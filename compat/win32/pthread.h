/*
 * compat/win32/pthread.h  –  Minimal pthreads shim for hipcc (MSVC target)
 *
 * Maps the pthread subset used by ds4_rocm_runtime.cuh onto native Windows
 * synchronisation primitives available since Windows Vista / Windows 10.
 *
 * Primitives covered:
 *   pthread_t            CreateThread / WaitForSingleObject
 *   pthread_mutex_t      CRITICAL_SECTION
 *   pthread_cond_t       CONDITION_VARIABLE (Windows Vista+)
 *   PTHREAD_MUTEX_INITIALIZER / PTHREAD_COND_INITIALIZER
 *   pthread_create / pthread_join
 *   pthread_mutex_lock / pthread_mutex_unlock
 *   pthread_cond_wait / pthread_cond_signal / pthread_cond_broadcast
 *
 * NOTE: This header is included by ds4_rocm.cu only when compiled by hipcc
 * on Windows (MSVC host toolchain).  The MinGW-based C sources use the real
 * pthread.h from <pthreads-w32> / libwinpthread supplied by MSYS2.
 */

#pragma once
#ifndef _WIN32_PTHREAD_SHIM_H_
#define _WIN32_PTHREAD_SHIM_H_

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <process.h>   /* _beginthreadex */
#include <errno.h>
#include <stdlib.h>
#include <time.h>      /* struct timespec */

/* ── pthread_t ─────────────────────────────────────────────────────────── */

typedef HANDLE pthread_t;

/* Thread start function trampoline: pthreads uses (void*)->void*, Windows
   _beginthreadex uses (void*)->unsigned.  We store the real start/arg in a
   small heap-allocated struct and convert the return value. */
struct _pthread_trampoline_arg {
    void *(*start_routine)(void *);
    void *arg;
};

static inline unsigned __stdcall _pthread_trampoline(void *p)
{
    struct _pthread_trampoline_arg *ta = (struct _pthread_trampoline_arg *)p;
    void *(*fn)(void *) = ta->start_routine;
    void *arg = ta->arg;
    free(ta);
    fn(arg);
    return 0;
}

static inline int pthread_create(pthread_t *thread, const void *attr,
                                  void *(*start_routine)(void *), void *arg)
{
    struct _pthread_trampoline_arg *ta =
        (struct _pthread_trampoline_arg *)malloc(sizeof(*ta));
    if (!ta) return ENOMEM;
    ta->start_routine = start_routine;
    ta->arg = arg;
    uintptr_t h = _beginthreadex(NULL, 0, _pthread_trampoline, ta, 0, NULL);
    if (h == 0) { free(ta); return EAGAIN; }
    *thread = (HANDLE)h;
    return 0;
}

static inline int pthread_join(pthread_t thread, void **retval)
{
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    if (retval) *retval = NULL;
    return 0;
}

static inline int pthread_detach(pthread_t thread)
{
    /* On Windows, "detaching" means we simply close the handle so the OS
       reclaims it when the thread exits.  The thread keeps running. */
    CloseHandle(thread);
    return 0;
}

/* ── pthread_mutex_t ───────────────────────────────────────────────────── */

typedef CRITICAL_SECTION pthread_mutex_t;

/* Value-initialiser for static mutexes.  We can't call InitializeCriticalSection
   at compile time, so we use a sentinel value and lazy-init on first lock.
   A simple approach: zero-init the CRITICAL_SECTION and call
   InitializeCriticalSection inside lock if not yet initialised.
   CRITICAL_SECTION is 40 bytes; all-zeros is a safe "not initialised" marker
   because the DebugInfo field would be NULL for a real CS. */
#define PTHREAD_MUTEX_INITIALIZER { 0 }

static inline int pthread_mutex_lock(pthread_mutex_t *m)
{
    /* Lazy init: DebugInfo == NULL means either uninitialised or a non-debug
       CS.  We use the SpinCount field (LockCount == -1 uninit heuristic).
       Simplest portable approach: always call Init if LockCount is 0xFFFFFFFF
       (the initial sentinel from zero-init). */
    if (m->LockCount == (LONG)-1 && m->RecursionCount == 0 && m->OwningThread == NULL)
        InitializeCriticalSection(m);
    EnterCriticalSection(m);
    return 0;
}

static inline int pthread_mutex_unlock(pthread_mutex_t *m)
{
    LeaveCriticalSection(m);
    return 0;
}

static inline int pthread_mutex_destroy(pthread_mutex_t *m)
{
    DeleteCriticalSection(m);
    return 0;
}

static inline int pthread_mutex_init(pthread_mutex_t *m, const void *attr)
{
    (void)attr;
    InitializeCriticalSection(m);
    return 0;
}

/* ── pthread_once ──────────────────────────────────────────────────────── */

typedef INIT_ONCE pthread_once_t;
#define PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT

static BOOL CALLBACK _pthread_once_callback(PINIT_ONCE once, PVOID param, PVOID *ctx)
{
    (void)once; (void)ctx;
    ((void (*)(void))param)();
    return TRUE;
}

static inline int pthread_once(pthread_once_t *once, void (*init_routine)(void))
{
    InitOnceExecuteOnce(once, _pthread_once_callback, (PVOID)init_routine, NULL);
    return 0;
}

/* ── pthread_cond_t ────────────────────────────────────────────────────── */

typedef CONDITION_VARIABLE pthread_cond_t;

#define PTHREAD_COND_INITIALIZER CONDITION_VARIABLE_INIT

static inline int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex)
{
    if (!SleepConditionVariableCS(cond, mutex, INFINITE))
        return GetLastError();
    return 0;
}

static inline int pthread_cond_timedwait(pthread_cond_t *cond, pthread_mutex_t *mutex,
                                         const struct timespec *abstime)
{
    /* abstime is absolute; convert to relative milliseconds */
    LARGE_INTEGER freq, now;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    int64_t nowns = (int64_t)(now.QuadPart * 1000000000LL / freq.QuadPart);
    int64_t absns = (int64_t)abstime->tv_sec * 1000000000LL + abstime->tv_nsec;
    int64_t relns = absns - nowns;
    DWORD ms = (relns <= 0) ? 0 : (DWORD)(relns / 1000000LL);
    
    if (!SleepConditionVariableCS(cond, mutex, ms)) {
        DWORD err = GetLastError();
        return (err == ERROR_TIMEOUT) ? ETIMEDOUT : err;
    }
    return 0;
}

static inline int pthread_cond_signal(pthread_cond_t *cond)
{
    WakeConditionVariable(cond);
    return 0;
}

static inline int pthread_cond_broadcast(pthread_cond_t *cond)
{
    WakeAllConditionVariable(cond);
    return 0;
}

static inline int pthread_cond_destroy(pthread_cond_t *cond)
{
    (void)cond;  /* CONDITION_VARIABLE doesn't require cleanup */
    return 0;
}

static inline int pthread_cond_init(pthread_cond_t *cond, const void *attr)
{
    (void)attr;
    InitializeConditionVariable(cond);
    return 0;
}

#endif /* _WIN32 */
#endif /* _WIN32_PTHREAD_SHIM_H_ */
