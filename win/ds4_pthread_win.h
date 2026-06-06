/* ds4_pthread_win.h — minimal pthread shim for native Windows GPU builds.
 *
 * The native-Windows CPU build (MinGW-w64) gets pthreads from winpthreads, but
 * the native-Windows ROCm/HIP build compiles the C host code with clang in the
 * MSVC ABI (to match the hipcc-built ds4_cuda.o), and the MSVC toolchain has no
 * <pthread.h>. This header implements exactly the pthread subset DS4 uses on
 * top of the Win32 threading primitives:
 *
 *   pthread_t, pthread_create, pthread_join
 *   pthread_mutex_t / _init / _lock / _unlock / _destroy
 *   pthread_cond_t  / _init / _wait / _signal / _broadcast / _destroy
 *   pthread_once_t  / pthread_once / PTHREAD_ONCE_INIT
 *
 * Header-only and self-contained. The entire body is guarded by _WIN32, and it
 * is only pulled in for the Windows GPU build (not the MinGW CPU build, which
 * already has real pthreads), so POSIX builds are completely unaffected.
 *
 * Only included from ds4_win.h, and only when DS4_WIN_PTHREAD is requested, so
 * the MinGW CPU build keeps using winpthreads.
 */
#ifndef DS4_PTHREAD_WIN_H
#define DS4_PTHREAD_WIN_H

#ifdef _WIN32

#include <windows.h>
#include <process.h>
#include <errno.h>

/* ---- threads ------------------------------------------------------------- */
typedef struct {
    HANDLE        handle;
    void         *(*start)(void *);
    void         *arg;
    void         *retval;
} ds4_pthread_state;
typedef ds4_pthread_state *pthread_t;

static unsigned __stdcall ds4_pthread_trampoline(void *p)
{
    ds4_pthread_state *st = (ds4_pthread_state *)p;
    st->retval = st->start(st->arg);
    return 0;
}

static inline int pthread_create(pthread_t *thread, const void *attr,
                                 void *(*start)(void *), void *arg)
{
    (void)attr;
    ds4_pthread_state *st = (ds4_pthread_state *)calloc(1, sizeof(*st));
    if (!st) return EAGAIN;
    st->start = start;
    st->arg   = arg;
    uintptr_t h = _beginthreadex(NULL, 0, ds4_pthread_trampoline, st, 0, NULL);
    if (h == 0) { free(st); return EAGAIN; }
    st->handle = (HANDLE)h;
    *thread = st;
    return 0;
}

static inline int pthread_join(pthread_t thread, void **retval)
{
    if (!thread) return EINVAL;
    WaitForSingleObject(thread->handle, INFINITE);
    if (retval) *retval = thread->retval;
    CloseHandle(thread->handle);
    free(thread);
    return 0;
}

/* ---- mutex (non-recursive; matches PTHREAD default) ---------------------- */
typedef SRWLOCK pthread_mutex_t;
#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT

static inline int pthread_mutex_init(pthread_mutex_t *m, const void *attr)
{
    (void)attr;
    InitializeSRWLock(m);
    return 0;
}
static inline int pthread_mutex_lock(pthread_mutex_t *m)
{
    AcquireSRWLockExclusive(m);
    return 0;
}
static inline int pthread_mutex_unlock(pthread_mutex_t *m)
{
    ReleaseSRWLockExclusive(m);
    return 0;
}
static inline int pthread_mutex_destroy(pthread_mutex_t *m)
{
    (void)m; /* SRWLOCK needs no teardown */
    return 0;
}

/* ---- condition variable -------------------------------------------------- */
typedef CONDITION_VARIABLE pthread_cond_t;
#define PTHREAD_COND_INITIALIZER CONDITION_VARIABLE_INIT

static inline int pthread_cond_init(pthread_cond_t *c, const void *attr)
{
    (void)attr;
    InitializeConditionVariable(c);
    return 0;
}
static inline int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m)
{
    /* SRWLOCK held exclusively → CONDITION_VARIABLE_LOCKMODE default (0). */
    return SleepConditionVariableSRW(c, m, INFINITE, 0) ? 0 : EINVAL;
}
static inline int pthread_cond_signal(pthread_cond_t *c)
{
    WakeConditionVariable(c);
    return 0;
}
static inline int pthread_cond_broadcast(pthread_cond_t *c)
{
    WakeAllConditionVariable(c);
    return 0;
}
static inline int pthread_cond_destroy(pthread_cond_t *c)
{
    (void)c; /* CONDITION_VARIABLE needs no teardown */
    return 0;
}

/* ---- one-time init ------------------------------------------------------- */
typedef INIT_ONCE pthread_once_t;
#define PTHREAD_ONCE_INIT INIT_ONCE_STATIC_INIT

static void (*ds4_once_fn)(void);
static BOOL CALLBACK ds4_once_trampoline(PINIT_ONCE io, PVOID param, PVOID *ctx)
{
    (void)io; (void)ctx;
    ((void (*)(void))param)();
    return TRUE;
}
static inline int pthread_once(pthread_once_t *once, void (*init)(void))
{
    InitOnceExecuteOnce(once, ds4_once_trampoline, (PVOID)init, NULL);
    return 0;
}

#endif /* _WIN32 */
#endif /* DS4_PTHREAD_WIN_H */
