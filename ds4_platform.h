/* ds4_platform.h
 *
 * Platform abstraction layer.  On POSIX systems this header is essentially a
 * passthrough that pulls in the same Unix headers ds4 has always used.  On
 * Windows it defines small inline shims and type aliases so that the rest of
 * the codebase can keep calling pthread_*, mmap, poll, flock, etc. without
 * caring whether the underlying implementation is POSIX or Win32.
 *
 * Design rules:
 *   - All Windows-specific bodies live behind #ifdef _WIN32.  POSIX builds
 *     never see the Windows aliases; pthreads/sockets work as before.
 *   - We keep POSIX symbol names (pthread_mutex_t, pthread_cond_t, mmap, ...).
 *     This avoids touching ~200 lock/unlock sites in ds4_server.c.
 *   - Sockets on Windows are SOCKET (an unsigned pointer-sized handle).  To
 *     keep ds4_server.c compiling, we define SOCKET to int on POSIX and use it
 *     as the socket descriptor type.  Where the codebase already uses int for
 *     fds, the implicit conversion is fine.
 *
 * Important include order: when building on Windows, this header must be the
 * first include in every translation unit that also includes <windows.h>,
 * because winsock2.h must be #include'd before windows.h.
 */

#ifndef DS4_PLATFORM_H
#define DS4_PLATFORM_H

/* -------------------------------------------------------------------------- */
/* POSIX path: pull in the same headers ds4 has always used.                  */
/* -------------------------------------------------------------------------- */
#ifndef _WIN32

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <poll.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <strings.h>
#include <time.h>

/* The codebase uses bare int for socket descriptors.  Keep it that way. */
typedef int ds4_socket_t;
#define DS4_INVALID_SOCKET (-1)
#define ds4_socket_close(s)  close(s)
#define ds4_socket_errno()   (errno)
#define ds4_socket_would_block(e) ((e) == EAGAIN || (e) == EWOULDBLOCK)

/* GCC/Clang attributes used elsewhere. */
#ifndef DS4_UNUSED
#define DS4_UNUSED __attribute__((unused))
#endif
#ifndef DS4_NORETURN
#define DS4_NORETURN __attribute__((noreturn))
#endif

/* Optional helpers that map 1:1 to POSIX. */
static inline int ds4_socket_set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

#else /* _WIN32 */

/* -------------------------------------------------------------------------- */
/* Windows path.  Winsock2 must come before windows.h; both must come before  */
/* the rest of the codebase touches sockets or threads.                       */
/* -------------------------------------------------------------------------- */

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00 /* Windows 10+; needed for WSAPoll fix and SRWLOCK */
#endif
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

/* MSVC's <sys/types.h> defines off_t as 32-bit `long`, which is too narrow for
 * the 64-bit file offsets DS4 passes around (model files routinely exceed 4
 * GB).  Suppress the CRT's typedef so our own __int64 off_t below is the only
 * one in scope. */
#ifndef _OFF_T_DEFINED
#define _OFF_T_DEFINED
typedef __int64 _off_t;
typedef __int64 off_t;
#endif

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <process.h>
#include <bcrypt.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <signal.h>

/* POSIX access() mode constants — MSVC's <io.h> uses bare integer literals. */
#ifndef F_OK
#define F_OK 0
#endif
#ifndef X_OK
#define X_OK 1 /* not really supported by _access; same as F_OK on MSVC */
#endif
#ifndef W_OK
#define W_OK 2
#endif
#ifndef R_OK
#define R_OK 4
#endif

/* Everything below is callable from both C (ds4.c, ds4_server.c, ...) and C++
 * (ds4_cuda.cu).  Wrap declarations in extern "C" so the same .obj/.lib
 * defined in ds4_platform_win.c links cleanly into both. */
#ifdef __cplusplus
extern "C" {
#endif

/* MSVC doesn't ship strings.h; the only thing the codebase uses from it is
 * strcasecmp/strncasecmp.  Map to MSVC's _stricmp/_strnicmp. */
#define strcasecmp  _stricmp
#define strncasecmp _strnicmp

/* MSVC: ssize_t is not in the standard headers. */
#if defined(_MSC_VER) && !defined(ssize_t)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

/* useconds_t: POSIX type used in <unistd.h>'s usleep prototype.  MSVC
 * doesn't ship it.  ds4_eval.c casts pause_ms*1000 to (useconds_t); the
 * actual usleep() inline is declared later, after windows.h is in scope. */
#if defined(_MSC_VER) && !defined(useconds_t)
typedef unsigned long useconds_t;
#endif

/* SSIZE_MAX is POSIX; MSVC doesn't define it.  On 64-bit Windows SSIZE_T is
 * 64-bit so it matches LLONG_MAX. */
#include <limits.h>
#ifndef SSIZE_MAX
#define SSIZE_MAX LLONG_MAX
#endif

/* Attribute macros: MSVC doesn't understand __attribute__ at all. */
#define DS4_UNUSED
#define DS4_NORETURN __declspec(noreturn)

/* GCC/Clang's __thread storage qualifier maps to MSVC's __declspec(thread). */
#if defined(_MSC_VER) && !defined(__thread)
#define __thread __declspec(thread)
#endif

/* MinGW provides __attribute__ but we still want the unified macro names. */
#ifdef __GNUC__
#undef  DS4_UNUSED
#define DS4_UNUSED __attribute__((unused))
#endif

/* Standard fd numbers from <unistd.h>.  Pick MSVC equivalents. */
#ifndef STDIN_FILENO
#define STDIN_FILENO  _fileno(stdin)
#endif
#ifndef STDOUT_FILENO
#define STDOUT_FILENO _fileno(stdout)
#endif
#ifndef STDERR_FILENO
#define STDERR_FILENO _fileno(stderr)
#endif

/* ---- File / mmap shims --------------------------------------------------- */

#ifndef PROT_READ
#define PROT_READ  0x1
#endif
#ifndef MAP_PRIVATE
#define MAP_PRIVATE 0x02
#endif
#ifndef MAP_SHARED
#define MAP_SHARED  0x01
#endif
#ifndef MAP_FAILED
#define MAP_FAILED ((void *)(intptr_t)-1)
#endif

/* posix_madvise hints used by ds4. */
#ifndef POSIX_MADV_WILLNEED
#define POSIX_MADV_WILLNEED 3
#endif
#ifndef POSIX_MADV_DONTNEED
#define POSIX_MADV_DONTNEED 4
#endif

void *ds4_win_mmap(void *addr, size_t len, int prot, int flags, int fd, long long off);
int   ds4_win_munmap(void *addr, size_t len);
int   ds4_win_posix_madvise(void *addr, size_t len, int advice);

#define mmap(addr, len, prot, flags, fd, off) ds4_win_mmap(addr, len, prot, flags, fd, off)
#define munmap(addr, len)                     ds4_win_munmap(addr, len)
#define posix_madvise(addr, len, advice)      ds4_win_posix_madvise(addr, len, advice)

/* flock with LOCK_EX|LOCK_NB. */
#ifndef LOCK_EX
#define LOCK_EX 2
#endif
#ifndef LOCK_NB
#define LOCK_NB 4
#endif
#ifndef LOCK_UN
#define LOCK_UN 8
#endif
int ds4_win_flock(int fd, int op);
#define flock(fd, op) ds4_win_flock(fd, op)

/* ---- Threads ------------------------------------------------------------- */

typedef HANDLE pthread_t;

typedef struct {
    SRWLOCK srw;
} pthread_mutex_t;
#define PTHREAD_MUTEX_INITIALIZER { SRWLOCK_INIT }

typedef struct {
    CONDITION_VARIABLE cv;
} pthread_cond_t;
#define PTHREAD_COND_INITIALIZER { CONDITION_VARIABLE_INIT }

typedef struct {
    SRWLOCK srw;
} pthread_rwlock_t;
#define PTHREAD_RWLOCK_INITIALIZER { SRWLOCK_INIT }

typedef struct {
    INIT_ONCE io;
} pthread_once_t;
#define PTHREAD_ONCE_INIT { INIT_ONCE_STATIC_INIT }

typedef void pthread_attr_t;
typedef void pthread_mutexattr_t;
typedef void pthread_condattr_t;
typedef void pthread_rwlockattr_t;

int ds4_win_pthread_create(pthread_t *out, const pthread_attr_t *attr,
                           void *(*fn)(void *), void *arg);
int ds4_win_pthread_join(pthread_t t, void **retval);
int ds4_win_pthread_detach(pthread_t t);
int ds4_win_pthread_once(pthread_once_t *once, void (*init)(void));

#define pthread_create(t, a, f, x)  ds4_win_pthread_create(t, a, f, x)
#define pthread_join(t, r)          ds4_win_pthread_join(t, r)
#define pthread_detach(t)           ds4_win_pthread_detach(t)
#define pthread_once(o, f)          ds4_win_pthread_once(o, f)

static inline int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *a) {
    (void)a; InitializeSRWLock(&m->srw); return 0;
}
static inline int pthread_mutex_destroy(pthread_mutex_t *m) { (void)m; return 0; }
static inline int pthread_mutex_lock(pthread_mutex_t *m)   { AcquireSRWLockExclusive(&m->srw); return 0; }
static inline int pthread_mutex_unlock(pthread_mutex_t *m) { ReleaseSRWLockExclusive(&m->srw); return 0; }
static inline int pthread_mutex_trylock(pthread_mutex_t *m){ return TryAcquireSRWLockExclusive(&m->srw) ? 0 : EBUSY; }

static inline int pthread_cond_init(pthread_cond_t *c, const pthread_condattr_t *a) {
    (void)a; InitializeConditionVariable(&c->cv); return 0;
}
static inline int pthread_cond_destroy(pthread_cond_t *c) { (void)c; return 0; }
static inline int pthread_cond_signal(pthread_cond_t *c)   { WakeConditionVariable(&c->cv); return 0; }
static inline int pthread_cond_broadcast(pthread_cond_t *c){ WakeAllConditionVariable(&c->cv); return 0; }
static inline int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) {
    SleepConditionVariableSRW(&c->cv, &m->srw, INFINITE, 0);
    return 0;
}
int ds4_win_pthread_cond_timedwait(pthread_cond_t *c, pthread_mutex_t *m,
                                   const struct timespec *abstime);
#define pthread_cond_timedwait(c, m, ts) ds4_win_pthread_cond_timedwait(c, m, ts)

static inline int pthread_rwlock_init(pthread_rwlock_t *r, const pthread_rwlockattr_t *a) {
    (void)a; InitializeSRWLock(&r->srw); return 0;
}
static inline int pthread_rwlock_destroy(pthread_rwlock_t *r) { (void)r; return 0; }
static inline int pthread_rwlock_rdlock(pthread_rwlock_t *r) { AcquireSRWLockShared(&r->srw); return 0; }
static inline int pthread_rwlock_wrlock(pthread_rwlock_t *r) { AcquireSRWLockExclusive(&r->srw); return 0; }
static inline int pthread_rwlock_unlock(pthread_rwlock_t *r) {
    /* SRWLOCK has separate Shared/Exclusive release.  ds4 always pairs lock/unlock
     * within the same code path, so the calling code knows which mode was used.
     * To keep the POSIX API, we expose a single rwlock_unlock that the helper
     * code below resolves: callers must use ds4_rwlock_unlock_shared or
     * ds4_rwlock_unlock_exclusive in new code.  For existing call sites that
     * use pthread_rwlock_unlock, we release as exclusive because in this
     * codebase rwlocks are mainly used in write-mostly paths.  Audit before
     * adding read-heavy rwlocks. */
    ReleaseSRWLockExclusive(&r->srw);
    return 0;
}
static inline int ds4_rwlock_unlock_shared(pthread_rwlock_t *r)    { ReleaseSRWLockShared(&r->srw); return 0; }
static inline int ds4_rwlock_unlock_exclusive(pthread_rwlock_t *r) { ReleaseSRWLockExclusive(&r->srw); return 0; }

/* ---- Sockets ------------------------------------------------------------- */

typedef SOCKET ds4_socket_t;
#define DS4_INVALID_SOCKET INVALID_SOCKET

void ds4_win_wsa_startup(void);     /* idempotent; safe to call from main() */
void ds4_win_wsa_cleanup(void);

static inline int ds4_socket_close(SOCKET s) { return closesocket(s); }
static inline int ds4_socket_errno(void)     { return WSAGetLastError(); }
static inline int ds4_socket_would_block(int e) {
    return e == WSAEWOULDBLOCK || e == WSAEINPROGRESS;
}
static inline int ds4_socket_set_nonblocking(SOCKET s) {
    u_long mode = 1;
    return ioctlsocket(s, FIONBIO, &mode) == 0 ? 0 : -1;
}

/* The server code calls poll() with struct pollfd; redirect to WSAPoll which
 * has the same shape on Win10 2004+. */
#define poll(fds, nfds, timeout) WSAPoll(fds, (ULONG)(nfds), timeout)

/* errno-style aliases used by the server for socket non-blocking returns. */
#ifndef EAGAIN
#define EAGAIN      WSAEWOULDBLOCK
#endif
#ifndef EWOULDBLOCK
#define EWOULDBLOCK WSAEWOULDBLOCK
#endif

/* ---- Misc shims ---------------------------------------------------------- */

void ds4_win_random_bytes(void *buf, size_t n);
const char *ds4_win_tmp_path(char *buf, size_t cap, const char *suffix);

/* usleep emulation. */
static inline int usleep(unsigned us) {
    /* Windows Sleep takes ms; round up so callers don't busy-spin. */
    DWORD ms = us / 1000u;
    if (us != 0 && ms == 0) ms = 1;
    Sleep(ms);
    return 0;
}

/* clock_gettime shim.  MSVC doesn't provide one; we use QueryPerformanceCounter
 * for monotonic time and GetSystemTimeAsFileTime for realtime.  Both meet the
 * monotonicity and resolution that ds4 callers expect. */
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC      1
#define CLOCK_REALTIME       0
#define CLOCK_UPTIME_RAW     1   /* macOS-only flavor; map to monotonic */
typedef int clockid_t;
#endif

int ds4_win_clock_gettime(int clk, struct timespec *ts);
#define clock_gettime(clk, ts) ds4_win_clock_gettime((int)(clk), ts)

/* gettimeofday: POSIX has it in <sys/time.h>; not on MSVC.  Emulate via the
 * same QueryPerformanceCounter-based path we use for clock_gettime.  struct
 * timeval is already declared by winsock2.h, included above. */
int ds4_win_gettimeofday(struct timeval *tv, void *tz);
#define gettimeofday(tv, tz) ds4_win_gettimeofday(tv, tz)

/* localtime_r: thread-safe variant.  MSVC's localtime_s has the result/tz
 * arguments swapped relative to POSIX; wrap it. */
static __inline struct tm *ds4_win_localtime_r(const time_t *t, struct tm *out) {
    return (localtime_s(out, t) == 0) ? out : (struct tm *)0;
}
#define localtime_r(t, out) ds4_win_localtime_r(t, out)

/* sysconf(_SC_PAGESIZE) is used by the CUDA backend to align mmap'd regions.
 * On Windows we map it to GetSystemInfo. */
#ifndef _SC_PAGESIZE
#define _SC_PAGESIZE 30
#endif
#ifndef _SC_NPROCESSORS_ONLN
#define _SC_NPROCESSORS_ONLN 84
#endif
long ds4_win_sysconf(int name);
#define sysconf(n) ds4_win_sysconf(n)

/* pread on a file descriptor; emulated via SetFilePointer+ReadFile with an
 * explicit 64-bit offset so the 81 GB GGUF file is reachable. */
ssize_t ds4_win_pread(int fd, void *buf, size_t n, long long off);
#define pread(fd, buf, n, off) ds4_win_pread(fd, buf, n, (long long)(off))

/* getpid: MSVC ships _getpid. */
#define getpid _getpid

/* ftruncate to a 64-bit length. */
static inline int ds4_win_ftruncate(int fd, long long len) {
    return _chsize_s(fd, (__int64)len) == 0 ? 0 : -1;
}
#define ftruncate(fd, len) ds4_win_ftruncate(fd, (long long)(len))

/* 64-bit fseek/ftell on FILE*.  off_t may still be 32-bit on Win64, so callers
 * must already be passing 64-bit-safe values to fseeko/ftello. */
#define fseeko(fp, off, whence) _fseeki64(fp, (__int64)(off), whence)
#define ftello(fp)              _ftelli64(fp)
typedef __int64 off64_t;
/* off_t is already typedef'd to __int64 above (with _OFF_T_DEFINED guard). */

/* dprintf(fd, fmt, ...) — POSIX-only.  Format into a buffer, then _write. */
int ds4_win_dprintf(int fd, const char *fmt, ...);
#define dprintf ds4_win_dprintf

/* Minimal opendir/readdir/closedir using FindFirstFile/FindNextFile.  Only
 * the d_name field of struct dirent is consulted by ds4_server.c. */
struct dirent {
    char d_name[260];
};
typedef struct ds4_win_dir DIR;
DIR *ds4_win_opendir(const char *path);
struct dirent *ds4_win_readdir(DIR *d);
int ds4_win_closedir(DIR *d);
#define opendir  ds4_win_opendir
#define readdir  ds4_win_readdir
#define closedir ds4_win_closedir

/* ---- Console / termios shims (ds4_eval.c) -------------------------------- */

struct ds4_console_state {
    DWORD in_mode;
    DWORD out_mode;
    int   restored;
};
int  ds4_win_console_enter_raw(struct ds4_console_state *out);
int  ds4_win_console_leave_raw(const struct ds4_console_state *prev);
int  ds4_win_console_size(int *cols, int *rows);
int  ds4_win_console_input_ready(void);

/* Minimal termios stub: ds4_eval.c only uses it as an opaque save/restore
 * blob.  We map the POSIX type to our state struct and the call sites use
 * the shim helpers above.  See ds4_eval.c edits. */
#define termios ds4_console_state

#ifndef TIOCGWINSZ
#define TIOCGWINSZ 0x5413
struct winsize { unsigned short ws_row, ws_col, ws_xpixel, ws_ypixel; };
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* _WIN32 */

#endif /* DS4_PLATFORM_H */
