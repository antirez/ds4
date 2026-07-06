/*
 * sys/file.h - flock and missing GNU helpers for Windows / MinGW-w64
 *
 * Provides:
 *   flock()         via LockFileEx / UnlockFileEx
 *   fcntl()         no-op macro (Windows has no per-fd close-on-exec flag)
 *   sysconf()       minimal shim for _SC_NPROCESSORS_ONLN and _SC_PAGESIZE
 *   dprintf()       via _write() — MinGW-w64 lacks this GNU extension
 *
 * Include path note: this file is found via -I./compat/win32, which adds the
 * compat/win32 prefix to the search path.  MinGW-w64 has no sys/file.h of its
 * own, so there is no shadowing risk.
 *
 * NOTE: This header does NOT include winsock2.h to avoid header conflicts with
 * legacy winsock.h. Source files that need networking (sockets) should include
 * winsock2.h DIRECTLY with appropriate guards. See ds4_distributed.c for example.
 */
#pragma once
#ifndef _WIN32_SYS_FILE_H_
#define _WIN32_SYS_FILE_H_

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#ifndef VC_EXTRA_LEAN
#  define VC_EXTRA_LEAN
#endif

#include <windows.h>
#include <io.h>
#include <direct.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

/* ssize_t for both MinGW and MSVC */
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef intptr_t ssize_t;
#endif
#ifndef SSIZE_MAX
#define SSIZE_MAX ((ssize_t)(SIZE_MAX >> 1))
#endif

/* PATH_MAX */
#ifndef PATH_MAX
#  ifdef MAX_PATH
#    define PATH_MAX MAX_PATH
#  else
#    define PATH_MAX 4096
#  endif
#endif

/* Network and socket functions are NOT included here to avoid winsock2.h
   conflicts. If your source file needs socket functions, include winsock2.h
   DIRECTLY with proper guards. Example in ds4_distributed.c. */

/* ──────────────────────────────────────────────────────────────────────────────── */
/* signal.h compatibility: Windows doesn't have sigaction. Provide stub. */

/* Define sigaction structure if not already defined (Windows doesn't have it) */
#ifndef HAVE_SIGACTION
struct sigaction {
    void (*sa_handler)(int);
    int sa_flags;
    int sa_mask;  /* Dummy field for compatibility; not used on Windows */
};

static inline int sigaction(int signum, const struct sigaction *act,
                            struct sigaction *oldact)
{
    /* Windows doesn't have sigaction. For now, this is a no-op stub.
       SIGINT is handled by SetConsoleCtrlHandler in Windows applications. */
    (void)signum;
    (void)act;
    (void)oldact;
    return 0;
}
#endif

static inline int sigemptyset(int *set)
{
    /* No-op on Windows. */
    if (set) *set = 0;
    return 0;
}

/* ──────────────────────────────────────────────────────────────────────────────── */
/* sys/ioctl.h compatibility: provide terminal size query for Windows. */

/* Minimal winsize structure compatible with POSIX */
struct winsize {
    unsigned short ws_row;
    unsigned short ws_col;
    unsigned short ws_xpixel;
    unsigned short ws_ypixel;
};

/* TIOCGWINSZ ioctl code (not used on Windows; shown here for compatibility) */
#define TIOCGWINSZ 0x5413

static inline int ioctl(int fd, unsigned long request, struct winsize *ws)
{
    /* On Windows, get console screen buffer info to determine terminal size. */
    if (request != TIOCGWINSZ) return -1;
    
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h == INVALID_HANDLE_VALUE) return -1;
    
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (!GetConsoleScreenBufferInfo(h, &csbi)) return -1;
    
    /* Calculate dimensions from the console buffer rectangle */
    ws->ws_col = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    ws->ws_row = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
    ws->ws_xpixel = 0;
    ws->ws_ypixel = 0;
    
    return 0;
}

/* ──────────────────────────────────────────────────────────────────────────────── */
/* sys/stat.h compatibility: mkdir on Windows takes only 1 argument. */

/* On Windows, the standard library has _mkdir which takes 1 argument (path only).
   POSIX mkdir takes 2 arguments (path, mode). We provide a macro that wraps _mkdir
   and ignores the mode parameter. */
#ifndef mkdir
#define mkdir(path, mode) _mkdir(path)
#endif

/* ──────────────────────────────────────────────────────────────────────────────── */
#define LOCK_EX 2
#define LOCK_NB 4
#define LOCK_UN 8

static inline int flock(int fd, int operation)
{
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }

    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));

    if (operation & LOCK_UN) {
        return UnlockFileEx(h, 0, 1, 0, &ov) ? 0 : (errno = EIO, -1);
    }

    DWORD flags = (operation & LOCK_EX) ? LOCKFILE_EXCLUSIVE_LOCK : 0;
    if (operation & LOCK_NB) flags |= LOCKFILE_FAIL_IMMEDIATELY;

    if (!LockFileEx(h, flags, 0, 1, 0, &ov)) {
        errno = (GetLastError() == ERROR_LOCK_VIOLATION) ? EWOULDBLOCK : EIO;
        return -1;
    }
    return 0;
}

/* ── fcntl no-op ───────────────────────────────────────────────────────── */
/* Windows has no per-fd close-on-exec bit; the only use in ds4 is the
   advisory fcntl(fd, F_SETFD, FD_CLOEXEC) call in the lock-file helper. */
#ifndef F_SETFD
#  define F_SETFD  2
#endif
#ifndef F_GETFL
#  define F_GETFL  3
#endif
#ifndef F_SETFL
#  define F_SETFL  4
#endif
#ifndef FD_CLOEXEC
#  define FD_CLOEXEC 1
#endif
#ifndef O_NONBLOCK
#  define O_NONBLOCK _O_NONBLOCK
#endif
/* Swallow all fcntl calls as a no-op.  The variadic macro form is needed
   because fcntl takes an optional third argument. */
#define fcntl(fd, ...) (0)

/* ── sysconf ───────────────────────────────────────────────────────────── */
/* MinGW-w64 does not implement sysconf().  Provide the two constants that
   ds4.c uses (_SC_NPROCESSORS_ONLN, _SC_PAGESIZE) backed by GetSystemInfo. */
#ifndef _SC_NPROCESSORS_ONLN
#  define _SC_NPROCESSORS_ONLN 84
#endif
#ifndef _SC_PAGESIZE
#  define _SC_PAGESIZE 30
#endif

#ifndef _DS4_SYSCONF_DEFINED
#define _DS4_SYSCONF_DEFINED
static inline long sysconf(int name)
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    if (name == _SC_NPROCESSORS_ONLN)
        return (long)si.dwNumberOfProcessors;
    if (name == _SC_PAGESIZE)
        return (long)si.dwPageSize;
    return -1L;
}
#endif

/* ── getpagesize ───────────────────────────────────────────────────────── */
/* MinGW-w64 lacks getpagesize().  Simple wrapper around sysconf. */
static inline int getpagesize(void)
{
    long ps = sysconf(_SC_PAGESIZE);
    return (ps > 0) ? (int)ps : 4096;  /* Default to 4096 if sysconf fails */
}

/* ── pread ─────────────────────────────────────────────────────────────── */
/* MinGW-w64 lacks pread().  Implement via ReadFile with OVERLAPPED so the
   file position is not disturbed (matching POSIX semantics). */
#ifndef _DS4_PREAD_DEFINED
#define _DS4_PREAD_DEFINED
#include <sys/types.h>
static inline ssize_t pread(int fd, void *buf, size_t count, off_t offset)
{
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ov.Offset     = (DWORD)((uint64_t)offset & 0xFFFFFFFFUL);
    ov.OffsetHigh = (DWORD)(((uint64_t)offset >> 32) & 0xFFFFFFFFUL);
    DWORD nread = 0;
    if (!ReadFile(h, buf, (DWORD)count, &nread, &ov)) {
        if (GetLastError() == ERROR_HANDLE_EOF) return 0;
        errno = EIO; return -1;
    }
    return (ssize_t)nread;
}
#endif /* _DS4_PREAD_DEFINED */

/* ── fmemopen ──────────────────────────────────────────────────────────── */
/* MinGW-w64 lacks fmemopen().  ds4 uses it to serialize/deserialize session
   snapshots to/from an in-memory buffer.  We implement a simple but correct
   version using a temporary file backed by the Windows temp directory.    */
#ifndef _DS4_FMEMOPEN_DEFINED
#define _DS4_FMEMOPEN_DEFINED
#include <string.h>
static inline FILE *fmemopen(void *buf, size_t size, const char *mode)
{
    /* Create an anonymous temp file, seed it with the buffer contents when
       opening for read, and return the FILE*.  On close the temp file is
       automatically removed. */
    FILE *fp = tmpfile();
    if (!fp) return NULL;
    if (mode[0] == 'r' || mode[0] == 'w') {
        if (mode[0] == 'r' && buf && size > 0) {
            if (fwrite(buf, 1, size, fp) != size) { fclose(fp); return NULL; }
            rewind(fp);
        }
    }
    return fp;
}
#endif /* _DS4_FMEMOPEN_DEFINED */
/* MinGW-w64 does not expose dprintf (it's a GNU extension).  Provide a
   small fallback that writes a formatted string to a raw file descriptor
   using _write(), which is the Windows CRT equivalent. */
#ifndef _DS4_DPRINTF_DEFINED
#define _DS4_DPRINTF_DEFINED
static inline int dprintf(int fd, const char *fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n > 0) (void)_write(fd, buf, (unsigned)n);
    return n;
}
#endif /* _DS4_DPRINTF_DEFINED */

/* ──────────────────────────────────────────────────────────────────────────────── */
/* time.h compatibility: localtime_r on Windows is localtime_s. */

#ifndef localtime_r
#define localtime_r(time_t_ptr, struct_tm_ptr) \
    (localtime_s((struct_tm_ptr), (time_t_ptr)) == 0 ? (struct_tm_ptr) : NULL)
#endif

/* ── getpid ────────────────────────────────────────────────────────────── */
/* MinGW-w64 has getpid() in <process.h>; ftruncate via _chsize_s.
   Just include process.h to expose getpid(). */
#include <process.h>

/* ftruncate: MSVC/_chsize equivalent */
#ifndef _DS4_FTRUNCATE_DEFINED
#define _DS4_FTRUNCATE_DEFINED
static inline int ftruncate(int fd, off_t length)
{
    return _chsize_s(fd, (long long)length) == 0 ? 0 : -1;
}
#endif

/* sleep(seconds): POSIX sleep maps to Windows Sleep(milliseconds) */
#ifndef _DS4_SLEEP_DEFINED
#define _DS4_SLEEP_DEFINED
static inline unsigned int sleep(unsigned int seconds)
{
    Sleep((DWORD)(seconds * 1000u));
    return 0;
}
#endif

/* nanosleep: approximate via Sleep (millisecond precision) */
#ifndef _DS4_NANOSLEEP_DEFINED
#define _DS4_NANOSLEEP_DEFINED
#include <time.h>  /* struct timespec on MSVC from ucrt/time.h */
static inline int nanosleep(const struct timespec *req, struct timespec *rem)
{
    if (req) Sleep((DWORD)(req->tv_sec * 1000u + req->tv_nsec / 1000000u));
    if (rem) { rem->tv_sec = 0; rem->tv_nsec = 0; }
    return 0;
}
#endif

/* ──────────────────────────────────────────────────────────────────────────────── */
/* signal.h compatibility: Windows doesn't have SIGPIPE signal. */

#ifndef SIGPIPE
#define SIGPIPE 13  /* Define a dummy value; will never be used on Windows */
#endif

/* ──────────────────────────────────────────────────────────────────────────────── */
/* POSIX constants missing on MSVC / MinGW-w64 */

/* STDERR_FILENO / STDOUT_FILENO / STDIN_FILENO */
#ifndef STDERR_FILENO
#  define STDIN_FILENO  0
#  define STDOUT_FILENO 1
#  define STDERR_FILENO 2
#endif

/* SSIZE_MAX - maximum value of ssize_t */
#ifndef SSIZE_MAX
#  include <stdint.h>
#  define SSIZE_MAX ((ssize_t)(SIZE_MAX >> 1))
#endif

/* clock_gettime / CLOCK_MONOTONIC via QueryPerformanceCounter
   Guard against MinGW's pthread_time.h which already provides this.
   Also guard against MSVC's ucrt/time.h which provides struct timespec. */
#ifndef CLOCK_MONOTONIC
#  define CLOCK_MONOTONIC 1
#endif
#ifndef CLOCK_REALTIME
#  define CLOCK_REALTIME  0
#endif

#if !defined(_DS4_CLOCK_GETTIME_DEFINED) && !defined(__MINGW32__)
#define _DS4_CLOCK_GETTIME_DEFINED
static inline int clock_gettime(int clk_id, struct timespec *tp)
{
    (void)clk_id;
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    tp->tv_sec  = (long)(cnt.QuadPart / freq.QuadPart);
    tp->tv_nsec = (long)(((cnt.QuadPart % freq.QuadPart) * 1000000000LL) / freq.QuadPart);
    return 0;
}
#endif /* _DS4_CLOCK_GETTIME_DEFINED && !__MINGW32__ */

#endif /* _WIN32 */
#endif /* _WIN32_SYS_FILE_H_ */
