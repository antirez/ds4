/*
 * compat/win32/unistd.h  -  POSIX unistd.h shim for Windows
 *
 * Used by both MinGW/gcc builds (via sys/file.h already covers most things)
 * and hipcc/MSVC builds (where MSVC has no unistd.h at all).
 *
 * Guards prevent redefinition when MinGW headers already provide functions.
 */

#pragma once
#ifndef _WIN32_UNISTD_H_
#define _WIN32_UNISTD_H_

#ifdef _WIN32

#include <io.h>
#include <sys/stat.h>
#include <stdint.h>
#include <stdlib.h>
#include <windows.h>
#include <process.h>
#include <time.h>
#include <fcntl.h>
#include <stdio.h>

/* ssize_t, SSIZE_MAX */
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef intptr_t ssize_t;
#endif
#ifndef SSIZE_MAX
#define SSIZE_MAX ((ssize_t)(SIZE_MAX >> 1))
#endif

/* off_t — file offset type.
 * NOTE: On MSVC, off_t is defined as 'long' (32-bit) by <sys/types.h>.
 * We cannot redefine the typedef here; instead, pread() uses int64_t directly
 * and callers must cast via (int64_t) not (off_t) for large-file offsets.
 */
#ifndef _OFF_T_DEFINED
#  define _OFF_T_DEFINED
   typedef __int64 off_t;
#endif

/* Standard file descriptors */
#ifndef STDIN_FILENO
#  define STDIN_FILENO  0
#  define STDOUT_FILENO 1
#  define STDERR_FILENO 2
#endif

/* getpid() */
#ifndef _DS4_GETPID_DEFINED
#define _DS4_GETPID_DEFINED
/* MSVC-only: MinGW's process.h already exposes getpid() */
#if defined(_MSC_VER) && !defined(__MINGW32__)
static inline int getpid(void) { return (int)_getpid(); }
#endif
#endif

/* close() - handles both CRT file descriptors and Winsock sockets.
 * _close() only works for CRT fds; sockets need closesocket().
 * We load closesocket lazily via GetProcAddress to avoid pulling in
 * winsock2.h here (which causes sockaddr redefinition conflicts). */
#if defined(_MSC_VER) && !defined(__MINGW32__)
static inline int close(int fd) {
    int r = _close(fd);
    if (r == -1 && errno == EBADF) {
        /* Not a CRT fd — try as a Winsock socket */
        typedef int (__stdcall *pfn_cs_t)(ULONG_PTR);
        static pfn_cs_t pfn_cs = NULL;
        if (!pfn_cs) {
            HMODULE ws2 = GetModuleHandleA("ws2_32.dll");
            if (ws2) pfn_cs = (pfn_cs_t)(void *)GetProcAddress(ws2, "closesocket");
        }
        if (pfn_cs && pfn_cs((ULONG_PTR)(unsigned)fd) == 0) { errno = 0; return 0; }
    }
    return r;
}
static inline int isatty(int fd) { return _isatty(fd); }
#endif

/* sysconf — guard prevents redefinition when sys/file.h already provided it */
#ifndef _SC_PAGESIZE
#  define _SC_PAGESIZE        30
#  define _SC_NPROCESSORS_ONLN 84
#endif
#if !defined(_DS4_SYSCONF_DEFINED) && defined(_MSC_VER) && !defined(__MINGW32__)
#define _DS4_SYSCONF_DEFINED
static inline long sysconf(int name)
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    if (name == _SC_NPROCESSORS_ONLN) return (long)si.dwNumberOfProcessors;
    if (name == _SC_PAGESIZE)         return (long)si.dwPageSize;
    return -1L;
}
#endif

/* pread */
#ifndef _DS4_PREAD_DEFINED
#define _DS4_PREAD_DEFINED
#include <errno.h>

/* pread — thread-safe positioned read on Windows.
 *
 * On Windows, concurrent ReadFile calls on a single synchronous HANDLE from
 * multiple threads is not safe. We use ReOpenFile() to obtain a private
 * FILE_FLAG_OVERLAPPED handle for each call, which gives us truly concurrent
 * positional I/O without any per-thread state or mutexes.
 *
 * The duplication is per-call (~2 µs overhead) but correct for the
 * multi-threaded ROCm streaming workers that share g_model_fd.
 */
static inline ssize_t pread(int fd, void *buf, size_t count, int64_t offset)
{
    if (count == 0) return 0;
    HANDLE orig = (HANDLE)_get_osfhandle(fd);
    if (orig == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }

    /* Get an overlapped-capable handle to the same file so that concurrent
     * pread calls from multiple threads are safe (18 ROCm streaming workers
     * all share g_model_fd). ReOpenFile with FILE_FLAG_OVERLAPPED gives each
     * caller its own independent positioned-read context. */
    HANDLE h = ReOpenFile(orig,
                          GENERIC_READ,
                          FILE_SHARE_READ | FILE_SHARE_WRITE,
                          FILE_FLAG_OVERLAPPED);
    const int reopen_ok = (h != INVALID_HANDLE_VALUE);
    if (!reopen_ok) h = orig;  /* single-threaded fallback */

    HANDLE ev = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (!ev) {
        if (reopen_ok) CloseHandle(h);
        errno = EIO; return -1;
    }

    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ov.Offset     = (DWORD)((uint64_t)offset & 0xFFFFFFFFUL);
    ov.OffsetHigh = (DWORD)(((uint64_t)offset >> 32) & 0xFFFFFFFFUL);
    ov.hEvent     = ev;

    DWORD nread = 0;
    BOOL ok = ReadFile(h, buf, (DWORD)count, &nread, &ov);
    if (!ok) {
        DWORD err = GetLastError();
        if (err == ERROR_IO_PENDING) {
            ok = GetOverlappedResult(h, &ov, &nread, TRUE);
            if (!ok) err = GetLastError();
        }
        if (!ok) {
            CloseHandle(ev);
            if (reopen_ok) CloseHandle(h);
            if (err == ERROR_HANDLE_EOF) return 0;
            errno = EIO; return -1;
        }
    }

    CloseHandle(ev);
    if (reopen_ok) CloseHandle(h);
    return (ssize_t)nread;
}
#endif

/* clock_gettime / CLOCK_MONOTONIC
   MinGW provides this via pthread_time.h; only define for pure MSVC. */
#ifndef CLOCK_MONOTONIC
#  define CLOCK_MONOTONIC 1
#  define CLOCK_REALTIME  0
#endif
/* MSVC's ucrt/time.h provides struct timespec; include it so the type is known. */
#include <time.h>
#if defined(_MSC_VER) && !defined(__MINGW32__) && !defined(_DS4_CLOCK_GETTIME_DEFINED)
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
#endif

/* strcasecmp — case-insensitive string comparison (POSIX)
   Provided as macro in strings.h for compatibility */
#ifndef strcasecmp
#  if defined(_MSC_VER) && !defined(__MINGW32__)
#    define strcasecmp(a, b) _stricmp((a), (b))
#    define strncasecmp(a, b, n) _strnicmp((a), (b), (n))
#  endif
#endif

/* mode_t — file mode type (for umask) */
#ifndef _MODE_T_DEFINED
#define _MODE_T_DEFINED
typedef unsigned int mode_t;
#endif

/* umask — MSVC has this but returns int; cast to mode_t if needed */
#ifndef _DS4_UMASK_COMPAT
#define _DS4_UMASK_COMPAT
/* MSVC's umask returns int but should be mode_t. Don't redefine since MSVC's
   corecrt_io.h already provides it. The caller can cast result if needed. */
#endif

/* S_IXUSR, S_IRWXG, S_IRWXO — stat mode bits */
#ifndef S_IXUSR
#  define S_IXUSR  0000100
#  define S_IRWXG  0000070
#  define S_IRWXO  0000007
#endif

/* mkstemp — create temporary file (POSIX; MSVC has _mktemp but it's less secure) */
#ifndef _DS4_MKSTEMP_DEFINED
#define _DS4_MKSTEMP_DEFINED
#include <io.h>
#include <fcntl.h>
#if defined(_MSC_VER) && !defined(__MINGW32__)
static inline int mkstemp(char *tmpl)
{
    /* mkstemp creates and opens a unique temporary file.
       MSVC has _mktemp (only creates name) and _open (raw descriptor).
       tmpl must be "XXXXXXXX" at end, where X are replaced with random chars. */
    if (_mktemp(tmpl) == NULL) return -1;
    return _open(tmpl, _O_RDWR | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
}
#endif
#endif

/* ftello — get file position (POSIX; MSVC has ftell returning long) */
#ifndef _DS4_FTELLO_DEFINED
#define _DS4_FTELLO_DEFINED
#if defined(_MSC_VER) && !defined(__MINGW32__)
static inline off_t ftello(FILE *stream)
{
    /* On Windows, use _ftelli64 for 64-bit file offsets */
    return (off_t)_ftelli64(stream);
}
static inline int fseeko(FILE *stream, off_t offset, int whence)
{
    /* On Windows, use _fseeki64 for 64-bit file offsets */
    return _fseeki64(stream, (int64_t)offset, whence);
}
#endif
#endif

/* localtime_r — thread-safe version of localtime
   Defined in sys/file.h as a macro that maps to localtime_s on MSVC */

/* SIGPIPE — signal number for broken pipe (Windows doesn't have POSIX signals) */
#ifndef SIGPIPE
#  define SIGPIPE 13
#endif

/* access() constants for mode parameter */
#ifndef F_OK
#  define F_OK 0  /* File exists */
#  define X_OK 1  /* Execute permission */
#  define W_OK 2  /* Write permission */
#  define R_OK 4  /* Read permission */
#endif

/* PATH_MAX — maximum path length */
#ifndef PATH_MAX
#  ifdef MAX_PATH
#    define PATH_MAX MAX_PATH  /* Windows MAX_PATH = 260 on MSVC */
#  else
#    define PATH_MAX 260
#  endif
#endif

/* useconds_t — microseconds integer type */
#ifndef _USECONDS_T_DEFINED
#define _USECONDS_T_DEFINED
typedef unsigned long useconds_t;
#endif

/* usleep — sleep for microseconds (POSIX; deprecated, but still used)
   Maps to Sleep(milliseconds) on Windows */
#ifndef _DS4_USLEEP_DEFINED
#define _DS4_USLEEP_DEFINED
#include <windows.h>
static inline int usleep(useconds_t usec)
{
    /* Convert microseconds to milliseconds, rounding up */
    Sleep((DWORD)((usec + 999) / 1000));
    return 0;
}
#endif
#if defined(_WINSOCK2API_) && !defined(_DS4_POLL_DEFINED)
#define _DS4_POLL_DEFINED
#ifndef POLLIN
#  define POLLIN   0x0001
#  define POLLOUT  0x0004
#  define POLLERR  0x0008
#  define POLLHUP  0x0010
#  define POLLNVAL 0x0020
#endif

static inline int poll(struct pollfd *fds, int nfds, int timeout)
{
    int result = WSAPoll((WSAPOLLFD *)fds, nfds, timeout);
    if (result == SOCKET_ERROR) return -1;
    return result;
}
#endif

#endif /* _WIN32 */
#endif /* _WIN32_UNISTD_H_ */
