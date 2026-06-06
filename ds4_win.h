/* ds4_win.h — minimal POSIX compatibility layer for native Windows builds.
 *
 * Provides just the POSIX surface ds4.c relies on that MinGW/UCRT lacks:
 *   - mmap / munmap / madvise (read-only file mappings)
 *   - sysconf(_SC_NPROCESSORS_ONLN / _SC_PAGESIZE)
 *   - flock / fcntl(F_SETFD,FD_CLOEXEC) / pread / ftruncate / dprintf  (instance lock)
 *   - fmemopen (fixed-buffer "wb"/"rb", temp-file backed with copy-back on close)
 *
 * Header-only, self-contained, no third-party deps. The whole body is guarded by
 * _WIN32, so this header is inert on POSIX platforms. ds4.c includes it in place
 * of <sys/mman.h> (and the other POSIX-only surface) behind #ifdef _WIN32, so the
 * native MinGW-w64 CPU build needs no extra include/search-path flags. MinGW
 * already provides pthread, clock_gettime and ftruncate.
 */
#ifndef DS4_WIN_H
#define DS4_WIN_H

#ifdef _WIN32

#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>

/* ---- mmap ---------------------------------------------------------------- */
#define PROT_NONE  0x0
#define PROT_READ  0x1
#define PROT_WRITE 0x2
#define PROT_EXEC  0x4
#define MAP_SHARED  0x01
#define MAP_PRIVATE 0x02
#define MAP_FAILED  ((void *)-1)
#define POSIX_MADV_NORMAL     0
#define POSIX_MADV_RANDOM     1
#define POSIX_MADV_SEQUENTIAL 2
#define POSIX_MADV_WILLNEED   3
#define POSIX_MADV_DONTNEED   4
#define MADV_WILLNEED POSIX_MADV_WILLNEED

#ifndef _SC_PAGESIZE
#define _SC_PAGESIZE         0x1
#endif
#ifndef _SC_NPROCESSORS_ONLN
#define _SC_NPROCESSORS_ONLN 0x2
#endif

/* ---- misc POSIX surface used by the GPU (HIP) host code ------------------ */
/* MinGW/UCRT supplies these; the clang-MSVC HIP toolchain (ds4_cuda.cu build)
 * does not. Guard each so the MinGW CPU build is unaffected. */
#ifndef STDIN_FILENO
#define STDIN_FILENO  0
#endif
#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
#define STDERR_FILENO 2
#endif

#ifndef SSIZE_MAX
#define SSIZE_MAX ((ssize_t)(((size_t)-1) >> 1))
#endif

/* ssize_t / off_t: MinGW defines these via <sys/types.h>; MSVC does not.
 * MSVC exposes _SSIZE_T_DEFINED once <BaseTsd.h> (pulled in by windows.h) and
 * the CRT have declared SSIZE_T; provide ssize_t/off_t only when absent. */
#if !defined(_SSIZE_T_DEFINED) && !defined(__MINGW32__) && !defined(_SSIZE_T_)
typedef SSIZE_T ssize_t;
#define _SSIZE_T_DEFINED
#endif
#if !defined(_OFF_T_DEFINED) && !defined(__MINGW32__)
/* MSVC <sys/types.h> already typedefs off_t to long; only define if missing. */
#ifndef _OFF_T_
typedef long long off_t;
#endif
#endif

/* clock_gettime / CLOCK_MONOTONIC: present in MinGW, absent in clang-MSVC.
 * MSVC's <time.h> already declares struct timespec, so we only supply the
 * clock id macros and the function. */
#if !defined(CLOCK_MONOTONIC) && !defined(__MINGW32__)
#include <time.h>
#define CLOCK_REALTIME  0
#define CLOCK_MONOTONIC 1
static inline int clock_gettime(int clk, struct timespec *ts)
{
    (void)clk;
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    ts->tv_sec  = (long long)(cnt.QuadPart / freq.QuadPart);
    long long rem = cnt.QuadPart % freq.QuadPart;
    ts->tv_nsec = (long)((rem * 1000000000LL) / freq.QuadPart);
    return 0;
}
#endif

/* ---- file locking / fd flags -------------------------------------------- */
#ifndef F_SETFD
#define F_SETFD    2
#endif
#ifndef FD_CLOEXEC
#define FD_CLOEXEC 1
#endif
#define LOCK_SH 1
#define LOCK_EX 2
#define LOCK_NB 4
#define LOCK_UN 8

static inline void *mmap(void *addr, size_t length, int prot, int flags,
                         int fd, long long offset)
{
    (void)addr; (void)flags; (void)prot;
    HANDLE fh = (HANDLE)_get_osfhandle(fd);
    if (fh == INVALID_HANDLE_VALUE) { errno = EBADF; return MAP_FAILED; }
    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (mh == NULL) { errno = ENOMEM; return MAP_FAILED; }
    DWORD off_hi = (DWORD)((uint64_t)offset >> 32);
    DWORD off_lo = (DWORD)((uint64_t)offset & 0xFFFFFFFFu);
    void *p = MapViewOfFile(mh, FILE_MAP_READ, off_hi, off_lo, length);
    CloseHandle(mh); /* view keeps the section alive */
    if (p == NULL) { errno = ENOMEM; return MAP_FAILED; }
    return p;
}

static inline int munmap(void *addr, size_t length)
{
    (void)length;
    return UnmapViewOfFile(addr) ? 0 : -1;
}

static inline int posix_madvise(void *addr, size_t length, int advice)
{
    (void)addr; (void)length; (void)advice;
    return 0; /* advisory only */
}
static inline int madvise(void *addr, size_t length, int advice)
{
    return posix_madvise(addr, length, advice);
}

static inline long sysconf(int name)
{
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    if (name == _SC_NPROCESSORS_ONLN) return (long)si.dwNumberOfProcessors;
    if (name == _SC_PAGESIZE)         return (long)si.dwPageSize;
    errno = EINVAL;
    return -1;
}

static inline int flock(int fd, int op)
{
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    OVERLAPPED ov; memset(&ov, 0, sizeof(ov));
    if (op & LOCK_UN) {
        return UnlockFileEx(h, 0, MAXDWORD, MAXDWORD, &ov) ? 0 : -1;
    }
    DWORD f = 0;
    if (op & LOCK_EX) f |= LOCKFILE_EXCLUSIVE_LOCK;
    if (op & LOCK_NB) f |= LOCKFILE_FAIL_IMMEDIATELY;
    if (!LockFileEx(h, f, 0, MAXDWORD, MAXDWORD, &ov)) {
        errno = (GetLastError() == ERROR_LOCK_VIOLATION) ? EWOULDBLOCK : EACCES;
        return -1;
    }
    return 0;
}

static inline int fcntl(int fd, int cmd, ...)
{
    (void)fd; (void)cmd;
    return 0; /* F_SETFD/FD_CLOEXEC is a no-op: Windows handles aren't inherited by default */
}

static inline long long ds4_pread(int fd, void *buf, size_t count, long long offset)
{
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    OVERLAPPED ov; memset(&ov, 0, sizeof(ov));
    ov.Offset     = (DWORD)((uint64_t)offset & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD)((uint64_t)offset >> 32);
    DWORD got = 0;
    if (!ReadFile(h, buf, (DWORD)count, &got, &ov)) {
        if (GetLastError() == ERROR_HANDLE_EOF) return 0;
        errno = EIO; return -1;
    }
    return (long long)got;
}
#define pread(fd, buf, count, offset) ds4_pread((fd), (buf), (size_t)(count), (long long)(offset))

/* ftruncate: provided by MinGW <unistd.h>; absent in the MSVC ABI build. */
#if !defined(__MINGW32__)
static inline int ftruncate(int fd, long long length)
{
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    LARGE_INTEGER li; li.QuadPart = length;
    if (!SetFilePointerEx(h, li, NULL, FILE_BEGIN)) { errno = EINVAL; return -1; }
    if (!SetEndOfFile(h)) { errno = EIO; return -1; }
    return 0;
}
#endif

static inline int dprintf(int fd, const char *fmt, ...)
{
    char buf[512];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) return -1;
    if (n > (int)sizeof(buf)) n = (int)sizeof(buf);
    return _write(fd, buf, (unsigned)n);
}

/* ---- fmemopen (temp-file backed, fixed buffer) --------------------------- */
typedef struct { FILE *fp; void *buf; size_t cap; int writeback; } ds4_memstream;
#define DS4_MEMSTREAM_MAX 16
static ds4_memstream ds4_ms_tab[DS4_MEMSTREAM_MAX];
static CRITICAL_SECTION ds4_ms_cs;
static volatile LONG ds4_ms_init = 0;

static inline void ds4_ms_ensure(void)
{
    if (InterlockedCompareExchange(&ds4_ms_init, 1, 0) == 0)
        InitializeCriticalSection(&ds4_ms_cs);
}

static inline FILE *ds4_tmpfile(void)
{
    char dir[MAX_PATH], path[MAX_PATH];
    if (!GetTempPathA(sizeof(dir), dir)) return NULL;
    if (!GetTempFileNameA(dir, "ds4", 0, path)) return NULL;
    /* open read/write, delete on close */
    return fopen(path, "wb+TD"); /* T=temporary, D=delete-on-close (MSVCRT ext) */
}

static inline FILE *fmemopen(void *buf, size_t size, const char *mode)
{
    ds4_ms_ensure();
    int writing = (mode && (strchr(mode, 'w') || strchr(mode, 'a') || strchr(mode, '+')));
    FILE *fp = ds4_tmpfile();
    if (!fp) return NULL;
    if (!writing && buf && size) {
        if (fwrite(buf, 1, size, fp) != size) { fclose(fp); return NULL; }
        rewind(fp);
    }
    EnterCriticalSection(&ds4_ms_cs);
    for (int i = 0; i < DS4_MEMSTREAM_MAX; i++) {
        if (ds4_ms_tab[i].fp == NULL) {
            ds4_ms_tab[i].fp = fp; ds4_ms_tab[i].buf = buf;
            ds4_ms_tab[i].cap = size; ds4_ms_tab[i].writeback = writing ? 1 : 0;
            break;
        }
    }
    LeaveCriticalSection(&ds4_ms_cs);
    return fp;
}

static inline int ds4_win_fclose(FILE *fp)
{
    if (fp && ds4_ms_init) {
        EnterCriticalSection(&ds4_ms_cs);
        for (int i = 0; i < DS4_MEMSTREAM_MAX; i++) {
            if (ds4_ms_tab[i].fp == fp) {
                if (ds4_ms_tab[i].writeback && ds4_ms_tab[i].buf && ds4_ms_tab[i].cap) {
                    fflush(fp); rewind(fp);
                    fread(ds4_ms_tab[i].buf, 1, ds4_ms_tab[i].cap, fp); /* copy back */
                }
                ds4_ms_tab[i].fp = NULL; ds4_ms_tab[i].buf = NULL;
                ds4_ms_tab[i].cap = 0;   ds4_ms_tab[i].writeback = 0;
                break;
            }
        }
        LeaveCriticalSection(&ds4_ms_cs);
    }
    return fclose(fp); /* real fclose — macro defined only after this header */
}

#endif /* _WIN32 */

/* Redirect fclose AFTER all helpers above so ds4_win_fclose's own call hits the
 * real fclose. Source files including this header get the memory-stream-aware one. */
#ifdef _WIN32
#define fclose(fp) ds4_win_fclose(fp)
#endif

#endif /* DS4_WIN_H */
