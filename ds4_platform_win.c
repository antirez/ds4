/* ds4_platform_win.c
 *
 * Windows implementations of the POSIX functions ds4 needs.  This file is
 * compiled only on Windows (see CMakeLists.txt); the Makefile-driven Linux
 * and macOS builds never see it.
 *
 * Implements:
 *   - mmap / munmap / posix_madvise on file descriptors via CreateFileMapping
 *   - flock (LOCK_EX|LOCK_NB) via LockFileEx
 *   - pthread_create/join/detach via _beginthreadex
 *   - pthread_once via InitOnceExecuteOnce
 *   - pthread_cond_timedwait (the inline-only variants in ds4_platform.h
 *     handle the non-timed cond ops directly)
 *   - WSA startup / cleanup
 *   - random bytes via BCryptGenRandom (replaces /dev/urandom reads)
 *   - Console raw mode and window size for the ds4-eval TUI
 *
 * Notes on cudaHostRegister and ds4_cuda.cu:
 *   The model loader in ds4.c calls mmap() with MAP_SHARED so the GGUF file
 *   can be pinned via cudaHostRegister later from ds4_cuda.cu.  Our mmap
 *   shim uses MapViewOfFile, which returns a base address that is page
 *   aligned and committable, exactly what cudaHostRegister expects.  The
 *   NVIDIA documented gotcha (cudaErrorInvalidValue on contiguous mappings
 *   >4MB) does not appear on modern drivers (>= 555) for read-only mappings,
 *   but the registration call in ds4_cuda.cu now also tries
 *   cudaHostRegisterIoMemory as a fallback if the plain Mapped|ReadOnly call
 *   fails.  See ds4_cuda.cu changes.
 */

#ifdef _WIN32

#include "ds4_platform.h"

#include <bcrypt.h>
#include <io.h>
#include <stdarg.h>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "bcrypt.lib")

/* -------------------------------------------------------------------------- */
/* mmap / munmap / posix_madvise                                              */
/* -------------------------------------------------------------------------- */

/* Each successful mmap returns a base pointer.  To munmap correctly we also
 * need to close the file-mapping handle we created.  Keep a small bounded
 * registry keyed by base address.  This is fine because ds4 mmaps a small
 * number of long-lived regions (the GGUF file, the on-disk KV cache files). */

#define DS4_MMAP_REG_CAP 64
struct ds4_mmap_entry {
    void  *base;
    HANDLE mapping;
};
static struct ds4_mmap_entry g_mmap_reg[DS4_MMAP_REG_CAP];
static SRWLOCK               g_mmap_reg_lock = SRWLOCK_INIT;

static void ds4_mmap_reg_add(void *base, HANDLE mapping) {
    AcquireSRWLockExclusive(&g_mmap_reg_lock);
    for (int i = 0; i < DS4_MMAP_REG_CAP; i++) {
        if (g_mmap_reg[i].base == NULL) {
            g_mmap_reg[i].base = base;
            g_mmap_reg[i].mapping = mapping;
            ReleaseSRWLockExclusive(&g_mmap_reg_lock);
            return;
        }
    }
    ReleaseSRWLockExclusive(&g_mmap_reg_lock);
    /* Overflow: leak the mapping handle.  We will still UnmapViewOfFile in
     * munmap, but the kernel mapping object stays until process exit.  At 64
     * slots the codebase never reaches this in practice. */
}

static HANDLE ds4_mmap_reg_take(void *base) {
    HANDLE h = NULL;
    AcquireSRWLockExclusive(&g_mmap_reg_lock);
    for (int i = 0; i < DS4_MMAP_REG_CAP; i++) {
        if (g_mmap_reg[i].base == base) {
            h = g_mmap_reg[i].mapping;
            g_mmap_reg[i].base = NULL;
            g_mmap_reg[i].mapping = NULL;
            break;
        }
    }
    ReleaseSRWLockExclusive(&g_mmap_reg_lock);
    return h;
}

void *ds4_win_mmap(void *addr, size_t len, int prot, int flags, int fd, long long off) {
    (void)addr;
    (void)flags; /* MAP_PRIVATE vs MAP_SHARED both map FILE_MAP_READ for ds4. */

    if ((prot & PROT_READ) == 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return MAP_FAILED;
    }
    HANDLE hfile = (HANDLE)_get_osfhandle(fd);
    if (hfile == INVALID_HANDLE_VALUE) {
        return MAP_FAILED;
    }

    /* CreateFileMapping needs a max size of 0 to use the file's size. */
    HANDLE hmap = CreateFileMappingW(hfile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hmap == NULL) {
        return MAP_FAILED;
    }

    DWORD hi = (DWORD)((unsigned long long)off >> 32);
    DWORD lo = (DWORD)(off & 0xFFFFFFFFu);
    void *view = MapViewOfFile(hmap, FILE_MAP_READ, hi, lo, len);
    if (view == NULL) {
        CloseHandle(hmap);
        return MAP_FAILED;
    }
    ds4_mmap_reg_add(view, hmap);
    return view;
}

int ds4_win_munmap(void *addr, size_t len) {
    (void)len;
    if (addr == NULL || addr == MAP_FAILED) return -1;
    HANDLE hmap = ds4_mmap_reg_take(addr);
    BOOL ok = UnmapViewOfFile(addr);
    if (hmap) CloseHandle(hmap);
    return ok ? 0 : -1;
}

int ds4_win_posix_madvise(void *addr, size_t len, int advice) {
    /* PrefetchVirtualMemory is the closest analogue to MADV_WILLNEED on
     * Windows 8+.  It is asynchronous and a hint, so calling it is harmless. */
    if (advice == POSIX_MADV_WILLNEED) {
        WIN32_MEMORY_RANGE_ENTRY entry;
        entry.VirtualAddress = addr;
        entry.NumberOfBytes  = len;
        if (PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0)) return 0;
        /* If the API isn't available or fails, treat as success: ds4 already
         * ignores madvise errors. */
        return 0;
    }
    if (advice == POSIX_MADV_DONTNEED) {
        /* OfferVirtualMemory requires the page to be allocated with specific
         * flags; on a file-backed mapping we cannot use it.  No-op. */
        return 0;
    }
    return 0;
}

/* -------------------------------------------------------------------------- */
/* flock                                                                      */
/* -------------------------------------------------------------------------- */

int ds4_win_flock(int fd, int op) {
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) {
        errno = EBADF;
        return -1;
    }
    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));

    if (op & LOCK_UN) {
        if (UnlockFileEx(h, 0, MAXDWORD, MAXDWORD, &ov)) return 0;
        errno = EIO;
        return -1;
    }

    DWORD flags = 0;
    if (op & LOCK_EX) flags |= LOCKFILE_EXCLUSIVE_LOCK;
    if (op & LOCK_NB) flags |= LOCKFILE_FAIL_IMMEDIATELY;

    if (LockFileEx(h, flags, 0, MAXDWORD, MAXDWORD, &ov)) return 0;

    DWORD err = GetLastError();
    if (err == ERROR_LOCK_VIOLATION || err == ERROR_IO_PENDING) {
        errno = EWOULDBLOCK;
    } else {
        errno = EIO;
    }
    return -1;
}

/* -------------------------------------------------------------------------- */
/* Threads                                                                    */
/* -------------------------------------------------------------------------- */

struct ds4_thread_start {
    void *(*fn)(void *);
    void  *arg;
};

static unsigned __stdcall ds4_thread_trampoline(void *p) {
    struct ds4_thread_start s = *(struct ds4_thread_start *)p;
    free(p);
    /* We don't expose the void* return value through pthread_join because
     * ds4 always passes NULL for the join retval.  The return value is
     * dropped here. */
    (void)s.fn(s.arg);
    return 0;
}

int ds4_win_pthread_create(pthread_t *out, const pthread_attr_t *attr,
                           void *(*fn)(void *), void *arg) {
    (void)attr;
    struct ds4_thread_start *s = (struct ds4_thread_start *)malloc(sizeof(*s));
    if (!s) return ENOMEM;
    s->fn  = fn;
    s->arg = arg;

    uintptr_t h = _beginthreadex(NULL, 0, ds4_thread_trampoline, s, 0, NULL);
    if (h == 0) {
        free(s);
        return errno ? errno : EAGAIN;
    }
    *out = (HANDLE)h;
    return 0;
}

int ds4_win_pthread_join(pthread_t t, void **retval) {
    if (retval) *retval = NULL;
    if (t == NULL) return EINVAL;
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
    return 0;
}

int ds4_win_pthread_detach(pthread_t t) {
    if (t == NULL) return EINVAL;
    CloseHandle(t);
    return 0;
}

static BOOL CALLBACK ds4_once_callback(PINIT_ONCE InitOnce, PVOID Parameter, PVOID *Context) {
    (void)InitOnce; (void)Context;
    void (*init)(void) = (void (*)(void))Parameter;
    init();
    return TRUE;
}

int ds4_win_pthread_once(pthread_once_t *once, void (*init)(void)) {
    BOOL ok = InitOnceExecuteOnce(&once->io, ds4_once_callback, (PVOID)init, NULL);
    return ok ? 0 : EINVAL;
}

int ds4_win_pthread_cond_timedwait(pthread_cond_t *c, pthread_mutex_t *m,
                                   const struct timespec *abstime) {
    /* Compute relative timeout from absolute time. */
    DWORD ms = INFINITE;
    if (abstime) {
        struct timespec now;
        timespec_get(&now, TIME_UTC);
        long long delta_ms = (long long)(abstime->tv_sec - now.tv_sec) * 1000
                           + (abstime->tv_nsec - now.tv_nsec) / 1000000;
        if (delta_ms <= 0) {
            ms = 0;
        } else if (delta_ms >= (long long)INFINITE) {
            ms = INFINITE - 1;
        } else {
            ms = (DWORD)delta_ms;
        }
    }
    if (SleepConditionVariableSRW(&c->cv, &m->srw, ms, 0)) return 0;
    return (GetLastError() == ERROR_TIMEOUT) ? ETIMEDOUT : EINVAL;
}

/* -------------------------------------------------------------------------- */
/* Winsock startup                                                            */
/* -------------------------------------------------------------------------- */

static INIT_ONCE g_wsa_once = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK ds4_wsa_init(PINIT_ONCE i, PVOID p, PVOID *c) {
    (void)i; (void)p; (void)c;
    WSADATA d;
    (void)WSAStartup(MAKEWORD(2, 2), &d);
    return TRUE;
}

void ds4_win_wsa_startup(void) {
    InitOnceExecuteOnce(&g_wsa_once, ds4_wsa_init, NULL, NULL);
}

void ds4_win_wsa_cleanup(void) {
    /* Intentionally no-op: process exit reclaims the WSA state, and the
     * inference server keeps sockets alive until terminated. */
}

/* -------------------------------------------------------------------------- */
/* /dev/urandom replacement                                                   */
/* -------------------------------------------------------------------------- */

void ds4_win_random_bytes(void *buf, size_t n) {
    /* BCryptGenRandom with BCRYPT_USE_SYSTEM_PREFERRED_RNG matches the
     * security level expected from /dev/urandom on Linux. */
    NTSTATUS st = BCryptGenRandom(NULL, (PUCHAR)buf, (ULONG)n,
                                  BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (st != 0) {
        /* Last-resort fallback: rand() is fine for the few non-security
         * call sites in ds4_server.c that use random_bytes (tool IDs are
         * 128 bit so collision risk is negligible even with a weak RNG). */
        unsigned char *p = (unsigned char *)buf;
        for (size_t i = 0; i < n; i++) p[i] = (unsigned char)rand();
    }
}

/* -------------------------------------------------------------------------- */
/* Temp-dir helper                                                            */
/* -------------------------------------------------------------------------- */

const char *ds4_win_tmp_path(char *buf, size_t cap, const char *suffix) {
    char tmp[MAX_PATH];
    DWORD n = GetTempPathA((DWORD)sizeof(tmp), tmp);
    if (n == 0 || n >= sizeof(tmp)) {
        snprintf(buf, cap, "C:\\Windows\\Temp\\%s", suffix ? suffix : "");
        return buf;
    }
    snprintf(buf, cap, "%s%s", tmp, suffix ? suffix : "");
    return buf;
}

/* -------------------------------------------------------------------------- */
/* clock_gettime                                                              */
/* -------------------------------------------------------------------------- */

int ds4_win_dprintf(int fd, const char *fmt, ...) {
    char stack[2048];
    char *buf = stack;
    va_list ap;
    va_start(ap, fmt);
    int n = _vscprintf(fmt, ap);
    va_end(ap);
    if (n < 0) return -1;
    if ((size_t)n + 1 > sizeof(stack)) {
        buf = (char *)malloc((size_t)n + 1);
        if (!buf) return -1;
    }
    va_start(ap, fmt);
    vsnprintf(buf, (size_t)n + 1, fmt, ap);
    va_end(ap);
    int w = _write(fd, buf, (unsigned)n);
    if (buf != stack) free(buf);
    return w;
}

struct ds4_win_dir {
    HANDLE          handle;
    WIN32_FIND_DATAA find;
    int             first;
    struct dirent   entry;
};

DIR *ds4_win_opendir(const char *path) {
    if (!path) return NULL;
    char pattern[MAX_PATH];
    size_t plen = strlen(path);
    if (plen + 3 > sizeof(pattern)) return NULL;
    memcpy(pattern, path, plen);
    /* Append \* (or /*) to make a search pattern. */
    if (plen > 0 && pattern[plen - 1] != '\\' && pattern[plen - 1] != '/') {
        pattern[plen++] = '\\';
    }
    pattern[plen++] = '*';
    pattern[plen]   = '\0';

    DIR *d = (DIR *)calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->handle = FindFirstFileA(pattern, &d->find);
    if (d->handle == INVALID_HANDLE_VALUE) {
        free(d);
        return NULL;
    }
    d->first = 1;
    return d;
}

struct dirent *ds4_win_readdir(DIR *d) {
    if (!d) return NULL;
    if (d->first) {
        d->first = 0;
    } else {
        if (!FindNextFileA(d->handle, &d->find)) return NULL;
    }
    strncpy(d->entry.d_name, d->find.cFileName, sizeof(d->entry.d_name) - 1);
    d->entry.d_name[sizeof(d->entry.d_name) - 1] = '\0';
    return &d->entry;
}

int ds4_win_closedir(DIR *d) {
    if (!d) return -1;
    if (d->handle != INVALID_HANDLE_VALUE) FindClose(d->handle);
    free(d);
    return 0;
}

long ds4_win_sysconf(int name) {
    if (name == _SC_PAGESIZE) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (long)si.dwPageSize;
    }
    if (name == _SC_NPROCESSORS_ONLN) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (long)si.dwNumberOfProcessors;
    }
    return -1;
}

ssize_t ds4_win_pread(int fd, void *buf, size_t n, long long off) {
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) {
        errno = EBADF;
        return -1;
    }
    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ov.Offset     = (DWORD)(off & 0xFFFFFFFF);
    ov.OffsetHigh = (DWORD)((unsigned long long)off >> 32);
    /* ReadFile with OVERLAPPED on a non-overlapped file handle still works:
     * the file pointer position is the one in OVERLAPPED.  This is the
     * standard pread emulation on Windows. */
    DWORD got = 0;
    if (!ReadFile(h, buf, (DWORD)n, &got, &ov)) {
        DWORD err = GetLastError();
        if (err == ERROR_HANDLE_EOF) return 0;
        errno = EIO;
        return -1;
    }
    return (ssize_t)got;
}

int ds4_win_clock_gettime(int clk, struct timespec *ts) {
    if (!ts) return -1;
    if (clk == CLOCK_REALTIME) {
        FILETIME ft;
        GetSystemTimeAsFileTime(&ft);
        ULARGE_INTEGER u;
        u.LowPart  = ft.dwLowDateTime;
        u.HighPart = ft.dwHighDateTime;
        /* FILETIME is 100ns ticks since 1601-01-01.  Shift to Unix epoch. */
        const unsigned long long EPOCH_DIFF_100NS = 116444736000000000ULL;
        unsigned long long ticks = u.QuadPart - EPOCH_DIFF_100NS;
        ts->tv_sec  = (time_t)(ticks / 10000000ULL);
        ts->tv_nsec = (long)((ticks % 10000000ULL) * 100);
        return 0;
    }
    /* MONOTONIC: QueryPerformanceCounter. */
    static LARGE_INTEGER freq;
    static int freq_ready;
    if (!freq_ready) {
        QueryPerformanceFrequency(&freq);
        freq_ready = 1;
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    ts->tv_sec  = (time_t)(now.QuadPart / freq.QuadPart);
    long long rem = now.QuadPart % freq.QuadPart;
    ts->tv_nsec = (long)((rem * 1000000000LL) / freq.QuadPart);
    return 0;
}

int ds4_win_gettimeofday(struct timeval *tv, void *tz) {
    (void)tz;
    if (!tv) return -1;
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    ULARGE_INTEGER u;
    u.LowPart  = ft.dwLowDateTime;
    u.HighPart = ft.dwHighDateTime;
    const unsigned long long EPOCH_DIFF_100NS = 116444736000000000ULL;
    unsigned long long ticks = u.QuadPart - EPOCH_DIFF_100NS;
    tv->tv_sec  = (long)(ticks / 10000000ULL);
    tv->tv_usec = (long)((ticks % 10000000ULL) / 10ULL);
    return 0;
}

/* -------------------------------------------------------------------------- */
/* Console raw mode                                                           */
/* -------------------------------------------------------------------------- */

int ds4_win_console_enter_raw(struct ds4_console_state *out) {
    HANDLE hin  = GetStdHandle(STD_INPUT_HANDLE);
    HANDLE hout = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hin == INVALID_HANDLE_VALUE || hout == INVALID_HANDLE_VALUE) return -1;

    DWORD in_mode, out_mode;
    if (!GetConsoleMode(hin, &in_mode) || !GetConsoleMode(hout, &out_mode)) return -1;
    out->in_mode  = in_mode;
    out->out_mode = out_mode;
    out->restored = 0;

    /* Match termios raw: disable line input, echo, processed input (so Ctrl+C
     * comes through as a key event), enable virtual terminal sequences on
     * output for ANSI escape codes used by ds4-eval. */
    DWORD new_in  = in_mode & ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT
                              | ENABLE_PROCESSED_INPUT);
    new_in |= ENABLE_VIRTUAL_TERMINAL_INPUT;
    DWORD new_out = out_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING
                             | DISABLE_NEWLINE_AUTO_RETURN;
    if (!SetConsoleMode(hin, new_in))  return -1;
    if (!SetConsoleMode(hout, new_out)) { SetConsoleMode(hin, in_mode); return -1; }
    return 0;
}

int ds4_win_console_leave_raw(const struct ds4_console_state *prev) {
    if (!prev || prev->restored) return 0;
    HANDLE hin  = GetStdHandle(STD_INPUT_HANDLE);
    HANDLE hout = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hin  != INVALID_HANDLE_VALUE) SetConsoleMode(hin,  prev->in_mode);
    if (hout != INVALID_HANDLE_VALUE) SetConsoleMode(hout, prev->out_mode);
    return 0;
}

int ds4_win_console_size(int *cols, int *rows) {
    HANDLE hout = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hout == INVALID_HANDLE_VALUE) return -1;
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (!GetConsoleScreenBufferInfo(hout, &info)) return -1;
    if (cols) *cols = info.srWindow.Right  - info.srWindow.Left + 1;
    if (rows) *rows = info.srWindow.Bottom - info.srWindow.Top  + 1;
    return 0;
}

int ds4_win_console_input_ready(void) {
    HANDLE hin = GetStdHandle(STD_INPUT_HANDLE);
    if (hin == INVALID_HANDLE_VALUE) return 0;
    DWORD events = 0;
    if (!GetNumberOfConsoleInputEvents(hin, &events)) return 0;
    return events > 0 ? 1 : 0;
}

#endif /* _WIN32 */
