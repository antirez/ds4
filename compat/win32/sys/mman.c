/*
 * sys/mman.c - mmap/munmap/mlock/munlock for Windows
 *
 * Based on mman-win32 (MIT License)
 * https://github.com/alitrack/mman-win32
 */
#include "mman.h"

#ifdef _WIN32

#include <windows.h>
#include <errno.h>
#include <io.h>

#ifndef FILE_MAP_EXECUTE
#define FILE_MAP_EXECUTE 0x0020
#endif

static DWORD mman_prot_to_page(int prot)
{
    if (prot == PROT_NONE) return 0;
    if (prot & PROT_EXEC)
        return (prot & PROT_WRITE) ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ;
    return (prot & PROT_WRITE) ? PAGE_READWRITE : PAGE_READONLY;
}

static DWORD mman_prot_to_access(int prot)
{
    DWORD access = 0;
    if (prot == PROT_NONE) return access;
    if (prot & PROT_READ)  access |= FILE_MAP_READ;
    if (prot & PROT_WRITE) access |= FILE_MAP_WRITE;
    if (prot & PROT_EXEC)  access |= FILE_MAP_EXECUTE;
    return access;
}

void *mmap(void *addr, size_t len, int prot, int flags, int fildes, OffsetType off)
{
    HANDLE fm, h;
    void *map = MAP_FAILED;

    const DWORD offLow  = (DWORD)(off & 0xFFFFFFFFUL);
    const DWORD offHigh = (sizeof(OffsetType) > 4) ? (DWORD)((off >> 32) & 0xFFFFFFFFUL) : 0;
    const DWORD protect = mman_prot_to_page(prot);
    const DWORD access  = mman_prot_to_access(prot);
    const OffsetType maxSize = off + (OffsetType)len;
    const DWORD maxLow  = (DWORD)(maxSize & 0xFFFFFFFFUL);
    const DWORD maxHigh = (sizeof(OffsetType) > 4) ? (DWORD)((maxSize >> 32) & 0xFFFFFFFFUL) : 0;

    errno = 0;
    if (len == 0 || prot == PROT_EXEC) { errno = EINVAL; return MAP_FAILED; }

    h = (flags & MAP_ANONYMOUS) ? INVALID_HANDLE_VALUE
                                : (HANDLE)_get_osfhandle(fildes);
    if (h == INVALID_HANDLE_VALUE && !(flags & MAP_ANONYMOUS)) {
        errno = EBADF; return MAP_FAILED;
    }

    fm = CreateFileMapping(h, NULL, protect, maxHigh, maxLow, NULL);
    if (!fm) { errno = EPERM; return MAP_FAILED; }

    map = (flags & MAP_FIXED)
        ? MapViewOfFileEx(fm, access, offHigh, offLow, len, addr)
        : MapViewOfFile(fm, access, offHigh, offLow, len);

    CloseHandle(fm);
    if (!map) { errno = EPERM; return MAP_FAILED; }
    return map;
}

int munmap(void *addr, size_t len)
{
    (void)len;
    return UnmapViewOfFile(addr) ? 0 : (errno = EPERM, -1);
}

int msync(void *addr, size_t len, int flags)
{
    (void)flags;
    return FlushViewOfFile(addr, len) ? 0 : (errno = EPERM, -1);
}

int mlock(const void *addr, size_t len)
{
    return VirtualLock((LPVOID)addr, len) ? 0 : (errno = EPERM, -1);
}

int munlock(const void *addr, size_t len)
{
    return VirtualUnlock((LPVOID)addr, len) ? 0 : (errno = EPERM, -1);
}

#endif /* _WIN32 */
