/*
 * sys/mman.h - mmap/munmap/mlock/munlock for Windows
 *
 * Based on mman-win32 (MIT License)
 * https://github.com/alitrack/mman-win32
 *
 * Implements POSIX memory-mapping over CreateFileMapping / MapViewOfFile.
 * Deliberately does NOT define POSIX_MADV_* so the existing
 * #if defined(POSIX_MADV_WILLNEED) guards in ds4.c remain compiled out —
 * those are advisory hints only and skipping them is correct on Windows.
 */
#pragma once
#ifndef _WIN32_SYS_MMAN_H_
#define _WIN32_SYS_MMAN_H_

#ifdef _WIN32

#include <stdint.h>
#include <sys/types.h>

/* Determine the offset type for mmap based on pointer size. */
#if defined(_WIN64)
typedef int64_t OffsetType;
#else
typedef uint32_t OffsetType;
#endif

#define PROT_NONE   0
#define PROT_READ   1
#define PROT_WRITE  2
#define PROT_EXEC   4

#define MAP_FILE      0
#define MAP_SHARED    1
#define MAP_PRIVATE   2
#define MAP_TYPE      0xf
#define MAP_FIXED     0x10
#define MAP_ANONYMOUS 0x20
#define MAP_ANON      MAP_ANONYMOUS

#define MAP_FAILED ((void *)-1)

#define MS_ASYNC      1
#define MS_SYNC       2
#define MS_INVALIDATE 4

#ifdef __cplusplus
extern "C" {
#endif

void *mmap(void *addr, size_t len, int prot, int flags, int fildes, OffsetType off);
int   munmap(void *addr, size_t len);
int   msync(void *addr, size_t len, int flags);
int   mlock(const void *addr, size_t len);
int   munlock(const void *addr, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* _WIN32 */
#endif /* _WIN32_SYS_MMAN_H_ */
