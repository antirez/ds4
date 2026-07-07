/*
 * compat/win32/sys/time.h  –  Minimal POSIX sys/time.h shim for Windows
 *
 * Provides struct timeval for socket code that needs it.
 */

#pragma once
#ifndef _WIN32_SYS_TIME_H_
#define _WIN32_SYS_TIME_H_

#ifdef _WIN32

#include <time.h>
#include <winsock2.h>  /* winsock2.h provides struct timeval on Windows */
#include <windows.h>
#include <stdint.h>

/* Socket shutdown constants */
#ifndef SD_RECEIVE
#  define SHUT_RD   SD_RECEIVE
#  define SHUT_WR   SD_SEND
#  define SHUT_RDWR SD_BOTH
#endif

/* gettimeofday — get current time (not precise on Windows but close enough) */
#ifndef _DS4_GETTIMEOFDAY_DEFINED
#define _DS4_GETTIMEOFDAY_DEFINED
#if defined(_MSC_VER) && !defined(__MINGW32__)
static inline int gettimeofday(struct timeval *tv, void *tz)
{
    (void)tz;  /* timezone parameter ignored on Windows */
    
    if (!tv) return -1;
    
    /* Get current time in 100-nanosecond intervals since 1601 */
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    
    /* Convert to 64-bit value */
    uint64_t t = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    
    /* Epoch difference between Windows (1601) and Unix (1970) in 100-ns units */
    static const uint64_t EPOCH_DIFF = 116444736000000000ULL;
    
    /* Convert to Unix time */
    if (t < EPOCH_DIFF) return -1;
    t = (t - EPOCH_DIFF) / 10;  /* Convert to microseconds */
    
    tv->tv_sec = (long)(t / 1000000);
    tv->tv_usec = (long)(t % 1000000);
    
    return 0;
}
#endif
#endif

#endif /* _WIN32 */
#endif /* _WIN32_SYS_TIME_H_ */
