/*
 * compat/win32/sys/socket.h  –  Minimal POSIX sys/socket.h shim for Windows
 *
 * Provides socket constants and includes winsock2.h.
 */

#pragma once
#ifndef _WIN32_SYS_SOCKET_H_
#define _WIN32_SYS_SOCKET_H_

#ifdef _WIN32

#ifndef NOMINWINSOCK
#  define NOMINWINSOCK
#endif
#ifndef _WINSOCKAPI_
#  define _WINSOCKAPI_
#endif

/* Prevent winsock.h conflicts; include windows.h first with guards */
#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>

/* POSIX socket shutdown constants */
#ifndef SHUT_RD
#  define SHUT_RD    SD_RECEIVE
#  define SHUT_WR    SD_SEND
#  define SHUT_RDWR  SD_BOTH
#endif

/* poll() — use WSAPoll on Windows */
#ifndef _DS4_POLL_DEFINED
#define _DS4_POLL_DEFINED

/* Auto-initialize Winsock on first socket call.
 * WSAStartup is required on Windows before any socket operations.
 * We call it lazily so source files don't need to manage it explicitly. */
static inline int ds4_wsa_ensure_init(void) {
    static int s_inited = 0;
    if (!s_inited) {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
        s_inited = 1;
    }
    return 1;
}

/* Wrap socket() to auto-initialize Winsock */
#ifndef _DS4_SOCKET_WRAP
#define _DS4_SOCKET_WRAP
#ifdef socket
#  undef socket
#endif
static inline int ds4_socket(int af, int type, int protocol) {
    ds4_wsa_ensure_init();
    SOCKET s = socket(af, type, protocol);
    if (s == INVALID_SOCKET) { errno = EINVAL; return -1; }
    return (int)s;
}
#define socket ds4_socket
#endif

/* pollfd structure for poll() — winsock2 defines WSAPOLLFD, we'll use it */
#ifndef POLLIN
#  define POLLIN   0x0001
#  define POLLOUT  0x0004
#  define POLLERR  0x0008
#  define POLLHUP  0x0010
#  define POLLNVAL 0x0020
#endif

/* Use winsock2's WSAPOLLFD directly for poll() */
static inline int poll(struct pollfd *fds, int nfds, int timeout)
{
    /* On Windows, use WSAPoll which works with socket FDs and regular FDs */
    int result = WSAPoll((WSAPOLLFD *)fds, nfds, timeout);
    /* WSAPoll returns SOCKET_ERROR (-1) on error, not -1 */
    if (result == SOCKET_ERROR) return -1;
    return result;
}
#endif

/* if_nametoindex() — get interface index by name */
#ifndef _IFNAMSIZ
#  define _IFNAMSIZ 16
#endif
#ifndef _DS4_IF_NAMETOINDEX_DEFINED
#define _DS4_IF_NAMETOINDEX_DEFINED
static inline unsigned int if_nametoindex(const char *ifname)
{
    (void)ifname;
    /* On Windows, just return 1 (first interface index).
       A full implementation would use GetAdaptersInfo or GetIfEntry2,
       but for simple multicast/setsockopt usage, 1 is a safe default. */
    return 1;
}
#endif

#endif /* _WIN32 */
#endif /* _WIN32_SYS_SOCKET_H_ */
