/* ds4_sockets_win.h — minimal Berkeley-sockets-over-Winsock shim.
 *
 * main's refactor moved the distributed runtime (ds4_distributed.c) into
 * CORE_OBJS, so it now links into *every* binary — including ds4-bench. That
 * file is a full TCP coordinator/worker transport written against the POSIX
 * sockets API (<arpa/inet.h>, <netdb.h>, <sys/socket.h>, <poll.h>, …). The
 * Windows HIP/MSVC-ABI build has no such headers; this shim supplies just the
 * surface ds4_distributed.c uses, mapped onto Winsock2 / ws2tcpip.
 *
 * Scope: enough for ds4_distributed.c to *compile and link* on Windows so that
 * ds4-bench.exe builds. The bench never enters distributed serving, so the
 * runtime fidelity of a few calls (notably dup() of a socket and WSAStartup
 * lifetime) is not exercised by the bench. Anything that would need real
 * runtime parity for `--role coordinator/worker` on Windows is called out in
 * win/README.md as a follow-up.
 *
 * Header-only, self-contained. Whole body guarded by _WIN32 (and not pulled in
 * by the MinGW CPU build, which is GPU-less and does not link the distributed
 * runtime into the bench), so POSIX builds are byte-for-byte unchanged.
 */
#ifndef DS4_SOCKETS_WIN_H
#define DS4_SOCKETS_WIN_H

#ifdef _WIN32

/* winsock2.h must precede windows.h; WIN32_LEAN_AND_MEAN stops a later
 * <windows.h> from dragging in the legacy <winsock.h>. Include the Winsock
 * headers first and let the include guards settle the order across TUs. */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>      /* if_nametoindex */
#include <io.h>            /* _close fallback for non-socket fds */
#include <signal.h>        /* signal/SIG_IGN (MSVC CRT; SIGPIPE absent) */
#include <errno.h>
#include <stdio.h>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")

/* ---- POSIX errno aliases for the Winsock failure codes ds4_distributed.c
 * inspects. recv/send/poll set errno via the wrappers below. -------------- */
#ifndef EINTR
#define EINTR        WSAEINTR
#endif
#ifndef EWOULDBLOCK
#define EWOULDBLOCK  WSAEWOULDBLOCK
#endif
#ifndef EINPROGRESS
#define EINPROGRESS  WSAEINPROGRESS
#endif

/* socklen_t is defined by ws2tcpip.h on recent SDKs; guard just in case. */
#ifndef _SOCKLEN_T_DEFINED
#ifndef socklen_t
typedef int socklen_t;
#endif
#endif

/* poll(): provided as WSAPoll on Windows Vista+. struct pollfd / POLL* and
 * nfds_t come from winsock2.h. SHUT_RDWR maps to SD_BOTH. */
#ifndef SHUT_RD
#define SHUT_RD   SD_RECEIVE
#define SHUT_WR   SD_SEND
#define SHUT_RDWR SD_BOTH
#endif
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0    /* Windows never raises SIGPIPE on a dead socket */
#endif

/* ds4_distributed.c calls poll() directly; route it to WSAPoll. */
#define poll(fds, n, timeout) WSAPoll((fds), (ULONG)(n), (int)(timeout))

/* close() is used on socket fds (overwhelmingly) and, in one path, on the fd
 * returned by mkstemp(). Try closesocket() first; if the descriptor is not a
 * socket, fall back to the CRT _close(). This keeps both cases correct. */
static __inline int ds4_win_close(int fd)
{
    if (closesocket((SOCKET)fd) == 0) return 0;
    if (WSAGetLastError() == WSAENOTSOCK) return _close(fd);
    return -1;
}
#define close(fd) ds4_win_close(fd)

/* signal(SIGPIPE, SIG_IGN): Windows has no SIGPIPE; make it a no-op. The
 * generic signal()/SIGINT path that the MSVC CRT *does* support is unaffected
 * because ds4_distributed.c only ever ignores SIGPIPE. */
#ifndef SIGPIPE
#define SIGPIPE 13
#endif

/* WSA bootstrap: ds4_distributed.c has no WSAStartup call (it is POSIX code).
 * Run it once on first socket use via a constructor-style guard. clang-cl
 * supports __attribute__((constructor)); fall back to lazy init otherwise. */
static __inline void ds4_win_wsa_startup(void)
{
    static volatile long started = 0;
    if (InterlockedCompareExchange(&started, 1, 0) == 0) {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
    }
}

/* Wrap the entry-point socket-creating calls so Winsock is initialized before
 * first use, and translate the last Winsock error into errno for the EINTR /
 * EWOULDBLOCK checks in ds4_distributed.c. */
static __inline SOCKET ds4_win_socket(int af, int type, int proto)
{
    ds4_win_wsa_startup();
    SOCKET s = socket(af, type, proto);
    if (s == INVALID_SOCKET) errno = WSAGetLastError();
    return s;
}
#define socket(af, type, proto) ds4_win_socket((af), (type), (proto))

static __inline int ds4_win_getaddrinfo(const char *node, const char *service,
                                        const struct addrinfo *hints,
                                        struct addrinfo **res)
{
    ds4_win_wsa_startup();
    return getaddrinfo(node, service, hints, res);
}
#define getaddrinfo(n, s, h, r) ds4_win_getaddrinfo((n), (s), (h), (r))

/* recv/send/accept/connect set errno from the Winsock error so the POSIX-style
 * `errno == EINTR` retry loops in ds4_distributed.c behave. */
static __inline int ds4_win_recv(int s, void *buf, size_t len, int flags)
{
    int r = recv((SOCKET)s, (char *)buf, (int)len, flags);
    if (r < 0) errno = WSAGetLastError();
    return r;
}
#define recv(s, b, l, f) ds4_win_recv((s), (b), (l), (f))

static __inline int ds4_win_send(int s, const void *buf, size_t len, int flags)
{
    int r = send((SOCKET)s, (const char *)buf, (int)len, flags & ~MSG_NOSIGNAL);
    if (r < 0) errno = WSAGetLastError();
    return r;
}
#define send(s, b, l, f) ds4_win_send((s), (b), (l), (f))

static __inline int ds4_win_accept(int s, struct sockaddr *addr, socklen_t *len)
{
    SOCKET a = accept((SOCKET)s, addr, len);
    if (a == INVALID_SOCKET) { errno = WSAGetLastError(); return -1; }
    return (int)a;
}
#define accept(s, a, l) ds4_win_accept((s), (a), (l))

static __inline int ds4_win_connect(int s, const struct sockaddr *addr, socklen_t len)
{
    int r = connect((SOCKET)s, addr, len);
    if (r != 0) errno = WSAGetLastError();
    return r;
}
#define connect(s, a, l) ds4_win_connect((s), (a), (l))

/* setsockopt: ds4_distributed.c passes plain pointers (int / struct timeval).
 * Winsock wants `const char *`; cast through. SO_RCVTIMEO/SO_SNDTIMEO take a
 * DWORD-milliseconds value on Windows rather than a struct timeval, but the
 * bench does not serve, so the (compiling) cast is sufficient here. */
static __inline int ds4_win_setsockopt(int s, int level, int opt,
                                       const void *val, socklen_t len)
{
    int r = setsockopt((SOCKET)s, level, opt, (const char *)val, len);
    if (r != 0) errno = WSAGetLastError();
    return r;
}
#define setsockopt(s, lvl, o, v, l) ds4_win_setsockopt((s), (lvl), (o), (v), (l))

/* dup() of a socket: a faithful port needs WSADuplicateSocket; the bench never
 * takes this path (coordinator-only). _dup keeps the link resolving and the
 * call type-correct; flagged as a serving-mode follow-up in win/README.md. */
#ifndef dup
#define dup(fd) _dup(fd)
#endif

#endif /* _WIN32 */
#endif /* DS4_SOCKETS_WIN_H */
