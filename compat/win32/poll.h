#ifndef _COMPAT_WIN32_POLL_H_
#define _COMPAT_WIN32_POLL_H_

#ifdef _WIN32

/* DO NOT INCLUDE WINSOCK2.H HERE - causes conflicts.
   Instead, this header must be included AFTER winsock2.h is already included,
   and poll() will use WSAPOLLFD which is defined by winsock2.h */

#ifndef POLLIN
#  define POLLIN   0x0001
#  define POLLOUT  0x0004
#  define POLLERR  0x0008
#  define POLLHUP  0x0010
#  define POLLNVAL 0x0020
#endif

/* Forward declare WSAPoll from winsock2 */
#ifdef _WIN32
extern int WSAAPI WSAPoll(__inout_ecount(nfds) struct pollfd FAR * fdArray, __in ULONG nfds, __in INT timeout);
#  define SOCKET_ERROR (-1)
#endif

static inline int poll(struct pollfd *fds, int nfds, int timeout)
{
    int result = WSAPoll(fds, nfds, timeout);
    if (result == SOCKET_ERROR) return -1;
    return result;
}

#endif /* _WIN32 */
#endif /* _COMPAT_WIN32_POLL_H_ */

