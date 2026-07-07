#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <winsock2.h>
#include <ws2tcpip.h>

int main(void) {
    WSADATA wsa;
    int r = WSAStartup(MAKEWORD(2,2), &wsa);
    printf("WSAStartup: %d\n", r);

    SOCKET s = socket(AF_INET, SOCK_STREAM, 0);
    printf("socket() SOCKET=%llu  as int=%d\n", (unsigned long long)s, (int)s);
    if (s == INVALID_SOCKET) { printf("socket INVALID\n"); return 1; }

    int yes = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons(8000);
    inet_pton(AF_INET, "127.0.0.1", &sa.sin_addr);

    int br = bind(s, (struct sockaddr*)&sa, sizeof(sa));
    printf("bind: %d  WSAErr=%d  errno=%d  str='%s'\n",
           br, WSAGetLastError(), errno, strerror(errno));
    if (br != 0) { closesocket(s); WSACleanup(); return 1; }

    int lr = listen(s, 128);
    printf("listen: %d  WSAErr=%d  errno=%d  str='%s'\n",
           lr, WSAGetLastError(), errno, strerror(errno));

    closesocket(s);
    WSACleanup();
    return lr;
}
