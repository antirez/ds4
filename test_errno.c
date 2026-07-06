/* Test what errno maps to "value too large" in our build */
#include "compat/win32/sys/file.h"
#include "compat/win32/unistd.h"
#include "compat/win32/sys/socket.h"
#include <stdio.h>
int main(void) {
    for (int i = 0; i < 150; i++) {
        const char *s = strerror(i);
        if (s && strstr(s, "large")) printf("errno=%d: %s\n", i, s);
    }
    /* Also test actual socket listen */
    WSADATA wsa; WSAStartup(MAKEWORD(2,2), &wsa);
    SOCKET s2 = socket(AF_INET, SOCK_STREAM, 0);
    printf("socket=%llu (int=%d)\n", (unsigned long long)s2, (int)s2);
    struct sockaddr_in sa = {0};
    sa.sin_family = AF_INET; sa.sin_port = htons(8001);
    inet_pton(AF_INET, "127.0.0.1", &sa.sin_addr);
    int b = bind(s2, (struct sockaddr*)&sa, sizeof(sa));
    printf("bind=%d WSA=%d errno=%d '%s'\n", b, WSAGetLastError(), errno, strerror(errno));
    int l = listen(s2, 128);
    printf("listen=%d WSA=%d errno=%d '%s'\n", l, WSAGetLastError(), errno, strerror(errno));
    close((int)s2);
    return 0;
}
