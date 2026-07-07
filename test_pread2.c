#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>

int main() {
    const char *path = "C:/Users/wren/git/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";
    int fd = _open(path, _O_RDONLY | _O_BINARY);
    if (fd < 0) { printf("_open failed\n"); return 1; }
    HANDLE orig = (HANDLE)_get_osfhandle(fd);
    printf("orig handle: %p\n", orig);

    HANDLE h = ReOpenFile(orig, GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_WRITE, FILE_FLAG_OVERLAPPED);
    if (h == INVALID_HANDLE_VALUE) {
        printf("ReOpenFile FAILED: error=%lu\n", (unsigned long)GetLastError());
        _close(fd);
        return 1;
    }
    printf("ReOpenFile succeeded: %p\n", h);

    char buf[256];
    HANDLE ev = CreateEventW(NULL, TRUE, FALSE, NULL);
    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ov.hEvent = ev;
    ov.Offset = 0; ov.OffsetHigh = 0;
    DWORD nread = 0;
    BOOL ok = ReadFile(h, buf, sizeof(buf), &nread, &ov);
    if (!ok && GetLastError() == ERROR_IO_PENDING) {
        ok = GetOverlappedResult(h, &ov, &nread, TRUE);
    }
    printf("ReadFile@0: ok=%d nread=%lu err=%lu\n", (int)ok, (unsigned long)nread, (unsigned long)(ok?0:GetLastError()));
    CloseHandle(ev);

    /* Try at large offset (20 GiB) */
    HANDLE ev2 = CreateEventW(NULL, TRUE, FALSE, NULL);
    OVERLAPPED ov2;
    memset(&ov2, 0, sizeof(ov2));
    ov2.hEvent = ev2;
    uint64_t off = 20ULL*1024*1024*1024;
    ov2.Offset = (DWORD)(off & 0xFFFFFFFFUL);
    ov2.OffsetHigh = (DWORD)((off >> 32) & 0xFFFFFFFFUL);
    ok = ReadFile(h, buf, sizeof(buf), &nread, &ov2);
    if (!ok && GetLastError() == ERROR_IO_PENDING) {
        ok = GetOverlappedResult(h, &ov2, &nread, TRUE);
    }
    printf("ReadFile@20GiB: ok=%d nread=%lu err=%lu\n", (int)ok, (unsigned long)nread, (unsigned long)(ok?0:GetLastError()));
    CloseHandle(ev2);

    CloseHandle(h);
    _close(fd);
    return 0;
}
