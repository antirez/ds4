/* Minimal ioctl() shim for Windows (linenoise terminal support).
   This header provides ONLY terminal ioctl functions, NOT socket functions.
   It avoids including winsock2.h which causes header conflicts. */

#ifndef _COMPAT_WIN32_SYS_IOCTL_H_
#define _COMPAT_WIN32_SYS_IOCTL_H_

#include <windows.h>
#include <stdio.h>

/* Minimal ioctl command codes we support */
#ifndef TIOCGWINSZ
#  define TIOCGWINSZ 0x5413
#endif

/* Terminal window size structure */
#ifndef _HAVE_WINSIZE
#  define _HAVE_WINSIZE 1
struct winsize {
    unsigned short ws_row;
    unsigned short ws_col;
    unsigned short ws_xpixel;
    unsigned short ws_ypixel;
};
#endif

/* ioctl() for terminal operations (Windows version via GetConsoleScreenBufferInfo) */
static inline int ioctl(int fd, int request, ...) {
    va_list args;
    va_start(args, request);
    
    if (request == TIOCGWINSZ) {
        struct winsize *ws = va_arg(args, struct winsize *);
        if (ws == NULL) {
            va_end(args);
            return -1;
        }
        
        /* Get console handle from file descriptor.
           For stdin (0), stdout (1), stderr (2), use GetStdHandle. */
        HANDLE hConsole = INVALID_HANDLE_VALUE;
        if (fd == STDOUT_FILENO) {
            hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        } else if (fd == STDERR_FILENO) {
            hConsole = GetStdHandle(STD_ERROR_HANDLE);
        } else if (fd == STDIN_FILENO) {
            hConsole = GetStdHandle(STD_INPUT_HANDLE);
        } else {
            va_end(args);
            return -1;
        }
        
        if (hConsole == INVALID_HANDLE_VALUE) {
            va_end(args);
            return -1;
        }
        
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (!GetConsoleScreenBufferInfo(hConsole, &csbi)) {
            va_end(args);
            return -1;
        }
        
        /* Calculate window size from console buffer */
        ws->ws_col = (unsigned short)(csbi.srWindow.Right - csbi.srWindow.Left + 1);
        ws->ws_row = (unsigned short)(csbi.srWindow.Bottom - csbi.srWindow.Top + 1);
        ws->ws_xpixel = 0;
        ws->ws_ypixel = 0;
        
        va_end(args);
        return 0;
    }
    
    va_end(args);
    return -1;  /* Unsupported ioctl command */
}

#endif /* _COMPAT_WIN32_SYS_IOCTL_H_ */
