#ifndef _COMPAT_WIN32_DIRENT_H_
#define _COMPAT_WIN32_DIRENT_H_

/* Directory entry iteration for Windows */

#include <windows.h>
#include <string.h>
#include <io.h>
#include <stdlib.h>
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ino_t — inode number type 
   Note: MSVC's sys/stat.h uses _ino_t internally */
#ifndef _INO_T_DEFINED
#define _INO_T_DEFINED
typedef unsigned long _ino_t;
typedef unsigned long ino_t;
#endif

typedef struct {
    WIN32_FIND_DATAA data;
    HANDLE find_handle;
    int first;
} DIR;

struct dirent {
    ino_t d_ino;
    char d_name[MAX_PATH];
};

static DIR* opendir(const char* name)
{
    DIR* dir = (DIR*)malloc(sizeof(DIR));
    if (!dir) { errno = ENOMEM; return NULL; }
    
    char search_path[MAX_PATH + 2];
    snprintf(search_path, sizeof(search_path), "%s\\*", name);
    
    dir->find_handle = FindFirstFileA(search_path, &dir->data);
    if (dir->find_handle == INVALID_HANDLE_VALUE) {
        free(dir);
        errno = ENOENT;
        return NULL;
    }
    
    dir->first = 1;
    return dir;
}

static struct dirent* readdir(DIR* dir)
{
    static struct dirent entry;
    
    if (!dir || dir->find_handle == INVALID_HANDLE_VALUE) {
        errno = EBADF;
        return NULL;
    }
    
    BOOL found;
    if (dir->first) {
        found = TRUE;
        dir->first = 0;
    } else {
        found = FindNextFileA(dir->find_handle, &dir->data);
    }
    
    if (!found) {
        return NULL;
    }
    
    entry.d_ino = 0;
    strncpy(entry.d_name, dir->data.cFileName, sizeof(entry.d_name) - 1);
    entry.d_name[sizeof(entry.d_name) - 1] = '\0';
    
    return &entry;
}

static int closedir(DIR* dir)
{
    if (!dir) { errno = EBADF; return -1; }
    if (dir->find_handle != INVALID_HANDLE_VALUE) {
        FindClose(dir->find_handle);
    }
    free(dir);
    return 0;
}

static void rewinddir(DIR* dir)
{
    if (dir) { dir->first = 1; }
}

static long telldir(DIR* dir)
{
    (void)dir;
    return 0;
}

static void seekdir(DIR* dir, long pos)
{
    (void)dir;
    (void)pos;
}

#ifdef __cplusplus
}
#endif

#endif /* _COMPAT_WIN32_DIRENT_H_ */
