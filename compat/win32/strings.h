#ifndef _COMPAT_WIN32_STRINGS_H_
#define _COMPAT_WIN32_STRINGS_H_

/* BSD string functions compatibility (use string.h instead) */

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Provide strcasecmp / strncasecmp as MSVC equivalents */
#if defined(_MSC_VER) && !defined(__MINGW32__)
#define strcasecmp(a, b) _stricmp((a), (b))
#define strncasecmp(a, b, n) _strnicmp((a), (b), (n))
#endif

#ifdef __cplusplus
}
#endif

#endif /* _COMPAT_WIN32_STRINGS_H_ */
