#ifndef DS4_TOOL_TEXT_H
#define DS4_TOOL_TEXT_H

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

/* Tool bodies are not HTML. Escape only their own closing delimiter and
 * spellings that would otherwise collide with that escape. */
static inline bool ds4_tool_text_escaped_close(const char *s, const char *end) {
    if (*s != '&') return false;
    s++;
    while (!strncmp(s, "amp;", 4)) s += 4;
    return !strncmp(s, "lt;", 3) && !strncmp(s + 3, end + 1, strlen(end + 1));
}

static inline bool ds4_tool_text_needs_escape(const char *s, const char *end) {
    return !strncmp(s, end, strlen(end)) || ds4_tool_text_escaped_close(s, end);
}

static inline void ds4_tool_text_unescape(char *s, const char *end) {
    char *out = s;
    while (*s) {
        if (ds4_tool_text_escaped_close(s, end)) {
            if (!strncmp(s, "&amp;", 5)) {
                *out++ = '&';
                s += 5;
            } else {
                *out++ = '<';
                s += 4;
            }
        } else {
            *out++ = *s++;
        }
    }
    *out = '\0';
}

/* A streaming fragment must not split an escape that could become a closing
 * delimiter. Ordinary entities pass through unchanged. */
static inline bool ds4_tool_text_partial_escape(const char *s, size_t n,
                                                const char *end) {
    if (!n || *s != '&') return false;
    s++;
    n--;
    while (n >= 4 && !memcmp(s, "amp;", 4)) {
        s += 4;
        n -= 4;
    }
    if (n < 4 && !memcmp(s, "amp;", n)) return true;
    if (n < 3) return !memcmp(s, "lt;", n);
    if (memcmp(s, "lt;", 3)) return false;
    s += 3;
    n -= 3;
    size_t tail = strlen(end + 1);
    return n < tail && !memcmp(s, end + 1, n);
}

#endif
