#include "ds4_prompt_prefix.h"

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void prefix_error(char *error, size_t error_cap, const char *fmt, ...) {
    if (!error || error_cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, error_cap, fmt, ap);
    va_end(ap);
}

static bool prefix_marker_at(const char *data, size_t len, size_t pos,
                             ds4_prompt_prefix_role *role,
                             size_t *marker_len) {
    static const char user[] = "USER:";
    static const char assistant[] = "ASSISTANT:";
    if (len - pos >= sizeof(user) - 1 &&
        memcmp(data + pos, user, sizeof(user) - 1) == 0) {
        *role = DS4_PROMPT_PREFIX_USER;
        *marker_len = sizeof(user) - 1;
        return true;
    }
    if (len - pos >= sizeof(assistant) - 1 &&
        memcmp(data + pos, assistant, sizeof(assistant) - 1) == 0) {
        *role = DS4_PROMPT_PREFIX_ASSISTANT;
        *marker_len = sizeof(assistant) - 1;
        return true;
    }
    return false;
}

static char *prefix_copy_content(const char *data, size_t start, size_t end) {
    size_t len = end - start;
    char *content = malloc(len + 1);
    if (!content) return NULL;
    memcpy(content, data + start, len);
    content[len] = '\0';
    return content;
}

void ds4_prompt_prefix_free(ds4_prompt_prefix *prefix) {
    if (!prefix) return;
    for (size_t i = 0; i < prefix->count; i++)
        free(prefix->turns[i].content);
    free(prefix->turns);
    memset(prefix, 0, sizeof(*prefix));
}

int ds4_prompt_prefix_parse(ds4_prompt_prefix *out,
                            const char *data,
                            size_t len,
                            char *error,
                            size_t error_cap) {
    if (!out) {
        prefix_error(error, error_cap, "invalid prefix output");
        return -1;
    }
    memset(out, 0, sizeof(*out));
    if (!data || len == 0) {
        prefix_error(error, error_cap, "prefix file is empty");
        return -1;
    }
    if (memchr(data, '\0', len)) {
        prefix_error(error, error_cap, "prefix file contains a NUL byte");
        return -1;
    }

    size_t pos = 0;
    if (len >= 3 && (unsigned char)data[0] == 0xef &&
        (unsigned char)data[1] == 0xbb &&
        (unsigned char)data[2] == 0xbf) {
        pos = 3;
    }
    size_t line = 1;

    while (pos < len) {
        ds4_prompt_prefix_role role;
        size_t marker_len = 0;
        if (!prefix_marker_at(data, len, pos, &role, &marker_len)) {
            prefix_error(error, error_cap,
                         "line %zu must start with USER: or ASSISTANT:", line);
            ds4_prompt_prefix_free(out);
            return -1;
        }

        const ds4_prompt_prefix_role expected =
            (out->count % 2 == 0) ? DS4_PROMPT_PREFIX_USER
                                  : DS4_PROMPT_PREFIX_ASSISTANT;
        if (role != expected) {
            prefix_error(error, error_cap, "line %zu: expected %s:", line,
                         expected == DS4_PROMPT_PREFIX_USER ? "USER"
                                                            : "ASSISTANT");
            ds4_prompt_prefix_free(out);
            return -1;
        }

        size_t content_start = pos + marker_len;
        if (content_start < len &&
            (data[content_start] == ' ' || data[content_start] == '\t')) {
            content_start++;
        }

        size_t next = len;
        size_t next_line = line;
        size_t scan_line = line;
        for (size_t i = content_start; i < len; i++) {
            if (data[i] != '\n') continue;
            scan_line++;
            ds4_prompt_prefix_role ignored_role;
            size_t ignored_len;
            if (i + 1 < len &&
                prefix_marker_at(data, len, i + 1,
                                 &ignored_role, &ignored_len)) {
                next = i + 1;
                next_line = scan_line;
                break;
            }
        }

        size_t content_end = next;
        if (content_end > content_start && data[content_end - 1] == '\n') {
            content_end--;
            if (content_end > content_start && data[content_end - 1] == '\r')
                content_end--;
        }
        if (content_end == content_start) {
            prefix_error(error, error_cap, "line %zu has an empty %s turn",
                         line, role == DS4_PROMPT_PREFIX_USER ? "USER"
                                                              : "ASSISTANT");
            ds4_prompt_prefix_free(out);
            return -1;
        }

        if (out->count == SIZE_MAX / sizeof(out->turns[0])) {
            prefix_error(error, error_cap, "too many turns in prefix file");
            ds4_prompt_prefix_free(out);
            return -1;
        }
        ds4_prompt_prefix_turn *turns =
            realloc(out->turns, (out->count + 1) * sizeof(out->turns[0]));
        if (!turns) {
            prefix_error(error, error_cap, "out of memory reading prefix file");
            ds4_prompt_prefix_free(out);
            return -1;
        }
        out->turns = turns;
        char *content = prefix_copy_content(data, content_start, content_end);
        if (!content) {
            prefix_error(error, error_cap, "out of memory reading prefix file");
            ds4_prompt_prefix_free(out);
            return -1;
        }
        out->turns[out->count++] = (ds4_prompt_prefix_turn) {
            .role = role,
            .content = content,
        };

        if (next == len) {
            pos = len;
        } else {
            pos = next;
            line = next_line;
        }
    }

    if (out->count == 0) {
        prefix_error(error, error_cap, "prefix file contains no turns");
        return -1;
    }
    if (out->turns[out->count - 1].role != DS4_PROMPT_PREFIX_ASSISTANT) {
        prefix_error(error, error_cap,
                     "prefix file must end with an ASSISTANT turn");
        ds4_prompt_prefix_free(out);
        return -1;
    }
    return 0;
}

int ds4_prompt_prefix_load(ds4_prompt_prefix *out,
                           const char *path,
                           char *error,
                           size_t error_cap) {
    if (!out || !path || !path[0]) {
        prefix_error(error, error_cap, "invalid prefix file path");
        return -1;
    }
    memset(out, 0, sizeof(*out));

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        prefix_error(error, error_cap, "cannot open prefix file %s: %s",
                     path, strerror(errno));
        return -1;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        prefix_error(error, error_cap, "cannot seek prefix file %s: %s",
                     path, strerror(errno));
        fclose(fp);
        return -1;
    }
    long size = ftell(fp);
    if (size < 0 || (uintmax_t)size > SIZE_MAX - 1) {
        prefix_error(error, error_cap, "cannot size prefix file %s", path);
        fclose(fp);
        return -1;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        prefix_error(error, error_cap, "cannot rewind prefix file %s: %s",
                     path, strerror(errno));
        fclose(fp);
        return -1;
    }

    size_t len = (size_t)size;
    char *data = malloc(len + 1);
    if (!data) {
        prefix_error(error, error_cap, "out of memory reading prefix file");
        fclose(fp);
        return -1;
    }
    if (len && fread(data, 1, len, fp) != len) {
        prefix_error(error, error_cap, "cannot read prefix file %s", path);
        free(data);
        fclose(fp);
        return -1;
    }
    if (fclose(fp) != 0) {
        prefix_error(error, error_cap, "cannot close prefix file %s: %s",
                     path, strerror(errno));
        free(data);
        return -1;
    }
    data[len] = '\0';

    int rc = ds4_prompt_prefix_parse(out, data, len, error, error_cap);
    free(data);
    if (rc != 0 && error && error_cap && error[0]) {
        char detail[256];
        snprintf(detail, sizeof(detail), "%s", error);
        prefix_error(error, error_cap, "%s: %s", path, detail);
    }
    return rc;
}
