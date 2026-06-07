#include "ds4_acp.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4_ACP_JSON_MAX_NESTING 256

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} acp_buf;

static void acp_oom(const char *what) {
    perror(what);
    exit(1);
}

static void acp_buf_append(acp_buf *b, const char *s, size_t n) {
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 256;
        while (cap < b->len + n + 1) cap *= 2;
        char *p = realloc(b->ptr, cap);
        if (!p) acp_oom("ds4-acp: realloc");
        b->ptr = p;
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void acp_buf_puts(acp_buf *b, const char *s) {
    acp_buf_append(b, s, strlen(s));
}

static void acp_buf_putc(acp_buf *b, char c) {
    acp_buf_append(b, &c, 1);
}

static char *acp_buf_take(acp_buf *b) {
    if (!b->ptr) {
        char *p = malloc(1);
        if (!p) acp_oom("ds4-acp: malloc");
        p[0] = '\0';
        return p;
    }
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static char *acp_xstrndup(const char *s, size_t n) {
    char *p = malloc(n + 1);
    if (!p) acp_oom("ds4-acp: malloc");
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static bool acp_utf8_valid(const char *s, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned char c = (unsigned char)s[i++];
        if (c < 0x80) continue;
        int need = 0;
        if (c >= 0xc2 && c <= 0xdf) need = 1;
        else if (c >= 0xe0 && c <= 0xef) need = 2;
        else if (c >= 0xf0 && c <= 0xf4) need = 3;
        else return false;
        if (i + (size_t)need > n) return false;
        unsigned char c1 = (unsigned char)s[i];
        if (c == 0xe0 && c1 < 0xa0) return false;
        if (c == 0xed && c1 >= 0xa0) return false;
        if (c == 0xf0 && c1 < 0x90) return false;
        if (c == 0xf4 && c1 >= 0x90) return false;
        for (int j = 0; j < need; j++) {
            unsigned char cc = (unsigned char)s[i + (size_t)j];
            if ((cc & 0xc0) != 0x80) return false;
        }
        i += (size_t)need;
    }
    return true;
}

void ds4_acp_json_ws(const char **p) {
    while (**p && isspace((unsigned char)**p)) (*p)++;
}

static bool json_lit(const char **p, const char *lit) {
    size_t n = strlen(lit);
    if (strncmp(*p, lit, n) != 0) return false;
    *p += n;
    return true;
}

static int json_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool json_u16(const char **p, uint32_t *out) {
    if ((*p)[0] != '\\' || (*p)[1] != 'u') return false;
    uint32_t cp = 0;
    for (int i = 0; i < 4; i++) {
        int h = json_hex((*p)[2 + i]);
        if (h < 0) return false;
        cp = (cp << 4) | (uint32_t)h;
    }
    *p += 6;
    *out = cp;
    return true;
}

static void json_utf8(acp_buf *b, uint32_t cp) {
    char tmp[4];
    if (cp <= 0x7f) {
        tmp[0] = (char)cp;
        acp_buf_append(b, tmp, 1);
    } else if (cp <= 0x7ff) {
        tmp[0] = (char)(0xc0 | (cp >> 6));
        tmp[1] = (char)(0x80 | (cp & 0x3f));
        acp_buf_append(b, tmp, 2);
    } else if (cp <= 0xffff) {
        tmp[0] = (char)(0xe0 | (cp >> 12));
        tmp[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
        tmp[2] = (char)(0x80 | (cp & 0x3f));
        acp_buf_append(b, tmp, 3);
    } else {
        tmp[0] = (char)(0xf0 | (cp >> 18));
        tmp[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
        tmp[2] = (char)(0x80 | ((cp >> 6) & 0x3f));
        tmp[3] = (char)(0x80 | (cp & 0x3f));
        acp_buf_append(b, tmp, 4);
    }
}

bool ds4_acp_json_string(const char **p, char **out) {
    ds4_acp_json_ws(p);
    if (**p != '"') return false;
    (*p)++;
    acp_buf b = {0};
    while (**p && **p != '"') {
        unsigned char c = (unsigned char)**p;
        if (c < 0x20) goto fail;
        if (c == '\\') {
            (*p)++;
            switch (**p) {
            case '"': acp_buf_putc(&b, '"'); (*p)++; break;
            case '\\': acp_buf_putc(&b, '\\'); (*p)++; break;
            case '/': acp_buf_putc(&b, '/'); (*p)++; break;
            case 'b': acp_buf_putc(&b, '\b'); (*p)++; break;
            case 'f': acp_buf_putc(&b, '\f'); (*p)++; break;
            case 'n': acp_buf_putc(&b, '\n'); (*p)++; break;
            case 'r': acp_buf_putc(&b, '\r'); (*p)++; break;
            case 't': acp_buf_putc(&b, '\t'); (*p)++; break;
            case 'u': {
                (*p)--;
                uint32_t cp, lo = 0;
                if (!json_u16(p, &cp)) goto fail;
                if (cp >= 0xd800 && cp <= 0xdbff) {
                    const char *q = *p;
                    if (!json_u16(&q, &lo) ||
                        lo < 0xdc00 || lo > 0xdfff)
                        goto fail;
                    *p = q;
                    cp = 0x10000 + (((cp - 0xd800) << 10) | (lo - 0xdc00));
                } else if (cp >= 0xdc00 && cp <= 0xdfff) {
                    goto fail;
                }
                json_utf8(&b, cp);
                break;
            }
            default:
                goto fail;
            }
        } else {
            acp_buf_putc(&b, (char)c);
            (*p)++;
        }
    }
    if (**p != '"') goto fail;
    (*p)++;
    *out = acp_buf_take(&b);
    return true;

fail:
    free(b.ptr);
    return false;
}

static bool json_number(const char **p) {
    const char *s = *p;
    if (*s == '-') s++;
    if (*s == '0') {
        s++;
    } else if (isdigit((unsigned char)*s)) {
        while (isdigit((unsigned char)*s)) s++;
    } else {
        return false;
    }
    if (*s == '.') {
        s++;
        if (!isdigit((unsigned char)*s)) return false;
        while (isdigit((unsigned char)*s)) s++;
    }
    if (*s == 'e' || *s == 'E') {
        s++;
        if (*s == '+' || *s == '-') s++;
        if (!isdigit((unsigned char)*s)) return false;
        while (isdigit((unsigned char)*s)) s++;
    }
    *p = s;
    return true;
}

static bool json_skip_value_depth(const char **p, int depth);

static bool json_skip_array_depth(const char **p, int depth) {
    if (depth >= DS4_ACP_JSON_MAX_NESTING) return false;
    ds4_acp_json_ws(p);
    if (**p != '[') return false;
    (*p)++;
    ds4_acp_json_ws(p);
    if (**p == ']') {
        (*p)++;
        return true;
    }
    while (**p) {
        if (!json_skip_value_depth(p, depth + 1)) return false;
        ds4_acp_json_ws(p);
        if (**p == ']') {
            (*p)++;
            return true;
        }
        if (**p != ',') return false;
        (*p)++;
    }
    return false;
}

static bool json_skip_object_depth(const char **p, int depth) {
    if (depth >= DS4_ACP_JSON_MAX_NESTING) return false;
    ds4_acp_json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    ds4_acp_json_ws(p);
    if (**p == '}') {
        (*p)++;
        return true;
    }
    while (**p) {
        char *key = NULL;
        if (!ds4_acp_json_string(p, &key)) return false;
        free(key);
        ds4_acp_json_ws(p);
        if (**p != ':') return false;
        (*p)++;
        if (!json_skip_value_depth(p, depth + 1)) return false;
        ds4_acp_json_ws(p);
        if (**p == '}') {
            (*p)++;
            return true;
        }
        if (**p != ',') return false;
        (*p)++;
    }
    return false;
}

static bool json_skip_value_depth(const char **p, int depth) {
    ds4_acp_json_ws(p);
    if (**p == '"') {
        char *s = NULL;
        bool ok = ds4_acp_json_string(p, &s);
        free(s);
        return ok;
    }
    if (**p == '{') return json_skip_object_depth(p, depth);
    if (**p == '[') return json_skip_array_depth(p, depth);
    if (json_lit(p, "true") || json_lit(p, "false") || json_lit(p, "null"))
        return true;
    return json_number(p);
}

bool ds4_acp_json_skip_value(const char **p) {
    return json_skip_value_depth(p, 0);
}

bool ds4_acp_json_raw_value(const char **p, char **out) {
    ds4_acp_json_ws(p);
    const char *start = *p;
    if (!ds4_acp_json_skip_value(p)) return false;
    *out = acp_xstrndup(start, (size_t)(*p - start));
    return true;
}

static bool json_id_value(const char *raw) {
    const char *p = raw;
    ds4_acp_json_ws(&p);
    if (*p == '"' || *p == '-' || isdigit((unsigned char)*p)) return true;
    return !strcmp(p, "null");
}

bool ds4_acp_object_get_raw(const char *json, const char *key, char **out) {
    const char *p = json;
    ds4_acp_json_ws(&p);
    if (*p != '{') return false;
    p++;
    ds4_acp_json_ws(&p);
    while (*p && *p != '}') {
        char *k = NULL;
        if (!ds4_acp_json_string(&p, &k)) return false;
        ds4_acp_json_ws(&p);
        if (*p != ':') {
            free(k);
            return false;
        }
        p++;
        if (!strcmp(k, key)) {
            free(k);
            return ds4_acp_json_raw_value(&p, out);
        }
        free(k);
        if (!ds4_acp_json_skip_value(&p)) return false;
        ds4_acp_json_ws(&p);
        if (*p == ',') {
            p++;
            ds4_acp_json_ws(&p);
        } else if (*p != '}') {
            return false;
        }
    }
    return false;
}

bool ds4_acp_object_get_string(const char *json, const char *key, char **out) {
    char *raw = NULL;
    *out = NULL;
    if (!ds4_acp_object_get_raw(json, key, &raw)) return false;
    const char *p = raw;
    bool ok = ds4_acp_json_string(&p, out);
    ds4_acp_json_ws(&p);
    if (*p) ok = false;
    if (!ok) {
        free(*out);
        *out = NULL;
    }
    free(raw);
    return ok;
}

void ds4_acp_request_free(ds4_acp_request *r) {
    free(r->id_json);
    free(r->method);
    free(r->params_json);
    memset(r, 0, sizeof(*r));
}

ds4_acp_parse_result ds4_acp_parse_request(const char *json,
                                           ds4_acp_request *out,
                                           char *err, size_t err_len) {
    memset(out, 0, sizeof(*out));
    const char *p = json;
    bool saw_jsonrpc = false;
    bool saw_method = false;
    bool jsonrpc_ok = false;
    ds4_acp_json_ws(&p);
    if (*p != '{') {
        snprintf(err, err_len, "expected JSON object");
        return DS4_ACP_PARSE_JSON;
    }
    p++;
    ds4_acp_json_ws(&p);
    while (*p && *p != '}') {
        char *key = NULL;
        if (!ds4_acp_json_string(&p, &key)) {
            snprintf(err, err_len, "invalid object key");
            ds4_acp_request_free(out);
            return DS4_ACP_PARSE_JSON;
        }
        ds4_acp_json_ws(&p);
        if (*p != ':') {
            free(key);
            snprintf(err, err_len, "expected ':'");
            ds4_acp_request_free(out);
            return DS4_ACP_PARSE_JSON;
        }
        p++;
        if (!strcmp(key, "jsonrpc")) {
            char *v = NULL;
            if (!ds4_acp_json_string(&p, &v)) {
                free(key);
                snprintf(err, err_len, "invalid jsonrpc");
                ds4_acp_request_free(out);
                return DS4_ACP_PARSE_REQUEST;
            }
            saw_jsonrpc = true;
            jsonrpc_ok = !strcmp(v, "2.0");
            free(v);
        } else if (!strcmp(key, "id")) {
            free(out->id_json);
            out->id_json = NULL;
            if (!ds4_acp_json_raw_value(&p, &out->id_json)) {
                free(key);
                snprintf(err, err_len, "invalid id");
                ds4_acp_request_free(out);
                return DS4_ACP_PARSE_JSON;
            }
            out->has_id = true;
        } else if (!strcmp(key, "method")) {
            free(out->method);
            out->method = NULL;
            if (!ds4_acp_json_string(&p, &out->method)) {
                free(key);
                snprintf(err, err_len, "invalid method");
                ds4_acp_request_free(out);
                return DS4_ACP_PARSE_REQUEST;
            }
            saw_method = true;
        } else if (!strcmp(key, "params")) {
            free(out->params_json);
            out->params_json = NULL;
            if (!ds4_acp_json_raw_value(&p, &out->params_json)) {
                free(key);
                snprintf(err, err_len, "invalid params");
                ds4_acp_request_free(out);
                return DS4_ACP_PARSE_JSON;
            }
            out->has_params = true;
        } else if (!ds4_acp_json_skip_value(&p)) {
            free(key);
            snprintf(err, err_len, "invalid value");
            ds4_acp_request_free(out);
            return DS4_ACP_PARSE_JSON;
        }
        free(key);
        ds4_acp_json_ws(&p);
        if (*p == ',') {
            p++;
            ds4_acp_json_ws(&p);
        } else if (*p != '}') {
            snprintf(err, err_len, "expected ',' or '}'");
            ds4_acp_request_free(out);
            return DS4_ACP_PARSE_JSON;
        }
    }
    if (*p != '}') {
        snprintf(err, err_len, "unterminated object");
        ds4_acp_request_free(out);
        return DS4_ACP_PARSE_JSON;
    }
    p++;
    ds4_acp_json_ws(&p);
    if (*p) {
        snprintf(err, err_len, "trailing data");
        ds4_acp_request_free(out);
        return DS4_ACP_PARSE_JSON;
    }
    if (!saw_jsonrpc || !jsonrpc_ok || !saw_method ||
        (out->has_id && !json_id_value(out->id_json)))
    {
        snprintf(err, err_len, "invalid JSON-RPC request");
        ds4_acp_request_free(out);
        return DS4_ACP_PARSE_REQUEST;
    }
    return DS4_ACP_PARSE_OK;
}

char *ds4_acp_json_escape(const char *s, size_t n) {
    acp_buf b = {0};
    bool valid_utf8 = acp_utf8_valid(s, n);
    acp_buf_putc(&b, '"');
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"': acp_buf_puts(&b, "\\\""); break;
        case '\\': acp_buf_puts(&b, "\\\\"); break;
        case '\b': acp_buf_puts(&b, "\\b"); break;
        case '\f': acp_buf_puts(&b, "\\f"); break;
        case '\n': acp_buf_puts(&b, "\\n"); break;
        case '\r': acp_buf_puts(&b, "\\r"); break;
        case '\t': acp_buf_puts(&b, "\\t"); break;
        default:
            if (c < 0x20) {
                char tmp[8];
                snprintf(tmp, sizeof(tmp), "\\u%04x", c);
                acp_buf_puts(&b, tmp);
            } else if (!valid_utf8 && c >= 0x80) {
                char tmp[8];
                snprintf(tmp, sizeof(tmp), "\\u%04x", c);
                acp_buf_puts(&b, tmp);
            } else {
                acp_buf_putc(&b, (char)c);
            }
            break;
        }
    }
    acp_buf_putc(&b, '"');
    return acp_buf_take(&b);
}
