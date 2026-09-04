/* GLP (GGUF Layer Projection) reader for ds4's directional steering.
 *
 * Pure C99: no CUDA, no Metal, no ds4.c internals.  See ds4_glp.h for the
 * rationale and the spec reference.
 *
 * The GGUF parse here is intentionally a small independent implementation
 * rather than a call into ds4.c's loader.  ds4.c's parser is static, tied to
 * ds4_model, and calls ds4_die() on malformed input; a steering file is user
 * input on a code path that must be able to report a refusal and continue, and
 * every refusal in this file is covered by tests/test_glp.c without a
 * model. */

#include "ds4_glp.h"

#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define DS4_GLP_MAGIC        0x46554747u /* "GGUF", little endian. */
#define DS4_GLP_GGUF_VERSION 3
#define DS4_GLP_TENSOR_F32   0           /* ggml type id for F32. */
#define DS4_GLP_SPEC_VERSION 1

/* Directions are a few hundred KB of F32 by construction. A cap keeps a
 * malformed header from making us walk a multi-gigabyte tensor directory
 * looking for direction.<N>, and it is three orders of magnitude above any
 * real vector. */
#define DS4_GLP_MAX_FILE_BYTES (256ull * 1024ull * 1024ull)

enum {
    GLP_KV_UINT8   = 0,
    GLP_KV_INT8    = 1,
    GLP_KV_UINT16  = 2,
    GLP_KV_INT16   = 3,
    GLP_KV_UINT32  = 4,
    GLP_KV_INT32   = 5,
    GLP_KV_FLOAT32 = 6,
    GLP_KV_BOOL    = 7,
    GLP_KV_STRING  = 8,
    GLP_KV_ARRAY   = 9,
    GLP_KV_UINT64  = 10,
    GLP_KV_INT64   = 11,
    GLP_KV_FLOAT64 = 12,
};

typedef struct {
    const char *ptr;
    uint64_t    len;
} glp_str;

typedef struct {
    glp_str  key;
    uint32_t type;
    uint64_t value_pos;
} glp_kv;

typedef struct {
    glp_str  name;
    uint32_t ndim;
    uint64_t dim0;
    uint32_t type;
    uint64_t abs_offset;
} glp_tensor;

typedef struct {
    const uint8_t *base;
    uint64_t       size;
    uint64_t       pos;
    int            bad;
} glp_cursor;

typedef struct {
    int            fd;
    const uint8_t *map;
    uint64_t       size;
    uint64_t       alignment;
    uint64_t       tensor_data_pos;
    uint64_t       n_kv;
    uint64_t       n_tensors;
    glp_kv        *kv;
    glp_tensor    *tensors;
} glp_file;

/* ------------------------------------------------------------------ */
/* Errors                                                             */
/* ------------------------------------------------------------------ */

/* Every message names the file, what was wrong, and -- where the failure is
 * one that degrades silently rather than erroring -- what it would have cost
 * to apply the file anyway.  A refusal a user does not understand gets
 * worked around with the override flag, which defeats the point. */
static int glp_fail(char *err, size_t errlen, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

static int glp_fail(char *err, size_t errlen, const char *fmt, ...) {
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* Cursor                                                             */
/* ------------------------------------------------------------------ */

static int glp_has(glp_cursor *c, uint64_t n) {
    if (c->bad) return 0;
    if (n > c->size || c->pos > c->size - n) {
        c->bad = 1;
        return 0;
    }
    return 1;
}

static int glp_read(glp_cursor *c, void *dst, uint64_t n) {
    if (!glp_has(c, n)) return 0;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return 1;
}

static int glp_u32(glp_cursor *c, uint32_t *v) { return glp_read(c, v, 4); }
static int glp_u64(glp_cursor *c, uint64_t *v) { return glp_read(c, v, 8); }

static int glp_string(glp_cursor *c, glp_str *s) {
    uint64_t len;
    if (!glp_u64(c, &len)) return 0;
    if (!glp_has(c, len)) return 0;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return 1;
}

static uint64_t glp_scalar_size(uint32_t type) {
    switch (type) {
    case GLP_KV_UINT8:
    case GLP_KV_INT8:
    case GLP_KV_BOOL:    return 1;
    case GLP_KV_UINT16:
    case GLP_KV_INT16:   return 2;
    case GLP_KV_UINT32:
    case GLP_KV_INT32:
    case GLP_KV_FLOAT32: return 4;
    case GLP_KV_UINT64:
    case GLP_KV_INT64:
    case GLP_KV_FLOAT64: return 8;
    default:             return 0;
    }
}

/* GGUF allows arrays of arrays; depth is bounded so a hostile header cannot
 * recurse us off the stack. */
static int glp_skip_value(glp_cursor *c, uint32_t type, int depth) {
    if (depth > 4) {
        c->bad = 1;
        return 0;
    }
    if (type == GLP_KV_STRING) {
        glp_str s;
        return glp_string(c, &s);
    }
    if (type == GLP_KV_ARRAY) {
        uint32_t elem_type;
        uint64_t len;
        if (!glp_u32(c, &elem_type)) return 0;
        if (!glp_u64(c, &len)) return 0;
        if (elem_type == GLP_KV_STRING || elem_type == GLP_KV_ARRAY) {
            for (uint64_t i = 0; i < len; i++) {
                if (!glp_skip_value(c, elem_type, depth + 1)) return 0;
            }
            return 1;
        }
        const uint64_t sz = glp_scalar_size(elem_type);
        if (sz == 0) {
            c->bad = 1;
            return 0;
        }
        if (len != 0 && sz > UINT64_MAX / len) {
            c->bad = 1;
            return 0;
        }
        if (!glp_has(c, len * sz)) return 0;
        c->pos += len * sz;
        return 1;
    }
    const uint64_t sz = glp_scalar_size(type);
    if (sz == 0) {
        c->bad = 1;
        return 0;
    }
    if (!glp_has(c, sz)) return 0;
    c->pos += sz;
    return 1;
}

static uint64_t glp_align_up(uint64_t v, uint64_t a) {
    if (a == 0) return v;
    const uint64_t rem = v % a;
    return rem == 0 ? v : v + a - rem;
}

/* ------------------------------------------------------------------ */
/* File open / close                                                  */
/* ------------------------------------------------------------------ */

static void glp_close(glp_file *f) {
    if (!f) return;
    free(f->kv);
    free(f->tensors);
    if (f->map) munmap((void *)f->map, (size_t)f->size);
    if (f->fd >= 0) close(f->fd);
    memset(f, 0, sizeof(*f));
    f->fd = -1;
}

static int glp_open(glp_file *f, const char *path, char *err, size_t errlen) {
    memset(f, 0, sizeof(*f));
    f->fd = -1;

    if (!path || !path[0]) return glp_fail(err, errlen, "no steering file path");

    const int fd = open(path, O_RDONLY);
    if (fd < 0) return glp_fail(err, errlen, "%s: cannot open", path);

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return glp_fail(err, errlen, "%s: cannot stat", path);
    }
    if (st.st_size < 24) {
        close(fd);
        return glp_fail(err, errlen, "%s: too small to be a GGUF file", path);
    }
    if ((uint64_t)st.st_size > DS4_GLP_MAX_FILE_BYTES) {
        close(fd);
        return glp_fail(err, errlen,
                        "%s: %llu bytes is not a steering vector. A GLP file "
                        "carries no weights; ours are a few hundred KB of F32. "
                        "Did you pass the model?",
                        path, (unsigned long long)st.st_size);
    }

    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        return glp_fail(err, errlen, "%s: cannot mmap", path);
    }
    f->fd   = fd;
    f->map  = (const uint8_t *)map;
    f->size = (uint64_t)st.st_size;

    glp_cursor c = { f->map, f->size, 0, 0 };
    uint32_t magic = 0, version = 0;
    if (!glp_u32(&c, &magic) || !glp_u32(&c, &version) ||
        !glp_u64(&c, &f->n_tensors) || !glp_u64(&c, &f->n_kv)) {
        glp_close(f);
        return glp_fail(err, errlen, "%s: truncated GGUF header", path);
    }
    if (magic != DS4_GLP_MAGIC) {
        glp_close(f);
        return glp_fail(err, errlen, "%s: not a GGUF file", path);
    }
    if (version != DS4_GLP_GGUF_VERSION) {
        glp_close(f);
        return glp_fail(err, errlen,
                        "%s: GGUF v%u; GLP is specified on GGUF v3",
                        path, version);
    }
    /* Every KV entry serializes at least a key length, a type tag and a
     * one-byte value (13 bytes); every tensor directory entry at least a name
     * length, an ndim, one dim, a type and an offset (32). A count the
     * remaining bytes cannot minimally contain is a hostile header. Reject
     * before calloc so a small file cannot amplify itself into a multi-GB
     * allocation. */
    const uint64_t remaining = f->size - c.pos;
    if (f->n_kv > remaining / 13 || f->n_tensors > remaining / 32) {
        glp_close(f);
        return glp_fail(err, errlen, "%s: GGUF header counts exceed file size", path);
    }

    f->alignment = 32;
    if (f->n_kv) {
        f->kv = calloc((size_t)f->n_kv, sizeof(f->kv[0]));
        if (!f->kv) {
            glp_close(f);
            return glp_fail(err, errlen, "%s: out of memory", path);
        }
    }
    for (uint64_t i = 0; i < f->n_kv; i++) {
        glp_kv *kv = &f->kv[i];
        if (!glp_string(&c, &kv->key) || !glp_u32(&c, &kv->type)) {
            glp_close(f);
            return glp_fail(err, errlen, "%s: truncated GGUF metadata", path);
        }
        kv->value_pos = c.pos;
        if (kv->key.len == 17 && memcmp(kv->key.ptr, "general.alignment", 17) == 0 &&
            kv->type == GLP_KV_UINT32) {
            glp_cursor t = { f->map, f->size, kv->value_pos, 0 };
            uint32_t a = 0;
            if (glp_u32(&t, &a) && a != 0) f->alignment = a;
        }
        if (!glp_skip_value(&c, kv->type, 0)) {
            glp_close(f);
            return glp_fail(err, errlen, "%s: malformed GGUF metadata value for a key", path);
        }
    }

    if (f->n_tensors) {
        f->tensors = calloc((size_t)f->n_tensors, sizeof(f->tensors[0]));
        if (!f->tensors) {
            glp_close(f);
            return glp_fail(err, errlen, "%s: out of memory", path);
        }
    }
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        glp_tensor *t = &f->tensors[i];
        uint64_t dims[4] = {1, 1, 1, 1};
        if (!glp_string(&c, &t->name) || !glp_u32(&c, &t->ndim)) {
            glp_close(f);
            return glp_fail(err, errlen, "%s: truncated GGUF tensor directory", path);
        }
        if (t->ndim == 0 || t->ndim > 4) {
            glp_close(f);
            return glp_fail(err, errlen, "%s: tensor with %u dimensions", path, t->ndim);
        }
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!glp_u64(&c, &dims[d])) {
                glp_close(f);
                return glp_fail(err, errlen, "%s: truncated GGUF tensor dims", path);
            }
        }
        uint64_t rel = 0;
        if (!glp_u32(&c, &t->type) || !glp_u64(&c, &rel)) {
            glp_close(f);
            return glp_fail(err, errlen, "%s: truncated GGUF tensor directory", path);
        }
        t->dim0       = dims[0];
        t->abs_offset = rel; /* made absolute below, once the data start is known */
    }

    f->tensor_data_pos = glp_align_up(c.pos, f->alignment);
    if (f->tensor_data_pos > f->size) {
        glp_close(f);
        return glp_fail(err, errlen, "%s: tensor data starts past end of file", path);
    }
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        glp_tensor *t = &f->tensors[i];
        if (t->abs_offset > f->size - f->tensor_data_pos) {
            glp_close(f);
            return glp_fail(err, errlen, "%s: tensor offset past end of file", path);
        }
        t->abs_offset += f->tensor_data_pos;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Metadata accessors                                                 */
/* ------------------------------------------------------------------ */

static int glp_key_is(const glp_kv *kv, const char *key) {
    const size_t len = strlen(key);
    return kv->key.len == len && memcmp(kv->key.ptr, key, len) == 0;
}

static const glp_kv *glp_find(const glp_file *f, const char *key) {
    for (uint64_t i = 0; i < f->n_kv; i++) {
        if (glp_key_is(&f->kv[i], key)) return &f->kv[i];
    }
    return NULL;
}

/* Copy a string value into a fixed buffer.  Returns 1 if the key existed and
 * was a string.  Truncation is fine here: these fields are for the human
 * reading the startup line, never for a comparison that decides behaviour. */
static int glp_get_str(const glp_file *f, const char *key, char *out, size_t outlen) {
    if (outlen) out[0] = '\0';
    const glp_kv *kv = glp_find(f, key);
    if (!kv || kv->type != GLP_KV_STRING) return 0;
    glp_cursor c = { f->map, f->size, kv->value_pos, 0 };
    glp_str s;
    if (!glp_string(&c, &s)) return 0;
    if (!outlen) return 1;
    size_t n = s.len < outlen - 1 ? (size_t)s.len : outlen - 1;
    memcpy(out, s.ptr, n);
    out[n] = '\0';
    return 1;
}

static int glp_get_u32(const glp_file *f, const char *key, uint32_t *out) {
    const glp_kv *kv = glp_find(f, key);
    if (!kv) return 0;
    glp_cursor c = { f->map, f->size, kv->value_pos, 0 };
    if (kv->type == GLP_KV_UINT32) return glp_u32(&c, out);
    if (kv->type == GLP_KV_INT32) {
        int32_t v = 0;
        if (!glp_read(&c, &v, sizeof(v)) || v < 0) return 0;
        *out = (uint32_t)v;
        return 1;
    }
    if (kv->type == GLP_KV_UINT64) {
        uint64_t v = 0;
        if (!glp_u64(&c, &v) || v > UINT32_MAX) return 0;
        *out = (uint32_t)v;
        return 1;
    }
    return 0;
}

static int glp_get_f32(const glp_file *f, const char *key, float *out) {
    const glp_kv *kv = glp_find(f, key);
    if (!kv) return 0;
    glp_cursor c = { f->map, f->size, kv->value_pos, 0 };
    if (kv->type == GLP_KV_FLOAT32) return glp_read(&c, out, sizeof(*out));
    if (kv->type == GLP_KV_FLOAT64) {
        double v = 0.0;
        if (!glp_read(&c, &v, sizeof(v))) return 0;
        *out = (float)v;
        return 1;
    }
    return 0;
}

/* Any key in the pre-GLP internal namespace.  Files written before
 * 2026-08-22 used "dspark.*"; the rebrand renamed the keys with no
 * compatibility alias because no external files existed.  Detecting it is
 * worth the loop: such a file has the right tensors and no glp.mode, so
 * without this check it would be diagnosed as "an additive control vector"
 * and refused for the wrong reason. */
static int glp_has_legacy_namespace(const glp_file *f) {
    for (uint64_t i = 0; i < f->n_kv; i++) {
        if (f->kv[i].key.len > 7 && memcmp(f->kv[i].key.ptr, "dspark.", 7) == 0) return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Public: probe and hook names                                       */
/* ------------------------------------------------------------------ */

int ds4_glp_is_gguf(const char *path) {
    if (!path || !path[0]) return -1;
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    unsigned char m[4];
    const ssize_t got = read(fd, m, sizeof(m));
    close(fd);
    if (got != (ssize_t)sizeof(m)) return 0;
    return (m[0] == 'G' && m[1] == 'G' && m[2] == 'U' && m[3] == 'F') ? 1 : 0;
}

const char *ds4_glp_hook_name(ds4_glp_hook hook) {
    switch (hook) {
    case DS4_GLP_HOOK_RESID_POST_LAYER: return "residual_stream_post_layer";
    case DS4_GLP_HOOK_FFN_OUT:          return "ffn_out_pre_residual";
    case DS4_GLP_HOOK_ATTN_OUT:         return "attn_out_pre_residual";
    default:                            return "unknown";
    }
}

static ds4_glp_hook glp_hook_from_name(const char *name) {
    if (!name || !name[0]) return DS4_GLP_HOOK_UNKNOWN;
    if (!strcmp(name, "residual_stream_post_layer")) return DS4_GLP_HOOK_RESID_POST_LAYER;
    if (!strcmp(name, "ffn_out_pre_residual"))       return DS4_GLP_HOOK_FFN_OUT;
    if (!strcmp(name, "attn_out_pre_residual"))      return DS4_GLP_HOOK_ATTN_OUT;
    return DS4_GLP_HOOK_UNKNOWN;
}

/* ------------------------------------------------------------------ */
/* Spec checks shared by read_info and load                           */
/* ------------------------------------------------------------------ */

static int glp_cmp_u32(const void *a, const void *b) {
    const uint32_t x = *(const uint32_t *)a;
    const uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

/* Finiteness of a serialized float, tested on its bytes. This file is built
 * with -ffast-math, under which the compiler assumes every float-typed value
 * is finite and folds even memcpy/bit-test NaN checks away -- so the value
 * must stay integer-typed all the way through the test. NaN and Inf both
 * have all exponent bits set. */
static int glp_bytes_f32_finite(const void *p) {
    uint32_t bits;
    memcpy(&bits, p, sizeof(bits));
    return (bits & 0x7f800000u) != 0x7f800000u;
}

/* Parse "direction.<N>" and return N, or -1 if the name is not one. */
static long glp_direction_index(const glp_str *name) {
    static const char pfx[] = "direction.";
    const size_t pfxlen = sizeof(pfx) - 1;
    if (name->len <= pfxlen) return -1;
    if (memcmp(name->ptr, pfx, pfxlen) != 0) return -1;

    long v = 0;
    for (uint64_t i = pfxlen; i < name->len; i++) {
        const char ch = name->ptr[i];
        if (ch < '0' || ch > '9') return -1;
        v = v * 10 + (ch - '0');
        if (v > 1000000) return -1;
    }
    return v;
}

static int glp_read_metadata(const glp_file *f,
                             const char     *path,
                             ds4_glp_info   *info,
                             char           *err,
                             size_t          errlen) {
    memset(info, 0, sizeof(*info));
    info->rank = 1;

    /* (1) glp.mode.  Absence means "add" for a plain llama.cpp control
     * vector, which is the honest default -- but ds4 cannot apply "add", so
     * either way an absent mode is a refusal.  Separate the two so the
     * message names the actual situation. */
    if (!glp_get_str(f, "glp.mode", info->mode, sizeof(info->mode))) {
        if (glp_has_legacy_namespace(f)) {
            return glp_fail(err, errlen,
                            "%s: carries only pre-GLP \"dspark.*\" metadata. The keys "
                            "were renamed to \"glp.*\" with no compatibility alias; "
                            "re-export or re-download the vector.",
                            path);
        }
        return glp_fail(err, errlen,
                        "%s: no glp.mode. Absent means additive (h += v), which is "
                        "what every control vector written before this key existed "
                        "is -- and ds4 projects (h -= a(h.v)v). Refusing to guess: "
                        "applying a projective direction additively pushes every "
                        "token along the direction instead of removing it, and "
                        "nothing downstream would detect it.",
                        path);
    }
    if (strcmp(info->mode, "project") != 0) {
        return glp_fail(err, errlen,
                        "%s: glp.mode=\"%s\", but ds4 implements projective ablation "
                        "only (y -= scale * v * dot(v, y)). Refusing to apply rather "
                        "than falling back to a different operation.",
                        path, info->mode);
    }

    /* (2) glp.spec_version says which contract the glp.* keys are written to.
     * Deliberately distinct from general.version, which says which build of
     * the vector this is. */
    if (glp_get_u32(f, "glp.spec_version", &info->spec_version) &&
        info->spec_version > DS4_GLP_SPEC_VERSION) {
        return glp_fail(err, errlen,
                        "%s: glp.spec_version=%u, this build implements %d. A later "
                        "spec may change what these keys mean; refusing rather than "
                        "guessing.",
                        path, info->spec_version, DS4_GLP_SPEC_VERSION);
    }

    /* (3) rank.  ds4's steering buffer holds one direction per layer, and
     * rank>1 needs an orthonormal basis to avoid subtracting overlapping
     * components more than once. */
    glp_get_u32(f, "glp.rank", &info->rank);
    if (info->rank == 0) info->rank = 1;
    if (info->rank != 1) {
        return glp_fail(err, errlen,
                        "%s: glp.rank=%u. ds4 stores one direction per layer; "
                        "rank>1 is expressible in the container but not implemented "
                        "here.",
                        path, info->rank);
    }

    info->has_alpha_default = glp_get_f32(f, "glp.alpha_default", &info->alpha_default);
    if (info->has_alpha_default) {
        /* Test the serialized bytes, never the float value: see
         * glp_bytes_f32_finite. */
        const glp_kv *akv = glp_find(f, "glp.alpha_default");
        int finite = 1;
        if (akv && akv->type == GLP_KV_FLOAT32) {
            finite = glp_bytes_f32_finite(f->map + akv->value_pos);
        } else if (akv && akv->type == GLP_KV_FLOAT64) {
            uint64_t bits;
            memcpy(&bits, f->map + akv->value_pos, sizeof(bits));
            finite = (bits & 0x7ff0000000000000ull) != 0x7ff0000000000000ull;
        }
        if (!finite) {
            return glp_fail(err, errlen,
                            "%s: glp.alpha_default is not finite. It is the "
                            "scale the projection multiplies by; refusing rather "
                            "than steering by NaN.",
                            path);
        }
    }

    glp_get_str(f, "glp.hook_point", info->hook_name, sizeof(info->hook_name));
    info->hook = glp_hook_from_name(info->hook_name);

    /* Where the direction was captured, when the producer says.  Absent means
     * "same as hook_point" (a natively derived vector); a value that differs
     * marks a transferred vector -- surfaced as a warning at load and in
     * --dir-steering-info, never a refusal. */
    glp_get_str(f, "glp.derived_at", info->derived_at, sizeof(info->derived_at));

    glp_get_str(f, "general.base_model.0.name",         info->base_model,     sizeof(info->base_model));
    glp_get_str(f, "general.base_model.0.organization", info->base_org,       sizeof(info->base_org));
    glp_get_str(f, "general.base_model.0.version",      info->base_version,   sizeof(info->base_version));
    glp_get_str(f, "general.base_model.0.repo_url",     info->base_repo_url,  sizeof(info->base_repo_url));
    glp_get_str(f, "glp.content_sha256",                info->content_sha256, sizeof(info->content_sha256));
    glp_get_str(f, "glp.created",                       info->created,        sizeof(info->created));
    glp_get_str(f, "glp.method",                        info->method,         sizeof(info->method));
    glp_get_str(f, "glp.contrast",                      info->contrast,       sizeof(info->contrast));
    return 0;
}

/* Walk the tensor directory: validate every direction.<N> and resolve the
 * shape.  Does not read tensor bytes. */
static int glp_scan_tensors(const glp_file *f,
                            const char     *path,
                            ds4_glp_info   *info,
                            char           *err,
                            size_t          errlen) {
    uint32_t n_embd = 0, n_dirs = 0;
    uint32_t lo = UINT32_MAX, hi = 0;

    for (uint64_t i = 0; i < f->n_tensors; i++) {
        const glp_tensor *t = &f->tensors[i];
        const long idx = glp_direction_index(&t->name);
        if (idx < 0) continue;

        /* (4) direction.N applies at layer N, no offset, so layer 0 cannot be
         * expressed and direction.0 is invalid rather than "layer 0". */
        if (idx == 0) {
            return glp_fail(err, errlen,
                            "%s: direction.0 is invalid. direction.N applies at layer "
                            "N with no offset, so layer 0 cannot be expressed in this "
                            "container.",
                            path);
        }
        if (t->type != DS4_GLP_TENSOR_F32) {
            return glp_fail(err, errlen,
                            "%s: direction.%ld is ggml type %u, not F32. Directions "
                            "are a few hundred KB and must stay F32: the projection "
                            "runs in fp32 on the activation regardless of the model's "
                            "weight dtype.",
                            path, idx, t->type);
        }
        if (t->ndim != 1) {
            return glp_fail(err, errlen,
                            "%s: direction.%ld has %u dimensions, expected 1-D of "
                            "n_embd.",
                            path, idx, t->ndim);
        }
        if (t->dim0 == 0 || t->dim0 > UINT32_MAX) {
            return glp_fail(err, errlen, "%s: direction.%ld has length %llu",
                            path, idx, (unsigned long long)t->dim0);
        }
        if (n_embd == 0) {
            n_embd = (uint32_t)t->dim0;
        } else if ((uint32_t)t->dim0 != n_embd) {
            return glp_fail(err, errlen,
                            "%s: inconsistent n_embd across directions (%u then %llu)",
                            path, n_embd, (unsigned long long)t->dim0);
        }
        if (t->abs_offset > f->size ||
            (uint64_t)t->dim0 * 4ull > f->size - t->abs_offset) {
            return glp_fail(err, errlen, "%s: direction.%ld data past end of file",
                            path, idx);
        }
        /* The bytes are read as float * straight out of the mapping, so the
         * absolute offset must be float-aligned; anything else is an
         * unaligned scalar load. */
        if (t->abs_offset % 4 != 0) {
            return glp_fail(err, errlen,
                            "%s: direction.%ld offset %llu is not float-aligned",
                            path, idx, (unsigned long long)t->abs_offset);
        }
        if ((uint32_t)idx < lo) lo = (uint32_t)idx;
        if ((uint32_t)idx > hi) hi = (uint32_t)idx;
        n_dirs++;
    }

    if (n_dirs == 0) {
        return glp_fail(err, errlen,
                        "%s: no direction.<N> tensors. This is a GGUF file but not a "
                        "control vector.",
                        path);
    }
    info->n_embd    = n_embd;
    info->n_dirs    = n_dirs;
    info->layer_min = lo;
    info->layer_max = hi;

    /* (5) Cross-check the tensor names against glp.layer_ids_zero_based.  The
     * two are written from the same source, so a disagreement means a broken
     * exporter -- which is what happened to us once: an exporter wrote
     * direction.11..39 for directions derived at layers 10..38 while this
     * field still read 10..38.  The names are what execute, so the mismatch
     * shifted the whole stack one layer, and a one-layer shift degrades
     * rather than fails.  The file carried the evidence and nothing looked at
     * it.
     *
     * Compare the exact sets, not a summary: count/min/max alone cannot see a
     * duplicate direction.<N> (which would load twice, the second silently
     * overwriting the first row of the dense buffer) or a declared list that
     * differs only in the interior. */
    uint32_t *ids = malloc((size_t)n_dirs * sizeof(ids[0]));
    if (!ids) return glp_fail(err, errlen, "%s: out of memory", path);
    uint32_t n_ids = 0;
    for (uint64_t i = 0; i < f->n_tensors; i++) {
        const long idx = glp_direction_index(&f->tensors[i].name);
        if (idx >= 1) ids[n_ids++] = (uint32_t)idx;
    }
    qsort(ids, n_ids, sizeof(ids[0]), glp_cmp_u32);
    for (uint32_t i = 1; i < n_ids; i++) {
        if (ids[i] == ids[i - 1]) {
            free(ids);
            return glp_fail(err, errlen,
                            "%s: direction.%u appears twice. The second would "
                            "overwrite the first in the layer-indexed buffer.",
                            path, ids[i]);
        }
    }

    char declared[1024];
    if (glp_get_str(f, "glp.layer_ids_zero_based", declared, sizeof(declared)) &&
        declared[0]) {
        uint32_t d_n = 0;
        const char *p = declared;
        while (*p) {
            while (*p == ' ' || *p == ',') p++;
            if (!*p) break;
            if (*p < '0' || *p > '9') { d_n = 0; break; }
            while (*p >= '0' && *p <= '9') p++;
            d_n++;
        }
        uint32_t *d_ids = d_n ? malloc((size_t)d_n * sizeof(d_ids[0])) : NULL;
        if (d_n && !d_ids) {
            free(ids);
            return glp_fail(err, errlen, "%s: out of memory", path);
        }
        if (d_ids) {
            d_n = 0;
            p = declared;
            while (*p) {
                while (*p == ' ' || *p == ',') p++;
                if (!*p) break;
                uint32_t v = 0;
                while (*p >= '0' && *p <= '9') v = v * 10 + (uint32_t)(*p++ - '0');
                d_ids[d_n++] = v;
            }
            qsort(d_ids, d_n, sizeof(d_ids[0]), glp_cmp_u32);
        }
        if (d_ids && (d_n != n_ids || memcmp(d_ids, ids,
                                             (size_t)n_ids * sizeof(ids[0])) != 0)) {
            free(d_ids);
            free(ids);
            return glp_fail(err, errlen,
                            "%s: glp.layer_ids_zero_based does not match the "
                            "direction tensors' layer set. The tensor names are "
                            "what get applied, so this file would steer the "
                            "wrong layers. Re-export it.",
                            path);
        }
        free(d_ids);
    }
    free(ids);
    return 0;
}

/* (6) Hook point.  Both the enum value and the raw string matter: an
 * unrecognised name is as much a refusal as a recognised mismatch, because
 * either way we would be applying the direction somewhere it was not
 * calibrated. */
static const char *glp_hook_flag(ds4_glp_hook hook) {
    switch (hook) {
    case DS4_GLP_HOOK_RESID_POST_LAYER: return "--dir-steering-resid";
    case DS4_GLP_HOOK_FFN_OUT:          return "--dir-steering-ffn";
    case DS4_GLP_HOOK_ATTN_OUT:         return "--dir-steering-attn";
    default:                            return NULL;
    }
}

static int glp_check_hook(const char        *path,
                          const ds4_glp_info *info,
                          ds4_glp_hook       target,
                          int                allow_mismatch,
                          char              *err,
                          size_t             errlen) {
    if (!info->hook_name[0]) {
        /* No declared hook.  Pre-hook_point files exist and the field is not
         * in the required set, so this is a warning the caller prints, not a
         * refusal. */
        return 0;
    }
    if (info->hook == target) return 0;
    if (allow_mismatch) return 0;

    /* When the file's hook is one ds4 implements, the fix is to steer that
     * site, not to override: the file is fine, the site choice is not.  An
     * unrecognised hook has no such remedy. */
    const char *site_flag = glp_hook_flag(info->hook);
    if (site_flag) {
        return glp_fail(err, errlen,
                        /* Remedy first, rationale second: this message is written
                         * into a caller-sized buffer, and a truncation should cost
                         * the explanation, not the fix. */
                        "%s: glp.hook_point=\"%s\" but this projection applies at "
                        "\"%s\". Steer the declared site instead (%s), or pass "
                        "--dir-steering-allow-hook-mismatch and re-tune the scale. "
                        "These are different tensors, and the alpha was "
                        "calibrated at one site and does not transfer silently: "
                        "applying it here degrades rather than errors -- the "
                        "failure this field exists to prevent.",
                        path, info->hook_name, ds4_glp_hook_name(target),
                        site_flag);
    }
    return glp_fail(err, errlen,
                    "%s: glp.hook_point=\"%s\" is not a hook this build "
                    "implements (ds4 implements residual_stream_post_layer, "
                    "ffn_out_pre_residual, attn_out_pre_residual). Refusing to "
                    "apply the direction somewhere it was not calibrated; "
                    "--dir-steering-allow-hook-mismatch overrides this, and the "
                    "scale then has to be re-tuned.",
                    path, info->hook_name);
}

/* ------------------------------------------------------------------ */
/* Public: read_info                                                  */
/* ------------------------------------------------------------------ */

int ds4_glp_read_info(const char *path, ds4_glp_info *info, char *err, size_t errlen) {
    if (!info) return glp_fail(err, errlen, "ds4_glp_read_info: null info");
    if (errlen) err[0] = '\0';

    glp_file f;
    int rc = glp_open(&f, path, err, errlen);
    if (rc != 0) return rc;

    rc = glp_read_metadata(&f, path, info, err, errlen);
    if (rc == 0) rc = glp_scan_tensors(&f, path, info, err, errlen);

    /* Norms, for the inspect path only: enough to spot a direction that is not
     * unit (which silently rescales the effective alpha, quadratically) or a
     * slot that was left zero. */
    if (rc == 0) {
        info->norm_min = 0.0f;
        info->norm_max = 0.0f;
        int first = 1;
        for (uint64_t i = 0; i < f.n_tensors; i++) {
            const glp_tensor *t = &f.tensors[i];
            if (glp_direction_index(&t->name) < 1) continue;
            const float *src = (const float *)(const void *)(f.map + t->abs_offset);
            double sum = 0.0;
            for (uint64_t k = 0; k < t->dim0; k++) sum += (double)src[k] * (double)src[k];
            const float norm = (float)sqrt(sum);
            if (first) {
                info->norm_min = info->norm_max = norm;
                first = 0;
            } else {
                if (norm < info->norm_min) info->norm_min = norm;
                if (norm > info->norm_max) info->norm_max = norm;
            }
        }
    }

    glp_close(&f);
    return rc;
}

/* ------------------------------------------------------------------ */
/* Public: load                                                       */
/* ------------------------------------------------------------------ */

int ds4_glp_load(const char   *path,
                 ds4_glp_hook  target_hook,
                 int           allow_hook_mismatch,
                 float        *dirs,
                 uint32_t      n_layers,
                 uint32_t      n_embd,
                 ds4_glp_info *info,
                 char         *err,
                 size_t        errlen) {
    ds4_glp_info local;
    if (!info) info = &local;
    if (errlen) err[0] = '\0';
    if (!dirs || n_layers == 0 || n_embd == 0) {
        return glp_fail(err, errlen, "ds4_glp_load: bad destination buffer");
    }

    /* Zeroed up front and left zeroed on every failure path: a zero row is a
     * no-op projection, so a caller that ignores our return value degrades to
     * unsteered rather than to garbage. */
    memset(dirs, 0, (size_t)n_layers * n_embd * sizeof(dirs[0]));

    glp_file f;
    int rc = glp_open(&f, path, err, errlen);
    if (rc != 0) return rc;

    rc = glp_read_metadata(&f, path, info, err, errlen);
    if (rc == 0) rc = glp_scan_tensors(&f, path, info, err, errlen);
    if (rc == 0) {
        rc = glp_check_hook(path, info, target_hook, allow_hook_mismatch, err, errlen);
    }

    /* A transferred vector (captured at one site, calibrated for another) is
     * legal and loadable, but invisible without this line: nothing else in the
     * log would say the direction was estimated on a different distribution
     * than the one it is about to edit. */
    if (rc == 0 && info->derived_at[0] && info->hook_name[0] &&
        strcmp(info->derived_at, info->hook_name) != 0) {
        fprintf(stderr,
                "ds4: %s: transferred vector -- derived at \"%s\", applied at "
                "\"%s\". glp.alpha_default belongs to the apply site.\n",
                path, info->derived_at, info->hook_name);
    }

    /* Shape must match the model exactly.  n_embd and the layer count are the
     * two things quantisation preserves, which is why the same vector pairs
     * with any quantisation of the same base checkpoint -- and also why a
     * mismatch here means a different model, not a different quant. */
    if (rc == 0 && info->n_embd != n_embd) {
        rc = glp_fail(err, errlen,
                      "%s: directions are %u wide, this model's n_embd is %u. A "
                      "direction is specific to the checkpoint it was derived from.",
                      path, info->n_embd, n_embd);
    }
    if (rc == 0 && info->layer_max >= n_layers) {
        rc = glp_fail(err, errlen,
                      "%s: direction.%u would apply at layer %u, but this model has "
                      "%u steerable layers (0..%u).",
                      path, info->layer_max, info->layer_max, n_layers, n_layers - 1);
    }

    if (rc != 0) {
        glp_close(&f);
        memset(dirs, 0, (size_t)n_layers * n_embd * sizeof(dirs[0]));
        return rc;
    }

    info->norm_min = 0.0f;
    info->norm_max = 0.0f;
    int first = 1;

    for (uint64_t i = 0; i < f.n_tensors; i++) {
        const glp_tensor *t = &f.tensors[i];
        const long idx = glp_direction_index(&t->name);
        if (idx < 1) continue;

        /* direction.N at row N. No offset. The upstream loader reads the other
         * way -- common_control_vector_load_one() stores direction.N at data
         * offset (N-1)*n_embd, which looks like "N-1 is the layer" -- but
         * llama_adapter_cvec::apply() then fills tensors[il] from
         * (il-1)*n_embd, so the two -1s cancel and tensors[il] holds
         * direction.il, used at graph layer il. Confirmed by measurement
         * (tests/test-cvec-layer-map.cpp in the llama.cpp fork), not by
         * reading it a third time. */
        float *dst = dirs + (uint64_t)idx * n_embd;
        const float *src = (const float *)(const void *)(f.map + t->abs_offset);
        memcpy(dst, src, (size_t)n_embd * sizeof(dst[0]));

        /* A NaN or Inf in the direction makes every activation it touches
         * NaN: the dot is NaN, the subtraction is NaN, and the model's output
         * is NaN with no error raised. Refuse at load instead. The test reads
         * the mapping as integers: float-typed NaN checks fold away under
         * -ffast-math (see glp_bytes_f32_finite). */
        const uint32_t *src_bits = (const uint32_t *)(const void *)(f.map + t->abs_offset);
        int finite = 1;
        for (uint32_t k = 0; k < n_embd; k++) {
            if ((src_bits[k] & 0x7f800000u) == 0x7f800000u) { finite = 0; break; }
        }
        if (!finite) {
            glp_close(&f);
            memset(dirs, 0, (size_t)n_layers * n_embd * sizeof(dirs[0]));
            return glp_fail(err, errlen,
                            "%s: direction.%ld contains a NaN or Inf. The "
                            "projection multiplies the activation by dot(v, y); "
                            "a non-finite direction poisons every token.",
                            path, idx);
        }

        /* ds4's op removes the component exactly at scale 1 only for a unit
         * direction, and the projector v^ v^T is invariant to ||v||, so
         * rescaling here cannot change the intended operation -- whereas
         * leaving it off-unit scales the removal by ||v||^2. */
        double sum = 0.0;
        for (uint32_t k = 0; k < n_embd; k++) sum += (double)dst[k] * (double)dst[k];
        const float norm = (float)sqrt(sum);
        if (first) {
            info->norm_min = info->norm_max = norm;
            first = 0;
        } else {
            if (norm < info->norm_min) info->norm_min = norm;
            if (norm > info->norm_max) info->norm_max = norm;
        }
        if (norm <= 1e-12f) {
            glp_close(&f);
            memset(dirs, 0, (size_t)n_layers * n_embd * sizeof(dirs[0]));
            return glp_fail(err, errlen,
                            "%s: direction.%ld has norm %g. A zero direction cannot "
                            "be normalised and would steer nothing at that layer; "
                            "the file is malformed rather than partially covering.",
                            path, idx, (double)norm);
        }
        if (fabsf(norm - 1.0f) > 1e-3f) {
            const float inv = 1.0f / norm;
            for (uint32_t k = 0; k < n_embd; k++) dst[k] *= inv;
            info->n_renormalized++;
        }
    }

    glp_close(&f);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Public: print                                                      */
/* ------------------------------------------------------------------ */

static void glp_print_field(FILE *out, const char *label, const char *value) {
    if (value && value[0]) fprintf(out, "  %-14s %s\n", label, value);
}

static float glp_default_scale(const char *path, ds4_glp_hook hook,
                               const char *flag, float fallback,
                               int *adopted) {
    if (adopted) *adopted = 0;
    if (ds4_glp_is_gguf(path) != 1) return fallback;

    ds4_glp_info info;
    char err[64];
    if (ds4_glp_read_info(path, &info, err, sizeof(err)) != 0) return fallback;
    if (!info.has_alpha_default) return fallback;
    if (info.hook != hook) return fallback;

    /* Adopted even when the value is 0: a file that declares alpha_default=0
     * means "no steering by default", which is a value, not an absence. */
    fprintf(stderr,
            "ds4: %s defaulted to %g from glp.alpha_default\n",
            flag, (double)info.alpha_default);
    if (adopted) *adopted = 1;
    return info.alpha_default;
}

float ds4_glp_default_ffn_scale(const char *path, float fallback, int *adopted) {
    return glp_default_scale(path, DS4_GLP_HOOK_FFN_OUT,
                             "--dir-steering-ffn", fallback, adopted);
}

float ds4_glp_default_resid_scale(const char *path, float fallback, int *adopted) {
    return glp_default_scale(path, DS4_GLP_HOOK_RESID_POST_LAYER,
                             "--dir-steering-resid", fallback, adopted);
}

int ds4_glp_inspect_main(const char *path) {
    if (ds4_glp_is_gguf(path) != 1) {
        fprintf(stderr,
                "ds4: %s is not a GGUF file. The legacy raw steering format is "
                "a headerless blob of n_layers * n_embd floats and carries "
                "nothing to inspect.\n",
                path ? path : "(null)");
        return 2;
    }
    ds4_glp_info info;
    char err[768];
    if (ds4_glp_read_info(path, &info, err, sizeof(err)) != 0) {
        fprintf(stderr, "ds4: %s\n", err);
        return 1;
    }
    ds4_glp_print_info(stdout, path, &info);
    return 0;
}

void ds4_glp_print_info(FILE *out, const char *path, const ds4_glp_info *info) {
    if (!out || !info) return;

    fprintf(out, "GLP vector: %s\n", path ? path : "(unnamed)");
    fprintf(out, "  %-14s %s\n", "mode", info->mode[0] ? info->mode : "(absent)");
    fprintf(out, "  %-14s %u\n", "spec_version", info->spec_version);
    fprintf(out, "  %-14s %s%s\n", "hook_point",
            info->hook_name[0] ? info->hook_name : "(absent)",
            (info->hook_name[0] && info->hook == DS4_GLP_HOOK_UNKNOWN)
                ? "  [not a hook this build knows]" : "");
    if (info->derived_at[0]) {
        const int transferred = info->hook_name[0] &&
            strcmp(info->derived_at, info->hook_name) != 0;
        fprintf(out, "  %-14s %s%s\n", "derived_at", info->derived_at,
                transferred ? "  [TRANSFERRED: captured at a different site; "
                              "alpha_default belongs to the apply site]" : "");
    }
    if (info->has_alpha_default) {
        fprintf(out, "  %-14s %g\n", "alpha_default", (double)info->alpha_default);
    } else {
        fprintf(out, "  %-14s (absent)\n", "alpha_default");
    }
    fprintf(out, "  %-14s %u\n", "rank", info->rank);
    fprintf(out, "  %-14s %u directions, n_embd=%u, layers %u..%u\n",
            "coverage", info->n_dirs, info->n_embd, info->layer_min, info->layer_max);
    /* Coverage is the lever that moved our numbers -- 6 layers 18%,
     * 16 layers 3.8%, 29 layers 0.0% -- and alpha saturates above ~4, so
     * a file with few layers underperforms regardless of how it was derived.
     * Print it where anyone comparing two vectors will see it. */
    fprintf(out, "  %-14s %.6f .. %.6f%s\n", "direction norm",
            (double)info->norm_min, (double)info->norm_max,
            info->n_renormalized ? "  [rescaled to unit]" : "");

    glp_print_field(out, "base model", info->base_model);
    glp_print_field(out, "base org", info->base_org);
    glp_print_field(out, "base revision", info->base_version);
    glp_print_field(out, "repo", info->base_repo_url);
    glp_print_field(out, "method", info->method);
    glp_print_field(out, "contrast", info->contrast);
    glp_print_field(out, "created", info->created);
    /* content_sha256 covers tensor bytes only, not metadata: glp.created makes
     * the file non-reproducible, so the hash is what lets two people confirm
     * they hold the same direction regardless of when it was packaged. */
    glp_print_field(out, "content sha256", info->content_sha256);
}
