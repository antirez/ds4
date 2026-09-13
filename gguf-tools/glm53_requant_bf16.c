/*
 * Requantize GLM 5.3 dense BF16 tensors to a smaller type, in an existing GGUF.
 *
 * The shipped GLM-5.3-Flash Q4_K artifact stores blk.N.kda_{q,k,v,output},
 * output.weight and token_embd.weight as BF16 while its experts are Q4_K.
 * The KDA projections are dense -- every one is read on every decoded token --
 * so at 34 KDA layers they alone account for roughly 8.5 GiB of the per-token
 * read traffic, more than all routed experts combined.  output.weight is a
 * further full matvec per token.
 *
 * glm53_quantize.py already assigns q8_0 to exactly these groups on its q4
 * artifact (role="linear_attention", "embedding" and "output"), so the result
 * is a shape the loader and the generic matmul already accept.  This tool
 * produces it from an existing GGUF, without needing the source checkpoint.
 *
 * Everything other than the selected tensors is copied byte for byte, and the
 * quantization goes through the same quants.c facade the other tools use, so
 * the output differs from the input only in those tensors' type and payload.
 */

#include "quants.h"

#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    GV_U8 = 0, GV_I8 = 1, GV_U16 = 2, GV_I16 = 3, GV_U32 = 4, GV_I32 = 5,
    GV_F32 = 6, GV_BOOL = 7, GV_STR = 8, GV_ARR = 9, GV_U64 = 10,
    GV_I64 = 11, GV_F64 = 12
};

typedef struct {
    const char *name;
    uint64_t    name_len;
    uint32_t    n_dims;
    uint64_t    dims[4];
    uint64_t    ne;
    uint32_t    type;
    uint64_t    offset;
    uint32_t    new_type;
    uint64_t    new_offset;
    uint64_t    new_bytes;
} tinfo;

static const uint8_t *g_base, *g_cur, *g_end;

/* Set once the scratch output exists, so a die() anywhere below does not leave
 * a half-written file sitting next to the real one. */
static char *g_tmp_path;

static void die(const char *msg) __attribute__((noreturn));
static void die(const char *msg) {
    fprintf(stderr, "glm53-requant-bf16: %s\n", msg);
    if (g_tmp_path) unlink(g_tmp_path);
    exit(1);
}

/* Every size below is derived from attacker-controlled header fields, so each
 * multiply and add is checked rather than allowed to wrap into a small,
 * plausible-looking value that then passes a range test. */
static uint64_t mul_or_die(uint64_t a, uint64_t b) {
    if (a != 0 && b > UINT64_MAX / a) die("size overflow in the tensor table");
    return a * b;
}

static uint64_t add_or_die(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) die("size overflow in the tensor table");
    return a + b;
}

static uint64_t pad_or_die(uint64_t x, uint64_t n) {
    return add_or_die(x, (n - x % n) % n);
}

static void need(size_t n) {
    if ((size_t)(g_end - g_cur) < n) die("truncated gguf");
}

static uint32_t rd_u32(void) { need(4); uint32_t v; memcpy(&v, g_cur, 4); g_cur += 4; return v; }
static uint64_t rd_u64(void) { need(8); uint64_t v; memcpy(&v, g_cur, 8); g_cur += 8; return v; }

static const char *rd_str(uint64_t *len) {
    uint64_t n = rd_u64();
    need(n);
    const char *s = (const char *)g_cur;
    g_cur += n;
    if (len) *len = n;
    return s;
}

static size_t scalar_size(uint32_t t) {
    switch (t) {
        case GV_U8: case GV_I8: case GV_BOOL: return 1;
        case GV_U16: case GV_I16: return 2;
        case GV_U32: case GV_I32: case GV_F32: return 4;
        case GV_U64: case GV_I64: case GV_F64: return 8;
        default: return 0;
    }
}

/* Skips a metadata value, returning its u32 content when it is a plain u32
 * (used only to pick general.alignment out of the stream). */
static void skip_value(uint32_t t, int *is_u32, uint32_t *u32_out) {
    if (is_u32) *is_u32 = 0;
    if (t == GV_STR) { rd_str(NULL); return; }
    if (t == GV_ARR) {
        uint32_t et = rd_u32();
        uint64_t n  = rd_u64();
        if (et == GV_STR) {
            for (uint64_t i = 0; i < n; i++) rd_str(NULL);
        } else {
            size_t sz = scalar_size(et);
            if (!sz) die("array of unsupported element type");
            const uint64_t span = mul_or_die((uint64_t)sz, n);
            if (span > SIZE_MAX) die("metadata array is larger than this address space");
            need((size_t)span);
            g_cur += (size_t)span;
        }
        return;
    }
    size_t sz = scalar_size(t);
    if (!sz) die("unsupported metadata value type");
    if (t == GV_U32 && is_u32) { *is_u32 = 1; *u32_out = rd_u32(); return; }
    need(sz);
    g_cur += sz;
}

enum { SEL_KDA = 1u << 0, SEL_HEAD = 1u << 1, SEL_EMBD = 1u << 2 };

static int name_is(const char *name, uint64_t len, const char *want) {
    size_t wl = strlen(want);
    return len == wl && memcmp(name, want, wl) == 0;
}

static int name_ends(const char *name, uint64_t len, const char *suffix) {
    size_t sl = strlen(suffix);
    return len >= sl && memcmp(name + len - sl, suffix, sl) == 0;
}

/* The groups glm53_quantize.py's q4 artifact assigns to Q8_0: the
 * linear-attention projections (role="linear_attention") and the embedding and
 * output tensors (role="embedding"/"output"). */
static int selected(const char *name, uint64_t len, unsigned sel) {
    if (sel & SEL_KDA) {
        if (name_ends(name, len, ".kda_q.weight") ||
            name_ends(name, len, ".kda_k.weight") ||
            name_ends(name, len, ".kda_v.weight") ||
            name_ends(name, len, ".kda_output.weight")) return 1;
    }
    if ((sel & SEL_HEAD) && name_is(name, len, "output.weight")) return 1;
    if ((sel & SEL_EMBD) && name_is(name, len, "token_embd.weight")) return 1;
    return 0;
}

static unsigned parse_selection(const char *spec) {
    unsigned sel = 0;
    const char *p = spec;
    while (*p) {
        const char *comma = strchr(p, ',');
        size_t n = comma ? (size_t)(comma - p) : strlen(p);
        if      (n == 3 && !memcmp(p, "kda",  3)) sel |= SEL_KDA;
        else if (n == 4 && !memcmp(p, "head", 4)) sel |= SEL_HEAD;
        else if (n == 4 && !memcmp(p, "embd", 4)) sel |= SEL_EMBD;
        else if (n == 3 && !memcmp(p, "all",  3)) sel |= SEL_KDA | SEL_HEAD | SEL_EMBD;
        else die("--tensors takes a comma separated list of kda, head, embd, all");
        if (!comma) break;
        p = comma + 1;
    }
    if (!sel) die("--tensors selected nothing");
    return sel;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <in.gguf> <out.gguf> [--type q8_0|q4_K] [--tensors LIST]\n"
                "  Requantizes BF16 tensors that glm53_quantize.py's q4 artifact\n"
                "  assigns to q8_0.  LIST is a comma separated selection of:\n"
                "    kda   blk.N.kda_{q,k,v,output}.weight  (default)\n"
                "    head  output.weight\n"
                "    embd  token_embd.weight\n"
                "    all   all of the above\n", argv[0]);
        return 2;
    }
    const char *in_path = argv[1], *out_path = argv[2];
    const char *want = "q8_0";
    unsigned sel = SEL_KDA;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--type") && i + 1 < argc)         want = argv[++i];
        else if (!strcmp(argv[i], "--tensors") && i + 1 < argc) sel = parse_selection(argv[++i]);
        else die("unrecognised argument; run with no arguments for usage");
    }
    ds4q_type target;
    if      (!strcmp(want, "q8_0")) target = DS4Q_TYPE_Q8_0;
    else if (!strcmp(want, "q4_K")) target = DS4Q_TYPE_Q4_K;
    else die("target type must be q8_0 or q4_K");
    if (!ds4q_can_quantize(target)) die("quantizer cannot emit that type");

    int fd = open(in_path, O_RDONLY);
    if (fd < 0) die("cannot open input");
    struct stat st;
    if (fstat(fd, &st) != 0) die("cannot stat input");
    /* The input stays mmapped for the whole run, so an output that resolves to
     * the same file would pull the source out from under every read still to
     * come.  st_dev/st_ino catches the hard link and the symlink too, which a
     * string compare of the two paths would not. */
    struct stat out_st;
    if (stat(out_path, &out_st) == 0 &&
        out_st.st_dev == st.st_dev && out_st.st_ino == st.st_ino) {
        die("output resolves to the input; write to a new path instead");
    }
    const size_t in_size = (size_t)st.st_size;
    if (in_size < 24) die("input is too small to be a gguf file");
    void *map = mmap(NULL, in_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) die("cannot mmap input");
    g_base = (const uint8_t *)map;
    g_cur  = g_base;
    g_end  = g_base + in_size;

    need(4);
    if (memcmp(g_cur, "GGUF", 4) != 0) die("not a gguf file");
    g_cur += 4;
    const uint32_t version = rd_u32();
    if (version != 3) fprintf(stderr, "glm53-requant-bf16: warning: gguf version %u\n", version);
    const uint64_t n_tensors = rd_u64();
    const uint64_t n_kv      = rd_u64();
    /* A tensor-info entry costs at least 8+4+8+4+8 bytes and a kv pair at
     * least 8+4+1, so a count past these bounds is a corrupt header.  Reject
     * it here rather than at the calloc() it would otherwise size. */
    if (n_tensors > in_size / 32) die("implausible tensor count");
    if (n_kv > in_size / 13) die("implausible metadata count");

    uint32_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen; const char *key = rd_str(&klen);
        uint32_t vt = rd_u32();
        int is_u32 = 0; uint32_t v = 0;
        skip_value(vt, &is_u32, &v);
        if (is_u32 && klen == strlen("general.alignment") &&
            memcmp(key, "general.alignment", klen) == 0) {
            alignment = v ? v : 32;
        }
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 || alignment > 65536) {
        die("general.alignment is not a power of two in range");
    }
    const size_t kv_end = (size_t)(g_cur - g_base);

    tinfo *ts = calloc((size_t)n_tensors, sizeof(*ts));
    if (!ts) die("out of memory");
    for (uint64_t i = 0; i < n_tensors; i++) {
        tinfo *t = &ts[i];
        t->name = rd_str(&t->name_len);
        t->n_dims = rd_u32();
        if (t->n_dims == 0 || t->n_dims > 4) die("tensor with zero or more than 4 dimensions");
        t->ne = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->dims[d] = rd_u64();
            /* Both guard the ne/dims[0] divisions below and keep the product
             * from wrapping into a small, plausible-looking byte count. */
            if (t->dims[d] == 0) die("tensor with a zero-length dimension");
            /* ds4q_row_size takes an int64_t, so a dimension past INT64_MAX
             * would be reinterpreted as negative and silently return 0. */
            if (t->dims[d] > (uint64_t)INT64_MAX) die("tensor dimension out of range");
            t->ne = mul_or_die(t->ne, t->dims[d]);
        }
        t->type   = rd_u32();
        t->offset = rd_u64();
    }
    const size_t info_end  = (size_t)(g_cur - g_base);
    const size_t data_start = ds4q_pad(info_end, alignment);
    if (data_start > in_size) die("tensor data section starts past the end of the input");

    /* Plan: pick new types and lay the data section out again. */
    uint64_t cursor = 0, converted = 0, before = 0, after = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        tinfo *t = &ts[i];
        int convert = (t->type == DS4Q_TYPE_BF16) && selected(t->name, t->name_len, sel);
        /* row_size() returns 0 for a type this build does not know, for a row
         * that is not a whole number of blocks, and for anything out of range.
         * Treating that as "copy 0 bytes" would emit a file that still parses
         * but has quietly lost the payload, so stop instead. */
        const size_t row_bytes = ds4q_row_size((ds4q_type)t->type, (int64_t)t->dims[0]);
        if (row_bytes == 0) {
            fprintf(stderr, "glm53-requant-bf16: %.*s is type %" PRIu32 ", which this build cannot size\n",
                    (int)t->name_len, t->name, t->type);
            die("refusing to copy a tensor whose layout is unknown");
        }
        const uint64_t nrows = t->ne / t->dims[0];
        const uint64_t old_bytes = mul_or_die((uint64_t)row_bytes, nrows);
        if (t->offset > in_size - data_start ||
            old_bytes > in_size - data_start - t->offset) {
            die("tensor data runs past the end of the input");
        }
        if (convert && (t->dims[0] % (uint64_t)ds4q_block_size(target)) != 0) {
            fprintf(stderr, "glm53-requant-bf16: %.*s row %" PRIu64 " not a multiple of the block size; leaving as is\n",
                    (int)t->name_len, t->name, t->dims[0]);
            convert = 0;
        }
        t->new_type = convert ? (uint32_t)target : t->type;
        t->new_bytes = convert
            ? mul_or_die((uint64_t)ds4q_row_size(target, (int64_t)t->dims[0]), nrows)
            : old_bytes;
        cursor = pad_or_die(cursor, alignment);
        t->new_offset = cursor;
        cursor = add_or_die(cursor, t->new_bytes);
        if (convert) { converted++; before += old_bytes; after += t->new_bytes; }
    }
    if (!converted) die("no matching BF16 tensors found -- nothing to do");
    fprintf(stderr,
            "glm53-requant-bf16: %" PRIu64 " tensors -> %s, %.2f GiB -> %.2f GiB (saves %.2f GiB per full read)\n",
            converted, ds4q_type_name(target),
            before / 1073741824.0, after / 1073741824.0, (before - after) / 1073741824.0);

    /* Build the file beside its destination and rename it into place at the
     * end: out_path then either still holds whatever it held before, or holds
     * a complete result, and never a truncated one.
     *
     * The scratch name comes from mkstemp rather than the pid.  A predictable
     * name opened with fopen("wb") reintroduces exactly the bug the input/
     * output inode check above closes: if that path is a symlink or hard link
     * to the input, the open truncates the mapped source.  mkstemp picks an
     * unpredictable name and opens O_CREAT|O_EXCL, which neither follows a
     * symlink nor reuses an existing file. */
    const size_t tmp_len = strlen(out_path) + 8;
    g_tmp_path = malloc(tmp_len);
    if (!g_tmp_path) die("out of memory");
    snprintf(g_tmp_path, tmp_len, "%s.XXXXXX", out_path);
    const int out_fd = mkstemp(g_tmp_path);
    if (out_fd < 0) die("cannot create the scratch output");
    /* Belt and braces: confirm what we hold is a fresh regular file and is not
     * the input, before a single byte is written. */
    struct stat tmp_st;
    if (fstat(out_fd, &tmp_st) != 0) die("cannot stat the scratch output");
    if (!S_ISREG(tmp_st.st_mode) ||
        (tmp_st.st_dev == st.st_dev && tmp_st.st_ino == st.st_ino)) {
        die("scratch output is not a fresh regular file");
    }
    /* mkstemp creates 0600; a model file should follow the umask like any
     * other output this tool used to produce. */
    const mode_t mask = umask(0);
    (void)umask(mask);
    (void)fchmod(out_fd, (mode_t)(0666 & ~mask));
    FILE *out = fdopen(out_fd, "wb");
    if (!out) die("cannot open output");
    /* Header and metadata are copied verbatim; tensor-info entries keep their
     * width, so the data section still begins at the same offset. */
    if (fwrite(g_base, 1, kv_end, out) != kv_end) die("write failed");
    for (uint64_t i = 0; i < n_tensors; i++) {
        tinfo *t = &ts[i];
        fwrite(&t->name_len, 8, 1, out);
        fwrite(t->name, 1, t->name_len, out);
        fwrite(&t->n_dims, 4, 1, out);
        for (uint32_t d = 0; d < t->n_dims; d++) fwrite(&t->dims[d], 8, 1, out);
        fwrite(&t->new_type, 4, 1, out);
        if (fwrite(&t->new_offset, 8, 1, out) != 1) die("write failed");
    }
    static const uint8_t zeros[4096] = {0};
    size_t here = (size_t)ftello(out);
    if (here != info_end) die("tensor info section changed size unexpectedly");
    while (here < data_start) {
        size_t n = data_start - here;
        if (n > sizeof(zeros)) n = sizeof(zeros);
        fwrite(zeros, 1, n, out);
        here += n;
    }

    ds4q_quantize_init(target);
    const int64_t CHUNK = 256; /* rows per pass, keeps the f32 staging small */
    for (uint64_t i = 0; i < n_tensors; i++) {
        tinfo *t = &ts[i];
        const size_t want_at = data_start + t->new_offset;
        size_t at = (size_t)ftello(out);
        while (at < want_at) {
            size_t n = want_at - at;
            if (n > sizeof(zeros)) n = sizeof(zeros);
            fwrite(zeros, 1, n, out);
            at += n;
        }
        const uint8_t *src = g_base + data_start + t->offset;
        if (t->new_type == t->type) {
            if (fwrite(src, 1, t->new_bytes, out) != t->new_bytes) die("write failed");
            continue;
        }
        const int64_t ncols = (int64_t)t->dims[0];
        const int64_t nrows = (int64_t)(t->ne / t->dims[0]);
        float *f32 = malloc((size_t)ncols * CHUNK * sizeof(float));
        void  *qbuf = malloc((size_t)ds4q_row_size(target, ncols) * CHUNK);
        if (!f32 || !qbuf) die("out of memory");
        for (int64_t r = 0; r < nrows; r += CHUNK) {
            const int64_t rows = (r + CHUNK <= nrows) ? CHUNK : (nrows - r);
            const uint16_t *bf = (const uint16_t *)src + (size_t)r * ncols;
            for (int64_t k = 0; k < rows * ncols; k++) f32[k] = ds4q_bf16_to_f32(bf[k]);
            size_t wrote = ds4q_quantize_chunk(target, f32, qbuf, 0, rows, ncols, NULL);
            if (fwrite(qbuf, 1, wrote, out) != wrote) die("write failed");
        }
        free(f32);
        free(qbuf);
        fprintf(stderr, "  %.*s -> %s\n", (int)t->name_len, t->name, ds4q_type_name(target));
    }
    if (fclose(out) != 0) die("close failed");
    if (rename(g_tmp_path, out_path) != 0) die("cannot move the finished file into place");
    free(g_tmp_path);
    g_tmp_path = NULL;
    munmap(map, in_size);
    close(fd);
    fprintf(stderr, "glm53-requant-bf16: wrote %s\n", out_path);
    return 0;
}
