#include "ds4_engine_internal.h"



/* Positional rows (C++ has no array designators); the /(n)/ comments carry
 * the GGUF type ids, and the {NULL, 0, 0} rows are the id gaps. */
static const gguf_type_info gguf_types[] = {
    /* 0*/ {"f32",      1,   4},
    /* 1*/ {"f16",      1,   2},
    /* 2*/ {"q4_0",    32,  18},
    /* 3*/ {"q4_1",    32,  20},
    /* 4*/ {NULL,       0,   0},
    /* 5*/ {NULL,       0,   0},
    /* 6*/ {"q5_0",    32,  22},
    /* 7*/ {"q5_1",    32,  24},
    /* 8*/ {"q8_0",    32,  34},
    /* 9*/ {"q8_1",    32,  40},
    /*10*/ {"q2_k",   256,  84},
    /*11*/ {"q3_k",   256, 110},
    /*12*/ {"q4_k",   256, 144},
    /*13*/ {"q5_k",   256, 176},
    /*14*/ {"q6_k",   256, 210},
    /*15*/ {"q8_k",   256, 292},
    /*16*/ {"iq2_xxs",256,  66},
    /*17*/ {"iq2_xs", 256,  74},
    /*18*/ {"iq3_xxs",256,  98},
    /*19*/ {"iq1_s",  256,  50},
    /*20*/ {"iq4_nl",  32,  18},
    /*21*/ {"iq3_s",  256, 110},
    /*22*/ {"iq2_s",  256,  82},
    /*23*/ {"iq4_xs", 256, 136},
    /*24*/ {"i8",       1,   1},
    /*25*/ {"i16",      1,   2},
    /*26*/ {"i32",      1,   4},
    /*27*/ {"i64",      1,   8},
    /*28*/ {"f64",      1,   8},
    /*29*/ {"iq1_m",  256,  56},
    /*30*/ {"bf16",     1,   2},
    /*31*/ {NULL,       0,   0},
    /*32*/ {NULL,       0,   0},
    /*33*/ {NULL,       0,   0},
    /*34*/ {NULL,       0,   0},
    /*35*/ {NULL,       0,   0},
    /*36*/ {NULL,       0,   0},
    /*37*/ {NULL,       0,   0},
    /*38*/ {"fp8_e4m3", 32,  33},   /* MXFP8: E4M3 + per-32 E8M0 (8.25 bpw) */
    /*39*/ {"mxfp4",    32,  17},   /* MXFP4: E2M1 (2/byte) + per-32 E8M0 (4.25 bpw) */
    /* CUTLASS block-scaled MXFP4 (see cutlass_mxfp4_expert_layout() below):
     * NOT a uniform per-element byte rate, so block_elems=0 here on purpose
     * -- this entry exists only so tensor_type_name() has something to
     * print; it makes tensor_nbytes() refuse (rather than silently
     * miscompute) if this type is ever routed through the generic path. */
    /*40*/ {"cutlass_mxfp4", 0, 0},
    /* MXFP8_LT (pre-stored MXFP8): de-interleaved E4M3 data + swizzled E8M0
     * scale. For the shipped 128-aligned shapes the total byte size equals the
     * type-38 size (out*(in/32)*33), so it reuses the {32,33} accounting here
     * and loads/mmaps through the generic path. See DS4_TENSOR_MXFP8_LT. */
    /*41*/ {"mxfp8_lt", 32, 33},
};

static_assert(sizeof(gguf_types) / sizeof(gguf_types[0]) == 42,
              "gguf_types rows must line up with GGUF type ids");



static uint64_t scalar_value_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}



static bool skip_value(ds4_cursor *c, uint32_t type, int depth) {
    if (depth > 8) {
        cursor_error(c, "metadata array nesting is too deep");
        return false;
    }

    uint64_t scalar = scalar_value_size(type);
    if (scalar != 0) return cursor_skip(c, scalar);

    if (type == GGUF_VALUE_STRING) {
        ds4_str ignored;
        return cursor_string(c, &ignored);
    }

    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type;
        uint64_t len;

        if (!cursor_u32(c, &item_type)) return false;
        if (!cursor_u64(c, &len)) return false;

        uint64_t item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                cursor_error(c, "metadata array is too large");
                return false;
            }
            return cursor_skip(c, len * item_size);
        }

        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, depth + 1)) return false;
        }
        return true;
    }

    cursor_error(c, "unknown GGUF metadata type");
    return false;
}



const gguf_type_info *tensor_type(uint32_t type) {
    uint32_t n = sizeof(gguf_types) / sizeof(gguf_types[0]);
    if (type >= n || gguf_types[type].name == NULL) return NULL;
    return &gguf_types[type];
}



const char *tensor_type_name(uint32_t type) {
    const gguf_type_info *info = tensor_type(type);
    return info ? info->name : "unknown";
}



static bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    const gguf_type_info *info = tensor_type(type);
    if (!info || info->block_elems == 0) return false;
    uint64_t blocks = (elements + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}



/* CUTLASS Sm120 block-scaled MXFP4 (type 40) is expert-major: one
 * ColumnMajor E2M1 data blob (N*K/2 bytes) followed by one swizzled E8M0
 * scale-factor blob per expert. The CUTLASS Sm1xxBlkScaledConfig tile atom
 * pads both N and K up to a multiple of 128 for the SF blob, so this is
 * NOT a uniform per-element byte rate and cannot live in gguf_types[].
 * (For DS4's actual rich-expert shapes both dims are already multiples of
 * 128, so sf_bytes == n*k/32 exactly with no padding -- but callers must
 * not assume that holds for arbitrary shapes.) */
void cutlass_mxfp4_expert_layout(uint64_t k, uint64_t n,
                                  uint64_t *data_bytes, uint64_t *sf_bytes,
                                  uint64_t *stride) {
    uint64_t k_pad = (k + 127) / 128 * 128;
    uint64_t n_pad = (n + 127) / 128 * 128;
    *data_bytes = n * k / 2;
    *sf_bytes = (n_pad / 32) * k_pad;
    *stride = *data_bytes + *sf_bytes;
}



ds4_cursor cursor_at(const ds4_model *m, uint64_t pos) {
    ds4_cursor c = {
        .base = m->map,
        .size = m->size,
        .pos = pos,
        .error = {0},
    };
    return c;
}



static ds4_kv *model_find_kv(const ds4_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (ds4_streq(m->kv[i].key, key)) return &m->kv[i];
    }
    return NULL;
}



static bool model_get_string(const ds4_model *m, const char *key, ds4_str *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_string(&c, out);
}



bool model_get_u32(const ds4_model *m, const char *key, uint32_t *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_u32(&c, out);
}



static bool model_get_u64(const ds4_model *m, const char *key, uint64_t *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_UINT64) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    return cursor_u64(&c, out);
}



bool model_get_u64_compat(const ds4_model *m, const char *key, uint64_t *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_UINT64) {
        return cursor_u64(&c, out);
    }
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = v;
        return true;
    }
    return false;
}



bool model_get_f32_compat(const ds4_model *m, const char *key, float *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_FLOAT32) {
        return cursor_read(&c, out, sizeof(*out));
    }
    if (kv->type == GGUF_VALUE_FLOAT64) {
        double v = 0.0;
        if (!cursor_read(&c, &v, sizeof(v))) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == GGUF_VALUE_INT32) {
        int32_t v = 0;
        if (!cursor_read(&c, &v, sizeof(v))) return false;
        *out = (float)v;
        return true;
    }
    return false;
}



bool model_get_bool(const ds4_model *m, const char *key, bool *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_BOOL) return false;
    ds4_cursor c = cursor_at(m, kv->value_pos);
    uint8_t v = 0;
    if (!cursor_read(&c, &v, sizeof(v))) return false;
    *out = v != 0;
    return true;
}



bool model_get_array(const ds4_model *m, const char *key, ds4_array_ref *out) {
    ds4_kv *kv = model_find_kv(m, key);
    if (!kv || kv->type != GGUF_VALUE_ARRAY) return false;

    ds4_cursor c = cursor_at(m, kv->value_pos);
    if (!cursor_u32(&c, &out->type)) return false;
    if (!cursor_u64(&c, &out->len)) return false;
    out->data_pos = c.pos;
    return true;
}



void model_close(ds4_model *m) {
    if (!m) return;
    free(m->kv);
    free(m->tensors);
    if (m->map) munmap((void *)m->map, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}



static void model_prefetch_cpu_mapping(const ds4_model *m) {
    if (!m || !m->map || m->size == 0) return;

    /*
     * CPU generation touches expert weights according to router decisions, so a
     * long decode can fault in model pages that the prompt never touched. On
     * current Darwin kernels we have seen those late file-backed faults trigger
     * an OS-level VM panic in map-count accounting. This hint does not copy or
     * pin the GGUF; it just asks the kernel to start bringing the read-only
     * mapping into the page cache before token generation reaches it.
     */
#if defined(POSIX_MADV_WILLNEED)
    const int rc = posix_madvise((void *)m->map, (size_t)m->size, POSIX_MADV_WILLNEED);
    if (rc != 0) {
        ds4_log(stderr,
                DS4_LOG_WARNING,
                "ds4: warning: POSIX_MADV_WILLNEED failed for CPU model mapping: %s\n",
                strerror(rc));
    }
#else
    (void)m;
#endif
}



/* Read the GGUF metadata table.  Values stay in the mmap; we store offsets so
 * later validation can decode only the keys it needs. */
static void parse_metadata(ds4_model *m, ds4_cursor *c) {
    m->kv = (ds4_kv *)calloc((size_t)m->n_kv, sizeof(m->kv[0]));
    if (!m->kv) ds4_die("out of memory while allocating metadata table");

    m->alignment = 32;

    for (uint64_t i = 0; i < m->n_kv; i++) {
        ds4_kv *kv = &m->kv[i];

        if (!cursor_string(c, &kv->key)) ds4_die(c->error);
        if (!cursor_u32(c, &kv->type)) ds4_die(c->error);

        kv->value_pos = c->pos;

        if (ds4_streq(kv->key, "general.alignment") &&
            kv->type == GGUF_VALUE_UINT32)
        {
            ds4_cursor tmp = cursor_at(m, kv->value_pos);
            uint32_t alignment;
            if (cursor_u32(&tmp, &alignment) && alignment != 0) {
                m->alignment = alignment;
            }
        }

        if (!skip_value(c, kv->type, 0)) ds4_die(c->error);
    }
}



/* Read the tensor directory and convert relative GGUF offsets to absolute
 * mmap offsets.  Tensor bytes are still never copied here. */
static void parse_tensors(ds4_model *m, ds4_cursor *c) {
    m->tensors = (ds4_tensor *)calloc((size_t)m->n_tensors, sizeof(m->tensors[0]));
    if (!m->tensors) ds4_die("out of memory while allocating tensor table");

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];

        if (!cursor_string(c, &t->name)) ds4_die(c->error);
        if (!cursor_u32(c, &t->ndim)) ds4_die(c->error);
        if (t->ndim == 0 || t->ndim > DS4_MAX_DIMS) {
            ds4_die("tensor has an unsupported number of dimensions");
        }

        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) ds4_die(c->error);
            if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) {
                ds4_die("tensor element count overflow");
            }
            t->elements *= t->dim[d];
        }

        if (!cursor_u32(c, &t->type)) ds4_die(c->error);
        if (!cursor_u64(c, &t->rel_offset)) ds4_die(c->error);

        if (t->type == DS4_TENSOR_CUTLASS_MXFP4) {
            if (t->ndim != 3) ds4_die("CUTLASS MXFP4 tensor must be 3D [K,N,n_expert]");
            uint64_t data_bytes, sf_bytes, stride;
            cutlass_mxfp4_expert_layout(t->dim[0], t->dim[1], &data_bytes, &sf_bytes, &stride);
            if (t->dim[2] != 0 && stride > UINT64_MAX / t->dim[2]) {
                ds4_die("tensor element count overflow");
            }
            t->bytes = stride * t->dim[2];
        } else if (!tensor_nbytes(t->type, t->elements, &t->bytes)) {
            ds4_log(stderr,
                DS4_LOG_WARNING,
                "ds4: warning: tensor %.*s has unsupported GGUF type %u\n",
                (int)t->name.len, t->name.ptr, t->type);
        }
    }

    m->tensor_data_pos = align_up(c->pos, m->alignment);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        ds4_tensor *t = &m->tensors[i];
        if (t->rel_offset > UINT64_MAX - m->tensor_data_pos) {
            ds4_die("tensor offset overflow");
        }
        t->abs_offset = m->tensor_data_pos + t->rel_offset;
        if (t->bytes != 0 &&
            (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset))
        {
            ds4_die("tensor points outside GGUF file");
        }
        if (t->bytes > m->max_tensor_bytes) {
            m->max_tensor_bytes = t->bytes;
        }
    }
}



/* Open and map the GGUF once.  GPU needs a shared mapping for no-copy
 * MTLBuffers; CPU uses a private read-only mapping to avoid Darwin VM stress.
 * Tokenizer-only callers pass prefetch_cpu=false so inspecting tokens never
 * walks the huge tensor payload. */
void model_open(ds4_model *m, const char *path, bool gpu_mapping,
                       bool prefetch_cpu) {
    memset(m, 0, sizeof(*m));
    m->fd = -1;

    int fd = open(path, O_RDONLY);
    if (fd == -1) ds4_die_errno("cannot open model", path);

    struct stat st;
    if (fstat(fd, &st) == -1) ds4_die_errno("cannot stat model", path);
    if (st.st_size < 32) ds4_die("model file is too small to be GGUF");

    /*
     * GPU wraps slices of this mapping as no-copy MTLBuffers, so the GPU
     * path keeps the file-backed shared mapping. The CPU path only reads the
     * weights through normal pointers and should not inherit GPU's VM policy:
     * use a private read-only mapping there.
     *
     * This is deliberately defensive against an OS-level Darwin VM bug observed
     * while the CPU backend streams the very large GGUF through a shared mmap:
     * the kernel can panic in VM map-count accounting instead of returning a
     * normal user-space failure. Keeping CPU inference off the shared mapping
     * avoids that VM accounting path while preserving normal file-backed reads.
     */
    const int mmap_flags = gpu_mapping ? MAP_SHARED : MAP_PRIVATE;
    void *map = (void *)mmap(NULL, (size_t)st.st_size, PROT_READ, mmap_flags, fd, 0);
    if (map == MAP_FAILED) ds4_die_errno("cannot mmap model", path);

    m->fd = fd;
    m->map = (const uint8_t *)map;
    m->size = (uint64_t)st.st_size;

    ds4_cursor c = cursor_at(m, 0);
    uint32_t magic;
    if (!cursor_u32(&c, &magic)) ds4_die(c.error);
    if (magic != DS4_GGUF_MAGIC) ds4_die("model is not a GGUF file");
    if (!cursor_u32(&c, &m->version)) ds4_die(c.error);
    if (!cursor_u64(&c, &m->n_tensors)) ds4_die(c.error);
    if (!cursor_u64(&c, &m->n_kv)) ds4_die(c.error);

    if (m->version != 3) ds4_die("only GGUF v3 is supported");

    /* Sanity-cap the 64-bit header counts before they size calloc()s: every
     * metadata entry and tensor record needs well over 24 bytes of file, so
     * counts beyond size/24 can only come from a corrupt or hostile header. */
    if (m->n_kv > m->size / 24 || m->n_tensors > m->size / 24) {
        ds4_die("GGUF header kv/tensor counts exceed what the file could hold");
    }

    parse_metadata(m, &c);
    parse_tensors(m, &c);

    if (!gpu_mapping && prefetch_cpu) model_prefetch_cpu_mapping(m);
}



static void print_size(uint64_t bytes) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    printf("%.2f GiB", (double)bytes / gib);
}



void model_summary(const ds4_model *m) {
    ds4_str name = {0, 0};
    ds4_str arch = {0, 0};
    uint32_t layers = 0;
    uint64_t ctx_train = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t n_swa = 0;
    uint32_t indexer_heads = 0;
    uint32_t indexer_head_dim = 0;
    uint32_t indexer_top_k = 0;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_expert_groups = 0;
    uint32_t n_group_used = 0;
    uint64_t tensor_bytes = 0;
    uint64_t params = 0;

    model_get_string(m, "general.name", &name);
    model_get_string(m, "general.architecture", &arch);
    model_get_u32(m, "deepseek4.block_count", &layers);
    model_get_u64(m, "deepseek4.context_length", &ctx_train);
    model_get_u32(m, "deepseek4.attention.head_count", &n_head);
    model_get_u32(m, "deepseek4.attention.head_count_kv", &n_head_kv);
    model_get_u32(m, "deepseek4.attention.key_length", &head_dim);
    model_get_u32(m, "deepseek4.attention.sliding_window", &n_swa);
    model_get_u32(m, "deepseek4.attention.indexer.head_count", &indexer_heads);
    model_get_u32(m, "deepseek4.attention.indexer.key_length", &indexer_head_dim);
    model_get_u32(m, "deepseek4.attention.indexer.top_k", &indexer_top_k);
    model_get_u32(m, "deepseek4.expert_count", &n_expert);
    model_get_u32(m, "deepseek4.expert_used_count", &n_expert_used);
    model_get_u32(m, "deepseek4.expert_group_count", &n_expert_groups);
    model_get_u32(m, "deepseek4.expert_group_used_count", &n_group_used);

    for (uint64_t i = 0; i < m->n_tensors; i++) {
        tensor_bytes += m->tensors[i].bytes;
        params += m->tensors[i].elements;
    }

    printf("model: %.*s\n", (int)name.len, name.ptr);
    printf("arch:  %.*s\n", (int)arch.len, arch.ptr);
    printf("gguf:  v%u, %" PRIu64 " metadata keys, %" PRIu64 " tensors\n",
        m->version, m->n_kv, m->n_tensors);
    if (layers) printf("layers: %u\n", layers);
    if (ctx_train) printf("train context: %" PRIu64 "\n", ctx_train);
    if (n_head || n_head_kv || head_dim || n_swa) {
        printf("attention: heads=%u kv_heads=%u head_dim=%u swa=%u\n",
               n_head, n_head_kv, head_dim, n_swa);
    }
    if (indexer_heads || indexer_head_dim || indexer_top_k) {
        printf("indexer: heads=%u head_dim=%u top_k=%u\n",
               indexer_heads, indexer_head_dim, indexer_top_k);
    }
    if (n_expert || n_expert_used || n_expert_groups || n_group_used) {
        printf("experts: count=%u used=%u groups=%u groups_used=%u\n",
               n_expert, n_expert_used, n_expert_groups, n_group_used);
    }
    printf("file size: ");
    print_size(m->size);
    printf("\n");
    printf("tensor bytes described by GGUF: ");
    print_size(tensor_bytes);
    printf("\n");
    printf("logical parameters: %.2f B\n", (double)params / 1000000000.0);

    printf("tensor types:\n");
    for (uint32_t type = 0; type < sizeof(gguf_types)/sizeof(gguf_types[0]); type++) {
        uint64_t count = 0;
        uint64_t bytes = 0;
        for (uint64_t i = 0; i < m->n_tensors; i++) {
            if (m->tensors[i].type == type) {
                count++;
                bytes += m->tensors[i].bytes;
            }
        }
        if (count != 0) {
            printf("  %-8s %5" PRIu64 " tensors, ", tensor_type_name(type), count);
            print_size(bytes);
            printf("\n");
        }
    }

}



ds4_tensor *model_find_tensor(const ds4_model *m, const char *name) {
    const size_t len = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].name.len == len &&
            memcmp(m->tensors[i].name.ptr, name, len) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}



static int accelerator_tensor_span_cmp(const void *a, const void *b) {
    const accelerator_tensor_span *sa = (const accelerator_tensor_span *)a;
    const accelerator_tensor_span *sb = (const accelerator_tensor_span *)b;
    if (sa->off < sb->off) return -1;
    if (sa->off > sb->off) return 1;
    if (sa->end < sb->end) return -1;
    if (sa->end > sb->end) return 1;
    return 0;
}



static uint64_t accelerator_cuda_preload_span_bytes(void) {
    uint64_t mb = 1024;
    const char *env = getenv("DS4_CUDA_WEIGHT_PRELOAD_SPAN_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 64) mb = 64;
    if (mb > 4096) mb = 4096;
    return mb * 1048576ull;
}



static bool accelerator_span_filter_contains(uint64_t off,
                                             uint64_t bytes,
                                             const uint64_t *span_offsets,
                                             const uint64_t *span_sizes,
                                             uint32_t span_count) {
    if (span_count == 0) return true;
    if (bytes == 0) return true;
    const uint64_t end = off + bytes;
    if (end < off) return false;
    for (uint32_t i = 0; i < span_count; i++) {
        const uint64_t span_end = span_offsets[i] + span_sizes[i];
        if (span_end < span_offsets[i]) return false;
        if (off >= span_offsets[i] && end <= span_end) return true;
    }
    return false;
}



static bool accelerator_prepare_model_tensor_spans(const ds4_model *m,
                                                   const uint64_t *span_offsets,
                                                   const uint64_t *span_sizes,
                                                   uint32_t span_count,
                                                   const char *skip_prefix,
                                                   uint64_t *prepared_out) {
    uint64_t cap = m->n_tensors;
    if (cap == 0) {
        if (prepared_out) *prepared_out = 0;
        return true;
    }

    accelerator_tensor_span *spans = (accelerator_tensor_span *)xmalloc((size_t)cap * sizeof(spans[0]));
    uint64_t nspan = 0;
    for (uint32_t i = 0; i < span_count; i++) {
        if (span_offsets[i] > m->size ||
            span_sizes[i] == 0 ||
            span_sizes[i] > m->size - span_offsets[i]) {
            free(spans);
            return false;
        }
    }
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const ds4_tensor *t = &m->tensors[i];
        if (t->bytes == 0) continue;
        /* --expert-overlay swapped tensors live in the donor file's mapping
         * (donor-relative offsets that can exceed this model's size); they are
         * prepared separately by accelerator_prepare_expert_overlay. */
        if (t->ext_map) continue;
        if (skip_prefix &&
            t->name.len > strlen(skip_prefix) &&
            memcmp(t->name.ptr, skip_prefix, strlen(skip_prefix)) == 0) continue;
        if (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset) {
            free(spans);
            return false;
        }
        if (!accelerator_span_filter_contains(t->abs_offset, t->bytes,
                                              span_offsets, span_sizes, span_count)) {
            continue;
        }
        spans[nspan++] = (accelerator_tensor_span){
            .off = t->abs_offset,
            .end = t->abs_offset + t->bytes,
        };
    }
    if (nspan == 0) {
        free(spans);
        if (prepared_out) *prepared_out = 0;
        return true;
    }

    qsort(spans, (size_t)nspan, sizeof(spans[0]), accelerator_tensor_span_cmp);

    const uint64_t max_span = accelerator_cuda_preload_span_bytes();
    const int tty = ds4_log_is_tty(stderr);
    const uint64_t progress_step = (tty ? 2ull : 16ull) * 1073741824ull;
    uint64_t next_progress = progress_step;
    double last_progress = now_sec();
    uint64_t prepared = 0;
    uint64_t merged = 0;

    const char *accelerator_name = "CUDA";

    fprintf(stderr, "%sds4: %s preparing model tensor mappings%s",
            tty ? "\r\033[K" : "",
            accelerator_name,
            tty ? ": 0.00 GiB" : "\n");
    fflush(stderr);

    for (uint64_t i = 0; i < nspan;) {
        uint64_t off = spans[i].off;
        uint64_t end = spans[i].end;
        i++;
        while (i < nspan &&
               spans[i].off <= end + 65536u &&
               spans[i].end - off <= max_span) {
            if (spans[i].end > end) end = spans[i].end;
            i++;
        }
        char label[96];
        snprintf(label, sizeof(label), "tensor-span:%" PRIu64, merged);
        if (ds4_gpu_cache_model_range(m->map, m->size, off, end - off, label) == 0) {
            if (tty) fputc('\n', stderr);
            fprintf(stderr,
                    "ds4: accelerator failed to prepare model tensor span %" PRIu64
                    " at offset %" PRIu64 "\n",
                    merged, off);
            free(spans);
            return false;
        }
        prepared += end - off;
        merged++;

        const double now = now_sec();
        if (prepared >= next_progress || now - last_progress >= (tty ? 2.0 : 10.0)) {
            if (tty) {
                fprintf(stderr, "\r\033[Kds4: %s preparing model tensor mappings: %.2f GiB",
                        accelerator_name,
                        (double)prepared / 1073741824.0);
            } else {
                fprintf(stderr, "ds4: %s prepared model tensor mappings %.2f GiB\n",
                        accelerator_name,
                        (double)prepared / 1073741824.0);
            }
            fflush(stderr);
            last_progress = now;
            while (next_progress <= prepared) next_progress += progress_step;
        }
    }

    if (tty) fputc('\n', stderr);
    free(spans);
    if (prepared_out) *prepared_out = prepared;
    return true;
}



bool accelerator_cache_model_tensors(ds4_backend backend,
                                            const ds4_model *m,
                                            const uint64_t *span_offsets,
                                            const uint64_t *span_sizes,
                                            uint32_t span_count,
                                     const char *skip_prefix) {
    if (backend != DS4_BACKEND_CUDA) return true;
    if (!m || !m->map || m->size == 0) return false;
    /* Register each MXFP8 weight's offset so the workhorse matmul executes
     * ONLY registered tensors (per-tensor routing; unregistered offsets are
     * rejected at dispatch). Runs before the DIRECT_MODEL early-out so it
     * applies in the mmap path too. */
    uint64_t n_fp8 = 0, n_fp8_lt = 0;
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        const ds4_tensor *t = &m->tensors[i];
        if (t->type == DS4_TENSOR_FP8_E4M3) {
            ds4_gpu_register_fp8_weight(t->abs_offset);
            n_fp8++;
        } else if (t->type == DS4_TENSOR_MXFP8_LT) {
            /* Same FP8 matmul path, but flag the offset as pre-stored so the
             * resolver points cuBLASLt at the mmap instead of converting. */
            ds4_gpu_register_fp8_weight(t->abs_offset);
            ds4_gpu_register_fp8_lt_weight(t->abs_offset);
            n_fp8++;
            n_fp8_lt++;
        }
    }
    if (n_fp8 > 0)
        fprintf(stderr, "ds4: %llu MXFP8 workhorse weights detected -> FP8 matmul path"
                " (%llu pre-stored MXFP8_LT, zero-copy)\n",
                (unsigned long long)n_fp8, (unsigned long long)n_fp8_lt);
    if (getenv("DS4_CUDA_DIRECT_MODEL") != NULL) {
        return true;
    }

    const double t0 = now_sec();
    uint64_t prepared = 0;
    if (!accelerator_prepare_model_tensor_spans(m, span_offsets, span_sizes, span_count, skip_prefix, &prepared)) {
        return false;
    }
    const double t1 = now_sec();
    const char *accelerator_name = "CUDA";
    fprintf(stderr,
            "ds4: %s startup model preparation covered %.2f GiB of tensor spans in %.3fs\n",
            accelerator_name, (double)prepared / 1073741824.0, t1 - t0);
    return true;
}



/* Return the in-place tensor payload inside the mapped GGUF (or inside the
 * overlay file's mapping for --expert-overlay swapped tensors). */
const void *tensor_data(const ds4_model *m, const ds4_tensor *t) {
    return (const uint8_t *)tensor_map_base(m, t) + t->abs_offset;
}



/* Pre-populate device access for --expert-overlay swapped tensor spans.
 * Must run at startup: the lazy weight-cache copy path cannot execute inside
 * GPU graph encode (synchronous cudaMemcpy during capture fails), which is
 * where an unprepared overlay span would otherwise first be touched. */
bool accelerator_prepare_expert_overlay(ds4_backend backend,
                                        const ds4_model *base,
                                        const ds4_model *overlay) {
    if (backend != DS4_BACKEND_CUDA) return true;
    uint64_t prepared = 0;
    for (uint64_t i = 0; i < base->n_tensors; i++) {
        const ds4_tensor *t = &base->tensors[i];
        if (t->ext_map != overlay->map || t->bytes == 0) continue;
        char label[96];
        snprintf(label, sizeof(label), "overlay:%.*s",
                 (int)(t->name.len < 80 ? t->name.len : 80), t->name.ptr);
        if (ds4_gpu_cache_external_range(overlay->map, overlay->fd,
                                         t->abs_offset, t->bytes, label) == 0) {
            fprintf(stderr, "ds4: failed to prepare expert-overlay span for %.*s\n",
                    (int)t->name.len, t->name.ptr);
            return false;
        }
        prepared += t->bytes;
    }
    fprintf(stderr, "ds4: expert overlay prepared %.2f GiB for device access\n",
            (double)prepared / 1073741824.0);
    return true;
}



/* Swap routed-expert tensor entries in the base model for same-named entries
 * from an overlay GGUF (--expert-overlay FILE:PREFIX). Only 3D `*_exps.*`
 * tensors whose name starts with the prefix are swapped, so the overlay never
 * touches FP8-registered dense weights or streaming offset tables. A swapped
 * entry keeps the overlay file's abs_offset and records the overlay mapping in
 * ext_map/ext_size; consumers resolve the payload via tensor_data()/
 * tensor_map_base(). Returns the number of tensors swapped. */
static bool str_span_contains(const char *s, size_t len, const char *needle) {
    const size_t nlen = strlen(needle);
    if (len < nlen) return false;
    for (size_t i = 0; i + nlen <= len; i++) {
        if (memcmp(s + i, needle, nlen) == 0) return true;
    }
    return false;
}

uint32_t model_apply_expert_overlay(ds4_model *base, const ds4_model *overlay,
                                    const char *prefix) {
    const size_t plen = strlen(prefix);
    uint32_t swapped = 0;
    for (uint64_t i = 0; i < base->n_tensors; i++) {
        ds4_tensor *t = &base->tensors[i];
        if (t->name.len < plen || memcmp(t->name.ptr, prefix, plen) != 0) continue;
        /* GGUF names are length-prefixed, not NUL-terminated: every match
         * below must stay bounded by name.len. */
        if (t->ndim != 3 || !str_span_contains(t->name.ptr, t->name.len, "_exps.")) continue;
        const ds4_tensor *ov = NULL;
        for (uint64_t j = 0; j < overlay->n_tensors; j++) {
            const ds4_tensor *c = &overlay->tensors[j];
            if (c->name.len == t->name.len &&
                memcmp(c->name.ptr, t->name.ptr, t->name.len) == 0) {
                ov = c;
                break;
            }
        }
        if (!ov) {
            fprintf(stderr, "ds4: expert overlay is missing tensor: %.*s\n",
                    (int)t->name.len, t->name.ptr);
            exit(1);
        }
        if (ov->ndim != t->ndim ||
            memcmp(ov->dim, t->dim, sizeof(ov->dim)) != 0) {
            fprintf(stderr, "ds4: expert overlay shape mismatch for %.*s\n",
                    (int)t->name.len, t->name.ptr);
            exit(1);
        }
        if (ov->type == DS4_TENSOR_CUTLASS_MXFP4) {
            /* The CUTLASS grouped-GEMM prefill path device-asserts when its
             * expert weights come from an overlay range (observed 2026-07-02,
             * root cause not yet chased). The measurement harness only needs
             * plain MXFP4/IQ2/Q2K donors, so refuse rather than corrupt. */
            fprintf(stderr, "ds4: expert overlay does not support CUTLASS "
                            "type-40 donor tensors yet (%.*s)\n",
                    (int)t->name.len, t->name.ptr);
            exit(1);
        }
        t->type = ov->type;
        t->rel_offset = ov->rel_offset;
        t->abs_offset = ov->abs_offset;
        t->elements = ov->elements;
        t->bytes = ov->bytes;
        t->ext_map = overlay->map;
        t->ext_size = overlay->size;
        if (t->bytes > base->max_tensor_bytes) base->max_tensor_bytes = t->bytes;
        swapped++;
    }
    return swapped;
}



/* Optional startup pass that touches tensor pages before timing generation. */
void model_warm_weights(const ds4_model *m) {
    const uint64_t start = m->tensor_data_pos;
    const uint64_t end = m->size;
    if (start >= end) return;

    const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
    const uint8_t *p = m->map;
    volatile uint64_t checksum = 0;
    const double t0 = now_sec();

    fprintf(stderr, "ds4: warming mapped tensor pages: %.2f GiB\n",
            (double)(end - start) / (1024.0 * 1024.0 * 1024.0));

#if defined(POSIX_MADV_WILLNEED)
    (void)posix_madvise((void *)(p + start), (size_t)(end - start), POSIX_MADV_WILLNEED);
#endif

    for (uint64_t off = start; off < end; off += page) {
        checksum += p[off];
    }
    checksum += p[end - 1];

    const double t1 = now_sec();
    fprintf(stderr, "ds4: warmed tensor pages in %.3fs (checksum=%llu)\n",
            t1 - t0, (unsigned long long)checksum);
}

