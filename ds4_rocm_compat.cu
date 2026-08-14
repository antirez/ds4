#include <hip/hip_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_gpu_mgpu.h"
#include "ds4_gpu.h"
#include "ds4_gpu_args.h"

ds4_gpu_ctx g_gpu[DS4_MAX_GPUS] = {};
int g_n_gpus = 1;
int g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS] = {{1}};

static int rocm_tier_valid(int tier) {
    return tier == 0 && g_n_gpus == 1;
}

/* ------------------------------------------------------------------------
 * Decode-island HIP graph capture.  Port of the CUDA implementation in
 * ds4_cuda.cu (design from the Entrpi/ds4 batched-serving fork): the
 * position-independent islands of a decode layer are captured once per
 * buffer-alias key and replayed afterwards.  A replayed graph re-runs
 * the captured kernels byte-for-byte, so replay output is bit-identical
 * to the eager encode it recorded; any capture failure retires the
 * entry and the island runs eagerly, also byte-identical to before.
 *
 * Stream handling mirrors CUDA: kernels normally launch on the legacy
 * default stream, which cannot be captured, so every launch site in the
 * backend goes through ds4_rocm_stream(), which returns the capture
 * stream while g_rocm_decode_graph_capturing is set.
 */
#define ROCM_DECODE_GRAPH_LAYERS   64u
#define ROCM_DECODE_GRAPH_ISLANDS  4u
#define ROCM_DECODE_GRAPH_VARIANTS 8u

struct rocm_decode_graph_entry {
    ds4_decode_graph_key key;
    hipGraphExec_t       exec;
    int                  state;   /* 0 empty, 1 warmed, 2 ready, 3 dead */
    uint64_t             hits;
};

static rocm_decode_graph_entry
    g_rocm_decode_graphs[ROCM_DECODE_GRAPH_LAYERS][ROCM_DECODE_GRAPH_ISLANDS]
                        [ROCM_DECODE_GRAPH_VARIANTS];
static hipStream_t g_rocm_decode_graph_stream = NULL;
static int g_rocm_decode_graph_capturing = 0;
static uint64_t g_rocm_decode_graph_replays = 0;
static uint64_t g_rocm_decode_graph_captures = 0;

extern "C" void ds4_rocm_set_capture_stream(void *stream);

/* Island policy: islands 0/1 are the upstream eval-path islands
 * (TO_QKV and FROM_ATTN_TO_FFN); island 2 is the speculative span's
 * batched attention-output stage (C2), island 3 is the span's shared
 * expert tail (stages D+E).  Live measurements on gfx1151 (MTP draft-2
 * verify path): the small eval islands replay SLOWER than eager (rounds
 * 91.0 -> 92.7 ms), and islands 2/3 replay byte-exact (SHA-gated) but
 * time-neutral (91.1 ms) -- their 60-240 us kernels already hide launch
 * latency, so the ~11 ms/span launch feed lives in the tiny per-row
 * chains: position-dependent stages A/B and the per-row router/MoE
 * stage C3, which is capture-unsafe (deterministic divergence with all
 * buffer pointers keyed; cause not isolated).  The machinery is kept
 * because capturing the small-op chains becomes attractive once they
 * are rows-batched and position-indirect; until then it is dormant, so
 * so the default is OFF (measured-neutral today) and the machinery is
 * staged infrastructure: DS4_ROCM_DECODE_GRAPHS=span captures islands
 * 2+, =all also captures the eval islands. */
#define ROCM_DECODE_GRAPH_MODE_OFF  0
#define ROCM_DECODE_GRAPH_MODE_SPAN 1
#define ROCM_DECODE_GRAPH_MODE_ALL  2
static int g_rocm_decode_graph_mode = ROCM_DECODE_GRAPH_MODE_OFF;

extern "C" int ds4_gpu_decode_graphs_supported(void) {
    static int init = 0;
    static int enabled = 0;
    if (!init) {
        init = 1;
        const char *s = getenv("DS4_ROCM_DECODE_GRAPHS");
        if (s && (strcmp(s, "span") == 0 || strcmp(s, "SPAN") == 0)) {
            g_rocm_decode_graph_mode = ROCM_DECODE_GRAPH_MODE_SPAN;
        } else if (s && (strcmp(s, "all") == 0 || strcmp(s, "ALL") == 0)) {
            g_rocm_decode_graph_mode = ROCM_DECODE_GRAPH_MODE_ALL;
        } else {
            g_rocm_decode_graph_mode = ROCM_DECODE_GRAPH_MODE_OFF;
        }
        enabled = g_rocm_decode_graph_mode != ROCM_DECODE_GRAPH_MODE_OFF;
        if (enabled)
            fprintf(stderr, "ds4: DS4_ROCM_DECODE_GRAPHS=%s - decode graph capture mode %d\n",
                    s, g_rocm_decode_graph_mode);
    }
    return enabled && g_n_gpus == 1;
}

static void rocm_decode_graph_entry_kill(rocm_decode_graph_entry *e) {
    if (e->exec) {
        (void)hipGraphExecDestroy(e->exec);
        e->exec = NULL;
    }
    e->state = 3;
}

extern "C" void ds4_gpu_decode_graphs_invalidate(void) {
    for (uint32_t il = 0; il < ROCM_DECODE_GRAPH_LAYERS; il++) {
        for (uint32_t is = 0; is < ROCM_DECODE_GRAPH_ISLANDS; is++) {
            for (uint32_t v = 0; v < ROCM_DECODE_GRAPH_VARIANTS; v++) {
                rocm_decode_graph_entry *e = &g_rocm_decode_graphs[il][is][v];
                if (e->exec) {
                    (void)hipGraphExecDestroy(e->exec);
                    e->exec = NULL;
                }
                e->state = 0;
                e->hits = 0;
                memset(&e->key, 0, sizeof(e->key));
            }
        }
    }
}

static rocm_decode_graph_entry *rocm_decode_graph_find(
        const ds4_decode_graph_key *key) {
    if (key->il >= ROCM_DECODE_GRAPH_LAYERS ||
        key->island >= ROCM_DECODE_GRAPH_ISLANDS) return NULL;
    rocm_decode_graph_entry *slot = NULL;
    for (uint32_t v = 0; v < ROCM_DECODE_GRAPH_VARIANTS; v++) {
        rocm_decode_graph_entry *e = &g_rocm_decode_graphs[key->il][key->island][v];
        if (e->state != 0 &&
            memcmp(&e->key, key, sizeof(*key)) == 0) return e;
        if (e->state == 0 && !slot) slot = e;
    }
    if (slot) {
        memcpy(&slot->key, key, sizeof(*key));
        slot->state = 0;   /* caller advances the state machine */
        return slot;
    }
    return NULL;           /* all variants busy with other keys: stay eager */
}

extern "C" int ds4_gpu_decode_graph_begin(const ds4_decode_graph_key *key) {
    if (!key || !ds4_gpu_decode_graphs_supported()) return -1;
    if (g_rocm_decode_graph_mode != ROCM_DECODE_GRAPH_MODE_ALL &&
        key->island < 2u) return -1;   /* eval islands: measured net-negative */
    if (g_rocm_decode_graph_capturing) return -1;   /* no nesting */
    rocm_decode_graph_entry *e = rocm_decode_graph_find(key);
    if (!e || e->state == 3) return -1;
    if (e->state == 0) {
        /* Warm pass: run eagerly once so lazy allocators (tmp scratch,
         * hipBLASLt workspaces) reach steady-state sizes before capture. */
        e->state = 1;
        return -1;
    }
    if (e->state == 2) {
        /* Replay on the legacy default stream: the captured chain is
         * single-stream, so it executes in order against the surrounding
         * eager work with no cross-stream sync edges.  (Replaying on the
         * capture stream raced the eager producers in live tests.) */
        hipError_t err = hipGraphLaunch(e->exec, (hipStream_t)0);
        if (err != hipSuccess) {
            fprintf(stderr, "ds4: decode graph replay failed (il=%u island=%u): %s\n",
                    key->il, key->island, hipGetErrorString(err));
            (void)hipGetLastError();
            rocm_decode_graph_entry_kill(e);
            return -1;     /* caller encodes eagerly; nothing was consumed */
        }
        e->hits++;
        g_rocm_decode_graph_replays++;
        return 1;
    }
    /* state == 1: capture this encode. */
    if (!g_rocm_decode_graph_stream) {
        /* Blocking flag (no hipStreamNonBlocking): the legacy default
         * stream implicitly serializes with blocking streams, so graph
         * replays on this stream stay ordered against the surrounding
         * eager work that still launches on the legacy stream. */
        if (hipStreamCreate(&g_rocm_decode_graph_stream) != hipSuccess) {
            g_rocm_decode_graph_stream = NULL;
            rocm_decode_graph_entry_kill(e);
            return -1;
        }
    }
    if (hipStreamBeginCapture(g_rocm_decode_graph_stream,
                              hipStreamCaptureModeGlobal) != hipSuccess) {
        (void)hipGetLastError();
        rocm_decode_graph_entry_kill(e);
        return -1;
    }
    g_rocm_decode_graph_capturing = 1;
    ds4_rocm_set_capture_stream(g_rocm_decode_graph_stream);
    return 0;
}

extern "C" int ds4_gpu_decode_graph_end(const ds4_decode_graph_key *key) {
    if (!key || !g_rocm_decode_graph_capturing) return -1;
    g_rocm_decode_graph_capturing = 0;
    ds4_rocm_set_capture_stream(NULL);
    hipGraph_t graph = NULL;
    hipError_t err = hipStreamEndCapture(g_rocm_decode_graph_stream, &graph);
    rocm_decode_graph_entry *e = rocm_decode_graph_find(key);
    if (err != hipSuccess || graph == NULL) {
        fprintf(stderr, "ds4: decode graph capture failed (il=%u island=%u): %s\n",
                key->il, key->island, hipGetErrorString(err));
        (void)hipGetLastError();
        if (graph) (void)hipGraphDestroy(graph);
        if (e) rocm_decode_graph_entry_kill(e);
        return -1;         /* caller re-encodes the island eagerly */
    }
    if (!e) {              /* cannot happen: begin() found it */
        (void)hipGraphDestroy(graph);
        return -1;
    }
    hipGraphExec_t exec = NULL;
    err = hipGraphInstantiate(&exec, graph, NULL, NULL, 0);
    (void)hipGraphDestroy(graph);
    if (err != hipSuccess || exec == NULL) {
        fprintf(stderr, "ds4: decode graph instantiate failed (il=%u island=%u): %s\n",
                key->il, key->island, hipGetErrorString(err));
        (void)hipGetLastError();
        rocm_decode_graph_entry_kill(e);
        return -1;
    }
    /* Capture recorded the work without executing it: launch now so this
     * token's island actually runs.  Legacy stream, same as replays. */
    err = hipGraphLaunch(exec, (hipStream_t)0);
    if (err != hipSuccess) {
        fprintf(stderr, "ds4: decode graph first launch failed (il=%u island=%u): %s\n",
                key->il, key->island, hipGetErrorString(err));
        (void)hipGetLastError();
        (void)hipGraphExecDestroy(exec);
        rocm_decode_graph_entry_kill(e);
        return -1;
    }
    e->exec = exec;
    e->state = 2;
    g_rocm_decode_graph_captures++;
    if (getenv("DS4_ROCM_DECODE_GRAPH_LOG") != NULL) {
        fprintf(stderr, "ds4: decode graph captured il=%u island=%u (total %llu)\n",
                key->il, key->island,
                (unsigned long long)g_rocm_decode_graph_captures);
    }
    return 0;
}

extern "C" void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key *key) {
    if (!g_rocm_decode_graph_capturing) return;
    g_rocm_decode_graph_capturing = 0;
    ds4_rocm_set_capture_stream(NULL);
    hipGraph_t graph = NULL;
    (void)hipStreamEndCapture(g_rocm_decode_graph_stream, &graph);
    if (graph) (void)hipGraphDestroy(graph);
    (void)hipGetLastError();
    if (key) {
        rocm_decode_graph_entry *e = rocm_decode_graph_find(key);
        if (e) rocm_decode_graph_entry_kill(e);
    }
}

extern "C" int ds4_gpu_init_multi(const ds4_gpu_config *cfg) {
    if (!cfg || cfg->n_gpus != 1) {
        fprintf(stderr, "ds4: ROCm supports one GPU per process\n");
        return 0;
    }
    g_gpu[0].device_id = cfg->device_indices[0];
    if (hipSetDevice(g_gpu[0].device_id) != hipSuccess) return 0;
    return ds4_gpu_init();
}

extern "C" int ds4_gpu_set_current_device(int tier) {
    if (!rocm_tier_valid(tier)) return 1;
    return hipSetDevice(g_gpu[0].device_id) == hipSuccess ? 0 : 1;
}

extern "C" int ds4_gpu_set_current_device_fenced(int tier) {
    return ds4_gpu_set_current_device(tier);
}

extern "C" int ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *t, int tier,
                                         uint64_t bytes) {
    if (!t) return 1;
    if (!rocm_tier_valid(tier)) return 2;
    if (bytes == 0) bytes = 1;
    if (hipMalloc(&t->ptr, (size_t)bytes) != hipSuccess) return 3;
    t->bytes = bytes;
    t->owner = 1;
    t->device_id = 0;
    return 0;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(int tier,
                                                         uint64_t bytes) {
    if (!rocm_tier_valid(tier)) return NULL;
    return ds4_gpu_tensor_alloc(bytes);
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed_on(int tier,
                                                             uint64_t bytes) {
    if (!rocm_tier_valid(tier)) return NULL;
    return ds4_gpu_tensor_alloc_managed(bytes);
}

extern "C" void ds4_gpu_tensor_free_in_place(ds4_gpu_tensor *t) {
    if (!t) return;
    if (t->owner && t->ptr) (void)hipFree(t->ptr);
    memset(t, 0, sizeof(*t));
}

extern "C" int ds4_gpu_tensor_device(const ds4_gpu_tensor *t) {
    return t ? 0 : -1;
}

extern "C" int ds4_gpu_tensor_copy_async(ds4_gpu_tensor *dst,
                                           const ds4_gpu_tensor *src,
                                           uint64_t bytes) {
    if (!dst || !src || bytes > dst->bytes || bytes > src->bytes) return 0;
    if (bytes == 0) return 1;
    return hipMemcpyAsync(dst->ptr, src->ptr, (size_t)bytes,
                          hipMemcpyDeviceToDevice, 0) == hipSuccess;
}

extern "C" int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor *dst,
                                          const ds4_gpu_tensor *src,
                                          uint64_t bytes) {
    return ds4_gpu_tensor_copy(dst, 0, src, 0, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev_default(ds4_gpu_tensor *dst,
                                                  const ds4_gpu_tensor *src,
                                                  uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev(dst, src, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev_ordered(ds4_gpu_tensor *dst,
                                                  const ds4_gpu_tensor *src,
                                                  uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev(dst, src, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev3(
        ds4_gpu_tensor *dst0, const ds4_gpu_tensor *src0, uint64_t bytes0,
        ds4_gpu_tensor *dst1, const ds4_gpu_tensor *src1, uint64_t bytes1,
        ds4_gpu_tensor *dst2, const ds4_gpu_tensor *src2, uint64_t bytes2) {
    return (bytes0 == 0 || ds4_gpu_tensor_copy_xdev(dst0, src0, bytes0)) &&
           (bytes1 == 0 || ds4_gpu_tensor_copy_xdev(dst1, src1, bytes1)) &&
           (bytes2 == 0 || ds4_gpu_tensor_copy_xdev(dst2, src2, bytes2));
}

extern "C" int ds4_gpu_tensor_copy_xdev3_default_dst(
        ds4_gpu_tensor *dst0, const ds4_gpu_tensor *src0, uint64_t bytes0,
        ds4_gpu_tensor *dst1, const ds4_gpu_tensor *src1, uint64_t bytes1,
        ds4_gpu_tensor *dst2, const ds4_gpu_tensor *src2, uint64_t bytes2) {
    return ds4_gpu_tensor_copy_xdev3(dst0, src0, bytes0, dst1, src1, bytes1,
                                     dst2, src2, bytes2);
}

extern "C" int ds4_gpu_tensor_wait_xdev(const ds4_gpu_tensor *src,
                                          int dst_tier) {
    return src && rocm_tier_valid(dst_tier);
}

extern "C" int ds4_gpu_tensor_wait_xdev_default(const ds4_gpu_tensor *src,
                                                  int dst_tier) {
    return ds4_gpu_tensor_wait_xdev(src, dst_tier);
}

extern "C" uint64_t ds4_gpu_tier_free_vram(int tier) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (!rocm_tier_valid(tier) ||
        hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
        return 0;
    }
    return (uint64_t)free_bytes;
}

extern "C" int ds4_gpu_args_probe_auto_cuda(
        const int *device_filter, int filter_len, ds4_gpu_config *out,
        size_t safety_margin_bytes, char *errbuf, size_t errbuflen) {
    if (!out) {
        if (errbuf && errbuflen) snprintf(errbuf, errbuflen, "internal: NULL out");
        return 1;
    }
    int visible = 0;
    hipError_t rc = hipGetDeviceCount(&visible);
    if (rc != hipSuccess || visible <= 0) {
        if (errbuf && errbuflen) {
            snprintf(errbuf, errbuflen, "hipGetDeviceCount failed: %s",
                     rc == hipSuccess ? "no devices" : hipGetErrorString(rc));
        }
        return 1;
    }
    if (filter_len > 1 || (!device_filter && visible > 1)) {
        if (errbuf && errbuflen) {
            snprintf(errbuf, errbuflen,
                     "ROCm supports one GPU per process; select one device");
        }
        return 1;
    }
    const int device = device_filter && filter_len == 1 ? device_filter[0] : 0;
    if (device < 0 || device >= visible || hipSetDevice(device) != hipSuccess) {
        if (errbuf && errbuflen) {
            snprintf(errbuf, errbuflen, "invalid ROCm device %d", device);
        }
        return 1;
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    rc = hipMemGetInfo(&free_bytes, &total_bytes);
    if (rc != hipSuccess) {
        if (errbuf && errbuflen) {
            snprintf(errbuf, errbuflen, "hipMemGetInfo failed: %s",
                     hipGetErrorString(rc));
        }
        return 1;
    }
    const size_t reserve_floor = (size_t)2ull * 1024ull * 1024ull * 1024ull;
    const size_t reserve_pct = free_bytes / 20u;
    const size_t reserve = reserve_floor > reserve_pct ? reserve_floor : reserve_pct;
    memset(out, 0, sizeof(*out));
    out->device_indices[0] = device;
    out->vram_bytes[0] = free_bytes > reserve ? free_bytes - reserve : 0;
    out->n_gpus = 1;
    out->safety_margin_bytes = safety_margin_bytes;
    return 0;
}

extern "C" void ds4_gpu_enable_q8_dequant_gemm(void) {
}

static int g_rocm_q8_cache_suppressed = 0;

extern "C" int ds4_gpu_q8_cache_suppressed(void) {
    return g_rocm_q8_cache_suppressed;
}

extern "C" void ds4_gpu_set_q8_cache_suppressed(int suppressed) {
    g_rocm_q8_cache_suppressed = suppressed != 0;
}

extern "C" int ds4_gpu_set_decode_fast_attention(int enabled) {
    (void)enabled;
    return 0;
}

extern "C" int ds4_gpu_set_decode_score_vec4(int enabled) {
    (void)enabled;
    return 0;
}

extern "C" int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_rows) {
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                      weight_offset, in_dim, out_dim, x,
                                      n_rows);
}

extern "C" int ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1, const void *model_map,
        uint64_t model_size, uint64_t weight0_offset,
        uint64_t weight1_offset, uint64_t in_dim, uint64_t out0_dim,
        uint64_t out1_dim, const ds4_gpu_tensor *x, uint32_t n_rows) {
    return ds4_gpu_matmul_q8_0_pair_tensor(
            out0, out1, model_map, model_size, weight0_offset, weight1_offset,
            in_dim, out0_dim, out1_dim, x, n_rows);
}

extern "C" int ds4_gpu_matmul_f16_router_rows_exact_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, const ds4_gpu_tensor *x, uint32_t n_rows) {
    return ds4_gpu_matmul_f16_tensor(out, model_map, model_size, weight_offset,
                                     4096u, 256u, x, n_rows);
}

extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
        ds4_gpu_tensor *q_out, const ds4_gpu_tensor *q,
        const void *model_map, uint64_t model_size,
        uint64_t q_weight_offset, uint32_t q_n,
        ds4_gpu_tensor *kv_out, const ds4_gpu_tensor *kv,
        uint64_t kv_weight_offset, uint32_t kv_n, uint32_t rows,
        uint32_t kv_n_head, uint32_t kv_head_dim, uint32_t n_rot,
        uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow, float eps) {
    return ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                   q_out, q, model_map, model_size, q_weight_offset, q_n,
                   kv_out, kv, kv_weight_offset, kv_n, rows, eps) != 0 &&
           ds4_gpu_rope_tail_tensor(
                   kv_out, rows, kv_n_head, kv_head_dim, n_rot, pos0,
                   n_ctx_orig, inverse, freq_base, freq_scale, ext_factor,
                   attn_factor, beta_fast, beta_slow) != 0;
}

extern "C" int ds4_gpu_embed_token_quant_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint32_t n_vocab,
        uint32_t token, uint32_t n_embd) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_embed_token_q8_0_tensor(out, model_map, model_size,
                                           weight_offset, n_vocab, token,
                                           n_embd);
}

extern "C" int ds4_gpu_embed_tokens_quant_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_vocab, uint32_t n_tokens,
        uint32_t n_embd) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_embed_tokens_q8_0_tensor(out, tokens, model_map, model_size,
                                            weight_offset, n_vocab, n_tokens,
                                            n_embd);
}

extern "C" int ds4_gpu_matmul_quant_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint64_t in_dim,
        uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (weight_type == 8u) {
        return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                          weight_offset, in_dim, out_dim, x,
                                          n_tok);
    }
    if (weight_type == 1u) {
        return ds4_gpu_matmul_f16_tensor(out, model_map, model_size,
                                         weight_offset, in_dim, out_dim, x,
                                         n_tok);
    }
    return 0;
}

extern "C" int ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint64_t in_dim,
        uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (weight_type == 8u) {
        return ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
                out, model_map, model_size, weight_offset, in_dim, out_dim,
                x, n_tok);
    }
    return ds4_gpu_matmul_quant_tensor(out, model_map, model_size,
                                       weight_offset, weight_type, in_dim,
                                       out_dim, x, n_tok);
}

extern "C" int ds4_gpu_glm_k_b_project_typed_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *kv_norm,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t kv_lora_dim,
        uint32_t qk_nope, uint32_t n_head) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_glm_k_b_project_tensor(out, kv_norm, model_map, model_size,
                                           weight_offset, n_tokens,
                                           kv_lora_dim, qk_nope, n_head);
}

extern "C" int ds4_gpu_glm_qk_lowrank_typed_tensor(
        ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *q,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_head, uint32_t kv_lora_dim,
        uint32_t qk_nope, uint32_t qk_dim) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_glm_qk_lowrank_q8_0_tensor(
            qk_low, q, model_map, model_size, weight_offset, n_head,
            kv_lora_dim, qk_nope, qk_dim);
}

extern "C" int ds4_gpu_glm_qk_lowrank_typed_batch_tensor(
        ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *q,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_head,
        uint32_t kv_lora_dim, uint32_t qk_nope, uint32_t qk_dim) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor(
            qk_low, q, model_map, model_size, weight_offset, n_tokens, n_head,
            kv_lora_dim, qk_nope, qk_dim);
}

extern "C" int ds4_gpu_glm_value_project_typed_batch_heads_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *lora,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_head,
        uint32_t kv_lora_dim, uint32_t value_dim) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(
            heads, lora, model_map, model_size, weight_offset, n_tokens,
            n_head, kv_lora_dim, value_dim);
}

extern "C" int ds4_gpu_glm_attention_indexed_decode_typed_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache, const void *model_map,
        uint64_t model_size, uint64_t value_weight_offset,
        uint32_t value_weight_type, const ds4_gpu_tensor *selected,
        uint32_t n_selected, uint32_t cache_cap, bool cache_f16,
        uint32_t n_head, uint32_t kv_lora_dim, uint32_t qk_nope,
        uint32_t qk_rope, uint32_t value_dim, uint32_t n_ctx_orig,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow) {
    if (value_weight_type != 8u) return 0;
    return ds4_gpu_glm_attention_indexed_decode_tensor(
            heads, q, qk_low, kv_lora_cache, k_rope_cache, model_map,
            model_size, value_weight_offset, selected, n_selected, cache_cap,
            cache_f16, n_head, kv_lora_dim, qk_nope, qk_rope, value_dim,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow);
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_typed_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache, const void *model_map,
        uint64_t model_size, uint64_t value_weight_offset,
        uint32_t value_weight_type, const ds4_gpu_tensor *selected,
        uint32_t n_tokens, uint32_t n_selected, uint32_t cache_cap,
        bool cache_f16, uint32_t n_head, uint32_t kv_lora_dim,
        uint32_t qk_nope, uint32_t qk_rope, uint32_t value_dim,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast,
        float beta_slow) {
    if (value_weight_type != 8u) return 0;
    return ds4_gpu_glm_attention_indexed_batch_tensor(
            heads, q, qk_low, kv_lora_cache, k_rope_cache, model_map,
            model_size, value_weight_offset, selected, n_tokens, n_selected,
            cache_cap, cache_f16, n_head, kv_lora_dim, qk_nope, qk_rope,
            value_dim, n_ctx_orig, freq_base, freq_scale, ext_factor,
            attn_factor, beta_fast, beta_slow);
}
