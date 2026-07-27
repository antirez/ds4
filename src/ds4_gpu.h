#ifndef DS4_GPU_H
#define DS4_GPU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Hyper-connection (HC) residual-stream storage precision (task #62).
 * The source model runs a BF16 residual (config torch_dtype: bfloat16 — see
 * ds4-source-numerics); our HC carriers were f32, i.e. 2x the precision AND
 * bandwidth of the source with no fidelity gain. We narrow the STORAGE of the
 * six swap-coupled HC residual carriers (cur_hc/after_attn_hc/after_ffn_hc and
 * their batched twins) to BF16, while every kernel keeps accumulating in f32 —
 * exactly what torch does (bf16 storage, f32 math). This macro is the element
 * size of a stored HC residual sample, shared by the CUDA kernels (typed via
 * ds4_hc_t in ds4_cuda_internal.h) and the C host stride/offset math.
 *
 * Define DS4_HC_F32 (compile flag) to restore f32 carriers — the fallback, and
 * the intermediate used to prove the storage-narrowing plumbing is a pure
 * no-op (f32 build must be byte-identical to the pre-change build) before the
 * BF16 flip. One compile-time switch; NO per-token/per-layer runtime branch. */
#ifdef DS4_HC_F32
#define DS4_HC_ELT_SIZE 4u
#else
#define DS4_HC_ELT_SIZE 2u
#endif


/* =========================================================================
 * GPU Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the DS4-specific GPU executor.
 *
 * The public GPU API is tensor-resident: activations, KV state, and scratch
 * buffers stay device-owned across the whole prefill/decode command sequence.
 */
typedef struct ds4_gpu_tensor ds4_gpu_tensor;

int ds4_gpu_init(void);
void ds4_gpu_cleanup(void);

/* Boot-time observability: the resolved gate/up MXFP4 tile width (NT=16 where
 * the device accepts the dynamic-SMEM opt-in, else NT=8) -- the same cached
 * value the mxfp4 prefill path launches at <1024u>.  Resolving it is idempotent
 * (a single cudaFuncSetAttribute) and does not alter any generated token. */
uint32_t ds4_gpu_moe_mxfp4_tile_width(void);

/* Running total of live tensor-alloc bytes (owned allocations only, views
 * excluded).  Snapshot around a session create to measure its true resident
 * cost; the server ledger commits that actual. */
uint64_t ds4_gpu_tensor_alloc_bytes_current(void);
ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);
void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor);
/* Raw device pointer without a synchronize (for building device pointer tables). */
void *ds4_gpu_tensor_device_ptr(const ds4_gpu_tensor *tensor);
int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count);
int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);

/* Batched D2D copy: prepare a device-side descriptor table over fixed tensor
 * allocations once (whole-tensor copies, byte counts multiples of 16; returns
 * NULL on any violation), then replay all copies with one kernel launch.
 * max_bytes is the largest descriptor's byte count (grid sizing). Built for the
 * spec-frontier snapshot/restore paths (~126 tiny per-layer copies per step). */
void *ds4_gpu_batched_copy_prepare(ds4_gpu_tensor **dst, ds4_gpu_tensor **src,
                                   const uint64_t *bytes, uint32_t n);
int ds4_gpu_batched_copy_run(void *handle, uint32_t n_descs, uint64_t max_bytes);
void ds4_gpu_batched_copy_free(void *handle);

int ds4_gpu_begin_commands(void);
int ds4_gpu_flush_commands(void);
int ds4_gpu_end_commands(void);
/* Decode CUDA-graph capture pair: begin returns 1 when the tape is being
 * captured (end replays it as one graph launch and syncs); 0 means graphs
 * are disabled (DS4_CUDA_NO_GRAPHS / unsupported) and the caller must use
 * the plain begin/end_commands pair instead. */
int ds4_gpu_decode_graph_begin(void);
int ds4_gpu_decode_graph_end(void);
int ds4_gpu_synchronize(void);

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size);
int ds4_gpu_set_model_fd(int fd);
int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map);
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
int ds4_gpu_cache_external_range(const void *host_base_key, int fd, uint64_t offset, uint64_t bytes, const char *label);
int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
void ds4_gpu_set_quality(bool quality);
void ds4_gpu_print_memory_report(const char *label);
/* cudaMemGetInfo passthrough (0/0 on failure) for diagnostics/samplers. */
void ds4_gpu_mem_info(uint64_t *free_out, uint64_t *total_out);

/* =========================================================================
 * Embeddings and Indexer Helpers.
 * =========================================================================
 *
 * These kernels seed HC state from token embeddings and implement the ratio-4
 * compressed-attention indexer that chooses visible compressed rows.
 */

int ds4_gpu_embed_token_hc_tensor(
        ds4_gpu_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc);

int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale);

int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

/* Banked (multi-session) mode: positions/seq_id are per-row int32 device
 * arrays (row t's absolute position and TRUE bank id), comp_cap the per-bank
 * compressed-row stride, n_banks the pool size; the comp cache operand is
 * the whole bank pool.  Per-row visible count = (qpos+1)/ratio (the engine's
 * emit-before-read rule); rows past it (and dead rows, seq_id out of pool)
 * score -INF.  Scalar n_comp = cross-bank superset (scan bound + scores-row
 * stride only).  NULL/NULL/0/1 = classic single-cache behavior bit-exactly.
 * Banked multi-token rows run the generic kernel (the WMMA tier stays
 * single-bank); banked n_tokens==1 keeps the direct-one fast tier so the
 * scan is bit-identical to classic single-token decode. */
int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale,
        const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *seq_id,
        const ds4_gpu_tensor *index_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks);

int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/* GPU argmax over n_vocab F32 logits. Writes the winning index as int32 at
 * out_idx[0]. Tie-break: lower index wins (matches host sample_argmax). */
int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab);

int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/* =========================================================================
 * Dense Projections, Norms, RoPE, and KV Rounding.
 * =========================================================================
 *
 * The graph uses these primitives for Q/KV projections, HC/output projections,
 * attention output projections, and DS4's tail-only RoPE.
 */

int ds4_gpu_matmul_mxfp8_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* Register one MXFP8 workhorse weight (attn_kv/q, attn_output, shared experts,
 * output head) by offset so the matmul above executes it; done once at load. */
void ds4_gpu_register_fp8_weight(uint64_t weight_offset);

/* Mark an already-fp8-registered offset as a pre-stored MXFP8_LT weight: the
 * device layout (de-interleaved E4M3 data + swizzled E8M0 scale) is already in
 * the mmap, so the matmul resolver skips the cudaMalloc+convert and points
 * cuBLASLt directly at g_model_device_base+offset. Done once at load. */
void ds4_gpu_register_fp8_lt_weight(uint64_t weight_offset);

/* Batched-prefill activation quantization cache.
 *
 * One normalized activation feeds several block-scaled MXFP8 projections per
 * layer; the per-GEMM activation quantization is a pure function of the buffer
 * and its shape, so it only has to run once.  Arm the cache immediately after
 * the activation is written (arming invalidates any earlier contents, which is
 * what makes a later hit safe even though the buffer pointer is reused every
 * layer); disarm when the activation is dead.  A backend without the
 * optimization may implement both as no-ops.  Purely a traffic optimization:
 * results are bit-identical either way.
 */
void ds4_gpu_mxfp8_act_cache_arm(const ds4_gpu_tensor *x, uint64_t n_tok, uint64_t in_dim);
void ds4_gpu_mxfp8_act_cache_disarm(void);

/* Optional fused GPU operations.
 *
 * These are acceleration hooks, not required backend primitives.  A backend
 * that does not provide the fused kernel must still define the symbol and
 * return 0.  Callers then use the portable sequence of required primitives.
 */
int ds4_gpu_matmul_mxfp8_pair_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight0_offset,
        uint64_t                weight1_offset,
        uint64_t                in_dim,
        uint64_t                out0_dim,
        uint64_t                out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_shared_gate_up_swiglu_mxfp8_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_matmul_f16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* plan-34 phase-2 inc 2/4: arm the M-neutral batched-matmul mode with a PREFIX
 * ROW COUNT. `n` = the number of leading DECODE rows in the batched step; those
 * rows run through the M-independent custom per-token kernels (byte-identical
 * across batch width), while the trailing prefill rows [n..M) take the fast
 * cuBLAS(Lt)/grouped tensor-core path. n==0 disarms (pure prefill / classic).
 * n==M arms the whole batch (decode-only, == inc-2). Set once at
 * multiseq_step_begin, cleared at step_end — never on a per-token path.
 * The query returns the count (MoE two-pass reads it to place the split;
 * inc-2/3 dense-GEMM callers treat nonzero as "armed"). */
void ds4_gpu_matmul_set_batch_mneutral(int n);
int  ds4_gpu_matmul_batch_mneutral(void);   /* query: decode-prefix row count (0 = disarmed) */

int ds4_gpu_matmul_bf16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor       *out_a,
        ds4_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_rms_norm_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_plain_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_head_rms_norm_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps);

/* positions (both RoPE entries below): optional int32 [n_tok] DEVICE array of
 * per-row absolute positions for multi-session banked batches (rows of
 * different sessions sit at unrelated positions).  NULL keeps the classic
 * consecutive pos0+t rule bit-exactly — the multiseq degeneracy invariant.
 * The launcher bounds-checks the array's SIZE; its VALUES are the caller's
 * contract (they are device-side, and a per-step D2H scan to validate them
 * would put host-visible work on the per-token path).  Values are used as
 * uint32 rotation positions: a negative entry rotates at a garbage angle
 * rather than faulting.  gpu_graph_multiseq_step_begin is the host-side
 * validator that every position is > 0 before any launch sees the array. */
int ds4_gpu_head_rms_norm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow,
        float             eps,
        const ds4_gpu_tensor *positions);

int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot);

int ds4_gpu_dsv4_fp8_kv_pack_tensor(
        const ds4_gpu_tensor *x,
        ds4_gpu_tensor       *packed,
        ds4_gpu_tensor       *scales,
        uint32_t               n_tok,
        uint32_t               head_dim);

/* Microscaling (MX) compressed-KV pack/dequant. fmt is DS4_MXKV_FMT_FP8/FP4.
 * Row layout [data][E8M0 scales], block 32; head_dim must be a multiple of 32.
 * `out` (pack) / `in` (dequant) is sized n_tok * DS4_MXKV_ROWBYTES(fmt,head_dim). */
int ds4_gpu_mxkv_pack_tensor(
        const ds4_gpu_tensor *x,
        ds4_gpu_tensor       *out,
        uint32_t               fmt,
        uint32_t               n_tok,
        uint32_t               head_dim);

int ds4_gpu_mxkv_dequant_tensor(
        const ds4_gpu_tensor *in,
        ds4_gpu_tensor       *out,
        uint32_t               fmt,
        uint32_t               n_tok,
        uint32_t               head_dim);

/* DS4_ATTN_PACK compressed-KV storage (value-preserving).  One packed row is
 * [n_nope e4m3 bytes][n_nope/64 E8M0 scale bytes][pad to 4B][n_rot f32 rope]
 * (712 B at head_dim 512 / n_rot 64).  The stored values are exactly the
 * ds4_gpu_dsv4_fp8_kv_quantize_tensor roundtrip for the nope dims and the
 * untouched f32 rope tail, so read-back is bit-identical to the f32 cache.
 * quantize_store additionally roundtrips the f32 source rows IN PLACE
 * (identical to the plain quantize entry) so stages/dumps stay consistent.
 * Requires n_rot == 64 and (head_dim - n_rot) % 64 == 0. */
int ds4_gpu_attn_pack_quantize_store_tensor(
        ds4_gpu_tensor *x,
        ds4_gpu_tensor *packed,
        uint32_t          out_row0,
        uint32_t          n_rows,
        uint32_t          head_dim,
        uint32_t          n_rot);

int ds4_gpu_attn_pack_dequant_tensor(
        const ds4_gpu_tensor *in,
        ds4_gpu_tensor       *out,
        uint32_t               n_rows,
        uint32_t               head_dim,
        uint32_t               n_rot);

/* Repack-only variant for session load: packs ALREADY-roundtripped f32 rows
 * with an exact integer-math scale bucket (value-idempotent; x unmodified).
 * The live emit path must keep using ds4_gpu_attn_pack_quantize_store_tensor,
 * whose fast-math scale matches ds4_gpu_dsv4_fp8_kv_quantize_tensor exactly. */
int ds4_gpu_attn_pack_repack_tensor(
        const ds4_gpu_tensor *x,
        ds4_gpu_tensor       *packed,
        uint32_t               out_row0,
        uint32_t               n_rows,
        uint32_t               head_dim,
        uint32_t               n_rot);

/* Gathered dequant of n_sel rows selected by `rows` (indices into a cap_rows MX
 * cache) into f32 `out`: [n_sel][head_dim] when transpose==0, or [head_dim][n_sel]
 * when transpose!=0 (builds a PV V^T operand). The attention gather primitive. */
int ds4_gpu_mxkv_gather_dequant_tensor(
        const ds4_gpu_tensor *cache,
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *rows,
        uint32_t               n_sel,
        uint32_t               cap_rows,
        uint32_t               head_dim,
        uint32_t               fmt,
        uint32_t               transpose);

int ds4_gpu_dsv4_indexer_qat_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_rows,
        uint32_t          head_dim);

/* QAT-roundtrip n_rows f32 rows of x in place AND store them MXKV-FP4-packed
 * into `packed` at rows [out_row0, out_row0+n_rows).  The f32 result in x is
 * bit-identical to ds4_gpu_dsv4_indexer_qat_tensor. */
int ds4_gpu_dsv4_indexer_qat_pack_tensor(
        ds4_gpu_tensor *x,
        ds4_gpu_tensor *packed,
        uint32_t          out_row0,
        uint32_t          n_rows,
        uint32_t          head_dim);

/* Tell the indexer score kernels the indexer compressed cache is stored
 * MXKV-FP4-packed (68 B/row at head_dim 128) instead of f32. */
void ds4_gpu_indexer_set_fp4(int on);

/* raw_f16 parameter convention (attention readers / raw KV writers below):
 * the flag describes the STORAGE FORMAT OF THE PASSED raw tensor for THIS
 * call — 1 means the raw operand is a __half cache, 0 means f32.  Raw values
 * are already f16-rounded at write time (the f32 store path roundtrips
 * through __float2half), so a __half cache reads back bit-identical floats;
 * f16 only halves storage and read traffic.  Callers must pass the format of
 * the specific buffer they hand in (the persistent layer ring may be __half
 * while e.g. batch/drafter buffers stay f32). */

int ds4_gpu_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow,
        const ds4_gpu_tensor *positions);

/* Release decode fused KV finalizer: after the standalone RoPE kernel, this
 * performs DS4's FP8 non-RoPE KV round trip and writes the F16-rounded raw
 * attention cache row in one dispatch. */
int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          raw_f16);

/* Reference/raw-cache primitive kept for prefill and diagnostics.  Decode uses
 * ds4_gpu_kv_fp8_store_raw_tensor unless a diagnostic reference path is
 * explicitly selected by the graph driver. */
int ds4_gpu_store_raw_kv_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim,
        uint32_t                raw_f16);

/* Banked mode (positions/seq_id non-NULL): row t stores to bank seq_id[t]'s
 * ring at slot positions[t] %% raw_cap over the whole pool (raw_cache = the
 * bank slab, byte-bounded by n_banks); pos0 is ignored.  Dead rows (seq_id
 * out of pool) store nothing.  NULL/NULL/1 = classic consecutive store. */
int ds4_gpu_store_raw_kv_batch_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim,
        uint32_t                raw_f16,
        const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *seq_id,
        uint32_t                n_banks);

/* =========================================================================
 * KV Compression and Attention.
 * =========================================================================
 *
 * Compressed layers maintain rolling score/KV state and append pooled rows at
 * ratio boundaries.  Attention kernels consume raw SWA rows, compressed rows,
 * and optional indexer masks.
 */

int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens);

int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0);

int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                comp_kv_fp8,
        uint32_t                comp_kv_pack,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                raw_f16);

int ds4_gpu_attention_prefill_raw_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                raw_f16);

/* Batched decode attention.  The trailing descriptor quad enables multi-
 * session banked mode: positions/seq_id are int32 [n_tokens] DEVICE arrays
 * (row t's absolute query position and TRUE bank id), comp_cap is the
 * per-bank compressed-row stride and n_banks the pool size; the raw/comp KV
 * operands are then whole bank pools addressed at seq_id*raw_cap /
 * seq_id*comp_cap, with the raw window, ring start, and visible compressed
 * count derived per row from the position.  Scalar n_raw/raw_start are
 * ignored AND unvalidated in banked mode (pass 0); raw_cap must still be
 * the true per-bank ring capacity.  The per-row visible compressed count is
 * (qpos+1)/ratio — identical to the engine's classic single-session decode,
 * which emits a step's compressed row BEFORE attention, so at an emit step
 * (qpos ≡ ratio-1 mod ratio) the row sees the row emitted that same step.
 * DRIVER CONTRACT: in banked mode every bank's compressed rows for the
 * current step — including same-step emits — must be written before the
 * attention launch.  Scalar n_comp is the cross-bank superset clamp, a
 * SAFETY bound only: if it bites (a bank lagging behind its own frontier,
 * e.g. mid-prefill), the row reads fewer rows instead of garbage but its
 * output DIVERGES from single-session — never co-schedule such a bank.
 * seq_id must carry TRUE bank ids in [0, n_banks): a row whose id is out
 * of range (stale slot, -1 sentinel) reads nothing and gets zero head
 * outputs — fail-visible, never a wild read.  Banked mode requires a
 * nonzero window <= 256 (the kernels' per-row raw scratch bound); banked
 * argument rejections return 0 and print the reason to stderr.  Pass
 * NULL/NULL/0/1 for the classic single-cache behavior — bit-exact. */
int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                non_causal,
        uint32_t                raw_f16,
        const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *seq_id,
        uint32_t                comp_cap,
        uint32_t                n_banks);

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                comp_kv_fp8,
        uint32_t                comp_kv_pack,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                non_causal,
        uint32_t                raw_f16,
        const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *seq_id,
        const ds4_gpu_tensor *comp_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks);

int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                comp_kv_fp8,
        uint32_t                comp_kv_pack,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                raw_f16,
        const ds4_gpu_tensor *positions,
        const ds4_gpu_tensor *seq_id,
        const ds4_gpu_tensor *comp_bank_ptrs,
        uint32_t                comp_cap,
        uint32_t                n_banks);

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                comp_kv_fp8,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                raw_f16);

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                comp_kv_fp8,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                raw_f16);

int ds4_gpu_attention_output_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

int ds4_gpu_attention_output_low_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads);

/* =========================================================================
 * Router, Shared Expert, and Routed MoE.
 * =========================================================================
 *
 * These kernels implement the FFN body: router probabilities/top-k or hash
 * routing, shared SwiGLU, and the IQ2_XXS/Q2_K/MXFP4 routed experts.
 */

int ds4_gpu_swiglu_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight);

int ds4_gpu_add_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t                n);

int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale);

int ds4_gpu_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits);

int ds4_gpu_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);


int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index);

int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens,
        bool                   *mid_is_f16);

/* CUTLASS Sm120 block-scaled MXFP4 grouped-expert FFN (DS4_TENSOR_CUTLASS_MXFP4, type 40).
 * out[T,out_dim] = down(swiglu(x.Wg^T, x.Wu^T)).Wd^T for T tokens ALL ROUTED TO ONE EXPERT
 * (the caller gathers per-expert rows via sorted_pairs before calling this, and scatters the
 * result back). Wg/Wu/Wd are that expert's CUTLASS-packed [data, then SF] blob pointers, sliced
 * by the caller from cutlass_mxfp4_expert_layout()'s stride/split-point. `scratch` must be at
 * least ds4_cutlass_expert_ffn_scratch_bytes(T,in_dim,mid_dim,out_dim) bytes; size once for the
 * layer's shape at the largest T a single expert can see and reuse across every expert and
 * every CUTLASS-typed layer sharing that shape -- this function does no allocation and no
 * synchronization, unlike ds4_cutlass_expert_ffn (used only by the standalone test). */
size_t ds4_cutlass_expert_ffn_scratch_bytes(int T, int in_dim, int mid_dim, int out_dim);
int ds4_cutlass_expert_ffn_scratch(
        float          *out,
        const float    *x,
        const uint8_t  *Wg_d,
        const uint8_t  *Wg_sf,
        const uint8_t  *Wu_d,
        const uint8_t  *Wu_sf,
        const uint8_t  *Wd_d,
        const uint8_t  *Wd_sf,
        const float    *weights,
        float           clamp,
        int             T,
        int             in_dim,
        int             mid_dim,
        int             out_dim,
        uint8_t        *scratch,
        size_t          scratch_bytes);

/* Small-batch (n_tokens 2..4) rich-expert FFN over the packed CUTLASS weights via direct
 * fp4 GEMV: one gate+up+swiglu launch and one down launch over all (token,expert) slots,
 * no sort, no host readback, f32 activations. down_out gets one pre-weighted FFN result
 * per slot at [slot*out_dim]; the caller sums the n_expert slices per token (moe_sum).
 * mid_scratch must hold n_tokens*n_expert*mid_dim floats. selected/rweights are the
 * device [n_tokens,n_expert] routing outputs. Returns 0 on success. */
int ds4_cutlass_expert_ffn_gemv_small(
        float          *down_out,
        float          *mid_scratch,
        const float    *x,
        const int32_t  *selected,
        const float    *rweights,
        const uint8_t  *gate_w,
        const uint8_t  *up_w,
        const uint8_t  *down_w,
        uint64_t        gate_stride,
        uint64_t        gate_data_bytes,
        uint64_t        down_stride,
        uint64_t        down_data_bytes,
        float           clamp,
        int             n_tokens,
        int             n_expert,
        unsigned        n_total_expert,
        int             in_dim,
        int             mid_dim,
        int             out_dim);

/* Grouped (ptr-array) MXFP4 prefill FFN: runs EVERY active expert's gate/up/down as a single
 * blockscaled grouped GEMM launch each -- replacing the per-expert host loop + blocking offsets
 * readback in routed_moe_launch_cutlass. Per-group problem shapes, A/B/D + SFA/SFB pointer arrays
 * and SF-layouts are built on device from `counts`/`padded_offsets`; there is no host readback.
 *
 * The caller must gather each expert's tokens to 128-ROW-PADDED offsets (padded_offsets[e], a
 * multiple of 128) into x_gathered[padded_total,in_dim] + w_gathered[padded_total], PRE-ZEROING
 * padding rows, because the SM120 SF atom spans 128 rows and each group's SF must start on a
 * 128-row boundary. Output ffn_out[padded_total,out_dim] holds one pre-weighted result per padded
 * row; the caller scatters the real rows into the flat down buffer (via the same padded map),
 * then moe_sum reduces. `padded_total` is a host upper bound (multiple of 128); inactive experts
 * carry M=0 and contribute nothing. `scratch` must be >= the *_scratch_bytes value. No host sync. */
size_t ds4_cutlass_grouped_moe_scratch_bytes(
        int padded_total, int n_total_expert, int in_dim, int mid_dim, int out_dim);
int ds4_cutlass_grouped_moe(
        float          *ffn_out,
        const float    *x_gathered,
        const float    *w_gathered,
        const uint8_t  *gate_w,
        const uint8_t  *up_w,
        const uint8_t  *down_w,
        uint64_t        gate_stride,
        uint64_t        gate_data_bytes,
        uint64_t        down_stride,
        uint64_t        down_data_bytes,
        float           clamp,
        int             n_total_expert,
        int             in_dim,
        int             mid_dim,
        int             out_dim,
        const uint32_t *counts,
        const uint32_t *padded_offsets,
        int             padded_total,
        uint8_t        *scratch,
        size_t          scratch_bytes);

/* Single-projection W4A8 GEMM for MIXED type-40 + iq2/q2k layers. Computes out[T,out_dim] =
 * x[T,in_dim] . W[out_dim,in_dim]^T for ONE expert's type-40 CUTLASS weight (data at W_d, swizzled
 * SFB at W_sf), packing x to E4M3 dynamic block-scaled activations -- bit-identical to a single
 * projection of the uniform grouped path. Caller gathers x contiguously (T = tokens for that
 * expert) and sizes scratch once via ds4_cutlass_proj_scratch_bytes(). No allocation, no sync. */
size_t ds4_cutlass_proj_scratch_bytes(int T, int in_dim, int out_dim);
int ds4_cutlass_proj_scratch(float *out, const float *x,
        const uint8_t *W_d, const uint8_t *W_sf, int T, int in_dim, int out_dim,
        uint8_t *scratch, size_t scratch_bytes);

/* Grouped single-projection W4A8 GEMM for MIXED layers -- one device-built ptr-array grouped GEMM
 * over 128-padded gathered activations: out[padded_total,out_dim] = x_gathered . W^T for every
 * active expert (W_base+e*W_stride data, +W_data_bytes swizzled SFB). No host readback, no per-expert
 * sync; bit-identical to the per-expert single-proj path (same pack + gather order + GEMM). Padding
 * rows must be pre-zeroed. Caller sizes scratch once via ds4_cutlass_grouped_proj_scratch_bytes(). */
size_t ds4_cutlass_grouped_proj_scratch_bytes(int padded_total, int n_total_expert, int in_dim, int out_dim);
int ds4_cutlass_grouped_proj(float *out, const float *x_gathered,
        const uint8_t *W_base, uint64_t W_stride, uint64_t W_data_bytes,
        int n_total_expert, int in_dim, int out_dim,
        const uint32_t *counts, const uint32_t *padded_offsets, int padded_total,
        uint8_t *scratch, size_t scratch_bytes);

/* Single-projection W4A8 GEMV for MIXED type-40 layers at decode/small-batch (n<=4): lean fp4-weight
 * GEMV with E4M3-roundtripped f32 activations (same function as the prefill grouped GEMM), one launch
 * over all (token,expert) slots, no per-expert loop/host sync. mid/down_out are pair-layout f32. */
int ds4_cutlass_gemv_gateup(float *mid, const float *x, const int32_t *selected, const float *rweights,
        const uint8_t *gate_w, const uint8_t *up_w, uint64_t gate_stride, uint64_t gate_data_bytes,
        float clamp, int n_tokens, int n_expert, unsigned n_total_expert, int in_dim, int mid_dim);
int ds4_cutlass_gemv_down(float *down_out, const float *mid, const int32_t *selected,
        const uint8_t *down_w, uint64_t down_stride, uint64_t down_data_bytes,
        int n_tokens, int n_expert, unsigned n_total_expert, int mid_dim, int out_dim);

/* Runtime dequant->fp4 weight packer for the 2-bit prefill path: quantizes a dequantized f32
 * weight [N,K] (N rows of K, RowMajor) to MXFP4 on-device (LOSSY) into CUTLASS B layout
 * (packed E2M1 `Bd` + swizzled ue8m0 `Bsf`), byte-identical to ds4_cutlass_pack_source so the
 * FFN above consumes it unchanged. N must be even. Sizes below give the two output regions. */
size_t ds4_cutlass_weight_data_bytes(int N, int K);
size_t ds4_cutlass_weight_sf_count(int N, int K);
void   ds4_cutlass_pack_weight_f32(uint8_t *Bd, uint8_t *Bsf, const float *W, int N, int K);

/* =========================================================================
 * Hyper-Connection Kernels.
 * =========================================================================
 *
 * HC kernels reduce four residual streams before a sublayer and expand the
 * sublayer output back into four streams afterward.
 */

int ds4_gpu_hc_split_sinkhorn_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *mix,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_gpu_hc_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_weighted_sum_split_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* Release decode fused HC pre-sublayer operation: split the HC mixer and
 * immediately reduce four HC streams into the active 4096-wide sublayer row. */
int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps);

/* Fused plain-RMSNorm + f16 HC-mix GEMV (decode, n_tok == 1).  Byte-identical
 * to rms_norm_plain_tensor() followed by matmul_f16_tensor(); see the kernel
 * comment in ds4_cuda_hc_router.cu for the order argument.  `x` is an HC
 * residual CARRIER (ds4_hc_t storage, DS4_HC_ELT_SIZE bytes/sample), not f32. */
int ds4_gpu_hc_norm_mix_f16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   eps);

int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps);

int ds4_gpu_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_add_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_shared_down_hc_expand_mxfp8_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_matmul_fp8_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* DSpark Markov + confidence heads */

int ds4_gpu_dspark_markov_step_model(
        ds4_gpu_tensor       *refined_logits,
        int32_t               *refined_id_dst,
        int32_t               *refined_id2_dst,
        const ds4_gpu_tensor *base_logits,
        const void             *dspark_model_map,
        uint64_t                dspark_model_size,
        uint64_t                markov_w1_offset,
        uint64_t                markov_w2_offset,
        int32_t                prev_token,
        uint32_t               vocab_size,
        uint32_t               embed_dim);

int ds4_gpu_dspark_hc_mean_reduce(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *after_ffn_hc,
        uint32_t               n_embd,
        uint32_t               n_hc);

int ds4_gpu_dspark_hc_mean_reduce_batch(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *hc_batch,
        uint32_t               n_embd,
        uint32_t               n_hc,
        uint32_t               n_tokens);

/* DSpark confidence head: per block position, confidence that the draft is
 * accepted. hidden = post-hc_head drafter hidden [n_positions, hidden_dim];
 * token_ids = block token per position; markov_w1/proj resolved from the dspark
 * model map. Drives confidence-scheduled verification (sizing the draft length). */
int ds4_gpu_dspark_confidence_score_model(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *hidden,
        const ds4_gpu_tensor *token_ids,
        const void             *dspark_model_map,
        uint64_t                dspark_model_size,
        uint64_t                markov_w1_offset,
        uint64_t                proj_offset,
        uint32_t                n_positions,
        uint32_t                hidden_dim,
        uint32_t                embed_dim,
        uint32_t                vocab_size);


/* ===========================================================================
 * Wrong-arch build trap (checked at startup by ds4_gpu_init).
 *
 * A plain `make` rebuilds stale CUDA objects WITHOUT CUDA_ARCH=sm_120f, so
 * they compile for nvcc's default arch (sm_75 on CUDA 13).  The binary still
 * links and loads on the GB10, then dies mid-run with an opaque device assert
 * the first time a kernel needing sm_120f features launches — and `make test`
 * cannot catch it.  Every nvcc-compiled TU that includes this header (all of
 * src/cuda/ does, directly or via ds4_cuda_internal.h) registers the
 * __CUDA_ARCH_LIST__ it was compiled with; ds4_gpu_init() verifies each TU
 * carries code for the device's SM family and aborts startup with an
 * actionable message instead.  Per-TU registration also catches MIXED builds
 * (one stale object rebuilt default-arch by a plain `make`).  Host C compiles
 * never define __CUDA_ARCH_LIST__, so this block is nvcc-only; the `inline`
 * list head merges into one copy per linked binary, so any subset of objects
 * links (no central symbol required).
 */
#if defined(__cplusplus) && defined(__CUDA_ARCH_LIST__)
typedef struct ds4_tu_archs {
    const char            *file;    /* __BASE_FILE__ of the registering TU */
    const int             *archs;   /* __CUDA_ARCH_LIST__ entries, e.g. 1200 */
    int                     n_archs;
    struct ds4_tu_archs   *next;
} ds4_tu_archs;
inline ds4_tu_archs *ds4_tu_archs_head = nullptr;
namespace {
const int ds4_tu_arch_list_[] = { __CUDA_ARCH_LIST__ };
ds4_tu_archs ds4_tu_archs_rec_ = {
    __BASE_FILE__, ds4_tu_arch_list_,
    (int)(sizeof(ds4_tu_arch_list_) / sizeof(ds4_tu_arch_list_[0])), nullptr
};
__attribute__((constructor)) void ds4_tu_archs_register_(void) {
    ds4_tu_archs_rec_.next = ds4_tu_archs_head;
    ds4_tu_archs_head = &ds4_tu_archs_rec_;
}
}
#endif

#endif
