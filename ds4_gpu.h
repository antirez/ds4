#ifndef DS4_GPU_H
#define DS4_GPU_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
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
#ifndef DS4_GPU_TENSOR_DEFINED
#define DS4_GPU_TENSOR_DEFINED
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
#endif

#ifndef DS4_GPU_ATTENTION_DECODE_ROW_DEFINED
#define DS4_GPU_ATTENTION_DECODE_ROW_DEFINED
#define DS4_GPU_ATTENTION_DECODE_BATCH_MAX 32u
typedef struct {
    uint64_t raw_kv;
    uint64_t comp_kv;
    uint64_t topk;
    uint32_t pos;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t window;
    uint32_t ratio;
    uint32_t indexed;
} ds4_gpu_attention_decode_row;
#endif

int ds4_gpu_init(void);
void ds4_gpu_cleanup(void);

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);
void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor);
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
/* Stable CUDA allocation identity, including a view's byte offset.  Unlike
 * the wrapper handle, this stays unchanged when an equivalent tensor view is
 * recreated and is therefore suitable for CUDA graph-cache keys. */
uintptr_t ds4_gpu_tensor_storage_key(const ds4_gpu_tensor *tensor);
#endif
int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count);
int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);
int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count);
int ds4_gpu_moe_handoff_pack_tensor(
        ds4_gpu_tensor       *packed,
        const ds4_gpu_tensor *ffn_norm,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_embd,
        uint32_t              n_expert);
int ds4_gpu_pack_slot_rows_f32_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *slots,
        uint32_t                n_rows,
        uint32_t                width,
        uint32_t                n_slots,
        uint32_t                slot_cap);

int ds4_gpu_begin_commands(void);
int ds4_gpu_flush_encoder(void);
int ds4_gpu_flush_commands(void);
#ifdef __APPLE__
/* Commit the current Metal batch without draining.  The completion hook runs
 * on a Metal-owned thread and must only publish thread-safe readiness state;
 * the next full command drain joins it before returning. */
int ds4_gpu_flush_commands_progress(void (*report)(void *ctx), void *ctx);
#endif
int ds4_gpu_commands_active(void);
#ifdef __APPLE__
int ds4_gpu_parallel_ffn_finish(void);
void ds4_gpu_parallel_ffn_abort(void);
int ds4_gpu_parallel_ffn_start(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *shared_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              model_dim,
        uint32_t              shared_dim,
        const ds4_gpu_tensor *x,
        float                 clamp);
int ds4_gpu_parallel_ffn_start_sliced(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *shared_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              model_dim,
        uint32_t              shared_dim,
        uint32_t              shared_lane_offset,
        uint32_t              shared_lane_count,
        const ds4_gpu_tensor *x,
        float                 clamp);

/* GPU-decided shared-expert lane split for two-rank TP decode: the split
 * kernels read the selected expert ids and take complementary lane ranges
 * sized to balance the bytes each rank streams (shift_q16 = routed expert
 * bytes / (2 * shared expert bytes) in Q16; 0 reproduces static halves). */
int ds4_gpu_parallel_ffn_start_split(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *shared_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              model_dim,
        uint32_t              shared_dim,
        const ds4_gpu_tensor *x,
        float                 clamp,
        const ds4_gpu_tensor *selected,
        uint32_t              tp_rank,
        uint32_t              tp_world,
        uint32_t              n_expert,
        uint32_t              n_expert_used,
        uint32_t              shift_q16);
#endif

/* out = a + b into this rank's TP slab slot for (layer, gate), publishing the
 * gate's checked flag from the same kernel; falls back to ds4_gpu_add_tensor
 * when the fold does not apply.  Call right before ds4_gpu_tp_gate_encode. */
int ds4_gpu_add_tensor_tp_flag(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t              n,
        uint32_t              layer,
        uint32_t              gate);

/* Register that the next TP partial producer for (layer, gate) may publish
 * the gate's checked flag itself (taken by the attention output K-slice
 * matvec when its output is that slot; otherwise ignored). */
void ds4_gpu_tp_flag_fold_request(uint32_t layer, uint32_t gate);

/* Deferred kv norm task: call before ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor
 * to run only its q task now and fold the kv task into the KV staging
 * kernel of the same layer; flush runs it standalone if nothing consumed it. */
void ds4_gpu_dsv4_qkv_norm_defer_kv_next(void);
int ds4_gpu_kv_norm_task_pending(void);
int ds4_gpu_kv_norm_task_flush(void);
int ds4_gpu_kv_norm_task_begin_concurrent(void);
void ds4_gpu_kv_norm_task_end_concurrent(void);
int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value);
int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value, const char *label);
int ds4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label);
#if defined(DS4_ROCM_BUILD) || \
    (!defined(__APPLE__) && !defined(DS4_NO_GPU))
int ds4_gpu_tensor_read_after_selected_event(const ds4_gpu_tensor *tensor,
                                             uint64_t offset,
                                             void *data,
                                             uint64_t bytes,
                                             uint64_t event_value,
                                             const char *label);
#endif
int ds4_gpu_end_commands(void);
#ifdef __APPLE__
/* Metal-only asynchronous command streams. Command encoding remains
 * serialized; at most eight committed streams may execute concurrently. */
void ds4_gpu_set_stream(int idx);
int ds4_gpu_current_stream(void);
int ds4_gpu_end_commands_async(void);
int ds4_gpu_wait_stream(int idx);
#endif
int ds4_gpu_synchronize(void);

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size);
int ds4_gpu_set_model_fd(int fd);
int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map);
#if defined(DS4_BENCH_CUDA) || \
    (!defined(__APPLE__) && !defined(DS4_ROCM_BUILD) && \
     !defined(DS4_NO_GPU))
/* CUDA benchmark/test controls.  These are deliberately explicit API hooks
 * rather than environment knobs: production callers must not depend on
 * strict dispatch or backend-internal model provenance. */
int ds4_cuda_test_model_range_is_device_resident(
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        uint64_t bytes,
        int logical_tier);
int ds4_cuda_test_model_range_device_ptr(
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        uint64_t bytes,
        int logical_tier,
        const void **device_ptr);
void ds4_cuda_test_set_q4_mmq_strict(int required);
#endif
/* Prepare a second, fully resident support GGUF without replacing the active
 * target-model mapping used by SSD streaming. */
int ds4_gpu_prepare_support_model(const void *model_map, uint64_t model_size,
                                  uint64_t map_offset, uint64_t map_size,
                                  uint64_t max_tensor_bytes);
int ds4_gpu_build_derived_artifacts(const void *model_map, uint64_t model_size,
                                    const char *model_path);
int ds4_gpu_model_range_replaced(const void *model_map, uint64_t offset,
                                 uint64_t bytes);
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
/* Add a secondary GGUF mapping without replacing the primary model mapping. */
int ds4_gpu_set_aux_model_map_range(const void *model_map,
                                    uint64_t model_size,
                                    uint64_t map_offset,
                                    uint64_t map_size);
int ds4_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes);
int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label);
int ds4_gpu_q8_cache_suppressed(void);
void ds4_gpu_set_q8_cache_suppressed(int suppressed);
#ifdef DS4_ROCM_BUILD
void ds4_gpu_release_q8_f16_cache(void);
#endif

/* Model-file ranges assigned to CUDA devices by the multi-GPU placement
 * planner. Metal keeps these declarations for the shared engine interface. */
#ifndef DS4_MAX_GPUS
#define DS4_MAX_GPUS 16
#endif
typedef struct {
    uint64_t source_offset;
    uint64_t bytes;
    int target_device;
} ds4_tensor_range;

int ds4_gpu_device_cache_tensors(int device_id,
                                 const ds4_tensor_range *ranges,
                                 int n_ranges);
int ds4_gpu_register_support_map(const void *map, uint64_t size, uint64_t bias);
int ds4_gpu_device_cache_support_tensors(int device_id,
                                         int entry_device_id,
                                         const ds4_tensor_range *ranges,
                                         int n_ranges,
                                         int from_main_map);
uint64_t ds4_gpu_tier_free_vram(int logical_tier);
int ds4_gpu_lookup_cache(uint64_t source_offset, uint64_t bytes,
                         int *out_device_id, void **out_device_ptr);
int ds4_gpu_lookup_cache_device(uint64_t source_offset, uint64_t bytes);

int ds4_gpu_pro_q4_expert_table_auto_available(void);
int ds4_gpu_preload_q4_expert_tables(const void *model_map, uint64_t model_size,
                                     uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
                                     uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
                                     uint32_t n_total_expert);
int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
void ds4_gpu_set_quality(bool quality);
void ds4_gpu_set_glm_model(bool enabled);
void ds4_gpu_set_ssd_streaming(bool enabled);
void ds4_gpu_set_glm_streaming_prefill_full_layer(bool enabled);

typedef struct ds4_gpu_q4_attn_q_b_f16_sidecar_desc {
    uint64_t weight_offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    uint32_t weight_type;
    uint32_t layer;
} ds4_gpu_q4_attn_q_b_f16_sidecar_desc;

/* Backend-neutral prefill preflight for optional Q4_K attn_q_b F16
 * acceleration.  A backend may prepare resident sidecars or reusable
 * transient scratch/pipelines according to its policy.  Prepare returns 1
 * when the selected path is ready, 0 for a policy/safety skip, and -1 when
 * strict mode requires an unavailable specialization. */
int ds4_gpu_prepare_q4_attn_q_b_f16_sidecars(
        const void *model_map,
        uint64_t model_size,
        const ds4_gpu_q4_attn_q_b_f16_sidecar_desc *descs,
        uint32_t count,
        uint32_t max_prefill_rows,
        uint64_t working_set_reserve_bytes,
        uint64_t *prepared_bytes);
/* Release sidecars at a quiescent backend lifecycle point.  Returns zero
 * only when pending GPU work could not be synchronized safely. */
int ds4_gpu_release_q4_attn_q_b_f16_sidecars(void);
uint64_t ds4_gpu_q4_attn_q_b_f16_cache_generation(void);
/* Evict resident sidecars before adding a graph to a live-session set. */
int ds4_gpu_make_room_for_q4_attn_q_b_f16_session(void);

#ifdef __APPLE__
int ds4_gpu_device_is_pre_m5_apple_silicon(void);
int ds4_gpu_device_is_m5_apple_silicon(void);
int ds4_gpu_set_decode_pipeline_fast_lookup(int enabled);
/* Strict test oracle for the fixed decode mul_mv pipeline lookup cache. */
int ds4_gpu_test_decode_pipeline_fast_lookup(void);
/* Strict test oracle for the extended decode mul_mv_ext (nsg + nxpsg) cache. */
int ds4_gpu_test_decode_pipeline_fast_lookup_ext(void);
/* Strict test oracle for the generated resident-prefill MXFP4 half LUT. */
int ds4_gpu_test_mxfp4_down_half_lut(uint16_t *legacy_bits,
                                     uint16_t *lut_bits);
typedef struct ds4_gpu_iq2_mid_only_oracle_report {
    uint64_t mid_words;
    uint64_t mid_mismatches;
    uint64_t canonical_gate_unwritten;
    uint64_t canonical_up_unwritten;
    uint64_t candidate_gate_writes;
    uint64_t candidate_up_writes;
    uint64_t masked_mid_mismatches;
    uint64_t masked_inactive_writes;
    uint64_t masked_canonical_gate_unwritten;
    uint64_t masked_canonical_up_unwritten;
    uint64_t masked_gate_writes;
    uint64_t masked_up_writes;
    uint64_t guard_byte_mismatches;
} ds4_gpu_iq2_mid_only_oracle_report;
/* Full-shape address-table oracle for the experimental M1 IQ2 mid-only
 * producer.  Return zero means setup/execution failure; numerical and
 * sentinel failures are reported explicitly in `report`. */
int ds4_gpu_test_iq2_addr_mid_only_oracle(
        ds4_gpu_iq2_mid_only_oracle_report *report);
typedef struct ds4_gpu_stream_expert_live_index_report {
    uint64_t scans;
    uint64_t entries;
    uint64_t fallbacks;
    uint64_t inserts;
    uint64_t removes;
    uint64_t reuse_scan_calls;
    uint64_t reuse_scan_entries;
    uint64_t resident_hash;
    uint32_t live_count;
    uint32_t cache_entries;
    uint32_t eligible;
    uint32_t active;
    uint32_t broken;
} ds4_gpu_stream_expert_live_index_report;
/* Test-only policy/state hooks for the IQ2 production-size SSD cache index. */
int ds4_gpu_test_stream_expert_live_index_policy(
        int ssd_streaming,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        int enable,
        int disable);
void ds4_gpu_test_stream_expert_live_index_report(
        ds4_gpu_stream_expert_live_index_report *report);
typedef struct ds4_gpu_exact_rows_persistent_report {
    uint64_t persistent_calls;
    uint64_t transient_calls;
    uint64_t persistent_fallbacks;
    uint64_t persistent_failures;
    uint64_t mapped_view_calls;
    uint32_t max_unique;
} ds4_gpu_exact_rows_persistent_report;
/* Test-only policy and counters for exact-row private cache snapshots. */
int ds4_gpu_test_exact_rows_persistent_policy(
        uint32_t configured_count,
        uint32_t unique_count,
        int      size_class_ok);
void ds4_gpu_test_exact_rows_persistent_report(
        ds4_gpu_exact_rows_persistent_report *report);
typedef struct ds4_gpu_q4_attn_q_b_f16_cache_report {
    uint64_t entries;
    uint64_t bytes;
    uint64_t lookups;
    uint64_t hits;
    uint64_t misses;
    uint64_t builds;
    uint64_t build_failures;
    uint64_t candidate_calls;
    uint64_t fallbacks;
    uint64_t rejects;
    uint64_t build_circuit_open;
    uint64_t transient_exact_views_created;
    uint64_t transient_exact_views_live;
    uint64_t model_exact_cache_entries;
    uint64_t model_exact_cache_bytes;
} ds4_gpu_q4_attn_q_b_f16_cache_report;
/* Test observability for the resident Metal Q4_K attn_q_b F16 sidecar. */
void ds4_gpu_test_q4_attn_q_b_f16_cache_report(
        ds4_gpu_q4_attn_q_b_f16_cache_report *report);
void ds4_gpu_test_q4_attn_q_b_f16_cache_reset(void);
int ds4_gpu_test_q4_attn_q_b_f16_projection_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tok);
typedef enum ds4_gpu_test_q4_qb_mm_arm {
    DS4_GPU_TEST_Q4_QB_MM_Q4_F32 = 0,
    DS4_GPU_TEST_Q4_QB_MM_Q4_F16 = 1,
    DS4_GPU_TEST_Q4_QB_MM_F16_F32 = 2,
    DS4_GPU_TEST_Q4_QB_MM_F16_F16 = 3,
    DS4_GPU_TEST_Q4_QB_MM_Q4_TRANSIENT_F16_F16 = 4,
    DS4_GPU_TEST_Q4_QB_MM_ARM_COUNT = 5,
} ds4_gpu_test_q4_qb_mm_arm;
/* Runtime capability probe for test-only matmul arms. */
int ds4_gpu_test_q4_attn_q_b_mm_arm_supported(
        ds4_gpu_test_q4_qb_mm_arm arm);
/* Strict projection-only resident benchmark hook.  F16-weight arms require
 * a READY sidecar; F16-RHS arms optionally include the production copy.  The
 * transient arm rebuilds its F16 weight matrix for every benchmark/oracle
 * projection and may consume either a prepacked or freshly copied F16 RHS. */
int ds4_gpu_test_q4_attn_q_b_mm_variant_tensor(
        ds4_gpu_tensor              *out_f32,
        ds4_gpu_tensor              *rhs_f16,
        const void                  *model_map,
        uint64_t                     model_size,
        uint64_t                     weight_offset,
        uint64_t                     in_dim,
        uint64_t                     out_dim,
        const ds4_gpu_tensor        *x_f32,
        uint32_t                     n_tok,
        ds4_gpu_test_q4_qb_mm_arm    arm,
        bool                         materialize_rhs);
int ds4_gpu_test_q4_attn_q_b_f16_working_set_policy(
        uint64_t recommended,
        uint64_t allocated,
        uint64_t additional);
typedef struct ds4_gpu_stream_test_stats {
    uint64_t tensor_live_bytes;
    uint64_t transient_references;
    uint32_t tensor_live_count;
    uint32_t pending_command_buffers;
    uint32_t last_command_buffers;
    uint32_t active_queue_mask;
    uint32_t model_residency_queue_mask;
    uint32_t q4_residency_queue_mask;
} ds4_gpu_stream_test_stats;
/* Test-only observability and lifetime injection for Metal stream oracles. */
void ds4_gpu_test_stream_stats(ds4_gpu_stream_test_stats *stats);
int ds4_gpu_test_hold_stream_transient(uint64_t bytes);
enum {
    DS4_GPU_TEST_MXFP4_PAIR_TAIL_CULL = 1u << 0,
    DS4_GPU_TEST_MXFP4_PAIR_COMPACT_TILE = 1u << 1,
    DS4_GPU_TEST_MXFP4_MAP_SCATTER = 1u << 2,
    DS4_GPU_TEST_MXFP4_DOWN_TAIL_CULL = 1u << 3,
    DS4_GPU_TEST_MXFP4_DOWN_HALF_LUT = 1u << 4,
    DS4_GPU_TEST_OUTPUT_HC_WEIGHTS4 = 1u << 5,
    DS4_GPU_TEST_HC_RMS_SCALE_PROJ = 1u << 6,
    DS4_GPU_TEST_STREAMING_LIVE_INDEX_FAILURE = 1u << 7,
    DS4_GPU_TEST_IQ2_SSD_GROUPED_PIPELINE_FAILURE = 1u << 8,
    DS4_GPU_TEST_ATTN_OUT_LOW_Q8_STATIC = 1u << 9,
    DS4_GPU_TEST_BATCH_ATTN_OUT_Q8_HC_FUSION = 1u << 10,
    DS4_GPU_TEST_BATCH_ATTN_OUT_Q4_HC_FUSION = 1u << 11,
    DS4_GPU_TEST_FLASH_ATTN_SMALL_PREFILL_NWG32 = 1u << 12,
    DS4_GPU_TEST_FLASH_ATTN_SMALL_PREFILL_NWG1_FAILURE = 1u << 13,
    DS4_GPU_TEST_REQUIRE_IQ2_TOP8_PAIR_SWIGLU = 1u << 14,
};
void ds4_gpu_test_set_flags(uint32_t flags);
double ds4_gpu_test_last_completed_gpu_ms(void);
uint32_t ds4_gpu_test_last_flash_attn_prefill_nwg(void);
int ds4_gpu_test_reset_flash_attn_tmp(void);
uint64_t ds4_gpu_test_flash_attn_tmp_bytes(void);
void ds4_gpu_release_zero_prefix_prefill_mask_cache(void);
#else
static inline int ds4_gpu_device_is_pre_m5_apple_silicon(void) { return 0; }
static inline int ds4_gpu_device_is_m5_apple_silicon(void) { return 0; }
#endif
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU)
typedef struct ds4_cuda_stream_selected_batch_io_report {
    uint64_t candidates;
    uint64_t attempts;
    uint64_t completed;
    uint64_t legacy_batches;
    uint64_t safe_fallbacks;
    uint64_t failures;
    uint64_t required_failures;
    uint64_t oracle_runs;
    uint64_t oracle_failures;
    uint64_t tasks;
    uint64_t segments;
    uint64_t reads;
    uint64_t bytes;
    int enabled;
    int required;
    int oracle;
} ds4_cuda_stream_selected_batch_io_report;
typedef struct ds4_cuda_stream_selected_event_pipeline_report {
    uint64_t candidates;
    uint64_t signals;
    uint64_t readbacks;
    uint64_t uploads;
    uint64_t compute_waits;
    uint64_t safe_fallbacks;
    uint64_t failures;
    uint64_t required_failures;
    uint64_t oracle_runs;
    uint64_t oracle_failures;
    int enabled;
    int required;
    int oracle;
} ds4_cuda_stream_selected_event_pipeline_report;
typedef struct ds4_cuda_iq2_ssd_grouped_report {
    uint64_t candidates;
    uint64_t eligible;
    uint64_t attempts;
    uint64_t completed;
    uint64_t not_applicable;
    uint64_t safe_fallbacks;
    uint64_t failures;
    uint64_t required_failures;
    uint64_t upload_waits;
    uint64_t lease_waits;
    uint64_t lease_records;
    uint64_t lease_drains;
    int enabled;
    int required;
    int stats;
} ds4_cuda_iq2_ssd_grouped_report;
typedef struct ds4_cuda_stream_expert_persistent_report {
    uint64_t plan_attempts;
    uint64_t plans_built;
    uint64_t commits;
    uint64_t rollbacks;
    uint64_t hits;
    uint64_t misses;
    uint64_t duplicates;
    uint64_t free_assignments;
    uint64_t evictions;
    uint64_t rejects;
    uint64_t budget_rejects;
    uint64_t class_rejects;
    uint64_t protected_rejects;
    uint64_t key_misses;
    uint64_t overflow_rejects;
    uint64_t oracle_runs;
    uint64_t oracle_failures;
    uint64_t arena_allocations;
    uint64_t arena_reuses;
    uint64_t arena_releases;
    uint64_t arena_failures;
    uint64_t arena_oracle_runs;
    uint64_t arena_oracle_failures;
    uint64_t epochs_attempted;
    uint64_t epochs_published;
    uint64_t all_hit_epochs;
    uint64_t miss_epochs;
    uint64_t miss_experts;
    uint64_t weight_bytes_uploaded;
    uint64_t remap_bytes_uploaded;
    uint64_t upload_failures;
    uint64_t fallbacks;
    uint64_t slot_invalidations;
    uint64_t poisons;
    uint64_t persistent_dispatches;
    uint64_t transient_dispatches;
    uint64_t runtime_oracle_runs;
    uint64_t runtime_oracle_failures;
    int enabled;
    int required;
    int stats;
    int oracle;
} ds4_cuda_stream_expert_persistent_report;
typedef struct ds4_cuda_q8_hc_expand_report {
    uint64_t candidates;
    uint64_t fused_attempts;
    uint64_t fused_completed;
    uint64_t split_attempts;
    uint64_t split_completed;
    uint64_t failures;
    uint64_t capture_candidates;
    uint64_t owned_forced_fused;
    uint64_t multi_gpu_forced_fused;
    uint64_t oracle_runs;
    uint64_t oracle_failures;
    int force_fused;
    int split_requested;
    int stats;
} ds4_cuda_q8_hc_expand_report;
/* CUDA-only policy/planner/scatter hooks.  The policy hook takes explicit
 * values so tests do not need to mutate process environment around once_flag
 * initialization. */
int ds4_cuda_test_q8_hc_expand_policy(
        int force_fused, int disable_fused, int n_gpus, int owned,
        int capture, int *fused_out, int *owned_forced_out,
        int *multi_gpu_forced_out, int *capture_out);
int ds4_cuda_test_q8_hc_expand_env_value(const char *value);
int ds4_cuda_test_q8_hc_expand_oracle(void);
void ds4_cuda_q8_hc_expand_get_report(
        ds4_cuda_q8_hc_expand_report *report);
int ds4_cuda_test_stream_selected_batch_policy(
        int enable, int disable, int require, int oracle,
        int *enabled_out, int *required_out, int *oracle_out);
int ds4_cuda_test_stream_selected_batch_env_value(const char *value);
int ds4_cuda_test_stream_selected_batch_plan(void);
int ds4_cuda_test_stream_selected_batch_copy(void);
void ds4_cuda_stream_selected_batch_io_get_report(
        ds4_cuda_stream_selected_batch_io_report *report);
int ds4_cuda_test_stream_selected_event_pipeline_policy(
        int enable, int disable, int require, int oracle,
        int *enabled_out, int *required_out, int *oracle_out);
int ds4_cuda_test_stream_selected_event_env_value(const char *value);
int ds4_cuda_test_stream_selected_event_pipeline(void);
int ds4_cuda_test_stream_selected_owner_device(void);
void ds4_cuda_stream_selected_event_pipeline_get_report(
        ds4_cuda_stream_selected_event_pipeline_report *report);
int ds4_cuda_test_iq2_ssd_grouped_policy(
        int enable, int disable, int require, int stats,
        int *enabled_out, int *required_out, int *stats_out);
int ds4_cuda_test_iq2_ssd_grouped_eligibility(
        int enabled, int ssd_streaming, int single_gpu, int gb10,
        int quality, int owned_filtered, int capture, int mmq,
        uint32_t n_tokens, uint32_t n_expert, int top6_unique,
        int raw_layout, int binding_valid);
int ds4_cuda_test_iq2_ssd_grouped_candidate(
        int iq2_path, int ssd_streaming, int allow_streaming,
        int owned_filtered, uint32_t n_tokens, uint32_t n_expert,
        int top6_unique, int raw_layout, int binding_valid);
int ds4_cuda_test_iq2_ssd_grouped_raw_layout(
        uint32_t gate_type, uint32_t down_type,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim,
        uint32_t out_dim);
int ds4_cuda_test_iq2_ssd_grouped_lease(void);
void ds4_cuda_iq2_ssd_grouped_get_report(
        ds4_cuda_iq2_ssd_grouped_report *report);
int ds4_cuda_test_stream_expert_persistent_policy(
        int enable, int disable, int require, int stats, int oracle,
        int *enabled_out, int *required_out, int *stats_out,
        int *oracle_out);
int ds4_cuda_test_stream_expert_persistent_env_value(const char *value);
int ds4_cuda_test_stream_expert_persistent_planner(void);
int ds4_cuda_test_stream_expert_persistent_arena(void);
int ds4_cuda_test_stream_expert_persistent_runtime(void);
void ds4_cuda_stream_expert_persistent_get_report(
        ds4_cuda_stream_expert_persistent_report *report);
int ds4_gpu_cuda_stream_selected_event_pipeline_enabled(void);
int ds4_gpu_cuda_stream_selected_event_pipeline_required(void);
int ds4_gpu_cuda_stream_selected_set_owner_device(void);
void ds4_gpu_cuda_stream_selected_event_note_candidate(void);
void ds4_gpu_cuda_stream_selected_event_note_fallback(void);
void ds4_gpu_cuda_stream_selected_event_note_failure(int required);
int ds4_gpu_signal_selected_readback_ready_async(uint64_t *event_value);
int ds4_gpu_stream_expert_cache_wait_selected_upload(
        uint64_t event_value, const char *label);
int ds4_gpu_cuda_stream_selected_event_abort(void);
#endif
void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts);
void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes);
uint64_t ds4_gpu_recommended_working_set_size(void);
uint32_t ds4_gpu_stream_expert_cache_configured_count(void);
uint32_t ds4_gpu_stream_expert_cache_current_count(void);
typedef struct ds4_gpu_stream_expert_table {
    const void *model_map;
    uint64_t    model_size;
    uint32_t    layer;
    uint32_t    n_total_expert;
    uint64_t    gate_offset;
    uint64_t    up_offset;
    uint64_t    down_offset;
    uint64_t    gate_expert_bytes;
    uint64_t    down_expert_bytes;
} ds4_gpu_stream_expert_table;
/* Reset only the prompt-local eviction heuristic.  The resident SSD expert
 * cache itself is intentionally kept warm across sessions. */
void ds4_gpu_stream_expert_cache_reset_route_hotness(void);
void ds4_gpu_stream_expert_cache_release_resident(void);
uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes);
int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected);
int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected);
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU)
/* Returns 1 on publication, 0 on a safe pre-enqueue rejection, and -1 on
 * a post-enqueue failure which callers must not retry. */
int ds4_gpu_stream_expert_cache_begin_selected_load_async(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected,
        uint64_t                          *upload_event_value);
#endif
int ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor              *selected,
        uint32_t                           n_selected);
#ifdef __APPLE__
/* The async selected-load worker registers itself so Metal cache paths never
 * wait on command buffers from that thread (they fail the load instead and
 * the caller retries synchronously). */
void ds4_gpu_stream_expert_cache_note_service_thread(void);
#endif
#if defined(DS4_ROCM_BUILD) || (!defined(DS4_NO_GPU) && !defined(__APPLE__))
int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_tokens,
        uint32_t                           n_selected);
#endif
#ifdef DS4_ROCM_BUILD
int ds4_gpu_stream_expert_cache_load_layer(
        const ds4_gpu_stream_expert_table *table);
int ds4_gpu_stream_expert_cache_seed_from_layer_selected(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor             *selected,
        uint32_t                          n_tokens,
        uint32_t                          n_seed_tokens,
        uint32_t                          n_selected);
int ds4_gpu_stream_expert_cache_finish_pending_batch(void);
int ds4_gpu_stream_expert_cache_release_layer_cache(void);
#endif
int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts);
#ifdef __APPLE__
/* Seed from mapped weights with blits appended to the active command buffer. */
int ds4_gpu_stream_expert_cache_seed_experts_gpu_copy(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts);
/* Exact speculative decode may compute up to five independent router rows
 * before executing their routed MoE tails.  begin_collect() first isolates
 * this layer from ordinary decode's pending selected-expert load and global
 * selected-id override.  prepare() then builds one immutable SSD address table
 * for the union of those rows, and set_row() arms exactly one routed-MoE call
 * at a time in increasing row order.  prepare() is a Metal command-stream
 * boundary: it waits for the router rows to become CPU-visible and reopens the
 * command batch.  release() is required on every success/error exit after
 * begin_collect(), and before collecting the next layer. */
int ds4_gpu_stream_expert_exact_rows_begin_collect(void);
int ds4_gpu_stream_expert_exact_rows_prepare(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor              *selected_rows,
        uint32_t                           n_rows,
        uint32_t                           n_selected);
int ds4_gpu_stream_expert_exact_rows_set_row(uint32_t row);
/* Commit the completed routed-tail command buffer without waiting.  The
 * backend retains every private address/overflow/cache buffer owned by the
 * exact-row scope until that command buffer completes, so release() may be
 * called immediately after a successful return.  This is an experimental
 * boundary used only when the caller explicitly enables asynchronous tails. */
int ds4_gpu_stream_expert_exact_rows_end_async(void);
void ds4_gpu_stream_expert_exact_rows_release(void);
#endif
void ds4_gpu_print_memory_report(const char *label);

/* Tensor-parallel per-layer gates (Metal only).  The encoder calls
 * ds4_gpu_tp_gate_encode() right after the kernels that produce a partial
 * block output in the TP slab: it closes the current encoder, makes the GPU
 * signal a shared event, queues the exchange on a service thread, and makes
 * the GPU wait for the CPU-signaled release before the combine kernel runs.
 * Sequence values are assigned internally and increase monotonically; both
 * ranks encode the identical gate sequence so values pair up by
 * construction.  The exchange callback runs on the service thread and must
 * return nonzero on success. */
typedef int (*ds4_gpu_tp_exchange_fn)(void *ud, uint32_t layer, uint32_t gate, uint64_t seq);
/* Bind one rank of the two-way split. slab is the transport slab tensor and
 * gpu_flags_off is the offset of its GPU-written gate-ready flag words. */
int ds4_gpu_tp_init(uint32_t rank,
                    ds4_gpu_tensor *slab, uint64_t gpu_flags_off,
                    uint64_t out_off, uint64_t vec_bytes,
                    ds4_gpu_tp_exchange_fn fn, void *ud);
void ds4_gpu_tp_shutdown(void);
/* Multi-session TP reuses slab slots across several encoded graph tapes.
 * Shared-event arrival is required in that mode to make each partial vector
 * CPU-visible before the transport thread reads it. */
void ds4_gpu_tp_set_session_batch_mode(int enabled);
/* Single-session flag gates use one exact arrival word per layer/gate, so
 * decode command buffers may be submitted in layer order without a later
 * monotonic event signal satisfying an earlier arrival. */
int ds4_gpu_tp_decode_split_flush_safe(void);
/* Weight ranges to pull into the GPU cache while the given gate (0 attention,
 * 1 FFN) waits for the peer: consumed by the next poll gate of that kind. */
int ds4_gpu_tp_gate_prefetch_plan(uint32_t gate,
                                  const void *model_map, uint64_t model_size,
                                  const uint64_t *offsets, const uint64_t *bytes,
                                  uint32_t count);
/* The coordinator-only DSpark support model does not participate in TP.
 * Suspend ownership only while encoding it; base-model verification remains
 * split across both ranks. */
void ds4_gpu_tp_suspend_expert_sharding(int suspend);
int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate);
/* Verify-block batch gates: one exchange per layer moving `rows` partial
 * rows at once (speculative verify).  The callback runs on the gate service
 * thread with the same ud as the row-gate exchange fn. */
typedef int (*ds4_gpu_tp_batch_exchange_fn)(void *ud, uint32_t layer,
                                            uint32_t rows, uint64_t seq);
void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn);
int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows);
/* Prefill batch gates: the service thread exchanges `bytes` between two
 * CPU-visible bounce tensors directly (payloads far beyond slab slots). */
typedef int (*ds4_gpu_tp_big_exchange_fn)(void *ud, uint32_t layer,
                                          uint64_t seq, const void *out,
                                          void *in, uint64_t bytes);
void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn);
int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                               const ds4_gpu_tensor *out_t,
                               ds4_gpu_tensor *in_t,
                               uint64_t bytes);
/* Pause/resume the DVFS keep-alive around work that keeps the GPU busy.
 * No-op when TP is not bound. */
void ds4_gpu_tp_keepalive_pause(int paused);
/* Split attention heads across the two TP ranks in the GLM batch-prefill
 * attention kernels (qk-low, attention-lora, value-project). The caller
 * zeroes the unowned head range of the heads buffer and combines the
 * attn-output partials over the TP big-gate exchange. */
void ds4_gpu_tp_set_attn_head_split(int enabled);
/* Skip the whole-file model residency set (TP sharding: only the
 * owned ranges are warmed; the rest must never be paged in). Call before
 * the model is mapped. */
void ds4_gpu_model_residency_skip(int skip);
/* Submit one trivial command buffer (first-submission costs paid at load). */
int ds4_gpu_warm_command_queue(void);
/* Nonzero after any gate exchange failed; the eval must abort. */
int ds4_gpu_tp_failed(void);

/* Tensor-parallel sliced projections (Metal decode path only).
 *
 * ds4_gpu_matmul_q8_0_kslice_tensor computes a k-range partial matvec:
 * out[out_dim] = W[:, k_off : k_off + k_cnt] @ x[x_elem_off : +k_cnt] where
 * W rows span full_in_dim quantized elements. Q8_0 slices use multiples of 32;
 * the generic dispatch also accepts Q4_K slices in multiples of 256. Partial
 * results from both ranks sum to the full projection.
 *
 * ds4_gpu_attention_output_q8_tp_tensor is the group-sliced attention output
 * pair: low projection for groups [group0, group0+group_cnt) plus the
 * matching k-slice of the expand projection, producing this rank's partial
 * attention block output (n_tokens == 1 only). */
int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                full_in_dim,
        uint64_t                k_off,
        uint64_t                k_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                x_elem_off);
/* CUDA multi-row variant. Each input row contains only the owned contiguous
 * K slice, while each output row spans the full projection width. */
int ds4_gpu_matmul_q8_0_kslice_rows_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              full_in_dim,
        uint64_t              out_dim,
        uint64_t              k_off,
        uint64_t              k_cnt,
        const ds4_gpu_tensor *x,
        uint64_t              n_rows);
int ds4_gpu_matmul_quant_kslice_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                full_in_dim,
        uint64_t                k_off,
        uint64_t                k_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                x_elem_off);
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

int ds4_gpu_embed_token_q8_0_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd);

int ds4_gpu_embed_tokens_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd);

int ds4_gpu_embed_token_quant_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          weight_type,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd);

int ds4_gpu_embed_tokens_quant_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd);

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
        float                   scale);

int ds4_gpu_dspark_markov_argmax_tensor(ds4_gpu_tensor *out_idx,
                                        const ds4_gpu_tensor *logits_row,
                                        const void *model_map,
                                        uint64_t model_size,
                                        uint64_t w1_offset,
                                        uint64_t w2_offset,
                                        uint32_t prev_token,
                                        uint32_t vocab,
                                        uint32_t rank);

/* Optional GPU-resident DSpark proposer tail.  The backend keeps the Markov
 * token chain on-device across all draft rows, stops it at the first rejected
 * confidence row, and returns proposals plus the evaluated confidence logits
 * in one result/readback.  Callers must re-evaluate the returned prefix with
 * the established CPU sigmoid policy and fall back if the device stopped a
 * row that policy accepts.
 *
 * This acceleration hook is deliberately fail-closed: it returns zero for
 * disabled or unsupported inputs and callers must use the established
 * per-row path.  CUDA requires DS4_CUDA_DSPARK_DEVICE_PROPOSER=1; Metal uses
 * DS4_METAL_DSPARK_DEVICE_PROPOSER=1.  The corresponding NO_DEVICE_PROPOSER
 * variables are unconditional kill switches.  The first implementation
 * supports Q8_0 w1, w2, and confidence weights with 32-aligned hidden/rank
 * dimensions.  The caller-owned result tensor must provide
 * DS4_GPU_DSPARK_DEVICE_PROPOSAL_BYTES; the first sizeof(result) bytes are the
 * public payload and the remainder is private per-call backend state. */
#define DS4_GPU_DSPARK_MAX_DRAFTS 6u
#define DS4_GPU_DSPARK_DEVICE_PROPOSAL_BYTES 2048u
typedef struct {
    int32_t  tokens[DS4_GPU_DSPARK_MAX_DRAFTS];
    float    confidence_logits[DS4_GPU_DSPARK_MAX_DRAFTS];
    uint32_t proposal_len;
    uint32_t confidence_len;
    uint32_t status;       /* 1: complete; 0: device-side failure */
    uint32_t reserved;
} ds4_gpu_dspark_device_proposal;

int ds4_gpu_dspark_markov_confidence_q8_tensor(
        ds4_gpu_tensor       *out_result,
        const ds4_gpu_tensor *logits_rows,
        const ds4_gpu_tensor *hidden_rows,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              w1_offset,
        uint64_t              w2_offset,
        uint64_t              confidence_offset,
        uint32_t              first_prev_token,
        uint32_t              vocab,
        uint32_t              rank,
        uint32_t              hidden_dim,
        uint32_t              n_drafts,
        float                 confidence_threshold,
        int                   reuse_confidence0,
        float                 confidence0);
int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

int ds4_gpu_indexer_top1_value_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *values,
        const ds4_gpu_tensor *scores,
        uint32_t              n_comp,
        uint32_t              n_tokens,
        uint32_t              index_offset);

int ds4_gpu_matmul_q8_0_top1_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *values,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              index_offset);

int ds4_gpu_set_decode_fast_attention(int enabled);
int ds4_gpu_set_decode_score_vec4(int enabled);

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

int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_decode_mpp_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_quant_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_quant_rows_scalar_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* Optional fused GPU operations.
 *
 * These are acceleration hooks, not required backend primitives.  A backend
 * that does not provide the fused kernel must still define the symbol and
 * return 0.  Callers then use the portable sequence of required primitives.
 * Backends that return nonzero from a fused half-output operation must also
 * implement the matching half-input HC expansion helpers below.
 */
int ds4_gpu_matmul_q8_0_pair_tensor(
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

int ds4_gpu_matmul_q4_K_pair_decode_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight0_offset,
        uint64_t              weight1_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x);

/* Optional dense Q4_K pair for decode, microbatch, and prefill. Backends
 * return 1 when the pair was encoded, 0 to request separate fallback
 * matmuls from the graph, and -1 after a required/attempted-path error. */
int ds4_gpu_matmul_q4_K_pair_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight0_offset,
        uint64_t              weight1_offset,
        uint64_t              in_dim,
        uint64_t              out0_dim,
        uint64_t              out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t              n_tok);

/* Metal decode compound for AProjQ4: Q-A/KV Q4_K pair plus the attention
 * and indexer F16 compressor pairs/state stores. Returns 1 when encoded,
 * 0 to use the separate fallback, and -1 on an attempted-path error. */
int ds4_gpu_q4_K_pair_quad_compressor_store_tensor(
        ds4_gpu_tensor       *qr,
        ds4_gpu_tensor       *kv_raw,
        ds4_gpu_tensor       *out0_kv,
        ds4_gpu_tensor       *out0_score,
        ds4_gpu_tensor       *out1_kv,
        ds4_gpu_tensor       *out1_score,
        ds4_gpu_tensor       *state0_kv,
        ds4_gpu_tensor       *state0_score,
        ds4_gpu_tensor       *state1_kv,
        ds4_gpu_tensor       *state1_score,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_a_offset,
        uint64_t              kv_offset,
        uint64_t              weight0_kv_offset,
        uint64_t              weight0_score_offset,
        uint64_t              weight1_kv_offset,
        uint64_t              weight1_score_offset,
        uint64_t              ape0_offset,
        uint32_t              ape0_type,
        uint64_t              ape1_offset,
        uint32_t              ape1_type,
        uint32_t              in_dim,
        uint32_t              q_rank,
        uint32_t              kv_dim,
        uint32_t              width0,
        uint32_t              width1,
        const ds4_gpu_tensor *x,
        uint32_t              ratio,
        uint32_t              pos);

/* Multi-row decode projections that preserve the one-row reduction order. */
int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);
int ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight0_offset,
        uint64_t              weight1_offset,
        uint64_t              in_dim,
        uint64_t              out0_dim,
        uint64_t              out1_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);

int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor       *out_h,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
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

int ds4_gpu_router_shared_gate_up_q8_0_tensor(
        ds4_gpu_tensor       *router_logits,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              router_weight_offset,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              in_dim,
        uint64_t              router_out_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        float                 clamp,
        bool                  router_only);
#ifdef __APPLE__
int ds4_gpu_router_project_select_fused_tensor(
        ds4_gpu_tensor       *router_logits,
        ds4_gpu_tensor       *probs,
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              router_weight_offset,
        uint64_t              bias_offset,
        bool                  has_bias,
        const ds4_gpu_tensor *x);
#endif
int ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *prequant,
        uint32_t                expert_split,
        bool                    home_rank);

int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
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

int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
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
        uint64_t                n_tok,
        float                   clamp);

int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
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
        uint64_t                n_tok,
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

/* CUDA batch path: fold an input RMS normalization into the FP16 activation
 * conversion used by the following projection. Returns 0 without touching
 * out when the optimized path is unavailable. */
int ds4_gpu_matmul_f16_rms_fold_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   norm_eps);

/* Exact multi-row form of the DeepSeek 4096x256 F16 router projection. */
int ds4_gpu_matmul_f16_router_rows_exact_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        const ds4_gpu_tensor *x,
        uint32_t                n_rows);

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

/* Optional Metal decode fusion. Returns 1 when the paired projection and
 * recurrent compressor-state store were encoded, 0 when the optimized path
 * is unavailable, and -1 on an attempted-path error. */
int ds4_gpu_matmul_f16_pair_compressor_store_tensor(
        ds4_gpu_tensor       *out_kv,
        ds4_gpu_tensor       *out_score,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_kv_offset,
        uint64_t                weight_score_offset,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                in_dim,
        uint32_t                width,
        const ds4_gpu_tensor *x,
        uint32_t                ratio,
        uint32_t                pos);

/* Optional Metal ratio-4 decode fusion.  Returns 1 when both paired
 * compressor projections and their recurrent-state stores were encoded in a
 * single dispatch, 0 when the caller should keep the established separate
 * paths, and -1 on an attempted-path error. */
int ds4_gpu_f16_quad_compressor_store_auto_available(void);
int ds4_gpu_matmul_f16_quad_compressor_store_tensor(
        ds4_gpu_tensor       *out0_kv,
        ds4_gpu_tensor       *out0_score,
        ds4_gpu_tensor       *out1_kv,
        ds4_gpu_tensor       *out1_score,
        ds4_gpu_tensor       *state0_kv,
        ds4_gpu_tensor       *state0_score,
        ds4_gpu_tensor       *state1_kv,
        ds4_gpu_tensor       *state1_score,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight0_kv_offset,
        uint64_t              weight0_score_offset,
        uint64_t              weight1_kv_offset,
        uint64_t              weight1_score_offset,
        uint64_t              ape0_offset,
        uint32_t              ape0_type,
        uint64_t              ape1_offset,
        uint32_t              ape1_type,
        uint64_t              in_dim,
        uint32_t              width0,
        uint32_t              width1,
        const ds4_gpu_tensor *x,
        uint32_t              ratio,
        uint32_t              pos);

/* Decode-only M5 fusion: emit-path compressor row finalize (norm + rope +
 * fp8/commit + indexer qat) in one dispatch.  Bit-exact vs the separate
 * dispatches.  Returns 1 when fused, 0 to fall back. */
int ds4_gpu_dsv4_comp_row_finalize_tensor(
        ds4_gpu_tensor       *attn_stage,
        ds4_gpu_tensor       *attn_cache,
        uint32_t              attn_comp_row,
        uint64_t              attn_norm_offset,
        ds4_gpu_tensor       *index_cache,
        uint32_t              index_comp_row,
        uint64_t              index_norm_offset,
        ds4_gpu_tensor       *attn_state_kv,
        ds4_gpu_tensor       *attn_state_score,
        ds4_gpu_tensor       *index_state_kv,
        ds4_gpu_tensor       *index_state_score,
        const void           *model_map,
        uint64_t              model_size,
        uint32_t              pos,
        uint32_t              n_rot,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 rms_eps);

/* Decode-only M5 fusion: q_a/kv Q8 pair projection + F16 quad compressor
 * projection/store in one dispatch.  Bit-exact vs the separate dispatches.
 * Returns 1 when fused, 0 to fall back, -1 on error. */
int ds4_gpu_qkv_pair_quad_compressor_store_tensor(
        ds4_gpu_tensor       *qr,
        ds4_gpu_tensor       *kv_raw,
        ds4_gpu_tensor       *out0_kv,
        ds4_gpu_tensor       *out0_score,
        ds4_gpu_tensor       *out1_kv,
        ds4_gpu_tensor       *out1_score,
        ds4_gpu_tensor       *state0_kv,
        ds4_gpu_tensor       *state0_score,
        ds4_gpu_tensor       *state1_kv,
        ds4_gpu_tensor       *state1_score,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_a_offset,
        uint64_t              kv_offset,
        uint64_t              weight0_kv_offset,
        uint64_t              weight0_score_offset,
        uint64_t              weight1_kv_offset,
        uint64_t              weight1_score_offset,
        uint64_t              ape0_offset,
        uint32_t              ape0_type,
        uint64_t              ape1_offset,
        uint32_t              ape1_type,
        uint32_t              in_dim,
        uint32_t              q_rank,
        uint32_t              kv_dim,
        uint32_t              width0,
        uint32_t              width1,
        const ds4_gpu_tensor *x,
        uint32_t              ratio,
        uint32_t              pos);

int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_repeat_hc_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_repeat_hc_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *rows,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

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

int ds4_gpu_add_rms_norm_weight_tensor(
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
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

int ds4_gpu_dsv4_qkv_rms_norm_kv_rope_fp8_store_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t              kv_weight_offset,
        uint32_t              kv_n,
        ds4_gpu_tensor       *raw_cache,
        uint64_t              raw_cap,
        uint32_t              raw_row,
        uint32_t              n_rot,
        uint32_t              pos0,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 eps);

int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
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
        uint32_t                kv_n_head,
        uint32_t                kv_head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        bool                    inverse,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   eps);

int ds4_gpu_head_rms_norm_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps);

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
        float             eps);

/* Returns 1 when the backend-specific fused projection and its consumers were
 * accepted/encoded successfully, 0 only while the caller may safely use its
 * generic projection path, and -1 when a required specialization could not be
 * honored or work failed after an output writer was encoded (so replay is
 * unsafe). Completion is established by the enclosing backend synchronization.
 * q_half is optional: backends whose graph owns F16 projection staging may use
 * it, while CUDA/ROCm can emit F32 directly into out. */
int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *q_half,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tok,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              n_rot,
        uint32_t              pos0,
        uint32_t              n_ctx_orig,
        bool                  inverse,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 eps);

int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot);

int ds4_gpu_dsv4_indexer_qat_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_rows,
        uint32_t          head_dim);



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
        float             beta_slow);

int ds4_gpu_glm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow);

int ds4_gpu_glm_kv_lora_rms_norm_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_raw,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        float                 eps);

int ds4_gpu_glm_k_b_project_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_norm,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              n_head);

int ds4_gpu_glm_k_b_project_typed_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *kv_norm,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              n_head);

int ds4_gpu_glm_store_compact_kv_tensor(
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_norm,
        const ds4_gpu_tensor *kv_raw,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16);

int ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_weight_offset,
        uint32_t              q_n,
        ds4_gpu_tensor       *kv_lora_cache,
        ds4_gpu_tensor       *k_rope_cache,
        const ds4_gpu_tensor *kv_raw,
        uint64_t              kv_weight_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_rope,
        bool                  cache_f16,
        float                 eps);

int ds4_gpu_glm_store_indexer_k_tensor(
        ds4_gpu_tensor       *indexer_key_cache,
        const ds4_gpu_tensor *raw_k,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              bias_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              rot_dim,
        uint32_t              n_ctx_orig,
        float                 eps,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16);

/* GLM-5.3 pools four normalized indexer keys with a learned, per-channel
 * softmax. Partial pools are retained in tail_k/tail_gate across calls. */
int ds4_gpu_glm53_indexer_pool_update_tensor(
        ds4_gpu_tensor       *pool_cache,
        ds4_gpu_tensor       *tail_k,
        ds4_gpu_tensor       *tail_gate,
        const ds4_gpu_tensor *raw_k,
        const ds4_gpu_tensor *gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              norm_weight_offset,
        uint64_t              norm_bias_offset,
        uint64_t              ape_offset,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              head_dim,
        uint32_t              pool_size,
        float                 eps,
        bool                  cache_f16);

int ds4_gpu_glm53_expand_pool_selection_tensor(
        ds4_gpu_tensor       *raw_selected,
        const ds4_gpu_tensor *pool_selected,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              selected_pools,
        uint32_t              index_topk,
        uint32_t              pool_size,
        uint32_t              output_width);

int ds4_gpu_glm_build_kv_cache_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16);

int ds4_gpu_glm_build_kv_cache_flash_tensor(
        ds4_gpu_tensor       *key_cache,
        ds4_gpu_tensor       *value_cache,
        const ds4_gpu_tensor *kv_raw,
        const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              kv_raw_dim,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  cache_f16);

int ds4_gpu_glm_attention_full_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16);

int ds4_gpu_glm_fill_selected_range_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_selected);

int ds4_gpu_glm_fill_selected_range_batch_tensor(
        ds4_gpu_tensor *selected,
        uint32_t        n_tokens,
        uint32_t        pos0,
        uint32_t        n_selected,
        uint32_t        pad_row);

int ds4_gpu_glm_indexer_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t        n_tokens,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        rot_dim,
        uint32_t        pos0,
        uint32_t        n_ctx_orig,
        float           freq_base,
        float           freq_scale,
        float           ext_factor,
        float           attn_factor,
        float           beta_fast,
        float           beta_slow);

int ds4_gpu_glm_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16);

int ds4_gpu_glm_indexer_scores_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16);

int ds4_gpu_glm53_indexer_scores_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache,
        uint32_t              n_rows,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              pool_size,
        uint32_t              n_head,
        uint32_t              head_dim,
        float                 scale,
        bool                  cache_f16);

int ds4_gpu_glm_qk_lowrank_q8_0_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim);

int ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim);

int ds4_gpu_glm_qk_lowrank_typed_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim);

int ds4_gpu_glm_qk_lowrank_typed_batch_tensor(
        ds4_gpu_tensor       *qk_low,
        const ds4_gpu_tensor *q,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_dim);

int ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *lora,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              value_dim);

int ds4_gpu_glm_value_project_typed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *lora,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              weight_type,
        uint32_t              n_tokens,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              value_dim);

int ds4_gpu_glm_attention_indexed_decode_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_rope_tail_decode_rows_tensor(
        ds4_gpu_tensor                     *x,
        const ds4_gpu_attention_decode_row *rows,
        uint32_t                            n_rows,
        uint32_t                            n_head,
        uint32_t                            head_dim,
        uint32_t                            n_rot,
        uint32_t                            n_ctx_orig,
        bool                                inverse,
        float                               freq_base,
        float                               freq_scale,
        float                               ext_factor,
        float                               attn_factor,
        float                               beta_fast,
        float                               beta_slow);

int ds4_gpu_glm_attention_indexed_decode_typed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_decode_split_group8_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *partial_lora,
        ds4_gpu_tensor       *partial_ms,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        bool                  selected_rows_valid,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        uint32_t              block_rows,
        uint32_t              n_blocks,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor(
        ds4_gpu_tensor       *heads,
        ds4_gpu_tensor       *partial_lora,
        ds4_gpu_tensor       *partial_ms,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_selected,
        bool                  selected_rows_valid,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        uint32_t              block_rows,
        uint32_t              n_blocks,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_batch_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_batch_typed_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              value_weight_offset,
        uint32_t              value_weight_type,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              value_dim,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_sort_i32_rows_asc_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *src,
        uint32_t              row_width,
        uint32_t              n_rows);

int ds4_gpu_glm_attention_indexed_batch_lora_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        uint32_t              n_tokens,
        uint32_t              pos0,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

/* Dense causal MLA over the shared compact latent cache. qk_low and lora_out
 * are [token, head, kv_lora_dim]; the F16 cache is shared by all heads. */
int ds4_gpu_glm_attention_dense_compact_lora_causal_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        uint32_t              q_row0,
        uint32_t              n_q,
        uint32_t              n_kv,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_dim);

int ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
        ds4_gpu_tensor       *lora_out,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected,
        uint32_t              n_tokens,
        uint32_t              n_selected,
        uint32_t              cache_cap,
        bool                  cache_f16,
        uint32_t              n_head,
        uint32_t              kv_lora_dim,
        uint32_t              qk_nope,
        uint32_t              qk_rope,
        uint32_t              n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow);

int ds4_gpu_glm_attention_flash_staged_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16);

int ds4_gpu_glm_attention_flash_tensor(
        ds4_gpu_tensor       *heads,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              pos0,
        uint32_t              n_tokens,
        uint32_t              cache_len,
        uint32_t              cache_cap,
        uint32_t              n_head,
        uint32_t              qk_dim,
        uint32_t              value_dim,
        bool                  cache_f16);

/* Release decode fused KV finalizer: after the standalone RoPE kernel, this
 * performs DS4's FP8 non-RoPE KV round trip and writes the F16-rounded raw
 * attention cache row in one dispatch. */
int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot);

/* Exact multi-session form of the decode KV finalizer. KV rows are
 * contiguous, while each output row is written to its session-private cache. */
int ds4_gpu_kv_fp8_store_raw_decode_rows_tensor(
        ds4_gpu_tensor        *kv,
        ds4_gpu_tensor *const *raw_caches,
        const uint32_t        *raw_caps,
        const uint32_t        *raw_rows,
        uint32_t               n_rows,
        uint32_t               head_dim,
        uint32_t               n_rot);

/* Reference/raw-cache primitive kept for prefill and diagnostics.  Decode uses
 * ds4_gpu_kv_fp8_store_raw_tensor unless a diagnostic reference path is
 * explicitly selected by the graph driver. */
int ds4_gpu_store_raw_kv_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim);

/* Store with float(half(x)) rounding. If the batch exceeds raw_cap, retain
 * only its final raw_cap rows at (pos0 + source_row) % raw_cap. */
int ds4_gpu_store_raw_kv_batch_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim);

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
        float                   rms_eps,
        bool                    state_already_stored,
        bool                    decode_one_token,
        bool                    defer_finalize);

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
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_decode_heads_rope_tensor(
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
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                n_rot,
        uint32_t                pos0,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        int                    *fused_inv_rope);

/* Multi-session decode over contiguous Q/head rows and private KV caches.
 * The row table is copied into CUDA launch parameters, so no device-side
 * descriptor upload or synchronization is required. */
int ds4_gpu_attention_decode_rows_rope_tensor(
        ds4_gpu_tensor                       *heads,
        const void                           *model_map,
        uint64_t                              model_size,
        uint64_t                              sinks_offset,
        const ds4_gpu_tensor                 *q,
        const ds4_gpu_attention_decode_row   *rows,
        uint32_t                              n_rows,
        uint32_t                              n_head,
        uint32_t                              head_dim,
        uint32_t                              n_rot,
        uint32_t                              n_ctx_orig,
        float                                 freq_base,
        float                                 freq_scale,
        float                                 ext_factor,
        float                                 attn_factor,
        float                                 beta_fast,
        float                                 beta_slow);
/* Diagnostic/public form of the dk=512 gathered decode-attention KV staging
 * step. The compressed source must be F16; dst writes chronological raw-ring
 * rows followed by compressed rows and must not overlap either source. */
int ds4_gpu_flash_kv_stage_f16_tensor(
        ds4_gpu_tensor       *dst,
        const ds4_gpu_tensor *raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_raw,
        const ds4_gpu_tensor *comp,
        uint32_t                comp_is_f16,
        uint32_t                n_comp,
        uint32_t                head_dim);

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
        uint32_t                head_dim);

/* Rectangular raw prefill attention: q is a view of the n_q query rows at
 * token positions [q_row0, q_row0 + n_q) of the chunk, raw_kv keeps all
 * n_kv rows, heads receives n_q output rows.  Used by the TP prefill row
 * split; the square entry above is the q_row0 = 0, n_q = n_kv case. */
int ds4_gpu_attention_prefill_raw_heads_range_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                q_row0,
        uint32_t                n_q,
        uint32_t                n_kv,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

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
        uint32_t                head_dim);

int ds4_gpu_attention_noncausal_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
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
        uint32_t                head_dim);

int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
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
        uint32_t                head_dim);

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

/* Rectangular static-mixed prefill attention: q is a view of the n_q query
 * rows at token positions [q_row0, q_row0 + n_q) of the chunk, while raw_kv
 * keeps all n_tokens rows and comp_kv all n_comp compressed keys.  Used by
 * the TP prefill row split; the square entry above is q_row0 = 0,
 * n_q = n_tokens. */
int ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                q_row0,
        uint32_t                n_q,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

/* DeepSeek Vision-Exp attention over the current prefill chunk. The raw cache
 * is chronological from raw_start and may include the preceding SWA rows.
 * Synthetic image spans in tokens are made bidirectional as specified by the
 * checkpoint; text and compressed keys retain the normal causal masks. */
int ds4_gpu_attention_visual_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        const int32_t          *tokens,
        uint32_t                vocab_size,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
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
#ifdef __APPLE__
/* Optional resident Metal output-B + HC4 epilogues.  Return 1 when fused work
 * was encoded, 0 before writing anything when ineligible, and -1 after an
 * attempted-path failure (the caller must not replay the fallback then). */
int ds4_gpu_attention_output_q8_batch_hc_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              out_a_offset,
        uint64_t              out_b_offset,
        uint64_t              group_dim,
        uint64_t              rank,
        uint32_t              n_groups,
        uint64_t              out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t              n_tokens,
        uint32_t              n_hc);
#endif
/* Returns 1 when the batch path ran, 0 for the ordinary row fallback, and -1
 * for a post-enqueue failure or a backend REQUIRE diagnostic.  The caller
 * must not retry the row fallback after -1. */
int ds4_gpu_attention_output_q4_K_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint32_t                out_b_type,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);
#ifdef __APPLE__
int ds4_gpu_attention_output_q4_K_batch_hc_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              out_a_offset,
        uint64_t              out_b_offset,
        uint32_t              out_b_type,
        uint64_t              group_dim,
        uint64_t              rank,
        uint32_t              n_groups,
        uint64_t              out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t              n_tokens,
        uint32_t              n_hc);
#endif

int ds4_gpu_attention_output_q8_batch_f16_tensor(
        ds4_gpu_tensor       *out_h,
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

int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads);
/* Q4_K grouped low projection: positive is success, zero is a clean
 * pre-enqueue fallback, and negative is a required/possibly-enqueued failure. */
int ds4_gpu_attention_output_low_q4_K_slice_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads,
        int                    resident_decode);

int ds4_gpu_attention_output_low_q8_rows_exact_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        const ds4_gpu_tensor *heads,
        uint32_t                n_rows);

int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups_total,
        uint32_t                group0,
        uint32_t                group_cnt,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads);

/* =========================================================================
 * Router, Shared Expert, and Routed MoE.
 * =========================================================================
 *
 * These kernels implement the FFN body: router probabilities/top-k or hash
 * routing, shared SwiGLU, and the IQ2_XXS/Q2_K/Q4_K routed experts.
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

int ds4_gpu_add3_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *c,
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

/* DeepSeek Vision-Exp prefill may mix ordinary vocabulary IDs and synthetic
 * image IDs in one batch. Text rows keep the normal/hash route; image rows use
 * the checkpoint's visual selection bias. Routing weights always come from
 * the original, unbiased scores. */
int ds4_gpu_router_select_batch_visual_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        bool                    has_bias,
        bool                    hash_mode,
        const void             *vision_map,
        uint64_t                vision_size,
        uint64_t                visual_bias_offset,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens,
        uint32_t                vocab_size,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);

int ds4_gpu_glm_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale);

int ds4_gpu_glm_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);

int ds4_gpu_glm_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   swiglu_clamp,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        bool                    force_resident);

int ds4_gpu_glm_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   swiglu_clamp,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride,
        bool                    force_resident);

int ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   swiglu_clamp,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride);

int ds4_gpu_routed_moe_set_selected_override(const int32_t *selected, uint32_t n_selected);
void ds4_gpu_set_glm_mtp_verify_mode(bool enabled);
#ifdef DS4_ROCM_BUILD
int ds4_gpu_dspark_gfx1151_fast_path(void);
void ds4_gpu_set_dspark_verify_mode(bool enabled);
#endif

int ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        uint64_t              in_start,
        uint64_t              in_count,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t              n_embd,
        uint32_t              n_hc);

int ds4_gpu_routed_moe_one_owned_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              gate_type,
        uint32_t              down_type,
        uint64_t              gate_expert_bytes,
        uint64_t              gate_row_bytes,
        uint64_t              down_expert_bytes,
        uint64_t              down_row_bytes,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t              n_total_expert,
        uint32_t              n_expert,
        uint32_t              resident_expert_base,
        uint32_t              resident_expert_count,
        float                 clamp,
        const ds4_gpu_tensor *x,
        ds4_gpu_tensor       *down_output,
        bool                  pack_fixed3,
        ds4_gpu_tensor       *shared_prequant);

int ds4_gpu_routed_moe_batch_owned_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              gate_type,
        uint32_t              down_type,
        uint64_t              gate_expert_bytes,
        uint64_t              gate_row_bytes,
        uint64_t              down_expert_bytes,
        uint64_t              down_row_bytes,
        uint32_t              expert_in_dim,
        uint32_t              expert_mid_dim,
        uint32_t              out_dim,
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        uint32_t              n_total_expert,
        uint32_t              n_expert,
        uint32_t              resident_expert_base,
        uint32_t              resident_expert_count,
        float                 clamp,
        const ds4_gpu_tensor *x,
        uint32_t              layer_index,
        uint32_t              n_tokens,
        bool                 *mid_is_f16);

int ds4_gpu_routed_moe_owned_slots_combine_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_slots,
        const ds4_gpu_tensor *selected,
        uint32_t              out_dim,
        uint32_t              expert_split);

int ds4_gpu_routed_moe_owned_slots_combine_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_slots,
        const ds4_gpu_tensor *selected,
        uint32_t              out_dim,
        uint32_t              expert_split,
        uint32_t              rows);

int ds4_gpu_routed_moe_owned_packed_combine_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_packed,
        const ds4_gpu_tensor *selected,
        uint32_t              out_dim,
        uint32_t              expert_split);

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
        const ds4_gpu_tensor *add_in,
        uint32_t                layer_index,
        bool                    force_resident);

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
        bool                   *mid_is_f16,
        bool                    force_resident);

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

#ifdef __APPLE__
/* Metal DSpark prefill capture: materialize every reduced HC row and mirror
 * the final row in the same compute dispatch. `out` and `last_out` must refer
 * to non-overlapping storage; production uses separate persistent tensors. */
int ds4_gpu_hc_weighted_sum_capture_last_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *last_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc);
#endif

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

/* Exact one-row HC decode fusion: unweighted RMSNorm followed by the narrow
 * F16 HC-mix projection. The Metal and CUDA implementations are specialized
 * for the 16384 -> 24 DS4 Flash shape and preserve their standalone reduction
 * trees. */
int ds4_gpu_hc_rms_norm_mix_f16_available(void);
int ds4_gpu_hc_rms_norm_mix_f16_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n,
        uint32_t              out_dim,
        float                 eps);

#ifdef __APPLE__
/* Exact one-row continuation of HC RMSNorm+mix: split/Sinkhorn, HC collapse,
 * and the following weighted RMSNorm are encoded in the producer dispatch. */
int ds4_gpu_hc_rms_norm_mix_split_norm_f16_available(void);
int ds4_gpu_hc_rms_norm_mix_split_norm_f16_tensor(
        ds4_gpu_tensor       *mix,
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *residual_hc,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              mix_weight_offset,
        uint64_t              scale_offset,
        uint64_t              base_offset,
        uint64_t              norm_weight_offset,
        uint32_t              n,
        uint32_t              mix_dim,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              sinkhorn_iters,
        float                 eps,
        float                 hc_eps,
        float                 norm_eps);
int ds4_gpu_hc_expand_add_rms_norm_mix_split_norm_f16_tensor(
        ds4_gpu_tensor       *mix,
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_prev,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              mix_weight_offset,
        uint64_t              scale_offset,
        uint64_t              base_offset,
        uint64_t              norm_weight_offset,
        uint32_t              n,
        uint32_t              mix_dim,
        uint32_t              n_embd,
        uint32_t              n_hc,
        uint32_t              sinkhorn_iters,
        float                 eps,
        float                 hc_eps,
        float                 norm_eps);

#endif

/* Batched HC RMSNorm followed by its narrow F16 mixer projection. On the
 * tuned Metal path, scale_scratch stores one float per row instead of the
 * full normalized HC tensor; other shapes retain the established fallback. */
int ds4_gpu_hc_rms_scale_project_f16_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *scale_scratch,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                in_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *x,
        uint32_t                n_rows,
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
int ds4_gpu_hc_expand_add_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
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

int ds4_gpu_hc_expand_split_half_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out_h,
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

int ds4_gpu_hc_expand_add_split_half_add_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add_h,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
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

int ds4_gpu_shared_down_hc_expand_add_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *routed_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *home_slots,
        const ds4_gpu_tensor *peer_packed,
        const ds4_gpu_tensor *selected,
        uint32_t                expert_split,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_matmul_q8_0_hc_expand_tensor(
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

int ds4_gpu_glm53_embedding_bf16(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        const ds4_gpu_tensor *token_ids,
        uint32_t              n_tokens,
        uint32_t              n_embd,
        uint32_t              n_vocab);

int ds4_gpu_glm53_matmul_bf16(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_rows);

int ds4_gpu_glm53_matmul_bf16_qkv(
        ds4_gpu_tensor       *out_q,
        ds4_gpu_tensor       *out_k,
        ds4_gpu_tensor       *out_v,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_q_offset,
        uint64_t              weight_k_offset,
        uint64_t              weight_v_offset,
        uint32_t              in_dim,
        uint32_t              out_dim,
        const ds4_gpu_tensor *x);

#ifndef DS4_GLM53_VISION_TYPES_DEFINED
#define DS4_GLM53_VISION_TYPES_DEFINED
#define DS4_GLM53_VISION_LAYERS 24u

typedef struct {
    uint64_t norm1;
    uint64_t qkv_weight;
    uint64_t qkv_bias;
    uint64_t q_norm;
    uint64_t k_norm;
    uint64_t attn_proj_weight;
    uint64_t attn_proj_bias;
    uint64_t norm2;
    uint64_t gate_weight;
    uint64_t gate_bias;
    uint64_t up_weight;
    uint64_t up_bias;
    uint64_t down_weight;
    uint64_t down_bias;
} ds4_glm53_vision_layer_weights;

typedef struct {
    uint64_t patch_weight;
    uint64_t patch_bias;
    uint64_t post_norm;
    uint64_t downsample_weight;
    uint64_t downsample_bias;
    uint64_t merger_proj;
    uint64_t merger_norm;
    uint64_t merger_norm_bias;
    uint64_t merger_gate;
    uint64_t merger_up;
    uint64_t merger_down;
    ds4_glm53_vision_layer_weights layer[DS4_GLM53_VISION_LAYERS];
} ds4_glm53_vision_weights;
#endif

/* Encode normalized, block-major image patches into 4096-wide language-model
 * embeddings. GPU implementations keep every intermediate on device. */
int ds4_gpu_glm53_vision_encode(
        float                          *out,
        const float                    *patches,
        uint32_t                        grid_h,
        uint32_t                        grid_w,
        const void                     *model_map,
        uint64_t                        model_size,
        const ds4_glm53_vision_weights *weights);

#ifndef DS4_DEEPSEEK4_VISION_TYPES_DEFINED
#define DS4_DEEPSEEK4_VISION_TYPES_DEFINED
#define DS4_DEEPSEEK4_VISION_LAYERS 32u
#define DS4_DEEPSEEK4_LANGUAGE_LAYERS 43u
#define DS4_DEEPSEEK4_MTP_LAYERS 3u

typedef struct {
    uint64_t norm1;
    uint64_t qkv_weight;
    uint64_t qkv_bias;
    uint64_t attn_proj_weight;
    uint64_t attn_proj_bias;
    uint64_t norm2;
    uint64_t mlp_w1;
    uint64_t mlp_w2;
} ds4_deepseek4_vision_layer_weights;

typedef struct {
    uint64_t patch_weight;
    uint64_t patch_bias;
    uint64_t post_norm;
    uint64_t aligner_w1;
    uint64_t aligner_w1_bias;
    uint64_t aligner_w2;
    uint64_t aligner_w2_bias;
    uint64_t image_start;
    uint64_t image_pad;
    uint64_t image_newline;
    uint64_t image_end;
    uint64_t visual_router_bias[DS4_DEEPSEEK4_LANGUAGE_LAYERS];
    uint64_t mtp_visual_router_bias[DS4_DEEPSEEK4_MTP_LAYERS];
    uint64_t hash_router_bias[3];
    ds4_deepseek4_vision_layer_weights layer[DS4_DEEPSEEK4_VISION_LAYERS];
} ds4_deepseek4_vision_weights;
#endif

/* Encode row-major normalized 14x14 RGB patches. The output is the natural
 * row-major 3x3-aligned grid; N-layout permutation and sentinels are applied
 * by the prompt layer once the image's token position is known. */
int ds4_gpu_deepseek4_vision_encode(
        float                              *out,
        const float                        *patches,
        uint32_t                            grid_h,
        uint32_t                            grid_w,
        const void                         *model_map,
        uint64_t                            model_size,
        const ds4_deepseek4_vision_weights *weights);

/* Replace token rows with projected image embeddings and repeat each row into
 * every GLM hyperconnection stream. Must be called in an active command batch. */
int ds4_gpu_glm53_scatter_image_hc(
        ds4_gpu_tensor       *hc,
        const ds4_gpu_tensor *image,
        uint32_t              dst_row,
        uint32_t              image_row,
        uint32_t              rows,
        uint32_t              total_rows,
        uint32_t              n_embd,
        uint32_t              n_hc);

/* GLM-5.3 Kimi Delta Attention. Recurrent and convolution state stay FP32. */
int ds4_gpu_glm53_kda_decode(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *k,
        const ds4_gpu_tensor *v,
        const ds4_gpu_tensor *raw_gate,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_conv_offset,
        uint64_t              k_conv_offset,
        uint64_t              v_conv_offset,
        uint64_t              a_log_offset,
        uint64_t              dt_bias_offset,
        uint64_t              output_norm_offset,
        uint32_t              n_heads,
        uint32_t              n_rows,
        float                 gate_lower_bound,
        float                 norm_eps);

int ds4_gpu_glm53_kda_prefill(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *conv_state,
        ds4_gpu_tensor       *recurrent_state,
        ds4_gpu_tensor       *q,
        ds4_gpu_tensor       *k,
        ds4_gpu_tensor       *v,
        ds4_gpu_tensor       *raw_gate,
        const ds4_gpu_tensor *raw_beta,
        const ds4_gpu_tensor *output_gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              q_conv_offset,
        uint64_t              k_conv_offset,
        uint64_t              v_conv_offset,
        uint64_t              a_log_offset,
        uint64_t              dt_bias_offset,
        uint64_t              output_norm_offset,
        uint32_t              n_heads,
        uint32_t              n_tokens,
        float                 gate_lower_bound,
        float                 norm_eps);

/* Q4_K sibling of the decode attention-output/HC compound. */
int ds4_gpu_matmul_q4_K_hc_expand_available(void);
int ds4_gpu_matmul_q4_K_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t              n_embd,
        uint32_t              n_hc);

/* Decode-island CUDA graph capture (CUDA backend; Metal/ROCm/CPU stub it
 * out and stay eager).  Design ported from the Entrpi/ds4 batched-serving
 * fork's per-layer decode graph capture.  The key identifies a captured
 * island: layer, island index, and the stable device-storage addresses that
 * the captured kernels bake in (never short-lived view-wrapper addresses).
 * ds4_cuda.cu mirrors this struct
 * byte-for-byte (it does not include this header); keep both in sync. */
typedef struct ds4_decode_graph_key {
    uint32_t il;
    uint32_t island;    /* 0: layer top to pre-rope; 1: attn-out to layer end */
    uint32_t variant;
    uint32_t _pad;
    void    *cur_hc;
    void    *after_attn_hc;
    void    *after_ffn_hc;
    void    *attn_norm;
} ds4_decode_graph_key;

/* Exact-N uses a disjoint CUDA graph-cache domain so its five batch-row
 * activation addresses cannot evict the ordinary decode variants. */
#define DS4_DECODE_GRAPH_VARIANT_EXACTN 0x80000000u

int  ds4_gpu_decode_graphs_supported(void);
/* 1: replayed (island already executed; skip encoding it)
 * 0: capturing (encode the island, then call _end)
 * -1: run eagerly */
int  ds4_gpu_decode_graph_begin(const ds4_decode_graph_key *key);
/* 0: capture committed and launched; -1: capture failed (entry retired;
 * the caller must re-encode the island eagerly -- no work was executed). */
int  ds4_gpu_decode_graph_end(const ds4_decode_graph_key *key);
void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key *key);
void ds4_gpu_decode_graphs_invalidate(void);
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)
void ds4_gpu_decode_graph_counters(
        uint64_t *captures,
        uint64_t *replays,
        uint64_t *warms,
        uint64_t *no_slots,
        uint64_t *failures);
#endif

#ifdef __cplusplus
}
#endif

#endif
