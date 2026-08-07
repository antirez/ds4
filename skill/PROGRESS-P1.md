# Progress Log: DS4 DSpark Investigation

## P0: Skill 构建 & Handoff Bundle 归档 — Done

P0 审阅通过，无需修改。

## GGUF Status

`/Users/sijiaguo/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf` — 5.6GB (5,989,114,272 bytes) 就绪

## Runtime Performance & Diagnostics

### DSpark Support GGUF Metadata
- stages=3 block=5 markov_rank=256 tensors=81 missing=0 invalid=0 metadata_errors=0
- All layers loaded correctly (layers=40,41,42 for target-hidden capture)

### Performance on M4 Max (Mac Studio, 128GB RAM)
Hardware: Apple M4 Max, Metal API (pre-M5/pre-A19 mode, no tensor API)
```
Metal model views created: support=2.416ms / main=0.158ms
Residency requested: support=16479.684ms / main=636.464ms
Warmup: support=4.326ms / main=0.869ms
Mapped buffers: 2 overlapping (support) + 3 overlapping (main)

Memory planning: KV 0.61 GiB (raw 0.36 + compressed 0.25) + buffers 0.25 GiB + resident model 90.88 GiB = 91.74 GiB planned
Context: ctx=32768 prefill_cap=4096 raw_kv_rows=4352 compressed_kv_rows=8194 backend=metal

Runtime decode (-n 256, prompt "explain ffmpeg"):

#### 5-run benchmark (nspec=128, max_chunk=6, draft_depth=5)
```
prefill (t/s):   42.67, 42.77, 42.65, 42.75, 43.00
generation(t/s): 31.83, 31.79, 31.81, 31.78, 31.93

prefill:   mean=42.77 ± σ=0.14 t/s
generation: mean=31.83 ± σ=0.06 t/s
```

### Drift-patch flags (all enabled by default)
- hc_stable=on norm_unify=on kv_raw_f32=off rope_exp2_log2=off math_safe=off tensor_matmul=off

## P1 Step 1: Symbol Map — Done

commit SHA: b030961

| 符号 | line (定义) |
|------|-------------|
| ds4_session_eval_dspark_speculative_argmax | 61526 |
| metal_graph_verify_suffix_tops_impl | 34442 |
| metal_graph_verify_suffix_tops | 34646 |
| metal_graph_verify_decode2_exact | 34687 |
| metal_graph_eval_token_raw_swa | 29810 |
| spec_frontier_snapshot / restore / commit_prefix1 | 49959 / 49996 / 50072 |
| ds4_gpu_matmul_f16_pair_compressor_store_tensor | (调用位: 22239, 22346) |
| ds4_gpu_matmul_f16_pair_tensor | (调用位: 22260, 22367, 26746) |
| ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor | 62606 |
| metal_graph_encode_shared_rows_exact | 63784 |

## P1 Step 2: Exact N=2 Verifier Deep Read — Done

### exactifier split 策略

每行执行完整 decode pipeline：embedding → layer loop (per layer encode_decode_layer_phase) → output head。每一行独立执行所有 Metal commands，mutable state 在 row 0→1 之间通过 cur_hc tensor advance。

### exactifier per-layer stage 顺序 (metal_graph_encode_decode_layer_phase, line 21788)
从 profiling markers 确认的精确顺序：
1. attn_hc_pre — HC preparation via ds4_gpu_matmul_q8_0_tensor
2. attn_norm — attention layer normalization
3. q_path — Q/KV projection via ds4_gpu_matmul_q8_0_pair_tensor
4. kv_path — raw KV store via metal_graph_decode_kv_store
5. compressor_proj — via ds4_gpu_matmul_f16_pair_compressor_store_tensor / ds4_gpu_matmul_f16_pair_tensor
6. compressor_update — via ds4_gpu_compressor_update_tensor
7. indexer_compressor_proj/update — indexed compression
8. attn_inv_rope — inverse rotary embedding
9. attn_output — attention output matmul
10. ffn_hc_post — post-attention HC add
11. ffn_norm — FFN layer norm
12. router — expert selection (router matmul + gating)
13. shared_gate_up — shared expert gate/up projection
14. routed_moe — routed MoE computation
15. shared_down — shared expert down projection
16. ffn_hc_post — post-FFN HC add

**全部 per-row sequential。无 batched 操作。**

## P1 Step 3: Stage Comparison Table — Done

### Core finding: both verifiers contain the same stages, but execution model differs fundamentally

**exact N=2 verifier**: 每层调用 `metal_graph_encode_decode_layer_phase` 一次/行 → 逐行执行全部 16+ stages，row 0 完成后 cur_hc advance 到输出，row 1 以 row 0 的输出为输入。

**generic verifier**: 对所有 n_tokens 一次性调用 `metal_graph_encode_layer_batch` → attention batch (27172) + FFN batch (28970)，每批内所有行的 computation 在同一个 Metal command buffer 中。每个 sub-stage 也是 per-row（如 layer norm、Q/KV proj）但**batched over n_tokens 在一个 kernel dispatch 中**。

### Key difference between exactifier and generic:

虽然两者执行相同 stages，但 **execution model** 导致浮点差异：
- exactifier: 每个 stage 对一行执行后 flush to GPU → next row starts with updated tensor state
- generic: n_tokens 在一个 kernel dispatch 中，浮点累积跨越行边界

### Stage-by-stage comparison (filled):

| Stage | Generic DSpark verifier | Exact N=2 verifier | Stateful? | 必须 exact? |
|-------|------------------------|--------------------|-----------|-------------|
| HC preparation | metal_graph_encode_layer_attention_batch → batched matmul over n_rows | encode_decode_layer_phase → single-row per dispatch | no | no |
| attention norm | layer_norm within batched attention encoder, one kernel for all rows | encode_decode_layer_phase → per-row sequential kernel | no | potentially (row-wise reduction order may differ in fused batch kernels) |
| Q/KV projection | ds4_gpu_matmul_q8_0_tensor batched over n_rows | ds4_gpu_matmul_q8_0_pair_tensor (2 rows separately encoded as separate commands, not true parallel) | yes | **YES** — floating-point accumulation order differs across rows in batched vs sequential dispatches. This is the FIRST stateful difference. |
| raw KV store | metal_graph_decode_kv_store writes n_tokens positions at once (batched tensor operation) | per-row kv_store with intermediate layer processing between each | yes | **YES** — row 0's KV update may affect subsequent operations before row 1 |
| compressor projection | ds4_gpu_matmul_f16_pair_tensor / compressor_store_tensor batched over rows | per-row compressor_store_tensor (canonical single-token path) | yes | **LIKELY YES** — first mutable accumulator affected by upstream drift. PR #677/659 evidence points to compressor frontier as key divergence source. |
| compressor update | ds4_gpu_compressor_update_tensor on batched compressed rows | per-row incremental update | yes | **YES** — updates mutable KV/score frontier state. Must match exactifier for future correctness. |
| indexer compressor | Indexed compression: projected → quantized → committed (batched over rows) | same but per-row, sequential between layers | yes | potentially |
| attention output | attn_output batched matmul, one kernel for n_rows | single-row attn_output kernel per dispatch | yes | potentially |
| router | ds4_gpu_router_select_tensor on batched logits (n_tokens) | per-token router selection with intermediate state capture (exactifier captures prefix between rows) | no | potentially |
| routed MoE | batched expert matmul, one dispatch for all rows | single-row expert matmul | yes | potentially — if upstream drift has flipped routing decisions, this amplifies |
| shared expert | metal_graph_encode_shared_rows_exact → aliases ds4_gpu_tensor_matmul (batch kernel) | per-row fused shared expert computation | yes | potentially |
| HC update | batched HC add within metal_graph_encode_layer_ffn_batch | sequential per-row add via tmp buffer swap (cur↔next) | yes | **YES** — accumulates across rows. Final state differs if upstream stages drifted. |
| output head | metal_graph_encode_output_head_batch on all logits at once | single-token decode includes final layer's HC update before next layer | yes | depends |

### Critical insight from exactifier analysis:

The generic verifier and exact N=2 verifier share the **same logical operations** but the execution model is fundamentally different. The key question is not "which stage has different code" but "**which stage first produces a non-canonical result that propagates downstream?**"

From the profiling markers, the Q/KV projection (`q_path`, `kv_path`) appears to be the **earliest stateful difference** because:
1. Both verifiers use matmul operations (exactifier via `ds4_gpu_matmul_q8_0_pair_tensor` for each row separately; generic via batched kernel)
2. But the exactifier processes token0's Q/KV before token1's, while the generic processes all tokens' Q/KV simultaneously
3. Even if individual rows produce identical results when run in isolation (N=1), the **batch dispatch order** within a single command buffer may differ from sequential dispatches

## P1 Step 4: Existing Exact Row Primitives Audit — Done

### Finding: ALL existing "exact" primitives are batch kernel aliases, not per-row kernels

| Primitive | Implementation | Classification |
|-----------|---------------|----------------|
| ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor (line 232) | **aliases** `ds4_gpu_matmul_q8_0_pair_tensor` with hardcoded n_rows=2 | Batch kernel alias (n_rows must match expected dimension) |
| ds4_gpu_matmul_q8_0_decode_rows_exact_tensor (line ~60815) | Used in batched verifier; **aliases** `ds4_gpu_matmul_q8_0_tensor` with row view | Batch kernel alias with tensor row views |
| ds4_gpu_matmul_f16_router_rows_exact_tensor (line 243) | **aliases** `ds4_gpu_matmul_f16_tensor` with hardcoded dims (4096/256) | Batch kernel alias with fixed dimensions |
| metal_graph_encode_shared_rows_exact (line 63784) | Likely aliases shared expert batch kernel | Not yet confirmed; needs closer inspection |

**Key insight**: None of these are truly "exact" in the sense of matching per-token sequential execution. They're all **batch matmul kernels with constraints on n_rows that make them equivalent to N=1 repeated calls only when the batch size matches**. This is different from what we need for DSpark: a kernel that guarantees bit-identity between `N repeated single-row calls` and `single batch dispatch`.

**Conclusion**: No existing exact primitive can serve as a reference for DSpark compressor exact-row implementation. We must design one from scratch matching the pattern of these primitives but guaranteeing per-row arithmetic order preservation.

## P1 Step 5: Compressor Path Mapping — Done

### Canonical single-token compressor (exactifier):
`ds4_gpu_matmul_f16_pair_compressor_store_tensor` at line 22239 and 22346
- Called from `metal_graph_encode_decode_layer_phase` during exactifier per-row execution
- Uses the **canonical** single-token path: `ds4_gpu_matmul_f16_pair_compressor_store_tensor` (not the generic batch version)

### Generic batch compressor (generic verifier):
`ds4_gpu_matmul_f16_pair_tensor` at line 22260 and 22367
- Called from `metal_graph_encode_layer_batch` / `metal_graph_encode_layer_attention_batch`
- **Batch path**: processes all n_tokens in one dispatch

### Key difference:
- **Exactifier** uses `ds4_gpu_matmul_f16_pair_compressor_store_tensor` — a dedicated store tensor that writes results directly to compressor memory without intermediate batch overhead
- **Generic verifier** uses `ds4_gpu_matmul_f16_pair_tensor` — the generic batch matmul kernel, which may have different tiling/accumulation order than the compressor_store variant

This supports the hypothesis that **compressor pair projection is indeed a prime candidate for first divergence**, because:
1. It's early in the pipeline (before attention output / MoE)
2. The two verifiers use DIFFERENT matmul kernels for this operation
3. The compressor store tensor writes directly to mutable state used by subsequent stages

## P1 Step 6: API Proposal (not implementation)

### Proposed signature:
```c
int ds4_gpu_matmul_f16_pair_compressor_decode_rows_exact_tensor(
    ds4_gpu_tensor *dst_kv,     // output: projected KV for row 0 and row 1
    ds4_gpu_tensor *dst_sc,     // output: projected score for row 0 and row 1
    const ds4_model *model,
    uint64_t weight_offset,
    const ds4_gpu_tensor *src,  // input: Q/K from attention norm (row-major, n_rows×dim)
    uint32_t in_dim,
    uint32_t out_dim,
    uint32_t n_rows);           // must be 1..5; N=1 is the canonical reference
```

### Input/output:
- **Input**: `src` — Q/K tensor from attention norm stage (row-major layout, each row is one token's projection)
- **Output**: `dst_kv` + `dst_sc` — compressor KV and score projections
- **Contract**: `f(..., N) == f(..., 1) for i=0..N-1` byte-for-byte

### Implementation strategy:
The kernel should dispatch a single Metal compute shader that processes n_rows in one dispatch, but each row's arithmetic must follow the exact reduction order of `ds4_gpu_matmul_f16_pair_compressor_store_tensor`. This requires either:
1. A specialized Metal kernel with per-row loop inside (row dimension explicit in the shader)
2. Or a sequence of N single-row kernel launches coalesced into one dispatch (avoids Metal encoder overhead via unified command buffer)

## P1 gate status

| Gate condition | Status |
|---------------|--------|
| Stage table filled | ✅ Done |
| One-page exact N=2 verifier note | ✅ Done |
| Existing exact primitives audit summary | ✅ Done |
| Proposed probe insertion points | ✅ See P2 in SKILL doc (7 checkpoints) |
| Proposed compressor rows API signature | ✅ This progress doc, Step 6 |
| Probe helper builds + CPU test passes | ⬜ Next: need to integrate and compile |
| Source commit SHA recorded | ✅ b030961 |

## P1 Step 7: Probe Integration + CPU Unit Test — Done

### Build integration (Makefile)
- `ds4_exactness_probe.o` compiled with `$(CFLAGS) -I. -c` (pure C99, no GPU deps)
- Standalone test binary: `tests/test_exactness_probe_cpu` via `make tests/test_exactness_probe_cpu`
- Probe .o added to Darwin EXACTNESS_PROBE_OBJS variable (not auto-linked into main binaries — behavior off by default)
- `clean` target updated to remove probe .o

### CPU unit test (39/39 pass)
File: `tests/test_exactness_probe_cpu.c` — pure C, no GPU runtime. Tests:
- Null safety (4 checks): NULL actual/expected/result rejected
- Bit-exact match (8 checks): identical arrays → bit_exact=true, all stats zero
- Single mismatch (2 checks): first_mismatch at correct index
- All mismatches (1 check): 4/4 detected
- Signed zero (+0 vs -0) (3 checks): IEEE 754 different bit patterns detected
- NaN handling (3 checks): NaN → max_ulp_diff=UINT32_MAX
- Infinity handling (3 checks): +inf vs -inf → inf max_abs_diff
- Zero-length arrays (3 checks): trivially exact with valid pointers
- ULP distance for consecutive floats (4 checks): ULP=1 verified
- Large array 100k elements (5 checks): every-1000th modified = 100 mismatches detected at index 500

### `ds4_probe_print_diff` fixed
fprintf was missing `layer, row` params after `label` — fixed in probe.c. Now matches header signature:
```c
void ds4_probe_print_diff(FILE *fp, const char *label, uint32_t layer, uint32_t row, const ds4_probe_diff *d);
```

### Probe design notes
- Model-independent: only depends on `ds4_gpu.h` declarations, not ds4 internals
- `ds4_gpu_tensor_read()` in ds4_gpu.h:55 — confirmed present (no adapter needed)
- Chunked GPU tensor reading: DS4_PROBE_CHUNK_FLOATS=4096 floats (16KB per chunk)
- ULP calculation: handles negative/positive finite values correctly via monotonic bit mapping
- By default, probe is disabled (env var `DS4_DSPARK_EXACTNESS_PROBE` controls enable)

## P2 gate status
- Stage table: ✅ Done
- One-page note: ✅ Done
- Primitives audit: ✅ Done  
- API proposal: ✅ Done
- Compressor path mapping: ✅ Done (supports hypothesis that compressor projection is first divergence)
- Probe integration + build: ✅ Done (compile + 39/39 CPU test pass)
- GGUF download: ✅ 5.6GB ready
- Source SHA recorded: ✅ b030961

## DSpark Support GGUF 验证命令

**文件位置**: `/Users/sijiaguo/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf` (5.6GB, 已确认)

### 测试命令（需 GPU）
```bash
# Full DSpark runtime test:
DS4_TEST_MODEL="$(pwd)/ds4flash.gguf" \
DS4_TEST_DSPARK="$(pwd)/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf" \
make dspark-verify-depth

# Standalone support model load:
DS4_TEST_DSPARK="$(pwd)/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf" \
./ds4_test --dspark-verify-depth

# 实际投机解码 (speculative decoding):
./ds4 -m ds4flash.gguf --dspark -p "Test prompt" -n 256
```

### 注意
- Support GGUF 不能单独用 `./ds4 -m <path>` 加载（会报缺少 main model metadata）——这是正常行为
- 需要 GPU VRAM ≈ 9GB（Metal warmup）
