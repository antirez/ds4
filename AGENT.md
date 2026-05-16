# Agent Notes

`ds4.c` is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase with
Objective-C only where Metal requires it and Metal kernels under `metal/`.

## Goals

- Keep the production path as whole-model Metal graph inference.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained
  attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed
  kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is
  intentional.
- Prefer short Metal smoke tests for build verification.

## Layout

- `ds4.c`: model loading, tokenizer, CPU reference code, Metal graph scheduling,
  sessions, disk-cache payload serialization.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: compute kernels.
- `ds4_cuda.cu`: CUDA backend.  Single TU; mirrors ds4_metal.m's role on
  NVIDIA / DGX Spark.  Dispatches quantized matmuls either through its own
  bespoke kernels or - when `DS4_CUDA_USE_MMQ=1` is set - through the
  vendored llama.cpp `mul_mat_q` kernels in `cuda/mmq/`.
- `cuda/mmq/`: vendored llama.cpp ggml-cuda kernels (`mmq.cuh`, `mma.cuh`,
  `vecdotq.cuh`, `quantize.{cu,cuh}`, `mmid.{cu,cuh}`, `common.cuh`,
  `ggml-common.h`, `vendors/cuda.h`) pinned to upstream commit `5c0e9468`,
  plus the ds4-side adapter (`ds4_ggml_stubs.{h,cu}`, `ds4_mmq.{h,cu}`,
  redirect `ggml.h` / `ggml-impl.h` / `ggml-cuda.h`).  See
  `cuda/mmq/VENDOR.md` for the symbol-resolution table and the upstream
  re-sync procedure.  Phases 0-7 of the lift are documented in
  `local/docs/ds4_mmq_lift_plan.html` (in the auto-round companion repo).
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material.

## CUDA environment variables

The CUDA backend has a few opt-in switches that affect the inner matmul
dispatch.  Defaults preserve the historical behavior; set the variables
to opt in to the newer paths.

- `DS4_CUDA_USE_MMQ` (default on): the vendored llama.cpp `mul_mat_q`
  kernels in `cuda/mmq/` are the default CUDA path for quantized
  matmuls - Q8_0 dense (attention projections, shared expert, lm_head)
  and the routed-MoE block when the GGUF uses IQ2_XXS for gate/up and
  Q2_K for down (the V4 Flash configuration).  Other quant
  combinations fall through to the existing kernels.  Set
  `DS4_CUDA_USE_MMQ=0` (or `off` / `no` / `false`) to disable and
  revert to the native Q8 warp kernels (the
  `matmul_q8_0_preq_*_kernel` family).  The legacy Q8&rarr;FP16
  expansion cache plus `cublasGemmEx` pipeline that previously sat
  between mmq and the native kernels was deleted in Step 1 of the
  optimization plan - mmq is fast enough that the cache was no longer
  earning its complexity.  Validated on RTX PRO 6000 Blackwell
  (sm_120, CUDA 13.0) against V4 Flash: prefill 357-1041 tok/s vs
  357-373 baseline (sustained ~2.80x), gen within run-to-run
  variance.
- `DS4_CUDA_MMQ_MOE_MIN_TOKENS=N`: minimum `n_tokens` at which the
  routed-MoE mmq path activates.  Default 2.  At `n_tokens=1` mmq's
  matrix-matrix-shaped path has higher per-launch fixed cost than the
  vector path; that case is handled by the mmvq decode branch below.
  Override to 1 to force mmq even at decode (likely slower than mmvq).
- `DS4_CUDA_MMQ_X_MAX=N`: clip `get_mmq_x_max_host` to N (rounded down
  to a multiple of 8) when sweeping tile widths.  Diagnostic only; the
  vanilla 128 wins on sm_120 (RTX PRO 6000 Blackwell) so the default
  is unchanged.  Step 4 of the optimization plan ran a {32, 64, 96,
  128} sweep against V4 Flash: X=32 lost ~20%, X=64 lost ~6%, X=96
  was within +/-1% of default but mixed across ctx points.  May be
  useful on other arches.
- `DS4_CUDA_NO_MMVQ_DECODE` (default unset): opt-out of the vendored
  llama.cpp `mul_mat_vec_q` (mmvq) decode path.  Step 6 of the
  optimization plan: mmvq is structurally optimal for the n_tokens=1
  routed-MoE and dense attention projection cases (one CUDA block per
  output row, no wasted column tiling).  Wires in two places:
    1. `routed_moe_launch` for the V4 Flash IQ2_XXS gate/up + Q2_K
       down MoE shape AND the Q4_K-only MoE shape, gated on
       `n_tokens * n_expert_used <= MMVQ_MAX_BATCH_SIZE` (8 on
       Blackwell).  Two separate `ds4_mmq_<type>_moe_vec` calls
       preserve the DeepSeek V4 clamp epilogue exactly; the fused
       `ds4_mmq_<type>_moe_pair_vec` entry exists but is not yet
       wired (fusion applies silu without clamp).
    2. `cuda_matmul_q8_0_tensor_labeled` for n_tok=1 (attention
       projection decode) via `ds4_mmq_q8_0_dense_vec`.
  Set to `1` (or any non-empty value) to fall through to the legacy
  paths and the existing mmq path.
- `DS4_CUDA_MMVQ_DECODE_MAX_TOKENS=N` (default 1): cap on n_tokens
  routed through the mmvq decode branch in `routed_moe_launch`.  Valid
  range 0-8.  0 disables (same as `DS4_CUDA_NO_MMVQ_DECODE=1` for the
  MoE path).  Values 2-8 extend mmvq coverage to short-prefill batches
  but require n_tokens * n_expert_used &le; 8 for the down matmul.
- `DS4_CUDA_MOE_GRAPHS` (**default OFF as of 2026-05-16-rev; previously
  default ON**): CUDA Graph capture+replay for the mmvq routed-MoE decode
  block and (Step 8.2) for the n_tok=1 dense Q8_0 vec path used by
  attention projections.  Each kernel sequence is captured into a
  `cudaGraphExec_t` on first execution with a given
  (layer-shape, buffer-pointer) tuple and replayed via
  `cudaGraphLaunch` on subsequent calls.  The MoE cache holds 256
  entries (one per layer shape-class); the dense Q8_0 cache holds
  1024 entries (one per attention projection per layer).  Replay
  eliminates ~5-15&micro;s of CPU&harr;driver round-trip per kernel
  launch, the dominant overhead at decode where individual kernels
  are small.  Requires an explicit non-default stream (`g_moe_stream`,
  shared across both caches) for capture; routes the ds4_mmq pool's
  `cudaMallocAsync` through the same stream via the thread-local
  `ds4_pool_set_stream()` so allocations don't invalidate capture.
  Each cache slot falls through cleanly to the un-captured path on
  any capture error.
  
  **Why default OFF:**  The original default-ON flip was validated only
  by 10/10 bit-identical greedy decode against the graphs-OFF baseline
  on the same branch.  That gate proves self-determinism but does NOT
  prove exact-byte equivalence vs the legacy path
  (`DS4_CUDA_USE_MMQ=0`).  External proof-harness validation on
  GB10/Spark in May 2026 observed gross output corruption on a 32-token
  smoke with graphs ON that disappears with graphs OFF, implying an
  undeclared cross-stream dependency in the captured region (likely a
  pool-allocated buffer touched by an uncaptured kernel on stream=0).
  Re-enabling graphs as the default requires root-causing and fixing
  that hazard.  Opt-in for experiments:  `DS4_CUDA_MOE_GRAPHS=1`
  (or `on` / `yes` / `true`).

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and Metal are available. Use live server tests only when intentionally
testing the API surface.
