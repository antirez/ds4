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

The CUDA backend dispatches every Q8_0 dense matmul (attention
projections, shared expert, lm_head, attn_output_b) through one of three
strategies, defaulting to `mmq` with a downgrade chain on init failure.
cuBLAS is initialized unconditionally at backend startup regardless of
the chosen strategy; we observed that this triggers CUDA driver-state
that makes mmq ~4x faster on sm_121 vs a binary that skips
`cublasCreate`, so the cublas path remains resident even when it isn't
selected.

|  Strategy | When picked                                | What it runs                                      |
|-----------|--------------------------------------------|---------------------------------------------------|
| `mmq`     | default; on every CUDA arch we've validated| vendored llama.cpp `mul_mat_q` (cuda/mmq/)        |
| `cublas`  | explicit override or fallback if mmq init fails | `cuda_q8_f16_ptr` Q8->FP16 cache + `cublasGemmEx` |
| `warp8`   | explicit override or last-resort fallback  | native `matmul_q8_0_preq_*_kernel` family         |

The strategy is logged once on first dispatch, e.g.:

    ds4: CUDA Q8_0 dispatch: mmq (sm_120, 1792 GB/s memory bandwidth) [default]
    ds4: CUDA Q8_0 dispatch: mmq (sm_121, 546 GB/s memory bandwidth) [default]
    ds4: CUDA Q8_0 dispatch: cublas (sm_121, 546 GB/s memory bandwidth) [DS4_CUDA_PREFILL_PATH=cublas]

(The bandwidth figure is informational; we don't tier on it.)

Benchmarked deltas at ctx=2048, V4 Flash IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8,
promessi_sposi prompt, no weight server:

| Arch                                       | mmq    | cublas | warp8 |
|--------------------------------------------|--------|--------|-------|
| PRO 6000 Blackwell sm_120 (~1.8 TB/s GDDR7) | **1092** | 373   | 373   |
| GB10 Spark sm_121 (~546 GB/s LPDDR5X)       | **458**  | 401   | 56    |

As of the in-process VMM arena commit (`cuda: route fd-cache through the
VMM arena when supported`), single-process runs on discrete GPUs also
land in the ~2 MiB-page layout that the weight server uses: PRO 6000 mmq
prefill at ctx=2048 went from 1076 (arena baseline) to 2196 with the
in-process VMM allocator -- within 0.07% of the weight server's 2197
ceiling. The win persists across the entire ctx sweep (1.87x at
ctx=32768). See `local/docs/ds4_inprocess_vmm_plan.html` for the design
and `local/docs/ds4_vmm_landing/pod_{arena,vmm}.csv` for the A/B.

The weight server remains the right answer when more than one process
holds the same model concurrently (proof harness, multi-profile sweeps,
MTP correctness) -- it amortises the ~60 s base upload across N workers.
For a single ds4-bench, ds4-server, or one-shot ./ds4 -p ..., the
in-process allocator now gives the same prefill ceiling with no setup
tax. Opt out via `DS4_CUDA_VMM_ARENA=0` (escape hatch for profiler
quirks); auto-disabled when the worker is already importing weights
from a weight server (`DS4_CUDA_WEIGHT_IPC_MANIFEST` set).

The MTP verifier (Option D, `DS4_CUDA_MTP_VERIFIER_USE_MMQ`) forces
warp8 inside the verifier regardless of the chosen strategy, because
the drafter is trained against legacy decoding and only warp8 is
bit-identical to that distribution.

Env-var surface:

- `DS4_CUDA_PREFILL_PATH=mmq|cublas|warp8|auto` (default `auto` ->
  mmq).  Explicit override.  `auto` and unset both resolve to mmq;
  `cublas` opts into the FP16-expansion + cublasGemmEx path; `warp8`
  forces the simplest fallback (useful for debugging or for the MTP
  drafter-bit-equivalence cases when not using Option D).
- `DS4_CUDA_USE_MMQ=0` (legacy): equivalent to
  `DS4_CUDA_PREFILL_PATH=cublas` for back-compat with the prior
  override.  `DS4_CUDA_PREFILL_PATH` takes precedence if both are set.
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
- `DS4_CUDA_MTP_VERIFIER_USE_MMQ` (default unset, behaves as `0`):
  repro switch for Bug 2.  By default, every MTP verifier call is
  bracketed in `ds4.c` with `ds4_gpu_set_mtp_verifier(1)` /
  `ds4_gpu_set_mtp_verifier(0)`; the CUDA backend honors this by
  routing all Q8_0 dense matmuls (and the routed-MoE dispatch, via
  the same `ds4_cuda_use_mmq()` gate) onto the legacy native kernels
  for the duration of one verifier call.  This is the Option D
  hybrid: mmq for prefill and non-MTP decode, legacy native kernels
  inside the verifier.  Necessary because mmq's stream-k + MMA FP32
  reduction order drifts ~1 ULP/layer from the legacy `warp8`
  kernel; the MTP drafter is trained against legacy-style decoding,
  so an mmq verifier flips tight-margin argmax tokens and collapses
  draft acceptance (analyst measured 0/314 on GB10 with mmq verifier
  active).  Set to `1` (or `on` / `yes` / `true`) to bypass the gate
  and let mmq run inside the verifier &mdash; reproduces the broken
  behavior, useful only for bisection.  See
  `local/docs/ds4_mmq_mtp_correctness_plan.html` in the auto-round
  companion repo for the full mechanism and validation plan.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and Metal are available. Use live server tests only when intentionally
testing the API surface.

### Multi-process testing with the weight server

For any test that spawns more than one `ds4` process against the same model
- the proof harness (`tests/ds4_proof.py`)
- multi-profile bench sweeps
- MTP correctness work (loads base + MTP gguf into the same device)

use the `ds4_weight_server` to share weights via CUDA VMM or IPC.  Without it,
each process pays the full base-model upload cost (~67 s for V4 Flash Q2 on
PRO 6000) and an MTP gguf alongside the base may OOM due to single-allocation
fragmentation even when total free VRAM is sufficient.

For *single*-process runs the in-process VMM arena (auto-enabled wherever
the driver supports `cuMemCreate`) already gives the same 2 MiB-page
layout the weight server uses, so the prefill ceiling is reached without
a sidecar -- no manifest, no broker socket, no lock file. The weight
server is only worth the setup cost when N >= 2 processes share the
same model on the same card.

Two patterns:

**Proof harness (recommended)** — let the harness manage lifecycle:
```
python3 tests/ds4_proof.py --plan PLAN.json --bin ./ds4 --base ds4flash.gguf \
    --start-weight-server --weight-server-bin ./ds4_weight_server \
    --weight-server-backend vmm --weight-server-scope base|mtp|both \
    --weight-server-reserve-gb 8 \
    --weight-server-manifest /tmp/ws_manifest.json
```

**Standalone** — long-lived server + multiple clients:
```
# launch (background; writes manifest when ready)
./ds4_weight_server --base ds4flash.gguf --mtp gguf/...-MTP-*.gguf \
    --manifest /tmp/ws_manifest.json --backend vmm --scope both \
    --reserve-gb 8 &

# every client process
DS4_CUDA_WEIGHT_IPC_MANIFEST=/tmp/ws_manifest.json ./ds4-bench ...
DS4_CUDA_WEIGHT_IPC_MANIFEST=/tmp/ws_manifest.json ./ds4 ...
```

Notes:
- `--reserve-gb` must leave room for context buffers + transient allocs.
  Empirically on PRO 6000 (96 GiB total, ~80.8 GiB model): `8` works,
  `12` fails preflight.  On Spark / GB10 (128 GiB) the analyst-validated
  setting is `--reserve-gb 24`.
- `--backend vmm` uses CUDA managed memory and is the post-2026-05-16
  default; `ipc` is the older path.
- `--scope` controls which weights to upload: `base`, `mtp`, or `both`.
  Use `base` for non-MTP work to save VRAM.
- If the server crashed mid-run, clean up
  `/tmp/ds4_weight_server_cuda*.lock` and any lingering processes
  (`pkill -9 -f ds4_weight_server`) before retrying.
- For a single one-shot `./ds4 -p ...` invocation the load cost is
  amortized over generation and the weight server is not worth setting
  up; the in-process VMM arena gives the same prefill ceiling with no
  setup tax. Use the weight server only when more than one process
  shares the model.

In-process VMM arena env vars:
- `DS4_CUDA_VMM_ARENA=0`: disable the in-process VMM allocator; fall
  back to `cudaMalloc`-backed arena. Escape hatch for profiler quirks
  with VMM-mapped ranges; unset (or any non-`0` value) enables.
- `DS4_CUDA_VMM_ARENA_CHUNK_MB=N`: minimum chunk size per
  `cuMemCreate`. Default 0 (chunk = request size, rounded up to
  granularity; matches the weight server's per-range allocation).
  Override only if the driver's per-process mapping count becomes a
  concern.
- Hard gated off when `DS4_CUDA_WEIGHT_IPC_MANIFEST` is set, because
  the sidecar already provides identical VMM ranges and running both
  would double-allocate the model.
