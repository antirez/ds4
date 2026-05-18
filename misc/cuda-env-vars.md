# CUDA backend env-var reference

The CUDA backend (`ds4_cuda.cu`) mirrors `ds4_metal.m`'s role on NVIDIA. It
dispatches Q8_0 dense matmuls through one of three kernel families, has its
own n_tok=1 mmvq decode path, optionally captures the decode-block kernel
sequence into a `cudaGraphExec_t`, and can either allocate weight memory
in-process (default) or import it from the `ds4_weight_server` sidecar.

Every CUDA-specific env var is below, with the intent behind each default.

## Q8_0 dispatcher

cuBLAS is initialised unconditionally at backend startup regardless of the
selected strategy: on sm_121 we observed that this triggers CUDA driver state
making mmq ~4&times; faster than a binary that skips `cublasCreate`, so the
cublas path stays resident even when not selected.

| Strategy | When picked                                       | What it runs                                       |
|----------|---------------------------------------------------|----------------------------------------------------|
| `mmq`    | default; every CUDA arch we've validated          | vendored llama.cpp `mul_mat_q` (`cuda/mmq/`)       |
| `cublas` | explicit override or fallback if mmq init fails   | `cuda_q8_f16_ptr` Q8&rarr;FP16 cache + `cublasGemmEx` |
| `warp8`  | explicit override or last-resort fallback         | native `matmul_q8_0_preq_*_kernel` family          |

Logged once on first dispatch, e.g.:

    ds4: CUDA Q8_0 dispatch: mmq (sm_120, 1792 GB/s memory bandwidth) [default]

The bandwidth figure is informational; we don't tier on it.

## Env-var inventory

- `DS4_CUDA_PREFILL_PATH=mmq|cublas|warp8|auto` (default `auto` &rarr; mmq).
  Explicit override. `auto` and unset both resolve to mmq.

- `DS4_CUDA_USE_MMQ=0` (legacy alias). Equivalent to
  `DS4_CUDA_PREFILL_PATH=cublas`. The newer variable takes precedence.

- `DS4_CUDA_MMQ_MOE_MIN_TOKENS=N` (default 2). Minimum `n_tokens` at which
  the routed-MoE mmq path activates. At n=1 mmq's matrix-matrix-shaped path
  has higher per-launch cost than the vector path; that case is handled by
  the mmvq decode branch.

- `DS4_CUDA_MMQ_X_MAX=N`. Clip `get_mmq_x_max_host` to N (rounded down to a
  multiple of 8) when sweeping tile widths. Diagnostic only; the vanilla
  128 wins on sm_120.

- `DS4_CUDA_NO_MMVQ_DECODE`. Opt-out of the vendored `mul_mat_vec_q` decode
  path. mmvq is structurally optimal for n_tok=1 routed-MoE and dense
  attention projection (one block per output row, no column-tile waste).
  Wires into `routed_moe_launch` and `cuda_matmul_q8_0_tensor_labeled`.

- `DS4_CUDA_MMVQ_DECODE_MAX_TOKENS=N` (default 1). Cap on n_tokens routed
  through the mmvq decode branch in `routed_moe_launch`. Range 0&ndash;8;
  0 disables. Values 2&ndash;8 extend mmvq coverage to short-prefill
  batches but require `n_tokens * n_expert_used &le; 8` for the down
  matmul.

- `DS4_CUDA_MOE_GRAPHS=0` (default on). Opt-out of CUDA Graph
  capture+replay around the mmvq routed-MoE decode block and the n_tok=1
  dense Q8_0 vec path. Each captured launch is bracketed by
  `cudaEventRecord` / `cudaStreamWaitEvent` so g_moe_stream and stream=0
  stay correctly ordered across the boundary.

- `DS4_CUDA_MTP_VERIFIER_USE_MMQ` (default 0). Bisection switch. Normally
  `ds4.c` brackets every MTP verifier call with
  `ds4_gpu_set_mtp_verifier(1/0)` and the CUDA backend routes Q8_0
  matmuls onto `warp8` for the duration. mmq's stream-k + MMA FP32
  reduction order drifts ~1 ULP/layer from warp8; the drafter is trained
  against legacy decoding so an mmq verifier flips tight-margin tokens
  (0/314 acceptance on GB10 with mmq verifier active). Set to 1 to
  reproduce the broken behavior for bisection.

## In-process VMM weight arena

The arena allocates each weight tensor in its own CUDA Driver VMM
region (`cuMemCreate` &rarr; `cuMemAddressReserve` &rarr; `cuMemMap`
&rarr; `cuMemSetAccess`), giving every tensor its own
2&nbsp;MiB-aligned virtual address.  This matches what the
out-of-process `ds4_weight_server` provides imported workers.  On
discrete GPUs this is worth roughly 2&times; prefill; on integrated
GPUs it's neutral-to-positive.

### Why per-tensor chunks specifically

The chunk-size bisect we ran during development clarified the
mechanism.  VMM with one large chunk (e.g.
`DS4_CUDA_VMM_ARENA_CHUNK_MB=1792`) performs identically to the
cudaMalloc-backed arena (~1080 t/s prefill on PRO 6000), even though
the underlying memory is still 2&nbsp;MiB-paged.  The actual
differentiator is **per-tensor 2&nbsp;MiB-aligned base addresses**:
when each weight tensor sits at its own fresh
`cuMemAddressReserve`-handed VA, matmul kernels' tile-load coalescing
and L2 spatial-locality patterns improve enough to roughly double
prefill.  Pack the same VMM-paged memory into one big chunk and the
bases land at sub-granularity offsets &mdash; the perf advantage
disappears.

This also unifies cleanly with the drift below: same root cause, two
effects you cannot separate.

### Known trade-off: FP32 reduction-order drift vs official vectors

Per-tensor VMM-allocated weight ranges produce a small but real
**reduction-order drift** in the matmul kernels relative to the
cudaMalloc-backed arena.  The same cache/tile-arrival-order behavior
that gives the 2&times; perf win also changes the order in which tile
partial sums reach the FP32 accumulator; FP32 is non-associative, so
the order matters.  This is structural to the kernels' parallel
reduction strategy, not a misuse of the API.

Investigation established:

1. The uploaded weight bytes are byte-identical between the two
   allocators (verified by post-upload checksum of all 138 weight
   ranges).
2. Kernels do not read past tensor bounds (verified by poisoning the
   chunk tail with 0xAB instead of zero &mdash; output unchanged).
3. The drift is shared by both the vendored mmq family and the legacy
   `warp8` native kernels and is therefore upstream of the Q8_0
   dispatcher.  Same drift on PRO 6000 sm_120 and GB10 sm_121.
4. Logit-level magnitude is small (~0.08 logprob units at step 0)
   &mdash; bounded, deterministic, of the same shape as the documented
   mmq-vs-warp8 ULP-per-layer drift behind `DS4_CUDA_MTP_VERIFIER_USE_MMQ`
   (Option D).  Most tokens are unaffected; only tight-margin choices
   flip.

**Observable cost:** in `./ds4_test --logprob-vectors`, one of four
test vectors (`short_code_completion`, step 1: the `c` language tag
after triple-backticks) flips to a textually-equivalent but
byte-different alternative under the VMM-arena default.  The other
seven failures in that test family are pre-existing on the CUDA
backend and reproduce identically with `DS4_CUDA_VMM_ARENA=0`.

**Workaround for users who need official-vector byte equivalence:**
set `DS4_CUDA_VMM_ARENA=0` to use the cudaMalloc-backed arena.  Prefill
ceiling drops by ~50% on discrete GPUs in exchange for the parity.

### Env vars

- `DS4_CUDA_VMM_ARENA=0`. Disable; fall back to the cudaMalloc-backed
  arena.  Also the workaround for the reduction-order drift above.

- `DS4_CUDA_VMM_ARENA_CHUNK_MB=N`. Minimum chunk size per `cuMemCreate`.
  Default 0 (chunk = request size, rounded up to the driver-reported
  granularity; matches the weight server's per-range allocation).
  Values 1024+ collapse the per-tensor placement and forfeit the perf
  benefit; useful only for bisection.

- `DS4_CUDA_WEIGHT_IPC_MANIFEST=/path/to/manifest.json`. Worker-side
  import path for weights owned by `ds4_weight_server`. When set, the
  in-process VMM arena is hard-gated off because the sidecar already
  provides identical VMM ranges and running both would double-allocate
  the model. See `misc/proof-harness/README.md` for the sidecar
  lifecycle.
