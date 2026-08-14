---
format: patchmd/v0.1
id: laguna-s21-cuda
status: ready
kind: persistent-customization
upstream:
    base: 7e3dbef7e336433f487c172a3308e26b39fa75a3
relationships:
    depends_on: []
    conflicts_with: []
authors:
    - name: Unknown
license: MIT
---
# Intent

Add a native CUDA inference backend for the Laguna S 2.1 model already
supported by the Metal backend.

# Motivation

Laguna S 2.1 currently has model loading and a Metal execution path, but
cannot run end-to-end on NVIDIA GPUs through ds4's CUDA backend.

# Required behavior

- A CUDA build must load a supported Laguna S 2.1 GGUF and execute its
  model-specific forward path on NVIDIA GPUs.
- CUDA inference must implement the same Laguna S 2.1 model mechanics and
  tensor semantics as the existing Metal reference path.
- Existing CUDA model backends must continue to compile and retain their
  existing dispatch behavior.

# Invariants

- Laguna S 2.1 tensor validation, tokenizer/chat behavior, and public
  CLI/server APIs remain shared with the existing model implementation.
- Metal, CPU/reference, ROCm, SSD-streaming, and distributed paths are not
  changed unless a shared correction is required for CUDA correctness.
- Correctness takes priority over a faster CUDA path with unexplained logits,
  attention, or KV-cache drift.

# Non-goals

- Redesigning Laguna S 2.1 architecture or its GGUF layout.
- Introducing a generic model runtime or C++.
- Optimizing unrelated model backends.
- Adding DFlash or another speculative decoder before raw Laguna prefill and
  decode performance are competitive.

# Assumptions

- The `origin/laguna-s2.1` branch at
  `7e3dbef7e336433f487c172a3308e26b39fa75a3` is the authoritative Metal
  reference and integration base.
- Existing CUDA kernels and graph orchestration for other supported models
  should be reused where their tensor semantics match Laguna S 2.1.
- The attached CUDA host is a single NVIDIA GB10 (`sm_121`). Laguna remains
  full-residency and single-GPU; multi-GPU placement, tensor parallelism, and
  SSD streaming stay explicitly unsupported.

# Open questions

None.

# Acceptance

- Build the CUDA targets with the repository's supported `make cuda-*`
  workflow.
- Run focused CUDA unit/regression tests that do not require the full model.
- With a Laguna S 2.1 GGUF, run a deterministic prompt or logits comparison
  against a trusted backend and exercise decode plus multi-token prefill.
- Report sustained performance with at least 2K, 4K, and 8K prompt contexts
  and 256 generated tokens per point. Short smoke/profile runs are diagnostic
  evidence only, not representative throughput claims.
- Inspect model dispatch and compile/test existing CUDA paths to guard against
  regressions.

# Adaptation guidance

Keep model mechanics in the shared graph code and expose only narrow,
backend-neutral GPU primitives in `ds4_gpu.h`. Implement CUDA kernels and
launch wrappers in `ds4_cuda.cu`, following existing CUDA error handling,
stream, tensor, and multi-GPU conventions. Prefer reusable primitives over
Laguna-only kernels where semantics are identical.

The Laguna graph already exposes the required backend-neutral hooks. CUDA must
replace its current no-op Laguna stubs and add the quantized primitives that
the official GGUFs need: Q8_0 signal-path operations for the revised recipe,
legacy Q4_K/Q6_K dense operations, per-head RMSNorm plus prefix NeoX/YaRN
RoPE, f16 ring-cache GQA attention with the learned softplus gate, and Q4_K
routed gate/up/down projections. The earlier recipe's Q6_K routed down
projections remain supported. Decode and multi-token prefill both require
coverage.

# Decisions

- Keep the shared Laguna graph as the sole owner of model scheduling. CUDA
  implements the graph's existing hooks and extends generic Q4_K/Q6_K tensor
  operations rather than adding a parallel CUDA-only model graph.
- Route both the revised Q4_K/Q4_K/Q4_K experts and the legacy
  Q4_K/Q4_K/Q6_K experts through the same Laguna CUDA MoE implementation.
  The revised graph keeps its shared expert on the Q8_0 matmul path.
- Preserve the revised checkpoint's signal-path accuracy during prefill.
  Metal multiplies Q8_0 weights by floating-point activations, whereas CUDA's
  generic Q8_0 path requantizes activations to Q8_0. Repeating that extra
  quantization across 48 Laguna layers corrupts generation. Revised Laguna
  therefore expands Q8_0 signal weights to a bounded F16 cache and uses
  cuBLAS for multi-token projections. If the cache budget is exhausted, it
  streams one expanded matrix through reusable scratch rather than falling
  back to activation requantization. Decode retains the native Q8_0 kernels
  after the accurate prefill state has been established.
- Use direct quantized F32-reduction kernels for decode and quality mode.
  Normal multi-token Q4_K/Q6_K matmul dequantizes one matrix at a time into a
  reusable f16 scratch slab and uses cuBLAS, avoiding persistent dequantized
  copies of the approximately 70 GiB model.
- Preserve f16 KV-cache storage and online-softmax attention semantics from
  Metal. Prefill stages the current chunk before attention and commits it to
  the ring only after all causal queries finish, so sliding-window wraparound
  cannot overwrite keys still needed by early queries.
- Reuse CUDA's Q8_K activation quantization for Laguna MoE gate/up and down
  dot products. Routed weights are fused into the intermediate values; each
  output row then reduces selected experts in slot order.
- Optimize the raw Laguna path first. Reuse generic CUDA expert sorting as a
  reference, but specialize execution for Laguna's top-10 routing, Q4_K/Q6_K
  down projections, and 1024-wide expert intermediates instead of masking
  direct-path performance with speculative decoding.
- Treat other model backends as references, not constraints. Prefer a
  Laguna-specific kernel or schedule when its top-10 routing, alternating
  Q4_K/Q6_K down weights, 1024-wide experts, or shared-expert structure allows
  a measurably better implementation without weakening correctness.
- For multi-token sparse MoE, sort routed pairs by expert and evaluate eight
  pairs per weight row. Process each down projection in 1024-row chunks,
  reusing the already-quantized mid buffer for pair terms and reducing top-10
  slots in their original order. This covers both Q4_K and Q6_K down layers
  without atomics or a full token-by-slot-by-output scratch allocation.
- During prefill, evaluate the three or six query heads sharing a Laguna KV
  head in one CUDA block. Preserve each head's original 128-thread reduction
  tree and causal key order while loading the common K/V row once per group.
  Keep decode head-parallel: grouping decode heads reduced block-level
  parallelism and lost throughput at every sustained frontier.
- For decode histories longer than 256 keys, retain one block per query head
  and stripe keys over eight warps. Merge the eight online-softmax partials in
  shared memory. This increases parallelism without repeating the rejected
  grouped-head schedule; `DS4_CUDA_LAGUNA_NO_SPLIT_DECODE=1` retains the
  original serial-history kernel.
- On Blackwell (`sm_120+`), use sixteen warps for histories of at least 2K.
  The wider schedule is capability- and shape-gated; either
  `DS4_CUDA_LAGUNA_NO_BLACKWELL=1` or
  `DS4_CUDA_LAGUNA_NO_BLACKWELL_SPLIT16=1` restores the portable eight-warp
  path. Both schedules explicitly broadcast each warp's complete query/key
  dot product before the online-softmax update.
- Use aligned 16-byte Q4_K loads in Laguna's one-token gate/up and Q4_K-down
  kernels, guarded by tensor stride checks and
  `DS4_CUDA_LAGUNA_NO_VEC_Q4=1`. Convert the normalized attention activation
  to f16 once and reuse it across the four cuBLAS Q/K/V/gate projections;
  `DS4_CUDA_LAGUNA_NO_SHARED_QKVG_ACTIVATION=1` restores the generic calls.
- Finish portable raw-kernel optimization before adding hardware-specific
  paths. Any later Blackwell/GB10 specialization must be capability- and
  shape-gated, retain a rollback switch and portable fallback, preserve the
  validated logits contract, and demonstrate a measured win on supported
  hardware.

# Provenance

Explicitly requested:

- Create a CUDA backend for Laguna S 2.1.
- Use the existing Metal implementation for this model and CUDA kernels for
  other models as references.

Inferred from repository conventions:

- Preserve the existing shared Laguna graph rather than duplicating model
  scheduling inside the CUDA runtime.
- Add synthetic low-level CUDA numeric tests because the full Laguna GGUF
  cannot fit on the currently available filesystem.

Verification evidence:

- `make cuda-spark -j2` passes with CUDA 13.0 on the GB10, building the CLI,
  server, benchmark, evaluator, and agent.
- `make cpu -j2` passes, guarding the shared graph and backend dispatch edits.
- `make cuda-regression` passes both the existing long-context CUDA test and
  the new Laguna test.
- `tests/cuda_laguna_smoke` passes patterned Q4_K/Q6_K batch and decode
  matmuls, Q4_K embedding, YaRN and SWA-compatible head RMSNorm/RoPE, gated
  GQA prefill/decode including ring wraparound, and decode/batch routed/shared
  MoE with both revised Q4_K and legacy Q6_K down projections.
- The current pinned 68,248,759,648-byte Poolside GGUF contains 141 Q4_K,
  386 Q8_0, and 287 F32 tensors. It loads successfully and runs end-to-end on
  the GB10 CUDA backend. The earlier 75,173,103,200-byte recipe with 239
  Q4_K, 48 Q6_K, 240 F16, and 287 F32 tensors remains accepted.
- A deterministic 128-token revised-weight generation remains coherent
  through the full sample. The default no-thinking smoke answers the
  one-sentence sky-color prompt directly and coherently. The same GGUF also
  generates coherent text in Poolside's Laguna llama.cpp branch, confirming
  the downloaded checkpoint and tokenizer are intact.
- On the revised weights, the sustained raw sweep with 256 generated tokens
  per point measures 201.87/22.36 tok/s prefill/generation at 2K context,
  160.93/21.64 at 4K, and 132.42/20.33 at 8K. Results are stored in
  `speed-bench/laguna_s21_gb10_revised.csv`.
- Before the accurate-prefill correction, the same revised-weight sweep
  appeared to measure 137.22/21.83, 113.51/21.84, and 100.62/20.46 tok/s,
  but deterministic generation was corrupt. Those invalid measurements are
  retained only as diagnostic history and are not published as results.
- On the legacy weights, the original sustained single-process benchmark with
  256 generated tokens
  per point measured 48.05/9.06 tok/s prefill/generation at 2K context,
  42.57/7.29 at 4K, and 36.79/5.23 at 8K. With expert-sorted MoE, chunked
  Q4_K/Q6_K down, grouped prefill GQA, split-history decode attention, aligned
  Q4_K decode loads, and a shared Q/K/V/gate f16 activation, the same sweep
  measures 148.08/15.25, 125.52/15.25, and 106.95/14.56 tok/s. Relative to
  the original baseline, prefill improves by 208%, 195%, and 191%, while
  generation improves by 68%, 109%, and 178%. Results are stored in
  `speed-bench/laguna_s21_gb10.csv`.
- Initial Nsight Systems profiling attributed 89.7% of 512-token prefill GPU
  time to routed experts: 60.7% to Q4_K gate/up and 29.0% to Q4_K/Q6_K down.
  After expert sorting and chunked down, a follow-up profile measured 19.3%
  gate/up, 38.0% Q6_K down, 7.4% Q4_K down, and 31.4% causal attention.
  This motivated grouped prefill GQA, which raises sustained 8K prefill from
  69.65 to 107.50 tok/s by avoiding repeated K/V loads.
- A decode-focused profile after split-history attention attributes about
  44% of decode GPU time to the five f16 cuBLAS projections, 19% to Q4_K
  gate/up, 12% to MoE down, 8% to split attention, and 6% to the Q6_K output
  matmul. The existing direct CUDA f16 matvec was rejected after lowering a
  2K/64-token diagnostic from 15.19 to 12.06 tok/s.
- In a paired 256-token GB10 comparison, Blackwell's sixteen-warp attention
  raises revised-weight steady generation from 21.84 to 22.49 tok/s at 2K,
  20.49 to 21.77 at 4K, and 18.37 to 20.44 at 8K over the portable
  eight-warp schedule: gains of 3.0%, 6.2%, and 11.3%. The paired control is
  stored in `speed-bench/laguna_s21_gb10_revised_portable.csv`. At a 4K
  prompt it also reproduces the scalar attention path's complete 32-token
  greedy continuation.
- The optimized and rollback paths produce byte-identical full-model logits
  for sorted gate/up, chunked Q4_K/Q6_K down, grouped prefill GQA, aligned
  Q4_K decode loads, and the shared Q/K/V/gate activation. The synthetic CUDA
  regression compares every output element from the scalar, portable
  eight-warp, and default (sixteen-warp on GB10) attention paths at the 2,048
  key activation threshold and across 4,096 keys within a 2e-4
  absolute/relative tolerance.
  A grouped decode experiment was rejected despite identical logits because
  it regressed sustained generation to 8.10/6.17/4.17 tok/s.
- `sh -n download_model.sh` passes. The Laguna download target accepts the
  pinned model's exact byte size and rejects an existing file with a different
  size before invoking the Hugging Face client.
- The broader `make test -j2` run completed its long-context, tensor
  equivalence, local-golden, server, and other checks, but returned six
  model-dependent tool-call/golden/kernel failures on this host. None arose
  from the focused Laguna regression; this non-green aggregate result is
  recorded as a validation exception rather than hidden.
- Full-model revised-weight prefill and generation complete successfully at
  all three sustained benchmark frontiers.

# Reference realization

The reference realization is based on upstream 7e3dbef7e336433f487c172a3308e26b39fa75a3 and stored in `reference.patch`.
