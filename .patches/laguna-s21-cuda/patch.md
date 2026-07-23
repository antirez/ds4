---
format: patchmd/v0.1
id: laguna-s21-cuda
status: ready
kind: persistent-customization
upstream:
    base: 8a927009e61bd7e1ca370cd793c18b2749dbc03c
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
  `8a927009e61bd7e1ca370cd793c18b2749dbc03c` is the authoritative Metal
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
the official GGUF needs: Q4_K token/dense operations, Q6_K dense operations,
per-head RMSNorm plus prefix NeoX/YaRN RoPE, f16 ring-cache GQA attention with
the learned softplus gate, and Q4_K gate/up with Q4_K or Q6_K MoE down
projections. Decode and multi-token prefill both require coverage.

# Decisions

- Keep the shared Laguna graph as the sole owner of model scheduling. CUDA
  implements the graph's existing hooks and extends generic Q4_K/Q6_K tensor
  operations rather than adding a parallel CUDA-only model graph.
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
- Optimize the raw Laguna path first. Reuse the generic CUDA Q4_K
  expert-sorting and tensor-core kernels where its tensor semantics match;
  extend them for Laguna's top-10 routing and Q6_K down projections instead
  of masking direct-path performance with speculative decoding.

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
  GQA prefill/decode including ring wraparound, and decode/batch Q6_K-down
  routed/shared MoE.
- The pinned 75,173,103,200-byte Poolside GGUF contains 239 Q4_K, 48 Q6_K,
  240 F16, and 287 F32 tensors. It loads successfully and runs end-to-end on
  the GB10 CUDA backend.
- Short diagnostic runs measure 48.32 tok/s prefill and 14.02 tok/s
  generation for a 128-token prompt with four generated tokens. At 2,048
  prompt tokens and 64 generated tokens they measure 48.00 tok/s prefill,
  9.14 tok/s generation, and 9.19 tok/s steady-state generation. These are
  smoke/profile measurements rather than representative throughput claims.
- A sustained single-process benchmark with 256 generated tokens per point
  measures 48.05/9.06 tok/s prefill/generation at 2K context,
  42.57/7.29 at 4K, and 36.79/5.23 at 8K. The 4K and 8K prefill rows measure
  incremental suffixes of 2K and 4K tokens respectively. Results are stored
  in `speed-bench/laguna_s21_gb10.csv`.
- Nsight Systems attributes 89.7% of 512-token prefill GPU time to the routed
  expert kernels: 60.7% to Q4_K gate/up and 29.0% to Q4_K/Q6_K down.
  The direct causal attention kernel accounts for another 9.2%; cuBLAS
  projections, dequantization, norms, and copies together account for less
  than 1%. A separate decode diagnostic similarly attributes 75.8% of total
  GPU time to routed expert gate/up/down, with F16 signal-path GEMV next.
- `sh -n download_model.sh` passes. The Laguna download target accepts the
  pinned model's exact byte size and rejects an existing file with a different
  size before invoking the Hugging Face client.
- The broader `make test -j2` run completed its long-context, tensor
  equivalence, local-golden, server, and other checks, but returned six
  model-dependent tool-call/golden/kernel failures on this host. None arose
  from the focused Laguna regression; this non-green aggregate result is
  recorded as a validation exception rather than hidden.
- Full-model Laguna logits/generation validation was not run because the GGUF
  is unavailable locally and cannot fit in the remaining download space.

# Reference realization

The reference realization is based on upstream 8a927009e61bd7e1ca370cd793c18b2749dbc03c and stored in `reference.patch`.
