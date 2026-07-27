# Agent Notes

Pulsar (`pulsar`) is a DeepSeek V4 Flash specific inference engine, not a
generic GGUF runner. This tree is the **CUDA-only fork** of antirez's upstream
project, targeting the NVIDIA DGX Spark (GB10, `sm_121`, ~121 GB usable
unified memory). The Metal, ROCm, and CPU inference backends were fully
removed; `cuda` is the only backend. The engine still contains shared
host-side C math (`attention.c`, `hc.c`, `layers.c`, `moe.c`, `experts.c`)
that the live GPU path calls — it is not dead CPU-backend code.

## Goals

- Keep the production path as whole-model CUDA graph inference on GB10.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF. The model
  must be fully resident: a GGUF that does not fit is rejected at load, never
  partially streamed.
- Preserve correctness before speed. Do not keep a faster path with
  unexplained attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Keep the implementation small, sharp, easy to understand. Don't introduce
  slop: fragile case-patching, dead code, or needless complexity.
- Comment inference code where model mechanics, cache lifetime, memory policy,
  or API orchestration are not obvious locally. Prefer comments beside the
  implementation over separate design documents.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- C++ is confined to the single CUTLASS TU (`src/cuda/pulsar_mxfp4_cutlass.cu`).
  Everything else is C (or CUDA C).

## Layout

Public headers: `src/pulsar.h` (engine API) and `src/pulsar_gpu.h` (GPU graph API).

- `src/engine/` — 21 TUs + `pulsar_engine_internal.h`: GGUF parsing, tokenizer,
  weight binder (`weights.c`), quant format/kernel tables, GPU graph
  orchestration (`gpu_graph_alloc.c`, `gpu_graph_state.c`, `gpu_prefill.c`,
  `gpu_decode.c`, `gpu_diag.c`), sessions + KV payload serialization, imatrix
  collection, steering, and the shared host math TUs.
- `src/cuda/` — 7 kernel TUs + `pulsar_cuda_internal.h`: runtime/memory
  (`pulsar_cuda_runtime.cu`), matmuls incl. the cuBLASLt MXFP8 path
  (`pulsar_cuda_matmul.cu`), attention, MoE, indexer, norm/KV, HC router; plus
  `pulsar_mxfp4_cutlass.cu` (CUTLASS grouped GEMM for MXFP4 experts) and
  `pulsar_iq2_tables_cuda.inc`.
- `src/server/` — 14 TUs + `pulsar_server_internal.h`: HTTP server,
  OpenAI/Responses/Anthropic endpoints and streaming, prompt rendering, disk
  KV cache, exact-DSML tool replay.
- `src/agent/` — 20 TUs + `pulsar_agent_internal.h`: native coding agent (tools,
  terminal UI, sessions, compaction).
- `src/cli/` — `pulsar_cli.c`, `pulsar_bench.c`, `pulsar_eval.c` entry points.
- `src/lib/` — shared pieces: help text, kvstore.
- `src/vendor/` — linenoise, rax.
- `tests/` — C runners; `pulsar_test.c` and `pulsar_agent_test.c` are unity builds
  that `#include` the server/agent source lists.
- `cutlass/` — git submodule (v4.5.2), **required** for the MXFP4 expert path.
- `gguf-tools/` — offline quantization/imatrix tooling that produces the GGUFs
  this fork loads.

Internal-header convention: a symbol is de-static'd and declared in the
module's `pulsar_*_internal.h` only when another TU of the same module needs it.
Everything else stays `static`. The cross-module surface is `src/pulsar.h`,
`src/pulsar_gpu.h`, and the `src/lib/*.h` headers only.

## Build

```sh
git submodule update --init cutlass   # once
make cuda-spark          # DGX Spark / GB10 (CUDA_ARCH=sm_120f)
make cuda-generic        # other local CUDA GPUs (CUDA_ARCH=native)
make cuda CUDA_ARCH=sm_N # explicit -arch, e.g. cross-builds
```

Binaries land in the repo root: `pulsar`, `pulsar-server`, `pulsar-agent`, `pulsar-bench`,
`pulsar-eval`. Note `pulsar_mxfp4_cutlass.cu` needs the `sm_120f` family for the
mxf4 block-scale MMA; the Makefile handles its flags.

## Testing

- `make test` runs `./pulsar-eval --self-test-extractors`, `./pulsar_agent_test`,
  and `./pulsar_test`. The eval self-test and agent test need no model;
  `pulsar_test` loads a model (`DS4_TEST_MODEL`, default `./ds4flash.gguf`).
- `pulsar_test` distinguishes **gating** internal-correctness tests (any failure
  fails the suite) from **informational** ones: `logprob-vectors` compares
  the 2-bit production model against full-precision official-API logprobs (a
  drift dashboard, expected mismatches), and `think-tool-recovery` is
  run-to-run flaky from the nondeterministic batched-prefill float-atomic
  down-sum. Informational mismatches are reported but exit 0.
- `make cuda-regression` runs `tests/cuda_long_context_smoke`: GPU kernel
  smoke tests, no model required.
- `./pulsar_test --logprob-vectors` compares against official-API vectors and
  pins `DS4_CUDA_PREFILL_CHUNK=2048`.
- imatrix collection (`--imatrix-dataset` / `--imatrix-out`) requires `--cuda`.

## Validation Culture

- **Refactors** (code movement, TU splits, renames) must produce byte-identical
  greedy output versus the pre-change binary.
- **Numerics changes** (kernels, quantization, activation formats) are
  perplexity-gated (`--perplexity-file`) plus the deterministic `pulsar-eval`
  q1..q4 token-count gate; recent reference point on the 97 GB zero-Q8 Flash
  oracle: ppl 7.3216, decode 12.35 t/s, long-prompt prefill 162 t/s.
- **One pulsar process at a time on the GB10.** Two ~97 GB model mappings OOM the
  box; the instance lock (`DS4_LOCK_FILE`) is intentional.

## Supported Weight Formats (binder-enforced)

| Tensor group | Accepted formats |
| --- | --- |
| Attention projections, shared experts | MXFP8 (FP8 E4M3 + per-32 E8M0 scales) |
| Routed experts gate/up/down | exactly `IQ2_XXS`/`IQ2_XXS`/`Q2_K`, or all three `MXFP4` |
| Output head | `BF16` or MXFP8 |
| Norms, embeddings, indexer, HC | `F32`/`F16` |

Legacy `Q4_K`/`Q8_0` weights are rejected at load with one clear error
(`weights_reject_unsupported_types` in `src/engine/weights.c`). `Q8_K` exists
only as *activation* quantization inside the routed-expert (MoE) kernels.

## Compute Paths

- **Prefill:** dense matmuls run cuBLASLt block-scaled MXFP8×MXFP8
  (`VEC32_UE8M0`) on tensor cores, including the attention-output projections.
  MXFP4 expert prefill runs the CUTLASS mxf4×mxf4 block-scaled grouped GEMM
  (GB10 / `sm_120f` family).
- **Decode:** custom fused kernels, memory-bound; activations stay raw f32
  except inside MoE (Q8_K).
- **FP8 KV cache** is currently fake-quant: E4M3 rounding with f32 storage.

## Environment Variables

All runtime tuning/diagnostic gates use the `DS4_CUDA_*` prefix (this fork
renamed every `DS4_METAL_*` gate; there are no compatibility aliases).
Also: `DS4_FP8_NO_MXCORE`, `DS4_TEST_MODEL`, `DS4_LOCK_FILE`, `DS4_GGUF_DIR`
(download script).

## Deferred Work

- Real packed FP8 KV-cache storage (replace the fake-quant f32 path).
- Move MoE decode off Q8_K activation quantization.
