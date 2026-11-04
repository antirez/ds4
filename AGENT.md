# Agent Notes

`ds4.c` is a DeepSeek V4 Flash/Pro specific inference engine; the model shape
(Flash or Pro) is selected from the GGUF at `ds4_engine_open()` time. It is not
a generic GGUF runner. The goal is a small, readable, high-performance C
codebase with Objective-C only where Metal requires it and Metal kernels under
`metal/`.

## Goals

- Keep the production path as whole-model Metal graph inference.
- Always make sure that the SSD streaming, CUDA, ROCm, distributed inference, Metal default inference are not affected by fixes to other parts of the code.
- Keep model loading mmap-backed for the Metal default case; do not eagerly copy the full GGUF. Keep the model loading for SSD streaming of routed experts explicit: allocated buffers, fast reads from disk, always try to hide loading of missing routed experts by loading them while performing the inference of the shared expert and routed experts already in RAM. Always try to hide loading of layers for prefill in SSD streaming mode using the inference time of the current layer as the next one is loaded.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV checkpoints.
- Keep MTP speculative decoding (`--mtp`, `--mtp-draft`) correct on every backend that supports it: speculative acceptance must never change the tokens a non-speculative run would produce.

## Quality Rules

- Keep the implementation small, sharp, easy to understand. Try to write elegant code in a state of grace. Don't settle for the first thing that comes to mind, try to find the most minimal and better working design. Don't introduce slop: very fragile code that just patches specific cases, dead code, useless code and code ways more complicated of how it should be.
- Comment important inference code where the model mechanics, cache lifetime, memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Never run the CPU inference path on macOS: it has exposed a kernel VM bug with very large mappings that can crash the machine and require a reboot. On macOS use `make cpu` only to confirm the CPU build still compiles.
- Do not run multiple huge model processes concurrently. The instance lock is intentional.

## Layout

The build produces five binaries: `ds4` (CLI), `ds4-server`, `ds4-bench`,
`ds4-eval`, `ds4-agent`.

- `ds4.c`: model loading, tokenizer, CPU reference code, Metal graph scheduling,
  sessions, disk-cache payload serialization.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: compute kernels.
- `ds4_cuda.cu` / `ds4_rocm.cu` (+ `ds4_gpu.h`, `rocm/*.cuh`): CUDA / ROCm backends.
- `ds4_agent.c` (+ `ds4_web.c`): integrated coding agent (alpha).
- `ds4_distributed.c`: transformer layer slices split across machines over TCP.
- `ds4_ssd.c`: SSD streaming of routed MoE experts.
- `ds4_kvstore.c` (+ `rax.c`): disk KV cache store keyed by token prefix (radix tree).
- `ds4_bench.c` / `ds4_eval.c`: throughput benchmarking and eval/grading harnesses.
- `gguf-tools/`: offline GGUF generation, imatrix collection, quantization,
  quality scoring.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material.

This list is not complete, check the files for more info.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and Metal are available. `ds4_test` also exposes focused sub-checks
(`--server`, `--logprob-vectors`, `--long-context`, `--tool-call-quality`,
`--metal-kernels`); see `CONTRIBUTING.md` for when to run each. Use live server
tests only when intentionally testing the API surface.

At every major change where one of the following could be affected, make sure to:

1. Test the normal Metal path and that speed is still at the level it was.
2. Test the SSD streaming path.
3. Test the distributed inference if it could be affected, but ask the user before doing so.
4. Check if CUDA or ROCm could be broken after the change, and ask the user to give you access to the relevant machine to actually test if everything is still fine.
