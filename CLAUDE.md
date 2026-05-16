# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

DwarfStar 4 (DS4) is a single-model native inference engine for **DeepSeek V4 Flash** specifically. It is not a generic GGUF runner. The codebase is intentionally narrow: one model, three production backends (Metal on macOS, CUDA on Linux/Windows, CPU for diagnostics only), and one HTTP server that speaks OpenAI/Anthropic/Responses wire formats. See `AGENT.md` for the project's quality rules — particularly **no C++ in the main path** and **do not add permanent semantic variants behind flags**.

## Build commands

Two parallel build systems coexist; do NOT reach for one when the other is canonical for the platform:

| Platform | Command | Notes |
|---|---|---|
| macOS | `make` | Default Metal build |
| Linux CUDA | `make cuda-spark` (DGX Spark/GB10) or `make cuda-generic` | `make cuda CUDA_ARCH=sm_120` for explicit arch |
| Linux/macOS CPU | `make cpu` | Diagnostics only — never the production target |
| Windows MSVC + CUDA | `cmake -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build --config Release` | Requires CUDA Toolkit ≥ 12.8 (sm_120 Blackwell) and Windows 10 ≥ build 19041. Builds `ds4-server`, `ds4-bench`, `ds4-eval` only — `ds4` CLI deferred (needs linenoise port to Console API). |

The Makefile and CMakeLists.txt are independent. The Makefile path is unchanged by the Windows port; all Windows-specific code is `#ifdef _WIN32`-gated.

## Tests

```sh
make test                      # builds ds4_test and runs --all
./ds4_test --server            # API + prompt rendering + KV bookkeeping
./ds4_test --logprob-vectors   # token-level diff vs official continuations
./ds4_test --long-context      # long-context fact recall regression
./ds4_test --tool-call-quality # live DSML emission
./ds4_test --metal-kernels     # numeric kernel checks
./ds4_test --cuda-regression   # CUDA-only smoke (CUDA build only)
```

Default model path is `./ds4flash.gguf`. Override with `DS4_TEST_MODEL=/path/to.gguf`. See `CONTRIBUTING.md` before opening a PR; speed regressions must include before/after `ds4-bench` CSVs from the same machine and backend.

## Architecture

### Three binaries share one core

`ds4.c` contains the engine: GGUF loader (mmap-based), tokenizer, CPU reference, Metal/CUDA graph scheduling, sessions, and disk-cache payload serialization. The three production binaries link against it:

- `ds4-server` (`ds4_server.c`) — HTTP server with `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, `/v1/completions`. Request parsing runs in client threads; **inference is serialized through one graph worker** that owns the live `ds4_session` and KV checkpoint. There is no batching across requests.
- `ds4-bench` (`ds4_bench.c`) — measures *instantaneous* prefill/generation throughput at context frontiers (not whole-run averages); writes CSV.
- `ds4-eval` (`ds4_eval.c`) — split-screen TUI that runs an embedded 75-question GPQA/SuperGPQA/AIME mix against the live model.
- `ds4` (`ds4_cli.c` + `linenoise.c`) — interactive REPL. NOT built on Windows.

### Public boundary

`ds4.h` is the only header the CLI/server should depend on. Never let HTTP/CLI code touch tensor internals — that's the rule the file's header comment enforces. Backends are selected at runtime via `ds4_backend` (METAL, CUDA, CPU).

### KV cache is first-class disk state

This is the project's defining design choice. The compressed DeepSeek V4 KV cache plus modern SSDs make on-disk persistence the resume mechanism. Two layers:

1. **Live in-memory checkpoint** — single mutable session; new unrelated requests *evict* the prior one.
2. **Disk KV cache** — `--kv-disk-dir DIR --kv-disk-space-mb N`. Files are SHA1-of-rendered-prefix `.kv` blobs containing the full DS4 session payload (token IDs, logits, raw/compressed KV rows, indexer rows). Format documented in `README.md` "Disk KV Cache" section. **Cache files use plain `read`/`write`, NOT mmap** — because the model is already mapped and we don't want to compete for VM mappings.

Saves happen at four moments: `cold` (after first long prompt), `continued` (every ~10k tokens at aligned frontiers), `evict` (before being replaced), `shutdown`.

### Tool call state replay

Agent clients send normalized OpenAI/Anthropic JSON tool-call objects but DeepSeek emits DSML text. If we re-rendered slightly differently, the rendered byte prefix would no longer match the live KV checkpoint and we'd rebuild from scratch. Two layers solve this:

1. **Exact replay map** — every tool call gets an unguessable API tool ID; the server keeps `tool_id → exact sampled DSML block` in a bounded radix-tree-backed map (`rax.c` is the radix tree). When the client sends the tool ID back, the renderer uses the **exact bytes the model sampled**. This map is also persisted into KV cache files.
2. **Canonicalization fallback** — only used if exact replay is missing or `--disable-exact-dsml-tool-replay`. After a tool-call turn, the server compares live sampled tokens with what the next request will render, and may rewrite the live checkpoint or fall back to an older disk KV snapshot.

During generation, **temperature is forced to 0 for DSML protocol structure** (tags, parameter headers, JSON punctuation) but uses request sampling for `string=true` payloads (file contents, edit text). Don't merge those decoding policies.

### Backend dispatch

- macOS: Metal graph in `ds4_metal.m`; kernels in `metal/*.metal`.
- Linux/Windows: CUDA in `ds4_cuda.cu` (C++ via nvcc — note `_MSC_VER`-aware `DS4_CUDA_UNUSED` macro because MSVC host compiler doesn't understand `__attribute__`).
- The model GGUF stays mmap'd; the CUDA backend calls `cudaHostRegister` on that mapping for pinned host memory, and rotates pages with `posix_madvise(DONTNEED)` (no-op on Windows).

### Platform abstraction (`ds4_platform.h` / `ds4_platform_win.c`)

On POSIX, `ds4_platform.h` is a thin passthrough that pulls in the same Unix headers DS4 has always used. On Windows it tires in `winsock2.h` BEFORE `windows.h` and provides shims so the rest of the codebase keeps using POSIX symbol names (pthread_*, mmap, poll, flock, clock_gettime, dprintf, opendir/readdir, termios, /tmp). This avoids touching the ~200 mutex lock/unlock sites in `ds4_server.c`.

`ds4.h` includes `ds4_platform.h` first. New platform-conditional code should be `#ifdef _WIN32`-gated and verified against `make cpu` to ensure Linux/macOS still compile cleanly.

### Source layout

- `ds4.c` — engine core (~18k lines)
- `ds4.h` — public engine boundary
- `ds4_gpu.h` — backend interface used by Metal/CUDA implementations
- `ds4_metal.m` + `metal/*.metal` — Metal runtime + kernels
- `ds4_cuda.cu` + `ds4_iq2_tables_cuda.inc` — CUDA backend
- `ds4_platform.{h,c}` — POSIX/Windows abstraction
- `ds4_server.c` — HTTP server (~15k lines; tests live in `#else` of `#ifndef DS4_SERVER_TEST`)
- `ds4_cli.c` + `linenoise.c` — REPL
- `ds4_bench.c`, `ds4_eval.c` — speed and capability runners
- `rax.{c,h}` — radix tree backing the tool-id replay map
- `tests/ds4_test.c` — single C runner for all regression tests
- `gguf-tools/` — offline model-building, imatrix collection, quality scoring (separate sub-projects with their own README)
- `dir-steering/` — single-vector activation steering
- `speed-bench/` — benchmark CSVs and Python plot script
- `misc/` — design notes for ANTHROPIC live continuation and Responses API behaviors

## Common debugging entry points

```sh
./ds4 --dump-tokens -p "..."                          # tokenize + identify DSML specials, exit
./ds4 --dump-logprobs /tmp/out.json --temp 0 -p "..." # greedy continuation + top-k alternatives per step
./ds4-server --trace /tmp/ds4-trace.txt ...           # rendered prompts, cache decisions, tool parser events
```

For DSML token boundaries: the close marker starts as two tokens: `</` and `｜DSML｜`.

## Key constraints to respect

- **Do not introduce C++** in the production path (CUDA `.cu` is the exception, with limited STL: `std::unordered_map`, `std::vector`).
- **Do not link against GGML or llama.cpp.** Quant tables and CPU dot-product code are adapted under MIT, but the engine is standalone.
- **Do not run the CPU path on macOS for large inference** — there is a known kernel VM bug that can crash the system.
- **The instance lock is intentional** (`/tmp/ds4.lock` on POSIX, `%TEMP%\ds4.lock` on Windows). Don't bypass it.
- **Disk KV cache directory is disposable** — when behavior looks suspicious, stop the server and `rm -rf` it.
