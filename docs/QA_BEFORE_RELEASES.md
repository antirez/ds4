# QA Before Releases

This is the release gate for Pulsar.  Run it before tagging or pushing a
release build.  The goal is not to prove every code path exhaustively; it is to
exercise the paths that have historically regressed: CUDA graph inference,
disk KV cache, server APIs, and the agent TUI/tool state
machine.

This fork is CUDA-only and targets the DGX Spark (GB10, ~128 GB unified
memory); all release testing happens on that host.  Do not run multiple huge
model processes at the same time.  Record the commit, GGUF file, context size,
and any non-default flags for every manual run.

## 1. Repository And Build Sanity

- Start from a clean tree except intentional release notes:
  `git status --short`.
- Build the release binaries:
  `make clean && make cuda-spark`.
- Run whitespace checks before committing:
  `git diff --check`.
- Confirm `./pulsar --help`, `./pulsar-server --help`, and `./pulsar-agent --help` render
  cleanly, with readable section colors and no broken wrapping.

## 2. Core Regression Tests

- Run the default suite:
  `make test`.
- Run the CUDA smoke regression:
  `make cuda-regression`.
- Run the vector checks explicitly after any tokenizer, template, KV, kernel,
  quantization, or prompt-rendering change:
  `./pulsar_test --logprob-vectors`
  and `./pulsar_test --local-golden-vectors`.
- Run server tests when HTTP, SSE, prompt rendering, cache policy, or tool-call
  replay changed:
  `./pulsar_test --server`.
- Run `./pulsar-eval --self-test-extractors`.

## 3. Flash Inference Path

Use the normal Flash GGUF that 128 GB users run.

- One-shot CLI:
  `./pulsar -m ds4flash.gguf --ctx 32768 --nothink -p "Explain C pointers in one paragraph."`
- Thinking and max-thinking prompts:
  run one short coding prompt with default thinking and one with max thinking.
- Long-context recall:
  run the long name/number or archive recall test used for catching attention
  and MoE routing drift.
- Logprob sanity:
  `./pulsar --nothink --temp 0 --dump-logprobs /tmp/ds4-logprobs.json --logprobs-top-k 20 -p "..."`
  and inspect that the continuation is sane.
- Speed sanity:
  run `pulsar-bench` with `speed-bench/promessi_sposi.txt` and compare prefill,
  generation speed, and KV bytes with the last known good numbers for the same
  machine.
- Run a longer prompt that exercises routed experts past a few thousand tokens.

## 4. Disk KV Cache

Disk KV cache bugs are high impact for server users.

- Start the server with:
  `./pulsar-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192`.
- Run the same request twice and verify the second request hits cache.
- Fill the cache enough to trigger eviction; verify the newly-written entry is
  not evicted and useful anchors are retained.
- Test rejection of incompatible checkpoints when model, quantization, context,
  or raw/compressed KV layout changes.
- Test stripped agent sessions: `/strip <id>` then `/switch <id>` should rebuild
  by prefill and render sane history.

## 5. Server APIs

The server must keep compatibility across OpenAI, Responses, and Anthropic
clients.

- `GET /v1/models/deepseek-v4-flash` and `GET /v1/models/deepseek-v4-pro`
  should both serve whichever GGUF is loaded.
- Test OpenAI chat completion, OpenAI Responses, and Anthropic messages.
- Test SSE streaming with thinking enabled and disabled.
- Test keepalive during long prefill and confirm clients do not time out.
- Test `--trace` and confirm rendered prompts, cache decisions, generated text,
  and tool-parser events are useful without leaking unrelated state.

## 6. pulsar-agent

The agent is the most stateful component.  Test it manually, not only by build.

- Startup banner, status bar, help, `/power`, `/save`, `/list`, `/switch`,
  `/history`, `/compact`, `/new`, `/del`, and `/strip`.
- Ctrl+C during generation, during prefill, during a web fetch, and during a
  long tool call.  After `Stopped by user`, typing a new prompt must work.
- Queue messages while the model is busy.  Queued messages must not skip tool
  execution; after tool results, the queued user text must be provided.
- Read/search/edit/write tools:
  create a temp project, ask for edits, verify old/new and `[upto]` anchored
  edits fail safely on ambiguous matches and do not require retyping whole files.
- Real coding edit loop:
  delete `/tmp/mymandel`, ask pulsar-agent to create a small C ASCII Mandelbrot
  program there, build and run it, then in a second user turn ask for a small
  modification that should naturally use the edit tool, such as changing the
  ASCII character ramp or output dimensions.  Verify the agent edits the
  existing file instead of rewriting the whole project, and that the final
  program still builds and runs.
- Bash tools:
  test short output, large output truncation, non-zero exit output, long-running
  jobs, `bash_status`, and `bash_stop`.
- TUI:
  test multiline prompt editing, history navigation, queued prompt display,
  status bar fill to terminal width, syntax highlighting in Markdown/code blocks,
  and SSH/remote terminal flicker.

## 7. Download Script And Model Files

- Test `download_model.sh` in a temporary directory so local weights are not
  overwritten.
- Test one Flash target enough to verify URL, resume, Hugging Face CLI/curl
  behavior, file naming, and symlink policy.
- Verify legacy removed targets fail clearly.
- Verify README model names match the script and Hugging Face repository.

## 8. Performance And Power

- Run `pulsar-bench` on the release machine and compare with tracked CSV baselines.
- Test `--power 100` is not throttled.
- Test `--power 50` visibly reduces duty cycle in CLI, server, agent, eval, and
  bench where practical.
- Confirm context buffer size, raw KV rows, compressed KV rows, and mmap behavior
  match expectations for 32k, 100k, and any release-advertised context size.

## 9. Release Sign-off

Do not sign off until:

- `make test` and `make cuda-regression` passed on the GB10 host.
- The Flash inference path (CLI, thinking modes, long-context recall) was
  exercised.
- Disk KV cache was exercised.
- Server API streaming was exercised.
- Agent interruption and tool loops were exercised manually.
- Speed is within expected variance for the same hardware and model.
- Any skipped item is written down with the reason.
