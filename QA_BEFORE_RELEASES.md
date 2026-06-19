# QA Before Releases

This is the release gate for DwarfStar.  Run it before tagging or pushing a
release build.  The goal is not to prove every code path exhaustively; it is to
exercise the paths that have historically regressed: Metal graph inference,
CUDA, ROCm, SSD streaming, distributed execution, disk KV cache, server APIs, and the
agent TUI/tool state machine.

Do not run multiple huge model processes at the same time.  Record the commit,
hardware, GGUF file, context size, and any non-default flags for every manual
run.

Preferred release test hosts:

- CUDA / DGX Spark: `toor@192.168.0.180`.
- Metal / distributed Mac testing: `mac-m5max-it` and `mac-m5max-us`.
- ROCm: The Strix Halo system at antirez@strixhalo (Framework Desktop).

The Mac hosts have DNS entries and are reached through an internet VPN.  They
are connected to each other over WiFi and also through a Thunderbolt 5
point-to-point link.  The TB5 route is the preferred distributed-inference
network when it is available, but it can be fragile and sometimes only works
when `ds4` is executed in the foreground.  Prefer these machines for release
testing, especially distributed inference.  Local fallback testing on this
machine is acceptable when needed; it is an M3 Max with 128 GB RAM.
The Strix Halo system is reachable via the VPN as well and has a local WiFi
address in the same lan of the M5 Max systems. The CUDA hosts are in a
different remote lan and are accessible via a different VPN active
in this system.

## 1. Repository And Build Sanity

- Start from a clean tree except intentional release notes:
  `git status --short`.
- Build the normal local target:
  `make clean && make`.
- Build CPU-only binaries as a compile check only:
  `make clean && make cpu`.
- Run whitespace checks before committing:
  `git diff --check`.
- Confirm `./ds4 --help`, `./ds4-server --help`, and `./ds4-agent --help` render
  cleanly, with readable section colors and no broken wrapping.

## 2. Core Regression Tests

- Run the default suite:
  `make test`.
- Run the vector checks explicitly after any tokenizer, template, KV, kernel,
  quantization, or prompt-rendering change:
  `./ds4_test --logprob-vectors`
  and `./ds4_test --local-golden-vectors`.
- Run server tests when HTTP, SSE, prompt rendering, cache policy, or tool-call
  replay changed:
  `./ds4_test --server`.
- Run `./ds4-eval --self-test-extractors`.

## 3. Metal Flash Path

Use the normal Flash GGUF that 128 GB users run.

- One-shot CLI:
  `./ds4 -m ds4flash.gguf --ctx 32768 --nothink -p "Explain C pointers in one paragraph."`
- Thinking and max-thinking prompts:
  run one short coding prompt with default thinking and one with max thinking.
- Long-context recall:
  run the long name/number or archive recall test used for catching attention
  and MoE routing drift.
- Logprob sanity:
  `./ds4 --nothink --temp 0 --dump-logprobs /tmp/ds4-logprobs.json --logprobs-top-k 20 -p "..."`
  and inspect that the continuation is sane.
- Speed sanity:
  run `ds4-bench` with `speed-bench/promessi_sposi.txt` and compare prefill,
  generation speed, and KV bytes with the last known good numbers for the same
  machine.

## 4. Metal PRO Path

PRO support is experimental, but release builds must not break it silently.

- If a PRO-capable machine is available, run a short PRO q2 prompt and verify
  the correct template, thinking behavior, and endpoint aliases.
- For PRO Q4 distributed builds, test only on the intended high-memory machines.
- If PRO cannot be run locally, at least build all binaries and review changes
  touching model shape, tensor lookup, routed expert mapping, template logic,
  and KV payload compatibility.

## 5. SSD Streaming

SSD streaming is a capacity path, so test both correctness and user experience.

- Flash q2/q2-q4 streaming:
  `./ds4 -m ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 32GB -p "..."`
- Regression test mixed-quant Flash SSD streaming. Use the mixed q2/q4 GGUF
  with boosted Q4 routed-expert layers and a prompt long enough to exercise the
  selected-address prefill path; it must not fail with "model range is not
  covered by mapped model views":
  `./ds4 -m gguf/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed.gguf --ssd-streaming --ssd-streaming-cache-experts 16GB --ctx 4096 --tokens 1 --nothink --prompt-file /tmp/ds4_600tok_prompt.txt`.
- Cold streaming measurement:
  run once with `--ssd-streaming-cold` and verify no deadlock, missing expert,
  or impossible slowdown.
- Confirm startup reports cache budget and that generation does not stall on
  repeated expert misses for a small interactive prompt.
- If streaming cache internals changed, test the same prompt twice and compare
  first-token/logprob sanity between runs.

## 6. CUDA / DGX Spark

Before a release, ask the user for CUDA access if it is not already configured.
Use the DGX Spark / GB10 host `toor@192.168.0.180`.  Do not claim CUDA is
release-ready without this pass.

- Fetch or push the exact release commit to the CUDA machine.
- Build:
  `make clean && make cuda-spark`.
- Run:
  `make cuda-regression`.
- Run a short CLI prompt with the Flash GGUF and record generation t/s.
- Run a longer prompt that exercises routed experts past a few thousand tokens.
- If CUDA Q4, distributed, streaming hooks, tensor span loading, or model cache
  code changed, test the specific GGUF and split mode that uses that path.
- Verify that any CUDA-only warning fixes are also clean on macOS and do not
  change Metal behavior.

## 7. ROCm / Strix Halo

Use the Strix Halo Framework Desktop via the VPN hostname `strixhalo`
(`antirez@strixhalo`).  This host validates the ROCm backend; do not use it as
a substitute for CUDA or Metal release testing.

- Fetch or push the exact release commit to the Strix Halo machine.
- Build:
  `make clean && make strix-halo`.
- Use the q2 Flash imatrix GGUF for release smoke tests:
  `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.
- Do not use the mixed q2-q4 or Q4 Flash GGUFs for routine Strix Halo QA yet.
  They are dangerous on this machine for now because the ROCm path can hit
  system OOM instead of failing cleanly.
- Run a short CLI prompt:
  `./ds4 -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --ctx 4096 --nothink -p "Reply with exactly: OK"`.
- Run one longer prompt if ROCm kernels, backend hooks, tensor loading, model
  cache, KV cache, or graph prefill code changed.
- Record startup memory/cache messages, prefill speed, generation speed, and
  whether the backend reports `ROCm backend initialized`.

## 8. Distributed Inference

Distributed code has regressed around route setup, KV snapshots, request IDs,
and split model loading.  Test it whenever distributed, KV, session, or model
loading code changes.

- Prefer `mac-m5max-it` and `mac-m5max-us` for Metal distributed tests.  Use the
  TB5 point-to-point link when it is working; otherwise note that the run used
  WiFi/VPN routing.
- Start workers first, then the coordinator.
- Test a small prompt and a longer prompt.
- Verify the coordinator waits for a complete route and exits cleanly.
- Verify `Ctrl+C` returns control after the current distributed token or chunk
  drains.
- Save and restore a distributed KV snapshot if that code changed.
- If CUDA distributed is relevant, test across the CUDA hosts and record
  generation speed, not just "it works".

## 9. Disk KV Cache

Disk KV cache bugs are high impact for server users.

- Start the server with:
  `./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192`.
- Run the same request twice and verify the second request hits cache.
- Fill the cache enough to trigger eviction; verify the newly-written entry is
  not evicted and useful anchors are retained.
- Test rejection of incompatible checkpoints when model, quantization, context,
  or raw/compressed KV layout changes.
- Test stripped agent sessions: `/strip <id>` then `/switch <id>` should rebuild
  by prefill and render sane history.

## 10. Server APIs

The server must keep compatibility across OpenAI, Responses, and Anthropic
clients.

- `GET /v1/models/deepseek-v4-flash` and `GET /v1/models/deepseek-v4-pro`
  should both serve whichever GGUF is loaded.
- Test OpenAI chat completion, OpenAI Responses, and Anthropic messages.
- Test SSE streaming with thinking enabled and disabled.
- Test keepalive during long prefill and confirm clients do not time out.
- Test `--trace` and confirm rendered prompts, cache decisions, generated text,
  and tool-parser events are useful without leaking unrelated state.

## 11. ds4-agent

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
  delete `/tmp/mymandel`, ask ds4-agent to create a small C ASCII Mandelbrot
  program there, build and run it, then in a second user turn ask for a small
  modification that should naturally use the edit tool, such as changing the
  ASCII character ramp or output dimensions.  Verify the agent edits the
  existing file instead of rewriting the whole project, and that the final
  program still builds and runs.
- Bash tools:
  test short output, large output truncation, non-zero exit output, long-running
  jobs, `bash_status`, and `bash_stop`.
- Web tools:
  `google_search` and `visit_page` should ask for visible Chrome approval with
  timeout, open pages without stealing focus when possible, extract Markdown,
  close tabs, and handle consent/privacy walls as tool errors the model can see.
- TUI:
  test multiline prompt editing, history navigation, queued prompt display,
  status bar fill to terminal width, syntax highlighting in Markdown/code blocks,
  and SSH/remote terminal flicker.

## 12. Download Script And Model Files

- Test `download_model.sh` in a temporary directory so local weights are not
  overwritten.
- Test one Flash target and one PRO target enough to verify URL, resume, Hugging
  Face CLI/curl behavior, file naming, and symlink policy.
- Verify legacy removed targets fail clearly.
- Verify README model names match the script and Hugging Face repository.

## 13. Performance And Power

- Run `ds4-bench` on the release machine and compare with tracked CSV baselines.
- Test `--power 100` is not throttled.
- Test `--power 50` visibly reduces duty cycle in CLI, server, agent, eval, and
  bench where practical.
- Confirm context buffer size, raw KV rows, compressed KV rows, and mmap behavior
  match expectations for 32k, 100k, and any release-advertised context size.

## 14. Release Sign-off

Do not sign off until:

- macOS Metal Flash passed.
- CUDA was tested on the CUDA machine or the release notes explicitly say CUDA
  was not validated.
- ROCm was tested on Strix Halo or the release notes explicitly say ROCm was
  not validated.
- Disk KV cache was exercised.
- Server API streaming was exercised.
- Agent interruption and tool loops were exercised manually.
- Speed is within expected variance for the same hardware and model.
- Any skipped item is written down with the reason.
