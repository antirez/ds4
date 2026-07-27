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

- CUDA / DGX Spark: `toor@192.168.60.184`.
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
- Treat compiler warnings as build failures. Save each release and test build's
  complete output and require no `warning:` or NVCC `warning #` lines. Fix the
  source when possible; use a narrow target-specific suppression only when a
  test deliberately compiles a partial translation unit.
- Repeat the warning-free build gate on the release hardware:
  `make clean && make` on Metal,
  `make clean && make cuda-spark` on DGX Spark,
  `make clean && make cuda-generic CUDA_HOME=/usr` on the multi-GPU CUDA host,
  `make clean && make strix-halo` on Strix Halo.
- Run whitespace checks before committing:
  `git diff --check`.
- Confirm `./ds4 --help`, `./ds4-server --help`, and `./ds4-agent --help` render
  cleanly, with readable section colors and no broken wrapping.

## 2. Core Regression Tests

- Run the default suite:
  `make test`.
- Run `tests/test_gpu_args_cli.sh` explicitly after changing executable option
  parsing or multi-GPU placement. Invalid values and device/budget count
  mismatches must reach the shared GPU parser in all four binaries; an
  `unknown option` response from a binary that advertises the flag is a
  release blocker. On CUDA, also start `ds4-server` once with
  `--gpu-vram auto` and the intended `--gpu-devices` list and preserve the
  resolved layout line.
- Run the vector checks explicitly after any tokenizer, template, KV, kernel,
  quantization, or prompt-rendering change:
  `./ds4_test --logprob-vectors`
  and `./ds4_test --local-golden-vectors`.
- Run server tests when HTTP, SSE, prompt rendering, cache policy, or tool-call
  replay changed:
  `./ds4_test --server`.
- Run `./ds4-eval --self-test-extractors`.

## 3. Official Continuation Quality Gates

These tests are release-blocking after tokenizer, template, KV-cache, attention,
MoE routing, quantization, logit, or model-graph changes.  They are
teacher-forced continuation checks against hosted-model output and API
top-logprob slices, so do not replace them with one sampled chat answer.

- Build the scorer:
  `make -C gguf-tools quality-score`.
- Run the tracked DeepSeek V4 Flash smoke vectors:
  `./ds4_test --logprob-vectors`.
  This covers the official Flash API vectors in `tests/test-vectors/`, including
  short prompts and long-prompt attention cases.
- Run the 100-case DeepSeek V4 Flash fixture for every released Flash GGUF:
  `gguf-tools/quality-testing/score_official /path/to/deepseek-v4-flash.gguf gguf-tools/quality-testing/data/flash/manifest.tsv /tmp/flash.tsv 4096`.
- Run the 100-case GLM 5.2 OpenRouter fixture for every released GLM GGUF:
  `gguf-tools/quality-testing/score_official models/GLM-5.2-UD-Q4_K_XL.gguf gguf-tools/quality-testing/data/glm52-openrouter-100/manifest.tsv /tmp/glm52-q4.tsv 4096`.
  Current Q4 XL reference band: first-token match `95/100`, API top-1 agreement
  about `0.942`, and API pair-order agreement about `0.880`.
- Run the same GLM fixture for reduced-precision GLM release files.  The Q2
  routed reference is lower quality but should stay near first-token match
  `92/100`, API top-1 agreement about `0.890`, and API pair-order agreement
  about `0.800` unless the quantization changed deliberately.
- Run the 100-case DeepSeek V4 PRO fixture for every released PRO GGUF:
  `gguf-tools/quality-testing/score_official /path/to/deepseek-v4-pro.gguf gguf-tools/quality-testing/data/pro/manifest.tsv /tmp/pro.tsv 4096`.
- For SSD streaming, run the same official-continuation scorer once with full
  residency and once with `--ssd-streaming` for the release model.  The summary
  and API agreement should stay in the same quality band.
- Compare any candidate against the previous release or last-known-good output:
  `python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv`.
  Treat a large first-token-match drop, a clear NLL regression, or a material
  API top-1/pair-order regression as a blocker unless the release notes call out
  an intentional quality tradeoff.
- Keep the raw `summary` and `api_summary` lines in the release notes or QA log.
  Do not use stale manifests from `misc/` as release evidence.

## 4. Metal Flash Path

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

### DSpark / DeepSpec Runtime

DSpark is opt-in, but it mutates the verifier, target-hidden capture, support
model loading, and scheduler paths.  Run these whenever DSpark support,
speculative verification, confidence/scheduler policy, target hidden capture,
tiny routed-MoE verifier kernels, or shared `--mtp` support-model code changes:

- Default greedy acceptance fixture:
  `DS4_DSPARK_MODEL=/Users/antirez/ds4/ds4flash.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support.gguf make dspark-acceptance`.
- 64-token guardrail:
  `DS4_DSPARK_FIXTURE_TOKENS=64 DS4_DSPARK_MODEL=/Users/antirez/ds4/ds4flash.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support.gguf make dspark-acceptance`.
- Fixed-block partial-accept fallback:
  `DS4_DSPARK_FIXTURE_CONFIDENCE=0 DS4_DSPARK_FIXTURE_TOKENS=8 DS4_DSPARK_FIXTURE_REQUIRE_PARTIAL=1 DS4_DSPARK_MODEL=/Users/antirez/ds4/ds4flash.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support.gguf make dspark-acceptance`.
- DSpark verifier invariant smoke:
  `DS4_TEST_MODEL=/Users/antirez/ds4/ds4flash.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support.gguf make dspark-verify-depth`.
- If shared support-model or verifier structures changed, also run legacy MTP:
  `make mtp-verify-depth` with `DS4_TEST_MTP` set to a one-stage MTP support
  GGUF, or confirm the target skips only because the optional file is missing.
- Record `c_add` `accepted_draft`, `errors=0`, `verify_layer`, and `net_saved`
  for both 32-token and 64-token runs.  A faster run with lower proposal
  quality is a regression unless it was an intentional scheduler change.
- If verifier MoE kernels changed, run one diagnostic `c_add` profile with
  `DS4_DSPARK_VERIFY_SELECTED_PROFILE=1` or the Metal MoE stage profiler and
  record the selected-expert footprint or stage timing in the DSpark log.

### Session Microbatching And Metal TP

Run these gates whenever session scheduling, batched decode, mixed
prefill/decode, QKV projection, shared or routed experts, tensor parallelism,
or backend fallback selection changes.

- On a single Metal machine, run the full-vocabulary exact-logit oracle with
  2, 4, 8, and 16 sessions:
  `DS4_TEST_MODEL=/path/to/ds4flash.gguf DS4_TEST_SESSION_COUNT=N make test-metal-session-batch`.
  Compatible resident Q8 runs must report `native_shared=1 native_qkv=1` at
  every tested count. The 16-session run covers row counts above the old
  artificial eight-row limit.
- Repeat the four-session oracle with
  `DS4_METAL_SESSION_BATCH_SHARED=0` and with
  `DS4_METAL_SESSION_BATCH_QKV=0`. The first run must use the complete fallback;
  the second may batch the shared expert only. Both must remain bit-exact.
- The oracle must cover reversed row ordering, at least six decode steps, and a
  mixed prefill/decode call. Any nonzero differing-logit count is a blocker;
  argmax-only agreement is insufficient.
- Benchmark 1, 2, 4, 8, and 16 simultaneous resident sessions on the same host
  and model. Record model-step latency and aggregate decode tokens/second, not
  only request completion speed. The current Metal path batches QKV and part of
  the shared expert, but still runs attention, routed experts, shared down, and
  the output head per session. Treat flat aggregate scaling as unfinished
  implementation work, not evidence that Metal cannot benefit from batching.
- On `mac-m5max-it` and `mac-m5max-us`, run the same oracle in physical TP mode
  over explicit `tcp` and `rdma` transports. Set `DS4_TEST_TP_MODE=leader` on
  the leader and `DS4_TEST_TP_MODE=worker DS4_TEST_TP_LEADER_HOST=HOST` on the
  worker, with a unique `DS4_TEST_TP_PORT`. Run at least 2 and 4 sessions and
  preserve both logs. TP currently uses the ordered per-session fallback, so
  native single-machine row-grid flags must remain off in the TP logs.
- For the current TB5 MacBook link, US is `10.99.0.2` on `en1`/`rdma_en1` and
  IT is `10.99.0.1` on `en6`/`rdma_en6`; both use GID index 1. Before testing,
  require `rdma_ctl status` to report `enabled` and `ibv_devinfo -v` to show
  `PORT_ACTIVE` plus the corresponding `::ffff:10.99.0.x` GID. Force the
  device and GID with `--rdma-device NAME --rdma-gid-index 1` if automatic
  selection is ambiguous. A working TB IP ping alone is not RDMA evidence.
- Kill the TP worker during one batch with `DS4_TEST_TP_DISCONNECT=1` on the
  leader. The operation must fail cleanly, invalidate every affected session,
  and return control without hanging.
- Verify unsupported combinations explicitly: GLM, DSpark/MTP support models,
  SSD streaming, quality/reference modes, steering, resident Q4 expert
  overlap, and CPU-router modes must use the established exact fallback or
  reject the combination before evaluation. They must not partially activate
  native batching.

## 5. Metal PRO Path

PRO support is experimental, but release builds must not break it silently.

- If a PRO-capable machine is available, run a short PRO q2 prompt and verify
  the correct template, thinking behavior, and endpoint aliases.
- For PRO Q4 distributed builds, test only on the intended high-memory machines.
- If PRO cannot be run locally, at least build all binaries and review changes
  touching model shape, tensor lookup, routed expert mapping, template logic,
  and KV payload compatibility.

## 6. GLM 5.2

GLM has a different template, model shape, MTP block, attention layout,
tensor-parallel gate width, and streaming policy. Flash or PRO success does not
substitute for this matrix.

- On a 512 GB Metal machine, run short greedy prompts with both the Q4 XL and
  reduced-precision Q2 release GGUFs. Cover thinking and no-thinking templates,
  and verify the server reports the GLM model family rather than a DeepSeek
  alias internally.
- Run the OpenRouter smoke vectors explicitly:
  `DS4_TEST_MODEL=/path/to/glm.gguf
  DS4_TEST_VECTOR_FILE=tests/test-vectors/glm-openrouter/official.vec
  ./ds4_test --logprob-vectors`.
  Preserve the report as a diagnostic. The hosted vectors include very
  low-probability top-20 tails whose membership is not stable after GLM routed
  expert quantization, so an individual `official top token missing locally`
  assertion is not by itself a release blocker. Selected-token mismatches must
  remain consistent with the model's 100-case first-token band, and the
  section 3 scorer is the release gate for aggregate GLM quality.
- Run the 100-case Q4 XL and Q2 official fixtures from section 3 and preserve
  both `summary` and `api_summary` lines. Compare against the documented Q4 and
  Q2 reference bands independently.
- Run `tests/glm_long_context_smoke.sh` with the release-advertised context on
  the 512 GB Metal host. The generated continuation must begin with `>` and
  contain none of the known token-corruption markers.
- Exercise integrated GLM MTP with `--glm-mtp-timing` on a deterministic
  prompt. Compare the greedy text to
  a non-MTP run, require clean speculative cycles, and record acceptance and
  timing. Also run once with MTP disabled to prove ordinary decode remains the
  default.
- Run the Metal session oracle with 2 and 4 GLM sessions. It must report
  `family=glm native_shared=0 native_qkv=0` and remain exact, including mixed
  prefill/decode; the DeepSeek-only row-grid kernels must not activate.
- Run resident and SSD-streaming GLM Q2 prompts with the same greedy input.
  Compare first token and top-logprob sanity, and record the selected full-layer
  prefix and dynamic expert-cache budget.
- Run physical two-machine GLM TP over TCP and RDMA with short and long prompts.
  Record prefill/decode speed, transport, rank residency, and clean shutdown.
  Repeat one run with `--tensor-parallel-token-prefill` as the exact-arithmetic
  diagnostic. Use a GGUF whose routed-expert type has ownership-aware GLM TP
  kernels. Also test a Q4-routed GLM file as a negative gate: until Q4 ownership
  kernels are implemented, both ranks must reject it clearly before evaluation
  rather than loading a partial split or hanging.
- On the eight-GPU CUDA host, run one resident GLM Q2 prompt, a long-context
  prompt, integrated GLM MTP, and concurrent server requests. Use ordinary
  eight-GPU layer placement for GLM; do not pass the Flash-specific
  `--cuda-tensor-parallel` option. Multi-tier GLM prefill must
  report progress through the tier-switching token-major path, and decode,
  cache updates, and output-head/logit assembly must complete without CPU spill.
  Auto-placement must reserve each layer's compact DSA/indexer cache and the
  graph workspace before loading weights; a late graph-allocation failure is a
  release blocker. Confirm the long-context layout stays within every device's
  budget and uses additional tiers when the cache no longer fits on the earlier
  ones.
  The long-context harness can select this backend with
  `DS4_GLM_BACKEND=cuda` and pass placement flags through
  `DS4_GLM_EXTRA_ARGS="--gpu-vram auto --gpu-devices 0,2,4,6,1,3,5,7"`.
- Through `ds4-server`, exercise OpenAI chat, Responses, and Anthropic requests
  against GLM, including thinking and SSE. DeepSeek compatibility endpoint
  aliases may resolve to the loaded model, but rendered prompts and generated
  text must use the GLM template.

## 7. SSD Streaming

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

## 8. CUDA / DGX Spark

Before a release, ask the user for CUDA access if it is not already configured.
Use the DGX Spark / GB10 host `toor@192.168.60.184`.  Do not claim CUDA is
release-ready without this pass.

- Fetch or push the exact release commit to the CUDA machine.
- Build:
  `make clean && make cuda-spark`.
- Require both the DGX Spark build and the eight-GPU CUDA build to complete
  without compiler warnings.
- Run:
  `make cuda-regression`.
- Run a short CLI prompt with the Flash GGUF and record generation t/s.
- Run a longer prompt that exercises routed experts past a few thousand tokens.
- On the eight-GPU CUDA host, run the full-vocabulary decode oracle:
  `DS4_TEST_MODEL=/path/to/flash.gguf make test-cuda-session-batch`.
  Preserve the per-batch timing for 2, 4, and 8 rows and require
  `nonexact_logits=0`. Run the released Q4 file and the reduced-precision Q2
  file: Q4 exercises grouped routed/shared stages, while unsupported Q2 native
  MoE shapes must retain the ordered exact fallback.
- With CUDA TP attention enabled, compatible Q4 runs must use grouped
  attention-core, QKV, KV-store, and attention-post by default and remain
  full-vocabulary exact against isolated decode. On the eight-L40S host, the
  16-row decode step must remain above 110 aggregate tokens/s. Repeat once with
  `DS4_CUDA_TP_ATTN=0` only as rollback coverage; it is not the production
  configuration.
- Run native mixed prefill/decode at the default frontier and at compressed
  context:
  `DS4_TEST_MODEL=/path/to/flash.gguf make test-cuda-mixed-batch` and
  `DS4_TEST_CONTEXT=4096 DS4_TEST_MIXED_INITIAL=2048 DS4_TEST_MIXED_ROUNDS=8
  DS4_TEST_MODEL=/path/to/flash.gguf make test-cuda-mixed-batch`.
  Every round must report exact logits and `mode=native`; a serialized fallback
  is a failure for the eight-GPU TP/EP topology. Under CUDA TP attention, the
  native mixed step must use the same exact grouped decode stages when their
  capability checks pass; record correctness and speedup separately. Also
  force an 800-row prefill quantum with
  `DS4_TEST_ALLOW_FALLBACK=1`; it must report the serialized safety fallback.
- Start `ds4-server` with 8 and 16 batched sessions and issue at least that many
  simultaneous requests with mixed prompt lengths. Verify no session mix-up,
  deadlock, or starvation and record aggregate generation throughput.
- On DGX Spark, verify the same public batch API and server concurrency use the
  single-GPU fallback without creating peer-only TP/EP state. The eight-GPU
  native oracle is not a valid Spark test because its topology is intentionally
  unavailable there.
- If CUDA Q4, distributed, streaming hooks, tensor span loading, or model cache
  code changed, test the specific GGUF and split mode that uses that path.
- Verify that any CUDA-only warning fixes are also clean on macOS and do not
  change Metal behavior.

## 9. ROCm / Strix Halo

Use the Strix Halo Framework Desktop via the VPN hostname `strixhalo`
(`antirez@strixhalo`).  This host validates the ROCm backend; do not use it as
a substitute for CUDA or Metal release testing.

- Fetch or push the exact release commit to the Strix Halo machine.
- Build:
  `make clean && make strix-halo`.
- Require the ROCm build to complete without compiler warnings.
- Use the q2 Flash imatrix GGUF for release smoke tests:
  `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.
- Do not use the mixed q2-q4 or Q4 Flash GGUFs for routine Strix Halo QA yet.
  They are dangerous on this machine for now because the ROCm path can hit
  system OOM instead of failing cleanly.
- Run a short CLI prompt:
  `./ds4 -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --ctx 4096 --nothink -p "Reply with exactly: OK"`.
- Run one longer prompt if ROCm kernels, backend hooks, tensor loading, model
  cache, KV cache, or graph prefill code changed.
- Run the GLM Q2 release model through ROCm SSD streaming with at least four
  generated tokens:
  `./ds4 --rocm -m gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf --ssd-streaming --ctx 4096 --nothink --tokens 4 -p "Reply with exactly: OK"`.
  Startup must select a cache budget that passes the memory guard without an
  override, and both compact indexed prefill and decode must complete.
- Run one longer GLM prompt with the release-advertised Strix context after
  changes to GLM attention, typed quantized projections, streaming expert
  caches, or memory budgeting. Record the context, cache split, and whether
  the continuation stays free of token-corruption markers.
- Run the same GLM model with `--glm-mtp-timing --temp 0`. At least one draft
  verification cycle must complete without a `glm mtp step failed` message.
- Record startup memory/cache messages, prefill speed, generation speed, and
  whether the backend reports `ROCm backend initialized`.

## 10. Distributed Inference

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

## 11. Disk KV Cache

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

## 12. Server APIs

The server must keep compatibility across OpenAI, Responses, and Anthropic
clients.

- `GET /v1/models/deepseek-v4-flash` and `GET /v1/models/deepseek-v4-pro`
  should both serve whichever GGUF is loaded.
- Test OpenAI chat completion, OpenAI Responses, and Anthropic messages.
- Test SSE streaming with thinking enabled and disabled.
- Test keepalive during long prefill and confirm clients do not time out.
- On the eight-L40S CUDA TP target, start `ds4-server` with the release TP
  options and verify all 16 100k-context sessions allocate. Startup must report
  a 2048-token prefill cap; a silent fallback to 4096 is an OOM regression.
- Test `--trace` and confirm rendered prompts, cache decisions, generated text,
  and tool-parser events are useful without leaking unrelated state.

## 13. ds4-agent

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

## 14. Download Script And Model Files

- Test `download_model.sh` in a temporary directory so local weights are not
  overwritten.
- Test one Flash target and one PRO target enough to verify URL, resume, Hugging
  Face CLI/curl behavior, file naming, and symlink policy.
- Verify legacy removed targets fail clearly.
- Verify README model names match the script and Hugging Face repository.

## 15. Performance And Power

- Run `ds4-bench` on the release machine and compare with tracked CSV baselines.
- Test `--power 100` is not throttled.
- Test `--power 50` visibly reduces duty cycle in CLI, server, agent, eval, and
  bench where practical.
- Confirm context buffer size, raw KV rows, compressed KV rows, and mmap behavior
  match expectations for 32k, 100k, and any release-advertised context size.

## 16. Release Sign-off

Do not sign off until:

- macOS Metal Flash passed.
- GLM 5.2 Metal, official-quality, MTP, batching-fallback, and applicable TP or
  CUDA gates passed.
- Official continuation quality gates passed for every released model family.
- CUDA was tested on the CUDA machine or the release notes explicitly say CUDA
  was not validated.
- ROCm was tested on Strix Halo or the release notes explicitly say ROCm was
  not validated.
- Metal, CUDA, ROCm, CPU-only, and test builds completed without compiler
  warnings on every release target that was validated.
- Disk KV cache was exercised.
- Server API streaming was exercised.
- Agent interruption and tool loops were exercised manually.
- Speed is within expected variance for the same hardware and model.
- Metal 2/4/8/16-session exactness and forced fallback gates passed.
- Physical Metal TP batching and CUDA native decode/mixed batching passed when
  those backends are part of the release.
- Any skipped item is written down with the reason.
