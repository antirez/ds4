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

- CUDA / DGX Spark: `toor@192.168.4.180` and `toor@192.168.4.181`.
- Metal / distributed Mac testing: `mac-m5max-it` and `mac-m5max-us`.
- ROCm: The Strix Halo system at antirez@strixhalo (Framework Desktop).

`192.168.60.250` is permission-only. Never connect to it for QA, stop or start
its server, build there, or run tests or benchmarks there without asking
Salvatore and receiving explicit permission for that specific QA pass. Earlier
permission does not carry over to later work.

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
- After fetching a bundle, rewriting commits, or resetting a remote test tree,
  force a clean build. Do not trust incremental `make`: restored source mtimes
  can be older than a stale executable. Record the tested binary's commit or
  verify it was rebuilt from the selected tree before running remote QA.
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
  `make clean && make cuda-generic CUDA_HOME=/usr` on the multi-GPU CUDA host
  only after receiving permission for `192.168.60.250`,
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
  `DS4_TEST_MODEL=/path/to/0731.gguf
  DS4_TEST_VECTOR_FILE=tests/test-vectors/flash-0731/official.vec
  ./ds4_test --logprob-vectors`
  and
  `DS4_TEST_MODEL=/path/to/0731.gguf
  DS4_TEST_LOCAL_GOLDEN_FILE=tests/test-vectors/flash-0731/local-golden.vec
  ./ds4_test --local-golden-vectors`.
- Run server tests when HTTP, SSE, prompt rendering, cache policy, or tool-call
  replay changed:
  `./ds4_test --server`.
- Run `./ds4-eval --self-test-extractors`.

### Critical Input And Server Regression Pass

Run these checks after changing parsers, server generation, model loading,
distributed snapshots, caches, DSpark, or CUDA build rules. Keep the item
numbers in the QA report so omissions are visible.

1. Send malformed OpenAI, Responses, and Anthropic requests with repeated
   owned string or array fields under ASan. Each request must fail cleanly and
   a following valid request must still work. Run `./ds4_test --server` too.
2. Replay at least 4,096 assistant/tool-result pairs through the Responses and
   Anthropic validators. Validation must remain linear-time and preserve the
   same accepted and rejected histories as a short replay.
3. Feed a distributed worker a snapshot header whose declared lengths exceed
   the configured and protocol limits. It must reject the header before a
   large allocation or payload read, without growing RSS materially.
4. Run malformed safetensors and GGUF fixtures through the loader and
   `gguf-tools/deepseek4-quantize` under ASan and UBSan. Truncated files,
   impossible dimensions, and overflowing tensor sizes must be rejected.
5. With a checkpoint-matched DSpark drafter, compare temperature-zero output
   against a no-drafter run for 400 and 800 generated tokens. Record the first
   output difference, acceptance, direct commits, replay fallbacks, and decode
   speed. Byte identity is not required: DSpark commits the batched verifier
   state, whose floating-point operation order differs from one-token decode.
   Verifier errors, invalid text, or a material continuation-quality regression
   remain release blockers. Use `--dspark-strict` for the target-only control.
6. Exercise unterminated and twice-closed reasoning in streaming and
   non-streaming OpenAI, Responses, and Anthropic requests, with and without
   tools. Reasoning must never leak into answer content.
7. On real Blackwell hardware, build the CUDA targets for `sm_120` or `sm_120a`
   and for DGX Spark `sm_121`. Confirm the emitted architecture flags retain
   the architecture-specific feature suffix and run `make cuda-regression`.
8. Build with CUDA 12.8 or newer and require the CUDA translation units to
   compile warning-free, including the `FLT_MAX` users.
9. Force a conversation past the in-memory KV threshold, restore the same disk
   checkpoint twice, and confirm the checkpoint file remains present after
   both successful loads. Corrupt checkpoints must still be rejected.
10. Run `make dspark-verify-depth` with matching 0731 target and drafter files.
    Strict capture must skip layers without a compressor and compare every
    captured compressor layer.
11. Send the same long GLM 5.2 prompt twice to one server session. The second
    request must report `cache_source: memory-rewind`, reuse through one token
    before the prompt boundary, and produce the same greedy output as a fresh
    session.
12. Run `./ds4_test --think-tool-recovery`, then repeat through all three HTTP
    APIs. A complete tool block inside unclosed reasoning must be recovered
    once, preceding prose must remain reasoning, and no synthetic continuation
    may be generated.
13. Run `./ds4_agent_test` under ASan with agent-cache strings whose declared
    lengths exceed the remaining file. Loading must fail without allocating
    the declared size, and a valid cache must still load.
14. Run the server parser tests under UBSan with `NaN`, positive infinity, and
    negative infinity where integer JSON fields are expected. Conversion must
    be defined and clamped, with no sanitizer report.

## 3. Official Continuation Quality Gates

These tests are release-blocking after tokenizer, template, KV-cache, attention,
MoE routing, quantization, logit, or model-graph changes.  They are
teacher-forced continuation checks against hosted-model output and API
top-logprob slices, so do not replace them with one sampled chat answer.

- Build the scorer:
  `make -C gguf-tools quality-score`.
- Match every Flash GGUF to the fixture captured from the same checkpoint.
  The current release checkpoint is 0731 and uses
  `tests/test-vectors/flash-0731/`; the older undated GGUF uses the preserved
  `tests/test-vectors/flash-pre-0731/` fixture. Never report a cross-checkpoint
  failure as a quality regression. New checkpoints require a new
  `flash-CHECKPOINT/` directory before release QA; do not replace an older
  fixture. Checkpoint-labelled GGUFs such as `-0731` must use the fixture with
  the same label.
- Run the tracked DeepSeek V4 Flash 0731 smoke vectors:
  `DS4_TEST_MODEL=/path/to/0731.gguf
  DS4_TEST_VECTOR_FILE=tests/test-vectors/flash-0731/official.vec
  ./ds4_test --logprob-vectors`.
  This covers short prompts and long-prompt attention cases. The runner defaults
  to this fixture, but release logs should keep the path explicit.
- Run the 100-case DeepSeek V4 Flash fixture for every released Flash GGUF:
  `gguf-tools/quality-testing/score_official /path/to/deepseek-v4-flash.gguf gguf-tools/quality-testing/data/flash/manifest.tsv /tmp/flash.tsv 4096`.
  This manifest is also for the 0731 checkpoint. A later checkpoint needs a
  separately named 100-case fixture and must not be scored against this one.
- Treat the native MXFP4 Flash GGUF as a separate release artifact. Run the
  same 100-case fixture on Metal, resident CUDA, and CUDA SSD streaming when
  those backends are advertised; compare each result with the Metal baseline.
  For resident multi-GPU CUDA, pass the normal placement flags to the scorer,
  for example `--gpu-vram auto --gpu-devices 0,2,4,6,1,3,5,7
  --cuda-tensor-parallel`.
- Run the 100-case GLM 5.2 OpenRouter fixture for every released GLM GGUF:
  `gguf-tools/quality-testing/score_official models/GLM-5.2-UD-Q4_K_XL.gguf gguf-tools/quality-testing/data/glm52-openrouter-100/manifest.tsv /tmp/glm52-q4.tsv 4096`.
  Current Q4 XL reference band: first-token match `95/100`, API top-1 agreement
  about `0.942`, and API pair-order agreement about `0.880`.
- Run the 100-case GLM 5.3 Flash fixture separately for both release artifacts:
  `gguf-tools/quality-testing/score_official /path/to/GLM-5.3-Flash-Q2.gguf gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv /tmp/glm53-q2.tsv 4096`
  and repeat with `GLM-5.3-Flash-Q4_K.gguf`. The current Q2 reference is
  average NLL `0.458030488`, first-token match `89/100`, and average greedy
  prefix `7.37`; Q4 is `0.299917952`, `90/100`, and `9.66`. These are fresh
  GLM-5.3 Z.AI FP8 continuations and must not be replaced by GLM 5.2 fixtures.
  The Q4 layout with Q8 KDA projections, embedding, and output head scored
  `0.300804038`, `90/100`, and `9.48` on M3 Ultra. Its paired BF16-layout
  control scored `0.300477636`, `90/100`, and `9.48`; the Q8 layout won 54 of
  100 cases despite its `0.109%` higher aggregate NLL.
- Run the same GLM fixture for reduced-precision GLM release files.  The Q2
  routed reference is lower quality but should stay near first-token match
  `92/100`, API top-1 agreement about `0.890`, and API pair-order agreement
  about `0.800` unless the quantization changed deliberately.
- Match every PRO GGUF to its checkpoint. The June preview uses
  `gguf-tools/quality-testing/data/pro/manifest.tsv`; the August 0813 release
  uses `gguf-tools/quality-testing/data/pro-0813/manifest.tsv`. Never interpret
  a cross-checkpoint score as a quantization result.
- Run the 100-case DeepSeek V4 PRO 0813 fixture for the new release GGUF:
  `gguf-tools/quality-testing/score_official /path/to/deepseek-v4-pro-0813.gguf gguf-tools/quality-testing/data/pro-0813/manifest.tsv /tmp/pro-0813.tsv 4096 --ssd-streaming`.
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
- For native MXFP4 changes, run `make mxfp4-dot-test test-mxfp4-metal`, then a
  short greedy prompt and the section 3 continuation fixture with the MXFP4
  GGUF. The synthetic fused-MoE test and full-model quality gate must both pass.

### DSpark / DeepSpec Runtime

DSpark is opt-in, but it mutates the verifier, target-hidden capture, support
model loading, and scheduler paths.  Run these whenever DSpark support,
speculative verification, confidence/scheduler policy, target hidden capture,
tiny routed-MoE verifier kernels, or shared `--mtp-model` support-model code changes:

Use the 0731 DSpark support GGUF only with a Flash 0731 target. A support model
from another checkpoint can have plausible acceptance statistics while
producing a different greedy continuation.

Normal DSpark runs commit accepted target-verifier state directly. The batched
verifier and one-token decode use the same graph with different floating-point
operation order, so `output_match=0` against the baseline is diagnostic rather
than a failure. `--dspark-strict` remains the byte-identical target-only mode.

- Default greedy acceptance fixture:
  `DS4_DSPARK_MODEL=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-acceptance`.
- 64-token guardrail:
  `DS4_DSPARK_FIXTURE_TOKENS=64 DS4_DSPARK_MODEL=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-acceptance`.
- Opportunistic sampled acceptance fixture:
  `DS4_DSPARK_FIXTURE_TEMPERATURE=1 DS4_DSPARK_FIXTURE_TOP_P=0.95 DS4_DSPARK_FIXTURE_MIN_P=0.05 DS4_DSPARK_FIXTURE_SEED=12345 DS4_DSPARK_FIXTURE_TOKENS=32 DS4_DSPARK_MODEL=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-acceptance`.
  Require proposals, a direct commit, and zero errors. Output identity with the
  baseline is not expected: normally evaluated boundary tokens are sampled,
  while verified DFlash/target greedy matches are committed without sampling.
- Exact sampled acceptance fixture:
  repeat the command above with `DS4_DSPARK_FIXTURE_EXACT_SAMPLING=1`.
  This mode must preserve the requested target distribution. Seeded output
  identity is not expected because speculative accept/reject decisions consume
  additional random numbers.
- `make test` includes a 100,000-draw distribution check for general `p/q`
  correction and for the point-mass DFlash proposal used at runtime. Both
  histograms must remain within the stated tolerance of the target
  distribution.
- Fixed-block direct partial commit:
  `DS4_DSPARK_FIXTURE_CONFIDENCE=0 DS4_DSPARK_FIXTURE_TOKENS=8 DS4_DSPARK_FIXTURE_REQUIRE_PARTIAL=1 DS4_DSPARK_MODEL=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-acceptance`.
- DSpark verifier invariant smoke:
  `DS4_TEST_MODEL=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf DS4_DSPARK_SUPPORT=/Users/antirez/ds4/gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf make dspark-verify-depth`.
- If shared support-model or verifier structures changed, also run legacy MTP:
  `make mtp-verify-depth` with `DS4_TEST_MTP` set to a one-stage MTP support
  GGUF, or confirm the target skips only because the optional file is missing.
- Record `c_add` `accepted_draft`, `direct_full`, `direct_partial`,
  `replay_fallbacks`, `errors=0`, `verify_layer`, `net_saved`, and
  `output_match` for both 32-token and 64-token runs. At least one direct commit
  must occur. A faster run with lower proposal quality is a regression unless
  it was an intentional scheduler change.
- If verifier MoE kernels changed, run one diagnostic `c_add` profile with
  `DS4_DSPARK_VERIFY_SELECTED_PROFILE=1` or the Metal MoE stage profiler and
  record the selected-expert footprint or stage timing in the DSpark log.
- On Metal, benchmark at least one predictable code continuation and one
  deliberately unpredictable prose continuation at temperature 1. For the
  128-token M5 Max hash-table prompt, ordinary sampling measured
  41.71/44.49/44.73 t/s and opportunistic DSpark measured
  47.15/49.08/48.19 t/s, a median gain of about 8.3%. A surreal-prose control
  measured 44.46 t/s ordinary and 41.66 t/s opportunistic. DSpark remains
  opt-in because prompts with little useful speculation can still be slower.

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
  mixed prefill/decode call. Any nonzero differing-logit count is a blocker
  unless the model-specific section below declares a measured full-logit
  tolerance; argmax-only agreement is insufficient. When
  `DS4_TEST_LIVE_CONTROLS=1` is used, its interleaved serial evaluations are a
  correctness stress and must remain excluded from the reported batch timing.
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
  preserve both logs. GLM 5.3 uses native row batching below token 4096 in TP
  too; its model-specific tolerance applies. Other unsupported TP shapes must
  select their established ordered fallback.
- For the current TB5 MacBook link, US is `10.99.0.2` on `en1`/`rdma_en1` and
  IT is `10.99.0.1` on `en6`/`rdma_en6`; both use GID index 1. Before testing,
  require `rdma_ctl status` to report `enabled` and `ibv_devinfo -v` to show
  `PORT_ACTIVE` plus the corresponding `::ffff:10.99.0.x` GID. Force the
  device and GID with `--rdma-device NAME --rdma-gid-index 1` if automatic
  selection is ambiguous. A working TB IP ping alone is not RDMA evidence.
- Kill the TP worker during one batch with `DS4_TEST_TP_DISCONNECT=1` on the
  leader. The operation must fail cleanly, invalidate every affected session,
  and return control without hanging. When remote control latency exceeds the
  default one-second marker window, set
  `DS4_TEST_TP_DISCONNECT_DELAY_MS=10000` and signal the worker as soon as
  `TP_DISCONNECT_READY` appears. Repeat over TCP and RDMA by sending `SIGSTOP`
  to the exact worker PID at that marker, then resume it after the leader
  returns. Startup must report the default `gate-timeout=750ms`; both paused
  runs must fail the gate and invalidate all affected sessions without a Metal
  GPU watchdog error. Before accepting that deadline, run one normal
  GLM-5.3 batch over each transport and one GLM-5.2 IQ2 RDMA prompt using the
  larger 108 GiB shard. `DS4_TP_GATE_TIMEOUT_MS` is a diagnostic override, not
  a setting required for normal inference.
- Set `DS4_TEST_TP_IDENTITY_MISMATCH=1` on the test leader once and require both
  ranks to reject the hello before inference. Also reflect a leader hello from
  a test peer without changing its role; the leader must reject two peers that
  both claim the coordinator rank. Separately point a worker at an unused port
  with `DS4_TP_TIMEOUT_SEC=1`; it must return a connection error rather than
  retrying indefinitely.
- Verify unsupported combinations explicitly. GLM 5.2, GLM 5.3 after the
  4,096-token sparse boundary, DSpark support models, quality/reference modes,
  and CPU-router modes must use their established exact fallback or reject the
  combination before evaluation. GLM 5.3 below the boundary, including
  directional steering, and supported SSD-streaming configurations have native
  batching and must pass their model-specific oracle instead of being forced
  off.

## 5. Metal PRO Path

PRO support is experimental, but release builds must not break it silently.

- If a PRO-capable machine is available, run a short PRO q2 prompt and verify
  the correct template, thinking behavior, and endpoint aliases.
- For PRO Q4 distributed builds, test only on the intended high-memory machines.
- If PRO cannot be run locally, at least build all binaries and review changes
  touching model shape, tensor lookup, routed expert mapping, template logic,
  and KV payload compatibility.

## 6. GLM 5.2 And GLM 5.3

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
- Exercise integrated GLM MTP with `--mtp-timing` on a deterministic
  prompt. Compare the greedy text to
  a non-MTP run, require clean speculative cycles, and record acceptance and
  timing. Also run once with MTP disabled to prove ordinary decode remains the
  default.
- Run the Metal session oracle with 2 and 4 GLM 5.2 sessions. It must report
  `family=glm native_shared=0 native_qkv=0` and remain exact, including mixed
  prefill/decode; the DeepSeek-only row-grid kernels must not activate.
- Run resident and SSD-streaming GLM Q2 prompts with the same greedy input.
  Compare first token and top-logprob sanity, and record the selected full-layer
  prefix and dynamic expert-cache budget.
- Run physical two-machine GLM TP over TCP and RDMA with short and long prompts.
  Record prefill/decode speed, transport, rank residency, and clean shutdown.
  The long prompt must cross the 4,096-token indexed-attention boundary. After
  changes to split attention, score at least 100 teacher-forced tokens beyond
  that boundary against the unsplit reference and run the 100-case Q2 fixture
  once through physical TP, preserving its `summary` and `api_summary` lines.
  Repeat one run with `--tensor-parallel-token-prefill` as the exact-arithmetic
  diagnostic. Test both Q2 and Q4 routed-expert files whose types have
  ownership-aware GLM TP kernels. Each rank must map only its owned experts and
  match the accepted single-host graph. A routed type without ownership-aware
  kernels must still reject clearly before evaluation rather than loading a
  partial split or hanging.
  The released GLM 5.2 IQ2_XXS file keeps `indexer.proj.weight` in FP32. Its
  loader and Metal graph must accept that established layout; GLM 5.3 may use
  its quantized or BF16 indexer projection instead. A short exact-output smoke
  on the current two-M5 RDMA setup returned `GLM52_OK` with 108.63 GiB planned
  per rank.
- With explicit permission for the current QA pass, run one resident GLM Q2
  prompt, a long-context prompt, integrated GLM MTP, and concurrent server
  requests on the eight-GPU CUDA host. Use ordinary eight-GPU layer placement
  for GLM; do not pass the Flash-specific
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
- Compare a non-tool prompt and a complete assistant tool-call/tool-result
  transition against the model's `chat_template.jinja` byte for byte. Include
  reasoning effort, a client system message, a function description and JSON
  schema, assistant reasoning, and an observation. Token-count agreement alone
  is insufficient. The fixed one-tool fixture is 901 bytes with SHA-256
  `f1718f3ebb1c41532bcd5eedd9ebd5c84ae930b018e31962d8743efdbc5affc3`.
  Run `./ds4_test --server` as the model-free regression for the same exact
  schema instructions and transition delimiters.

### GLM 5.3 Flash

GLM 5.3 adds mHC, recurrent KDA, a pool-4 DSA indexer, and a different MTP
block. A GLM 5.2 pass does not cover these paths.

- Run the section 3 GLM 5.3 Q2 and Q4 100-case fixtures before and after any
  graph, quantization, attention, KDA, mHC, TP, or cache change.
- Build and run the focused primitive test:
  `make tests/test_glm53_kda && ./tests/test_glm53_kda`.
  It covers BF16 projections, pool-4 state construction and expansion, grouped
  scorer arithmetic and causal visibility, recurrent KDA prefill versus
  sequential decode, and exact repeated causal-attention output on ROCm.
- Treat session construction as the attention-memory admission point. Every
  owned DSA cache and indexer pool/tail, every KDA recurrent state, and the
  complete supported prefill workspace must allocate before a request is
  accepted. A first prefill must not grow a per-layer cache. Check both a
  4,096-token session and a long session in the memory report.
- Keep GLM-5.3 in absorbed MLA form: attend densely over the shared compact
  latent cache through token 4,096, then use the pool-4 sparse selector. Do not
  restore the 2.75 GiB expanded per-head K/V cache as a presumed quality fix.
  The complete Q2 fixture on the compact Metal graph scored NLL `0.458177271`,
  first-token agreement `90/100`, and average greedy prefix `7.390`, matching
  the accepted release band. Fresh Z.AI FP8 long-context controls also favored
  compact attention: weighted NLL was `0.539823254` versus `0.820997888` for
  expanded K/V over 24 synthetic cases, and `0.808160860` versus `0.820309487`
  over 12 natural source-context cases. Extending dense attention to 16K made
  the natural set worse at `0.825709092`, so keep the 4K crossover.
- At 100K on an M5 Max, require the compact Q2 plan to remain near 94.09 GiB:
  89.87 GiB model, 1.11 GiB compact history, and 3.11 GiB fixed graph buffers.
  An 8,192-token one-shot control on the same graph reached 479.09 prefill and
  29.89 steady decode t/s. Repeat the continuation fixture after changing the
  compact cache type, absorbed projections, FlashAttention staging, or the 4K
  crossover; numerical similarity to the old expanded graph is not the gate.
- On this 128 GB M3 Max, run the resident Q2 through the generic non-NAX Metal
  path. Repeat the 4,096-4,100 boundary, official-continuation, MTP, snapshot,
  server-session, and continued-prefill gates used on M5. Record the different
  M3 performance floor rather than borrowing the M5 result. Run Q4 only with a
  bounded SSD-streaming cache; never try to make it fully resident.
- On one M5 Max, run resident Q2 with a prompt whose actionable instruction
  begins after token 4096. The model must recover the tail instruction and
  complete a tool or exact-output task. A short coherent continuation at token
  4109 is not sufficient: it previously missed a broken sparse selector.
- Run physical Q4 50/50 TP over the explicit TB5 RDMA devices. Graduate context
  allocation through 10K, 25K, and 50K, checking both ranks before advancing.
  The original 50K Q4 run reported 102.20 GiB per rank. Correct accounting for
  the dynamically allocated mHC/KDA prefill workspace adds about 0.44 GiB, so
  the expected plan is about 102.64 GiB per rank; record the exact new value on
  the next run. It remains below the fixed 110 GiB ceiling. Swap must not grow
  from idle, SSH must stay responsive, and both roles must exit cleanly.
- The 25K gate must place a read/edit/test task after a long inert archive and
  include one harmless tool failure that the agent must recover from. The 50K
  gate must contain at least 30K live prompt tokens and a real source repair:
  reproduce a failing test, edit the implementation without weakening tests,
  and pass a warning-strict build. Inspect the resulting diff manually.
- Confirm the logs select `rdma_en1` with GID 1 on US and `rdma_en6` with GID 1
  on IT. A TCP fallback does not satisfy this gate. Keep Q2 and Q4 in
  `~/ds4/gguf` on both hosts after testing.
- Run ordinary greedy decode, opportunistic MTP, and `--mtp-exact-sampling`.
  Greedy MTP must preserve the accepted continuation and provide a measured
  gain; the current short Q2 control improved from 34.21 to 41.97 t/s.
- After directional-steering changes, verify that a zero `45 x 4096` GLM vector
  is output-identical to the unsteered CLI for both FFN and attention hooks. A
  46-row file must be rejected with the expected 737,280-byte size. Build a
  `/path/to/glm53-direction.f32` test vector with the command in
  `dir-steering/README.md`, then run it through ordinary decode, `--mtp`, and a
  two-session `ds4-server` smoke. Run the native batch oracle with:

  ```sh
  DS4_TEST_MODEL=/path/to/GLM-5.3-Flash-Q2.gguf \
  DS4_TEST_SESSION_COUNT=4 DS4_TEST_LOGIT_TOLERANCE=0.001 \
  DS4_TEST_DIRECTIONAL_STEERING_FILE=/path/to/glm53-direction.f32 \
  DS4_TEST_DIRECTIONAL_STEERING_FFN=1 \
  DS4_TEST_DIRECTIONAL_STEERING_ATTN=0.25 \
  make test-metal-session-batch
  ```

  It must cover native decode and mixed prefill/decode without changing any
  selected token. Repeat a short physical TP run over RDMA with the same file
  and scales on both ranks. On CUDA and ROCm release targets, require a
  warning-free build and one short steered GLM 5.3 Q2 prompt. Finally rerun a
  held-out target/control sweep; an effective edit that makes control answers
  repetitive or incoherent does not pass.
- Run the two- and four-session GLM 5.3 server oracle below token 4096 on one
  M5 and physical TP. For both native single-M5 and TP paths, set
  `DS4_TEST_LOGIT_TOLERANCE=0.001`: row-batched reductions may differ from the
  serial launch order, but every selected token must match and the maximum
  full-logit delta must remain below that bound. Also run the serial rollback
  with zero tolerance. The current four- and eight-session Q2 maxima are both
  `0.0001297`; the current six-step two-session RDMA maximum is `0.000175238`. Past
  4096, require the exact ordered fallback until a sparse native batch oracle
  proves full-vocabulary correctness.
- Exercise every pool remainder at the dense-to-sparse boundary with prompts
  ending at tokens 4096 through 4100. Token 4096 must remain bit-identical to
  the serial control. The four sparse cases must retain the same greedy token,
  contain no nonfinite logits, and pass a multi-token exact-output task. Small
  batched-reduction logit differences are acceptable only when the official
  continuation and long-task gates remain in band. Build each prompt against
  `--dump-tokens`; word counts are not a valid substitute for rendered-token
  counts. The current `.180` CUDA reference returned exactly `BOUNDARY_OK` at
  all five frontiers.
- Run the session snapshot test across the sparse boundary:
  `DS4_TEST_MODEL=/path/to/GLM-5.3-Flash-Q2.gguf
  DS4_TEST_SNAPSHOT_PROMPT=/path/to/a-4k-plus-prompt.txt
  DS4_TEST_SNAPSHOT_CTX=8192 ./ds4_test --session-snapshot`.
  Restored top logits before and after one continued token must match the
  uninterrupted session within the test's `1e-6` tolerance.
- Repeat that command with `DS4_TEST_GLM_MTP=1`. The test must replay 16
  integrated-MTP cycles across the snapshot, including both one- and two-token
  outcomes, and match every committed token plus the final top-eight logits.
  The M5 Max reference produced 10 one-token cycles, 6 two-token cycles, and
  22 committed tokens without a verifier failure. It then synced back to the
  original long prompt, reproduced its top-eight logits within `1e-6`, and
  completed four more MTP cycles. The current `.180` CUDA run produced 5
  one-token and 11 two-token outcomes, committed 27 tokens, and also passed.
- Measure continued prefill as actual appends to one live session, not only as
  a single cold prompt. On an M5 Max, run:
  `./ds4-bench -m /path/to/GLM-5.3-Flash-Q2.gguf --metal
  --prompt-file /path/to/a-25k-prompt.txt --ctx-start 4096 --ctx-max 12288
  --ctx-alloc 16384 --step-incr 2048 --gen-tokens 0 --csv /tmp/glm53.csv`.
  The current 2K append results are 452.56, 423.74, 409.79, and 396.72 t/s at
  4K, 6K, 8K, and 10K resident prefixes. A warmed first append below 350 t/s
  requires investigation. The serial rollback control measured 29.68 t/s.
  On `.180` CUDA, the current 4K, 6K, and 8K append results are 522.66, 506.20,
  and 502.50 t/s.
- Keep the 33987-token Q4 TP agent run as the final long-state gate. The old
  serial sparse path took about 24 minutes to reach its first tool call. The
  batched path processed a 34023-token initial suffix in 109.846 seconds
  (309.74 t/s), completed the full read/edit/test task in 178.93 seconds, and
  used explicit RDMA. A faster result must still pass the tail task, fixture
  inspection, and official-continuation gates.
- On one DGX Spark, run Q2 through CUDA and repeat the primitive, official
  continuation, 4,096-4,100 boundary, continued-prefill, snapshot, MTP, server,
  and coding-agent gates. Validate independently on `.180` and `.181`; they are
  separate single-host runs, not CUDA TP. Q4 and Spark-to-Spark RDMA are not
  supported in this pass. The accepted `.180` 100-case reference is average
  NLL `0.461783551`, first-token agreement `90/100`, and average greedy prefix
  `7.49`.
- For the default compact CUDA graph, dump the complete first-token logits at
  a 1,024-token frontier twice with the same context allocation. Both files
  must be byte-identical. Run the 100-case GLM-5.3 fixture from the same linked
  objects, then run the two-session single-GPU oracle with
  `DS4_TEST_CUDA_SINGLE_GPU=1 DS4_TEST_SESSION_COUNT=2`. Require
  `nonexact_logits=0`. Finally run the fused D2R kernels under CUDA memcheck;
  an argmax-only comparison or coherent text does not replace these gates.

### GLM 5.3 Vision

Vision is a separate sidecar on Metal, single-GPU CUDA, and ROCm, and has its
own release gate. Text-only GLM success does not exercise image preprocessing,
the vision graph, multimodal prompt spans, or image-aware KV identity.

- Download `glm53-vision` and verify
  `GLM-5.3-Flash-Vision-Encoder.gguf` has SHA-256
  `ae23e14c6979e889051b2e4a39351abcdafb161e18e606fae4d8c40095a4bf3a`.
- Build `tests/test_glm53_vision_engine` and
  `tests/test_glm53_vision_prompt`. Run them with the release Q2 text GGUF, the
  vision sidecar, and a fixed PNG. The prompt test must generate a visual
  answer, reuse an unchanged image without repeated prefill, and rebuild when
  only the image fingerprint changes. It must also hold image-token positions
  fixed, replace the visual embedding with zeros, and observe changed output
  logits. Use a 4,096-token test session so a large screenshot plus the output
  cannot hit the old 2,048-token test ceiling. After explicit session
  invalidation, and again after restoring the zeroed embedding, require the
  complete image-conditioned logits to match the original within `1e-6`.
  This catches a compact-prefill path that processes placeholders but silently
  ignores the image data, as well as incomplete multimodal state rebuilds.
- Keep one accepted Metal embedding from a fixed image and compare CUDA and
  ROCm output with `tests/compare_glm53_vision_embeddings.py`. Require finite
  output, cosine similarity at least `0.995`, mean absolute error at most
  `0.001`, and maximum absolute error at most `0.06`. This permits normal BF16
  GEMM ordering differences but rejects a changed vision graph.
- Run a fixed model-level vision fixture containing photographs, screenshots,
  diagrams, readable text, spatial questions, and unrelated-image controls.
  Compare complete answers with the official GLM-5.3-Flash vision service and
  repeat through CLI, server, and `ds4-agent`. A valid decoder, expected image
  token count, or plausible but ungrounded prose does not pass this gate. With
  a local vision-enabled server running, require:

  ```sh
  python3 tests/run_glm53_vision_quality.py
  ```

  to report `6/6 passed`. For agent tests, expose only the raster fixtures;
  source SVGs or expected-answer files beside them let the agent bypass vision
  with text tools.
- Run the decoder over RGB, RGBA, grayscale, and palette PNG, baseline and
  progressive JPEG, EXIF orientation, truncated files, wrong CRCs, huge
  dimensions, and decompression-bomb fixtures under ASan and UBSan. Invalid
  files must fail without a sanitizer report or large allocation.
- In `./ds4`, submit one PNG and one JPEG with `/read`, then continue each chat
  with a text turn. Repeat once with `--mtp`; verification after image prefill
  must complete without a GLM MTP failure.
- In `ds4-agent`, require `view_image` to inspect a real file and use the
  resulting multimodal observation in a later read/edit/test tool loop. Image
  observations must enter as user-role multimodal turns; GLM loses grounding
  when several images are packed into a tool-response role. Run the five-image
  fixture in one turn so the prompt exceeds 4K, and require the same facts as
  the official Z.AI control. Text-only tool observations must remain tool-role.
- Through `ds4-server`, test OpenAI Chat data URIs, Responses `input_image`,
  and Anthropic base64 image blocks. Include two images in one message and an
  image in a later turn. Local paths, `file:` URLs, remote URLs, malformed
  base64, unsupported media, more than 16 images, and bodies over 64 MiB must
  return 4xx without reading local files or making network requests.
- Run Q4 across `mac-m5max-us` and `mac-m5max-it` over explicit TB5 RDMA with
  `--vision` on both ranks. The leader must encode once, both ranks must keep
  matching multimodal KV state, and the answer must remain correct. Record
  image encode, prefill, first-token, and decode timing separately.
- Build CPU, CUDA, and ROCm targets warning-free after vision changes. Run the
  encoder comparison, prompt replay test, and six-case server fixture on one
  DGX Spark and on `strixhalo`; ROCm Q2 must use bounded SSD streaming.

## 7. SSD Streaming

SSD streaming is a capacity path, so test both correctness and user experience.

- Flash q2/q2-q4 streaming:
  `./ds4 -m ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 32GB -p "..."`
- Regression test mixed-quant Flash SSD streaming. Use the mixed q2/q4 GGUF
  with boosted Q4 routed-expert layers and a prompt long enough to exercise the
  selected-address prefill path; it must not fail with "model range is not
  covered by mapped model views":
  `./ds4 -m gguf/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf --ssd-streaming --ssd-streaming-cache-experts 16GB --ctx 4096 --tokens 1 --nothink --prompt-file /tmp/ds4_600tok_prompt.txt`.
- Cold streaming measurement:
  run once with `--ssd-streaming-cold` and verify no deadlock, missing expert,
  or impossible slowdown.
- Confirm startup reports cache budget and that generation does not stall on
  repeated expert misses for a small interactive prompt.
- After changing model-map or memory accounting, test automatic sizing, an
  impossible large target such as `--ssd-streaming-cache-experts 500GB`, and
  `--ssd-streaming-cache-experts 1`. The large target must be reduced below
  the final memory-guard budget instead of failing or pressuring the machine
  into swap. The one-slot run must select direct per-layer reads and complete
  correctly without pretending that the selected-expert cache can hold one
  token's routed set. Preserve the startup lines showing the effective cache,
  global or per-layer decode map, and total planned memory.
- If streaming cache internals changed, test the same prompt twice and compare
  first-token/logprob sanity between runs.
- On an idle M5 Max, run the full GLM 5.3 Q2 SSD-streaming regression with the
  16 GiB expert budget. Use the release GGUF and verify its checksum before
  comparing results. The current reference file is
  `GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf`, SHA-256
  `059b36accd4c9acf73099da9f703b574d627869d619b7c4c316aa856e33d472e`.
  Discard one warm-up run, then take the median of three runs of each command:

  ```sh
  GLM_SSD_MODEL=/path/to/GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf

  ./ds4 -m "$GLM_SSD_MODEL" --ssd-streaming \
    --ssd-streaming-cache-experts 16GB --ctx 1024 --tokens 16 \
    --nothink --temp 0 --seed 1 \
    -p "$(head -c 2500 tests/test-vectors/glm-openrouter/prompts/long_memory_archive.txt)"

  ./ds4 -m "$GLM_SSD_MODEL" --ssd-streaming \
    --ssd-streaming-cache-experts 16GB --ctx 1024 --tokens 64 \
    --nothink --temp 0 --seed 1 \
    -p "Write the word apple exactly 100 times, separated by one space. Do not stop early and output nothing else."
  ```

  The first command must report 463 input tokens, produce a coherent answer
  about component gamma, and keep median prefill at or above 11.3 t/s. The
  second must emit all 64 requested output tokens and keep median generation at
  or above 5.5 t/s. The M5 Max reference medians are 12.59 and 6.14 t/s. Startup
  should plan about 18.03 GiB at this context and initially restrict the model
  map to the token embedding. GLM must demand-fill its expert cache by default;
  an ordinary run and `--ssd-streaming-cold` should have comparable cache-miss
  counts and speed unless an explicit preload count or diagnostic cap is used.
  A memory guard or static decode map that accounts nearly the full 196.58 GiB
  GGUF, a Metal OOM, repeated garbage tokens, or a compact-attention result
  that omits the RoPE score is a release blocker.

## 8. CUDA / DGX Spark

Before a release, ask the user for CUDA access if it is not already configured.
Use either DGX Spark / GB10 host, `toor@192.168.4.180` or
`toor@192.168.4.181`. Do not claim CUDA is release-ready without this pass.

Both Sparks normally run vLLM. Before stopping it, record its process, service
or container, model, ports, and exact launch command. Confirm all vLLM workers
have exited before loading DwarfStar. At the end, stop every DwarfStar process,
restore the exact vLLM service, and verify its original ports and model health.
Do not use high-performance Hugging Face Xet mode while vLLM is resident.

- Fetch or push the exact release commit to the CUDA machine.
- Build:
  `make clean && make cuda-spark`.
- Require both the DGX Spark build and the eight-GPU CUDA build to complete
  without compiler warnings. The eight-GPU build is performed only after
  receiving explicit permission to use `192.168.60.250` for this QA pass.
- Run:
  `make cuda-regression`.
- For native MXFP4 changes, run
  `make test-mxfp4-cuda CUDA_ARCH=native` on the multi-GPU CUDA host only after
  receiving explicit permission for `192.168.60.250`, and
  `make test-mxfp4-cuda CUDA_ARCH=sm_121` on DGX Spark. Dense MMQ, routed MMQ,
  routed MMVQ, fused gate/up, and fused down must pass. The Spark run must also
  pass the Blackwell K-tile guard. This synthetic parity test does not replace
  full-model continuation scoring.
- With that permission, run the native MXFP4 GGUF resident on the multi-GPU
  host, and run it with `--ssd-streaming` on DGX Spark. Use the same greedy prompt and continuation
  fixture on both. Record prefill and generation speed, require finite logits,
  and compare quality with the Metal MXFP4 result. Blackwell MMQ quantizes
  activations to native FP4 for batched work; decode MMVQ keeps Q8 activations,
  so quality must be checked rather than inferred from kernel-only parity.
- Run a short CLI prompt with the Flash GGUF and record generation t/s.
- Run a longer prompt that exercises routed experts past a few thousand tokens.
- With explicit permission for this QA pass, run the full-vocabulary decode
  oracle on the eight-GPU CUDA host:
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
- With explicit permission for the eight-GPU host, start `ds4-server` with 8
  and 16 batched sessions and issue at least that many simultaneous requests
  with mixed prompt lengths. Verify no session mix-up, deadlock, or starvation
  and record aggregate generation throughput.
- On DGX Spark, verify the same public batch API and server concurrency use the
  single-GPU fallback without creating peer-only TP/EP state. The eight-GPU
  native oracle is not a valid Spark test because its topology is intentionally
  unavailable there.
- For GLM 5.3, use the resident Q2 artifact only. Require the dedicated CUDA
  primitive and continuation gates in section 6, then record prefill,
  generation, MTP, continued-prefill, server aggregate throughput, and peak
  memory. Do not attempt the 178 GiB Q4 artifact on one 128 GB Spark.
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
- After MXFP4 or ROCm routed-MoE changes, run `make test-mxfp4-rocm`. Require
  zero `failures` for both `mid` and `out` at 1, 3, 32, 128, and 512 tokens,
  followed by `MXFP4 ROCm routed MoE: PASS`.
- Use the q2 Flash imatrix GGUF for release smoke tests:
  `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`.
- Do not use the mixed q2-q4 or Q4 Flash GGUFs for routine Strix Halo QA yet.
  They are dangerous on this machine for now because the ROCm path can hit
  system OOM instead of failing cleanly.
- Run a short CLI prompt:
  `./ds4 -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf --ctx 4096 --nothink -p "Reply with exactly: OK"`.
- For DeepSeek Flash and GLM 5.3 Flash decode, confirm the default path uses
  prequantized Q8 activations. Repeat the same greedy run with
  `DS4_ROCM_Q8_PREQUANT_DECODE=0` only as a diagnostic control. The default
  must be materially faster and must still pass the matching continuation-
  quality gate. `--quality` must stay on the full-FP32 activation path.
- Test DSpark with the matched 0731 target and support files:
  `DS4_BIN=./ds4 DS4_DSPARK_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf DS4_DSPARK_SUPPORT=gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf DS4_DSPARK_FIXTURE_TOKENS=64 sh tests/dspark_acceptance_fixture.sh`.
  Require proposals, accepted draft tokens, at least one direct state commit,
  zero verifier errors, and zero replay fallbacks. Record ordinary and DSpark
  generation speed separately. When direct verifier-state handling changes,
  also compare with a test-only build of its immediate replay predecessor; the
  direct build must be faster. DSpark is not currently expected to beat
  ordinary ROCm decode, so do not describe it as a ROCm speedup without a new
  measurement.
- Repeat one opportunistic DSpark run with
  `--temp 1 --top-p 0.95 --min-p 0.05`, then repeat it with
  `--mtp-exact-sampling`. The current 128-token code references are
  16.26 t/s ordinary, 12.28 t/s opportunistic, and 13.52 t/s exact, with no
  verifier errors. This is a correctness gate, not a ROCm speed claim; the
  ROCm batched verifier is still too expensive.
- Run one longer prompt if ROCm kernels, backend hooks, tensor loading, model
  cache, KV cache, or graph prefill code changed.
- Run the GLM Q2 release model through ROCm SSD streaming with at least four
  generated tokens:
  `./ds4 --rocm -m gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf --ssd-streaming --ctx 4096 --nothink --tokens 4 -p "Reply with exactly: OK"`.
  Startup must select a cache budget that passes the memory guard without an
  override, and both compact indexed prefill and decode must complete.
- Repeat the ROCm GLM smoke with an overlarge byte target and with a one-expert
  target. The byte target is a hint and must be reduced using current Linux
  `MemAvailable` as well as the backend limit. The one-expert target must use
  the per-layer fallback. After each run, confirm SSH remains responsive and
  no OOM kill, GPU reset, or reboot was recorded.
- Run one longer GLM prompt with the release-advertised Strix context after
  changes to GLM attention, typed quantized projections, streaming expert
  caches, or memory budgeting. Record the context, cache split, and whether
  the continuation stays free of token-corruption markers.
- Run the same GLM model with `--mtp-timing --temp 0`. At least one draft
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
- In batched mode, close clients while their requests are queued, prefilling,
  and streaming decode. Repeat across OpenAI chat, Responses, Anthropic, and
  completions. Abandoned work must stop at the next backend-safe boundary, and
  a valid request after each cancellation must complete normally.
- For the repeatable chat-completions cancellation and slot-reuse gate, run
  `python3 tests/test_server_batching.py --url http://127.0.0.1:8000 --pairs 2
  --workers 4 --case short-sampled --max-tokens 12 --cancel-first 4`. Then run
  at least twelve short four-request waves against the same four-slot server.
  Every pair must remain deterministic and the server must answer `/v1/models`
  after malformed JSON and an over-context request.
- Only after receiving explicit permission for this QA pass, start
  `ds4-server` on the eight-L40S CUDA TP target with the release TP options and
  verify all 16 100k-context sessions allocate. Startup must report a
  2048-token prefill cap; a silent fallback to 4096 is an OOM regression.
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
  create a temp project and ask for edits. By default, verify that exact old/new
  replacements work and the tool prompt does not advertise `[upto]`. In a
  separate `--edit-upto` run, verify anchored edits fail safely on ambiguous
  matches and do not require retyping whole files.
- Real coding edit loop:
  delete `/tmp/mymandel`, ask ds4-agent to create a small C ASCII Mandelbrot
  program there, build and run it, then in a second user turn ask for a small
  modification that should naturally use the edit tool, such as changing the
  ASCII character ramp or output dimensions.  Verify the agent edits the
  existing file instead of rewriting the whole project, and that the final
  program still builds and runs.
- For GLM-5.3 integrated MTP, repeat a long-context edit/build/test task after
  the prompt crosses token 4,096. Require real file and shell tool calls,
  accepted and rejected draft cycles, and no `glm mtp: GLM 5.3 verify failed`
  message. Continue the same session after one harmless tool error and after a
  snapshot restore so a hidden MTP fallback or damaged recurrent state is not
  mistaken for success.
- With a matched DSpark support file and temperature 1, repeat a coding-tool
  turn that crosses sampled prose, greedy DSML structure, parameter text, and
  back to sampled prose. Require the tool to execute, the final answer to be
  valid, and DSpark stats to show zero verifier errors and replay fallbacks.
  Run the opportunistic default and `--mtp-exact-sampling`. The M5 Max
  opportunistic smoke created, compiled, and ran a C program printing
  `OPPORTUNISTIC_OK`, accepting 196 of 235 draft tokens.
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

## 16. Speed Regression

Performance is a release gate. A correct result that is unexpectedly much
slower still needs an explanation before release.

Use the same commit, GGUF checksum, prompt, context frontier, generated-token
count, power setting, and backend flags as the reference run. Let the machine
become idle, discard the first warm-up run, then record the median of three
runs. Do not compare different model checkpoints or quantizations. For batched
tests, record aggregate and per-session decode speed.

- A slowdown over 5% requires a clean rerun and investigation.
- A repeatable slowdown over 10% in prefill, decode, or aggregate batched
  decode is a release blocker unless the change and tradeoff are documented.
- Keep the complete `ds4-bench` CSV. A single short-prompt average is not enough
  to detect a context-dependent regression.
- Compare startup time and peak memory as well as tokens per second when model
  loading, caches, streaming, or temporary arenas changed.
- Run the backend-specific batch tests in sections 4 and 8. Fast single-session
  decode does not substitute for aggregate multi-session throughput.

These are the last known good observations available when this gate was added.
They are reference points for matching hardware and workloads, not performance
claims across different models or contexts.

| System and backend | Model and workload | Prefill | Decode |
| --- | --- | ---: | ---: |
| MacBook Pro M3 Max 128 GB, Metal | Flash q2, 11,709-token prompt | 250.11 t/s | 21.47 t/s |
| MacBook Pro M3 Max 128 GB, Metal | Flash 0731 q2, opportunistic temperature-1 128-token code prompt | - | 29.16 t/s ordinary; 28.18 t/s DSpark |
| MacBook Pro M5 Max 128 GB, Metal | Flash q2, 11,707-token prompt | 463.44 t/s | 25.90 t/s |
| MacBook Pro M5 Max 128 GB, Metal | Flash 0731 q2, opportunistic temperature-1 128-token code prompt | - | 44.49 t/s ordinary median; 48.19 t/s DSpark median |
| Mac Studio M3 Ultra 512 GB, Metal | Flash q2, 11,709-token prompt | 468.03 t/s | 27.39 t/s |
| Mac Studio M3 Ultra 512 GB, Metal | Flash q4, 12,018-token prompt | 448.82 t/s | 26.62 t/s |
| Mac Studio M3 Ultra 512 GB, Metal | GLM 5.3 Flash Q4 with Q8 KDA/head, 2,048-token prompt | 437.62 t/s | 24.74 t/s |
| Two M5 Max 128 GB Macs, Metal TP over TB5 RDMA | GLM 5.2 IQ2_XXS, 4,096-token prefill and 256-token teacher-forced decode | about 214 t/s | about 16.7 t/s |
| MacBook Pro M5 Max 128 GB, Metal | GLM 5.3 Flash Q2, resident short prompt | 86.68 t/s | 34.45 t/s; 41.97 t/s greedy MTP |
| MacBook Pro M5 Max 128 GB, Metal | GLM 5.3 Flash Q2, 8,192-token compact-attention prompt | 479.09 t/s | 29.89 t/s steady |
| MacBook Pro M5 Max 128 GB, Metal | GLM 5.3 full Q2, SSD streaming with 16 GiB expert budget; 463-token prefill / forced 64-token decode | 12.59 t/s median | 6.14 t/s median |
| Two M5 Max 128 GB Macs, Metal TP over TB5 RDMA | GLM 5.3 Flash Q2, short prompt | 29.06 t/s | 32.70 t/s |
| MacBook Pro M5 Max 128 GB, Metal | GLM 5.3 Flash Q2, 24,988/49,948-token long prompts | 424.80 / 421.75 t/s | 29.50 / 28.10 t/s |
| Two M5 Max 128 GB Macs, Metal TP over TB5 RDMA | GLM 5.3 Flash Q2, 10,819-token prompt | 468.97 t/s | 22.85 t/s |
| Two M5 Max 128 GB Macs, Metal TP over TB5 RDMA | GLM 5.3 Flash Q4, 34,023-token agent prefill suffix | 309.74 t/s | full coding task: 178.93 s wall |
| MacBook Pro M5 Max 128 GB, Metal | GLM 5.3 Flash Q2, 2/4/8 resident decode sessions | - | 54.49 / 73.96 / 86.17 aggregate rows/s |
| DGX Spark GB10, CUDA | GLM 5.3 Flash Q2, 2,048-token prefill and 16-token decode | 531.39 t/s | 14.35 t/s at 2,048 context |
| DGX Spark GB10, CUDA | GLM 5.3 Flash Q2, live-session 4K/6K/8K continued prefills | 522.66 / 506.20 / 502.50 t/s | - |
| DGX Spark GB10, CUDA | GLM 5.3 Flash Q2, strict two-session 24-step oracle | - | 16.5 aggregate rows/s; byte-exact full logits |
| DGX Spark GB10, CUDA | Flash q2, 7,047-token prompt | 343.81 t/s | 13.75 t/s |
| DGX Spark GB10, CUDA | Flash q2 DSpark, 64-token C fixture | - | 24.48 t/s direct; 13.93 t/s replay predecessor |
| DGX Spark GB10, CUDA | pre-0731 Flash q2, exact-sampled 128-token code prompt | - | 18.17 t/s ordinary; 18.30 t/s DSpark |
| DGX Spark GB10, CUDA | pre-0731 Flash q2, opportunistic temperature-1 128-token code prompt | - | 18.32 t/s ordinary; 18.43 t/s DSpark |
| Strix Halo gfx1151, ROCm | Flash IQ2 resident, short section 9 smoke | - | 17.27 t/s; FP32 rollback 9.70 t/s |
| Strix Halo gfx1151, ROCm | Flash IQ2 resident, 4,096-token context | - | 14.82 t/s; FP32 rollback 8.76 t/s |
| Strix Halo gfx1151, ROCm | GLM 5.3 Flash Q2 resident, 64-token prompt and 128-token decode | 47.18 t/s median | 14.25 t/s median steady; 8.64 t/s FP32 rollback |
| Strix Halo gfx1151, ROCm | GLM 5.3 Flash Q2 resident, 4,096-token prefill | 80.08 t/s; scalar-attention rollback 23.35 t/s | - |
| Strix Halo gfx1151, ROCm | Flash IQ2 DSpark, 64-token C fixture | - | 11.40 t/s direct; 9.77 t/s replay predecessor; 16.70 t/s ordinary |
| Strix Halo gfx1151, ROCm | Flash 0731 IQ2, exact-sampled 128-token code prompt | - | 16.55 t/s ordinary; 12.68 t/s DSpark |
| Strix Halo gfx1151, ROCm | Flash 0731 IQ2, temperature-1 128-token code prompt | - | 16.26 t/s ordinary; 12.28 t/s opportunistic; 13.52 t/s exact |
| 8x L40S, CUDA TP | Flash q4, 2,048-token prefill benchmark | 1,524.84 t/s | 46.93 t/s |
| 8x L40S, CUDA TP | Flash q4, 16-row decode oracle | - | 126.0 aggregate t/s |

The 8x L40S values are retained from the last recorded run on `192.168.60.250`.
They are historical references only: never connect to that host or interrupt
its production server without explicit permission for the current QA pass. If
permission is granted, the existing hard floor remains 110 aggregate t/s for
the 16-row decode oracle.

## 17. Release Sign-off

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
- The speed-regression gate passed on every validated backend, with any skipped
  baseline or intentional slowdown documented.
- Metal 2/4/8/16-session exactness and forced fallback gates passed.
- Physical Metal TP batching and CUDA native decode/mixed batching passed when
  those backends are part of the release.
- Any skipped item is written down with the reason.
