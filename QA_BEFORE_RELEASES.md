# QA Before Releases

This is the release gate for DwarfStar.  Run it before tagging or pushing a
release build.  The goal is not to prove every code path exhaustively; it is to
exercise the paths that have historically regressed: Metal graph inference,
CUDA, ROCm, SSD streaming, distributed execution, disk KV cache, server APIs, and the
agent TUI/tool state machine.

Keep this file procedural: commands, pass/fail criteria, safety constraints and
reproducible reference measurements. Put per-run results, failed experiments,
retries and investigation notes in a separate QA report, not here.

Do not run multiple huge model processes at the same time. Record the commit,
hardware, GGUF checksum, prompt, context size and non-default flags for every
manual run. Report skipped checks and unresolved failures explicitly.

Preferred release test hosts:

- CUDA / DGX Spark: `antirez@dgxspark1` and `antirez@dgxspark2`.
  Use their direct 200 Gb/s RoCE link for network TP, not the VPN or WiFi.
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
- On Metal, run `make test-metal-command-memory` after command-buffer or copy
  changes. With API validation enabled, repeated compute/copy/flush batches
  must preserve their output and keep the CPU heap bounded after warmup.
  The target also checks row, session, speculative and large-prefill TP gates
  with ordered loopback exchanges; physical RDMA remains a separate test.
  Long model runs must not accumulate temporary driver objects per token;
  fixed KV and expert-cache budgets alone do not rule out this growth.

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
    captured compressor layer. Repeat with the matching Vision Exp pair.
    The test also verifies a six-token seed-plus-draft block and restores each
    retained prefix, comparing compressor and index-cache row counts against
    ordinary decode. Short output comparisons alone can miss stale frontiers.
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
- Run `make test-quality-api` and
  `python3 tests/test_glm_reference_render.py` after scorer or fixture changes.
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
  For the Q4 layout with Q8 KDA projections, embedding, and output head,
  the M3 Ultra reference is `0.300804038`, `90/100`, and `9.48`.
- For GLM 5.3 attention changes, also run the eight long Z.AI FP8 cases.
  First run `python3 gguf-tools/quality-testing/render_glm_references.py
  gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-long`, then
  score its `manifest-rendered.tsv` at context 8192 with `--rendered-prompt`.
  This preserves the official low-effort template and returned reasoning
  before scoring the answer. Every prompt crosses the 2,051-token boundary.
  Compare the default path against the scalar control on the same fixture.
  Z.AI supplies no output logprobs; do not report zero logprob-error columns
  as API parity. The 100-case reference numbers above use a no-thinking
  prefix and are not directly comparable to rendered-prefix scores.
- Run the GLM 5.2 fixture for reduced-precision GLM 5.2 release files. The Q2
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
- On Metal, benchmark predictable code and deliberately unpredictable prose at
  temperature 1. Compare ordinary, opportunistic and exact-sampled decoding;
  inspect output quality and acceptance as well as throughput. Low acceptance
  can make speculation slower than ordinary decode.

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
- Before physical Metal TP, check both hosts against the link and memory setup
  in [the distributed guide](docs/DISTRIBUTED.md). Reboots can reset
  member-interface IPv4 aliases and `iogpu.wired_limit_mb`. An active port is not
  enough, and a shard above the effective GPU residency limit may page heavily.
  Preserve OS headroom; do not bypass the memory guard to make a test fit.
  Match explicit `--prefill-chunk` overrides on both ranks. A different chunk
  schedule can fail during prefill synchronization rather than at the handshake.
  The batch oracle needs extra sessions for mixed prefill and serial controls.
  Use an admitted context for its short-prompt checks, then test long contexts
  separately rather than skipping the mixed checks.
- On `mac-m5max-it` and `mac-m5max-us`, run the same oracle in physical TP mode
  over explicit `tcp` and `rdma` transports. Set `DS4_TEST_TP_MODE=leader` on
  the leader and `DS4_TEST_TP_MODE=worker DS4_TEST_TP_LEADER_HOST=HOST` on the
  worker, with a unique `DS4_TEST_TP_PORT`. Run at least 2 and 4 sessions and
  preserve both logs. GLM 5.3 uses native row batching through token 2051 in TP
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
  2,051-token dense-attention limit, DSpark support models, quality/reference modes,
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
  The long prompt must cross the 2,048-token indexed-attention boundary. After
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
  its quantized or BF16 indexer projection instead.
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
- Run `make test-glm-attention` on Metal and CUDA, or
  `make test-glm-attention-rocm` on ROCm. Check FP16/FP32 padded selections,
  entirely empty selections, continued causal attention, and normalization
  against the independent numerical references. A bug in shared code or a
  copied kernel must be fixed and tested in every affected backend, including
  DeepSeek paths when affected.
- Treat session construction as the attention-memory admission point. Every
  owned DSA cache and indexer pool/tail, every KDA recurrent state, and the
  complete supported prefill workspace must allocate before a request is
  accepted. A first prefill must not grow a per-layer cache. Check both a
  4,096-token session and a long session in the memory report.
- Keep GLM-5.3 in absorbed MLA form. The model selects 2,048 tokens from
  complete four-token pools, plus the incomplete tail (at most three tokens).
  Dense attention is equivalent through 2,051 visible tokens; at 2,052 it
  must use the selector. This boundary must not depend on context allocation,
  SSD streaming, backend, or prefill chunk size. Padding IDs must be masked,
  including in batched prefill. Do not restore the 2.75 GiB expanded per-head
  K/V cache as a presumed quality fix.
- At 100K on an M5 Max, require the compact Q2 plan to remain near 94.09 GiB:
  89.87 GiB model, 1.11 GiB compact history, and 3.11 GiB fixed graph buffers.
  Repeat the continuation fixture after changing the
  compact cache type, absorbed projections, FlashAttention staging, or the
  crossover; numerical similarity to the old expanded graph is not the gate.
- On this 128 GB M3 Max, run the resident Q2 through the generic non-NAX Metal
  path. Repeat the 2,048-2,052 and 4,096-4,100 boundaries, official-continuation,
  MTP, snapshot, server-session, and continued-prefill gates used on M5. Record the different
  M3 performance floor rather than borrowing the M5 result. Run Q4 only with a
  bounded SSD-streaming cache; never try to make it fully resident.
- On one M5 Max, run resident Q2 with a prompt whose actionable instruction
  begins after token 4096. The model must recover the tail instruction and
  complete a tool or exact-output task. A short coherent continuation alone
  is not a sparse-selector correctness check.
- Run physical Q4 50/50 TP over the explicit TB5 RDMA devices. Graduate context
  allocation through 10K, 25K, and 50K, checking both ranks before advancing.
  Include mHC/KDA prefill workspace in the memory plan; the 50K reference is
  about 102.64 GiB per rank. Keep the plan within each host's admitted budget.
  Swap must not grow
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
  Check continuation quality, accepted/rejected drafts and measured speed;
  ordinary decode must remain available for low-acceptance prompts.
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
- Run the two- and four-session GLM 5.3 server oracle below token 2052 on one
  M5 and physical TP. For both native single-M5 and TP paths, set
  `DS4_TEST_LOGIT_TOLERANCE=0.001`: row-batched reductions may differ from the
  serial launch order, but every selected token must match and the maximum
  full-logit delta must remain below that bound. Also run the serial rollback
  with zero tolerance. Past
  2051, require the exact ordered fallback until a sparse native batch oracle
  proves full-vocabulary correctness.
- Exercise every pool remainder at the dense-to-sparse boundary with prompts
  ending at tokens 2048 through 2056, then repeat 4096 through 4100 for the
  prefill-work boundary. The sparse cases must retain the same greedy token,
  contain no nonfinite logits, and pass a multi-token exact-output task. Small
  batched-reduction logit differences are acceptable only when the official
  continuation and long-task gates remain in band. Build each prompt against
  `--dump-tokens`; word counts are not a valid substitute for rendered-token
  counts.
- Run the session snapshot test across the sparse boundary:
  `DS4_TEST_MODEL=/path/to/GLM-5.3-Flash-Q2.gguf
  DS4_TEST_SNAPSHOT_PROMPT=/path/to/a-4k-plus-prompt.txt
  DS4_TEST_SNAPSHOT_CTX=8192 ./ds4_test --session-snapshot`.
  Restored top logits before and after one continued token must match the
  uninterrupted session within the test's `1e-6` tolerance.
- Repeat that command with `DS4_TEST_GLM_MTP=1`. The test must replay 16
  integrated-MTP cycles across the snapshot, including both one- and two-token
  outcomes, and match every committed token plus the final top-eight logits.
  Then sync back to the original long prompt, reproduce its top-eight logits
  within `1e-6`, and complete further MTP cycles on the reused session.
- Measure continued prefill as actual appends to one live session, not only as
  a single cold prompt. On an M5 Max, run:
  `./ds4-bench -m /path/to/GLM-5.3-Flash-Q2.gguf --metal
  --prompt-file /path/to/a-25k-prompt.txt --ctx-start 4096 --ctx-max 12288
  --ctx-alloc 16384 --step-incr 2048 --gen-tokens 0 --csv /tmp/glm53.csv`.
  Repeat on CUDA and compare each append with a matched control. Record
  latency as well as throughput; test small tool-result appends too.
- Run the final Q4 TP coding task with at least 30K live prompt tokens over
  explicit RDMA. Require the tail instruction, read/edit/test task, independent
  code checks and prefix reuse to pass without premature compaction.
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
  vision sidecar, and a fixed PNG. Set `DS4_TEST_VISION_REPEATS=3` for the
  encoder test to require finite, repeatable embeddings. Run
  `python3 tests/test_compare_glm53_vision_embeddings.py` to check that the
  comparison rejects invalid reference embeddings as well as invalid candidates.
  The prompt test must generate a visual answer, reuse an unchanged image
  without repeated prefill, and rebuild when
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
- Cache and I/O optimizations must not change routed expert counts, weight
  precision or activation precision. With the same model and prefill chunks,
  require identical complete logits and deterministic output across cache
  policies. Test numerical-kernel changes separately against a reference.
- Run `MTL_DEBUG_LAYER=1 make test-metal-ssd-experts` on an idle Metal host.
  It exercises all eight routed slots through hits and evictions, requires byte-identical
  outputs with different cache capacities, and replaces more than 4,096
  layer mappings while retaining a separate auxiliary model mapping.
  For cached-batch changes, also run
  `MTL_DEBUG_LAYER=1 ./tests/test_metal_ssd_experts --full-glm-shape`.
  This uses full GLM's 6144/2048 expert dimensions, repeated batch sizes and
  evictions, and checks intermediate activations and output byte-for-byte.
  Include the 255/256/257-token dispatch boundary and an undersized cache.
  Repeat with `DS4_METAL_DISABLE_METAL4=1` to cover the conventional kernels.
- Run `MTL_DEBUG_LAYER=1 make test-glm-attention` and a real server coding
  session with validation enabled. Irregular context lengths must work, not
  just round benchmark sizes. Preserve validation failures and fix their cause;
  disabling validation is not a passing result.
- On Metal, compare the complete prefill logits with the previous executable
  at 2K/3K and 8K/12K frontiers, with generation and continued prefill between
  them. Exercise initial whole-layer reads, later selected-expert reads, cold
  caches and a mixed-size expert layer. Record initial and continued speed
  separately, including the first decode step rather than only steady speed.
  Save the exact commands and prompt files with each comparison. Equal context
  lengths do not imply equal input tokens. For short tool-result appends, test
  16, 43 and 114 tokens as well, followed by generation: a faster append must
  not merely move its I/O cost into the next decode steps.
- Deny the first static-weight `mlock` in a test build or interposer. Startup
  must leave those weights pageable and continue with correct output. Do not
  confuse this with denying every lock: the existing expert cache also needs
  locked buffers. Keep external memory monitoring enabled; do not deliberately
  trigger system OOM or a GPU watchdog reset.
- Run the section 12 coding-client and prefix-replay checks with a model
  larger than RAM and automatic SSD cache sizing, for both DeepSeek and GLM.
  A resident model pass does not exercise cache evictions after tool results.
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

  The first command must report 463 input tokens and keep median prefill at
  or above 11.3 t/s. It truncates the archive before its final question, so it
  is a timing input, not a sufficient quality test. Repeat with an explicit
  final question asking which component reports anomalies; require gamma. The
  second must emit all 64 requested output tokens and keep median generation at
  or above 5.5 t/s. The M5 Max reference medians are 12.59 and 6.14 t/s. Startup
  should plan about 39.63 GiB at this context, including 22.55 GiB of static
  model mapping, and initially restrict the model
  map to the token embedding. GLM must demand-fill its expert cache by default;
  an ordinary run and `--ssd-streaming-cold` should have comparable cache-miss
  counts and speed unless an explicit preload count or diagnostic cap is used.
  A memory guard or static decode map that accounts nearly the full 196.58 GiB
  GGUF, a Metal OOM, repeated garbage tokens, or a compact-attention result
  that omits the RoPE score is a release blocker.

  Also run `python3 tests/test_glm_rope_prefill.py --model "$GLM_SSD_MODEL"`
  under the external memory guard. This checks all next-token logits against
  the general attention path, then requires a correct answer to a complete
  question. Use the full, non-Flash checkpoint: Flash has no RoPE contribution
  and cannot catch this dispatch regression. `make test-glm-attention` covers
  nonzero RoPE against a double-precision reference with F16/F32 caches,
  initial and continued prefill, and incomplete head groups.

  Also test an 8K prompt followed by a 4K append at a 16K context with the
  automatic cache and an explicit cache target. This crosses compact-indexer
  warmup, which reads weights from earlier layers after prefill releases their
  mappings. Compare full logits and monitor memory throughout, not just startup.
  For `ds4-bench`, use `DS4_BENCH_FORCE_SNAPSHOT=1` when the extra snapshot fits;
  otherwise its large-payload fallback replays the prefix and changes the cache
  state before the append. Record which restoration method was used.

### SSD Performance References

Use these only with the same model, cache budget, prompt and restoration
method. Measure first-decode latency after each append as well as steady speed.
The full-GLM references use `GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf` on
M5 Max, a 16K context and a 35.36 GiB expert cache with snapshot restoration:

| Workload | Prefill | Generation |
| --- | ---: | ---: |
| Initial 8K, 16 output tokens | about 91 t/s | about 3.7 t/s |
| 4K append, 16 output tokens | about 84 t/s | about 4.0 t/s |

For short appends, use the code-audit prompt, 1024 initial tokens, an 8K
allocation, a 66 GB cache hint (61.35 GiB effective), forced snapshot restore
and 64 greedy output tokens at each frontier:

| Added tokens | Append latency | Generation afterward |
| ---: | ---: | ---: |
| 16 | 2.9 s, three-run median | 5.11 t/s |
| 43 | 5.4 s, single run | 5.34 t/s |
| 114 | 8.7 s, single run | 5.39 t/s |

Compare with `DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR=1` as
an ablation, requiring identical full logits and generated text. Keep cache
misses/application read bytes separate from measured physical SSD traffic.

A stuck `pread`, unopenable GGUF or filesystem wait surviving cancellation is
not a passing timeout test. Stop inference, check the filesystem independently
and repeat the task only after the host is healthy. Do not increase timeouts
or count a pass on another host as resolving that failure.

## 8. CUDA / DGX Spark

Before a release, ask the user for CUDA access if it is not already configured.
Use either DGX Spark / GB10 host, `antirez@dgxspark1` or
`antirez@dgxspark2`. Do not claim CUDA is release-ready without this pass.

Check for existing inference services first. Before stopping one, record its
process, service or container, model, ports, and exact launch command. Confirm
all workers have exited before loading DwarfStar. At the end, stop the test
processes, restore the original service, and verify its ports and model health.
Do not use high-performance Hugging Face Xet mode while a model is resident.

- Fetch or push the exact release commit to the CUDA machine.
- Build:
  `make clean && make cuda-spark`.
- Require both the DGX Spark build and the eight-GPU CUDA build to complete
  without compiler warnings. The eight-GPU build is performed only after
  receiving explicit permission to use `192.168.60.250` for this QA pass.
- Run:
  `make cuda-regression`.
- Build and run `tests/test_cuda_pool`, also under Compute Sanitizer.
  Exercise repeated initialization/cleanup, graph replay and pre-existing
  release thresholds above and below the runtime's scratch budget. Cleanup
  must restore the prior threshold and trim unused retained allocations.
- Run `tests/test_cuda_grouped_q8`, also under Compute Sanitizer. Grouped
  Q8 outputs must match scalar rows exactly for short verify blocks, ragged
  shapes, compact group slices and the 8/16/32-row boundaries; output tails
  must stay untouched.
  Compare whole-model prefill and decode, including V4.1, before accepting
  a primitive speedup. Repeat paired timings if a regression appears.
- Run `tests/test_cuda_tokentile`, also under Compute Sanitizer, after
  short-block attention changes. Compare against the double-precision CPU
  reference across raw-ring wrap, compression boundaries, head tails and
  peaky queries; cover both sides of the short-row dispatch boundary.
- Run `make test-cuda-indexed-one` after changing single-row indexed
  attention. Require exact scalar-path agreement across ring wrap, masked
  compressed rows, head tails and graph replays with changed queries. Check
  the double-precision reference and run memcheck, racecheck and synccheck.
  Compare full-model logits after fresh and continued prefill, and repeat
  physical Q2/MXFP4 speculative and server cancellation tests.
- Run `tests/test_cuda_q8_rows --slices` after changing compact shared-expert
  projections. Compare both rank halves, an asymmetric middle slice and the
  final 32-column block against the double-precision reference; check output
  tails and invalid ranges under Compute Sanitizer too.
- Run `tests/test_cuda_dspark`, also under Compute Sanitizer, after proposal
  changes. Cover ties, negative scores, vocabulary tails, invalid previous
  tokens and output canaries against the CPU Markov/argmax reference. Exercise
  every supported Markov rank and a full-size vocabulary with a ragged tail.
- Run `tests/test_cuda_compressor` after changing batched verification state.
  Pooled rows, rolling state and every saved prefix must match sequential
  updates exactly across compression boundaries. Run Compute Sanitizer too.
- Run `make test-cuda-copy-spans` on Spark after changing speculative cache
  copies. Snapshot, restore and every accepted-prefix slot must match ordered
  copies byte for byte, including graph replay and untouched destination
  guards. Check unaligned spans, multiple launches, invalid bounds and overlap
  fallback under Compute Sanitizer. Repeat single-host and physical TP
  speculative tests with Q2, and TP tests with MXFP4.
- Run `tests/test_cuda_attn_side`, including Compute Sanitizer memcheck and
  initcheck, racecheck and synccheck, after changing projection overlap. Check
  every intermediate, delayed side-stream input, paired projections for one
  through eight rows, graph capture/replay, and scratch failure. Repeat a
  single-host Q2 run to cover asynchronous model-copy ordering without TP.
- Run `make test-cuda-router-stream` after changing router dispatch or CUDA
  stream handling. Graph replay with new logits and delayed side-stream
  inputs must match eager execution for each router launch mode. Run all four
  Compute Sanitizer tools; confirm graph and side-stream checks did not skip.
- Run `make test-cuda-q8-0-gemv CUDA_ARCH=sm_121` after changing dense Q8
  verification kernels. Require bit-exact agreement with MMQ on the same
  quantized activations, including non-finite values, unsupported shapes,
  delayed side-stream inputs and graph replay with changed inputs. Run
  memcheck, initcheck, racecheck and synccheck. Compare full-model Q2 and
  MXFP4 outputs, acceptance decisions and timings on both TP ranks.
- After aligned Q8 scratch changes, run `make test-cuda-q8-scratch
  CUDA_ARCH=sm_121`, also under Compute Sanitizer. Dense and paired outputs
  must be exact with reused scratch, an undersized buffer, and captured graph
  replays after input changes. Include full-model prefill and decode logits;
  a short token comparison cannot detect stale scratch contents.
- After CUDA DSpark changes, run `make test-cuda-dspark-moe CUDA_ARCH=sm_121`
  and `compute-sanitizer --error-exitcode 1 tests/test_cuda_dspark_moe --check-only`.
  Cover one through eight rows, distinct/shared/partly overlapping experts,
  invalid negative selections, and Q2 down assignments spanning several launch
  tiles. Deduplicated gate/up rows must match separate one-token calls exactly.
  Run the Q8 scratch test too: every small batch must equal its separate rows.
  Repeat the section 4 greedy, opportunistic, exact-sampling and forced-partial
  fixtures with matching 0731 weights. CUDA retains all five intermediate
  prefixes of a six-token seed/draft block; ordinary partial accepts must not
  need replay. Keep the scheduler's seed count separate from successful drafts.
  Exercise a live speculative cache before and after continued prefill at
  2K, 4K and 8K frontiers. Also compare the official continuation scorer with
  `DS4_METAL_PREFILL_CHUNK=6` in both control and candidate builds: this shared
  diagnostic override exercises the small CUDA batches, unlike ordinary
  teacher-forced single-token scoring alone.
- For native MXFP4 changes, run
  `make test-mxfp4-cuda CUDA_ARCH=native` on the multi-GPU CUDA host only after
  receiving explicit permission for `192.168.60.250`, and
  `make test-mxfp4-cuda CUDA_ARCH=sm_121` on DGX Spark. Dense MMQ, routed MMQ,
  routed MMVQ, fused gate/up, and fused down must pass, including odd rows,
  K=256/768 and large batches. Require the same Q8-activation accuracy bounds
  on Blackwell as on older devices; do not relax them for native FP4 hardware.
  Run `make test-mmvq-cuda CUDA_ARCH=sm_121` under Compute Sanitizer to check
  tail loads, routed output bounds, masked experts and output canaries.
  On Spark, also run `tests/test_mxfp4_staged_cuda`: compare staged gate/up
  outputs exactly, including misaligned rows and large-width fallbacks.
  Run memcheck and racecheck when changing its shared-memory staging.
  For shared MMVQ changes, also run `make test-mmvq-rocm` on Strix Halo.
  These primitive tests do not replace full-model continuation scoring.
- With that permission, run the native MXFP4 GGUF resident on the multi-GPU
  host, and run it with `--ssd-streaming` on DGX Spark. Use the same greedy prompt and continuation
  fixture on both. Record prefill and generation speed, require finite logits,
  and compare quality with the Metal MXFP4 result. Both MMQ and MMVQ must keep
  eight-bit activations. Compare ordinary scalar decoding with the actual
  multi-row speculative verifier against checkpoint-matched official
  continuations, reporting teacher-forced NLL and API top-token agreement.
  A small prefill chunk alone does not exercise the verifier's cache commits.
- Run a short CLI prompt with the Flash GGUF and record generation t/s.
- Run a longer prompt that exercises routed experts past a few thousand tokens.
- Run `tests/test_deepseek_indexer FLASH_Q2_GGUF` on one otherwise idle GPU.
  It checks that decode selects the model's 512 highest-scoring compressed
  rows across the attention boundaries, including continued prefill.
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
- Run `make test-linux-memory test-rocm-memory` on an otherwise idle Strix.
  Admission checks must exclude `CmaFree` from Linux `MemAvailable`, even when
  `CmaTotal` reports zero, and refuse oversized pinned allocations before
  entering the driver. Keep an independent process-group memory watchdog for
  model tests: monitor `max(0, MemAvailable - CmaFree)`, not `MemAvailable`
  alone, and stop below 3 GiB usable. Never run primitive benchmarks or builds
  beside a resident model, even if the model process is paused. Keep logs
  outside `/tmp` so a reboot does not erase the failure evidence.
- After resident cache changes, exercise plain Flash and checkpoint-matched
  DSpark at the default context and prefill capacity. Required weights and
  session buffers must fit before optional Q8-to-FP16 expansion. Record the
  lowest usable RAM and inspect the kernel journal for OOM/GPU errors after
  each run; a monitor-terminated run is not a pass.
- After GLM attention changes, run `make test-glm53-kda-rocm
  test-glm-attention-rocm`. Repeat the attention test with
  `DS4_ROCM_GLM_SELECTED_ATTN_HEAD_TILE=1` to cover the single-head fallback.
  Require correct handling of padded, out-of-range, and entirely empty
  selections in both FP32 and FP16 caches, causal masking on continued
  prefill, and reference agreement for indexer normalization and split
  attention. `tests/test_glm_attention_rocm --bench` measures the attention
  primitive; it does not replace whole-model timing. Unload all models
  before this benchmark; do not run it beside a paused resident scorer.
- Compare before/after official continuation scores using the same GGUF and
  the matching fixture directory from section 3. For GLM 5.3 Flash use
  `gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv`.
  Record NLL, first-token matches, and matching-prefix length. These GLM
  prompts are short: they do not exercise the sparse-attention boundary.
  Also test initial and continued prefill across 2051 and 4096 tokens, including
  non-multiple-of-four frontiers, and compare with the scalar attention
  control. Do not claim a long-context quality improvement from unchanged
  short-prompt scores. Use the long Z.AI FP8 fixture and its rendered-prefix
  procedure in section 3 for this comparison.
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
  zero verifier errors, and no unexplained replay fallbacks. With
  `DS4_DSPARK_SPEC_LOG=1`, a five-of-six commit may report `prefix-extended`:
  ROCm retains four prefix snapshots and replays just the fifth token.
  This bounded fallback is expected; other replays need investigation.
  Record ordinary and DSpark
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
  `MemAvailable - CmaFree` as well as the backend limit. The one-expert target must use
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
- After tool-parser changes, run `make test-frontends` and
  `make test-session-state`. These targets do not load model weights. Repeat
  the frontend suites with ASan/UBSan for parser and buffer changes.
- Test both DeepSeek DSML and GLM tool syntax. Split closing delimiters at
  every byte boundary; sampled argument text must not become greedy syntax
  merely because a token ends in `<`. Literal thinking/tool markers inside
  arguments must not end the call. Check ordinary HTML entities, literal
  closing-wrapper escapes, and repeated escape spellings through rendering,
  parsing, streaming and canonical history replay.
- Truncate a tool argument using a small output budget and an explicit client
  stop, with and without streaming. Preserve `length` versus `stop`, never
  invent a complete action from an unfinished argument, and do not retry after
  a client stop. Model-error recovery must share the original output budget;
  usage includes the failed attempt. Partial SSE arguments may remain partial.
- Run a real Pi or other coding-client read/write/edit/build/test loop with
  GLM MTP and matching DeepSeek DSpark, in default and exact-sampling modes.
  Discard unseen speculative suffixes after tool/stop markers. Exact sampling
  must resample when the parser changes sampling mode; default opportunistic
  decoding should retain already-verified greedy drafts across that change.
  Check tool-result continuation through all three API formats. Responses
  requires full input replay, not an unsupported `previous_response_id`.
- Exercise actual `ds4-server` coding sessions, not only direct ds4-agent or
  isolated HTTP requests. Use Pi, OpenCode or another supported coding client
  for several read/edit/build/test rounds, including a harmless tool failure
  and recovery. Grow the conversation past 4K tokens and continue it.
  Capture `--trace` and check tool IDs, raw tool replay, rendered prompts,
  cache source, matched-prefix length and continued-prefill size after each
  tool result. An unchanged history should retain its reusable prefix;
  investigate repeated full prefills or unexplained canonicalization rebuilds.
  Conversely, editing an earlier message must invalidate the changed suffix.
  Compare a continued request with a fresh replay of the same full history,
  checking tool arguments, coherent output and context isolation. Record any
  necessary rebuild and its latency rather than counting a working tool alone
  as proof that prefix matching works.
- With vision enabled, run `python3 tests/test_server_vision_cache.py --url
  http://127.0.0.1:8000 --output /tmp/vision-cache-qa` against an otherwise idle
  server. Repeat for GLM and DeepSeek, with ordinary and batched sessions.
  Appending an image must retain the matching text/image prefix; replaying old
  images must reuse their encoder output. Changed, removed or reordered old
  images must not reuse incompatible KV. Include visible-history replay that
  omits hidden reasoning, concurrent requests, and a roughly 50K-token prefix
  (`--archive-lines 4200 --append-only`, with a sufficiently large `--ctx`).
- Run `python3 tests/test_server_vision_agent.py --url http://127.0.0.1:8000
  --pi /path/to/pi --output /tmp/vision-agent-qa` for both vision models. Pi must
  actually read the two images, edit the program and pass the independent
  output checks through Chat Completions, Responses and Anthropic. Image tool
  results must arrive as images, not disappear or become literal placeholders.
  Check cached-token accounting after every image-bearing tool continuation.
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

For an unterminated tool call, capture raw model output, source revision,
output budget and stop reason before blaming inference, parsing or memory.
Literal-marker copying needs both deterministic parser tests and real-model
checks: the model can change the text before the parser sees it.

## 13. ds4-agent

The agent is the most stateful component.  Test it manually, not only by build.

- Startup banner, status bar, help, `/power`, `/save`, `/list`, `/switch`,
  `/history`, `/compact`, `/new`, `/del`, and `/strip`.
- Ctrl+C during generation, during prefill, during a web fetch, and during a
  long tool call.  After `Stopped by user`, typing a new prompt must work.
- Queue messages while the model is busy.  Queued messages must not skip tool
  execution; after tool results, the queued user text must be provided.
- Force context pressure with `python3 tests/test_agent_compaction.py --binary
  ./ds4-agent --model MODEL.gguf --vision VISION.gguf --ctx 4096
  --output /tmp/agent-compaction-qa` (requires `pyte` and a C compiler; use 8192
  for DeepSeek). Check nearly full input, mid-response compaction, the compiled
  code oracle, output-budget accounting, oversized-input rejection and a real
  tool task afterward. An unfinished assistant response must resume without
  dropping partial words or code lines. An unfinished tool call must never run.
  Repeat with `--think --tokens 3000`, GLM `--mtp`, and DeepSeek
  `--dspark MATCHING_SUPPORT.gguf`. Summaries must record unfinished work,
  not invent its solution; validate the resulting code independently.
- Restore an already full session saved by the previous release and continue a
  coding task. Summary generation must have reserved space or summarize a
  bounded prefix while retaining every unsummarized token. Cancel compaction
  and continue again; do not lose the original conversation on failure. Repeat
  manual compaction, queued input, save/restore and thinking-enabled generation.
  Check task completion and constraints, not just the absence of an error.
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
  valid, and DSpark stats to show zero verifier errors and no unexplained
  replay fallbacks (see the bounded-prefix exception in section 9).
  Run the opportunistic default and `--mtp-exact-sampling`.
- After message-format changes, run `make test-session-state`. Text-only
  observations through the multimodal API must equal ordinary message tokens
  for Flash, PRO and GLM, including closing-wrapper escaping. They must not
  require a vision model.
  Rendering/image-observation errors must be reported directly without
  compacting the conversation as though it had run out of context.
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

Check speculative stop-boundary rewinds separately from compaction. Record
any required prefix rebuild and its latency; a successful compaction test
does not establish that speculative cache reuse is efficient.

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
- For speculative decoding, retain the generated text and draft-acceptance
  statistics. Repetition loops can inflate throughput; a faster bad answer is
  not a win. Separate changed continuation/acceptance from kernel cost using
  matched teacher-forced ordinary-decode measurements. Test code and prose.
- Compare startup time and peak memory as well as tokens per second when model
  loading, caches, streaming, or temporary arenas changed.
- Run the backend-specific batch tests in sections 4 and 8. Fast single-session
  decode does not substitute for aggregate multi-session throughput.

### CUDA And ROCm References

Use the corrected GLM 5.3 sparse-attention boundary at 2051 tokens. Do not
compare larger prompts against old dense-to-4096 measurements. On Strix Halo,
GLM 5.3 Flash Q2 with Promessi Sposi, 8192 allocated context and 32 generated
tokens has these reference points: initial 4096 prefill about 77 t/s, a 2048
append about 68 t/s, decode at 6144 about 11.7 t/s. Repeat as three-run medians.
The rendered long Z.AI fixture references are NLL 0.661721610 on ROCm and
0.680093110 on CUDA, both 6/8 first-token matches. Use section 3's template.

For Flash 0731 Q2 on ROCm, record cold and warm first-request costs separately.
Reference warm 2K prefill/append is about 207 t/s and 4K decode about 14.7 t/s;
required-buffer admission precedes optional weight expansion. Include that
expansion in startup-to-first-response timing rather than hiding it in warmup.
The matching 100-case reference is NLL 0.398181736, 56/100 first-token matches.

Single-Spark resident Flash 0731 Q2, Promessi Sposi, 9216 allocated context,
128 teacher-forced decode tokens, three-run medians:

| Context | Prefill | Decode |
| ---: | ---: | ---: |
| 2048 | 823.49 t/s | 19.25 t/s |
| 4096 | 899.89 t/s | 16.22 t/s |
| 8192 | 931.17 t/s | 15.97 t/s |

Single-Spark DSpark references use matched 0731 Q2 weights, 4096 allocated
context, prefill chunk 512 and 256 output tokens. Use the C hash-table and
unpredictable-prose prompts and sampling settings below:

| Workload | Ordinary decode | DSpark |
| --- | ---: | ---: |
| C hash table, temperature 0 | 19.72 t/s | 31.41 t/s |
| C hash table, temperature 1 | 19.53 t/s | 29.98 t/s |
| Unpredictable prose, temperature 1 | 19.53 t/s | 18.81 t/s |

Temperature 1 here uses opportunistic sampling, not exact sampling. Repeat
live-frontier sweeps at 2K/4K/8K as well; high short-code acceptance does not
establish a general speedup. Use the six-token scorer and continued-state
oracles in section 8 when changing small-batch CUDA kernels.

### Metal Kernel And Speculation Gates

For M5 dense-kernel changes, run `MTL_DEBUG_LAYER=1 make test-metal-dense-mpp`.
It checks Q8 decode and Q4_0/Q4_K prefill against exactly representable CPU
dots, including repeated calls, partial token tiles and untouched output tails.
Keep host threadgroup allocations in sync with kernel staging: the dense
double-buffered TensorOps kernel needs 8 KiB, not 4 KiB.

For M5 routed-prefill changes, run `make test-metal-moe-prefill`. It compares
gate/up, FP16 intermediate, expert partials and final outputs against the
unpacked path, including empty experts, tile tails and scratch reuse.
Then build `make metal-prefill-variant-bench metal-decode-schedule-bench` and
use a resident Flash Q2 or mixed Q2/Q4 model for interleaved full-logit checks:

```sh
speed-bench/metal_prefill_variant_bench -m "$MODEL" \
  --prompt-file speed-bench/promessi_sposi.txt --prefix-tokens 2048 \
  --warmup-tokens 2048 --candidate-env DS4_METAL_DISABLE_ROUTED_MPP_PACKED
speed-bench/metal_prefill_variant_bench -m "$MODEL" \
  --prompt-file speed-bench/promessi_sposi.txt --prefix-tokens 16384 \
  --initial-tokens 12288 --warmup-tokens 2048 \
  --candidate-env DS4_METAL_DISABLE_ROUTED_MPP_PACKED
speed-bench/metal_decode_schedule_bench -m "$MODEL" \
  --prompt-file speed-bench/promessi_sposi.txt --prefix-tokens 8192 \
  --ctx 9216 --tokens 256 --candidate-env DS4_METAL_DISABLE_ROUTED_MPP_PACKED
```

Here `control` is the default fast path; `candidate` disables packing.
Require exact logits and no decode regression. Packing is an M5
resident-prefill optimization; test streaming and conventional kernels
separately when their dispatch or shared buffers change.

For Metal DSpark changes, test matching 0731 and Vision Exp target/drafter
pairs. Run the acceptance fixture at temperature 0, at temperature 1 with
forced partial acceptance, and with `--mtp-exact-sampling`. Require zero
verifier errors and consistent seed/draft counters. A batched seed is a
normally chosen target token, not a successful draft prediction.

Resident M5 Q2 uses seed batching for longer proposals; short proposals first
decode the seed normally. Poor acceptance windows pause drafting for 32
cycles before trying again. `DS4_DSPARK_SEED_BATCH=0` retains the previous
schedule for diagnostic comparisons. This also applies to two-M5 TP with Q2,
mixed Q2/Q4 and MXFP4 experts. Streaming, exact sampling and other device/weight
combinations retain their existing defaults.

Compare code and prose, not just a high-acceptance copy prompt. Use at least
256 generated tokens at temperatures 0 and 1; also sweep 2K/4K/8K/16K live
frontiers with continued prefill. Record ordinary decode, old DSpark and
default DSpark separately. Seed batching must not make low-acceptance cases
worse than the previous DSpark path. It does not promise to beat ordinary
decode on every prompt. Run the six-session full-logit batch oracle too,
since a six-row verifier shares routed kernels with session batching.

M5 DSpark reference medians, 256 generated tokens, 4096 allocated context:

| Model and prompt | DSpark |
| --- | ---: |
| 0731 Q2, C hash table, temperature 0 | 62.95 t/s |
| 0731 Q2, C hash table, temperature 1 | 61.58 t/s |
| Vision Exp mixed Q2/Q4, C hash table, temperature 0 | 48.86 t/s |
| Vision Exp mixed Q2/Q4, C hash table, temperature 1 | 47.24 t/s |

Use the matching drafter and `--nothink --top-p 0.95 --min-p 0.05 --seed 12345`.
The C prompt is: "Write a complete C hash table implementation with string keys,
insert, find, delete, and a test main. Output only C code."

For two-M5 TP, test the MXFP4 and mixed Q2/Q4 Vision Exp models with their
matching drafter. Use the same sampling settings and 256-token C prompt above,
plus the prompt "Write an unpredictable surreal scene with constantly changing
imagery and no repeated phrases." Reference means of two reversed-order runs,
explicit RDMA and 50/50 residency:

| Model and prompt | TP DSpark |
| --- | ---: |
| MXFP4, C hash table, temperature 0 | 50.76 t/s |
| MXFP4, C hash table, temperature 1 | 48.15 t/s |
| MXFP4, prose, temperature 1 | 45.99 t/s |
| Mixed Q2/Q4, C hash table, temperature 0 | 49.52 t/s |
| Mixed Q2/Q4, C hash table, temperature 1 | 46.36 t/s |
| Mixed Q2/Q4, prose, temperature 1 | 44.16 t/s |

Ordinary MXFP4 TP reference on the C prompt is 53.19/50.93 t/s at temperature
0/1. Speculation need not beat ordinary decode on every prompt.

Also compare ordinary MXFP4 TP with and without
`DS4_METAL_DISABLE_ROUTED_MPP_PACKED=1`, using Promessi Sposi at 2K/4K/8K/16K
live frontiers. Repeat in reversed order and compare full frontier logits.
Investigate fresh-process variability with GPU/CPU clocks, temperature and
concurrent activity recorded. Require repeatable controls before attributing
small differences to a kernel. Include the matching official-continuation
fixture, not just a timing prompt.

Require `MTL_DEBUG_LAYER=1 make test-metal-moe-prefill` on both M5s. This covers
packed MXFP4 prefill, both expert-ownership halves, tiny batches and the actual
4096/2048/4096 static shapes. Static two-to-six-row outputs must be exact against
per-row decode. The synthetic static-shape model uses about 3.2 GiB of memory.
`DS4_METAL_DISABLE_TP_BATCH_MOE=1` restores per-row verifier experts for diagnosis;
`DS4_METAL_DISABLE_M5_TP_MXFP4_STATIC=1` disables the static specialization.
Neither is needed to enable the fast path.

Build `make tests/test_metal_tp_spec`. With the worker connected as usual,
run the coordinator with:

```sh
DS4_DSPARK_SCHEDULER=0 DS4_DSPARK_SPEC_LOG=1 \
  ./tests/test_metal_tp_spec "$MODEL" "$DSPARK" 10.99.0.2 9991 rdma_en1
```

This checks committed tokens against serial target logits at 127- and
4095-token prefixes, then appends to each live speculative cache. Require
six-token commits and exercise partial prefixes, including five of six tokens.
The test uses the verifier oracle's 2.0-logit near-argmax bound. Also run the acceptance checks with
`--mtp-exact-sampling` and with forced low-confidence proposals. Both ranks must
exit cleanly, with no verifier errors or unexplained replay fallbacks.
Exact sampling under TP still intentionally replays partially rejected blocks;
those replays are expected, and seed batching must remain disabled in this mode.
Record transport failures even if retries pass; investigate them rather than
hiding them by increasing timeouts.

Repeat the physical TP session oracle with two, four and six sessions. Do not
set `DS4_TEST_SKIP_MIXED`: ordinary decode and the mixed continued-prefill step
must both match full serial logits exactly. Check that the final TP residual
update is flushed before the output head; argmax agreement alone cannot catch
a stale residual. Test mixed Q2/Q4 on both one host and physical TP.

### Other Hardware References

These are comparison points for the stated model and workload, not portable
throughput promises. Establish a fresh matched baseline where the checkpoint,
prompt or settings are unavailable. Model-specific sections give additional
context sweeps and memory limits.

| System and backend | Model and workload | Prefill | Decode |
| --- | --- | ---: | ---: |
| M3 Ultra 512 GB, Metal | GLM 5.3 Flash Q4 with Q8 KDA/head, 2048-token prompt | 437.62 t/s | 24.74 t/s |
| Two M5 Max, Metal RDMA TP | GLM 5.2 IQ2_XXS, 4096-token prefill, 256 teacher-forced decode tokens | about 214 t/s | about 16.7 t/s |
| M5 Max, Metal | GLM 5.3 full Q2 SSD, 16 GiB expert budget, section 7 commands | 12.59 t/s median | 6.14 t/s median |
| DGX Spark, CUDA | GLM 5.3 Flash Q2, 2048-token prefill, 16 decode tokens | 531.39 t/s | 14.35 t/s |
| Strix Halo, ROCm | Flash 0731 IQ2, temperature-1 128-token code prompt | - | 16.26 ordinary; 12.28 opportunistic; 13.52 exact t/s |
| 8x L40S, CUDA TP | Flash Q4, 2048-token prefill benchmark | 1524.84 t/s | 46.93 t/s |
| 8x L40S, CUDA TP | Flash Q4, 16-row decode oracle | - | 126.0 aggregate t/s |

The 8x L40S values are retained from the last recorded run on `192.168.60.250`.
They are historical references only: never connect to that host or interrupt
its production server without explicit permission for the current QA pass. If
permission is granted, the existing hard floor remains 110 aggregate t/s for
the 16-row decode oracle.

## 17. DeepSeek V4.1 Flash

V4.1 is a different architecture and checkpoint, not a replacement filename for
V4 Flash. Use the matching vectors in
`gguf-tools/quality-testing/deepseek-v4.1-flash-20260910/manifest.tsv`.
The dataset README explains the temperature-one API logprobs: compare
teacher-forced probabilities and API top-token agreement, not sampled-prefix
length alone. Never refresh old Flash vectors through an API alias that now
serves V4.1.

Also score the 100 short general prompts in
`gguf-tools/quality-testing/deepseek-v4.1-flash-20260910-general/manifest.tsv`
when comparing quantizations. Keep this broader probability check separate from
the long sparse-boundary tests; neither substitutes for the other.

### Metal

- Audit the completed GGUF against its pinned source with
  `gguf-tools/deepseek41_validate_gguf.py --payload`, including native Engram rows.
  Repeat for calibrated weights using the actual imatrix. Record expert coverage;
  synthetic fixture statistics must never become a release calibration file.
  For Q4, pass `--quant q4` to both converter and validator. Convert from the
  original safetensors with the retained imatrix, not from the Q2 GGUF. A valid
  tensor inventory does not replace real-model quality tests for that recipe.
- Run `make test-engram test-deepseek41-gguf test-quality-api test-frontends`,
  the V4.1 manifest tests and the official RoPE/quantization primitive checks.
  Run `make test-download-model` to check verified downloads, damaged files,
  partial transfers and the default model link. Check the published artifacts
  using `ds41f-q2`, `ds41f-q4` and `ds41f-vision` on a runtime host. Q4 comes
  in two transport parts: check interrupted assembly, insufficient disk space,
  corrupt parts and the final complete-file checksum before opening the model.
- With real weights, run `tests/test_deepseek41_graph MODEL --session-fixture`
  and `tests/test_deepseek41_graph MODEL --partitions` under Metal API validation.
  Check restored continuation logits, compression carry, sliding-window wrap,
  cancellation and malformed snapshots. Zero-weight fixtures are not quality QA.
  On a sufficiently large resident Metal host, run
  `tests/test_deepseek41_prefill --resident MODEL speed-bench/promessi_sposi.txt`.
  This grows two sessions beyond 130K tokens while mixing scalar, layer-major
  and deferred prefills; it checks progress, complete state and restored decode.
  Omit `--resident` for the bounded SSD variant through 113K. Never run the
  resident variant on a 128 GiB Mac.
  Run `tests/test_deepseek41_graph MODEL --long-sessions PROMPT_FILE` with the
  dataset's `prompts/case_016.txt` for incremental prefill and exact restored
  continuation checks through 16385 tokens. This uses two sessions and a shared
  64 GiB expert-cache budget; monitor memory and do not overlap another inference job.
- After prefill scheduling changes, run
  `tests/test_deepseek41_graph MODEL --prefill-parity speed-bench/promessi_sposi.txt`
  under Metal API validation. This compares complete logits and all saved cache
  spans with token-major execution, including large continued chunks, snapshot
  restoration and cancellation. An in-flight partial layer stack must never
  be accepted as a valid snapshot.
  Run `make test-metal-ssd-experts test-metal-moe-prefill` too: seeded cache
  replacement must release physical buffers, not just reset entry counters.
  The SSD test covers IQ2 eight-expert and Q4/MXFP4 six-expert routes, cold/hot
  eviction, batched outputs and repeated physical-cache release. Its
  `--table-admission` mode must reject persistent full-expert tables during
  streaming, even with diagnostic table overrides. A resident one-row batch
  must not silently populate the SSD cache. For new large Q4 artifacts, start
  with a small cache and one continuation before the full scorer. Monitor host
  wired memory and swap independently of the engine's planned allocation;
  retained Metal residency sets can otherwise escape its accounting.
  `tests/test_metal_moe_prefill --ssd-address` checks the IQ2/Q2 address kernel
  across small and large row counts, checking index widths and output bounds.
  Compare short/long, initial/continued prefill with the same cache budget on
  the same host. Include first-token latency and steady decoding after prefill;
  a faster prefill that leaves a cold expert cache can lose overall.
- For the long-prompt resident-encoder schedule, run
  `tests/test_deepseek41_graph MODEL --encoder-parity speed-bench/promessi_sposi.txt`
  and `--encoder-long-parity` with the same prompt file. These need an automatic
  expert cache large enough to hold the encoder; use one inference process per
  128 GiB host. Compare complete logits and every cache span at 16,385 tokens,
  then append and decode, including snapshot restoration. `--encoder-cancel`
  separately checks interruption after pinning the full encoder and rebuilding
  the session; the short test also cancels a partially loaded encoder.
  Confirm the cache budget is restored and no encoder pages remain pinned.
  Memory must replace the expert cache, not sit beside it. A smaller budget or
  failed residency request must retain the two-layer path, with Engram on disk.
  Measure both schedules at 8k and 16k, including loading, first decode latency
  and at least 512 generated tokens. Use
  `DS4_METAL_DISABLE_V41_ENCODER_RESIDENCY=1` for the two-layer control.
- Score both single-host SSD streaming and two-host TP over real RDMA. Cover
  127/128/129, 511/512/513, 1023/1024/1025 and 16383/16384/16385 prompt tokens,
  then continued prefills and a real CLI/agent/server coding task with tool calls.
  Compare bootstrap and calibrated weights on the same held-out cases.
  Also use the `20260911-long` and `20260911-extended` V4.1 manifests for
  8/16/32K and 64/96K prefixes. Compare whole-prompt paired probability scores;
  sampled API output is not a full-logit oracle. Investigate persistent TP
  differences instead of assuming that a different reduction order explains them.
- Exercise native resident and TP batching with unrelated sessions. Run
  `tests/test_metal_session_batch` with `DS4_TEST_BATCH_ISOLATION=1`,
  `DS4_TEST_SESSION_COUNT=2`, `4` and `8`, and `DS4_TEST_DECODE_STEPS=32`.
  Supply `DS4_TEST_MODEL` and an admitted `DS4_TEST_CONTEXT_SIZE`; repeat two
  sessions at 64K using `DS4_TEST_PROMPT_FILE`. This oracle changes companion
  prompts and row order, checks complete target logits exactly, tests invalid
  batches without advancing state, and resumes after a mixed prefill/decode call.
  Isolation does not establish equivalence to serial inference: separately run
  `score_official --session-batch N` on the short and long manifests. Test
  physical Metal TP native batches with at least three rows: two rows use
  ordered execution by design. Confirm dispatch in the log instead of assuming
  that `--session-batch` selects the native path.
  Measure aggregate throughput without Metal validation, including two-session
  cases.
  Require the `native_ds41=1` trace for admitted native shapes, including mixed
  batches with nonzero `prefill_rows`; a passing ordered fallback is not evidence
  for the native path. Image-bearing sessions currently use that fallback.
  Real concurrent server requests must also cover cancellation and prefix reuse;
  a serial prefill followed by batched decode is not a native mixed-layer path.
  Run `tests/test_metal_moe_prefill --v41-decode` for rows 1-9, including both
  TP ownership partitions. Its `--v41-q4-decode` mode checks the Q4 shape at
  rows 1-8 against independent one-row calls, both ownership partitions and
  untouched output tails. Run both: IQ2 success does not cover Q4 dispatch.
  Run `tests/test_deepseek41_graph --batch-admission`
  and `tests/test_deepseek41_graph MODEL --batch-head`; the latter compares
  every output logit with the scalar head and selected rows with double math.
  Use `DS4_TEST_MIXED_SHAPES=1` with `tests/test_metal_session_batch` for 1-7
  appended tokens sharing a batch with independent decoders. Repeat at 73,728
  context on a large resident host, not the 128 GiB RDMA pair: this fixture
  allocates ten sessions. Check causal state and subsequent decode at 64K.
  Keep HTTP identical-output comparisons at a fixed prefill chunk geometry;
  the default idle and concurrent schedules can round differently. Also run
  the default schedule for independent coding correctness and prefix reuse.
- Run `tests/test_metal_tp_cancel` over the physical RDMA pair. Interrupt at
  small and large prefill boundaries, reject a partial snapshot, then rebuild
  and decode on the same connection. Both peers must exit normally. Repeat
  shared transport changes with older Flash and GLM, not just V4.1.
  Run `make test-session-state` and delayed physical `tests/test_metal_tp_bulk`
  exchanges for decode, verify and bulk shapes. Both receive windows must be
  posted before either peer sends. Keep matching protocol versions on both
  hosts; a longer timeout is not a fix for a missing receive window.
- Run `tests/test_tp_tcp` on macOS as well as Linux after transport changes.
  Exercise both real loopback TCP and socket pairs with tiny buffers, including
  stalled peers and half-closes. Darwin `sendmsg(MSG_DONTWAIT)` alone can still
  block; the exchange must restore descriptor flags on success and failure.
- For vision, audit the separate encoder GGUF against the pinned source and run
  `tests/test_deepseek41_graph --vision-routing`. Compare
  `--vision-encoder VISION_GGUF IMAGE OUTPUT.f32` with independently evaluated
  official BF16 encoder outputs for OCR, layout, diagram and screenshot images.
  Build `tests/test_glm53_vision_prompt` (the shared vision-prompt oracle) and
  run its `--logit-oracle` check for non-finite values. Test
  `tests/test_glm53_vision_prompt MODEL VISION_GGUF IMAGE` in resident,
  SSD and physical TP modes, including SSD `--quality`. Check image replay,
  changed embeddings and restored logits, not only a plausible description.
  Run the server vision-cache suite with tool-result images, changed/reordered
  images, multiple sessions and thinking replay. An unchanged image must retain
  its prefix; a changed old image must invalidate the affected cache. Exercise
  native agent image tools too. Keep the older Flash encoder regression gates.
  Run `python3 tests/test_agent_vision.py --binary ./ds4-agent --model MODEL
  --vision VISION_GGUF --ssd-streaming --output /tmp/agent-vision-qa`, then
  repeat with `--ctx 73728 --archive-words 50000` and a fresh output directory.
  It must inspect two images separately, edit code from their contents, pass
  an independent output check and retain the prefix at every tool continuation.
  Omit `--ssd-streaming` only on a host admitted for full residency. Allow
  headroom above the 50K input for image tokens, tools and system reminders;
  a task compacted into a short context does not pass the long-context gate.
- Verify that Engram tables remain unmapped and unpinned in every mode, including
  weight warming and TP. On 128 GiB Macs, never try full main-model residency;
  use SSD streaming or one half of a two-Mac TP setup. Increase context gradually
  and monitor memory/swap. Larger-Mac residency requires separate physical QA.
  With both released Q2 and Q4 GGUFs, run `tests/test_deepseek41_graph MODEL --memory-plan`.
  It rejects Q4 residency on 128/256 GiB hosts and a Q4 shard on a 128 GiB rank,
  while admitting larger hosts and fitting the SSD cache to its remaining budget.
  The Q2 variant checks simulated 128/256/512 GiB admission and TP shard sizing.
  Neither variant allocates weights or context buffers; this is not physical-host QA.
  Repeat startup with `--warm-weights`: TP must warm only its owned shard,
  SSD streaming must skip the full-file pass, and inspection must not warm weights.
- Exercise `--think-level 0`, `1`, `25`, `100`, `/think 25`, `/think`, `--think`
  and `--think-max`; reject malformed/out-of-range values. Check changing effort
  in a live conversation invalidates the old prefix without losing its messages.
- Run `tests/test_agent_compaction.py` first at 4K, then at least 64K with
  the appropriate SSD or resident configuration. Check real read/edit/test
  tasks, automatic and mid-generation compaction, interruption, text session
  save/restore, cached-prefix reuse, generation budgets and recovery after an
  oversized message. Compile and independently test the generated code.
  The small-context run must exercise mid-generation compaction. The long run
  must retain at least 75% of its context for the coding turn, not compact
  before it; allow room for the system reminder inserted after 50K tokens.
  Repeat with `--interrupt-only` and a fresh output directory to interrupt
  generation and verify a real tool call in the same conversation afterward.
  For a served V4.1 model, run `tests/test_server_vision_agent.py` with
  `--model deepseek-v4.1-flash` against Chat, Responses and Anthropic. Require
  independently checked image-driven code edits and cached-prefix reuse,
  not just successful HTTP responses.
- Unsupported backends, pipeline and speculative modes must fail explicitly until
  their V4.1 implementation is validated; an older graph/drafter is not a fallback.

### DSpark

Use `DeepSeek-V4.1-Flash-DSpark.gguf`, not a V4.0 drafter. The native support
model is 8.04 GiB and remains resident while the target can use SSD streaming.
On 128 GiB hosts, leave room for both models and context buffers. This path
supports one Metal, CUDA or ROCm host and two-rank Metal/CUDA TP. For TP,
pass the matching `--dspark --mtp-model DSPARK` options on both peers.

- Run `python3 tests/test_deepseek41_dspark_conversion.py` and audit every
  real tensor with `python3 gguf-tools/deepseek41_dspark.py --hf SOURCE
  --source-revision REVISION --out DSPARK --verify`. The source must be
  pinned and the three native draft shards
  present. Preserve MXFP4 experts and all source values; no extra quantization.
- Build `tests/test_deepseek41_dspark` for the backend under test. Run
  `tests/test_deepseek41_dspark --candidate-mask` to check the all-blocks
  shortcut and both sides of the 16K candidate-selection boundary. Run
  `tests/test_deepseek41_dspark TARGET DSPARK DUMP_DIRECTORY`, then
  `python3 tests/deepseek41_dspark_reference.py SOURCE DUMP_DIRECTORY`.
  CUDA uses `--expert-activation q8-mmq`; ROCm uses `--expert-activation q8-k`
  for its existing expert activation policy. This independent native-weight oracle checks target
  feature capture, all three stages, Markov logits and confidence.
- Run `tests/test_deepseek41_dspark TARGET DSPARK --runtime`. Require causal
  logits when rejected future tokens change, every retained prefix from zero
  through six, direct stop-boundary rewinds, continued decoding and save/load
  after verification. Exercise opportunistic and exact sampling. Loading an
  older target snapshot must warm missing draft features rather than reuse
  stale ones. Repeat the backend primitives, including 128-expert/top-3 routing.
- Repeat the runtime test on physical Metal RDMA and CUDA RoCE pairs:
  `tests/test_deepseek41_dspark TARGET DSPARK --runtime-tp HOST PORT DEVICE GID`.
  Start the normal `ds4` TP worker with context 2048 and the matching drafter.
  Require all prefix commits, correction rewinds, both sampling modes and
  cancellation to finish without a lost peer. Also reject mismatched DSpark
  settings at startup (`DS4_TEST_TP_MODE_MISMATCH=3` in `test_tp_link`).
- Build `gguf-tools/quality-testing/score_official_verify`. Compare the matching
  official manifest with `--mtp-model DSPARK` against the same command plus
  `--verify-batch 6`. Keep `--ssd-streaming --ssd-streaming-cache-experts 40gb`
  and context identical. Score the short/boundary fixture above and
  `deepseek-v4.1-flash-20260919-router/manifest-sensitivity.tsv` at context
  34816; also repeat its `manifest-long.tsv` with `--continued-prefill 32`.
  Report paired teacher-forced NLL and API top-token agreement. Greedy sampled
  prefix length and scalar-versus-verifier text equality are not quality gates.
  Repeat the scalar/verifier comparison on both physical TP pairs using the
  normal coordinator flags instead of SSD streaming; start a matching worker
  for each run. Verify the complete vocabulary, not just the local half.
- Run real CLI and agent code/prose requests at temperature zero and above
  zero, including `--mtp-exact-sampling`. Against an otherwise idle speculative
  server, run `tests/test_server_speculative_stop.py --model deepseek-v4.1-flash
  --output OUTPUT_DIRECTORY`. Require correct retrieval, tool results, SSE
  completion, prompt-prefix reuse and bounded finish latency. Repeat a real
  coding client's edit/build/test loop and cancellation near context capacity.
  Record acceptance and end-to-end timing; working speculation does not by
  itself establish a speedup.
- On ROCm, repeat at least ten actual server cold starts outside a debugger,
  alternating default and exact sampling. Send a request, check the answer,
  and shut down cleanly each time. Watch usable host memory, excluding CMA.
  If startup crashes, retain the binary and core or backtrace plus kernel
  logs; distinguish a userspace HIP failure from OOM or a GPU reset. A clean
  rerun is evidence against recurrence, not an explanation of a prior crash.

### Metal Reference Checks

The calibrated Q2 main weights occupy about 151.8 GiB, Q4 about 294.15 GiB;
188.8 GiB of Engram remains disk-only. Validate the release artifacts with
the converter/payload and download checks above, not old bootstrap filenames.

Compare against the corresponding stored fixture and execution schedule:

| Model / execution | Fixture | Mean NLL | API top-token agreement |
| --- | --- | ---: | ---: |
| Q2, M5 SSD | General 100 | 0.364576009 | 2697/2994 |
| Q2, M3 Ultra resident batched prefill | General 100 | 0.365680596 | 2705/2994 |
| Q2, Metal TP | General 100 | 0.361471738 | 2699/2994 |
| Q4, M5 SSD | General 100 | 0.246523256 | 2896/2994 |
| Q4, M3 Ultra resident | General 100 | 0.252409161 | 2896/2994 |
| Q4, M5 SSD | Long 9, 8K/16K/32K | 0.335651410 | 542/576 |
| Q4, M3 Ultra resident | Long 9, 8K/16K/32K | 0.328760094 | 538/576 |
| Q2, M3 Ultra resident | Extended 6, 64K/96K | 0.558586149 | 327/384 |
| Q2, Metal TP | Extended 6, 64K/96K | 0.560655370 | 328/384 |
| Q4, M3 Ultra resident | Extended 6, 64K/96K | 0.366465846 | 360/384 |

These are reference scores, not claims of bit-identical cross-backend quality.
Different prefill schedules and hardware can change rounding and expert routes.
Compare paired case scores on the same schedule, including the long fixtures;
unchanged top-token totals alone do not rule out a loss increase. A release
calibration must record coverage and retain the fallback for unobserved experts.

- For Q4 native batches through eight rows, retain the 0.0002 full-logit
  bound against scalar controls. Test the consumed fused outputs rather than
  optional gate/up scratch that a fused kernel need not write. Keep exact
  companion-isolation and cache-state checks separate from scalar parity.
- Keep the warm-SSD matrix-prefill threshold at 1024 tokens when at least
  half the experts fit unless a new threshold passes the nine long and six
  extended official fixtures. A speed win at 512 tokens is not sufficient.
- During Q4 SSD tests, verify that persistent full-expert tables cannot retain
  whole layers outside the cache budget. Start with a guarded 32 GiB cache,
  then increase only after wired memory and swap remain bounded. Planned
  allocation alone is insufficient evidence of safe physical residency.
- Repeat the explicit non-Metal-4 control when investigating M5 versus M3
  scores. Test residency, prefill geometry and kernel arithmetic separately;
  do not infer a transport or SSD error from cross-hardware logits alone.

Q4 resident reference on a 512 GiB M3 Ultra: default paths, disk-only Engram,
36,864-token allocation, no Metal validation or speculation. These are medians
of three complete sweeps after a warmup, with 128 output tokens per frontier.
Exclude loading. Only the first row is initial prefill; later rows are appends.

| Context frontier | Added tokens | Prefill t/s | Generation t/s |
| ---: | ---: | ---: | ---: |
| 4,096 | 4,096 | 341.75 | 18.69 |
| 8,192 | 4,096 | 339.10 | 18.67 |
| 16,384 | 8,192 | 641.49 | 18.32 |
| 32,768 | 16,384 | 715.96 | 18.11 |

```sh
./ds4-bench -m gguf/DeepSeek-V4.1-Flash-Q4.gguf \
  --prompt-file speed-bench/promessi_sposi.txt --ctx-start 4096 --ctx-max 32768 \
  --step-mul 2 --ctx-alloc 36864 --gen-tokens 128 --show-output --csv RESULT.csv
```

### CUDA SSD Streaming

V4.1 CUDA starts with text inference on a single DGX Spark. Test on
`toor@192.168.4.180` and `toor@192.168.4.181`, one model process per host.
Never load this Q2 model resident on a 128 GB Spark. Its 341 GiB file includes
189 GiB of disk-only Engram; the remaining weights still exceed RAM.

- Build with `make cuda-spark`, then `make CUDA_ARCH=sm_121
  tests/test_deepseek41_cuda tests/test_cuda_ssd_cache tests/test_cuda_q8_rows
  tests/test_deepseek41_prefill tests/test_cuda_ssd_batch
  tests/test_cuda_session_batch tests/test_cuda_mixed_batch`. Require warning-free builds.
- Run `tests/test_deepseek41_cuda`, its `--attention-output-large` mode,
  `tests/test_cuda_q8_rows`, and `tests/test_cuda_ssd_cache`. Check router ties,
  384 experts, long
  absolute RoPE positions, FP4/FP8 exponent range, masked sparse IDs, odd pair
  pooling, and the 8,192-row projection launch boundary. The cache oracle
  checks IQ2/Q2_K, Q4_K and MXFP4, eviction, remapped slots, small/zero and
  10,000-slot budgets, and prefill after its source becomes unreadable.
  Also check next-layer read-ahead: protect active inputs and pending slots,
  never publish partial reads, and evict unused read-ahead before demand-hot
  experts. Exercise cancellation, changed budgets/model descriptors and
  `tests/test_cuda_ssd_cache --prefetch-exit`; repeat with
  `DS4_CUDA_NO_DIRECT_IO=1` to cover buffered reads.
  Prefill must not accumulate full expert tensors in an unbounded second
  cache. Q8 row projections must match scalar execution
  with ragged shapes, untouched output tails and no padding after the weights.
- Repeat primitive and cache tests under Compute Sanitizer. On driver
  580.173.02, its host backtrace collector can itself crash; use
  `--show-backtrace device --report-api-errors no --error-exitcode 99` in
  that case. Device memory checking must remain enabled, with zero errors.
- Run `tests/test_deepseek41_prefill --cuda MODEL
  speed-bench/promessi_sposi.txt`. This uses two sessions and a 64 GiB cache
  budget, mixing scalar decode with 256/1K/4K continued prefills. Check full
  saved state, logits, progress callbacks, cancellation, restore and subsequent
  decoding. Partial layer stacks must not be accepted as snapshots.
  Repeat with `--cuda-small` for the 24 short/large append transitions and
  `--cuda-long` for decoder/deferred-state boundaries through 60K. Short
  CUDA IQ2/Q2_K SSD prefills use exact 2..8-row chunks below the 256-token
  matrix-prefill threshold. Check both sides of each dispatch boundary.
  Compare CUDA SSD read-ahead and medium one-sweep appends with
  `DS4_CUDA_DISABLE_SSD_PREFETCH=1` and
  `DS4_CUDA_DISABLE_SSD_MEDIUM_SWEEP=1`. The optimized path must preserve
  complete logits and saved state. Time actual `/read README.md`, fresh and
  continued 1K/2K/4K/8K/32K prompts, and the following decode, with both
  automatic and explicit cache budgets. Check whether prefills displace
  useful decode experts, and report any decode regression alongside the
  prefill gain. Unused read-ahead must not count as an actual demand hit.
- Score the same V4.1 short general and long manifests used on Metal, not the
  V4 Flash vectors. Exercise 8/16/32K sparse frontiers and continued prefill.
  Compare paired probability scores, not sampled-prefix length alone.
- Run `make test-engram` and the Engram reader under ASAN/UBSAN. Parallel disk
  reads must preserve duplicate-row order, BF16 values, bounded scratch,
  error reporting and cleanup after truncated or invalid data.
- Test a native agent editing task and a server coding session with real tool
  calls. Check follow-up prefix reuse, concurrent requests, streaming client
  cancellation and reuse of the freed slots. V4.1 IQ2/Q2_K SSD supports native
  batches of 2..8 rows. Run the CUDA session oracle with two/eight sessions,
  including sparse frontiers, and the mixed oracle with five prefill plus
  three decode rows. A 128-token prefill plus three decode rows must exercise
  the ordered fallback. Require native-path counters and compare full logits,
  cache state, snapshots, session reorder and invalidation with isolated runs.
  Repeat `tests/test_cuda_ssd_batch` normally and under Compute Sanitizer.
- Monitor available host memory and swap through initial/continued prefill,
  decoding and reloads. Test automatic and explicit expert budgets. Confirm
  resident mode rejects an oversized model before warming or allocating it.
  Include 8K prefill chunks with an automatic cache near 80 GiB: prefill must
  schedule the active experts, not every slot in the global decode cache.
  Repeat older V4 Flash Q2 resident and SSD tests after shared CUDA changes.

#### CUDA SSD Reference Workloads

Use `DeepSeek-V4.1-Flash-Q2.gguf`, disk-only Engram and no speculation.
Read-ahead should use the admitted cache and two 8 MiB staging buffers; it
starts at 2K tokens when two layers fit. Medium appends retain 2048-row
arithmetic partitions while visiting each layer once. Test those thresholds
with `--cuda-small` and `--cuda-long`, not just a large fresh prompt.

With a 64 GiB expert-cache hint (56.88 GiB dynamic plus 7.12 GiB prefill
reserve), 64K context, Promessi Sposi and 32 teacher-forced output tokens,
these single-run reference points cover fresh and continued prefill:

| Existing tokens | Added tokens | Prefill |
| ---: | ---: | ---: |
| 0 | 2048 | 84.90 t/s |
| 2048 | 2048 | 90.70 t/s |
| 0 | 4096 | 116.34 t/s |
| 4096 | 4096 | 116.31 t/s |
| 0 | 8192 | 228.44 t/s |
| 8192 | 4096 | 111.15 t/s |
| 0 | 32768 | 384.45 t/s |
| 32768 | 3241 | 87.77 t/s |

```sh
./ds4-bench --cuda -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ssd-streaming \
  --ssd-streaming-cache-experts 64GB --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 32768 --ctx-max 36009 --step-incr 3241 --ctx-alloc 65536 \
  --gen-tokens 32 --teacher-forced-decode --csv RESULT.csv
```

`64GB` denotes 64 GiB for this option. Check startup's resolved cache budget:
when memory is tight, the runtime can reduce it below the requested hint.
Repeat with automatic sizing, currently about 80 GiB on an idle Spark. A
medium append can displace decode-hot experts, so include at least 256 decode
tokens after it and compare total latency with read-ahead/sweep disabled.
For a roughly 3.2K `/read` prompt at 32K context, the automatic-cache reference
is about 100 t/s prefill and 9.3 t/s decode over 256 output tokens. Construct
and retain the actual prompt: the repository README's length changes.

Eight-session native batching reference: about 11 aggregate t/s at 1K context,
24 steps per session and automatic cache. Reproduce with
`tests/test_cuda_session_batch`, `DS4_TEST_CUDA_SINGLE_GPU=1`,
`DS4_TEST_SSD_CACHE_GIB=auto`, `DS4_TEST_SESSION_COUNT=8`,
`DS4_TEST_BATCH_ONLY=1` and `DS4_TEST_MODEL` set. Compare with
`DS4_CUDA_SESSION_BATCH_MOE=0`; require exact full-logit/frontier hashes.
Mixed five-prefill/three-decode rows must pass too; larger mixed prefills
must select the ordered fallback rather than claim native execution.

Quality reference bands: general 100 NLL 0.363135483, 2696/2994 API top tokens;
boundary 17 NLL 0.405201234, 126/141; long 9 NLL 0.558581488, 491/576.
Use the complete matching manifests and continued-prefill variants. A smaller
subset or an older Flash fixture is not a substitute for these gates.

### CUDA Network Tensor Parallelism

Use the two Sparks above with identical commits and matching Flash 0731 Q2,
Flash 0731 MXFP4 or V4.1 Q2 files, one GPU per rank. Expert shards are resident;
V4.1 Engram stays on disk. Do not combine
network TP with `--ssd-streaming`, `--cuda-tensor-parallel` or `--quality`.
The Linux transport uses RoCEv2. Spark shares pinned host-visible exchange
buffers with CUDA, with device/host staging as a fallback; this is not discrete
GPU GPUDirect. Both peers must use the same protocol version and agree on
expert splitting, verification exchanges and vocabulary-head splitting.
Incompatible peers must fail at the handshake, before inference.

For Flash 0731, score the full matching 100-case continuation set and its
continued-prefill variant against a single-host control. Run fresh 32K and
appended 8K prefills and compare full frontier logits with the prior build.
Use `tests/test_metal_tp_spec MODEL SUPPORT HOST PORT DEVICE GID --cuda`
with a CUDA worker for short/continued speculative verification, and
`tests/test_metal_tp_cancel MODEL PROMPT HOST PORT DEVICE GID --cuda` for
cancellation/rebuild. Repeat speculative verification with `--tcp` and a TCP
worker. Require no
errors or unexpected accepted-token replay; report scalar/batch logit gaps
rather than claiming exact greedy equality when the oracle allows a tolerance.
Compare default DSpark settings as well as unpruned diagnostic runs, on short
prose, code and a book continuation with at least a 4K prefix. Record accepted
tokens per cycle and time spent proposing/verifying. Inspect the continuations
for repetition; a changed passage is not a controlled throughput comparison.
Check the independent scalar-target oracle on prose too, not just coding
prompts. Report control failures and logit gaps without relaxing the tolerance
to make a candidate pass.

For default Q2 and MXFP4 DSpark over RDMA, pass `--dspark --mtp-model SUPPORT`
on both ranks and require the `TP DSpark drafter split active` message when
memory allows the optional halves. Repeat with
`DS4_TP_DISABLE_DSPARK_SPLIT=1` to check the local-drafter fallback. Run
`make test-dspark-history` for the shared cache-window and confidence inputs.
The history check covers local and split-drafter maintenance separately.
Run `make test-cuda-dspark-tp CUDA_ARCH=sm_121` and repeat its fixtures under
Compute Sanitizer. The shared-expert overlap fixture must match the inline
path bitwise at every supported batch width; run racecheck and initcheck too.
Check sampled and exact-sampling modes, partial and
seed-only commits, snapshot rebuild, continued prompts and server tool loops.
Rank-state or proposal disagreement is a failure, even if local drafting
later lets the request finish. Repeat without worker support, over TCP and
with `DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6=1`: supported configurations must keep
local drafting. Steering and tensor dumps also retain the local drafter.
Check the extra expert-half allocation against requested contexts and session
count; insufficient headroom must disable the optional split, not cause OOM.

For verifier arithmetic changes, teacher-force official continuation tokens
through the actual speculative verifier, not just the ordinary scorer's
prefill path. Compare NLL and API top-token agreement per prompt against the
prior build, with uncertainty intervals. A changed greedy prefix alone does
not establish a quality regression; nor does coherent text establish parity.
The speculative oracle also interleaves 256- and 8192-token sessions and
frees/recreates workspace borrowers. Require exact isolated-session logits.
In a multi-session server run, both peers must report shared prefill scratch;
the CLI worker must not allocate a separate full workspace per session.
Check memory headroom on both hosts, not just the coordinator's estimate.
Flash network session batching currently uses the mirrored ordered fallback;
its full logits must match isolated sessions, but do not report it as native
batching. The native batching requirements below apply to V4.1.

- Build `tests/test_tp_commands`, `tests/test_tp_tcp`, `tests/test_tp_rdma`,
  `tests/test_tp_link`, `tests/test_cuda_tp` and `tests/test_cuda_ssd_batch`.
  Run the command/TCP/RDMA unit tests under ASAN/UBSAN. Tiny socket buffers,
  stalled peers, half-close and disconnect must fail promptly, not deadlock.
  Inject one-sided head-row and verify-window failures. A failed context
  must refuse subsequent commands and gates without touching the transport;
  recovery requires a new connection, not a replay on the failed link.
  Repeat the physical link test with `DS4_TEST_TP_HEAD_ROW_FAULT=1` on both
  peers to check that a rejected head row sends no replay or decode command.
- Run the physical link test in both directions on each available direct
  link, with TCP and RDMA. Check the reported device, RoCEv2 GID and RC
  transport. A management-network ping is not an RDMA test. Stop a worker
  during exchange and require bounded coordinator failure.
  Exercise command-mailbox ring wrap, delayed credits, oversized SYNC
  followed by short commands, immediate STOP and peer exit. Disable the
  mailbox on just one rank: both must agree on the TCP command fallback.
  Verification begin records and head rows must belong to the current
  exchange, including after rollback; stale readiness must never suffice.
- Run `tests/test_cuda_tp_link` over RDMA and TCP, including staged buffers
  and Compute Sanitizer. Check one- and two-gate blocks, asymmetric offers,
  transitions back to scalar decode, and exact row payloads. Enable graph
  diagnostics on just one model peer: both must negotiate the replicated
  verifier, not exchange incompatible attention or shared-expert partials.
- Run `tests/test_cuda_tp` normally and under Compute Sanitizer. Check
  device-to-host visibility, row/batch/bulk exchange, growing staging buffers,
  output canaries, failed-peer propagation, rebind and cleanup. Transport
  registration must be released before its staging buffers are freed.
- Run `tests/test_cuda_ssd_batch --owned` and `--owned-mmq`, including under
  Compute Sanitizer. Cover 5120-wide inputs with more than eight rows, both
  ranks, empty local contributions, three/six selections and 384 experts.
  The IQ2 lookup tables must be initialized even when activations exceed
  the small shared cache. Poison scratch buffers: absent owned assignments
  must not read unwritten gate/up/down outputs or compute placeholder experts.
  `--owned-mmq-large` also checks 8191/8192/8193 rows through the global
  assignment-map path. V4.1 small owned batches must match each rank's scalar
  partials exactly, not merely produce a close combined sum.
- Run `tests/test_cuda_ssd_batch --owned-mxfp4`, also under Compute Sanitizer.
  Cover one through eight verifier rows and the 129-row matrix path, both
  owned halves, empty contributions, poisoned scratch and output canaries.
  The generic network entry must match explicit ownership without modifying
  router IDs or weights. Score the matching MXFP4 model through the actual
  verifier as well as ordinary prefill; a Q2-only pass does not cover it.
- Run `tests/test_cuda_mxfp4_split`, also under Compute Sanitizer, for the
  intermediate-dimension split. Gate/up rows must be exact; down-half sums
  must satisfy the reference tolerance. Check returned allocation bytes,
  invalid/unbound calls, and that startup retains no duplicate raw experts.
  Compare official verifier scores: the changed reduction is not bitwise.
- Run `tests/test_cuda_verify_head`, also under Compute Sanitizer. Split
  halves must match every full-head logit and its global top token, including
  ties and non-finite inputs. On the physical pair, test partial/full accepts,
  cancellation, snapshot rebuild, continued prefill and both sampling modes.
- Run `tests/test_cuda_tp_repack`, also under Compute Sanitizer. Compare
  aligned expert shards with independent raw shards and the whole expert
  table. Include empty rank contributions, poisoned scratch, output canaries,
  384 experts and the actual 5120/2304 dimensions. Small rows must be exact;
  large prefills must retain their validated accumulation order. The builder
  must reject invalid ranks and truncated mappings without touching Engram.
- Repeat the CUDA attention-output primitive checks with the TP projection
  enabled, including ragged rows, truncated weights and output tails.
- Run `tests/test_cuda_q8_rows`, including `--tp-head` under Compute
  Sanitizer. Both compact vocabulary halves must exactly match their rows
  in the full head, including native batches and untouched output tails.
  Check full logits after prefill, scalar decode, batch decode and restore;
  a missing worker half must fail rather than leave stale logits usable.
- Run `make test-cuda-reductions CUDA_ARCH=sm_121`, also with Compute
  Sanitizer memcheck and racecheck. Weighted RMS and FP32 projections must
  match the independent original reduction exactly, including zero inputs,
  mixed magnitudes, ragged widths, in-place normalization and output tails.
  Check model scores and timings too: a faster primitive alone does not
  establish an end-to-end speedup.
- Run `make test-cuda-shared CUDA_ARCH=sm_121` on Spark, also with
  Compute Sanitizer memcheck and racecheck. Concurrent shared/routed work
  must retain the serial result: exercise independent scratch users,
  captured graphs, changed inputs on replay, partial-launch failure and
  repeated cleanup/reinitialization. Verify model logits with graphs both
  enabled and disabled before accepting a scheduling change.
  Check the shared expert's gate, up and intermediate BF16 values too, with
  clamped/unclamped and large/small inputs, including the real 5120x2304 shape.
- Build `tests/test_cuda_tp_link` and run it on both hosts as
  `tests/test_cuda_tp_link RANK COORDINATOR PORT rdma DEVICE GID` (rank 0/1).
  Repeat under Compute Sanitizer, with TCP, and with
  `DS4_CUDA_DISABLE_TP_HOST_SLAB=1` for the staged fallback. Scalar and changing
  batch sizes must return exact payloads; a stopped peer must fail within
  the configured deadline. Check shutdown and reinitialization too.
- Run `tests/test_tp_rdma` under ASan/UBSan after transport changes. Cover
  negotiated 16/32/64 KiB completion accounting and malformed completions.
  On the physical pair, test current peers and one previous-version peer;
  Linux RC may negotiate larger messages, while old/Metal peers keep 16 KiB.
  Follow bulk prefill with decode and another bulk transfer on the same QP.
- Record CPU affinity on both ranks for paired timings. Spark has faster
  and slower CPU groups; compare the same allowed group and do not change
  system-wide CPU settings between runs.
- Run `tests/test_deepseek41_prefill --tensor-parallel-cuda MODEL
  speed-bench/promessi_sposi.txt LISTEN_HOST PORT RDMA_DEVICE GID` against
  a worker with `--ctx 16384`. Require full-state/logit agreement, dispatch
  boundaries, progress callbacks, cancellation, snapshots and both-rank
  prefix replay. A cancelled rank must not leave its peer's frontier valid.
  Repeat with `--tensor-parallel-cuda-small` for short-append boundaries;
  the control must mirror scalar execution on the worker as well.
- Score general100, boundary17, medium continued12 and long continued9
  against the saved V4.1 API continuations. Include 32-token appends at
  sparse 8/16/32K frontiers. Compare paired case scores, not just coherent
  text or exact agreement with a differently rounded execution path.
- Run native-agent editing and actual server tool loops on the pair. Check
  follow-up prefix reuse, concurrent streaming/nonstreaming requests,
  cancellation and slot reuse. Include five through eight active sessions
  so native CUDA TP batching is exercised; smaller groups run in order.
  Eight 4K contexts fit the tested Sparks; eight 8K workspaces are rejected
  by memory admission. Do not weaken that guard to make the test fit.
- Run `tests/test_cuda_session_batch` with `DS4_TEST_SESSION_COUNT=8`,
  `DS4_TEST_TP_LISTEN_HOST` set to the direct-link address and a matching
  worker. Require exact full-vocabulary logits, reordered rows, invalid-input
  rejection and snapshot replay. Repeat with `DS4_CUDA_SESSION_BATCH_MOE=0`.
  Also test five, six and seven sessions against isolated scalar controls.
  Run `tests/test_cuda_mixed_batch` with both a five-token and a 128-token
  prefill quantum, checking native eight-row work and the ordered fallback.
  Measure native and ordered throughput separately from compilation and
  diagnostic instrumentation; do not enable slower small groups by default.
- Check CUDA decode graphs against `DS4_CUDA_DECODE_GRAPHS=0`. Use
  `DS4_CUDA_DECODE_GRAPH_LOG=1` in a separate diagnostic run to confirm
  capture on both ranks, not silent fallback. Cover alternating batch/scalar
  decode, scratch growth after continued prefill, cancelled work, snapshots,
  and freeing/recreating sessions. Require exact logits and unchanged official
  continuation scores. Repeat resident Flash 0731 regression checks because
  the backends share graph infrastructure.
  V4.1 captures three regions per layer: input projections, attention output,
  and the FFN. Position-dependent attention and network gates stay outside
  capture. Require all 120 regions on each rank in the scalar diagnostic.
- Monitor host memory and swap throughout startup, prefill and reload. Each
  rank should retain about 80.57 GiB of weights: 71.19 GiB of derived expert
  artifacts and 9.38 GiB of raw tensors. Do not retain duplicate raw experts,
  unowned experts or disk-only Engram. Check allocation bytes as well as
  payload bytes:
  large ranges must not waste most of their arena blocks. Use identical
  prompts/context limits for paired prefill and decode timings.

#### CUDA TP Reference Workloads

Use the direct RoCE link, Q2 shards, disk-only Engram and no speculation.
A 1K Promessi Sposi prefix, 64K context and 2048 teacher-forced decode tokens
has a reference mean of about 21.8 t/s; 1K prefill is about 209 t/s. Capture
CPU affinity, temperature and clocks on both ranks. Repeat long runs rather
than discarding low results without explaining them.

Long-context reference points, 64K allocation and 512 teacher-forced decode
tokens per frontier (October 2, 2026):

| Existing tokens | Added tokens | Prefill | Decode |
| ---: | ---: | ---: | ---: |
| 0 | 32768 | 410.92 t/s | 17.97 t/s |
| 32768 | 8192 | 313.70 t/s | 17.93 t/s |

These are single measurements, not release medians. Prefix replay between
frontiers is excluded from append time; record it separately. Both ranks
used `taskset -c 5-9,15-19`; verify which CPU cores are faster on the test
hosts rather than assuming these IDs apply to every Spark. Eight-session
native decode is roughly 28 aggregate t/s at 1K allocation. Never report that
number as per-client throughput or a single-session result.

The stored 138-case quality references have mean NLL 0.364937088 (general
100), 0.401415795 (boundary 17), 0.523971083 (medium continued 12) and
0.566939577 (long continued 9). Require complete, finite score files and
paired comparisons, also with decode graphs disabled. Separately run the
64K/96K official vectors; the smaller set does not cover them.

## 18. Qwen3.8 Flash Next

- Use the self-contained Q2 and Q4 GGUFs with original BF16 n-grams. Old
  main-only files and quantized n-gram sidecars are not the release layout.
  Run `make test-qwen4-ngrams test-deepseek41-gguf` and
  `python3 -m unittest discover -s gguf-tools/tests -p test_qwen4_native_ngrams.py`.
  Repeat the reader with ASan/UBSan. Check exact BF16 values, duplicate and
  reordered rows, parallel reads, truncated files, invalid IDs and cleanup.
  Build `tests/test_qwen4_ngram_state` and run it with each real model under
  Metal validation and CUDA: prefill, decode and MTP failures must invalidate the live
  frontier; rebuilding and continuing must match an independent session exactly.
- Audit every copied main/MTP tensor against its input and every n-gram shard
  against the pinned BF16 source. Packaging must not requantize the calibrated
  experts. Verify the final checksum and download target before release.
- Confirm the table is outside the runtime mapping and all Metal residency
  views and CUDA model caches, including weight warming. Measure actual memory during short/long
  prefill and generation; adding 95.37 GiB on disk must not add that much RAM.
  Test one model at a time on an M5 Max. Keep space for the complete output
  plus a reserve during conversion; do not fill the system disk.
- Run ordinary and MTP decoding, including exact sampling, at small and large
  initial/continued prefixes. Follow the save/restore, rewind, checkpoint,
  logit-dump, vision and steering tests in `docs/QWEN38_FLASH_NEXT.md`.
  A table read failure must stop inference, not consume stale staging data.
- Exercise native-agent coding and real server tool-result continuations with
  prefix reuse, then an image turn and a text follow-up. Compare quality and
  speed against the old table separately: restoring original BF16 changes
  logits, so byte-identical text is not a quality requirement. Disk-only
  timings must be taken without concurrent downloads or model copies.
- Repeat focused DeepSeek and GLM checks after shared loader changes. Record
  skipped hardware or reference checks explicitly; coherent Qwen replies do
  not establish parity with the original HF model or a hosted API.
- On CUDA, run `make test-qwen4-cuda` and compute-sanitizer memcheck. Exercise
  Q2's padded down rows, Q4_K/MXFP4 experts, all dense formats, long recurrent
  scans, sparse causal selection and MTP snapshots against the CPU oracles.
  Tensor-core paths must pass the same tolerances as scalar kernels.
  Repeat with `DS4_TEST_QWEN4_ATTN_GROUPS=1` under memcheck, racecheck and
  synccheck. This covers grouped prefill and split decode attention against
  independent double-precision references, including empty/masked selections,
  large/small queries, head-group tails and long dense/sparse contexts.
  Repeat those sanitizer tools with `DS4_TEST_QWEN4_DENSE_ONLY=1` and
  `DS4_TEST_QWEN4_EXPERT_TILES=1`. Check residual-corrected dense projections,
  odd strides, large outputs, partial expert tiles and untouched output guards.
- Build `tests/test_qwen4_prefill` and run it with a real prompt through at
  least 8K context on both Metal and CUDA. Same-schedule replay must agree;
  record different-schedule probability differences separately. Nearly tied
  experts can amplify normal rounding, so a max-logit difference alone does
  not establish a state bug or a quality regression.
- For Metal native batching, run `tests/test_metal_session_batch` with
  `DS4_TEST_BATCH_ISOLATION=1` at two, four and eight sessions, including a
  sparse prefix. Reordered rows and changed companions must leave the target's
  complete logits identical. Separately score both official manifests with
  `--session-batch 1`, `4` and `8`; compare paired losses and API agreement.
  The scalar and batched reductions need not be bit-identical. Investigate
  greedy mismatches with their logit margins, not by loosening kernel tests.
  Run `tests/test_qwen4_kernels` under Metal validation and the real-model
  `tests/test_qwen4_ngram_state` for mixed ordinary/MTP cycles, failed batch
  reads, exact recovery and the final context slot.
- Start a four-slot Qwen Metal server with and without `--mtp`. Check concurrent
  tool calls, prefix reuse, cancellation, stop strings and one/two-token output
  limits. Repeat with `--mtp-exact-sampling`, mixing temperature-zero and
  sampled requests. Only the former may use greedy speculative acceptance.
  Compare seeded sampled replies in exact mode; default greedy MTP changes
  random draws with draft scheduling, so equal seeds need not give equal text.
  Benchmark ordinary and speculative batches on both prose and code, using
  `speed-bench/session_concurrency_bench`; record aggregate and per-session
  throughput. Keep n-grams on disk and monitor memory during arena growth,
  session destruction/recreation and engine cleanup.
  Run `python3 -m unittest discover -s tests -p test_serve_concurrency_bench.py`
  so truncated/error streams cannot silently enter the throughput results.
- Run `tests/test_server_story.py` with at least 49K server context: all sixteen
  story facts, the correction turn and cached-prefix reuse must pass. Also
  run `tests/test_agent_vision.py` with a long archive and
  `tests/test_server_vision_agent.py` through Pi's three supported APIs.
  Check the code with independent assertions, not the agent's own report.
  Run the CUDA session-batch oracle with Qwen to check isolation and reordered
  serial fallback; do not label its throughput as native batching.
  For `tests/test_cuda_mixed_batch`, set `DS4_TEST_CUDA_SINGLE_GPU=1` and
  `DS4_TEST_ALLOW_FALLBACK=1`; the full-logit oracle still requires exact replay.
  Repeat with explicit `--gpu-vram` admission: exclude disk-only n-grams,
  include each session's independent workspace, and reject insufficient
  resident-weight budgets. Mixed prefill/decode must use the Qwen fallback,
  never the DeepSeek graph.

### Qwen Quality And Speed References

Run both official Alibaba manifests in
`gguf-tools/quality-testing/data/qwen38-flash-alibaba-{100,long}` using the
commands in `docs/QWEN38_FLASH_NEXT.md`. Test default and `--quality` paths,
and repeat the long set with `--continued-prefill 1` and `256`. Check complete
case/token counts, finite results and alignment with the published tokenizer.
The hosted model is based on Flash Next but is not a byte-pinned checkpoint;
two split-emoji cases lack usable API token alignment and remain in NLL scoring.
Do not interpret missing API metadata as perfect agreement.

Current CUDA default-path references, short 100 and long 12 (2K-24K prefixes):

| Model / suite | Mean NLL | API top-token matches |
| --- | ---: | ---: |
| Q2 short | 0.353129 | 4950/5568 |
| Q4 short | 0.290299 | 5127/5568 |
| Q2 long | 0.155612 | 721/766 |
| Q4 long | 0.125160 | 745/766 |

Compare paired per-case losses and high-confidence API disagreements, not
only aggregate top-token counts. Fresh and continued schedules can differ
slightly; require exact same-schedule replay and independent task quality.
Do not change kernel tolerances merely to accept a faster implementation.

Single-Spark Q2/Q4, resident weights and disk-only BF16 n-grams, Promessi Sposi,
8192-token chunks and 128 teacher-forced decode tokens. These are two-run
means; use the section 16 procedure for release medians:

| Model | Initial 1024 prefill | Next 7168 prefill | Decode at 8192 |
| --- | ---: | ---: | ---: |
| Q2 | 516 t/s | 745 t/s | 22.6 t/s |
| Q4 | 513 t/s | 755 t/s | 21.2 t/s |

Also measure short continuations: eight 32-token Q4 appends after an 8K prefix
have a reference mean latency of about 273 ms each. Include following decode
so work is not merely deferred. Check ordinary and MTP generation on code
and prose, retaining text and acceptance statistics as required by section 16.
Repeat a fresh 32K Q2 prefix with `--prefill-chunk 32768`; its reference
prefill rate is about 771 t/s.

M5 Max Metal indicative single-run references, disk-only n-grams:

| Model | Initial 16K prefill | 16K append to 32K | Decode at 32K |
| --- | ---: | ---: | ---: |
| Q2 | 1492 t/s | 1438 t/s | 48.7 t/s |
| Q4 | 1458 t/s | 1330 t/s | 48.4 t/s |

At 8K context and a 1024-token chunk, planned memory is about 42.86/70.87 GiB
for Q2/Q4, excluding the 95.37 GiB disk table. Check physical memory and swap,
not only this plan. Repeat conventional Metal with Metal 4 disabled, and test
a physical pre-M5 device before advertising its performance or memory fit.

## 19. Release Sign-off

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
