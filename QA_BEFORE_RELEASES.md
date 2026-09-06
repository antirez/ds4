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
  The Q4 layout with Q8 KDA projections, embedding, and output head scored
  `0.300804038`, `90/100`, and `9.48` on M3 Ultra. Its paired BF16-layout
  control scored `0.300477636`, `90/100`, and `9.48`; the Q8 layout won 54 of
  100 cases despite its `0.109%` higher aggregate NLL.
- For GLM 5.3 attention changes, also run the eight long Z.AI FP8 cases.
  First run `python3 gguf-tools/quality-testing/render_glm_references.py
  gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-long`, then
  score its `manifest-rendered.tsv` at context 8192 with `--rendered-prompt`.
  This preserves the official low-effort template and returned reasoning
  before scoring the answer. Every prompt crosses the 2,051-token boundary.
  Compare the default path against the scalar control on the same fixture.
  Z.AI supplies no output logprobs; do not report zero logprob-error columns
  as API parity. The historical 100-case numbers above use a no-thinking
  prefix and are not directly comparable to rendered-prefix scores.
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
model loading, and proposal paths. Run these whenever DSpark support,
speculative verification, confidence policy, target hidden capture,
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
- For Metal DSpark verifier/proposer/replay changes, run this same-machine A/B
  matrix with `DS4_DSPARK_STATS=1`, greedy decoding, the same prompt and token
  limit, and no other environment changes:

  | Target expert cache | Expected DSpark depth | Legacy control | Candidate |
  | ---: | ---: | --- | --- |
  | 16 | 2 | `DS4_METAL_DSPARK_PROPOSER_BLOCK_MAX=0 DS4_METAL_DSPARK_ACCEPTANCE_ONLY_VERIFY=0 DS4_METAL_DSPARK_HEADLESS_REPLAY=0` | Leave proposer/headless unset; keep acceptance-only `=0` |
  | 32 | 5 | `DS4_DSPARK_SSD_VERIFY_BLOCK_MAX=5 DS4_METAL_DSPARK_ACCEPTANCE_ONLY_VERIFY=0 DS4_METAL_DSPARK_HEADLESS_REPLAY=0` | Keep the verifier cap, set acceptance-only `=1`, and leave proposer/headless unset |

  Use `--ssd-streaming-cache-experts 16` or `32` to match the row. The 0731
  top-6 verifier needs 30 effective slots for five draft rows; 32 leaves a small
  margin. Require byte-identical stdout between control and candidate and
  `errors=0`, `verifier_unavailable=0`, `proposed>0`, and
  `accepted_draft>0`. Record generation t/s, acceptance, `propose`, `verify`,
  `replay`, `prop_capped`, `prop_scheduled_rows`, `metal_accept_only`,
  `metal_verify_rows_saved`, and `metal_replay_headless`. In the candidate,
  eligible `N >= 3` verification cycles should save one target row; aligned
  ratio-4 boundaries intentionally remain on the legacy path. The
  depth-2 run exercises proposer capping and headless replay while retaining
  the legacy verifier; the depth-5 run should retain the checkpoint's native
  five proposal rows and exercise acceptance-only verification.
  On low-memory Metal, repeat the depth-5 candidate once with
  `DS4_METAL_DSPARK_PIN_MAIN_PROJ=1`. Require a startup log confirming the
  locked byte count, identical stdout and acceptance, and compare
  `prop_setup`, total `propose`, page faults, and generation t/s. A lock
  failure or a slower median keeps this optimization opt-in.
- For the experimental Metal SSD exact-2 verifier, repeat the depth-2 row
  above with `DS4_METAL_DSPARK_EXACT2=0` as the control and `=1` as the only
  candidate change. Set `DS4_DSPARK_FIXTURE_REQUIRE_EXACT2=1` only on the
  candidate. Require byte-identical stdout against both control and
  target-only output, `exact2_attempt>0`, `exact2_full>0`,
  `exact2_fallback=0`, and `errors=0`. Record generation t/s, `verify`, and
  `replay`; then repeat for at least 100 generated tokens to catch cumulative
  state drift. Do not infer that the generic five-row batch state is directly
  committable from this two-row result.
- For Metal exact-union or the AProjQ4/HC decode fusions, first run the
  model-backed oracle with the target AProjQ4 GGUF:
  `DS4_TEST_MODEL=/path/to/deepseek-v4-flash-aprojq4.gguf make test-metal-exactn-oracle`.
  Require its N=2..5 cases to be byte-identical to sequential decode for
  serialized KV/compressor state, logits, and the four-token continuation.
  The matrix must include full accepts for N=2,3,4,5, all N=5 partial prefixes
  1..4, and EOS in the first and a middle row. This is a correctness gate, not
  evidence of a speedup.
- The Q8 Q-A/KV compound rows below require a separate AProjQ8 target whose
  metadata includes both ratio-4 and ratio-128 compressor layers. An AProjQ4
  oracle cannot exercise that compound and is a failed coverage gate even if
  greedy output remains correct.
- Then run isolated, same-machine greedy A/B pairs with identical prompt,
  context, cache, token limit, and `DS4_DSPARK_STATS=1`. Change only the gate
  named by the row:

  | Metal fusion | Reference control | Candidate |
  | --- | --- | --- |
  | HC RMSNorm + F16 mixer on M1-M4 | `DS4_METAL_DISABLE_PRE_M5_HC_NORM_MIX_FUSE=1` | Leave the disable switch unset |
  | HC RMSNorm + F16 mixer on another Apple generation | Leave both HC norm/mix switches unset | `DS4_METAL_ENABLE_HC_NORM_MIX_FUSE=1` |
  | HC producer + split/Sinkhorn/destination RMSNorm on M1-M5 | `DS4_METAL_DISABLE_HC_PRODUCER_PRE_NORM_FUSE=1` | Leave the disable switch unset |
  | Q4 Q-A/KV + compressor store in exact-union | `DS4_METAL_DSPARK_EXACTN_UNION=1` with the Q4 enable switch unset | Keep exact-union `=1`; set `DS4_METAL_ENABLE_Q4_QKV_COMPRESSOR_FUSE=1` |
  | Q4 Q-A/KV + compressor store in ordinary `FULL` decode | Leave `DS4_METAL_ENABLE_Q4_QKV_COMPRESSOR_FUSE` unset | Set `DS4_METAL_ENABLE_Q4_QKV_COMPRESSOR_FUSE=1` |
  | Q8 Q-A/KV + compressor store in SSD `FULL` (AProjQ8) | Leave the Q8 enable/require switches unset | Set `DS4_METAL_ENABLE_Q8_QKV_COMPRESSOR_FUSE=1 DS4_METAL_REQUIRE_Q8_QKV_COMPRESSOR_FUSE=1` |
  | Q8 Q-A/KV + compressor store in SSD exact-union (AProjQ8) | `DS4_METAL_DSPARK_EXACTN_UNION=1` with the Q8 enable/require switches unset | Keep exact-union `=1`; set `DS4_METAL_ENABLE_Q8_QKV_COMPRESSOR_FUSE=1 DS4_METAL_REQUIRE_Q8_QKV_COMPRESSOR_FUSE=1` |
  | Q4 attention-output tiny batch in the generic verifier | Set `DS4_METAL_DSPARK_EXACTN_UNION=0 DS4_METAL_DSPARK_EXACTN=0 DS4_METAL_DSPARK_EXACT2=0`; leave tiny enable/require unset | Keep all three exact gates `=0`; set `DS4_METAL_REQUIRE_Q4_ATTN_OUT_TINY_BATCH=1` and require at least one proposed block of depth 3–5 (the acceptance-only suffix evaluates one fewer row) |
  | F16 attention+indexer quad compressor store in `FULL` decode | `DS4_METAL_DISABLE_COMPRESSOR_QUAD_STORE=1` | Leave the disable switch unset |
  | F16 attention+indexer quad compressor store in exact-union | `DS4_METAL_DSPARK_EXACTN_UNION=1 DS4_METAL_DISABLE_COMPRESSOR_QUAD_STORE=1` | Keep exact-union `=1`; leave the quad disable switch unset |
  | Exact ratio-4 one-row compressor pool on M1-M5 | `DS4_METAL_DISABLE_COMPRESSOR_EXACT_POOL_RATIO4=1` | Leave the disable switch unset |
  | Q4 attention-output B + HC expansion | `DS4_METAL_DISABLE_Q4_ATTN_OUT_HC_FUSE=1` | Leave the disable switch unset |
  | FlashAttention pad/block PSO memo | `DS4_METAL_DISABLE_PRE_M5_FLASH_ATTN_PAD_BLK_MEMO=1` | Leave the disable switch unset |
  | FlashAttention batched/vector PSO memo | `DS4_METAL_DISABLE_PRE_M5_FLASH_ATTN_BATCHED_MEMO=1` | Leave the disable switch unset |
  | Exact-union asynchronous routed tails | `DS4_METAL_DSPARK_EXACTN_UNION=1` with `DS4_METAL_DSPARK_EXACT_ROWS_ASYNC_TAILS` unset | Keep exact-union `=1`; set `DS4_METAL_DSPARK_EXACT_ROWS_ASYNC_TAILS=1` |

  The Q4 Q-A/KV compound is opt-in in both exact-union and ordinary `FULL`
  decode; it is enabled only when the explicit enable variable is present.
  Require byte-identical stdout and `errors=0`; for exact-union also
  require `exactn_union_attempt>0` and `exactn_union_error_fallback=0`.
  Partial-accept fallback is expected when the draft diverges. Record
  `exactn_union_full`, `exactn_union_partial_fallback`, `propose`, `verify`,
  `replay`, stage timings, page faults, and generation t/s. A candidate that
  is correct but slower remains disabled or opt-in according to its gate.
  For asynchronous tails, also repeat the model-backed oracle with the switch
  set and run enough exact-union cycles to cross cache eviction and raw-ring
  wrap boundaries. The candidate removes a CPU wait but retains private expert
  buffers until command-buffer completion; serialized state and process memory
  after synchronization must match the synchronous control.
  Before the model-backed runs, build `ds4_test` and run
  `./ds4_test --metal-kernels`. This covers the isolated compound HC, F16 quad
  compressor-store, exact ratio-4 pool, and tie-heavy Metal routing kernels.
  For the exact one-row pool candidate, repeat once with
  `DS4_METAL_REQUIRE_COMPRESSOR_EXACT_POOL_RATIO4=1`; the run must exercise the
  specialization instead of silently falling back. Also exercise the global
  kill switches plus the matching pre-M5 or M5 HC/pool rollback on the target
  machine. Treat the FlashAttention memo rows as host-dispatch A/B tests: the
  selected specialization and output must remain identical, and any timing
  comparison must use repeated warm runs.
- For the default M1 IQ2 address-table mid-only path, first run
  `make test-metal-iq2-midonly`. It must cover 12,288 full-shape top-6 mid
  words in both unmasked and complementary masked address-table modes with
  both mid mismatch counters at zero, no canonical unwritten rows, zero
  candidate gate/up writes, and zero guard mismatches. Then use the same greedy
  IQ2_XXS/Q2_K SSD-streaming model, prompt, cache state, and token count for
  three decode runs: leave all switches unset for the automatic candidate,
  set `DS4_METAL_REQUIRE_M1_IQ2_MID_ONLY=1` for fail-closed coverage, and set
  `DS4_METAL_DISABLE_M1_IQ2_MID_ONLY=1` for the canonical control and
  kill-switch fallback.
  Enable the routed-MoE stage profiler on one candidate layer and require path
  `iq2_stream_addr_mid_only_4096x2048` or
  `iq2_stream_addr_mask_mid_only_4096x2048`; absence of both is failed model
  coverage. Require byte-identical greedy output and top-logprobs, and report
  prefill separately from decode: this one-token routed producer is not a
  prefill optimization. Compare repeated hot-cache medians, then repeat a
  cold-cache sanity run to exclude a change in SSD cache behavior.
- For the removed Metal 512-column streaming top-k path, there is no runtime
  candidate gate. Compare the current binary with a build immediately before
  its removal only if historical timing is needed. First require
  `./ds4_test --metal-kernels` to pass, including tie-heavy routing cases, then
  require identical selected expert ids and greedy output. Correct deterministic
  ordering takes precedence over a timing difference.
- For the default CPU unrolled argmax, run `tests/test_sampling`, then compare
  an otherwise identical greedy workload with
  `DS4_CPU_DISABLE_UNROLLED_ARGMAX=1` (scalar control) and with the variable
  unset (candidate). Require identical tokens for ordinary, excluded-id,
  cross-lane-tie, and vocabulary-tail cases; record median generation t/s over
  repeated runs without claiming a speedup from the implementation alone.
- For the experimental resident-CUDA exact-2 verifier, use three controlled
  runs with verifier cap two on the same single-GPU host: native proposer plus
  legacy verifier
  (`DS4_CUDA_DSPARK_EXACT2=0 DS4_CUDA_DSPARK_PROPOSER_BLOCK_MAX=0 DS4_DSPARK_SSD_VERIFY_BLOCK_MAX=2`),
  two-row proposer plus legacy verifier
  (`DS4_CUDA_DSPARK_EXACT2=0 DS4_CUDA_DSPARK_PROPOSER_BLOCK_MAX=2 DS4_DSPARK_SSD_VERIFY_BLOCK_MAX=2`),
  and two-row proposer plus exact-2
  (`DS4_CUDA_DSPARK_EXACT2=1 DS4_CUDA_DSPARK_PROPOSER_BLOCK_MAX=2 DS4_DSPARK_SSD_VERIFY_BLOCK_MAX=2`).
  This separates the non-causal proposer-width change from the verifier and
  replay change. Then compare uncapped legacy DSpark against exact-2 as an
  end-to-end policy test. Keep SSD streaming and TP disabled. Require
  byte-identical stdout, `errors=0`, and `verifier_unavailable=0` from every
  run; set `DS4_DSPARK_FIXTURE_REQUIRE_EXACT2=1` on the exact-2 run so the
  fixture enforces `exact2_attempt>0` and `exact2_fallback=0`.
  Record `prop_scheduled_rows/cycles`, `propose`, `verify`, `replay`, `net_saved`,
  `miss_first`, `no_draft`, `avg_accept`, and generation t/s from every run.
- For resident CUDA exact-N, keep exact-2 disabled and compare
  `DS4_CUDA_DSPARK_EXACTN=0` against `=1` with the native five-row proposer
  and `DS4_DSPARK_SSD_VERIFY_BLOCK_MAX=5`. Repeat N=2,3,4,5 with explicit
  proposer/verifier caps, then exercise the kill switch with both
  `DS4_CUDA_DSPARK_EXACTN=1` and
  `DS4_CUDA_DISABLE_DSPARK_EXACTN=1`. Require byte-identical greedy stdout,
  `errors=0`, `verifier_unavailable=0`, `cuda_exactn_attempt>0`, and at least
  one `cuda_exactn_full`; partial cases must increment the partial and
  aggregate fallback counters, never the error counter, and continue
  identically through legacy replay. Include
  EOS as the first and a middle draft, a raw-ring wrap boundary, a context
  capacity cut, and prefill workspaces below five rows. Record
  `cuda_exactn_rows`, its full/partial/error counters, `snapshot`, `verify`,
  `replay`, acceptance, and generation t/s. Run with CUDA decode graphs both
  enabled and disabled. Do not promote the gate without a CUDA device build
  and serialized KV/compressor-state oracle; host syntax tests do not execute
  this path. On candidate fixture runs set
  `DS4_DSPARK_FIXTURE_REQUIRE_CUDA_EXACTN=1`; it requires aggregate
  `cuda_exactn_attempt>0` and `cuda_exactn_error_fallback=0`. It reports but
  does not reject aggregate `cuda_exactn_fallback`, because valid partial
  matches increment both the partial and aggregate fallback counters before
  legacy replay.
- For CUDA DSpark non-causal proposer attention, compare the reference with
  `DS4_CUDA_ENABLE_DSPARK_NONCAUSAL_ONLINE=0` against the candidate with `=1`.
  Repeat at proposal depths two and five, across every raw-ring start index,
  and once with both the enable variable and
  `DS4_CUDA_DISABLE_DSPARK_NONCAUSAL_ONLINE=1` to prove the kill switch restores
  the reference dispatch. On the short diagnostic runs also set
  `DS4_DSPARK_VERIFY_NONCAUSAL=1`; record all three reported `max_abs` and
  `max_rel` comparisons and reject non-finite values or a material error
  regression. Then run the acceptance fixture without the diagnostic host
  readbacks and require byte-identical target stdout, `errors=0`, and
  `verifier_unavailable=0`. Record proposal time, acceptance, generation t/s,
  and the startup dispatch log. Draft logits or acceptance may differ slightly
  because online softmax changes the floating-point reduction order; that is
  not permission for the verified target continuation to differ.
- For CUDA HC and tiny routed-MoE kernel changes, keep
  `DS4_CUDA_DSPARK_EXACT2` unset and repeat the resident acceptance fixture
  with these explicit A/B pairs: HC control
  `DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED=1` versus candidate with that variable
  absent; routed-MoE control `DS4_CUDA_DSPARK_TINY_ALIGNED_VEC=0` versus
  candidate `=1`. Require byte-identical stdout, `errors=0`, and
  `verifier_unavailable=0`; record `prop_chain`, `verify_layer`, total
  proposal/verify time, acceptance, and generation t/s. Also run
  `--decode-consistency 64` and the logprob-vector regression before enabling
  a numerically different kernel by default.
- For the CUDA AProjQ4 ports, run an isolated A/B for each dispatch:
  Q-A/KV pair control `DS4_CUDA_DISABLE_Q4_DENSE_PAIR=1` versus candidate with
  that variable absent; HC norm/mix control
  `DS4_CUDA_DISABLE_HC_NORM_MIX_FUSE=1` versus candidate
  `DS4_CUDA_ENABLE_HC_NORM_MIX_FUSE=1 DS4_CUDA_NO_F16_CUBLAS_ONE=1`; and Q4
  attention-output/HC control `DS4_CUDA_DISABLE_Q4_ATTN_OUT_HC_FUSE=1` versus
  the graph-compatible canonical candidate with that variable absent. Run a separate
  non-captured diagnostic with `DS4_CUDA_Q4_ATTN_OUT_HC_ORACLE=1`; require
  the summary to be present with `calls>0`, `skips=0`, and
  `epilogue_mismatches=0`, while `q8k_mismatches` records the expected
  numerical distance from the optional one-dispatch Q8_K experiment. A
  zero-call summary is a failed coverage gate. Only
  test `DS4_CUDA_Q4_ATTN_OUT_HC_Q8K_EXPERIMENT=1` as a promotion candidate if
  its oracle mismatches are also zero. Repeat the pair and attention-output
  cases with `DS4_CUDA_MMQ=0` to exercise the canonical Q8_K fallback
  separately from the default MMVQ/Q8_1 path. Require byte-identical stdout
  and full-logit/tensor equivalence before promoting an opt-in gate. Run with
  decode graphs both enabled and disabled, and record target, proposer,
  verifier, replay, acceptance, and generation t/s. A CUDA build and hardware
  run are mandatory; a host-only build does not compile the device kernels.
- When DSpark, support-model mapping, or SSD streaming changes, repeat both
  the acceptance fixture and verifier invariant on every advertised graph
  backend. Apply the backend and SSD options to the target-only baseline as
  well as the DSpark run:

  ```sh
  DS4_DSPARK_MODEL=/path/to/flash-0731.gguf \
  DS4_DSPARK_SUPPORT=/path/to/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  DS4_DSPARK_FIXTURE_BACKEND=cuda \
  DS4_DSPARK_FIXTURE_SSD_STREAMING=1 \
  DS4_DSPARK_FIXTURE_SSD_STREAMING_CACHE_EXPERTS=32 \
  DS4_DSPARK_FIXTURE_CONFIDENCE=0 \
  make dspark-acceptance

  DS4_TEST_MODEL=/path/to/flash-0731.gguf \
  DS4_DSPARK_SUPPORT=/path/to/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  DS4_TEST_SSD_STREAMING=1 \
  DS4_TEST_SSD_STREAMING_CACHE_EXPERTS=32 \
  make dspark-verify-depth
  ```

  On Strix Halo use the same variables with
  `DS4_DSPARK_FIXTURE_BACKEND=rocm make rocm-dspark-acceptance` and
  `make rocm-dspark-verify-depth`. Do not use the generic targets after a ROCm
  build: on non-Apple hosts their default object set is CUDA. For the 0731
  Flash layout, ROCm needs at least 30 expert slots; use 32 in release tests.
- The fixture must report aggregate `proposed>0`, `accepted_draft>0`,
  `verifier_unavailable=0`, and `errors=0`; stdout must remain byte-identical
  to the target-only SSD baseline. The verifier smoke must report
  `max_chunk>1`, `nspec>64`, and `worst_argmax_gap<=2`.
- Preserve baseline and DSpark `generation` t/s from the same fixture run,
  with the same host, model, cache, runtime settings, thermal state, and
  background load. A DSpark path that is materially slower than the target-only
  SSD path without a documented correctness tradeoff is a release blocker.
- On ROCm, also run one `DS4_DSPARK_PROBE=1`
  generation and require the non-causal attention and stage-chain probes to
  pass. This covers the HIP draft-attention kernel before the end-to-end
  verifier gate.
- If shared support-model or verifier structures changed, also run legacy MTP:
  `make mtp-verify-depth` with `DS4_TEST_MTP` set to a one-stage MTP support
  GGUF, or confirm the target skips only because the optional file is missing.
- Record `c_add` `accepted_draft`, `direct_full`, `direct_partial`,
  `replay_fallbacks`, `errors=0`, `verify_layer`, `net_saved`, and
  `output_match` for both 32-token and 64-token runs. At least one direct commit
  must occur. A faster run with lower proposal quality is a regression unless
  it was an intentional confidence-policy change.
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
  The complete Q2 fixture on the compact Metal graph scored NLL `0.458177271`,
  first-token agreement `90/100`, and average greedy prefix `7.390`, matching
  the accepted release band. Fresh Z.AI FP8 long-context controls also favored
  compact attention: weighted NLL was `0.539823254` versus `0.820997888` for
  expanded K/V over 24 synthetic cases, and `0.808160860` versus `0.820309487`
  over 12 natural source-context cases. These historical measurements used
  the old 4K cutoff; they are not evidence that it matches the reference graph.
- At 100K on an M5 Max, require the compact Q2 plan to remain near 94.09 GiB:
  89.87 GiB model, 1.11 GiB compact history, and 3.11 GiB fixed graph buffers.
  An 8,192-token one-shot control on the same graph reached 479.09 prefill and
  29.89 steady decode t/s. Repeat the continuation fixture after changing the
  compact cache type, absorbed projections, FlashAttention staging, or the
  crossover; numerical similarity to the old expanded graph is not the gate.
- On this 128 GB M3 Max, run the resident Q2 through the generic non-NAX Metal
  path. Repeat the 2,048-2,052 and 4,096-4,100 boundaries, official-continuation,
  MTP, snapshot, server-session, and continued-prefill gates used on M5. Record the different
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
- Run the two- and four-session GLM 5.3 server oracle below token 2052 on one
  M5 and physical TP. For both native single-M5 and TP paths, set
  `DS4_TEST_LOGIT_TOLERANCE=0.001`: row-batched reductions may differ from the
  serial launch order, but every selected token must match and the maximum
  full-logit delta must remain below that bound. Also run the serial rollback
  with zero tolerance. The current four- and eight-session Q2 maxima are both
  `0.0001297`; the current six-step two-session RDMA maximum is `0.000175238`. Past
  2051, require the exact ordered fallback until a sparse native batch oracle
  proves full-vocabulary correctness.
- Exercise every pool remainder at the dense-to-sparse boundary with prompts
  ending at tokens 2048 through 2056, then repeat 4096 through 4100 for the
  prefill-work boundary. The sparse cases must retain the same greedy token,
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
- On GB10, run `make test-cuda-q8-quantize CUDA_ARCH=sm_121` and
  `make bench-cuda-q8-prefill-quantize CUDA_ARCH=sm_121` before accepting the
  default-on tiled prefill activation producer. Require bitwise parity of
  bytes/scales for large batches, guard/tail safety and graph replay. Then
  compare model prefill/decode with `DS4_CUDA_DISABLE_Q8_PREFILL_TILED=1`
  against the unset default; keep the older global warp rollback unset in
  both arms. Follow [the three-way measurement protocol](speed-bench/cuda_q8_prefill_tiled.md).
  CPU-only tests and a kernel-only speedup do not establish a Q4/Q8 TPS gain.
- On a single GB10 (`sm_121`), validate the imported Q2 decode fast paths with
  the AProjQ8/OutQ8 Flash GGUF.  Compare the default against a rollback process
  that sets all of:
  `DS4_CUDA_NO_DIRECT_Q2_PREFILL=1`,
  `DS4_CUDA_NO_F16_PAIR_COMPRESSOR_STORE=1`,
  `DS4_CUDA_NO_F16_PAIR_COMPRESSOR_TRANSPOSE=1`,
  `DS4_CUDA_NO_F16_PAIR_COMPRESSOR_TRANSPOSE_PREFETCH8=1`,
  `DS4_CUDA_NO_Q8_FUSED_ALIGNED=1`,
  `DS4_CUDA_NO_Q8_ALIGNED_PERSISTENT=1`,
  `DS4_CUDA_NO_Q8_ALIGNED_DENSE_SCRATCH=1`, and
  `DS4_CUDA_NO_HC_SPLIT_NORM_SPLIT4096=1`.  Use separate processes, require
  byte-identical greedy stdout and per-token logprobs, then run the same pair
  under Compute Sanitizer.  Record prefill, decode, and steady decode rather
  than copying the upstream PR numbers into a release claim.
- Repeat the GB10 comparison with AProjQ4/OutQ8.  First run
  `make test-mmq-parity-cuda CUDA_ARCH=sm_121`; its Q4 cases must report zero
  bit mismatches for persistent scratch, grouped attention-A, and the
  opt-in K1024 persistent kernel.  For the model A/B, use
  `DS4_CUDA_NO_Q4_GB10_FAST=1` in the control and leave it unset in the
  candidate. Run a separate generic-verifier candidate with
  `DS4_CUDA_DISABLE_DSPARK_EXACTN=1`,
  `DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_BATCH=1`,
  `DS4_CUDA_REQUIRE_Q4_GROUPED_ATTN_A_BATCH=1`, and
  `DS4_CUDA_Q4_GROUPED_ATTN_A_ORACLE=1`; require `batch_candidates>0`,
  `batch_calls>0`, `batch_mismatches=0`, and `batch_skips=0`, plus
  byte-identical stdout. Build the resident prefill harness and run
  `./speed-bench/cuda_q4_prefill_bench --path mmq --case outa --tokens
  127,128,129,257,512,2048,4096 --samples 16 --warmup 4`; require bitwise
  equality between `pack8_mmq_unpack` and `grouped_8_grids`, finite/canary/CPU
  oracle success, and record the paired median. Run
  `make test-mmq-q4-grouped-q81-cuda CUDA_ARCH=sm_121`, then isolate the new
  Q8_1 front-end with
  `--grouped-q81-kernel --tokens 512,1024,2048,4096,6144,8192`; require
  byte-identical direct parity, bitwise final output, and a repeatable paired
  median win between `grouped_generic_q81` and
  `grouped_k4096_g8x2_q81`. Then compare full-model
  prefills with the default environment against the dominant
  `DS4_CUDA_NO_Q4_GROUPED_ATTN_A_PREFILL=1` rollback, and separately against
  the narrow `DS4_CUDA_NO_Q4_GROUPED_ATTN_A_Q81=1` rollback. Keep the
  single-grid and 16-warp experiments unset in this promotion comparison.
  Benchmark the K1024
  persistent kernel as a
  separate fail-closed arm with both
  `DS4_CUDA_ENABLE_Q4_K1024_PERSISTENT=1` and
  `DS4_CUDA_REQUIRE_Q4_K1024_PERSISTENT=1`; its rollback is
  `DS4_CUDA_NO_Q4_K1024_PERSISTENT=1`. Then run a non-captured oracle process
  with `DS4_CUDA_DECODE_GRAPHS=0`,
  `DS4_CUDA_Q4_K1024_PERSISTENT_ORACLE=1`, and
  `DS4_CUDA_Q4_K1024_PERSISTENT_STATS=1`; require `candidates>0`, `uses>0`,
  `oracle_calls>0`, `oracle_mismatches=0`, and `oracle_skips=0`. The parity
  test must also show a nonzero REQUIRE failure with the local kill set,
  proving admission fails before enqueue, and a canonical reference forced by
  that same kill. The counters are host dispatches and intentionally exclude
  CUDA graph replays. The persistent OutQ8
  vocabulary, compressor, HC split, direct routed-MoE paths, Q4 scratch,
  grouped attention-A, and canonical B+HC epilogue remain relevant, while
  the Q8-only attention-projection consumers are intentionally ineligible.
- Validate the experimental HC-to-consumer Q8_1 producer fold in separate
  processes. Use `DS4_CUDA_NO_Q8_FOLD=1` for the control and
  `DS4_CUDA_ENABLE_Q8_FOLD=1` for the candidate, first with the normal graph
  setting and then with `DS4_CUDA_DECODE_GRAPHS=0`. For the non-captured arm,
  also set `DS4_CUDA_Q8_FOLD_ORACLE=1` and require `hits>0`, `byte_calls>0`,
  `output_calls>0`, `byte_mismatches=0`, `output_mismatches=0`, and `skips=0`.
  The reached consumer must be reported as aligned Q8 or IQ2 MoE rather than
  inferred from producer counters alone. Require byte-identical greedy stdout
  and per-token logprobs, then repeat the control/candidate pair under Compute
  Sanitizer. Keep the oracle off for the graph-on timing arm: capture is an
  intentional fail-closed miss and is checked for safety, not fold coverage.
  Run these arms through the ordinary serialized inference dispatcher; the
  opt-in fold does not support concurrent host-thread submission to one CUDA
  stream.
- If the umbrella AProjQ4 A/B changes logits, do not attribute that change to
  "the Q4 fast path" as a unit. Run the fail-closed component matrix from a
  clean `cuda-spark` build. The output directory is intentionally explicit so
  the six independent-process arms, two oracle arms, raw logs, and diffs are
  retained:

  ```sh
  make clean && make cuda-spark
  make gguf-tools/quality-testing/score_official CUDA_ARCH=sm_121

  DS4_CUDA_Q4_MATRIX_SSD_STREAMING=1 \
  DS4_CUDA_Q4_MATRIX_SSD_CACHE=16GB \
  DS4_CUDA_Q4_MATRIX_DECODE_GRAPHS=1 \
  tests/cuda_q4_gb10_fast_matrix.sh \
    /path/to/DeepSeek-V4-Flash-AProjQ4-OutQ8.gguf \
    gguf-tools/quality-testing/data/flash/manifest.tsv \
    /tmp/q4-gb10-graphs-on

  DS4_CUDA_Q4_MATRIX_SSD_STREAMING=1 \
  DS4_CUDA_Q4_MATRIX_SSD_CACHE=16GB \
  DS4_CUDA_Q4_MATRIX_DECODE_GRAPHS=0 \
  tests/cuda_q4_gb10_fast_matrix.sh \
    /path/to/DeepSeek-V4-Flash-AProjQ4-OutQ8.gguf \
    gguf-tools/quality-testing/data/flash/manifest.tsv \
    /tmp/q4-gb10-graphs-off
  ```

  The matrix first proves that the three local rollback switches reproduce
  `DS4_CUDA_NO_Q4_GB10_FAST=1`; failure of `local_control` means the matrix is
  incomplete and no component claim is valid. It then enables exactly one of
  persistent Q8_1 scratch, grouped attention-A, or the graph-compatible B+HC
  call, with K1024 persistent kept disabled because it is a separate opt-in.
  The grouped and HC oracle summaries must have `calls>0`, `skips=0`, and zero
  relevant mismatches. `summary.txt` must say `promotion_gate=pass`. When it
  is blocked, use the named `*_differences` arms and their `.diff` or
  `.comparison.txt` files to identify a component; if no single arm differs
  but `default_fast` does, report an interaction rather than blaming an
  individual kernel. The tensor oracles and synthetic parity test are the
  bit-exact component gates; the top-128 smoke dump and every scorer TSV row
  are complementary end-to-end drift detectors, not a full-logit proof.
- Exercise CUDA DSpark at verifier/proposer depth 5 with the fast paths enabled
  and disabled.  Require identical final output, zero verifier errors, and
  matching full/partial acceptance histograms.  Test both the generic batch
  verifier (direct Q2 path) and CUDA exact-N (one-row decode paths); do not
  infer speculative speedup from the target-only benchmark.
- After aligned Q8 scratch changes, run `make test-cuda-q8-scratch
  CUDA_ARCH=sm_121`, also under Compute Sanitizer. Dense and paired outputs
  must be exact with reused scratch, an undersized buffer, and captured graph
  replays after input changes. Include full-model prefill and decode logits;
  a short token comparison cannot detect stale scratch contents.
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
- When ROCm Q4 code changes, run `make test-strix-rocm-q4-parity` and
  `make test-strix-rocm-q4-prefill` before attempting a model. The prefill
  oracle must compare the TILE8 default with
  `DS4_ROCM_DISABLE_Q4_PREFILL_TILE8=1` at K=256, 1024, and 4096 and at token
  counts covering a partial tile and the 128-token production chunk. Require
  bitwise dense, pair, Q4-attention-B, and Q8-attention-B parity with intact
  canaries. A REQUIRE-plus-DISABLE arm must fail before modifying output.
- For the Q4 F32 `attn_q_b` epilogue candidate, run
  `make test-rocm-q4-qb-epilogue ROCM_ARCH=gfx1151 DS4_TEST_REQUIRE_ROCM_DEVICE=1`.
  Require bitwise finite-output parity and intact guards against the forced
  legacy kernel, including signed zero, YaRN/inverse RoPE, token boundaries,
  and quality/SSD/generic-API exclusions. NaN payloads are not compared, but
  non-finite classification must agree. The host-only mapping/tree test is
  not HIP/GPU validation. Run the native benchmark and the fixed-4096-chunk
  model default/rollback A/B described in
  `speed-bench/rocm_q4_qb_f32_epilogue.md` before claiming any throughput win.
- Keep ROCm grouped attention-A decode opt-in until a model A/B wins. Its
  fail-closed test uses `DS4_ROCM_ENABLE_Q4_GROUPED_ATTN_A=1`,
  `DS4_ROCM_REQUIRE_Q4_GROUPED_ATTN_A=1`, and
  `DS4_ROCM_Q4_GROUPED_ATTN_A_STATS=1`; require dispatches and groups above
  zero, with zero fallbacks/failures and bitwise equality to the per-group
  reference. This synthetic coverage does not supersede the Q4-model OOM
  warning above.
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
  CUDA/ROCm retain four prefix snapshots and replay just the fifth token.
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
  valid, and DSpark stats to show zero verifier errors and no unexplained
  replay fallbacks (see the bounded-prefix exception in section 9).
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

September 5 ROCm attention review, GLM 5.3 Flash Q2, Promessi Sposi, 8,192
allocated context and 32 generated tokens (one matched before/after run):

| Workload | Previous build | Corrected default | Corrected scalar control |
| --- | ---: | ---: | ---: |
| Initial 4,096-token prefill | 80.65 t/s | 76.15 t/s | 33.81 t/s |
| 2,048-token append after 4,096 | 21.45 t/s | 67.28 t/s | 20.92 t/s |
| Decode at 6,144 | 11.70 t/s | 11.68 t/s | 11.69 t/s |

The previous build incorrectly used dense attention through 4,096 tokens.
The corrected graph switches after 2,051, following the pool selection rule;
its 5.6% initial-prefill cost must not be mistaken for timing noise. Older GLM
measurements above this boundary used the old graph. Continued prefill improves
about 3.2x over corrected scalar attention; decode is unchanged. Repeat these
measurements as three-run medians for release sign-off.

The complete stored GLM fixture is unchanged on ROCm: NLL `0.462125091`,
first-token matches `90/100`, mean greedy prefix `7.320`. CUDA scored
`0.461783551`, `90/100`, and `7.490`. These short prompts do not test the
corrected boundary. Independent attention tests pass on Metal, CUDA and ROCm;
whole-model warm/cold continued prefill passes on CUDA and ROCm. CUDA also
passes snapshot/MTP restore and reuse with a 2,771-token prompt. Fresh
long-context scoring now uses eight September 5 Z.AI FP8 references, covering
roughly 3K-7.3K prompt tokens and 740 answer tokens. Use the official low-effort
template and returned reasoning as the prefix; the provider requires reasoning.
Do not compare these answers against the no-thinking template.

| GLM Q2 long-reference path | NLL | First-token matches | Mean greedy prefix |
| --- | ---: | ---: | ---: |
| ROCm corrected default | 0.661721610 | 6/8 | 2.500 |
| ROCm old dense-cutoff control | 0.661579639 | 6/8 | 2.500 |
| ROCm corrected scalar control | 0.674282960 | 6/8 | 2.500 |
| CUDA corrected default | 0.680093110 | 6/8 | 2.500 |

The old-cutoff control retains the other correctness fixes. Its 0.021% NLL
difference is neutral (four case wins each). The corrected fast path is 1.86%
better than scalar on this set (six case wins versus two). This is focused
regression evidence, not a broad quality claim or API-logit parity: the API
returned no logprobs and may normalize reasoning whitespace. All runs completed
with at least 11.3 GiB usable RAM. CUDA's complete long-fixture TSV is identical
before and after the Q8 scratch change below.

The matched Flash 0731 Q2 ROCm fixture also has unchanged scores in all 100
cases: NLL `0.398181736`, first-token matches `56/100`, mean prefix `5.170`.
One paired timing run measured initial 2K prefill at `180.73/181.40` t/s,
a 2K append at `207.67/207.69` t/s, and 4K decode at `14.69/14.69` t/s.
Those timings predate the final memory-policy fixes. The subsequent DSpark
startup exhausted pinned memory: Linux's apparent availability included about
13.2 GiB of CMA pages that these allocations cannot use. After the reboot,
allocation checks exclude CMA, optional Q8 expansion is deferred until required
buffers exist, and tightly sized model arenas remove 13.38 GiB of wasted GLM
capacity. Do not reuse the earlier unadjusted memory minima as safety evidence.

The final build passed both complete 100-case fixtures with the same scores,
GLM snapshot/MTP/reuse at 2,771 prompt tokens, three identical finite vision
encodes, and strict two-session full-logit checks for GLM and Flash. All five
DSpark cases completed in greedy, opportunistic and exact-sampling modes with
zero verifier errors. Greedy outputs matched ordinary decode. One sampled
five-of-six commit used the documented one-token prefix extension. DSpark
kept at least 7.97 GiB usable; this is not a claim that it beats ordinary
decode on every prompt. GLM 5.2 SSD smokes passed with automatic sizing, a
500 GiB hint reduced to fit, and the one-expert per-layer fallback. There was
no new OOM, GPU reset or reboot during these final tests.

Final GLM timing was 77.33 t/s initial 4K prefill, 68.22 t/s for the 2K append,
and 11.74 t/s decode at 6K. Flash's initial 2K prefill was 154.81 t/s, about
15% below the earlier build; its append and 4K decode remained 207.18 and
14.68 t/s. Deferring optional cache construction moves work to the first
request. A separate same-engine test of three fresh sessions measured
168.63/207.41/206.90 t/s with byte-identical full logits, after 18.35 seconds
of engine startup. Record this cold-request cost; warm results alone do not
establish unchanged startup-to-first-response latency. These focused runs do
not replace the three-run release medians.

September 5 DGX Spark, fully resident Flash 0731 Q2, no speculative decoding:
three interleaved medians per build after warmup, 128 teacher-forced decode
tokens, Promessi Sposi, 9,216 allocated context. Reusing aligned Q8 scratch
removes per-call pool allocations without changing the computation.

| Context | Previous decode | Q8 scratch reuse | Gain | Prefill before / after |
| --- | ---: | ---: | ---: | ---: |
| 2,048 | 17.90 t/s | 19.25 t/s | 7.5% | 823.06 / 823.49 t/s |
| 4,096 | 15.23 t/s | 16.22 t/s | 6.5% | 898.69 / 899.89 t/s |
| 8,192 | 15.01 t/s | 15.97 t/s | 6.4% | 931.53 / 931.17 t/s |

Full prefill and post-decode logits matched exactly at all three frontiers.
All 100 Flash-0731 continuation scores were also identical: NLL `0.404714573`,
first-token matches `55/100`, mean prefix `4.890`. The Q8 scratch test covers
24 shapes, missing/undersized scratch, and graph replay with changing inputs;
Compute Sanitizer memcheck reported zero errors. This is a single-GPU result,
not a claim about multi-GPU speed or speculative acceptance.
The final CUDA and ROCm builds were warning-free; ROCm attention, KDA and memory
admission tests passed after removing the unused Q8 matvec kernels.

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
Require exact logits and no decode regression. On M5 Max, Flash Vision Exp
Q2 measured 734.40 versus 711.89 t/s at 2K; mixed Q2/Q4 measured 615.01
versus 604.62 t/s for a 4K append to 12K. Decode was unchanged. These are
interleaved sustained measurements, not cold-run peaks. Packing is an M5
resident-prefill optimization; it does not claim pre-M5 or streaming gains.

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

September 4 M5 measurements, 256 generated tokens, 4096 allocated context:

| Model and prompt | Previous DSpark | Default DSpark |
| --- | ---: | ---: |
| 0731 Q2, C hash table, temperature 0 | 55.71 t/s | 62.95 t/s |
| 0731 Q2, C hash table, temperature 1 | 50.95 t/s | 61.58 t/s |
| Vision Exp mixed Q2/Q4, C hash table, temperature 0 | 41.65 t/s | 48.86 t/s |
| Vision Exp mixed Q2/Q4, C hash table, temperature 1 | 43.56 t/s | 47.24 t/s |

These are three-run medians, alternating schedules with the matching drafter,
`--nothink --top-p 0.95 --min-p 0.05 --seed 12345`. The prompt is:
"Write a complete C hash table implementation with string keys, insert, find,
delete, and a test main. Output only C code."

For two-M5 TP, test the MXFP4 and mixed Q2/Q4 Vision Exp models with their
matching drafter. Use the same sampling settings and 256-token C prompt above,
plus the prompt "Write an unpredictable surreal scene with constantly changing
imagery and no repeated phrases." September 5 measurements (two repetitions,
reversed order on the second; both hosts on RDMA, 50/50 residency):

| Model and prompt | Previous TP DSpark | Default TP DSpark |
| --- | ---: | ---: |
| MXFP4, C hash table, temperature 0 | 41.09 t/s | 50.76 t/s |
| MXFP4, C hash table, temperature 1 | 41.47 t/s | 48.15 t/s |
| MXFP4, prose, temperature 1 | 38.62 t/s | 45.99 t/s |
| Mixed Q2/Q4, C hash table, temperature 0 | 40.60 t/s | 49.52 t/s |
| Mixed Q2/Q4, C hash table, temperature 1 | 41.74 t/s | 46.36 t/s |
| Mixed Q2/Q4, prose, temperature 1 | 37.64 t/s | 44.16 t/s |

Plain MXFP4 TP measured 53.19/50.93 t/s on that C prompt at temperature 0/1.
These DSpark gains are relative to previous DSpark, not a claim that speculation
always beats ordinary decode or that two hosts deliver twice the speed.

Also compare ordinary MXFP4 TP with and without
`DS4_METAL_DISABLE_ROUTED_MPP_PACKED=1`, using Promessi Sposi at 2K/4K/8K/16K
live frontiers. Repeat in reversed order and compare full frontier logits.
September 5 adjacent controls suggested roughly 3-5% faster prefill, but both
paths slowed during sustained runs; 16K results were variable. Five of six
runs were byte-identical at every frontier. One late control differed from
another control at 8K/16K (maximum 0.163/0.167, unchanged argmax), so investigate
reproducibility before attributing a difference to a new kernel.
The final adjacent old/new 2K/4K/8K runs measured 813.53/683.68/707.97 versus
888.45/703.16/691.96 prefill t/s; steady decode was 52.39/47.31/46.12 versus
52.33/47.38/46.29 t/s, with byte-identical frontier logits. This confirms
unchanged ordinary decode, not a uniform prefill gain at every context.

With DSpark, the continued-context 2K/4K/8K check measured 44.03/41.88/41.76
generation t/s versus 40.22/37.47/36.20 before these changes. All three prompt
frontiers matched the ordinary baseline byte-for-byte. The first 20 cases of
the checkpoint-matched Vision Exp continuation scorer were also byte-identical:
1280 target tokens, average NLL 0.179928298, first-token matches 19/20,
average common prefix 17 tokens. These are focused comparisons, not a full
100-case quality pass.

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
The test uses the existing verifier oracle's 2.0-logit near-argmax bound; the
September 5 MXFP4 maxima were 0 and 0.0984. Also run the acceptance checks with
`--mtp-exact-sampling` and with forced low-confidence proposals. Both ranks must
exit cleanly, with no verifier errors or unexplained replay fallbacks.
Exact sampling under TP still intentionally replays partially rejected blocks;
those replays are expected, and seed batching must remain disabled in this mode.
One September 5 two-session run failed at the initial RDMA big-gate barrier.
Both ranks exited cleanly; instrumented and ordinary retries, plus four- and
six-session runs, passed. The cause was not established. Keep transport failures
in the record even when reruns pass; do not hide them by increasing timeouts.

Repeat the physical TP session oracle with two, four and six sessions. Do not
set `DS4_TEST_SKIP_MIXED`: ordinary decode and the mixed continued-prefill step
must both match full serial logits exactly. A regression once left the final
TP residual update deferred in mixed mode, producing a maximum logit difference
of 23.7409 despite choosing the same next token. Testing only argmax misses it.
After flushing that update before the output head, September 5 MXFP4 TP tests
passed exactly at all three session counts; mixed Q2/Q4 passed at six sessions
both in TP and on a single host.

Do not discard slow runs as noise: separate-process tests sometimes varied
by about 20%, including ordinary decoding. Vision Exp prose showed slower
outliers; matched GPU-monitored reruns measured about 41.1 versus 38.4 t/s,
and alternating within one loaded engine measured 40.4 versus 36.7 t/s
(medians after warm-up).
Investigate fresh-process variability separately from kernel cost. Ordinary
decoding still wins on this low-acceptance prompt.

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
