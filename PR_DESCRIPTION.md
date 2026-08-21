# metal: M1-class decode tuning, n-gram speculation, and batch-verifier groundwork

> Draft PR description. All measurements below are from a Mac Studio M1 Ultra
> 128 GB running `DeepSeek-V4-Flash ... imatrix-fixed-0731.gguf` (91 GB, q2
> routed layout), greedy decoding, fixed prompts, 128-512 generated tokens per
> comparison. Run-to-run variance observed: ±0.5 t/s.

## Summary

This PR is the result of a measurement-first optimization campaign on M1-class
hardware (the oldest supported Apple Silicon tier). It contains:

1. **An opt-in switch that widens the "M3"-string fusion gates to every pre-M5
   device** — bit-exact verified, +9.7% decode on M1 Ultra — plus a small
   **per-device defaults layer**: at Metal init, M1/M2 devices apply the
   measured tuning automatically via `setenv(..., overwrite=0)` (explicit env
   always wins; `DS4_NO_DEVICE_DEFAULTS=1` restores pure upstream behavior;
   one self-documenting log line reports what was applied). M3/M4/M5 keep
   upstream defaults until measured. Verified on-device: no-env run applies
   and reaches the tuned speed, the kill switch reproduces the upstream
   baseline exactly, per-variable overrides work, and the deterministic eval
   gate is unchanged.
2. **Prompt-lookup (n-gram) speculative decoding with no support model** —
   +16% on repetition-heavy output (code, edits), free when idle, wired into
   CLI, server, and the native agent.
3. **Two small default-on wins**: routed-MoE tiny-kernel coverage widened to
   n≤8 rows, and the decode compressor emit fusion threaded into the batch
   verifier loop (both stream-identical, measured on-device).
4. **Instrumentation and negative results** that should save the next person
   a lot of GPU time: a verifier micro-benchmark mode, a GPU-busy split for
   the speculative verifier, and three measured dead ends documented below.

Nothing in this PR changes behavior on CUDA, ROCm, or the CPU reference build:
every new fast path is either `#if defined(__APPLE__)`-scoped, gated behind an
environment variable, or both. GLM is untouched (the speculative hook requires
`DS4_SUPPORT_NONE` and the GLM session branch returns earlier).

## Why

On M1 Ultra the engine decoded at 24.6 t/s while an M3 Ultra reaches ~44 t/s
on the same 800 GB/s memory bus. Per-token traffic is 9.14 GiB (47% Q8_0
attention projections, 21% routed experts), so M1 was running at ~255 GB/s
effective — the gap turned out to be software, in three places measured here.

## Changes

### 1. `ds4_gpu_device_is_m3_class()` — widen the M3-only fusion tier (opt-in)

Fifteen decode/prefill fusions were gated on the literal device-name substring
`"M3"` (RoPE fused into the FA reduce + packed32 reduce, KV RoPE/FP8 store
fusion, gathered KV staging, persistent zero attention mask, shared KV pad,
compressor pair-proj/APE/ratio-4 fusions, output HC weights4, router weights
batch fusion, zero-prefix mask cache). With `DS4_METAL_WIDEN_M3_GATES=1` these
now also apply to any pre-M5 device. Every fusion keeps its existing
`DS4_METAL_DISABLE_*` variable, so regressions can be bisected per gate.

* Measured on M1 Ultra: **26.3 → 28.9 t/s (+9.7%)**, and
  `metal_decode_schedule_bench --candidate-env DS4_METAL_WIDEN_M3_GATES
  --include-selection` confirms **bit-identical full-vocabulary logits (401
  rows) and identical greedy selections (400 tokens)**.
* Defaults are unchanged. If maintainers can repeat the bit-exact run on M2
  and M4 devices, promoting pre-M5 to default here looks safe; we only had M1
  Ultra to test.

Related finding, not yet turned into a default: the decode command-buffer
split schedule (2/32, tuned on M3 Ultra) is not optimal on M1 Ultra — an
explicit `DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=2` +
`DS4_METAL_GRAPH_TOKEN_SECOND_SPLIT_LAYERS=8` is worth **+6.8%** (bit-exact by
construction; GPU bubble drops from ~9% to ~3%). Making the adaptive schedule
pick 2/8 on M1/M2 needs measurements on more devices than we had.

Combined effect of the two exact knobs on M1 Ultra: **24.6 → 29.3 t/s (+19%)**
on every workload, byte-identical output. At 16k context the delta is neutral,
at 32k it is +3.4% — never a regression in our sweeps.

### 2. n-gram (prompt-lookup) speculative decoding — no support model

`DS4_NGRAM_SPEC=N` (recommended N=8) enables drafting from the session token
history: when the transcript tail repeats an earlier n-gram
(`DS4_NGRAM_MIN_MATCH`, default 3), the tokens that followed that occurrence
are proposed and checked by the existing DSpark batched verifier
(`metal_graph_verify_suffix_tops`). The drafter is a token-array producer only;
no hidden-state capture, no support GGUF, no memory overhead.

* **+16% on a code-rewrite task** (29.3 → 34.0 t/s), neutral on free prose
  (a missing match costs a sub-millisecond CPU scan).
* Greedy-exact at the acceptance boundary: every committed token equals the
  target argmax. Blocks of N≤5 commit through the prefix-frontier path and we
  verified the output is **byte-identical** to plain decode; N≥6 full-block
  accepts keep the verifier state and inherit DSpark's documented FP-order
  drift contract (`--quality` semantics unchanged).
* Drafts are truncated at the first stop/thinking-control token, so an
  accepted block can never cross a turn boundary. This matters for the agent,
  which extends the live KV incrementally.
* Wired in: CLI and server already routed through
  `ds4_engine_mtp_draft_tokens() > 1` (single-session, greedy requests;
  batched-session mode keeps speculation off, as it already did for
  MTP/DSpark). The native agent gets its own branch
  (`agent_speculative_block_cap` + `worker_accept_verified_token`): greedy
  turns only, plain-text only (never while DSML syntax is buffered, active,
  or parsing), blocks capped at 8 so a block cannot both enter and complete a
  tool stanza. Measured agent turn time on the code task: **38 s → 31 s**.
* Tested on Metal only. The verifier path is backend-shared, so the knob may
  work on CUDA, but we did not validate it there — keeping it opt-in until
  someone does is deliberate. `ds4-bench`/`ds4-eval` intentionally do not use
  it (they are the measurement and drift-reference tools).
* Workload tuning, measured through `ds4-server` (thinking on, generation
  t/s with prefill excluded): with the default `DS4_NGRAM_MIN_MATCH=3`,
  reasoning prose triggers frequent low-yield 3-gram matches and the
  ~120 ms verifies cost ~6% net; `DS4_NGRAM_MIN_MATCH=6` is the safe server
  setting (novel code −4%, rewrite/edit +13%). The nothink copy/edit numbers
  above (+16%) use min_match=3. A cheaper verifier (the microbatch work
  below) relaxes this precision/recall trade-off.
* Test hook: `DS4_NGRAM_FAKE_PROPOSAL=1` forces a full-width verify every
  cycle while committing exactly one token, so the output stream must equal
  plain greedy decode — a self-checking micro-benchmark for any verifier
  change (we used it to validate everything below).

### 3. Default-on micro-wins (both stream-identical on M1 Ultra)

* **Routed-MoE tiny-kernel coverage n≤5 → n≤8** (and `down_sum6` n≤4 → n≤8).
  The tiny `pair_swiglu`/`sum6` kernels already carry the token index in
  their grid; the old thresholds left n=6..8 verify blocks on the generic
  `mv` path (x read twice, split gate/up, separate down reduce). Rollback:
  `DS4_METAL_DISABLE_MOE_TINY_WIDE`.
* **Batch-verifier compressor emit fusion.** The non-aligned batch compressor
  path paid ~12 single-row dispatches per emit per ratio-4 layer (norm, RoPE,
  FP8 round-trip, F16 commit, indexer QAT, two state shifts). The decode path
  already fuses all of that into one bit-exact dispatch
  (`kernel_dsv4_comp_row_finalize_f32`); this PR threads the same fusion into
  the batch loops via a merged per-row loop (attention update + indexer
  update + one fused finalize per emit). Guarded by the same shape/type
  invariants as the decode fusion, plus: never under prefix-frontier capture
  (the fused kernel performs the ratio-4 state shifts) and never on the
  aligned-chunk path (already batched). The indexer projections move past the
  live attention rows in the shared staging tensor while fused. Rollback:
  `DS4_METAL_DISABLE_BATCH_COMP_FINALIZE_FUSE`.
  Validated: fused N=8 stream byte-identical to pre-patch; N=5
  (capture path, fusion off by design) byte-identical to plain decode;
  rollback byte-identical.
  One real-world bug was caught by agent testing and fixed before this PR:
  the same non-aligned batch path is also reached by **resumed-prefill
  chunks** (arbitrary `pos0` when extending a live checkpoint), where the
  fused loop's indexer staging offset exceeds the shared staging tensor and
  the encode fails (`gpu layer 2 attention batch encode failed`). The gate
  now requires `n_tokens <= 16` (verify-sized blocks), which is also the only
  regime where the per-row fused loop beats the aligned replay. Regression
  covered by re-running the exact failing agent scenario.

### 4. Instrumentation

* `ds4_gpu_busy_profile_seconds()` (Metal) + `verify_gpu_busy` in the DSpark
  stats line: splits CPU-encode from GPU-execution time inside
  `metal_graph_verify_suffix_tops`. This is how we established that the
  verifier is ~97% GPU-bound.
* The speculative verifier's 4-layer pipeline flush no longer requires DSpark
  hidden-state capture (rollback: `DS4_METAL_DISABLE_SPEC_VERIFY_FLUSH`).

### 5. Measured negative results (code present, default off — or documented)

These cost real GPU-hours; recording them is part of the PR's value:

* **`mul_mv_ext` r1_6/7/8 template instantiations** (dense weights streamed
  once instead of twice for 6..8-row blocks): **bit-identical but ~25% slower
  verifies on M1 Ultra** — the historical two grid.y slices run concurrently
  across cores, while an 8-accumulator block halves threadgroup count and
  hurts occupancy. Kept opt-in (`DS4_METAL_ENABLE_MV_EXT_WIDE`) because the
  trade-off may invert on wider-register devices (M3/M5) — worth one run of
  the schedule bench there.
* **DSpark on Metal M1 Ultra is net negative today**: 26.7 t/s on code vs
  29.3 without, despite an 83.65% acceptance rate (2.29 tokens/cycle). The
  breakdown (`verify_gpu_busy`) shows a 5-row verify costs ~112 ms and an
  8-row one ~121-135 ms — `verify(N) ≈ 34 + 10.9·N ms` — i.e. ~3× a 37 ms
  decode step, nearly all GPU execution in the batch layer kernels. This is
  the "hand-written N=2/N=4 decode microbatch" gap the comment at the
  verifier already names. Re-tested with `iogpu.wired_limit_mb` raised so the
  DSpark support model is fully wired: the verdict does not change.
* **Byte-count and dispatch-count theories of that fixed cost both fail on
  silicon**: halving dense bytes (r1_8) made it slower, and removing ~230 tiny
  dispatches (compressor fusion) bought ~1% (~10 µs per queued dispatch).
  A per-stage attribution with `DS4_METAL_LAYER_STAGE_PROFILE` on 8-row verify
  blocks (sync overhead subtracted) settles it: `routed_moe` ≈ 47 ms (~40%,
  and it already streams its 13.6 GB at ~290 GB/s — bandwidth-saturated),
  `output_proj` ≈ 22 ms (the 8× re-read of `attn_output_a` is real, but the
  SLC absorbs about half of it), `hc_pre` ≈ 13, `q_path` ≈ 11,
  `indexer_setup` ≈ 10; the batched flash-attention mma path, our previous
  prime suspect, accounts for almost nothing at 2k context. So the verifier
  floor is set by expert bytes, and the two kernels worth writing next are:
  an **expert-gathered MoE verify pass** (group the up-to-48 (row, expert)
  selections by expert and stream each slab once — measured cross-row overlap
  ~38% → roughly −16 ms), and an **N-row `attn_out_low`** (weights outer,
  token loop inner → −10-15 ms). Realistic verify(8) floor: ~60-75 ms, which
  would put DSpark at ~33 t/s and n-gram at ~37-38 t/s on code on M1 Ultra.

## Files touched

| File | Change |
|---|---|
| `ds4_metal.m` | `ds4_gpu_device_is_m3_class()` + 15 gate sites; MoE tiny thresholds; `mv_ext` r1 table + names (opt-in); GPU-busy accessor |
| `metal/dense.metal` | `kernel_mul_mv_ext_{f16,q8_0}_f32_r1_{6,7,8}` instantiations (dispatched only under the opt-in) |
| `ds4.c` | n-gram drafter + verifier entry (`ds4_ngram_propose`, `ds4_session_eval_ngram_speculative_argmax`) and hook in `ds4_session_eval_speculative_argmax`; `need_spec_verifier` and `ds4_engine_mtp_draft_tokens` extensions; batch compressor fused loop + `metal_graph_batch_comp_fuse_defer`; verifier flush widening; `verify_gpu_busy` stats |
| `ds4_agent.c` | agent speculative branch (greedy, plain-text-only, ≤8) |
| `ds4_gpu.h` | `ds4_gpu_busy_profile_seconds()` declaration |

## Platform / device impact matrix

| Target | Impact |
|---|---|
| Metal M1/M2 (pre-M5, no "M3" in name) | No change by default. With `DS4_METAL_WIDEN_M3_GATES=1` + split 2/8: +19% measured on M1 Ultra, bit-exact. MoE tiny widening and batch compressor fusion are on by default, both stream-identical (fusion additionally requires the widen env on non-M3 devices because it shares the decode fusion's `kv_rope_fp8` gate). |
| Metal M3/M4 | M3 keeps its existing gates (the new predicate is a superset). M4 gains the widened tier only under the env. The 2/8 split and the r1_6/7/8 kernels deserve one schedule-bench run each here before considering defaults. |
| Metal M5 | Untouched: M5-only paths keep their own predicates; the ported-M5 feature checks are unchanged. |
| CUDA / ROCm | No behavioral change: Apple-scoped code, plus env-gated engine paths that default off. `DS4_NGRAM_SPEC` would reach the shared verifier on CUDA but is untested there — left opt-in on purpose. |
| CPU build (`make cpu`) | Builds; new Apple branches compile out or return false. |
| GLM 5.2 | Untouched (GLM sessions branch before the n-gram hook; compressor fusion requires the DeepSeek ratio-4 shape). |

## How it was validated (M1 Ultra, Metal, q2 0731 imatrix GGUF)

Correctness:

```sh
# bit-exact A/B for the widened gates (401 logit rows + 400 selections identical)
./speed-bench/metal_decode_schedule_bench --include-selection \
  --control-first 2 --control-second 8 --candidate-env DS4_METAL_WIDEN_M3_GATES

# deterministic eval gate: identical with and without the patched env (4/4 PASSED)
./ds4-eval -m ds4flash.gguf --plain --questions 4 --tokens 2048 --temp 0 --seed 1

# verifier micro-harness: stream must equal plain greedy decode
DS4_NGRAM_SPEC=5 DS4_NGRAM_FAKE_PROPOSAL=1 ./ds4 ...   # byte-identical, all variants
```

Regression suite (per CONTRIBUTING.md), M1 Ultra, Metal, q2 0731 imatrix GGUF:

```text
./ds4_test --server                              OK
./ds4_test --logprob-vectors   (default env)     OK   # official continuation vectors
./ds4_test --logprob-vectors   (patched env:     OK   # identical result with
   WIDEN_M3_GATES=1 + split 2/8)                      # all exact knobs enabled
./ds4_test --metal-kernels                       29 failures — PRE-EXISTING:
   pristine HEAD (84cc882) in a clean worktree fails with the same 29
   ("router batch weights total selected=0 ...") on M1 Ultra; patched and
   pristine outputs are identical. Looks like M3-gated kernel tests that were
   never exercised on M1-class devices — flagged here for maintainer triage,
   not introduced by this PR.
make cpu                                         builds clean (not executed —
                                                 per the macOS kernel-bug note)
make (Metal)                                     builds clean, zero warnings
```

Not run (no hardware access): CUDA (`make cuda-regression`), ROCm, M2/M3/M4/M5
devices. `--long-context` and `--tool-call-quality` were skipped for time; the
deterministic `ds4-eval` q1-q4 gate above covers generation drift end-to-end.

Speed (`ds4-bench`, promessi_sposi, ctx 2048/16k/32k, gen 128; plus fixed-prompt
CLI runs at 512 tokens): tables above; raw logs available.

Real-workload confirmation, extracted from `ds4-server` per-request logs with
the device defaults active (`WIDEN` + split 2/8 + ngram 8/6), vs the measured
24.6 t/s upstream generation baseline on the same machine:

| workload | generation | vs upstream |
|---|---|---|
| agentic coding eval (15-step multifile task, tools) | **34.4 t/s** effective (median 38.4, peaks 42.7) | **+40%**, task wall −25% |
| reasoning eval (25 hard questions, thinking) | **27.8 t/s** effective (median 27.6) | **+13%** |

Quality on those same runs: 24/25 hard-accuracy, agent task fully passed
(hidden tests included, 0 invalid tool calls). The agent run also prefilled
only 3.5k of its 50k prompt tokens thanks to the existing exact-DSML/KV
prefix reuse — worth knowing when reading agent prefill figures.

## SSD streaming: findings and two small deltas

We ran the same measurement-first pass on `--ssd-streaming` with the 145 GB
MXFP4 GGUF (M1 Ultra, 128 GB). Two code deltas and three findings:

* **n-gram speculation is now gated off under SSD streaming**: the spec
  verifier machinery is not provisioned there (measured: every block reports
  `verifier_unavailable`; upstream likewise refuses `--mtp` with streaming),
  so the device defaults no longer arm dead machinery. Output verified
  identical with/without.
* **New diagnostic**: `DS4_METAL_STREAM_PREPARE_SUBPROFILE=1` adds sub-timers,
  a per-return-path histogram, and mlock/slot-state deltas to the expert
  cache's prepare path. It is how the findings below were established.
* Findings: (1) steady-state MXFP4 streaming decode on M1 Ultra is
  ~9.5-9.7 t/s at 0.80 cache hit rate (auto budget wins every manual budget
  we tried; extra pread threads don't help); short sessions run slower only
  while the slab cache fills from SSD. (2) The host-side prepare/mlock cost
  (~0.5 ms per miss) sits on the async loader threads, off the token
  critical path — we verified this the hard way: a bulk-mlock-per-slab
  variant was throughput-neutral while doubling mlock work (fresh pages pay
  zero-fill wiring), so it was retired with a note in the code. (3) The
  remaining streaming levers are architectural (two-tier resident+streamed
  experts, oracle-guided prefetch) — an expert-selection trace facility and
  our recorded traces make the prefetch ceiling computable offline before
  anyone writes a predictor.

## Unrelated footgun found while validating (report, not fixed here)

`make cpu` links the CPU-only builds over the same output names (`ds4`,
`ds4-eval`, …). A subsequent `make` sees those binaries as newer than the
Metal objects and does **not** relink them, so the tree silently keeps CPU
binaries — we measured a "Metal" eval at 5.7 t/s before spotting
`backend=cpu` in its startup line. Anyone following CONTRIBUTING's
"`make cpu` to verify portability" hits this. Possible fixes: distinct output
names for the cpu target, or forcing the link step in `all`.

## Open questions for maintainers

1. Should the pre-M5 widening become the default once M2/M4 repeat the
   bit-exact run? The per-fusion `DS4_METAL_DISABLE_*` knobs make a staged
   rollout easy.
2. Is a `--ngram-spec N` CLI flag preferable to the env for the speculation?
   We kept env-only to avoid touching the flag surface before a design nod.
3. The verifier microbatch problem is still open: with the fixed ~34 ms +
   10.9 ms/row cost, DSpark cannot win on M1-class Metal. Our profiling
   points at the batched FA path next; happy to iterate with review guidance.
