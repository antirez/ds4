# next-step.md — Qwen3.8-27B MTP speed work, handed off to Min

Written 2026-08-23 from Max (M5 Max, 128 GB). Everything below was measured on
Max unless stated. Code is already on GitHub; artifacts are rsynced into this
directory (see **Artifacts**).

## Where the code stands

| branch | commit | contents |
|---|---|---|
| `main` | `582a7b7` | exact MLX Q4_64A parity runtime (merged from ornith15) **+** MTP head KV sized by session ctx, BF16 CPU matvec |
| `ornith15` | `5edda49` | the parity runtime alone |

`main` is the branch to work on. Both are pushed to `origin` (audreyt/ds4) and
`git branch -f main origin/main` has already been run here, so `main` is
current. Build with the usual forced build; `.metal` files load from disk at
runtime relative to cwd.

```sh
cd ~/w/ds4 && make -B ds4 ds4-server ds4-bench ds4-eval ds4-agent -j8
```

## The objective that is still open

> do all levers and match mlx.fast speed, and carry what you can learn to ornith

Two of the levers are already implemented and measured (parity runtime, head-KV
sizing). The rest are identified, located in the source, and **not yet written**.
Nothing below is speculative about *where* the cost is — each item has a
measurement attached.

## Baselines you must reproduce before changing anything

Cold runs read ~30 % low. Run ≥3 reps, discard rep 1, take the median.
`accepted/round` is deterministic per config — if it matches across reps and
t/s moves, that is thermal noise, not your change.

### ds4 @ `582a7b7`, greedy, decode-only t/s

| prompt | no draft | `DS4_QWEN_MTP_K=3` | accepted/round |
|---|---|---|---|
| 523 tok, copy-style English | **23.9** | **53.5** (2.23×) | 1.46 |
| 2554 tok, Italian prose | 18.4 | 17.7 (0.96×) | 0.54 |
| 10295 tok | 9.7 | 14.8 (1.52×) | 0.70 |

Prefill, no draft: 226 t/s @523 tok, ~450 @2554, ~510 @10295.
Prefill with the MTP head loaded: **150–170 t/s at every length** — that is
lever 3 below.

Depth sweep at 523 tok (K=4/6/8 were single cold runs — **re-measure warm**):
K=2 48.0, K=3 53.5, K=4 37.7?, K=6 36.8?, K=8 34.8?.

### Pure MLX on the same weights (this is the real bar)

`mlx_lm.generate`, `EigenLabs-Qwen3.8-27B-4bit`, warm:

| prompt | MLX prefill | MLX decode |
|---|---|---|
| 523 tok | 679 t/s | 26.5 t/s |
| 2554 tok | 604 t/s | 29.7 t/s |
| 10295 tok | 610 / 469 t/s | 24.8 / 23.5 t/s |

**MLX decode is flat in context; ds4 collapses.** 9.7 vs 24.2 t/s at 10 k is the
single largest gap on the board. Short-context ds4 is 10 % behind (23.9 vs 26.5).

### mlx.fast (the Swift harness) on the identical 523-token seed, 256 tokens

| depth | seconds/token | t/s | accepted rate | mean draft |
|---|---|---|---|---|
| 0 (serial control) | 0.0712 / 0.0618 | 14.0 / **16.2** | — | 0 |
| 8 (its adaptive policy) | 0.0329 / 0.0300 | 30.4 / **33.3** | 0.843 | **5.28** (max 7, zero non-drafting rounds) |

So: **ds4 already beats mlx.fast's own harness** — 23.9 vs 16.2 serial, 53.5 vs
33.3 speculative — but mlx.fast extracts 5.45 tokens per round against ds4's
3.92, while ds4's round is ~2.2× cheaper. The remaining wins are (a) close the
pure-MLX decode gaps, (b) take mlx.fast's per-round yield with ds4's round cost.

## The levers, in expected-payoff order

### 1. Long-context decode cliff — host attention per token (biggest)

`ds4.c:17536-17561`. When `ds4_gpu_qwen_full_attn_tensor()` returns 0 the layer
falls back to `qwen_full_attn_from_qkv()` on the **host**: `ds4_gpu_end_commands()`,
readback of q/k/v, CPU attention over the whole context, writeback — per
full-attention layer, per token. Qwen3.8 has 16 full-attention layers, so a
single token pays 16 GPU syncs plus 16 CPU attentions.

It declines because the pooled KV is sized by `qwen_metal_pool_effective_max_ctx()`
(`ds4.c:16652-16673`), which returns **8192** unless something called
`qwen_metal_pool_note_requested_ctx()` before pool init. The CLI reports
`ctx=32768 … raw_kv_rows=4352`, so the request is not reaching the pool.

Do this: record the session ctx before the pool initialises (or size the pool
from it directly), exactly the way `582a7b7` now sizes the MTP head's private KV
from `g_qwen_pool.max_ctx` with a fallback ladder (`ds4.c:19224-19277`). Then
verify at 10 k that no layer takes the host path — the cheapest check is that
decode t/s stops collapsing (target ≥ 20 t/s at 10295 tokens).

Watch the memory: 64 layers × 32768 rows × 4 kv-heads × 256 dim × 4 B is far too
much if you allocate for every layer. Only the full-attention layers need rows.

### 2. Adaptive per-round draft depth (port mlx.fast's cost model)

ds4 drafts a fixed `DS4_QWEN_MTP_K` every round (`ds4.c:42003-42017`), which is
why the 2554-token prose case *loses* 4 %: it pays 4 forwards for 1.54 tokens.
mlx.fast ships a marginal-value rule and gets mean draft 5.28 with zero
non-drafting rounds on the easy prompt while collapsing to 0 on hard stretches.

Port it from `/Users/au/w/qwen38-challenge/Sources/MLXFastModel/Qwen36MTPBlockSession.swift`
(clone the repo on Min, it is absent here):

- `costModelDepth(offeredDepth:)` at **1072-1144** — the whole rule.
  `reach = Π p[j]`; extend while `reach > marginal[d] * (1 + expected) / cumulative[d]`;
  `expected += reach`.
- `positionAcceptEMA` init `0.85 * 0.98^d`, α = 0.15 — **837-839**;
  update rule `recordAcceptOutcome` — **1175-1181** (positions below the accept
  count observe a success, the position *at* it observes a failure unless the
  round stopped on a committed stop token, deeper positions observe nothing).
- `headStepCostRatio h = 0.18` — **873**, with the full fitting history at
  841-872. Uniform price: `marginal[d] = h`, `cumulative[d] = 1 + d*h` (**904-911**).
- Caps: `sdpaWidthWallDepthCap = 5` (**1034**), `segmentedVerifyDepthCap = 7`
  (**1041**), hard max 8.
- Top-2 confidence clamp at depths 0 and 1: `p = min(p, sigmoid(margin/2))` and
  `sigmoid(margin/3)` where `margin` is the pending primary's target top-2 logit
  gap (**1123-1131**). ds4 has those logits at verify time.

ds4's own h is measurable and close: verify at width 4 costs 72 ms against 42 ms
at width 1 → h ≈ 0.24. Fit it from a forced-depth sweep, do not inherit 0.18
blindly. Keep `DS4_QWEN_MTP_K` as the *offered* ceiling and let the policy pick
0..K; a returned 0 must cost exactly what plain decode costs.

### 3. MTP prefill priming is sequential per prompt token

`ds4.c:41939-41950` primes the head's KV by calling `qwen_mtp_draft_one_metal()`
once per prompt token. That is why prefill drops to 150–170 t/s whenever a head
is loaded (from 510 t/s at 10 k). Batch the priming the way the target's batched
prefill works — the head is one layer plus a vocab projection, so a batched
forward over the prompt window should cost a small fraction of the target's.

### 4. Short-context serial decode: 23.9 vs MLX 26.5, and prefill 226 vs 679

Not yet root-caused. `DS4_QWEN_PROFILE=1` prints per-command-buffer GPU time but
the print is gated to every 64th buffer, so it is nearly useless as-is; either
ungate it behind a second env var or use `/usr/bin/sample` on the running pid
(it exists; my one attempt raced the process). Suspect host round-trips per
token in the hybrid path and the fixed per-prefill overhead that amortises away
by 2 k tokens.

### 5. Then Ornith

`Ornith-1.5-35B` ships a **bundled nextn draft head** (`./ds4 --inspect` prints
`layers: 40 (+1 bundled nextn draft head, executed separately)`), and the
sidecar/APEX variants are synced. Everything above transfers:

- Levers 1–3 are model-independent code paths (pooled KV sizing, draft policy,
  head priming). Lever 2 especially: implement the policy once, keyed off the
  same profile counters, and both Qwen3.8 and Ornith get it.
- The head/trunk pairing discipline is the trap that already cost a day on
  qwen38 — read the `ds4-apex-mtp-head-pairing` skill before touching the APEX
  MTP artifacts, and never trust a t/s number without reading the generated text.
- Ornith baselines on Max today (523-tok prompt, n=128, no speculation):
  35B **269 t/s prefill / 88.2 t/s decode**; 9B **734 / 72.7**.

## Reproduction commands

```sh
cd ~/w/ds4
A=$PWD/qwen38-q4_64a-full.gguf                    # ds4-native Q4_64A scales
B=$PWD/gguf/Qwen3.8-27B-EigenLabs-Q4_64A.gguf     # MLX bf16 scales
H=$PWD/gguf/Qwen3.8-27B-MTP-bf16.gguf             # norm-corrected MTP head
F=$PWD/qwen38-mtp-fixtures/correctness_prompts/public_longcopy_gate_english_512.txt

# serial baseline
./ds4 -m "$A" --prompt-file "$F" -n 256 --temp 0 --raw

# speculative, fixed depth 3
DS4_QWEN_MTP_HEAD=$H DS4_QWEN_MTP_K=3 DS4_QWEN_MTP_PROFILE=1 \
  ./ds4 -m "$A" --prompt-file "$F" -n 256 --temp 0 --raw

# MLX-exact parity mode (bit-exact logits, ~half the decode, 1-row prefill)
DS4_Q4_64A_MLX_BF16=1 ./ds4 -m "$B" -p Hello --raw --dump-logits /tmp/l.json
# compare against mlx_lm.load('gguf/EigenLabs-Qwen3.8-27B-4bit') -> expect 0/248320 diffs

# long-context legs
./ds4 -m "$A" --prompt-file qwen38-mtp-fixtures/p8k.txt -n 128 --temp 0 --raw
```

**Artifact/mode pairing is a silent-garbage trap.** `$A` runs in the default mode
and emits `)[)[)[…` under `DS4_Q4_64A_MLX_BF16=1`; `$B` is the exact opposite.
Neither errors. `accepted=0 (0.00/round)` with a matched head is the tell. Always
read the text before quoting a number.

### mlx.fast side (needs a clone + build on Min)

```sh
git clone <qwen38-challenge remote> ~/w/qwen38-challenge   # absent on Min
cd ~/w/qwen38-challenge && swift build -c release
export MLXFAST_QWEN_MTP_TARGET_DIR=~/w/ds4/gguf/EigenLabs-Qwen3.8-27B-4bit
# fixtures already synced: ~/w/ds4/qwen38-mtp-fixtures/ab-{plan,reference}.json
./.build/release/mlxfast-swift mtp-timed \
  --weights ~/w/ds4/gguf/EigenLabs-Qwen3.8-27B-4bit \
  --mtp-head ~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head \
  --golden ~/w/ds4/qwen38-mtp-fixtures/ab-reference.json \
  --tokens 256 --mtp-depth 8        # --mtp-depth 0 is the serial control
# read parent_measured_seconds_per_token; t/s = 1/spt
```

Regenerate the golden if you change the seed prompt (needs `tokens + 4` rows):

```sh
./.build/release/mlxfast-swift mtp-verify --weights … --mtp-head … \
  --emitted ab-plan.json --generate 260 --mtp-depth 1 --output ab-reference.json
```

## Gates before any commit

```sh
make -B ds4 ds4-server ds4-bench ds4-eval ds4-agent -j8
./ds4_test --server
./ds4_agent_test; ./ds4-eval --self-test-extractors
./tests/test_layer_pack; ./tests/test_engine_mgpu_placement; ./tests/test_dflash_shape
./tests/test_q64a_quant; ./tests/test_gpu_args; ./tests/test_layer_stash; ./tests/test_gpu_args_cli.sh
make test-metal-session-batch DS4_TEST_MODEL=$PWD/qwen38-q4_64a-full.gguf   # exact_logits=1
DS4_TEST_MODEL=$PWD/ornith35.gguf ./tests/test_ornith15_bench.sh            # exact vs llama-cli
DS4_TEST_MODEL=$PWD/ornith9.gguf  ./tests/test_ornith9_bench.sh             # exact vs llama-completion
```

Serialize the 35 B runs (`sleep 5-10` between model loads); ds4 refuses to start
while another ds4/ds4-server holds the single-instance lock. Bare `./ds4_test`
(no `--server`) reports **20 pre-existing failures** against Ornith-9B — those
fixtures belong to the flash model; prove pre-existence by diffing suite
verdicts, do not chase them.

Also re-verify parity after *any* ds4.c change: `DS4_Q4_64A_MLX_BF16=1` +
`--dump-logits` against MLX must stay 0/248320 diffs.

## Artifacts synced into this directory

| path | size | what |
|---|---|---|
| `qwen38-q4_64a-full.gguf` | 14 G | Qwen3.8-27B, ds4-native Q4_64A scales — the fast path. **Not transferred**: the file already on Min as `Qwen3.8-27B-Q4_K_M.gguf` is byte-identical to it (`sha256 17cc0346cd373b4913c04975f624ac3fc456a5a3fbd08f99be82e7cb4aba439b`), so this name is a hardlink to it. That existing name is a **misnomer** — the file is Q4_64A, not Q4_K_M |
| `gguf/Qwen3.8-27B-EigenLabs-Q4_64A.gguf` | 14 G | same weights, MLX bf16 scales — parity mode only |
| `gguf/Qwen3.8-27B-MTP-bf16.gguf` | 821 M | **the norm-corrected MTP head** (llama.cpp's Qwen `+1` norm transform was wrongly applied to 7 tensors; this file is the fix) |
| `gguf/Qwen3.8-27B-MTP-f16.gguf` | 821 M | earlier F16 head, superseded — loses fidelity through BF16→F16 |
| `gguf/EigenLabs-Qwen3.8-27B-4bit/` | 14 G | MLX reference checkpoint (parity target, tokenizer.json, mlx.fast weights) |
| `~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head/` | 810 M | MLX MTP head checkpoint mlx.fast pins |
| `gguf/Ornith-1.5-9B-Q4_K_M.gguf` | 5.2 G | Ornith 9B target |
| `gguf/ornith1.5-9b-dflash-bf16-projection-Q4_K_M.gguf` | 730 M | Ornith 9B DFlash draft |
| `gguf/Ornith-1.5-35B-Q4_K_M.gguf` | 20 G | Ornith 35B target (bundled nextn head) |
| `gguf/mtp-Ornith-1.5-35B-A3B-NVFP4-frspec-owngen32768.gguf` | 901 M | Ornith 35B MTP sidecar |
| `gguf/Ornith-1.5-35B-A3B-APEX-MTP-Balanced.gguf` | 24 G | APEX mixed-precision + MTP |
| `gguf/Ornith-1.5-35B-A3B-APEX-Balanced.gguf` | 24 G | APEX mixed-precision, gate-ladder model |
| `qwen38-mtp-fixtures/` | small | `ab-plan.json`, `ab-reference.json` (260 mlx.fast golden rows), `p2k.txt` (2554 tok), `p8k.txt` (10295 tok), `correctness_prompts/` |

Symlinks `qwen38.gguf`, `ornith35.gguf`, `ornith9.gguf`, `ds4flash.gguf` are
created in this directory for the harnesses.

**The transfer is complete** (2026-08-23, ~23:05 CST). It ran over the **Thunderbolt 5
link at `audreyt@169.254.101.109` (`en2`)**, not `Min.local` — TB5 moved these
files at 200-390 MB/s against 14-21 MB/s over Wi-Fi/WARP, so use that address for
any future artifact push. Two openrsync notes that cost real time: this macOS
ships **openrsync** (protocol 29), so `--info=progress2` is rejected, and
`--partial` on a multi-GiB file stalls for minutes in the delta phase writing
nothing — use `rsync -aW` (whole-file). The driver is `/tmp/sync-min.sh` **on
Max**, log `/tmp/sync-min.log`; re-running it is idempotent.

Verified after transfer: all nine GGUFs byte-size-identical to Max,
`gguf/EigenLabs-Qwen3.8-27B-4bit/` 25 files / 14798168 KiB, and
`~/.cache/mlxfast/qwen3.8-27b-mtp-v1/mtp-head/` 5 files / 829512 KiB — both
matching Max exactly. Re-check any time with:

```sh
ls -l ~/w/ds4/*.gguf ~/w/ds4/gguf/*.gguf | awk '{print $5, $9}'
du -sh ~/w/ds4/gguf/EigenLabs-Qwen3.8-27B-4bit ~/.cache/mlxfast/qwen3.8-27b-mtp-v1
```

Sizes must match: `qwen38-q4_64a-full.gguf` and
`Qwen3.8-27B-EigenLabs-Q4_64A.gguf` are both exactly **15149087072** bytes and
are **byte-different** — that identical size is the pairing trap, not a copy
error. Verify with `cmp -s` if in doubt; anything still `.<name>.<random>` in the
directory is an in-flight rsync temp file.

## Min run results — 2026-08-24 (this machine)

All levers implemented on `main` (+ uncommitted working tree), gates green,
parity held. Numbers below are Min's; see **Machine caveat** before comparing.

### What the code changes are

1. **Grouped shadow attention kernel** (`metal/qwen_gdn.metal` +
   `ds4_metal.m`): the real long-context fix. The old
   `kernel_qwen_gqa_attn_decode_shadow` gave each of the 64 query heads its own
   threadgroup and made every one of them stream its kv-head's whole F16
   history — a 16× (`ngrp`) DRAM read amplification once the working set
   exceeds L2 (empirically pos ≳ 4 k). That, **not** pool sizing, was the
   10 k cliff: decode command buffers measured 7–8 s of pure GPU time per
   token at pos ~10.3 k while the pooled KV was correctly sized at 32768 rows
   and host-fallback tallies stayed zero. New kernel
   `kernel_qwen_gqa_attn_decode_shadow_grp`: one threadgroup per kv head, 512
   threads = one simdgroup per grouped q-head, per-head scalar op order
   provably identical (same BT=128 blocks, same online softmax). Greedy text
   is byte-identical to the old kernel at both 523 and 10295 tokens.
   `DS4_QWEN_SHADOW_GRP=0` restores the old dispatch; auto-falls back when
   `n_head/n_head_kv != 16`.
2. **Adaptive draft depth** upgraded to mlx.fast semantics:
   `DS4_QWEN_MTP_H` overrides h (default still 0.18), chooser cap flattened to
   7 like `segmentedVerifyDepthCap`, stop-token suppression in
   `qwen_mtp_update_ema` across all six call sites (hybrid, CPU, second Metal
   loop), optimism-transfer branch kept. Top-2 clamp NOT ported (verify passes
   expose top-1 only). `DS4_QWEN_MTP_AUTO` remains opt-in; its output is
   byte-identical to serial at 523.
3. **Batched MTP priming** (`qwen_mtp_priming_batched`, 256-row window):
   replaces the per-prompt-token head forwards. Byte-identical output;
   with-head prefill now equals no-head prefill (~209–217 t/s vs Max's
   sequential 150–170); under thermal load the A/B is 45.6 vs 17.8 t/s
   (2.6×). `DS4_QWEN_MTP_PRIMING_SEQ=1` restores the old loop.
4. **Instrumentation**: `DS4_QWEN_POOL_DEBUG=1` prints
   `req/ceiling -> max_ctx`; `DS4_QWEN_FALLBACK_TALLY=1` counts GDN/full-attn
   host fallbacks per layer via atexit; `DS4_QWEN_PROFILE_ALWAYS=1` prints
   every command buffer's GPU time with n/avg/max aggregates.
5. Early `note_requested_ctx` calls before all pre-generation `ensure_pool`
   warm sites (defensive; sizing was already correct on main).

### Measurements (Min, Apple M5 Max 128 GB, greedy)

| leg | old tree | new tree | note |
|---|---|---|---|
| serial @523 | — | **24.4 / 26.7** t/s (two fresh windows) | matches Max's 23.9 and the MLX bar 26.5 |
| K=3 spec @523 | — | **57.8** t/s, accepted 1.46/round | beats mlx.fast harness's 33.3 |
| AUTO @523 | — | lossless text, 210 acc / 45 rounds | drafted ~5/round |
| serial @10295 | 2.84–3.55 t/s | **9.89 / 9.91** t/s (reproducible) | **3.3× kernel win**, same-window ON/OFF pair 9.89 vs 2.70 |
| K=3 @10295 | 4.25 t/s (soaked) | accepted 0.70/round deterministic | counter cross-checks with Max exactly |
| prefill @10295 no head | — | 545 t/s fresh | Max saw ~510 |
| MLX reference (soaked Min) | 2.4–2.6 t/s @10k | ds4 new kernel 9.91 → **3.8× faster than pure MLX** | mlx_lm 0.31.3 in `~/.venvs/mlxq` |

h measured from the fresh K=3 leg: verify/round 54.4 ms vs serial 41.8 ms →
**h ≈ 0.10** for width 4 on fresh hardware (Max's 72-vs-42 ms fit gave 0.24;
treat 0.18 default as conservative middle; re-fit per machine with
`DS4_QWEN_MTP_H`). Depth sweep K∈{2,4,6,8} came back thermally confounded
(within-config reps perfectly consistent — same accept counts — but legs ran
in different thermal windows); do not quote it.

### Correctness matrix

- Goldens byte-match through every path: serial 523 & 10295, grouped kernel
  ON and OFF, batched priming, AUTO policy, post-rebuild binaries.
- Full gate suite green: unit tests, `exact_logits=1` Metal session batch,
  tiered parity (0 diffs > 1e-6, max 5e-10 — the strict-zero gate now needs
  mlx_lm 0.31.3, which itself shifted last-ulp; use `~/.venvs/mlxq/bin/python`),
  Ornith 35B and 9B exact-match regressions.
- Spec-vs-serial text at 10295 diverges at near-tie tokens ("po'" vs "poco")
  with BOTH kernels — pre-existing property of batched verify at long ctx, not
  this change; both outputs are coherent continuations. At 523 spec is exactly
  lossless (AUTO and fixed-K3 both match golden).

### Machine caveat (cost us half a day — read first)

Min soaks in ~60–90 s of sustained GPU load and then runs ANY framework
5–10× slow until minutes of idle: llama.cpp tg128 collapsed to 10.8 t/s on the
9B here (vs ~70 expected), mlx_lm to 2.4 t/s @10k. GPU clocks read 1620 MHz
100% residency throughout; powermode/battery/thermal warnings are clean — it
just loses sustain. Consequences: (a) never compare absolute numbers across
thermal windows; use same-window A/B or deep-cooled single shots (~10 min
idle); (b) the ≥20 t/s @10k target could not be certified on Min — 9.9 t/s is
the thermally-capped plateau and llama.cpp lands in the same zone, so the cap
is the machine, not the kernel; re-measure the absolute on Max (expect ~3×
over its documented 9.7); (c) Max's "short-context gap" (23.9 vs MLX 26.5)
looks like the same artifact — fresh Min hits 26.67.

### Next steps handed back

- Re-baseline absolutes on Max with the new kernel (serial + K=3 + AUTO at
  523/2554/10295; expect ~30 t/s class at 10k serial if sustain holds).
- Split-K along position for the grouped kernel (only 4 threadgroups/token/
  layer today → occupancy-bound beyond L2; flash-decoding style partial
  attention + merge would target the remaining 26.7→9.9 decay). Tune on Max.
- Decide default h (0.18 shipped; fresh-Min fit says 0.10) from a clean
  forced-depth sweep on Max using the profile counters, not wall-clock.
- Optional: port the top-2 confidence clamp once the verify pass exposes
  top-k (protects hard prompts; easy prompts unaffected).

## Ornith speed results — 2026-08-24 (Min, fresh-state discipline)

Follow-up session: applied the Qwen levers to both Ornith models and measured.
Commit `e5ce86d` + the ngrp generalization on top (uncommitted at this note's
time unless later committed).

### Shapes / routing facts

- Ornith-1.5-35B: arch `qwen35moe`, 40 blocks = 39 executable + 1 bundled
  nextn MoE head; heads=16 kv_heads=2 (**GQA group 8**), head_dim 256. Routes
  through the Qwen hybrid path; bundled nextn binds via `qwen_mtp_bind` and
  engages automatically (`DS4_QWEN_NEXTN_DRAFT=0` disables).
- Ornith-1.5-9B: arch `qwen35` dense 32L; heads=16 kv_heads=4 (**group 4**).
  DFlash sidecar pairs via `--dflash` (layers=6 block=16 draft_n<=7).
- The e5ce86d grouped shadow kernel gated on ngrp==16 → both Ornith models
  silently used the old kernel. Generalized to runtime ngrp 1..16
  (`metal/qwen_gdn.metal` + gate/dispatch in `ds4_metal.m`). Verified
  bitwise: grouped-serial ≡ legacy-serial across the full comparable span at
  ngrp=8, and the Qwen3.8 golden still matches byte-for-byte after the
  generalization rebuild.

### Measured (523-tok fixture, greedy, warmed + measured reps)

| leg | t/s | vs serial |
|---|---|---|
| 35B serial | **87.9–90.5 decode / ~1140 prefill** | — (Max doc said 269/88.2 — prefill was soaked there) |
| 35B nextn K=1 | **99.5 / 99.8** | **+13 % — the sweet spot** |
| 35B nextn K=3 | 57.7 | 0.64× — loses |
| 35B AUTO h=0.34 | 86.1 | over-drafted |
| 35B AUTO h=0.7 | 86.6 | collapses to ~no drafting (correct instinct) |
| 35B with-head prefill batched priming @10k | 263–275 vs 98–127 sequential | **2.1–2.8×**, text-identical |
| 9B serial | **75.9–77.7 / ~750** | matches Max 734/72.7 |
| 9B DFlash (warmed, -n256) | 53.1 (acc 220/230 = 95.6 %) | **0.70× — heavy draft can't win** |
| long ctx @10k both models | ON ≡ OFF, ≥18 t/s | no cliff: per-layer KV fits L2 at these shapes |

### Why speculation economics differ from Qwen3.8 here

Ornith35's head is a full MoE layer and every verified row re-routes experts:
marginal verify row ≈ +0.345 serial-token-times and each draft step ≈ 0.34
(h≈0.34 measured from profile counters) — so a round costs ≈ 1+0.69·d serial
times against yield Σp^i with p≈0.86. Break-even sits just past depth 1:
**fixed K=1 is optimal** (+13 %); deeper drafting always loses. The 9B DFlash
draft is target-class expensive (~9 ms/draft-token), so even 96 % acceptance
pays nothing. If you want more than +13 % on the 35B: cheaper verify rows
(batched-MoE-row kernel work) or a lighter draft head — not policy tuning.

### Correctness notes

- `tests/test_ornith15_bench.sh` and `test_ornith9_bench.sh` (exact vs
  llama-cli) pass through the generalized kernel.
- Ornith35 spec-vs-serial text diverges early EVEN UNDER THE OLD KERNEL
  (legacy K1 ≠ legacy serial @char 34): pre-existing MoE batched-verify
  near-tie behavior, not this change; outputs stay coherent. Same class as
  the Qwen3.8 @10k divergence recorded above.
- Watch out when scripting: `run(){ shift; ...$1... }` names output files
  after the first env assignment, not your tag — cost an hour of phantom
  diffs (fixed pattern in /tmp/orn3.sh).
