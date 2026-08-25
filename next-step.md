# next-step.md — Qwen3.8-27B MTP speed work

Rewritten 2026-08-24 on **Max** (M5 Max, 128 GB) after taking the Min handoff
back. Every number below is measured on Max unless stated. The previous
revision of this file is commit `e5ce86d`; read the **Correction** section
before trusting anything you remember from it.

## Where the code stands

| branch | commit | contents |
|---|---|---|
| `ornith15` | `46dd3e7` (nextn head-only guard) atop `5c9980e` (standalone-nextn-sidecar) atop the merge of `d6eb25f` + Min `5ee5427` | split-K shadow decode, ngrp gate fix (1..16), MTP verify-width counters, opt-in top-2 clamp, standalone nextn MTP head sidecars on `--mtp`/`--dflash`, nextn head-only bind guard, Min's Ornith measurements |
| `main` | `ornith15` minus `46dd3e7` and `5c9980e` | everything above except the standalone-nextn-sidecar commit and its guard fix |
| Min `5ee5427` | parallel work, merged in | independent ngrp 1..16 generalisation + full Ornith 35B/9B measurement |
| `e5ce86d` | common ancestor | grouped shadow kernel (dead on every shipped model), adaptive depth, batched MTP priming |

Min's `main` was fetched from `ssh://audreyt@192.168.1.77/Users/audreyt/w/ds4/`
(**not** `Min.local` as `au@`, and not `audreyt@Min.local` — only that exact
user+IP form authenticates from here). Min's `ornith15` branch is stale at
`5edda49`; Min commits to `main`, so fetch `main`.

Build and gates unchanged:

```sh
cd ~/w/ds4 && make -B ds4 ds4-server ds4-bench ds4-eval ds4-agent -j8
```

## The objective, and where it now stands

> do all levers and match mlx.fast speed, and carry what you can learn to ornith

**Serial decode now matches or beats pure MLX at every length measured, and is
flat in context** — which was the single largest gap on the board.

| prompt | ds4 @`e5ce86d` | ds4 @`d6eb25f` | pure MLX |
|---|---|---|---|
| 523 tok | 24.07 | **26.03** | 26.5 |
| 2554 tok | 18.40 | **25.52** | 29.7 |
| 10295 tok | 9.69 | **25.63** | 24.8 / 23.5 |

Medians of 3, greedy, decode-only t/s, `-n 256` (`-n 128` at 10295). The
`e5ce86d` column is the same binary with `DS4_QWEN_SHADOW_SPLITK=0`, measured in
the same thermal window, so it is a true A/B and not a cross-window comparison.

Prefill is unchanged: ~230 t/s @523, ~440 @2554, ~510 @10295, and with the MTP
head loaded ~205 @523 — Min's batched priming (lever 3) does hold up on Max, the
old 150–170 t/s ceiling is gone.

## 2026-08-25 — PR 1354 transfer (Q4 NAX split prefix)

`Layr-Labs/qwen-3.8-mtp-challenge#1354` (ox-alpha): pinned bf16 head wins vs
q2-q4 rerank (~26%); mlx packed GDN S<=2 does **not** transfer (ds4 already
fuses conv+core). Transferable hole: `ds4_gpu_matmul_quant_impl_tensor`
required `n_tok % 32 == 0` for Q4/Q6 NAX, so the 523-token fixture (and any
unaligned last prefill chunk) never hit `n128`. Ported the Q8_0
`split_nax_prefix` (`n_tok >= 192`, aligned prefix on NAX, remainder on tiled
mm). `DS4_METAL_Q4_NAX_SPLIT=0` restores the old gate.

Same binary, 523-tok longcopy, greedy `-n 256`, same window:

| leg | prefill | decode | acc | sha |
|---|---|---|---|---|
| serial | 421.07 | 20.55 | — | `59a33123…af30` |
| K=3 split=0 | 177.29 | 29.98 | 190/197 (1.46) | same |
| K=3 split=1 | **364.44** | 27.48 | 190/197 (1.46) | same |

Text is byte-identical to serial. Prefill is the win; decode t/s is window
noise. `test_metal_session_batch exact_logits=1`; both Ornith exact-match
harnesses green.

Tried and rejected same day: fused Q4_64A gate+up+SwiGLU on the live
`mul_mv_ext` path. Bit-exact vs unfused, but serial 27.4 vs 28.4 t/s and
MTP K=3 verify 58 vs 55 ms — concurrent gate/up already overlapped. Not
shipped. K=6/7 on longcopy is not faster than K=3 (58.6 / 54.5 vs 58.9)
and diverges from serial. Current K=3 is **2.11× serial** (58.9 / 27.9).
mlx.fast pinned-head longcopy is **~3.58×** (0.0106 vs 0.038 s/tok). Gap
is verify cost (~54 ms for w=4), not acceptance (copy is nearly lossless).

w=4 verify skip (soaked window, means only): full 101.6 ms, skip-FFN 22.9,
skip-GDN 44.6. FFN is the verify cost. Live n_tok=4 path is
`kernel_mul_mv_ext_q4_64a_f32_r1_4` (nsg=4), not `kernel_mul_mv_q4_64a_dense_f32`.
Occupancy A/B, all sha `57e2fe23…` / later `ed739af1…` = serial:
tiled mm (EXT_MAX=1) 198 ms verify — lose. Last paired run: nsg8 29 t/s /
113 ms vs nsg4 42 t/s / 77 ms. Default stays nsg=4.
`DS4_METAL_Q4_64A_NSG=8` remains an override. Still ~2.1× serial, not mlx.fast.



mlx.fast S=4 GDN: prefix-replay tape, no per-row SSM dump. ds4 verify writes
`gdn_state_steps` (spec_cap×64×48×128×128 f32). `DS4_QWEN_GDN_SNAP=0` skips
that. ABBA on longcopy K=3, sha `57e2fe23…`: snap-on 61.9/61.6 t/s verify
53.1/53.4 ms; snap-off **64.2/64.3 t/s** verify **50.9/50.9 ms**. Real, small.
Reject path still needs the dump or a replay; default stays snap-on. Not mlx.fast.

mlx.fast S=4 affine QMV (`qwen35_custom_affine4` + xsums) ported as
`DS4_METAL_Q4_64A_XSUMS=1` (`kernel_mul_mv_q4_64a_n4_xsums`, nsg=2, 4 rows/simd,
factored bias). ABBA longcopy K=3, sha `57e2fe23…`, log `n4 xsums QMV active`:
ext 61.23/62.19 t/s verify 53.9/53.1 ms; xsums **54.89/57.77 t/s** verify
**61.6/58.0 ms**. Bit-exact, slower. Default stays `mul_mv_ext` r1_4. Not mlx.fast.

Q4_64A fused gate/up+SwiGLU (`DS4_METAL_Q4_64A_PAIR=1`, ext r1_4 occupancy).
ABBA longcopy K=3, sha `57e2fe23…`, log `pair swiglu active`: baseline
62.15/61.96 t/s verify 53.1/53.2 ms; pair first 59.68 (cold compile) then
**62.99 t/s / 52.0 ms**. Bit-exact, noise-level. Default off. Not mlx.fast.

Q4_64A ext nxpsg A/B (`DS4_METAL_Q4_64A_NXPSG`): 8 (default) 61.93/61.82,
16 61.76 / outlier 65.25, 32 62.07. All sha `57e2fe23…`. Occupancy on K
is not the gap. Default 8.

Qwen3.8 GDN used legacy `kernel_qwen_gdn_core` (rows4 gated to 32-head).
`DS4_QWEN_GDN_ROWS4=1` on 48-head: log `core_rows4 active v_heads=48`, sha
`57e2fe23…`. Decode 62.18/61.97 vs 61.87/61.77 (noise). Prefill 425 vs 409.
skip-GDN live w=4 38ms vs 53ms → GDN block ~15ms, mostly Q4_64A in/out proj
not the SSM core. Default stays legacy core.

mlx.fast cost-model can offer depth 8 (near-flat verify). ds4 K=7 on longcopy:
56.95 t/s, w=8 mean 89 ms, draft 260 ms, repair 13.9 ms, acc 1.54/round, sha
`cc774c20…` (differs from K=3 `57e2fe23…` — extra blank line). K=3 stays
61.8 t/s / w=4 53 ms. Verify is not flat. Default K=3.

ds4 already `requestResidency` + queue add at load. `DS4_METAL_NO_RESIDENCY=1`
ABBA longcopy K=3, sha `57e2fe23…`: on 61.91/61.93 t/s prefill 410/408; off
61.72/61.69 t/s prefill 385/388. Decode not residency-bound.

GPU CB profile (`DS4_QWEN_PROFILE_ALWAYS=1`): serial n_tok=1 **33.4 ms GPU**.
K=3 verify n_tok=4 **~50 ms GPU** (one `command batch`); each MTP draft **3.3 ms
GPU** own CB (readback). 4-wide is 1.5× serial, not 4× — extra 17 ms is width-4
ALU/activation, not dispatch. Wall verify ≈ GPU busy. No commit.

GPU width: n_tok=1 33.4 ms, n_tok=2 36.0 ms (near-flat), n_tok=4 50 ms.
`DS4_METAL_Q4_64A_R1=2 NXPSG=16` (two Y-groups, w=2 occupancy) at n_tok=4:
57.3/57.5 t/s verify 58.5/58.3 ms vs default 61.9/61.6. 2× weight scan loses.
Keep r1=4. No commit.

Same-machine longcopy, mlxfast unloaded first. ds4 n=64 K=3 still **61.58 t/s**
(acc 1.50, w=4 53.8 ms). Cooled ds4 serial n=64 **22.0 t/s** (prefill 349; KV
grows — not the 30.7 n=16 GPU 33.4 ms). mlxfast mtp-timed n=64 K=3 **54.31 t/s**.
Local ds4 MTP > local mlx. Ranked ~94 t/s / 3.58× still unmatched. No commit.














## Correction: the grouped kernel never ran on Qwen3.8

The previous revision claimed a **3.3× long-context win** from
`kernel_qwen_gqa_attn_decode_shadow_grp`. That is wrong, and the way it was
wrong is worth keeping.

The grouped dispatch was gated on `n_head / n_head_kv == 16`. Qwen3.8's
full-attention layers are **24 query heads over 4 kv heads**, so `ngrp` is 6 and
the gate was false on every single decode. The model never executed the new
kernel. Evidence, all on Max:

- `DS4_QWEN_SHADOW_GRP=0` vs default: 9.82 vs 9.79 t/s at 10295, **byte-identical
  generated text**. Toggling a kernel that is genuinely running does not produce
  a character-identical result.
- Serial @10295 was 9.7 t/s at `582a7b7` and 10.0 t/s at `e5ce86d` — the
  "3.3× win" left Max exactly where it started.
- With split-K forced at every nsplit from 2 to 32, the output sha never moved,
  because split-K is gated behind the same `grp` flag.

Min's 2.84 → 9.89 t/s pair was its documented sustain collapse (Min runs any
framework 5–10× slow after ~60–90 s of GPU load), not the kernel. Min's own
Machine-caveat section predicted this failure mode; the same-window A/B was not
enough to escape it because *both* legs of that A/B ran the same kernel.

`DS4_QWEN_SHADOW_DEBUG=1` now prints, once per process, which geometry actually
ran. That one line would have caught this on day one:

```
ds4: qwen shadow decode: path=split nsplit=81 pos=10310 n_head=24 n_head_kv=4 head_dim=256 cap_rows=32768
```

Also corrected: the pool *was* sized properly all along on Max
(`DS4_QWEN_POOL_DEBUG=1` reports `req=32768 ceiling=32768 -> max_ctx=32768`).
The `raw_kv_rows=4352` in the memory line is the planner's GDN split, not the
attention pool, and was never the 10 k cliff. Host fallback tallies are zero.

## What actually fixed it

1. **ngrp gate generalised** to any group in 2..16, with the host dispatching
   `ngrp*32` threads rather than a hardcoded 512 so exactly `ngrp` simdgroups
   are live and the query-staging barrier stays uniform.
2. **Split-K along position.** Grouping by itself is worth nothing here — 9.75
   vs 9.69 t/s — because the decode is **occupancy-bound, not
   DRAM-amplification-bound**. One threadgroup per kv head is 4 threadgroups per
   layer per token on a 40-core GPU. `kernel_qwen_gqa_attn_decode_shadow_grp_split`
   runs one threadgroup per `(kv_head, split)` and
   `kernel_qwen_gqa_attn_decode_shadow_merge` recombines the partials with the
   same online-softmax rescale the single-pass kernel applies inline.

   nsplit sweep at 10295 (medians of 3):

   | nsplit | 2 | 4 | 8 | 16 | 24 | 32 | 41 | 81 |
   |---|---|---|---|---|---|---|---|---|
   | t/s | 14.6 | 18.4 | 21.3 | 22.9 | 23.6 | 24.1 | 24.8 | 25.9 |

   Monotone in threadgroup count, which is the proof that occupancy was the
   binding constraint. The auto rule therefore takes **one BT=128 block per
   split** (`nsplit = blocks_total`, capped 256): 5 splits @523, 20 @2554,
   81 @10295.

Split-K is **on by default**; `DS4_QWEN_SHADOW_SPLITK=0` restores the
single-pass dispatch and `2..256` forces a count for sweeps.

### Why defaulting it on is safe

- Generated text byte-identical to single-pass at 523, 2554 and 10295 for every
  nsplit tested.
- `--dump-logits` at 2554, `DS4_QWEN_SHADOW_SPLITK=0` vs default: **bit-identical**
  (md5 `272d8aa6070c725fe974c882fa5e27af` both ways). The reassociation is exact
  here because chunk boundaries are whole BT blocks, so each chunk replays the
  single-pass block order and the merge only re-applies scale factors the
  single-pass kernel would have applied anyway.
- MLX-exact parity unchanged: **0/248320 diffs, max_abs 0.0**.
- Full gate ladder green, including `exact_logits=1` metal-session-batch and
  both Ornith exact-match harnesses (the ngrp change is live for Ornith too).

## Standalone nextn head sidecars (--mtp/--dflash)

`5c9980e` teaches both draft-model flags to accept a standalone Qwen
nextn MTP head GGUF. `qwen_nextn_sidecar_install()` probes the file by
binding it (`qwen_mtp_bind_from`) and validating the result
(`qwen_mtp_is_valid`); only then does it arm the global
`g_qwen_mtp_sidecar_model`. On `--mtp` the nextn probe sits behind the
legacy-MTP and DSpark detections, so existing artifacts keep their
existing handling. On `--dflash` it runs before `dflash2_bind`, and a
file that fails the probe falls through to dflash2 plus the existing
dense-Qwen family gate exactly as before.

Binding prefers the armed CLI sidecar over the `DS4_QWEN_MTP_HEAD`
environment fallback, and the legacy session verifier stays off on this
path — drafting engages through `qwen_mtp_bind()` in the hybrid generate
paths instead. Sidecar storage is engine-owned, and `ds4_engine_close()`
disarms the pointer before unmapping, so the model cannot outlive its
map.

Acceptance rule: a file binds only if it reads as a Qwen MTP head in
the qwen35moe style — draft tensors under `blk.<n_layer>.nextn.*`,
including a full MoE block. Any non-matching file falls through
untouched.

The probe as originally shipped had a hole: binding a file satisfies
the probe whenever `blk.<N>.nextn.*` tensors are present, and a FULL
model bundles its own nextn head. Handing the 24 GB Ornith-35B APEX
full model to `--mtp` therefore armed it as a "standalone head"; GPU
placement absorbed the exec layers and generation emitted ~180
`Metal F32 tensor matmul range is outside the mapped model` warnings
plus corrupted text instead of the designed rejection.

`46dd3e7` closes this with `qwen_nextn_head_only()`: every
`blk.<k>` tensor must have `k == DS4_N_LAYER`. Post-guard the same
APEX file is rejected verbatim — `carries tensors outside blk.40;
not a standalone head` — falling through to the three-format
unsupported message with exit 1 and no arming. Re-validation after
the fix: the legit NVFP4 bare head still loads via `--mtp`
(`Qwen nextn MTP sidecar loaded`) and drafts (`drafted=23`,
coherent English; `accepted=0` on this artifact is a head-quality
fact, not a machinery failure); the `--dflash` legacy path is
untouched (`proposed=110 accepted=109 @47.36 t/s` banner intact);
the legacy-MTP rejection text is unchanged.

## Speculation economics inverted — read before quoting any MTP number

Split-K made the target forward ~2.6× cheaper at long context. The draft head
did not get cheaper, so **speculative decode is now a net loss beyond short
prompts**. With the head loaded, medians of 3:

| prompt | serial | K=1 | K=3 | AUTO |
|---|---|---|---|---|
| 523 | 25.45 | — | **51.81** (2.04×) | **52.49** (2.06×) |
| 2554 | 26.73 | **32.52** (1.22×) | 19.69 (0.74×) | 26.61 (1.00×) |
| 10295 | 25.67 | — | 16.85 (0.66×) | 21.98 (0.86×) |

At 523 speculation is still a 2× win and is **exactly lossless** — K=3 and AUTO
both reproduce the serial golden sha. At 2554 only depth 1 pays. At 10295
nothing pays. This is not a regression: MTP requires `DS4_QWEN_MTP_HEAD`, so the
default path never speculates.

## Default h stays 0.18

Fitted from the new per-width counters rather than wall clock. At 2554,
`verify(w) = 30.6 + 8.74·w` ms against a 37.4 ms plain step, and full-round cost
relative to plain decode is 1.30 / 1.90 / 2.50 / 4.02 / 5.27 at depth
1 / 2 / 3 / 5 / 7 — a least-squares `1 + d·h` fit gives h ≈ 0.59, but the
*marginal* price at depth 1 is only 0.30 and the average is dragged up by deep
rounds the policy should never choose.

Measured end to end with `DS4_QWEN_MTP_AUTO=1`:

| h | 523 | 2554 | 10295 |
|---|---|---|---|
| **0.18** | 49.56 | **30.89** | 21.37 |
| 0.35 | 51.39 | 26.59 | 20.44 |
| 0.59 | **52.58** | 24.88 | 21.38 |
| 0.90 | 25.89 (never drafts) | 25.08 | 24.07 |

0.18 wins on the mean across 523/2554 and is the shipped default. Min's 0.10 fit
would be strictly worse. `DS4_QWEN_MTP_H` still overrides per machine.

## Open items, in expected-payoff order

1. ~~**The adaptive policy latches off and cannot recover.**~~ **Fixed** — see
   the exploration-probe section below. The latch was real: `qwen_mtp_choose_k`
   returning 0 means `qwen_mtp_update_ema` sees `drafted == 0`, matches none of
   its branches, and the EMA freezes forever. It now probes. What is *not*
   fixed is the h fit itself: with the latch gone, h=0.18 is still the best
   tested value, so the cliff in the h table below was the latch, and the
   underlying price is genuinely ~0.3 at depth 1. Re-fitting h beyond the four
   tested values is still open.
2. **2554 is the only length still short of MLX** (25.5 vs 29.7). 523 and 10295
   are at or past the bar. Worth a `DS4_QWEN_PROFILE_ALWAYS=1` pass: decode is
   one command buffer per token, so the per-buffer aggregate localises it
   directly. At 2554 that buffer was 50.1 ms before split-K against 100.0 ms at
   10295, i.e. ~34 ms fixed + ~6.4 µs per context token; the fixed term is now
   the dominant cost at every length and nothing has profiled it yet.
3. **Ornith long-context split-K.** Short-context Ornith is already measured
   and wins (see below: 9B +11.3 %, 35B +22.4 % at 523 tokens, byte-identical).
   What is still unmeasured is Ornith at 10 k *with* split-K. Min's "no cliff
   at 10 k, grouped ON ≡ OFF" was about the grouped kernel, not split-K, so it
   does not settle this — and given split-K pays at 523 where there is no
   cliff at all, it plausibly pays at 10 k too.
4. **Top-2 confidence clamp** is implemented and off by default
   (`DS4_QWEN_MTP_TOP2=1`). It uses a real k=2 indexer reduction plus two 4-byte
   reads per position — no full-logits copy. Measured neutral at 523 (+1.8 %,
   identical text) and negative at 2554 (24.59 vs 26.61), so it stays opt-in
   until item 1 is fixed; the clamp only matters when the policy is actually
   free to choose depth.

## Exploration probe — the EMA latch fix

The adaptive policy was an absorbing Markov chain. Once `qwen_mtp_choose_k`
returned 0, every subsequent round had `drafted == 0`; `qwen_mtp_update_ema`
matches none of its branches in that case, so the EMA froze at whatever value
stopped the drafting and no amount of newly-predictable text could restart it.

A zero-draft round genuinely carries no evidence, so the only honest escape is
to spend a draft occasionally and re-observe. `qwen_mtp_choose_k` now returns
depth 1 after `interval` consecutive evidence-free rounds, and both it and
`qwen_mtp_update_ema` carry a `qwen_mtp_probe { quiet, interval, probing }`.

**The interval backs off, and a probe backs off even when it is accepted.**
That second part is the non-obvious one. Resetting to the eager interval on any
accepted draft looks right and is wrong: at 10 k context probes land ~0.3
tokens/round and speculation is *still* a net loss there, so "accepted" kept
re-arming the probe and cost 26 %. Accepted is not profitable. Only a round the
cost model itself priced and chose resets the interval; a probe always widens
it (doubling, capped at `QWEN_MTP_PROBE_MAX` 256). If drafting really is
worthwhile the EMA rises, the cost model starts choosing depth on its own, and
that resets the interval.

`DS4_QWEN_MTP_PROBE=0` restores the absorbing behaviour bit-for-bit; N sets the
eager interval (default 4).

### Measured, M5 Max, 2 reps

Inert at the shipped default h=0.18 — bit-identical output *and* identical
accept counts on all three fixtures, so the shipped configuration is provably
unaffected:

| leg | decode | accepted | sha |
|---|---|---|---|
| h=0.18 @523 | 50.88 | 210 (1.59/rnd) | `59a3312301db` = golden |
| h=0.18 @2554 | 22.94 | 113 (0.63/rnd) | `7bcf7cd460ab` = pre-fix |
| h=0.18 @10295 | 19.84 | 58 (0.63/rnd) | `4589b20f7f04` = pre-fix |

It only acts where the policy was actually stuck:

| leg | `PROBE=0` (latched) | probe on | effect |
|---|---|---|---|
| h=0.90 @523 (recoverable) | 23.5, acc **0** | **36.0**, acc 114 (0.88/rnd) | **+52 %**, output still golden |
| h=0.35 @2554 (marginal) | 24.7, acc 13 | 23.3, acc 71 (0.40/rnd) | −6 %, 5.5× the acceptance |
| h=0.90 @10295 (unprofitable) | 24.5, acc 0 | 21.4, acc 3 (0.04/rnd) | −12.5 %, bounded by backoff |

The 10295 row is the honest cost: where speculation cannot pay, the probe still
samples occasionally. Backoff is what keeps it at −12.5 % rather than the −26 %
the accept-resets-eagerly version cost. Anyone running long context with a
hand-raised h should set `DS4_QWEN_MTP_PROBE=0`.

**What is still unmeasured:** a fixture with a genuine mid-generation regime
change (hard stretch then predictable stretch). The h=0.90 @523 leg is a proxy
— it latches at round 0 because `ema[0]` inits to 0.85 < 0.90 on a prompt that
is uniformly predictable. Both prose fixtures stay hard throughout, so they can
only ever show the probe's cost, never its benefit; that asymmetry is why the
first round of measurement here looked purely negative.

## Reproduction

Prompts live in `qwen38-mtp-fixtures/` (gitignored; 523-token copy-style English
in `correctness_prompts/`, `p2k.txt` = 2554 tok, `p8k.txt` = 10295 tok). On Max
they were reconstructed from `~/w/qwen38-challenge/correctness_prompts/` and
`/tmp`; on Min they are already in place.

```sh
cd ~/w/ds4
A=$PWD/qwen38-q4_64a-full.gguf                    # ds4-native Q4_64A scales
B=$PWD/gguf/Qwen3.8-27B-EigenLabs-Q4_64A.gguf     # MLX bf16 scales, parity only
H=$PWD/gguf/Qwen3.8-27B-MTP-bf16.gguf             # norm-corrected MTP head
F=$PWD/qwen38-mtp-fixtures/correctness_prompts/public_longcopy_gate_english_512.txt

# serial, and the split-K A/B in one window
./ds4 -m "$A" --prompt-file "$F" -n 256 --temp 0 --raw
DS4_QWEN_SHADOW_SPLITK=0 ./ds4 -m "$A" --prompt-file "$F" -n 256 --temp 0 --raw

# which geometry actually ran (print this before believing any number)
DS4_QWEN_SHADOW_DEBUG=1 ./ds4 -m "$A" --prompt-file "$F" -n 8 --temp 0 --raw

# speculative + the per-width verify counters that fit h
DS4_QWEN_MTP_HEAD=$H DS4_QWEN_MTP_AUTO=1 DS4_QWEN_MTP_PROFILE=1 \
  ./ds4 -m "$A" --prompt-file "$F" -n 256 --temp 0 --raw

# MLX-exact parity (must stay 0/248320 after ANY ds4.c change)
DS4_Q4_64A_MLX_BF16=1 ./ds4 -m "$B" -p Hello --raw --dump-logits /tmp/l.json
```

MLX 0.31.3 is not installed system-wide on Max; create it on demand with
`python3 -m venv /tmp/mlxq && /tmp/mlxq/bin/pip install mlx-lm==0.31.3`, then
compare `/tmp/l.json` against `mlx_lm.load('gguf/EigenLabs-Qwen3.8-27B-4bit')`.

**Artifact/mode pairing is still a silent-garbage trap.** `$A` runs in the
default mode and emits `)[)[)[…` under `DS4_Q4_64A_MLX_BF16=1`; `$B` is the exact
opposite. Neither errors. Always read the text before quoting a number.

**`DS4_QWEN_MTP_K` overrides `DS4_QWEN_MTP_AUTO`** (`ds4.c:42434`). Setting both
runs a fixed depth and silently produces a fake "AUTO" measurement — this cost a
whole matrix here. For true adaptive depth, set `DS4_QWEN_MTP_AUTO=1` alone.

## Measurement discipline that held up

Max does **not** show Min's sustain collapse: within one matrix, reps land
inside 1–2 %. Across matrices separated by tens of minutes the same config can
move ~15 % (h=0.18 @2554 read 26.61 in one window and 30.89 in another), so any
claim must come from a same-window A/B. Discard rep 1; acceptance counts are
deterministic per config, so if `accepted/round` matches across reps and t/s
moves, that is machine state, not your change.

Qwen-hybrid-path window drift (new): the identical binary, fixture
and machine measured prefill 78 ↔ 205 t/s and decode 7 ↔ 11 t/s
within minutes of each other, while the Ornith models hold ±0.6 %.
Cause unknown (AC and thermal ruled out). Rule: never quote a Qwen
absolute t/s without an interleaved control leg of the SAME binary
in the same window; prefer the deterministic counters (accepts,
shas) for verdicts.

## Gates before any commit

```sh
make -B ds4 ds4-server ds4-bench ds4-eval ds4-agent -j8
./ds4_test --server; ./ds4_agent_test; ./ds4-eval --self-test-extractors
./tests/test_layer_pack; ./tests/test_engine_mgpu_placement; ./tests/test_dflash_shape
./tests/test_q64a_quant; ./tests/test_gpu_args; ./tests/test_layer_stash; ./tests/test_gpu_args_cli.sh
make q4k-dot-test; make mxfp4-dot-test
DS4_TEST_MODEL=$PWD/qwen38-q4_64a-full.gguf ./tests/test_metal_session_batch
DS4_TEST_MODEL=$PWD/gguf/Ornith-1.5-35B-Q4_K_M.gguf ./tests/test_ornith15_bench.sh
DS4_TEST_MODEL=$PWD/gguf/Ornith-1.5-9B-Q4_K_M.gguf  ./tests/test_ornith9_bench.sh
```

Serialize the model-loaded harnesses (`sleep 5-10` between); ds4 refuses to
start while another ds4/ds4-server holds the single-instance lock. Bare
`./ds4_test` without `--server` reports 20 pre-existing Ornith-9B fixture
failures — prove pre-existence by diffing suite verdicts, do not chase them.

Full ladder ran green twice today (pre- and post-guard-fix), and
MLX parity held at 0/248320 diffs, max_abs 0.0 across the guard fix.

## Ornith results — Min, 2026-08-24 (`5ee5427`, merged here)

Min worked this in parallel on the same day and reached the **same dead-gate
conclusion independently**, from the Ornith side rather than the Qwen side.
Its commit generalised the grouped kernel to runtime ngrp 1..16; this tree
takes that bound (wider than the 2..16 arrived at here) and layers split-K on
top. Min's numbers below are Min's own and were not re-measured on Max.

### Shapes — the old `ngrp == 16` gate excluded every model that ships here

| model | arch | heads / kv | ngrp | draft path |
|---|---|---|---|---|
| Ornith-1.5-35B | `qwen35moe`, 39 exec + 1 bundled nextn | 16 / 2 | **8** | bundled nextn, auto (`DS4_QWEN_NEXTN_DRAFT=0` disables) |
| Ornith-1.5-9B | `qwen35` dense 32L | 16 / 4 | **4** | DFlash sidecar via `--dflash` |
| Qwen3.8-27B | hybrid, 16 full-attn layers | 24 / 4 | **6** | MTP head sidecar |

head_dim is 256 on all three. Min verified the generalisation bitwise:
grouped-serial ≡ legacy-serial across the full comparable span at ngrp=8, and
the Qwen3.8 golden still matched byte-for-byte after the rebuild.

### Measured on Min (523-tok fixture, greedy, warmed)

| leg | t/s | vs serial |
|---|---|---|
| 35B serial | 87.9–90.5 decode / ~1140 prefill | — |
| 35B nextn K=1 | **99.5 / 99.8** | **+13 %, the sweet spot** |
| 35B nextn K=3 | 57.7 | 0.64× |
| 35B AUTO h=0.34 | 86.1 | over-drafts |
| 35B AUTO h=0.7 | 86.6 | collapses to no drafting |
| 35B batched priming @10k | 263–275 vs 98–127 sequential | **2.1–2.8×**, text-identical |
| 9B serial | 75.9–77.7 / ~750 | matches the old Max figure 734 / 72.7 |
| 9B DFlash warmed `-n 256` | 53.1 (acc 220/230 = 95.6 %) | **0.70×** |
| both models @10k | grouped ON ≡ OFF, ≥18 t/s | **no cliff at these shapes** |

The old Max figure of "35B 269 t/s prefill" was measured on a soaked machine;
Min's ~1140 fresh is the honest number.

### Split-K on Ornith — measured on Max after the merge

Min's ngrp 1..16 bound admits both Ornith models, so split-K now defaults on
for them too. That was worth checking rather than assuming, since Min had
found no long-context cliff on these shapes. It is a clear win at 523 tokens,
`-n 128`, two reps, byte-identical output on every leg:

| model | ngrp | split-K default | `SPLITK=0` | gain |
|---|---|---|---|---|
| Ornith-1.5-9B | 16/4 = 4 | 84.30 / 84.86 | 75.60 / 76.37 | **+11.3 %** |
| Ornith-1.5-35B | 16/2 = 8 | 110.61 / 109.69 | 89.76 / 89.63 | **+22.4 %** |

Both beat Min's own serial figures (9B 75.9–77.7, 35B 87.9–90.5) by exactly
the split-K margin, which is a useful cross-machine consistency check.

`DS4_QWEN_SHADOW_DEBUG=1` now prints **each distinct (path, nsplit)** rather
than one warn-once line, because a warn-once here is actively misleading: the
first shadow decode of a run happens during priming at a tiny `pos`, where
nsplit is always 1. The 35B run reports `grp:1 split:2 split:3 split:4
split:5 split:6` — the leading `grp:1` is priming, and every steady-state
token is in the split path. Reading only the first line had me briefly
conclude 35B was not using split-K at all.

### Why Ornith's speculation economics differ from Qwen3.8's

Ornith35's draft head is a full MoE layer and every verified row re-routes
experts, so a marginal verify row costs ≈0.345 serial-token-times and each
draft step ≈0.34 (h ≈ 0.34 measured from the profile counters). A round costs
≈ 1 + 0.69·d against a yield of Σpⁱ with p ≈ 0.86, so break-even sits just past
depth 1: **fixed K=1 is optimal and deeper always loses.** The 9B DFlash draft
is target-class expensive (~9 ms per draft token), so even 96 % acceptance pays
nothing. Getting past +13 % on the 35B needs cheaper verify rows (a batched
MoE-row kernel) or a lighter head — not policy tuning.

Note this is the same shape of finding as the Qwen3.8 result above: once the
target step is cheap, speculation stops paying. Two models, two paths, one
conclusion.

### Correctness notes from Min

- Both Ornith exact-match harnesses pass through the generalised kernel.
- Ornith35 spec-vs-serial text diverges early **even under the old kernel**
  (legacy K=1 ≠ legacy serial at char 34): pre-existing MoE batched-verify
  near-tie behaviour, not this change, and both outputs stay coherent. Same
  class as the Qwen3.8 @10k divergence.
- Scripting trap that cost Min an hour: `run(){ shift; ...$1... }` names output
  files after the first env assignment rather than the intended tag, producing
  phantom diffs.

## Re-measurement round 2 (2026-08-24, post latch-fix)

### Top-2 clamp with the policy free

Accept counts here are deterministic and load-independent — every
rep pair matched on counts *and* sha256 — so the verdict rests on
counters, not on the t/s columns, which are unusable this window
(see the drift bullet under measurement discipline).

| fixture | leg | accepted (/round) | sha256 |
|---|---|---|---|
| 512 (-n256) | default | 210 (1.59) | `59a33123…af30` |
| 512 (-n256) | `TOP2=1` | 210 (1.59) | `59a33123…af30` |
| p2k (-n256) | default | 113 (0.63) | `7bcf7cd4…03dac` |
| p2k (-n256) | `TOP2=1` | 102 (0.56) | `89572561…cbd3d` |

At 512 the clamp is fully neutral: identical accept profile and
byte-identical output on both legs. At p2k it drops accepted
113 → 102 **and** changes the output bytes under greedy decoding,
i.e. the clamp alters token selection somewhere outside pure
target-greedy verification (repair tokens are draft-schedule-
dependent). It stays opt-in.

### h re-fit around 0.18

No h in {0.12, 0.18, 0.26} separates on the counters. Acceptance is
flat: 210–211 @512, 0.63–0.65/round at p2k/p8k.

| fixture | h=0.12 | h=0.18 | h=0.26 |
|---|---|---|---|
| 512 (-n256) | 210 (1.59/rnd) | 210 (1.59/rnd) | 211 (1.62/rnd) |
| p2k (-n256) | 115 (0.64) | 113 (0.63) | 113 (0.65) |
| p8k (-n128) | 58 (0.63) | 58 (0.63) | 57 (0.65) |

Output bytes differ per h despite the equal accepts — repair tokens
depend on the draft schedule rather than pure target-greedy verify.
Decode t/s this window is contaminated by the drift noted under
measurement discipline, so the re-fit **remains open** pending a
trustworthy Qwen window. Shipped default h=0.18 unchanged.

### Ornith split-K at 10k

p8k.txt (~10295 tok), `-n 128` greedy, interleaved reps, rep2 pair:

| model | split-K default | `SPLITK=0` | gain |
|---|---|---|---|
| Ornith-1.5-9B | **78.91** | 17.35 | **+354.8 %** |
| Ornith-1.5-35B | **22.17** | 12.39 | **+78.9 %** |

sha256 is byte-identical across ALL legs per model (9B `438da396…`,
35B `99817f43…`). Disabled-leg variance is large (35B r1 read
4.54), so treat the percentages as approximate; the ordering is
unambiguous. Against +11.3 %/+22.4 % at 523 tokens, the split-K
gain grows with context. Default stays ON.

### Where the 2554 fixed cost lives

**Blocked by machine drift.** Qwen3.8 serial measured 5.3–11 t/s
decode / 66–82 prefill all day (doc baseline 26 / ~230), and an
interleaved control proved binary-invariance: the main build itself
swung prefill 205.6 → 78.5 between windows minutes apart on
identical hardware, AC power, pmset thermals clean. No Qwen
fixed-cost number is quotable today; the ~34 ms + 6.4 µs/tok model
can neither be confirmed nor refuted on Qwen.

Reference point that did measure clean: Ornith-35B serial p2k with
`DS4_QWEN_PROFILE_ALWAYS=1` — steady-state decode command buffer
median **8.51 ms** (p90 8.72, min 8.46, max 16.12), whole-run sum
2556.79 ms over 140 CBs, warmup fill CBs 0.48/0.56 ms, prefill
giants 510.94 + 458.82 + 347.68 ms. The old fixed-cost model does
not describe this config either (predicts ~47 ms/CB at 2048 ctx).
Refit deferred.

### Probe benefit on a mid-generation regime change

New fixture `regime_change.txt` (local-only, gitignored dir):
hard-prose prefix, then a repetitive field-report tail. At h=0.90
the documented EMA-freeze latch reproduces on it — `PROBE=0` sits at
drafted=0 / accepted=0 frozen for the entire run — while the probe
restarts drafting (drafted=9, accepted=8), so the structural benefit
is demonstrated. The recovery itself is modest:

| leg | decode t/s | drafted | accepted |
|---|---|---|---|
| h=0.50, probe on | 5.75 | 225 | 193 (1.40/rnd) |
| h=0.50, `PROBE=0` | 5.68 | 225 | 193 (1.40/rnd) |
| h=0.90, probe on | 2.87 | 9 | 8 |
| h=0.90, `PROBE=0` | 2.76 | 0 | 0 (latched) |

h=0.50 never latches (both legs byte-identical, 193 accepted).
Magnitudes come from the drifted window; the ordering and the
counts are trustworthy.
