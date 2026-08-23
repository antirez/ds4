# next-step.md — Qwen3.8-27B MTP speed work

Rewritten 2026-08-24 on **Max** (M5 Max, 128 GB) after taking the Min handoff
back. Every number below is measured on Max unless stated. The previous
revision of this file is commit `e5ce86d`; read the **Correction** section
before trusting anything you remember from it.

## Where the code stands

| branch | commit | contents |
|---|---|---|
| `ornith15` | `d6eb25f` | split-K shadow decode + ngrp gate fix + MTP verify-width counters + opt-in top-2 clamp |
| `main` | merge of the above | same tree |
| `e5ce86d` | previous tip | grouped shadow kernel (dead on Qwen3.8), adaptive depth, batched MTP priming |

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

1. **The adaptive policy latches off and cannot recover.** This is the biggest
   remaining defect and it is why the h table above has a cliff between 0.35 and
   0.18 rather than a smooth optimum. When `qwen_mtp_choose_k` returns 0,
   `qwen_mtp_update_ema` is called with `drafted == 0`, which matches none of its
   update branches — so the EMA freezes at whatever value stopped the drafting
   and the policy never drafts again. Reproduce: `DS4_QWEN_MTP_AUTO=1
   DS4_QWEN_MTP_H=0.35` at 2554 accepts 13 tokens in a whole 256-token run
   (0.07/round) versus 113 (0.63/round) at h=0.18. The fix is an exploration
   probe — force depth 1 every N consecutive zero rounds so the EMA keeps
   observing — after which h should be re-fitted, since 0.18 is currently
   partly chosen for latch-avoidance rather than for price.
2. **2554 is the only length still short of MLX** (25.5 vs 29.7). 523 and 10295
   are at or past the bar. Worth a `DS4_QWEN_PROFILE_ALWAYS=1` pass: decode is
   one command buffer per token, so the per-buffer aggregate localises it
   directly. At 2554 that buffer was 50.1 ms before split-K against 100.0 ms at
   10295, i.e. ~34 ms fixed + ~6.4 µs per context token; the fixed term is now
   the dominant cost at every length and nothing has profiled it yet.
3. **Carry split-K to Ornith.** The ngrp fix is already live for any model with
   a group of 2..16 and both Ornith exact-match gates pass, but no Ornith
   long-context A/B has been run. Baselines to beat, 523-tok prompt, n=128, no
   speculation: 35B 269 t/s prefill / 88.2 decode; 9B 734 / 72.7.
4. **Top-2 confidence clamp** is implemented and off by default
   (`DS4_QWEN_MTP_TOP2=1`). It uses a real k=2 indexer reduction plus two 4-byte
   reads per position — no full-logits copy. Measured neutral at 523 (+1.8 %,
   identical text) and negative at 2554 (24.59 vs 26.61), so it stays opt-in
   until item 1 is fixed; the clamp only matters when the policy is actually
   free to choose depth.

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
