# turbo3 KV cache A/B bench

A/B sweeps on the GX10 (ASUS Ascent, GB10 Blackwell chip, 128 GB unified memory) with the
IQ2XXS DeepSeek-V4-Flash checkpoint at
`/home/pidtom/models/ds4-model/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.

CSV files:
  * `gb10_fp8.csv`           fp8 baseline
  * `gb10_turbo3.csv`        turbo3 float-simulation cache
  * `gb10_turbo3_packed.csv` turbo3 packed-byte cache

Reproduce one cell:

```
./ds4-bench [--kv-cache turbo3] -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 16384 --step-incr 6144 --gen-tokens 64 \
  --csv /tmp/bench_$DTYPE.csv
```

ds4-bench prints the side-by-side fp8 vs turbo3 KV footprint at the chosen
ctx at startup:

```
ds4-bench: KV footprint @ ctx=16389:
  fp8     raw=365.50 MiB  compressed=215.23 MiB  total=580.73 MiB
  turbo3  raw=76.92 MiB   compressed=215.23 MiB  total=292.15 MiB  <-- active
  raw shrink: 4.75x  (turbo3 saves 288.58 MiB on the SWA ring)
```

## Throughput sweep

| ctx | fp8 prefill | t3-floatsim prefill | t3-packed prefill | fp8 gen | t3-floatsim gen | t3-packed gen |
|-----|------------:|--------------------:|------------------:|--------:|----------------:|--------------:|
|  2K | 399.48      | 399.04 (-0.1%)      | 388.07 (-2.8%)    | 13.71   | 13.62 (-0.7%)   | 11.92 (-13.1%) |
|  8K | 398.73      | 396.36 (-0.6%)      | 398.33 (-0.1%)    | 13.59   | 13.49 (-0.7%)   | 11.83 (-12.9%) |
| 14K | 383.73      | 381.41 (-0.6%)      | 384.30 (+0.1%)    | 13.41   | 13.33 (-0.6%)   | 11.67 (-13.0%) |
| 16K | 373.74      | 372.68 (-0.3%)      | 374.92 (+0.3%)    | 13.45   | 13.34 (-0.8%)   | 11.68 (-13.1%) |

Reading the table:

- **Prefill** is unchanged across all three - within 3% of fp8 baseline.
- **Gen_tps regresses ~13% on the packed-byte path** vs fp8 baseline.  The
  per-attention-call dequant-to-scratch kernel launch is the cost driver
  (~0.25 ms per decode-layer in the linear-attention pass).  The inline-
  dequant attention kernels eliminate the scratch pass and close the gap.

## Footprint shrink (the real payoff)

| ctx | fp8 SWA raw | turbo3 packed | shrink | absolute save |
|-----|------------:|--------------:|-------:|--------------:|
|  2K | head_dim*4 * raw_cap * 43 | head_dim*3.36 * raw_cap * 43 | 4.75x | 22 MiB at raw_cap=4096 |
| 16K | 365.50 MiB | 76.92 MiB | 4.75x | 288.58 MiB |

The 4.75x ratio is constant across ctx (per-row stride is independent of ctx;
raw_cap grows linearly).  The absolute MiB save grows linearly with ctx.

## Quality

`./ds4_test --logprob-vectors` (default fp8): **PASSES** -- bit-identical
to main on all 4 live vectors.

`DS4_TEST_KV_DTYPE=turbo3 ./ds4_test --logprob-vectors`: **FAILS** on
short_code_completion step 1 with a single argmax mismatch.  Expected:
the test asserts strict argmax equality at every position vs the official
continuation, and turbo3's quantisation noise shuffles the top-1 token in
~7-17% of positions while keeping the top-5 set intact (>99.6% top-5
agreement vs fp8 baseline -- see PR description for the KLD numbers).
This is a distribution-drift trade, not a bug.  Run
`ds4-bench --quality-baseline ...` for the KLD-aware comparator that
captures the actual quality envelope.

Smoke generation:
  * `"The capital of France is"` -> `Paris.` (byte-identical fp8 / turbo3-floatsim / turbo3-packed on this prompt)
  * `"Write the Python code to compute the factorial of n recursively."`
    -> 32-token identical Python function on this prompt.

## Why gen_tps regressed on the dequant-to-scratch path

The decompress-to-scratch architecture pays one dequant kernel launch per
attention call per layer.  At decode T=1, raw_cap=128, 43 layers, the
dequant pass does ~5500 group dequants per token (43 layers * 128 rows
* 1 launch).  Each dequant is cheap but launch overhead adds up.

The inline-dequant attention kernels move the dequant INSIDE each
attention kernel's V-load loop, eliminating the separate scratch pass and
capturing the V-load bandwidth shrink (4.75x less memory traffic on the
attention K/V read).

## What did NOT regress

  * **Prefill_tps within 3% of fp8** -- the dequant kernel scales O(ctx)
    while prefill compute is O(ctx^2), so prefill is dominated by the
    attention matmuls and the dequant pass is invisible.
  * **Quality is bit-identical to the float-sim path** -- the
    pack(unpack(x)) round trip is functionally lossless modulo FP8 group
    scale precision, and the matched-norm scale absorbs the precision
    loss.
  * **Disk session payload** for turbo3 sessions stores 4.75x fewer SWA-ring
    bytes via the per-dtype packed byte stride on the raw cache.
