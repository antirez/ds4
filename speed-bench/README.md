## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

### Metal decode schedule A/B

Build the balanced, same-engine Metal decode comparison with:

```
make metal-decode-schedule-bench
./speed-bench/metal_decode_schedule_bench \
  -m ds4flash.gguf \
  --include-selection
```

The harness prefills two sessions and alternates both variant order and
variant-to-session assignment. It aborts unless every full-vocabulary logit
row is bit-identical and, with `--include-selection`, both variants select the
same non-EOS token. Use `--candidate-env NAME` to measure a rollback control,
or `--help` to compare explicit split schedules.

To compare the default pre-M5 ratio-4 compressor pack/transpose fusion with the
legacy decode path, including token selection, use:

```
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION \
  --include-selection \
  --tokens 1024
```

### GLM-5.3 width-2 MTP verification A/B

Build and run the exact optimized/rollback comparison with a GLM-5.3 model:

```
make glm53-mtp-head-bench
./speed-bench/glm53_mtp_head_bench \
  /path/to/GLM-5.3-Flash-Q2.gguf
```

The harness uses one engine and two synchronized sessions. It runs three
512-token blocks, reverses arm order every 64 tokens, and uses
`DS4_GLM_MTP_DISABLE_DEFERRED_ROW1_HEAD=1` as the rollback control. It aborts
unless token IDs, acceptance schedules, positions, and full-vocabulary logits
are bit-identical after every chunk.

Pass a second argument to compare another dynamic rollback switch. The
snapshot-aware width-2 KDA path uses:

```
./speed-bench/glm53_mtp_head_bench \
  /path/to/GLM-5.3-Flash-Q2.gguf \
  DS4_METAL_DISABLE_GLM53_KDA_VERIFY2_SNAPSHOT_FUSION
```

Two complete 2026-08-31 M4 Max runs recorded:

| Run / block | Optimized t/s | Rollback t/s | Delta |
|---|---:|---:|---:|
| 1 / 1 | 29.136657 | 28.484798 | +2.2884% |
| 1 / 2 | 28.876846 | 28.477513 | +1.4023% |
| 1 / 3 | 28.473334 | 28.054908 | +1.4915% |
| 1 aggregate | 28.826356 | 28.337641 | +1.7246% |
| 2 / 1 | 29.598591 | 28.867628 | +2.5321% |
| 2 / 2 | 29.832179 | 29.228604 | +2.0650% |
| 2 / 3 | 29.882952 | 29.192804 | +2.3641% |
| 2 aggregate | 29.770724 | 29.095435 | +2.3209% |
| Combined | **29.290900** | **28.711500** | **+2.0180%** |

See `docs/research/glm53_kda_verify2_snapshot.md` for the exact model,
runtime revision, confidence calculation, whole-stack control, and Metal trace.

### Metal prefill variant A/B

Build the balanced prefill comparison. To compare the default resident pre-M5
MXFP4 pair tail-SIMDgroup cull against the original pair kernel, make the
rollback path the candidate:

```
make metal-prefill-variant-bench
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_PAIR_TAIL_SIMDGROUP_CULL
```

To isolate the default routed-down tail-SIMDgroup cull from the retained pair
default, use its down-specific rollback as the candidate:

```
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_DOWN_TAIL_SIMDGROUP_CULL
```

The harness uses one Metal engine and fresh sessions for every run. It warms
both variants with at least 32 tokens, alternates control/candidate order in
ABBA and BAAB blocks, poisons host logit buffers before copying, and aborts
unless every final full-vocabulary logit row is bit-identical. Defaults are an
8192-token prefix, an automatically sized 8193-token context, and two repeats;
use `--help` to override them.
