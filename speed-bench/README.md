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

The pre-M5 one-token Flash attention-output LOW projection also has an exact
fixed-shape Q8_0 kernel. Compare it with the generic rollback using:

```
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_ATTN_OUT_LOW_Q8_STATIC \
  --include-selection \
  --tokens 512
```

Balanced M3 Ultra A/B runs favored the fixed-shape kernel by 0.53%, 0.53%,
and 0.58% at a 2K-token prefix (43.03/42.80, 42.96/42.73, and 43.24/43.00
tok/s) and by 0.48% at an 8K-token prefix (38.44/38.26 tok/s). An independent
IQ2/Q2-model run at 2K gave 44.47/44.23 tok/s (+0.56%). All 1,909 compared
rows, 246,795,520 full-vocabulary logits, and 1,904 selected token IDs were
bit-identical.
Performance was measured on M3 Ultra; the exact host gate covers the shared
M1-M4 Flash shape and otherwise retains the generic kernel.

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

For resident pre-M5 MXFP4 prefills of 32 through 2047 tokens, the exact
scatter map, compact pair tile, pair/down tail culls, and down half-LUT now
use the same defaults as the established 2K+ path. Compare the complete
short-prefill extension with its aggregate rollback using:

```
./speed-bench/metal_prefill_variant_bench \
  --prefix-tokens 256 \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_SMALL_PREFILL
```

Balanced M3 Ultra A/B medians for the tuned path versus the rollback were
112.43/89.25 tok/s at 32 tokens (+26.0%), 111.95/89.30 at 33 (+25.4%),
180.48/142.59 at 64 (+26.6%), 275.72/219.98 at 128 (+25.3%),
389.27/316.93 at 256 (+22.8%), 512.94/439.39 at 512 (+16.7%),
173.94/157.08 at 1024 (+10.7%), and 632.30/593.70 at 2047 (+6.5%). Every
one of the 64 measured runs produced bit-identical full-vocabulary logits.
Performance was measured on M3 Ultra; the guarded default also covers the
shared resident M1-M4 path.

The harness uses one Metal engine and fresh sessions for every run. It warms
both variants with at least 32 tokens, alternates control/candidate order in
ABBA and BAAB blocks, poisons host logit buffers before copying, and aborts
unless every final full-vocabulary logit row is bit-identical. Defaults are an
8192-token prefix, an automatically sized 8193-token context, and two repeats;
use `--help` to override them.
