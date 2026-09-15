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

### DeepSeek V4.1 Flash streaming slab residency

On macOS 15+, `DS4_METAL_STREAMING_SLAB_RESIDENCY=1` attaches owned expert-cache
slabs to the existing Metal queue. It is opt-in; disk-backed model views and
Engram tables are excluded. Memory-pressure relief detaches the set until the
cache is rebuilt. See [the M2 Ultra investigation](v41_slab_residency_m2_ultra.md).

A model-free reproducer checks every GPU result and toggles queue attachment
OFF/ON/OFF/ON. The following allocates and locks approximately 104 GiB; run it
alone on a host with enough available memory:

```sh
make metal-slab-residency-bench test-metal-slab-residency
DS4_SLAB_BENCH_ALTERNATE_SMALL=1 ./speed-bench/metal_slab_residency_bench \
  toggle 26 4096 4080 24 6 > /tmp/slab-residency.csv
```

Arguments are mode, slab count, allocation MiB per slab, filled/locked MiB per
slab, iterations per phase and selected slabs per command buffer. Use eight
slabs instead of 26 for the 32 GiB control. No model file is opened.

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

Numeric runtime tuning controls can use `--candidate-value TEXT` (default `1`), and
`--prefill-chunk N` selects the same chunk size for both variants (default
4096). The control unsets the named variable. Use only controls read at
dispatch time: this harness keeps one engine alive and cannot compare settings
cached during initialization.

`DS4_QWEN4_TIMING=2` reports each prefill chunk's position, token count, and
stage times, including the attention mixer separately from GDN/attention.
These measurements include synchronization overhead and are for attribution;
use uninstrumented alternating runs for throughput claims.
`DS4_QWEN4_MOE_PROFILE=1` further splits tiled MoE into routing, expert-list
construction, routed/shared gate-up, routed/shared down, and reduction,
reporting one line per layer. Its synchronization also makes it diagnostic only.

Qwen routed MoE quantization specialization is enabled by default on M3 Ultra.
`DS4_QWEN4_MOE_MM_SPECIALIZE=0` restores the generic kernels; `=1` opts in on
other devices. To compare the final default against rollback, pass
`--candidate-env DS4_QWEN4_MOE_MM_SPECIALIZE --candidate-value 0` to the harness.
