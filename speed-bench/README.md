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

For Metal Tensor prefill experiments, treat matmul as the first optimization
surface: profile routed-MoE stages and dense Q8_0 attention projections, then
compare the current standard path, current Tensor auto path, and a default-off
candidate env switch with:

```
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label moe-matmul-first \
  --set-env DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1
```

Add `--run-drift-gate` before promoting a candidate. That reuses the
five-fixture `--quality` drift gate and writes a JSON summary beside the
benchmark CSVs.
