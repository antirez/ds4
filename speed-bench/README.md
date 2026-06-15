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
Record the machine, backend, model, and run parameters in `benchmarks.json`.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

<!-- BEGIN GENERATED BENCHMARK SUMMARY -->
## Benchmark Summary

Generated from the CSV files in this directory by `python3 speed-bench/update_summary.py`.

`@ 32k ctx` means the row where `ctx_tokens` is `32768`.

### DeepSeek V4 Flash q2

| Hardware | Best gen (t/s) | Gen @ 32k ctx (t/s) | Avg gen (t/s) | Best prefill (t/s) | Prefill @ 32k ctx (t/s) | Avg prefill (t/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Apple M4 Max | 26.8 | 24.5 | 24.6 | 344 | 248 | 250 |
| Apple M2 Ultra | 23.2 | 21.9 | 21.9 | 411 | 326 | 325 |
| NVIDIA DGX Spark / GB10 | 14.2 | 13.0 | 13.1 | 403 | 346 | 343 |

### DeepSeek V4 PRO q2

| Hardware | Best gen (t/s) | Gen @ 32k ctx (t/s) | Avg gen (t/s) | Best prefill (t/s) | Prefill @ 32k ctx (t/s) | Avg prefill (t/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Apple M3 Ultra | 12.4 | 9.56 | 9.90 | 183 | 139 | 149 |

<!-- END GENERATED BENCHMARK SUMMARY -->
