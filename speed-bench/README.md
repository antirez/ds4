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

<!-- BEGIN GENERATED BENCHMARK SUMMARY -->
## Benchmark Summary

Generated from the CSV files in this directory by `python3 speed-bench/update_summary.py`.

`@ 32k ctx` means the row where `ctx_tokens` is `32768`.

### DeepSeek V4 Flash q2

| Hardware | Best gen (t/s) | Gen @ 32k ctx (t/s) | Avg gen (t/s) | Best prefill (t/s) | Prefill @ 32k ctx (t/s) | Avg prefill (t/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Apple M4 Max | 26.76 | 24.52 | 24.57 | 343.76 | 247.91 | 250.39 |
| Apple M2 Ultra | 23.22 | 21.92 | 21.85 | 410.62 | 325.77 | 324.90 |
| NVIDIA DGX Spark / GB10 | 14.23 | 12.98 | 13.13 | 402.88 | 346.36 | 343.02 |

### DeepSeek V4 PRO q2

| Hardware | Best gen (t/s) | Gen @ 32k ctx (t/s) | Avg gen (t/s) | Best prefill (t/s) | Prefill @ 32k ctx (t/s) | Avg prefill (t/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Apple M3 Ultra | 12.42 | 9.56 | 9.90 | 183.06 | 138.82 | 149.28 |

<!-- END GENERATED BENCHMARK SUMMARY -->
