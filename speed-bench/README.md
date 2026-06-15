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

| Benchmark | Best gen | Gen @ 32k ctx | Avg gen | Best prefill | Prefill @ 32k ctx | Avg prefill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M4 Max | 26.76 t/s | 24.52 t/s | 24.57 t/s | 343.76 t/s | 247.91 t/s | 250.39 t/s |
| M2 Ultra | 23.22 t/s | 21.92 t/s | 21.85 t/s | 410.62 t/s | 325.77 t/s | 324.90 t/s |
| GB10 | 14.23 t/s | 12.98 t/s | 13.13 t/s | 402.88 t/s | 346.36 t/s | 343.02 t/s |
| PRO model M3 Ultra | 12.42 t/s | 9.56 t/s | 9.90 t/s | 183.06 t/s | 138.82 t/s | 149.28 t/s |

<!-- END GENERATED BENCHMARK SUMMARY -->
