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

For MTP speculative-decoding experiments, pass the optional MTP GGUF and keep
all other sweep parameters identical:

```
./ds4-bench \
  -m ds4flash.gguf \
  --mtp gguf/DeepSeek-V4-Flash-MTP.gguf \
  --mtp-draft 2 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv /tmp/ds4-mtp.csv
```

The CSV keeps the original throughput columns first, then appends context memory
breakdown columns and speculative-decode counters.  For SOTA compression work,
compare `context_raw_bytes`, `context_compressed_bytes`, `context_scratch_bytes`,
and `kvcache_bytes` alongside throughput.  For speculative decoding, compare
the actual `gen_tokens`, `mtp_extra_accept_rate`,
`mtp_accepted_extra_tokens`, and `mtp_eval_sec` against the same run without
`--mtp`. For model-free suffix decoding, run the same sweep with
`--suffix-decoding` and compare `suffix_draft_attempts`, `suffix_draft_hits`,
`suffix_accepted_tokens`, and `suffix_avg_draft_len`.

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.
