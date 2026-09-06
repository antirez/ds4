# DS4 throughput on MacBook Pro M5 Max (128 GB) — Metal, IQ2

Five back-to-back `ds4-bench` sweeps with the documented long-context command,
spaced by a passive cool-down between runs to keep thermal throttling from
dominating any single sweep.

## Setup

| | |
|---|---|
| Host | MacBook Pro Apple M5 Max (`Mac17,7`), 18 cores, 128 GB unified memory |
| OS | macOS 26.5 (build `25F71`), Darwin `25.5.0` |
| Backend | Metal (default on macOS) |
| Model | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf` |
| Model SHA-256 | `31598c67c8b8744d3bcebcd19aa62253c6dc43cef3b8adf9f593656c9e86fd8c` |
| DS4 commit | `920f987` |
| Context buffers | 1933.10 MiB at `ctx=65665`, `prefill_chunk=2048`, `raw_kv_rows=2304`, `compressed_kv_rows=16418` |

Command used for every run:

```sh
./ds4-bench -m ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 \
  --step-incr 2048 --gen-tokens 128
```

Five sequential sweeps, each followed by an idle cool-down before the next was
launched, to keep tail rows comparable to head rows within a single run.

## Per-run summary

`prefill avg` and `gen avg` are arithmetic means across all 32 frontiers in the
sweep; the `@64k` columns are the last frontier (`ctx_tokens = 65536`).

| run | wall (s) | prefill avg | gen avg | prefill @64k | gen @64k |
|----:|---------:|------------:|--------:|-------------:|---------:|
| 1 | 491 | 215.77 | 24.68 | 166.11 | 21.53 |
| 2 | 467 | 231.78 | 25.08 | 162.05 | 20.56 |
| 3 | 477 | 226.16 | 24.68 | 160.92 | 19.93 |
| 4 | 527 | 209.87 | 22.66 | 150.88 | 18.31 |
| 5 | 479 | 227.23 | 25.01 | 167.57 | 21.15 |

Run 4 is visibly the hottest pass: wall time +12% over the median, both
averages and the `@64k` numbers are the worst of the five. Runs 1, 2, 3 and 5
are tightly clustered, suggesting the cool-down between runs is otherwise doing
its job; run 4 is included as-is — the spread is part of the result.

## Aggregated per-frontier throughput (n = 5)

`σ` is the population standard deviation across the five runs at that frontier.

| ctx | KV (MiB) | prefill mean | prefill σ | prefill min/max | gen mean | gen σ | gen min/max |
|----:|---------:|-------------:|----------:|----------------:|---------:|------:|------------:|
| 2048 | 49.8 | 369.72 | 1.68 | 367.41 / 371.25 | 31.46 | 0.07 | 31.37 / 31.56 |
| 4096 | 76.6 | 316.62 | 4.07 | 308.93 / 320.03 | 30.85 | 0.17 | 30.61 / 31.02 |
| 6144 | 103.5 | 300.43 | 8.07 | 287.41 / 308.13 | 30.46 | 0.30 | 29.88 / 30.70 |
| 8192 | 130.4 | 293.18 | 4.23 | 287.25 / 299.70 | 30.21 | 0.40 | 29.44 / 30.58 |
| 10240 | 157.3 | 280.01 | 5.79 | 271.40 / 287.65 | 29.58 | 0.12 | 29.45 / 29.79 |
| 12288 | 184.2 | 272.51 | 7.61 | 262.67 / 284.56 | 29.03 | 0.47 | 28.24 / 29.61 |
| 14336 | 211.1 | 263.58 | 8.23 | 253.18 / 276.85 | 28.88 | 0.25 | 28.55 / 29.11 |
| 16384 | 237.9 | 261.95 | 6.76 | 253.12 / 272.18 | 28.50 | 0.80 | 27.12 / 29.31 |
| 18432 | 264.8 | 256.54 | 9.01 | 242.75 / 268.11 | 27.04 | 0.51 | 26.40 / 27.78 |
| 20480 | 291.7 | 253.61 | 6.66 | 244.86 / 263.49 | 26.96 | 0.47 | 26.30 / 27.36 |
| 22528 | 318.6 | 251.17 | 6.63 | 240.68 / 259.51 | 26.57 | 0.63 | 25.41 / 27.12 |
| 24576 | 345.5 | 247.47 | 7.40 | 237.63 / 256.34 | 26.30 | 0.67 | 25.01 / 26.88 |
| 26624 | 372.4 | 240.08 | 8.75 | 229.31 / 250.25 | 25.73 | 0.73 | 24.30 / 26.28 |
| 28672 | 399.2 | 234.19 | 10.30 | 221.75 / 246.17 | 25.46 | 0.82 | 23.84 / 26.08 |
| 30720 | 426.1 | 229.06 | 11.66 | 213.45 / 241.76 | 24.78 | 0.94 | 22.95 / 25.42 |
| 32768 | 453.0 | 222.39 | 13.50 | 202.83 / 238.00 | 24.29 | 1.27 | 21.81 / 25.22 |
| 34816 | 479.9 | 212.49 | 16.40 | 186.43 / 230.12 | 23.50 | 1.57 | 20.45 / 24.86 |
| 36864 | 506.8 | 203.31 | 18.22 | 172.81 / 222.65 | 22.92 | 2.13 | 18.80 / 24.67 |
| 38912 | 533.7 | 195.13 | 19.74 | 160.96 / 214.76 | 22.17 | 2.22 | 17.89 / 24.17 |
| 40960 | 560.5 | 189.39 | 18.41 | 157.57 / 208.13 | 21.89 | 2.00 | 18.09 / 23.75 |
| 43008 | 587.4 | 183.36 | 14.86 | 159.23 / 200.00 | 21.36 | 1.71 | 18.16 / 23.16 |
| 45056 | 614.3 | 178.57 | 13.04 | 156.37 / 194.05 | 21.01 | 1.63 | 18.01 / 22.90 |
| 47104 | 641.2 | 174.81 | 12.31 | 154.00 / 188.24 | 20.68 | 1.53 | 17.81 / 22.32 |
| 49152 | 668.1 | 171.82 | 10.44 | 153.41 / 182.24 | 20.45 | 1.34 | 17.89 / 21.78 |
| 51200 | 695.0 | 167.86 | 8.79 | 151.50 / 175.61 | 20.20 | 1.34 | 17.61 / 21.30 |
| 53248 | 721.8 | 165.84 | 7.91 | 150.99 / 172.05 | 20.17 | 1.14 | 17.99 / 21.05 |
| 55296 | 748.7 | 164.62 | 6.72 | 151.74 / 170.83 | 20.07 | 1.15 | 17.90 / 20.99 |
| 57344 | 775.6 | 164.83 | 6.44 | 152.20 / 170.15 | 20.15 | 1.03 | 18.20 / 20.90 |
| 59392 | 802.5 | 162.03 | 6.05 | 150.97 / 168.85 | 20.14 | 1.09 | 18.12 / 21.19 |
| 61440 | 829.4 | 161.08 | 5.72 | 151.35 / 168.03 | 20.29 | 1.09 | 18.32 / 21.43 |
| 63488 | 856.3 | 160.05 | 6.72 | 151.03 / 168.35 | 20.12 | 1.18 | 18.22 / 21.50 |
| 65536 | 883.1 | 161.51 | 5.86 | 150.88 / 167.57 | 20.30 | 1.13 | 18.31 / 21.53 |

## Observations

- **Cold context is fast.** At `ctx = 2048` Metal hits **~370 tok/s prefill /
  ~31.5 tok/s generation** with σ < 2 tok/s and σ < 0.1 tok/s respectively —
  the M5 Max is very repeatable when nothing is hot or memory-pressured.
- **Decode degrades smoothly with KV size.** Generation drops from 31.5 tok/s
  at 2k to ~20 tok/s at 64k — a ~35% fall over a 32× context expansion. KV
  attention reads dominate at the tail.
- **Prefill plateaus, doesn't collapse.** Prefill flattens at ~160 tok/s once
  context passes ~50k. The high-context floor is stable and reproducible.
- **Mid-context (~30k–40k) shows the highest run-to-run variance** (σ up to
  ~20 tok/s on prefill, ~2 tok/s on generation). The same rows in run 4 lag
  the others — those frontiers happen to land inside the thermally-loaded
  window of a run, while the head and tail of every sweep are tighter.
- **Best-row comparison to the README M3 Max single-run numbers**:
  at the head, M5 Max prefill peaks at **371 t/s** (M3 Max README short-prompt
  prefill: 58 t/s — measured a different way); the long-context generation
  floor here is ~20 t/s vs the README's 21.47 t/s at 11.7k tokens for M3 Max.
  These are not apples-to-apples (different prompt lengths and run shape),
  but the M5 Max sustains its decode rate noticeably farther into the
  context.

## Reproducing

The raw CSVs (one per run) and the captured `ds4-bench` stderr are in
`bench/results/run{1..5}.csv` and `bench/results/run{1..5}.log`. The aggregate
table above is regenerated from those CSVs.
