# HISA hierarchical indexer bench

`gb10_spark.csv` is raw `ds4-bench --csv` output from this branch on a
GB10 (ASUS Ascent, sm_121, 128 GB unified memory) running Qwen3.6-A3B
IQ2XXS with `--backend cuda --kv-cache turbo3 --comp-cache turbo3` and
inline-dequant comp_kv enabled (the configuration that exposes HISA at
long context).

Two rows, two context points:

| ctx | n_index_comp | HISA dispatch | gen_tps |
|---:|---:|---|---:|
| 65536 | 16394 | dormant (under gate, flat indexer runs) | 11.01 |
| 262144 | 65542 | active (over gate, HISA runs) | 7.61 |

The 64K row confirms zero regression when the gate keeps HISA off; the
256K row is the long-context point where HISA replaces the flat scan.

## Reproduce

This branch, applied on top of #243 (for `--kv-cache turbo3 --comp-cache
turbo3`):

```sh
make cuda-spark
./ds4-bench -m ds4flash.gguf \
  --backend cuda --kv-cache turbo3 --comp-cache turbo3 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 65536 --ctx-max 262144 --step-incr 196608 \
  --gen-tokens 32 \
  --csv speed-bench/hisa/gb10_spark.csv
```

`prefill_tps` is unchanged from the parent commit at both ctx points;
HISA is a decode-token optimization and the prefill batched-attention
path is untouched.

## What this CSV does not contain

A `--gen-tokens 128` sweep at the canonical 2K..256K step-incr=16384
granularity and matched per-dtype before/after CSVs across `fp8`,
`turbo4`, and `turbo3+comp` are queued.  The compact deltas in the PR
body for `fp8`, `turbo4`, and `turbo3+comp` come from separate
single-point runs from the same session that were not preserved as raw
`--csv` output; they will be regenerated and added here alongside the
full sweep.

## Perplexity

Teacher-forced PPL on the same model and prompt, 64 scored tokens:

```
ds4-bench: PPL teacher-forced  kv_cache=turbo3  tokens=64  scored=63  elapsed=4.06s
ds4-bench:   nll_avg=4.674636  ppl=107.193523
```

Identical to the parent-commit baseline at the same configuration.
