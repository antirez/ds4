# GLM-5.3 width-2 KDA snapshot fusion

Date: 2026-08-31  
Machine: Mac Studio, Apple M4 Max (16 CPU cores), 128 GiB RAM  
OS: macOS 26.5.2 (25F84)  
Upstream base: `b1b4ea03645434423e5cb4f39818fdc075e49825`  
Measured runtime revision: `48029e0`  
Model: `antirez/glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf`,
92,035.1 MiB, embedded width-2 MTP  
Model SHA-256: `e81fd6241c6e55a64e1e14e47a3eab61a173fa8d7e4b5c1d1848827119705b32`  
Backend/context: Metal, resident model, 8,192-token context

## Result

The snapshot-aware KDA verify path improves generated throughput by **2.02%**
in the paired aggregate, from **28.712 to 29.291 tokens/s**. All six 512-token
blocks were positive. The conservative two-sided 95% lower confidence bound
on the six within-machine paired log-speedups is **+1.53%**. This interval
covers one prompt, model, machine, and runtime configuration; it is not a
cross-workload confidence claim.

The complete width-2 stack reaches **29.291 tokens/s**, versus **24.563
tokens/s** for the exact upstream base under the same harness (**+19.25%**).
That comparison is an end-to-end result: base and final produced different
MTP acceptance schedules, so it must not be used to attribute the gain to an
individual change. The rollback-controlled result above isolates the final
KDA fusion.

The change is exact relative to the previous split path. Kernel tests require
byte equality for the row-0 convolution and recurrent snapshots, row-1 final
states, and both output rows. Model-backed resident and SSD-streaming tests
also require identical token IDs, speculative cycle sizes, positions, and
full-vocabulary logits after every cycle.

## Change

A rejected width-2 draft must keep KDA state after verifier row 0. The previous
path therefore ran KDA once for row 0, copied convolution and recurrent state
into the prefix buffer with two Metal copy encoders, then ran KDA again for
row 1. Across 34 KDA layers, every speculative cycle built 476 temporary
tensor views, encoded 102 additional KDA dispatches, and inserted 68 state
copy encoders.

The Metal-only `ds4_gpu_glm53_kda_verify2_snapshot()` operation keeps the
existing three-stage KDA structure:

1. prepare both rows and write row-0 Q/K/V convolution history in-kernel;
2. run both recurrent updates and write row-0 H in-kernel;
3. normalize both output rows with the existing output kernel.

The live state finishes after row 1. The packed prefix layout remains
convolution state followed by recurrent state for each KDA layer. A rejected
draft restores that prefix through the existing speculative transaction; an
accepted pair keeps the live row-1 state. Ordinary prefill, CUDA, ROCm, and CPU
paths do not call the new Metal primitive. The dynamic diagnostic rollback is:

```sh
DS4_METAL_DISABLE_GLM53_KDA_VERIFY2_SNAPSHOT_FUSION=1
```

## Paired throughput benchmark

The checked-in harness uses one engine and two synchronized sessions. It
alternates arm order every 64 generated tokens, runs three 512-token blocks,
and aborts unless token IDs, cycle sizes, acceptance/rejection counts,
positions, and the full vocabulary logits are byte-identical after each
chunk. The two processes were run sequentially; no other model process ran at
the same time.

```sh
make glm53-mtp-head-bench
./speed-bench/glm53_mtp_head_bench \
  /path/to/GLM-5.3-Flash-Q2.gguf \
  DS4_METAL_DISABLE_GLM53_KDA_VERIFY2_SNAPSHOT_FUSION
```

| Run / block | Fused t/s | Split rollback t/s | Delta |
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

Every block followed the same schedule in both arms: 124 single-token cycles,
194 double-token cycles, and 118 post-seed rejections. The combined row uses
`3072 / sum(elapsed)` for each arm, not an arithmetic mean of rates.

For the six paired block log-speedups: mean `0.02002799`, sample standard
deviation `0.00463404`; the one-sided 95% lower bound is `+1.6348%`, and the
more conservative two-sided 95% lower endpoint is `+1.5280%`.

## Whole-stack and ordinary decode controls

The same harness was compiled against the exact upstream base. Its two arms
are behaviorally identical because the rollback variable does not exist there;
together they provide 3,072 generated tokens. Base ran at `24.563300` t/s
(108 singles, 202 doubles, 106 post-seed rejections per 512-token block).
The two final optimized arms above also total 3,072 tokens and ran at
`29.290900` t/s (124 singles, 194 doubles, 118 rejections per block).

Ordinary non-MTP prefill/decode was measured once per revision with
`ds4-bench`, the checked-in `speed-bench/promessi_sposi.txt`, a 2,048-token
frontier, 128 generated tokens, and an 8,192-token allocation:

| Revision | Prefill t/s | Generation t/s | Steady generation t/s |
|---|---:|---:|---:|
| upstream `b1b4ea0` | 236.40 | 23.45 | 23.47 |
| final runtime `48029e0` | 237.78 | 23.54 | 23.56 |
| delta | +0.58% | +0.38% | +0.38% |

These single runs are a non-regression control, not a statistically powered
claim of an ordinary prefill/decode speedup.

## Metal System Trace

Two 96-token runs used the same prompt and retained the final eight seconds of
the `Metal System Trace` template. The rollback trace set only the diagnostic
environment variable above.

```sh
xcrun xctrace record --template 'Metal System Trace' --window 8s \
  --output glm53-kda-verify2-fused.trace --launch -- ./ds4 \
  -m /path/to/GLM-5.3-Flash-Q2.gguf --metal --ctx 8192 \
  --tokens 96 --temp 0 --nothink --mtp-timing \
  -p 'Explain, in detailed numbered steps, how an append-only write-ahead log, checkpoints, and idempotent replay recover a stateful service after a crash. Continue until every invariant and edge case is covered.'

xcrun xctrace record --template 'Metal System Trace' --window 8s \
  --env DS4_METAL_DISABLE_GLM53_KDA_VERIFY2_SNAPSHOT_FUSION=1 \
  --output glm53-kda-verify2-rollback.trace --launch -- ./ds4 \
  -m /path/to/GLM-5.3-Flash-Q2.gguf --metal --ctx 8192 \
  --tokens 96 --temp 0 --nothink --mtp-timing \
  -p 'Explain, in detailed numbered steps, how an append-only write-ahead log, checkpoints, and idempotent replay recover a stateful service after a crash. Continue until every invariant and edge case is covered.'
```

The optimized trace contained 62 representative verify command buffers with
9–13 GPU intervals each. Their weighted mean summed GPU time was `45.289 ms`
(range `44.557–46.200 ms`). The rollback trace contained 46 representative
verify buffers with 77–80 intervals each and a `45.980 ms` weighted mean
(range `44.898–47.150 ms`). The change therefore removes about 68 recorded
intervals and saves `0.691 ms` of GPU work per verify (`-1.503%`). The trace
matches the intended removal of two prefix-state copy encoders per each of 34
KDA layers and agrees with the paired end-to-end result.

## Correctness commands

```sh
make test-glm53-kda

DS4_TEST_MODEL=/path/to/GLM-5.3-Flash-Q2.gguf \
DS4_TEST_GLM_MTP=1 \
./ds4_test --session-snapshot

DS4_TEST_MODEL=/path/to/GLM-5.3-Flash-Q2.gguf \
DS4_TEST_GLM_MTP=1 \
DS4_TEST_SSD_STREAMING=1 \
DS4_TEST_SSD_STREAMING_CACHE_GB=16 \
./ds4_test --session-snapshot
```

Both model-backed runs completed 16 cycles as 9 singles and 7 doubles (23
committed tokens), including eight single-token cycles after the seed. The
resident run exercises the fused and split paths and requires byte-identical
cycle logits. SSD streaming intentionally remains on the established batch
fallback; its run confirms that the Apple-only fast path does not change the
streaming result.

The first automatic SSD-cache attempt was rejected during engine setup because
the 80% working-set policy left a one-expert cache, below the required 3.80 GiB
routed-prefill reserve. The explicit 16 GiB cache is the repository's intended
test configuration and passed; this setup failure produced no inference or
kernel result.
