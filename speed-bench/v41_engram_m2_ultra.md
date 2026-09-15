# DeepSeek V4.1 Flash: parallel uncached Engram decode reads

For current-upstream results and the known upstream router test failure, see
[2026-09-14 revalidation](#upstream-revalidation-2026-09-14).
The diagnosis and validation below describe the original 2026-09-13 base.

## Problem and diagnosis

One token reads 24 native Engram rows at each of two layers. Each row is only
264 bytes, but the current uncached reader serializes all 48 reads. A matched
microbenchmark showed that sorting/deduplicating through the prefill batch
reader alone did not improve latency (about 4.81 ms for both tables). Four
readers reduced it to about 1.37 ms whether rows were sorted or left in original
order. This change uses the smaller original-order implementation.

## Change

On macOS, `DS4_ENGRAM_PARALLEL_DECODE=1` partitions a 24-row read into four
six-row calls to the existing serial reader, using `dispatch_apply_f` to join
before return. Inputs are validated before dispatch. Each worker writes a
disjoint output range and records its own error; the caller propagates an error
after every worker has joined. It adds no row cache, sorting, heap allocation,
new descriptor, or change to the batched prefill reader. Other platforms and
row counts use the existing path. As with the serial reader, output can be
partially written after an IO/data error and callers must honor failure.

## Environment and results

Measured on 2026-09-13, Apple M2 Ultra, 192 GiB unified memory, macOS 15.7.4,
Metal, against upstream `bd66c402070042bf0a79ad6ece8242de4c93680c`.
Model: DeepSeek V4.1 Flash calibrated IQ2_XXS/Q2_K, 365,713,686,528 bytes;
SHA-256 `1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42`.
The model was reused without conversion. The machine's existing
`iogpu.wired_limit_mb=188000` was unchanged. One inference/Metal benchmark ran
at a time; ordinary desktop background services remained running.

All model benchmarks use `speed-bench/promessi_sposi.txt`, SSD streaming,
a 2,048-token prefill, 8,257 allocated context, default power and automatic
expert-cache sizing. The planner reports 135.26 GiB dynamic expert cache plus
7.12 GiB prefill headroom. Each process starts a fresh engine/cache; the OS/file
cache is not flushed. Order is control, candidate, candidate, control. Controls
use the same branch binary with the opt-in flag absent. The production path
with the flag absent is unchanged from the upstream base.

For independent model measurements, `DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS=1`
is present in both arms. The queue optimization is absent.

[Raw model CSV](v41_engram_m2_ultra.csv).

| Run | Variant | Prefill t/s | Decode t/s (512 tokens) | First token ms | Steady t/s (511 tokens) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | control | 68.60 | 10.27 | 389.896 | 10.34 |
| 2 | candidate | 68.68 | 10.80 | 402.655 | 10.87 |
| 3 | candidate | 68.69 | 10.80 | 368.530 | 10.86 |
| 4 | control | 68.80 | 10.32 | 385.098 | 10.38 |

Mean 512-token decode: 10.295 → 10.800 t/s (+4.9%).
All four 512-token decoded outputs are identical. These are two observations
per variant on one host, not release medians or a cross-device estimate.

Reproduce each arm with the candidate flag absent or set to `1`:

```sh
DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS=1 DS4_ENGRAM_PARALLEL_DECODE=1 \
./ds4-bench -m MODEL --ssd-streaming \
  --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 2048 \
  --ctx-alloc 8257 --gen-tokens 512 --show-output --csv /tmp/candidate.csv
```

The real-row microbenchmark (excluding pass 0) measures 4.787 ms serial
and 1.359 ms parallel per token for both tables. [Raw CSV](v41_engram_m2_ultra_rows.csv).

```sh
./speed-bench/engram_decode_bench MODEL 162955640832 384006168 264333279232 384016682
```

Those offsets are valid only for the exact GGUF identified above.

The checked-in row benchmark uses 128 deterministic 24-row sets per table and
eight alternating passes. It reads through the real Engram table API and checks
every output bit. This workload measures scattered uncached reads, not every
possible token distribution, storage device or shared-host load.

## Validation

- Clean default Metal build, CPU compile, and restored Metal executable links.
- `make test-engram`: serial and parallel modes, row order, duplicate IDs,
  signed zero, output guards, invalid final row rejected before writing,
  EDOM in each of four partitions, EIO on truncated input, and existing
  hash/history, batched-reader and numeric checks.
- `MTL_DEBUG_LAYER=1 DS4_ENGRAM_PARALLEL_DECODE=1
  tests/test_deepseek41_graph MODEL --session-fixture`: real-model session,
  snapshot/restore, cancellation and bounds checks.
- `make test-frontends test-deepseek41-gguf test-quality-api`.

The SDK 15 build retains two upstream unused Metal 4 symbol warnings. Full
legacy `make test` was not run against mismatched older Flash vectors. Other
platforms were not executed; CPU portability was compile-checked only.

## Upstream revalidation (2026-09-14)

Revalidated after merging upstream `a04f46fa423e45712c8c7e430eff422479f314a3`
(DeepSeek V4.1 CUDA support). The original measurements above remain historical.
The host, exact model, prompt, context allocation, cache policy, and independent
ABBA procedure are unchanged. Each variant was measured twice on this host.

Upstream added a non-macOS pthread path for large batched reads. This change
still applies only to macOS single-token 24-row reads; both paths and their
EDOM/EIO tests are retained.

[New raw model CSV](v41_engram_m2_ultra_20260914.csv).

| Run | Variant | Prefill t/s | Decode t/s | First token ms | Steady t/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | control | 68.43 | 10.20 | 341.775 | 10.26 |
| 2 | candidate | 68.65 | 10.72 | 331.508 | 10.78 |
| 3 | candidate | 68.63 | 10.76 | 338.298 | 10.82 |
| 4 | control | 68.66 | 10.21 | 353.569 | 10.26 |

Mean 512-token decode: 10.205 → 10.740 t/s (+5.2%).
All four decoded outputs match exactly.

Real-row reads, excluding pass 0: 4.898 → 1.418 ms per token
for both tables, with bitwise output checks. [Raw row CSV](v41_engram_m2_ultra_rows_20260914.csv).

Validation rerun: clean Metal build, CPU compilation and restored Metal links;
frontend, Engram, V4.1 GGUF and quality-tool unit tests. Under Metal API
validation, compact carry, index scores/top-k, general top-k, index projection,
embedding and TP attention subtests pass. The real-model session fixture
passes with parallel Engram reads enabled.

The full `test-deepseek41-metal` suite fails in the new upstream router test
at `tests/test_deepseek41_metal.c:105`. An unmodified checkout of the same
upstream commit reproduces exactly the same failure under Metal API validation:

```text
router n=256 mode=0 rows=1 row=0 expert=4
logit=-11.8886719 actual=0.00263455603 ref=0.00262947031
```

No kernel or tolerance was changed to bypass it. The full kernel suite is
therefore not green. CUDA/ROCm hardware and full legacy `make test` were not
executed. The two existing SDK 15 unused Metal 4 symbol warnings remain.

## Upstream integration (2026-09-15)

Merged upstream `9139e2ae58a41503968a500f36f75895c1ba63fc` and retained both
sets of test targets/documentation at the conflict. Clean Metal and CPU builds,
frontend/Engram/V4.1 GGUF/quality-tool tests and the real-model session fixture
with this optimization enabled pass. The session fixture ran with Metal API
validation.
The macOS decode switch remains opt-in.

Performance numbers above are the 2026-09-14 measurements, not a new benchmark
on this base. The unchanged upstream router failure is tracked by #1039 and
the separate accuracy fix #1044; neither is bundled into this optimization.
