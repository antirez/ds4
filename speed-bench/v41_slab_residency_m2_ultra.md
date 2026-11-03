# DeepSeek V4.1 Flash: owned Metal slab residency

For current-upstream results and the known upstream router test failure, see
[2026-09-14 revalidation](#upstream-revalidation-2026-09-14).
The diagnosis and validation below describe the original 2026-09-13 base.

## Problem and diagnosis

DeepSeek V4.1 Flash SSD decoding on this host fell to approximately 0.26 tokens/s
with the default owned expert-cache slabs. GPU execution itself did not explain
the multi-second token latency. Disabling slabs removed the severe cliff.
Registering allocations in an unattached residency set did not help; attaching
the set to the existing command queue did. Additional CPU `mlock` and warmup
alone did not remove the delay.

The model-free benchmark in this change allocates shared Metal buffers, writes
and locks them on the CPU, then reads small, dispersed samples through GPU
addresses with `useResource`. It performs no model-file IO. With a 104 GiB pool,
the initial queue-detached phase spent about 250 ms per large-buffer submission,
of which approximately 249.7 ms was in Metal's reported kernel/driver interval;
GPU execution was approximately 0.013 ms. Queue attachment reduced submission
wall time to approximately 0.180 ms. The 32 GiB control did not exhibit the cliff.
These per-phase means exclude the first four iterations of each phase.
These observations locate the delay in large-allocation submission/residency
handling, without claiming a particular private driver implementation or a
hardware-wide defect.

The queue-toggle run retains the residency set while detached. After its first
attachment, phase 2 remains fast: detachment alone is not an eviction command.
Do not interpret it as a reversible OFF/ON/OFF/ON performance experiment.
Fresh-process model comparisons below supply independent controls.

## Change

`DS4_METAL_STREAMING_SLAB_RESIDENCY=1` registers only owned expert-cache slab
buffers and attaches the set once to the existing queue on macOS 15+. It adds
no explicit `requestResidency`/`endResidency` pair. Cache teardown removes and
releases the set with the physical pool. Successful mlock-margin relief also
removes/releases it, preventing later slab allocation from reattaching until
the cache is rebuilt. Model mappings and disk-only Engram tables are excluded.
The feature remains opt-in pending evidence on other hosts and memory budgets.

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

[Raw model CSV](v41_slab_residency_m2_ultra.csv); model-free [104 GiB](v41_slab_residency_m2_ultra_repro_26.csv) and [32 GiB](v41_slab_residency_m2_ultra_repro_8.csv) CSVs.

| Run | Variant | Prefill t/s | Decode t/s (8 tokens) | First token ms | Steady t/s (7 tokens) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | control | 76.15 | 0.25 | 4103.550 | 0.25 |
| 2 | candidate | 70.95 | 9.57 | 329.048 | 13.83 |
| 3 | candidate | 69.16 | 9.67 | 316.315 | 13.71 |
| 4 | control | 60.11 | 0.25 | 3935.309 | 0.25 |

All four decoded outputs are identical. Eight-token decode intentionally keeps
the multi-second control bounded; its first-token cost materially affects the
aggregate. This is not a 512-token sustained-throughput claim. Prefill varies
from 60.11 to 76.15 t/s across controls and 69.16 to 70.95 t/s across candidates;
these two pairs do not establish a precise prefill speedup.

Reproduce each arm with the flag absent or set to `1`:

```sh
DS4_METAL_STREAMING_SLAB_RESIDENCY=1 ./ds4-bench -m MODEL --ssd-streaming \
  --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 2048 \
  --ctx-alloc 8257 --gen-tokens 8 --show-output --csv /tmp/slab-on.csv
```

## Validation

- Clean default Metal build, CPU compile, and restored Metal executable links.
- `make test-metal-slab-residency` with Metal API validation: protected expert
  survives relief; an unprotected slot is unlocked/evicted; the queue releases
  the set; allocation cannot reattach after relief; rebuilding reenables it;
  weak references confirm the set and old physical buffers are destroyed.
- `make test-metal-ssd-experts test-metal-moe-prefill`, plus SSD
  `--table-admission` and MoE `--ssd-address`, under Metal API validation.
- With the residency flag enabled and Metal API validation:
  `tests/test_deepseek41_graph MODEL --session-fixture` (real weights, snapshots,
  restored logits, prefix reuse, cancellation, boundary and malformed-state checks).
- `make test-frontends test-engram test-deepseek41-gguf test-quality-api`.
- Both 104 GiB and 32 GiB model-free runs check every GPU checksum.

The macOS SDK 15 build reports the same two pre-existing unused Metal 4 symbol
warnings as the upstream base. Full legacy `make test` was not run: its default
model vectors target the older Flash checkpoint; the V4.1 checks above were
used. This is single-host Metal evidence, not a release sign-off or validation
of CUDA, ROCm, RDMA, other Macs, or all memory-pressure conditions.

## Upstream revalidation (2026-09-14)

Revalidated after merging upstream `a04f46fa423e45712c8c7e430eff422479f314a3`
(DeepSeek V4.1 CUDA support). The original measurements above remain historical.
The host, exact model, prompt, context allocation, cache policy, and independent
ABBA procedure are unchanged. Each variant was measured twice on this host.

[New raw model CSV](v41_slab_residency_m2_ultra_20260914.csv).

| Run | Variant | Prefill t/s | Decode t/s | First token ms | Steady t/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | control | 72.13 | 0.24 | 4289.607 | 0.24 |
| 2 | candidate | 67.99 | 9.35 | 333.096 | 13.41 |
| 3 | candidate | 67.81 | 9.27 | 336.455 | 13.31 |
| 4 | control | 61.97 | 0.24 | 4037.232 | 0.24 |

Mean 8-token decode: 0.240 → 9.310 t/s (38.8×).
All four decoded outputs match exactly.

Validation rerun: clean Metal build, CPU compilation and restored Metal links;
frontend, Engram, V4.1 GGUF and quality-tool unit tests. Under Metal API
validation, compact carry, index scores/top-k, general top-k, index projection,
embedding and TP attention subtests pass. Slab lifecycle, SSD expert kernels,
MoE prefill, admission/address checks and the real-model session fixture pass.

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
Slab lifecycle and SSD expert/MoE kernel tests also pass.

Performance numbers above are the 2026-09-14 measurements, not a new benchmark
on this base. The unchanged upstream router failure is tracked by #1039 and
the separate accuracy fix #1044; neither is bundled into this optimization.

## Flash-1017 revalidation (2026-10-30)

Base: upstream `ee7e840` (refreshed again before publication). Host: Apple M2 Ultra,
192 GiB, macOS 15.7.4, `iogpu.wired_limit_mb=0` (unchanged for all runs).
Model: `AtomicChat/DeepSeek-V4.1-Flash-1017-Uncensored-DwarfStar-GGUF`,
revision `36c3f1413965b2df09f49e8bedd80dad1db76284`, file
`DeepSeek-V4.1-Flash-1017-Uncensored-IQ2_XXS-Q2_K-imatrix.gguf`.
Size: 365713686528 bytes; full-file SHA256 verified:
`09fb38b9482a43582027bb5ccf379d1827bd2380f5399fcc273af654b3a00ab1`.

Each ABBA comparison uses the same binary, a fresh process per run, the repository's
`speed-bench/promessi_sposi.txt`, 2048 prompt tokens, allocated context 8257, SSD
streaming and automatic expert-cache sizing. OS/file caches were not flushed.
This is two samples per arm on one host, not a cold-storage throughput test.

Measured code: `e5092ed3d2220f944aaf422c450cd97808494cd9`.

Switch: `DS4_METAL_STREAMING_SLAB_RESIDENCY=1`; 8 generated tokens.

Mean decode: **0.275 -> 10.665 tokens/s** (3778.2% increase).

| Run | Switch | Prefill t/s | Decode t/s | Steady decode t/s |
| --- | --- | ---: | ---: | ---: |
| 1 | 0 | 63.15 | 0.29 | 0.29 |
| 2 | 1 | 63.35 | 10.66 | 13.96 |
| 3 | 1 | 63.16 | 10.67 | 13.95 |
| 4 | 0 | 59.69 | 0.26 | 0.26 |

All four complete decoded texts are byte-identical within this comparison.

Reproduce each arm with the switch unset/set in ABBA order:

```sh
./ds4-bench -m "$MODEL" --ssd-streaming \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 2048 --ctx-alloc 8257 \
  --gen-tokens 8 --show-output --csv run.csv
```

The short 8-token result diagnoses the residency cliff; it is not a sustained throughput claim. Unmodified upstream measured 0.29 t/s with the same short workload. Lifecycle/pressure-relief/rebuild tests pass under Metal API validation, including a new injected residency-creation failure test that verifies the returned buffer remains GPU-usable. The reviewer-requested `fresh` flag removal and explicit warning/fallback are included. System swap remained zero in all four slab runs.

### Combined candidate and validation boundary

Separate integration tree `8da8039411e2297d04ec674c24d7bda6c6d44b42` combines
#1033, #1034 and #1234's FP4 precise-division fix, with native upstream Engram.
With slab residency and the queue enabled, measured decode was 16.26 t/s at 2K
(512 generated tokens), 16.14 at 8K and 14.66 at 32K (128 generated tokens each).
Steady decode was 16.37 / 16.57 / 15.45 t/s respectively. These are individual
integration observations, not isolated gains attributable to this PR.
System swap stayed at 37.75 MiB throughout those combined runs.

On that integration tree, Metal API validation passed the real-weight forced-eviction
check (exact full logits/history/cache spans, positions 121–144) and the public
session fixture (snapshots, prefix reuse, cancellation, bounds, final-head parity).
CPU compilation and Metal linking passed.

Pristine upstream's full Metal kernel test still fails FP4 E4M3 quantization
(mode 3, index 5428: 11.25 versus 7.5). The suite passed on the #1234 fix tree;
that kernel fix is intentionally not folded into either focused PR. Do not
interpret the two standalone PRs as independently fixing the upstream parity failure.
No fresh CUDA/ROCm hardware, 64K/128K, timed idle/resume, physical SSD-byte accounting,
or long-duration production soak is claimed by these measurements.
