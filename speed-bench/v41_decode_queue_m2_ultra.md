# DeepSeek V4.1 Flash: bounded streaming decode queue

For current-upstream results and the known upstream router test failure, see
[2026-09-14 revalidation](#upstream-revalidation-2026-09-14).
The diagnosis and validation below describe the original 2026-09-13 base.

## Problem and diagnosis

After removing the large-slab submission cliff, single-host SSD decoding still
waited at every layer, despite selected-expert ID readback already bounding
work on the same ordered Metal queue. A diagnostic profile with the companion
slab-residency fix held constant measured 81 versus 43 command buffers per token.
The mean of the last seven of eight tokens changed from 75.299 to 66.849 ms,
while GPU execution remained 41.379 versus 41.427 ms. Host wait time changed
from 58.847 to 51.324 ms. This profile explains the scheduling opportunity; it
is not the standalone performance measurement for this PR. The [per-token
profile CSV](v41_decode_queue_m2_ultra_profile.csv) records milliseconds;
`queued` denotes companion slab queue residency, not this PR's decode switch.

## Change

`DS4_METAL_ENABLE_V41_STREAM_DECODE_QUEUE=1` reuses the existing bounded queue
schedule for single-host, non-quality SSD decode. It keeps the drain after
layer 13 (before layer 14 overwrites shared Engram input), and at token completion.
Expert-ID readback and existing cache-replacement synchronization are preserved.
Resident, quality/imatrix and two-host TP eligibility remain unchanged.

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

For this independent comparison, `DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS=1`
is present in both arms. Otherwise the unrelated large-slab latency dominates
on this host. Neither companion optimization is required or enabled.

[Raw model CSV](v41_decode_queue_m2_ultra.csv).

| Run | Variant | Prefill t/s | Decode t/s (512 tokens) | First token ms | Steady t/s (511 tokens) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | control | 68.30 | 10.28 | 382.664 | 10.34 |
| 2 | candidate | 66.55 | 11.36 | 362.478 | 11.43 |
| 3 | candidate | 68.63 | 11.41 | 370.835 | 11.49 |
| 4 | control | 68.96 | 10.28 | 404.379 | 10.35 |

Mean 512-token decode: 10.280 → 11.385 t/s (+10.7%).
All four 512-token decoded outputs are identical. These are two observations
per variant on one host, not release medians or a cross-device estimate.

Reproduce each arm with the candidate flag absent or set to `1`:

```sh
DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS=1 DS4_METAL_ENABLE_V41_STREAM_DECODE_QUEUE=1 \
./ds4-bench -m MODEL --ssd-streaming \
  --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 2048 \
  --ctx-alloc 8257 --gen-tokens 512 --show-output --csv /tmp/candidate.csv
```

Balanced same-engine harness result:

```text
variant=control first_split=2 second_split=32 tokens=512 seconds=42.790230 tokens_per_second=11.9653
variant=candidate first_split=2 second_split=32 tokens=512 seconds=38.415040 tokens_per_second=13.3281
exact_rows=529 exact_floats=68389120 exact_selected_ids=528 vocab=129280
```

Command: the SSD streaming example in [README](README.md#metal-decode-schedule-ab).

## Validation

- Clean default Metal build, CPU compile, and restored Metal executable links.
- `MTL_DEBUG_LAYER=1 tests/test_deepseek41_graph MODEL --stream-decode-queue-parity`:
  two sessions share a 512-expert cache, forcing eviction. After restoring a
  120-token prefix, positions 121–144 compare every full-vocabulary float,
  finite logits, Engram history and all saved cache spans bit-for-bit.
- `MTL_DEBUG_LAYER=1 make test-deepseek41-metal`.
- `make test-frontends test-engram test-deepseek41-gguf test-quality-api`.
- Balanced same-engine decode harness, alternating variant order and assignment
  to sessions, with complete-logit and selected-token equality checks.

The SDK 15 build retains two upstream unused Metal 4 symbol warnings. Full
legacy `make test` was not run against mismatched older Flash vectors. These
focused V4.1 results do not establish release, multi-host or other-backend QA.

## Upstream revalidation (2026-09-14)

Revalidated after merging upstream `a04f46fa423e45712c8c7e430eff422479f314a3`
(DeepSeek V4.1 CUDA support). The original measurements above remain historical.
The host, exact model, prompt, context allocation, cache policy, and independent
ABBA procedure are unchanged. Each variant was measured twice on this host.

The opt-in single-host queue condition is now explicitly guarded by
`__APPLE__`; upstream CUDA decode and existing TP eligibility are preserved.

[New raw model CSV](v41_decode_queue_m2_ultra_20260914.csv).

| Run | Variant | Prefill t/s | Decode t/s | First token ms | Steady t/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | control | 67.40 | 10.13 | 394.734 | 10.19 |
| 2 | candidate | 66.23 | 11.30 | 330.683 | 11.37 |
| 3 | candidate | 66.39 | 11.33 | 362.067 | 11.40 |
| 4 | control | 68.84 | 10.21 | 353.083 | 10.27 |

Mean 512-token decode: 10.170 → 11.315 t/s (+11.3%).
All four decoded outputs match exactly.

Balanced same-engine harness:

```text
variant=control first_split=2 second_split=32 tokens=512 seconds=42.651979 tokens_per_second=12.0041
variant=candidate first_split=2 second_split=32 tokens=512 seconds=38.513994 tokens_per_second=13.2939
exact_rows=529 exact_floats=68389120 exact_selected_ids=528 vocab=129280
```

Validation rerun: clean Metal build, CPU compilation and restored Metal links;
frontend, Engram, V4.1 GGUF and quality-tool unit tests. Under Metal API
validation, compact carry, index scores/top-k, general top-k, index projection,
embedding and TP attention subtests pass. The forced-eviction real-model queue
parity test passes for complete logits, history and saved cache spans.

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
