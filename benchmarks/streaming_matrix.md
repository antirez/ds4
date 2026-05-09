# Streaming RAM Matrix

This file tracks constrained mlx-flash runs from `make test-constrained-ram`.
The raw data lives in `benchmarks/streaming_matrix.csv`.

## Current Default

The current default constrained test is:

```sh
make test-constrained-ram
```

with:

- `RAM_TEST_MB=16384`
- `RAM_TEST_STREAM_CACHE=16`
- `RAM_TEST_STREAM_WINDOW_MB=8`
- `RAM_TEST_PIN_MAX_MB=1`
- `RAM_TEST_COMPACT_CACHE_MB=0`
- `RAM_TEST_CTX=256`
- `RAM_TEST_TOKENS=2`

## Snapshot

| Run | Tokens | Cache | Window MiB | Compact Cache MiB | Output | Gen tok/s | Real s | Stream Peak MiB | Max RSS GiB | Stream Hit Rate | Compact Hit Rate |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Default tuned smoke | 2 | 16 | 8 | 0 | `Hello!` | 0.21 | 52.17 | 1911.64 | 1.76 | 43.5% | n/a |
| Tuned 4-token | 4 | 16 | 8 | 0 | `Hello! How can` | 0.26 | 56.88 | 1911.64 | 3.39 | 44.7% | n/a |
| Conservative 4-token | 4 | 1 | 1 | 0 | `Hello! How can` | 0.27 | 57.22 | 1010.02 | 3.38 | 29.3% | n/a |
| Tuned + compact tuple cache | 4 | 16 | 8 | 2048 | `Hello! How can` | 0.23 | 60.44 | 1911.64 | 4.65 | 44.7% | 1.6% |
| Conservative + compact tuple cache | 2 | 1 | 1 | 2048 | `Hello!` | 0.19 | 55.11 | 1010.02 | 2.28 | 27.7% | 0.0% |
| No-copy selected span | 2 | 16 | 8 | 0 | `Hello!` | 0.06 | 75.07 | 1911.64 | 0.05 | 41.4% | n/a |
| Per-expert slice cache, 512-entry cap | 4 | 16 | 8 | 8192 | `Hello! How can` | 0.21 | 63.70 | 1911.64 | 3.90 | 44.7% | 0.0% |
| Per-expert slice cache, 8192-entry cap | 4 | 16 | 8 | 8192 | `Hello! How can` | 0.21 | 61.94 | 1911.64 | 4.85 | 44.7% | 35.0% |

## Notes

- The persistent compact expert tuple cache is currently not a win. It adds RSS
  and memcpy pressure while producing very few hits on these short prompts.
- A wider stream cache/window reduces wrapper churn and evictions, but the
  current short tests are too dominated by prefill and one-token decode overhead
  to prove a large generation-rate gain.
- The tuned default stays well below the 16 GiB envelope. The highest observed
  RSS in this matrix is 4.85 GiB with the 8192-entry per-expert slice cache
  enabled, which is why that cache is still opt-in instead of the default.
- A no-copy selected-expert span is a clear regression. It avoids CPU copy work,
  but the min-to-max expert span pulls too much mmap-backed model data and drops
  generation to 0.06 tok/s.
- Per-expert slice caching can produce hits once the entry cap is high enough,
  but the current CPU copy and cache lookup path still does not improve tok/s.
  It needs a GPU-side gather or a different representation before it is worth
  making default.

## Columns

- `stream_cache`: `DS4_METAL_STREAM_CACHE`.
- `stream_window_mb`: `DS4_METAL_STREAM_WINDOW_MB`.
- `pin_max_mb`: `DS4_METAL_STREAM_PIN_MAX_MB`; small model views at or below
  this size avoid LRU eviction while an unpinned view is available.
- `compact_cache_mb`: `DS4_METAL_COMPACT_EXPERT_CACHE_MB`; set to `0` to keep
  compact expert buffers transient.
- `prefill_*`: counters at the post-prefill memory report.
- `final_*`: counters at the final memory report.

For longer comparisons, prefer `RAM_TEST_TOKENS=8` or `16` with the same short
prompt. That gives decode enough time for cache behavior to matter.
