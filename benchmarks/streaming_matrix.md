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
- `RAM_TEST_STREAM_CACHE=32`
- `RAM_TEST_STREAM_WINDOW_MB=8`
- `RAM_TEST_PIN_MAX_MB=1`
- `RAM_TEST_COMPACT_CACHE_MB=4096`
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
| GPU blit gather | 2 | 16 | 8 | 0 | `Hello!` | 0.30 | 48.49 | 1911.64 | 0.05 | 34.0% | n/a |
| GPU blit gather | 4 | 16 | 8 | 0 | `Hello! How can` | 0.35 | 52.21 | 1911.64 | 0.05 | 31.1% | n/a |
| Private one-encoder blit gather | 2 | 16 | 8 | 0 | `Hello!` | 0.29 | 48.65 | 1911.64 | 0.05 | 34.2% | n/a |
| Larger stream window | 2 | 16 | 64 | 0 | `Hello!` | 0.10 | 54.75 | 4096.09 | 0.05 | 69.1% | n/a |
| More small stream windows | 2 | 32 | 8 | 0 | `Hello!` | 0.34 | 47.41 | 3849.30 | 0.05 | 34.6% | n/a |
| More small stream windows | 4 | 32 | 8 | 0 | `Hello! How can` | 0.36 | 49.31 | 3849.30 | 0.05 | 31.4% | n/a |
| More small stream windows | 2 | 48 | 8 | 0 | `Hello!` | 0.32 | 41.80 | 5742.97 | 0.05 | 35.1% | n/a |
| GPU slice cache | 2 | 32 | 8 | 4096 | `Hello!` | 0.36 | 40.19 | 3849.30 | 0.05 | 34.6% | 0.0% |
| GPU slice cache | 4 | 32 | 8 | 4096 | `Hello! How can` | 0.36 | 46.86 | 3849.30 | 0.05 | 35.0% | 35.0% |
| GPU slice cache | 8 | 32 | 8 | 4096 | `Hello! How can I assist you today` | 0.37 | 59.43 | 3849.30 | 0.05 | 35.2% | 47.5% |
| GPU slice cache | 8 | 32 | 8 | 8192 | `Hello! How can I assist you today` | 0.38 | 61.45 | 3849.30 | 0.05 | 35.6% | 49.8% |
| GPU slice cache repeat | 2 | 32 | 8 | 4096 | `Hello!` | 0.37 | 39.45 | 3849.30 | 0.05 | 34.6% | 0.0% |
| Cleaned default smoke | 2 | 32 | 8 | 4096 | `Hello!` | 0.33 | 42.87 | 3849.30 | 0.05 | 34.6% | 0.0% |

## Failed Configurations

| Run | Tokens | Cache | Window MiB | Output | Real s | Stream Peak MiB | Reason |
|---|---:|---:|---:|---|---:|---:|---|
| Wide stream window | 2 | 64 | 64 | `FAILED_PREFILL_BUDGET` | 7.78 | 16320.38 | Filled the 16 GiB stream budget by prefill layer 8 before eviction could start. |

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
- GPU blit gather for selected expert compact buffers is the first clear win:
  the 2-token smoke improved from 0.21 to 0.30 tok/s and the 4-token comparison
  improved from 0.27 to 0.35 tok/s while keeping RSS tiny. It increases wrapper
  churn, but the CPU-copy removal more than pays for it on short decode runs.
- Collapsing the three routed expert blit encoders into one and making compact
  buffers private was safe under the cap, but did not improve the 2-token smoke
  run. This suggests the remaining cost is not encoder setup or shared-buffer
  CPU visibility alone.
- A 64 MiB stream window with 64 cache slots is too wide for the 16 GiB cap.
  The budget guard caught it during prefill at 16.0 GiB live stream views. Any
  wider-window test needs a smaller cache or earlier eviction.
- A 64 MiB window with 16 cache slots is safe and improves stream hit rate, but
  decode gets slower. Larger view ranges reduce wrapper churn while increasing
  eviction/advice and page-touch cost, so the current default stays at 8 MiB.
- Doubling the 8 MiB stream cache from 16 to 32 improves both the 2-token smoke
  and the 4-token comparison while remaining well inside the 16 GiB envelope.
  The constrained Makefile default now uses 32 stream windows.
- Increasing the 8 MiB stream cache to 48 stays well under the 16 GiB cap and
  lowers 2-token wall time, but generation tok/s does not beat 32 windows.
- A GPU-private per-expert slice cache with a 4 GiB budget improves the 2-token
  smoke and keeps total footprint under the 16 GiB cap. The first short run had
  no compact-cache hits, so this needs a longer run before making it default.
- At 4 tokens, the GPU-private slice cache gets the same 35% expert hit rate as
  the earlier CPU-copy slice cache but avoids the CPU-copy regression. It lowers
  wall time and wrapper churn while keeping generation tok/s tied with the best
  no-cache run.
- At 8 tokens, the 4 GiB GPU slice cache reaches a 47.5% hit rate but starts
  evicting. Generation improves only slightly to 0.37 tok/s, so the next check
  is whether an 8 GiB cache reduces churn without harming the cap.
- The 8 GiB GPU slice cache removes compact-cache evictions and reaches the
  best generation tok/s so far, but wall time regresses and peak footprint grows
  to about 6.27 GiB. It is still inside the 16 GiB envelope, but not an obvious
  default.
- Repeating the 2-token 4 GiB GPU-cache smoke produced 0.37 tok/s, confirming
  short-run noise but also confirming the setting is consistently ahead of the
  pre-cache baseline.
- The constrained Makefile default now uses the 4 GiB GPU slice cache. It is
  faster than no cache in the repeated smoke and does not pay the wall-time
  penalty seen with the 8 GiB cache.
- After removing the regressed span and CPU-copy cache experiments from the
  executable code, the default constrained smoke still passes under the cap.
  The 2-token result landed at 0.33 tok/s, within the noisy short-run band.

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
