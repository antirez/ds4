# Streaming RAM Matrix

This file tracks constrained mlx-flash runs from `make test-constrained-ram`.
The raw data lives in `benchmarks/streaming_matrix.csv`.

`generation_tps` is the primary speed metric. It comes from the `ds4:
generation:` timing line, not from `/usr/bin/time`; process `real_s` is kept
only as a secondary end-to-end cost that includes startup, mmap setup, hot-range
planning, prefill, and shutdown. For steady-state decode comparisons, measure
from first token onward and avoid treating `real_s` as tok/s.

## Current Default

The current default constrained test is:

```sh
make test-constrained-ram
```

with:

- `RAM_TEST_MB=16384`
- `RAM_TEST_RESIDENT_HOT_MB=8192`
- `RAM_TEST_STREAM_CACHE=32`
- `RAM_TEST_STREAM_WINDOW_MB=8`
- `RAM_TEST_STREAM_CACHE_RAM_MB=4096`
- `RAM_TEST_PIN_MAX_MB=1`
- `RAM_TEST_COMPACT_CACHE_MB=4096`
- `RAM_TEST_CTX=256`
- `RAM_TEST_TOKENS=2`

The largest local test envelope we currently allow is:

```sh
make test-constrained-ram-24gb
```

with `RAM_TEST_MB=24576`, `RAM_TEST_RESIDENT_HOT_MB=8192`,
`RAM_TEST_STREAM_CACHE_RAM_MB=8192`, and `RAM_TEST_COMPACT_CACHE_MB=8192`.

## Snapshot

Representative rows only. The CSV keeps the full exploration history.

| Run | RAM MiB | Hot MiB | Stream MiB | Compact MiB | Tokens | Gen tok/s | Max RSS GiB | Stream Evict | Compact Evict | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Early no-hot baseline | 16384 | 0 | 3849 peak | 4096 | 8 | 0.37 | 4.29 | 7767 | 1024 | GPU slice cache before planned hot residency. |
| 16 GiB default, first-token basis | 16384 | 8192 | 4096 | 4096 | 8 | 0.74 | 7.72 | 2871 | 1024 | Current conservative split; hot ranges do the main work. |
| 16 GiB more compact | 16384 | 8192 | 2048 | 6144 | 8 | 0.77 | 7.72 | 2751 | 0 | Compact evictions disappear, stream churn remains. |
| 24 GiB balanced | 24576 | 9216 | 7168 | 8192 | 8 | 0.77 | 8.25 | 2743 | 0 | All non-expert hot ranges fit; extra compact budget is unused. |
| 24 GiB stream-heavy | 24576 | 9216 | 9216 | 6144 | 8 | 0.72 | 8.25 | 2743 | 0 | More stream budget does not improve reuse. |
| 24 GiB optimized | 24576 | 8192 | 8192 | 8192 | 8 | 0.83 | 7.72 | 2751 | 0 | Direct-source compact misses; current fastest 8-token split. |
| 24 GiB optimized, longer check | 24576 | 8192 | 8192 | 8192 | 16 | 0.82 | 7.72 | 3116 | 0 | Stopped early at EOS but confirms the optimized split. |
| 24 GiB wider stream window | 24576 | 8192 | 8192 | 8192 | 8 | 0.46 | 7.72 | 2597 | 0 | 16 MiB windows overfetch and regress despite fewer wrappers. |
| 24 GiB low hot residency | 24576 | 4096 | 12288 | 8192 | 8 | 0.54 | 4.05 | 4674 | 0 | Lower RSS but far more streaming churn. |

## Conclusions

**Hot Residency**

Planned hot residency is the largest confirmed improvement. Moving frequently
used non-expert tensors into hot mmap-backed Metal views raises the 8-token
first-token-basis run from the earlier 0.37 tok/s baseline to 0.74 tok/s. The
current planner needs about 8.20 GiB to cover all selected non-expert hot ranges,
so setting hot residency much above 9 GiB will not help unless the planner starts
selecting another tensor class.

**Compact Expert Cache**

The compact expert cache is useful, but only up to the active expert working set.
At 4 GiB it still evicts 1024 entries in the 8-token run. At 6 GiB it removes
compact evictions and improves generation from 0.74 to 0.77 tok/s. An 8 GiB
budget leaves headroom for slightly longer or different prompts; the 16-token
optimized check reaches 6.79 GiB compact live.

**Stream-View Cache**

The stream-view cache is still the main churn signal. Even with 7-9 GiB of
stream-view budget, final stream evictions stay around 2740 and stream hits stay
around 75. That means larger stream residency alone is not turning into reuse;
the remaining cost is likely repeated per-token wrapping, blit/gather work, and
page churn for ranges that are not naturally reused by the current cache shape.
Increasing stream slots from 32 to 64 lowered evictions slightly but measured
0.81 tok/s, below the 32-slot baseline. Increasing the stream window to 16 MiB
reduced wrapper count but overfetched enough to drop decode to 0.46 tok/s.

**24 GiB Envelope**

The 24 GiB ceiling is safe in the tested splits. After optimizing compact-cache
misses, the best 24 GiB split is `8 GiB hot + 8 GiB stream + 8 GiB compact` at
0.83 tok/s for the 8-token run and 0.82 tok/s in the longer EOS-terminated
check. Larger stream allocations and the 9 GiB hot plan do not improve tok/s on
this prompt.

**Route / Compact / Gather**

The current sorted compact gather remains the best measured route through the
routed MoE path. Two targeted experiments were rejected: using CPU-known hash
route IDs to avoid the first three layer readbacks measured 0.78 tok/s, and
preserving router order in compact buffers measured 0.77 tok/s. The sorted
compact IDs appear to help source/cache locality more than route-order compact
layout helps the matvec kernels. Reusing private compact scratch buffers was
also tested and regressed to 0.80 tok/s; disabling that experiment returned the
same binary to 0.83 tok/s, so the code was removed.

## Failed Configurations

| Run | Tokens | Cache | Window MiB | Output | Real s | Stream Peak MiB | Reason |
|---|---:|---:|---:|---|---:|---:|---|
| Wide stream window | 2 | 64 | 64 | `FAILED_PREFILL_BUDGET` | 7.78 | 16320.38 | Filled the 16 GiB stream budget by prefill layer 8 before eviction could start. |

## Notes

- Planned hot residency is the major win so far. The first-token-basis 8-token
  run is 0.74 tok/s with hot ranges, versus 0.37 tok/s before hot residency.
- The 16 GiB conservative default is `8 GiB hot + 4 GiB compact + 4 GiB stream`.
  It stays well under the 16 GiB envelope in observed RSS, but still reports
  2871 stream evictions and 1024 compact expert evictions on the 8-token run.
- Moving 2 GiB from stream views to compact experts eliminates compact-cache
  evictions and improves generation slightly to 0.77 tok/s, but stream evictions
  remain high.
- The useful 24 GiB ceiling now provides a modest speed win after the
  direct-source compact miss optimization. The representative 24 GiB split is
  `8 GiB hot + 8 GiB compact + 8 GiB stream`, which reaches 0.83 tok/s. A
  stream-heavy `8 + 6 + 10` split ties on this prompt but leaves less compact
  headroom.
- The hot planner can cover all currently selected non-expert ranges with about
  8.20 GiB resident. Raising hot residency beyond 9 GiB will not help unless the
  planner starts selecting routed expert ranges or another hot tensor class.
- Stream-view eviction is tracked and remains the clearest churn signal. Larger
  stream budgets reduce peak budget pressure, but they have not improved reuse:
  stream hits stay around 75-78 on these 8-token runs while final stream
  evictions remain around 2740-2870.
- Compact expert eviction is also tracked. A 6 GiB compact expert budget is
  enough for the 8-token prompt, but 8 GiB is the safer 24 GiB setting because
  the longer check reaches 6.79 GiB compact live.
- The next likely speed path is reducing per-token streaming/blit work, not just
  increasing cache sizes. The latest code avoids the old cold-miss dependency
  chain of `source -> cached slice -> compact` by copying cold misses directly
  from the source view into the compact buffer while still populating the cache.
  The latest rejected experiments show that small route/readback or allocation
  changes are not enough; remaining candidates are deeper GPU-side
  gather/compute fusion without falling back to full-expert stream views.

## Columns

- `stream_cache`: `DS4_METAL_STREAM_CACHE`.
- `stream_window_mb`: `DS4_METAL_STREAM_WINDOW_MB`.
- `pin_max_mb`: `DS4_METAL_STREAM_PIN_MAX_MB`; small model views at or below
  this size avoid LRU eviction while an unpinned view is available.
- `compact_cache_mb`: `DS4_METAL_COMPACT_EXPERT_CACHE_MB`; set to `0` to keep
  compact expert buffers transient.
- Hot residency is tracked in run notes for now: `DS4_METAL_RESIDENT_HOT_MB`
  selects planned non-expert model ranges, and `DS4_METAL_STREAM_CACHE_RAM_MB`
  caps the remaining transient mmap-backed stream-view cache.
- `generation_tps`: primary decode-speed metric from the program timing line.
  Do not derive tok/s from `real_s`.
- `real_s`: `/usr/bin/time -l` process wall time; useful for regressions in
  setup/prefill/shutdown, not for steady-state decode speed.
- `prefill_*`: counters at the post-prefill memory report.
- `final_*`: counters at the final memory report.

For longer comparisons, prefer `RAM_TEST_TOKENS=8` or `16` with the same short
prompt. That gives decode enough time for cache behavior to matter.
