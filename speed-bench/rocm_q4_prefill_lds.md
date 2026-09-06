# ROCm Q4 prefill: simpler LDS staging and final-fence removal

Local source review, 2026-09-05, based on `fce65048` with pre-existing
uncommitted Metal/CUDA work preserved. No remote tester was contacted.
The development host has neither `hipcc` nor an AMD GPU: **HIP compilation,
GPU bitwise parity, occupancy and performance remain unverified**. No model
throughput improvement or percentage is claimed.

## Scope and changes

This note describes the scalar streaming schedule. The subsequent
[four-word copy extension](rocm_q4_prefill_lds_vector.md) has an independent
rollback. The `--case lds` benchmark below explicitly disables that extension
on both arms, retaining the original scalar-only comparison.

Only the existing exact Q4_K/Q8_K tiled-prefill kernels change:

- Standalone K1024 TILE4 copies the entire activation tile as consecutive
  32-bit words. The old block/word/token decomposition simplifies exactly to
  `dst[i] = src[i]` for its contiguous four-block token rows.
- Generic standalone/strided TILE8 and paired TILE8 copy by token pitch:
  `p = i / (8*73)`, `word = i % (8*73)`. The source is relative to the
  tile's first token/group/K block. `word < nb*73` preserves partial K tiles
  without decoding each word's 73-word block index again. Contiguous complete
  rows can use the flat copy. Inactive tokens and K-tail slots remain untouched.
- Generic TILE8 retains the producer-to-consumer barrier on every K tile
  and the consumer-to-next-producer barrier whenever another tile follows.
  The final reuse barrier is omitted: the remaining work reads registers
  and writes output, not LDS. The predicate is workgroup-uniform and remains
  outside row/tail conditionals.

For T=ceil((K/256)/8), generic TILE8 executes `2*T-1` workgroup barriers instead
of `2*T`. This is a barrier-count reduction, **not a latency estimate**:

| Kernel/shape | Before | After |
| --- | --- | --- |
| TILE8 K256 or K1024 | 2 | 1 |
| TILE8 K4096 | 4 | 3 |
| TILE8 K7168 | 8 | 7 |
| K1024 TILE4 | 1 | 1 (copy indexing only) |

Quantization bytes, weight dot products, floating-point accumulation order,
subgroup reductions, workgroup size, tile geometry, scratch size and stream
ordering are unchanged. The optimization also reaches the strided
attention-output-A/B batch calls that already select these exact kernels.
Decode/speculative N<=8, routed-expert kernels, direct WMMA and F16 sidecars
are unchanged. Existing admission for resident/SSD/quality paths is not widened;
in particular this does not make SSD K1024 TILE4 automatic. No SSD I/O code
changes. If WMMA or an F16 path owns a projection, this flag has no effect.

## Default and rollback

The new schedule is default-on only after an existing tiled-prefill path has
been selected. Set `DS4_ROCM_DISABLE_Q4_PREFILL_LDS_STREAM=1` for its original
loader/barrier instantiation. Any defined value, including `0` and empty,
disables; unset restores the default. This does not disable TILE8 itself.

Both schedules are compile-time kernel instantiations. A thread-local launch
counter lets tests reject a supposed default run that never enqueues the
candidate. There is no new architecture-specific instruction or precision mode.

## Verification

Executed locally:

```sh
make test-rocm-q4-lds-host
```

- 3,456 cases execute the actual production copy helper on the CPU against
  the old independent block/word mapping. Every thread's destination set is
  compared, including leading/trailing guards, all 1..8 token tails, every
  1..4/1..8 K tail, contiguous/strided input, nonzero offsets and 32/64/256
  cooperative threads. Source buffers must remain unchanged.
- 4,096 K-loop schedules verify that exactly the final reuse barrier is
  omitted. This is a host predicate check, not GPU synchronization emulation.
- The same tests pass with AddressSanitizer and UndefinedBehaviorSanitizer.
- The public-runtime oracle passes host C++ syntax checking. Its existing
  unused HIP-only helper warning remains when compiled without HIP headers.
- `make test-rocm-q4-parity` reports **SKIP (hipcc not found)**, not GPU PASS.

On the AMD tester:

```sh
make test-rocm-q4-parity ROCM_ARCH=gfx1151 DS4_TEST_REQUIRE_ROCM_DEVICE=1
```

Use the actual `ROCM_ARCH` for other supported GPUs. The extended prefill
oracle replays default/unset and rollback `1`/`0`/empty through the public API,
checks candidate launch evidence, and compares complete output allocations
bit for bit. Dense, K1024 TILE4, paired, grouped/strided and Q4-A/Q8-B cases
retain the existing comparisons against the old untiled/row-wise references
and output canaries. Run `--prefill-long` separately for the existing long
stress cases. GPU execution has not been performed locally.

The global environment-reference checker already fails on an unrelated
stale `external/system/HOME` metadata source. Only the new flag's metadata
and reference rows are added; the unrelated inventory is left alone.

## Isolated A/B for testers

```sh
make rocm-q4-prefill-bench ROCM_ARCH=gfx1151
./speed-bench/rocm_q4_prefill_bench --case lds \
  --tokens 9,16,17,30,33,128,256,512 --sets 4 --samples 16 --warmup 4
```

This new case compares `lds_legacy` with `lds_stream` at K4096/M1024 and
K1024/M32768, forcing identical TILE8 or TILE4 geometry on both sides. It
alternates ABBA/BAAB with rotating resident weights, validates bitwise output
and guards before and after timing, and fails if candidate launch evidence is
missing or appears in the rollback arm. WMMA is disabled for this comparison.

The reported interval uses HIP events around public-API dispatch and includes
the unchanged Q8_K quantization in both arms; it is **not matmul-only timing**
and may include GPU idle time while the host submits work. Allocation,
residency setup, readback, comparison and environment selection are outside
the timed interval. The benchmark itself still requires HIP compilation and
an AMD run. Check VGPRs, spills and occupancy as well as latency: simpler
source indexing alone does not prove fewer machine instructions or a gain.

For a real-model A/B, leave all other settings and cache state unchanged and
alternate default with the rollback in fresh runs. Compare greedy output and
prefill latency in both resident and SSD configurations that actually use
Q8_K+TILE8. A gain on these synthetic shapes cannot be treated as a measured
whole-model or SSD throughput improvement.
