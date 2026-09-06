# ROCm Q4 prefill: aligned four-word LDS copies

Source review based on `76115c71`. This extends the existing
[scalar LDS streaming schedule](rocm_q4_prefill_lds.md), not the dot product.
No remote tester was contacted. HIP compilation, GPU parity, generated
instructions, register pressure and performance are **pending**.

## Scope

The exact Q4_K/Q8_K TILE8 standalone/strided and paired prefill kernels, plus
the K1024 TILE4 specialization, can copy four consecutive 32-bit activation
words per thread iteration. Their vector-capable instantiations align the
shared tile to 16 bytes without changing its footprint. Fixed-size memcpy
with checked alignment keeps the raw Q8_K bytes and avoids typed aliasing
through a new vector type.

The helper requires 16-byte-aligned source and destination pointers and a
source token stride divisible by four words. Q8_K is 292 bytes and only
four-byte aligned: arbitrary group offsets and strides must not be assumed
to satisfy this condition. Incompatible tiles retain the previous scalar
streaming helper. Compatible partial K tiles use scalar copies for the last
one to three valid words, never reading padding or initializing unused slots.

At complete aligned tiles the copy loop has four times fewer iterations;
this is **not** a measured instruction-count or latency improvement. Confirm
that the HIP compiler emits wide global loads/LDS stores, and check VGPRs,
spills and occupancy. A compiler may split the copies or change scheduling.

Quantization, integer dots, floating-point accumulation/reduction order,
barriers, workgroup size (256 threads), grid, stream and scratch size remain
unchanged. Shared alignment is conditional so scalar rollback instantiations
retain their original alignment. Path admission is unchanged: decode N<=8,
MoE, direct WMMA and F16 sidecars do not use this helper. SSD can benefit only
on projections already selecting TILE8/TILE4; no storage policy or I/O changes.

## Default and rollback

Default-on inside the existing LDS streaming schedule:

```sh
DS4_ROCM_DISABLE_Q4_PREFILL_LDS_VECTOR=1
```

Any defined value, including `0` or empty, restores scalar streaming while
keeping its fence optimization. Unset re-enables the vector-capable helper.
The older `DS4_ROCM_DISABLE_Q4_PREFILL_LDS_STREAM=1` restores the original
loader and barriers and also disables the vector extension. WMMA/F16-owned
projections are unaffected by either flag.

## Verification

```sh
make test-rocm-q4-lds-host
make test-rocm-q4-parity ROCM_ARCH=gfx1151 DS4_TEST_REQUIRE_ROCM_DEVICE=1
```

Use the actual tester architecture in place of gfx1151 when appropriate.
The host test executes the production helper across 13,824 vector/fallback
cases: all four source/destination word alignments, all 1..8 token tails,
every 1..4/1..8 K tail, contiguous and strided sources, and 32/64/256 threads.
Each thread is checked in isolation against an independent block/word map;
output guards and unused slots must stay unchanged. Source allocations end
at the last valid word, also exercising read bounds under ASan. The existing
3,456 scalar cases and 4,096 barrier schedules remain.

The public-runtime GPU oracle now checks all 16 combinations of the two
rollback flags (unset, `1`, `0`, empty), including precedence. It requires
the expected launch counters and bitwise equality of complete output buffers
with guards across dense, paired, K1024 and strided attention cases. The new
counter proves selection of the vector-capable kernel, not that every tile
used wide instructions: device-side alignment fallback remains valid.

Local host tests pass, including ASan/UBSan. The runtime test source passes
host C++ syntax checking; without HIP this does not compile device kernels.
`make test-rocm-q4-parity` reports SKIP because hipcc is unavailable. The global
environment-documentation check retains its pre-existing stale HOME-source
failure. No GPU result or end-to-end percentage gain is claimed.

## Isolated tester A/B

```sh
make rocm-q4-prefill-bench ROCM_ARCH=gfx1151
./speed-bench/rocm_q4_prefill_bench --case lds_vector \
  --tokens 9,16,17,30,33,128,256,512 --sets 4 --samples 16 --warmup 4
```

This compares `lds_stream_scalar` with `lds_stream_vector` at K4096/M1024
and K1024/M32768. Both use the same quantizer, barrier schedule and tiling;
WMMA is disabled. The existing `--case lds` deliberately disables the new
extension on both arms, preserving its original baseline.

The harness rotates resident weight sets, alternates ABBA/BAAB, checks
bitwise outputs and guards before/after timing, and rejects wrong dispatches.
Public-API HIP-event intervals include the unchanged Q8_K quantization and
can include host-submission idle time; they are not matmul-only or SSD-I/O
measurements. Follow with model-output and prefill-latency A/B on the tester's
resident and SSD configurations, changing only the new rollback flag.
