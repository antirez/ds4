# ROCm Q4 prefill: K64 float4 activation loading

Source change, 2026-09-05, on top of `96f5b463` and the existing uncommitted
decode/Metal candidates. No remote tester was contacted. **HIP compilation,
GPU correctness, generated instructions and performance remain unverified.**
This is a candidate optimization, not evidence of a model t/s improvement.

## Scope

The direct Q4 WMMA K64/P80 kernel can load four adjacent F32 activations per
thread iteration instead of two. It uses the same two `__floats2half2_rn`
conversions and the same two half2 LDS stores; dequantization, WMMA operation
order and output stores are unchanged. On a full 64x64 activation tile,
source-level vector load iterations decrease from 2048 float2 to 1024 float4.
Bytes read, conversion/store count and synchronization do **not** decrease.
Actual load instructions, VGPR pressure and latency require compiler/GPU checks.

The existing 128/256-row geometries retain 8/16 wave32s, 256/512 threads,
10 KiB of LDS, the same launch bounds, stream and scratch allocation. No
six-wave experiment, persistent F16 sidecar or SSD frequency analysis is added.

The normal dispatcher uses the loader only after existing WMMA admission:
gfx1151 wave32, N=256..4096, outside quality mode, and only when K64 already
owns the launch. The activation base must be 16-byte aligned and both token
and group strides divisible by four floats. Incompatible pointers/strides keep
the previous float2/scalar path. K32, 64-row scalar kernels, K128, Q8_K TILE8,
MoE and decode are untouched. SSD retains its separate WMMA opt-in and device
residency contract; this change neither enables WMMA for streamed host weights
nor changes I/O or caching. Attention-output B keeps its existing TILE8 default.

This primarily targets dense projections and grouped attention-output A using
128-row K64. Natural aligned q_b launches normally use K128 or an F16 path,
so they will **not** use this loader unless existing controls select K64.

## Default and rollback

Enabled by default only within the scope above, with a one-flag opt-out:

```sh
DS4_ROCM_DISABLE_Q4_PREFILL_WMMA_K64_LOAD4=1 ./ds4 [your usual arguments]
```

Any defined value, even `0` or empty, disables; unset restores LOAD4. Unlike
the older K128 switch this flag is presence-based. To confirm real-model
coverage, use `DS4_ROCM_Q4_PREFILL_TILE8_STATS=1` in a separate diagnostic run
and inspect `wmma_k64_load4_calls`: nonzero with the new loader, zero on rollback.
That is successful enqueue evidence, not GPU completion or speed evidence.

## Native verification (no Python)

```sh
make test-rocm-q4-wmma-load-host
make test-rocm-q4-prefill-load4 ROCM_ARCH=gfx1151 DS4_TEST_REQUIRE_ROCM_DEVICE=1
```

The host test executes production addressing/admission helpers: 24,480 policy
combinations and 2,048 cooperative maps, all 1..64 token tails, both workgroup
sizes, nonzero token/K/group offsets, padded strides, exact source ends and
64-bit addressing. It checks disjoint LDS writes, zero-filled inactive token
rows, untouched LDS padding and output canaries. It does not emulate HIP
conversion, WMMA or synchronization.

The extended GPU runtime oracle prepares 600 default/rollback cases at
N=9/255/256/257/319 with byte offsets 0/4/8/12/16, K64 row128/256, row64,
K32, K128 and quality fallbacks. It requires exact launch counts and bitwise
whole-allocation parity with the original loader, poisons output before each
call, checks N/M-tail guards, and verifies unchanged input. Inputs include
signed zero, subnormal and FP16 rounding-boundary values. This oracle is
implemented but **not executed on AMD hardware here**.
The focused target fails on a visible non-gfx1151 device rather than skipping
the new kernel and reporting PASS. The broader `test-strix-rocm-q4-prefill`
target also includes these cases alongside the existing Q4 oracles.

Local checks: the host test passes normally and with ASan/UBSan; the existing
LDS and decode host tests still pass. The runtime test passes host C++ syntax
checking (the existing HIP-only unused-helper warning remains). Without `hipcc`
the optional GPU target reports SKIP and the required-device invocation above
fails, as intended; neither is GPU PASS. The Mac `all` build remains current.

## Isolated timing on the tester

```sh
make rocm-q4-prefill-bench ROCM_ARCH=gfx1151
./speed-bench/rocm_q4_prefill_bench --case wmma_load4 \
  --tokens 256,257,319,512,1024,4096 --sets 4 --samples 16 --warmup 4
```

This requires gfx1151 wave32. Explicit enqueue-only hooks compare original
LOAD2 and new LOAD4 with identical K64/P80 staging, independently of the
environment. Unsupported alignment/geometry in the LOAD4 hook fails rather
than silently benchmarking a fallback. Cases cover dense K4096/M1024, grouped
output-A (8 groups, K4096/M1024), output-B K8192/M4096 and q_b K1024/M32768.
Output-B and q_b are kernel diagnostics: ordinary output-B defaults to TILE8,
and ordinary q_b can be owned by K128/F16.

The harness rotates resident weights, uses ABBA/BAAB and reverses each set's
order on the next traversal (use 16 samples for four sets as above). It requires
bitwise results and guards for every set before timing and a final set after
timing. HIP events enclose kernel submission, not quantization, environment
selection, readback, model prefill or SSD I/O. Timings can still include host
submission gaps. Check the generated loads, register use, spills and occupancy
alongside timing; fewer loop iterations per thread is not a speed claim.

Follow with fresh-process A/B/B/A model runs changing only the new opt-out.
Use identical prompt tokens, prefill chunk, cache size and generation settings;
compare greedy output and prefill t/s. Check that LOAD4 is actually used before
attributing any whole-model gain. No percentage gain is currently established.
