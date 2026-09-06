# Q4 transient prefill dequantization: CUDA / ROCm

Candidate only: no GPU speedup or closed Q4/Q8 gap has been demonstrated.
The local Mac has neither a usable CUDA toolkit nor hipcc. CPU layout tests
and kernel-body simulation are not CUDA/HIP compilation or device proof.
No remote tester was contacted.

## Change and scope

The existing transient `attn_q_b` path expands a 1024x32768 Q4_K matrix into
64 MiB of F16 weights before GEMM. The old producer assigns 16 values per
thread and uses scalar byte loads/half stores with runtime row division.
The new shared CUDA/HIP kernel keeps those 16 values/thread, the grid and
stream, but addresses the contiguous Q4 blocks with shifts/masks, loads
one aligned `uint4` of packed bytes and publishes two `uint4` output vectors.
Scale/min metadata are decoded once per thread. It adds no scratch, barrier,
allocation, host/device copy or matrix cache; output size is unchanged.
Actual vector codegen, spills and performance must be inspected on the GPU.

Default admission: K=1024, M=32768, 256..8192 logical tokens, physically
resident weights, 16-byte-aligned disjoint input/output, outside quality and
SSD modes. CUDA is restricted to a single GB10; ROCm to gfx1151 wave32, with
a per-thread current-device capability cache shared with the Q4 epilogue.
Existing transient-path validation, memory ownership, circuit breakers and
GEMM ordering remain in force. With default thresholds the target is normally
4096-token chunks; this change does not lower the GEMM threshold.

Persistent sidecar prewarm and cache hits, direct Q4 MMQ/WMMA, Q8 and decode
are unchanged. GEMM accumulation/output type, optional F16-output behavior
and RMSNorm/RoPE selection are unchanged. All scalar fallback kernels remain
untouched. Misaligned/out-of-scope calls retain the old producer.

The new kernel uses the same F32 dequantization expression followed by a
single RN-to-F16 conversion. There is no intentional numerical change, but
compiler contraction/reassociation under production fast-math is part of
the GPU parity gate. Do not infer GPU bitwise parity from source expressions.

## Rollback

- CUDA: `DS4_CUDA_DISABLE_Q4_PREFILL_DEQUANT_VEC=1`.
- ROCm: `DS4_ROCM_DISABLE_Q4_PREFILL_DEQUANT_VEC=1`.

Both are presence-based: any defined value, **including `0`**, disables
the candidate. Unset the relevant flag for default-on behavior. CUDA reads
it at initialization; ROCm checks it at transient dispatch. Use separate
processes. Earlier Q8 producer and Q4 epilogue rollbacks are independent;
hold them fixed when comparing this change.

## Native tests and timing

```sh
make test-q4-prefill-dequant-host
make test-cuda-q4-prefill-dequant CUDA_ARCH=sm_121
make bench-cuda-q4-prefill-dequant CUDA_ARCH=sm_121
make test-rocm-q4-prefill-dequant ROCM_ARCH=gfx1151
make bench-rocm-q4-prefill-dequant ROCM_ARCH=gfx1151
```

Run the native commands on the corresponding GPU host, with fresh binaries
after compiler/architecture changes. The standalone harness requires only
the native toolkit/runtime, not a model, Python, HIP-CPU or full ds4 linking.
Native targets fail without the compiler/device. The host executable rejects
`--bench`; native timing also rejects devices outside production admission.

The host oracle checks 46080 shape/flag/alignment combinations, disjoint and
adjacent spans, pointer overflow/null rejection, 28560 chunk mappings, all
32768 six-bit metadata pairs across groups, nibbles and half-bit packing.
On hosts with `_Float16`, it also executes the actual vector kernel body
through a serial CPU shim and compares 5963776 half outputs against an
independent scalar calculation, including exceptional values and guards.
This passes strict/default flags, fast-math and ASan/UBSan locally. The shim
does not emulate GPU codegen, CUDA FTZ or performance.

The native oracle includes the production vector kernel and an independent
scalar reference matching the old backend algorithm. It covers partial
CTAs, aligned offsets, metadata/payload patterns, zeros, subnormals, large
values and NaN/Inf cases; finite outputs and infinities must match bitwise,
with only NaN payload differences allowed. It tests the full 1024x32768
matrix, buffer guards, input immutability, non-default streams and graph
replay. The test's scalar reference is not the linked backend entry point:
full-backend build and model checks are required separately.

The benchmark repeats full weight expansion four times per graph over
immutable inputs. Eight warmup calls plus an untimed graph replay precede
12 ABBA/BAAB samples per arm. Events exclude input/reset/readback, include
graph replay overhead; every sample checks output parity and guards. It
prints median/min/max microseconds per expansion, not model TPS. Check
CUDA Compute Sanitizer memcheck/racecheck/synccheck and AMD memory checking
as available, and inspect vector instructions and register spills before
acceptance. A kernel-only win is insufficient.

## Full-model acceptance

Use the usual resident AProjQ4 model, prompt, power policy and internal
prefill chunk (4096), with step increments 8192 and contexts through 32768.
Alternate default and the **dequantization-only** rollback in separate
processes; discard warmup/first frontier and retain at least three measured
runs per arm with distinct CSV names. Compare prefill TPS, steady decode
TPS and non-quality logits/greedy output. Keep a same-commit Q8 baseline.

Do not disable transient F16, enable a persistent sidecar cache, or change
F16-output/epilogue flags between arms. Those settings change or bypass the
work being measured. Confirm the trace contains
`ds4_q4_dequant_f16_vec16_kernel` for the default and the old scalar kernel
for rollback. If the normal model configuration bypasses transient expansion,
report that: do not force a different cache policy and call it a default win.

Local documentation checking still encounters the pre-existing stale
`external/system/HOME` source entry. New controls are recorded in curated
documentation and the metadata table; the generated inventory is not edited
by hand to hide that failure.
