# Q8 decode optimizations: CUDA, Metal and ROCm

This change applies two independent CUDA decode optimizations directly to
upstream `6289c516273979173abbc062209a81dd3706b804`. It does not require attention
requantization: a model with Q8 attention can use both paths when its F16
compressor and Q8 shared-expert tensors have the supported shapes. The Q8
activation quantizer and other upstream CUDA kernel dispatch remain unchanged.
Metal and ROCm extend the same projection/state and HC fusion approach using
their existing arithmetic and weight layouts, as described below.

## F16 compressor projections and state store

The two F16 projections and compressor ring write share one kernel. A
transposed half2 weight layout coalesces paired loads and prefetches eight
values per lane, preserving the canonical contiguous K chunks and ascending
32-lane sum. These kernels originate in `d12f4480`/`ff749b84` and were restored
in `4a87d9c`; this branch ports that restoration without the attention-Q4 series.

The automatic path requires a single GB10, K=4096, and one of these shapes:

| Output width | Compressor ratio |
| --- | --- |
| 256 or 1024 | 4 |
| 512 | 128 |

APE weights may be F16 or F32. Quality mode and other devices/shapes keep the
upstream path. Invalid buffers and aliases are checked before a writer is
submitted. The immutable transpose cache is bounded to 1 GiB, built outside
graph capture, and released with the model after invalidating captured
consumers. A cache miss during capture or an allocation/capacity refusal uses
the fused canonical-layout kernel. A submitted writer is never retried.

## Aligned Q8 shared-expert projection with HC expansion

The Q8 down projection and Hyper-Connection expansion reuse the existing
aligned model artifact on a single GB10, with K divisible by 1024 and output
width divisible by 128. Aligned int4 loads replace packed GGUF weight loads;
the upstream Q8_0 activation quantizer, F32 scales, reduction order and HC
epilogue are retained. No additional weight cache is allocated. Missing
artifacts use the upstream packed-weight path; graph capture requires warmed
scratch storage.

## Validation

Without a GPU:

```sh
make test-cuda-f16-compressor-host test-cuda-q8-hc-aligned-host
```

The tests extract production functions. F16 checks 72 input/position/type
cases in both contraction modes, including state rows, canaries and extreme
positions, plus 106 cache/dispatch fault checks. Q8 checks 40 kernel cases,
8192 independent integer-dot comparisons and 31 dispatch/fault cases in
strict and fast builds. Host tests use ASan/UBSan; they establish indexing,
arithmetic and host policy properties, not CUDA synchronization correctness.

On GB10, compile the unchanged native kernel bodies and check fresh-input
CUDA graph replays:

```sh
make test-cuda-f16-compressor test-cuda-q8-hc-aligned CUDA_ARCH=sm_121
make bench-cuda-f16-compressor CUDA_ARCH=sm_121
```

Native CUDA compilation and execution have not been verified on the local
Mac. No throughput percentage is claimed. The F16 benchmark excludes cache
preparation and reports eight balanced rounds of separate, fused canonical
and fused transposed kernel timings.

For model measurements, compare this branch against its upstream base with
the same Q8 GGUF, prompt, context, sampling settings and generated-token count.
Alternate binary order and verify output text. Report startup/cache costs
separately from steady decode. A same-binary reference run can disable both
paths with `DS4_CUDA_NO_F16_PAIR_COMPRESSOR_STORE=1` and
`DS4_CUDA_NO_Q8_FUSED_ALIGNED=1`. Both paths are enabled automatically when
eligible; the existing upstream rollback controls are also respected.

## Metal: reduce synchronization in existing fusions

Metal already fuses F16 compressor projections with the state write, and Q8
projections with HC expansion. The F16 pair, quad, and Q8 QKV-plus-quad kernels
now reduce both F16 projections through separate threadgroup-memory planes.
Partial sums and zero padding have disjoint writers, so one threadgroup
barrier replaces five threadgroup barriers and one device barrier. Final FP32
values write both the diagnostic outputs and the ring state, avoiding output
reloads. Pair/quad threadgroup storage grows from 256 to 512 bytes (512 to
1024 bytes for a four-row pair); the compound kernel already reserves 1024.

The shared-expert down+HC, general Q8+HC and vector-HC kernels use the same
disjoint ownership for partial sums and padding, removing one of their two
threadgroup barriers. Products, reduction trees, FP32 activations and HC
epilogues are unchanged. Ordinary matvecs remain numerical references. The
existing device/shape eligibility and automatic dispatch remain unchanged;
there is no new cache, repacking, or opt-in flag.

On a Mac with a Metal GPU:

```sh
make test-metal-decode-fusions
```

The native oracle builds the actual production shader functions and checks
all six fused entry points in strict and fast math modes against ordinary
matvecs followed by a separate state write or HC expansion. It checks bitwise
outputs and ring states, APE types, ratio/position boundaries, padded strides,
row tails and guard regions. This is a component correctness test, not an
end-to-end Q8 model throughput measurement.

Local validation on Apple M1 Max passed 68 cases in each math mode. A separate
integration check through the real backend's quad API passed 64 cases (512
bitwise tensor comparisons), including batched command encoding, nonzero
tensor/model offsets, both APE types and position wrap boundaries. The complete
Metal shader library also compiled and initialized successfully on that GPU.

A local fast-math microbenchmark compared the fused shaders from `4450d1f`
with the updated shaders on M1 Max. It used identical inputs, the respective
old/new threadgroup-memory sizes, one warmup round and eight rounds alternating
version order. GPU command-buffer time was divided by 256 repeated dispatches
for F16 and 2048 for Q8 HC. Every sample also checked outputs and state against
the separate numerical reference.

| Kernel (K / output width) | Old median, us | New median, us | Median paired new/old time |
| --- | ---: | ---: | ---: |
| F16 pair (4096 / 512) | 33.418 | 26.868 | 0.799 |
| F16 quad (4096 / 1024 + 256) | 63.660 | 48.591 | 0.761 |
| Q8 shared HC (2048 / 4096) | 38.409 | 38.248 | 1.003 |
| Q8 general HC (4096 / 4096) | 59.803 | 59.255 | 0.984 |
| Q8 vector HC (4096 / 4096) | 57.574 | 57.628 | 0.999 |

The F16 kernels were faster in all eight rounds, with about 20% and 24% lower
paired median time. Q8 HC showed no stable measurable improvement: paired
ratios crossed 1.0 for all three kernels. Absolute medians and paired ratios
are calculated independently. These repeated-dispatch measurements use warm
weights and do not establish a model token/s gain, or performance on M3/M5.
The direct shader benchmark also exercises kernels whose existing host gates
may select other paths on M1.

The [reproducible benchmark and real-decode profiler](METAL_DECODE_PROFILING.md)
now ship in the repository. They compare actual baseline/current shader
sources, save raw samples and provenance, and identify F16 versus compound
dispatches in the model's existing Metal timeline. This addresses the missing
reproduction tool noted in the [external M5 Max report](https://github.com/evanwtf/local-llm/issues/274#issuecomment-5608344436),
which passed the bitwise tests but found no clear end-to-end speedup.

## ROCm: fuse the F16 compressor state write

On gfx1151, the existing shared-X F16 pair kernel now has a variant that also
writes the selected compressor state row and adds APE to its score. This
eliminates the separate store launch and its projection reloads. Lane-strided
products, the eight-step unroll order, wave32 reduction, launch geometry and
shared-X allocation are identical to the existing pair kernel; CUDA's
transposed layout and different reduction are not used.

The automatic path requires K=4096 and the same width/ratio combinations in
the CUDA table above, with F16 or F32 APE. Quality mode, graph dumps, other
devices/shapes and other APE formats retain the previous path. Buffer/range
validation and overlap checks precede the writer. A failure after submission
is fatal, so a fallback cannot repeat a partially submitted state write.
There are no additional allocation paths, new caches, or new flags. ROCm's
existing Q8+HC kernel is unchanged by this extension.

```sh
make test-rocm-f16-compressor-host
# On Strix Halo with HIP and a visible GPU:
make test-rocm-f16-compressor ROCM_ARCH=gfx1151
make bench-rocm-f16-compressor ROCM_ARCH=gfx1151
```

The source-derived host oracle checks 216 arithmetic/state cases in each of
two contraction modes under ASan/UBSan, plus 87 dispatch and failure checks.
The HIP fixture compares the unchanged production kernel bodies in eager
execution and fresh-input graph replays; its benchmark alternates the old
pair-plus-store and fused paths in eager and graph execution. These are direct
HIP kernel graph tests; the ROCm backend's model-level graph API remains
unsupported. Native HIP compilation, execution and performance require an AMD
host and have not been verified on the local Mac.
