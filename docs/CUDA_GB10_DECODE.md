# GB10 decode optimizations

This change applies two independent CUDA decode optimizations directly to
upstream `6289c516273979173abbc062209a81dd3706b804`. It does not require attention
requantization: a model with Q8 attention can use both paths when its F16
compressor and Q8 shared-expert tensors have the supported shapes. The Q8
activation quantizer and all other upstream kernel dispatch remain unchanged.

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
