# Branch SSD and Quantized Kernel Notes

[SSD streaming](SSD_STREAMING.md) | [Environment variables](../ENVIRONMENT_VARIABLES.md)

These branch-specific notes describe kernel selection, rollback controls, and
validation limits. Kernel-only checks do not establish end-to-end throughput;
run commands from the repository root.

## Decode and SSD controls

CUDA Q4_K dense MMVQ decode and micro-batches on NVIDIA Turing+ fuse their NaN/Inf-to-zero
sanitizer into the matvec output store on eligible 1–8-token, four-row-aligned
shapes. The matvec fully overwrites those outputs, so both the pre-clear and
the separate sanitizer launch are skipped. The shared-activation pair applies
the same policy independently to each leg. Dot products, reduction order,
warp count, streams and quantization are unchanged; so are N>8 prefill, grouped/MoE
kernels, HIP and the persistent K1024 experiment. Set
`DS4_CUDA_DISABLE_Q4_MMVQ_EPILOGUE=1` to restore the previous path (any defined
value, even `0`, rolls back). `make test-cuda-q4-epilogue CUDA_ARCH=sm_121`
provides the focused GPU/graph oracle; this change has only host validation
locally, with GPU correctness and performance pending tester runs. See the
[test and measurement notes](../speed-bench/cuda_q4_mmvq_epilogue.md).

Metal Q8_0 dense decode matvecs and paired projections also use a single
reduction barrier on eligible non-TP/non-quality 4-simdgroup paths,
in both resident and SSD modes. Only the output-owning simdgroup performs the
final reduction, with the same sum order and launch geometry. Set
`DS4_METAL_DISABLE_Q8_MV_SINGLE_BARRIER=1` to restore their previous kernels.
This does not change prefill or the separate shared-expert SwiGLU policy.
`make bench-metal-q8-mv` runs bitwise checks followed by a resident synthetic
kernel-only A/B; full-model and SSD throughput still need separate tests.
Eight-simdgroup paths retain the legacy kernel after mixed local A/B results;
see [measurement notes](../speed-bench/metal_q8_mv_single_barrier.md).

Metal SSD decode enables two kernel specializations by default on eligible
non-quality, non-TP paths. Q4_K selected experts (six cache slots or
address tables) load activations once for gate/up and compute SwiGLU from
registers; the separate Q8_0 shared-expert gate/up kernel uses one threadgroup
barrier instead of two, with the same reduction order and launch geometry.
The Q8 shared-expert change can apply to both AProjQ4 and AProjQ8 models; the
Q4 change requires Q4_K routed-expert weights. Neither changes SSD reads,
cache policy, prefill, CUDA, or ROCm.

For tester A/B runs, set either rollback independently (any defined value,
including `0`, disables it):

```sh
DS4_METAL_DISABLE_SSD_Q4_PAIR_SHARED_X=1 \
DS4_METAL_DISABLE_SSD_Q8_SINGLE_BARRIER=1 \
./ds4 -m ./ds4flash.gguf --ssd-streaming
```

Unset both variables for the candidate run. Keep model, cache size, prompt,
context, and SSD/cache warmup identical between arms; compare each rollback
separately to attribute changes. `make test-metal-ssd-decode-kernels` runs a
GGUF-free bitwise GPU oracle with guarded buffers. It does not measure SSD or
end-to-end performance; throughput gains still require model-backed A/B tests.

CUDA's raw Q8_0 activation quantizers also use a default-on warp reduction
outside quality mode, for both resident and SSD decode/prefill. The ordinary
and group-slice producers drop six block barriers; the dual Q8_K/Q8_0 producer
drops seven in its Q8_0 phase without changing the Q8_K signed-max reduction.
Launch geometry, stream, rounding, and padding are preserved. This does not
change the separate MMQ Q8_1 path. Use
`DS4_CUDA_DISABLE_Q8_QUANT_WARP_REDUCE=1` in a fresh process to restore the
previous kernels. Test correctness with `make test-cuda-q8-quantize` on a CUDA
host, then measure end-to-end throughput with identical model/cache settings.

Metal SSD+DSpark also has an experimental, support-aware pre-cap for A/B tests.
Set `DS4_METAL_DSPARK_SAFE_EXPERT_COUNT=1` to convert a numeric count to bytes
and cap it, when measurable, after accounting for the target's non-routed
weights, the context/KV estimate, and a 2 GiB active reserve for the mmap-backed
support model. It does not yet price the complete batch-prefill workspace or a
separate routed-prefill transient reserve. Startup reports requested/effective
slots and the support reserve. If this policy cannot measure safe room, it
retains the explicit count with a warning; the normal final memory check remains
authoritative. The experiment does not affect `NGB` budgets or CUDA/ROCm.

## Metal grouped SSD prefill

For Metal IQ2_XXS/Q2_K models, eligible SSD prefill chunks automatically use
grouped address matmuls when the dynamic cache can retain the complete expert
domain. This includes normal 128-token chunks; the automatic range is 32–760
tokens on the 256-expert Flash model and requires, for example,
`--ssd-streaming-cache-experts 256`. Once that material condition is met,
selection is fail-closed by default; no environment prefix is required. Set
`DS4_METAL_ENABLE_IQ2_XXS_SSD_PREFILL_MM=0` or
`DS4_METAL_DISABLE_IQ2_XXS_SSD_PREFILL_MM=1` for the legacy sparse-matvec
rollback. `DS4_METAL_REQUIRE_IQ2_XXS_SSD_PREFILL_MM=0` keeps automatic
selection but permits a fallback, while explicit `REQUIRE=1` also rejects an
insufficient cache. Combining `REQUIRE=1` with `DISABLE=1` fails on an eligible
grouped-MM candidate, while short tail chunks retain their normal fallback. The
IQ2 live cache index remains automatic for its production shape, selected-load
early commit remains off unless explicitly enabled, and grouped-MM statistics
plus streaming timing summaries remain opt-in diagnostics. The grouped prefill
loader also skips `F_RDADVISE` for chunks of at least 32 tokens because it
immediately reads the same expert ranges with parallel `pread`; short chunks
retain the hint. Set
`DS4_METAL_ENABLE_STREAMING_PREFILL_EXPERT_READAHEAD=1` to restore the old
hint-plus-read sequence for cold-storage A/B tests.

## ROCm Q4 tiled prefill

The exact ROCm Q4 tiled-prefill kernels now use simpler Q8_K activation
staging and omit the final shared-memory reuse barrier. The change is default
on within the existing TILE8/K1024 TILE4 paths; set
`DS4_ROCM_DISABLE_Q4_PREFILL_LDS_STREAM=1` to restore their previous
loaders/barriers. Decode, WMMA, quantization and tile geometry are unchanged.
Host checks pass, but HIP compilation, GPU parity and performance measurement
remain pending. See [the Q4 LDS tester notes](../speed-bench/rocm_q4_prefill_lds.md).

Aligned tiles additionally use four-word LDS copies, with scalar fallback
for misaligned addresses/strides and partial words. Set
`DS4_ROCM_DISABLE_Q4_PREFILL_LDS_VECTOR=1` to restore scalar streaming while
retaining the improved fence schedule. The older LDS_STREAM rollback disables
both changes. GPU code generation and gains still need AMD validation; see
[the isolated vector-copy A/B](../speed-bench/rocm_q4_prefill_lds_vector.md).
