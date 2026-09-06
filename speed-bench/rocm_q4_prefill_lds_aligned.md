# ROCm Q4 output-B prefill: aligned Q8_K in LDS

This candidate targets the exact Q8_K + Q4_K TILE8 output-B projection.
It has **no measured speedup yet**. HIP compilation, generated instructions,
GPU parity and end-to-end prefill throughput must be verified on gfx1151.

Local M1 Max validation: default Metal build and CPU build pass; the aligned
host oracle passes strict, fast-math and ASan/UBSan builds (19,200 policies,
1,536 writer-ownership tiles, 1,408 K-loop tiles). The existing packed LDS
oracle also passes. A host shim of the production dot passed 64,000 bitwise
comparisons per compiler mode; this does not emulate HIP or GPU scheduling.
The native target fails explicitly here because hipcc/device is unavailable.
`make environment-docs` remains blocked by the pre-existing stale
`external/system/HOME` source reference; the new rollback is documented in
the curated reference and metadata table.

## Scope and arithmetic

The new layout is selected automatically for resident, non-quality gfx1151
wave32 strided TILE8 with K=8192, M=4096, one group and N=256..4096.
Input/output token strides must describe contiguous rows. Output A retains
its existing WMMA dispatch; standalone matmul reaches this TILE8 path when
WMMA is disabled. Decode, grouped A, pairs, other shapes, SSD and quality
mode retain their existing dispatch.

The external Q8_K scratch stays packed at 292 bytes per block. Only its LDS
copy changes to 304 bytes: `d` at byte 0, 12 padding bytes, `qs` at byte 16,
and `bsums` at byte 272. The 8-token/8-block tile rises from 18,688 to 19,456
LDS bytes. A wave copies each packed block without reading inactive tokens
or K blocks; padding is never consumed. Aligned fixed-size copies expose
four activation words per load to the dot loop.

Q8 quantization, Q4 scales/minima, integer dot order, FP32 accumulation and
reduction, workgroup/grid and streaming barriers retain their source-level
arithmetic. This is not proof of the generated GPU arithmetic or speed.
Inspect wide LDS loads, VGPRs, spills and occupancy; require bitwise GPU
parity before accepting performance results.

## Rollback and verification

`DS4_ROCM_DISABLE_Q4_PREFILL_LDS_ALIGNED=1` restores the existing packed LDS
vector path. Any defined value, including `0` or empty, disables the new
layout. Either older `DS4_ROCM_DISABLE_Q4_PREFILL_LDS_VECTOR` or
`DS4_ROCM_DISABLE_Q4_PREFILL_LDS_STREAM` also prevents it, preserving that
flag's original scalar-streaming or original-loader/barrier behavior.

```sh
make test-rocm-q4-lds-aligned-host
make test-rocm-q4-lds-aligned ROCM_ARCH=gfx1151
make bench-rocm-q4-lds-aligned ROCM_ARCH=gfx1151
./tests/test_rocm_q4_lds_aligned --bench --tokens 4096 --samples 16
```

The host oracle checks staging ownership, payloads, tails and scope. The
native oracle requires gfx1151 wave32 and fails when HIP/device is missing.
It covers N256/257/511/512/1024/2048/4096, zeros, excluded shapes/modes,
rollback precedence, input/output guards and actual dispatch counters. Its
composed attention case checks both A's intermediate and B's output bitwise
while A keeps WMMA enabled in both arms.

The native benchmark defaults to N2048 and N4096; `--tokens` accepts any
N256..4096. File-backed weight spans are uploaded and residency checked before
timing. Four warmups precede balanced ABBA/BAAB samples. HIP events cover
the public output-B Q8_K quantizer and TILE8 operation, potentially including
host-submission gaps; uploads, output resets and readbacks are outside the
interval. Every timed result checks parity, guards and dispatch. These are
projection measurements, not whole-model prefill tokens/s.

## Whole-model acceptance

Build the same ROCm binary and use the identical resident Q4 model for both
arms. Unset both older LDS rollbacks and all profiling flags beforehand.
Run these in separate processes; repeat in ABBA then BAAB order, with unique
CSV names per run, after discarding a warmup for each arm:

```sh
env DS4_ROCM_DISABLE_Q4_PREFILL_LDS_ALIGNED=1 ./ds4-bench --rocm \
  -m /path/aprojq4.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 32768 --step-incr 8192 --prefill-chunk 4096 \
  --gen-tokens 128 --csv q4-rollback.csv
env -u DS4_ROCM_DISABLE_Q4_PREFILL_LDS_ALIGNED ./ds4-bench --rocm \
  -m /path/aprojq4.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 32768 --step-incr 8192 --prefill-chunk 4096 \
  --gen-tokens 128 --csv q4-aligned.csv
```

Keep prompt, context, power and residency settings equal. Require unchanged
frontier logits and generated output in a separate parity run using
`--dump-frontier-logits-dir` and `--show-output`. In a separate diagnostic
run, `DS4_ROCM_Q4_PREFILL_TILE8_STATS=1` reports `lds_aligned_calls`; positive
counts attest enqueue selection, not completion, timing or correctness.
Keep profiling out of throughput runs. An 8% tokens/s gain requires a 7.407%
prefill-time reduction; the projection benchmark alone cannot establish it.
