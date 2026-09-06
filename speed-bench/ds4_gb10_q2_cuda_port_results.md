# GB10 Q2 CUDA port results

This records the Q2 CUDA baseline and the exact kernel work promoted for the
NVIDIA GB10 / DGX Spark path. The starting tree was `b030961` (`main`) and the
model was:

```text
gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
```

The test machine was an NVIDIA GB10 (`sm_121`) with nominal 128 GB unified
memory, driver 580.173.02, and CUDA 13.0. Builds used:

```sh
make cuda-spark
```

## Branch audit and selected work

`origin/mxfp4-m3` is 24 commits ahead of the starting tree, but its net CUDA
backend delta is only four compatibility lines. Its throughput work is in
Metal. The MXFP4 LUT/tile kernels are not directly applicable to this
IQ2_XXS/Q2_K model, and CUDA already has equivalents for several exact Metal
fusions (router top-k/weights, Q/KV RMS+RoPE, compressor pool math, and HC
split/weighted-sum/norm).

The two useful remaining ideas were removal of intermediate materialization and
an exact producer/store epilogue:

1. **Direct Q2 prefill.** The existing dormant direct D2R MMQ now has a
   production dispatch. IQ2 gate/up accumulators remain in registers and
   weighted SwiGLU is quantized directly to Q8_1 for the Q2_K down MMQ. It
   avoids materializing F32 gate, up, and mid tensors.
2. **F16 compressor projection/store.** The established ordered F16 pair
   matvec now optionally writes the compressor state in lane 0 of the same
   kernel. This removes 62 dependent state-store launches per Flash decode
   token (42 ratio-4 attention/indexer calls and 20 ratio-128 attention calls).

Both defaults are limited to GB10. The direct path has shape, residency,
aligned-artifact, streaming, scratch, and token-count fallbacks. Its public MMQ
contract returns `DS4_MMQ_NOT_APPLICABLE` only before any enqueue; negative
results are never retried. It is disabled while graph intermediates are being
dumped because its scratch intentionally overwrites the otherwise-dead
gate/up/mid buffers. The compressor fusion is limited to the three validated
4096-wide Flash/Pro calls: `(ratio,width) = (4,256), (4,1024), (128,512)`.

Rollback switches are:

```text
DS4_CUDA_NO_DIRECT_Q2_PREFILL=1
DS4_CUDA_NO_F16_PAIR_COMPRESSOR_STORE=1
```

The existing ordered-pair rollback switches also dominate the compressor
fusion.

Two experiments were not promoted: exact score split DIM2 was neutral (808.41
prefill tok/s in its initial 2K screen), and a QKV RoPE/FP8/cache-store
prototype preserved output but did not provide a worthwhile speedup.

## Reproducible baseline

The original `b030961` benchmark binary was retained on the author's test
host at `/tmp/ds4-bench-main-baseline` with SHA-256:

```text
6cf857252c1dbbc64add2b94857b3be568615c6e6e863961d4df938ad9f3ec98
```

Four interleaved 2,048-token runs of the ordinary path used
`speed-bench/promessi_sposi.txt`:

| run | prefill tok/s | decode tok/s | steady decode tok/s |
|---:|---:|---:|---:|
| 1 | 810.26 | 17.78 | 17.93 |
| 2 | 809.01 | 17.78 | 17.94 |
| 3 | 815.58 | 17.73 | 17.89 |
| 4 | 807.88 | 17.61 | 17.77 |
| **mean** | **810.6825** | **17.725** | **17.8825** |

## Performance

The final aggregate was measured in ABBA order with 2,048 prefill tokens and
512 generated tokens. The control set both rollback variables; the optimized
arm unset both. The per-run command body was:

```sh
./ds4-bench -m ds4flash.gguf --cuda \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 2048 --step-incr 2048 \
  --gen-tokens 512 --csv OUT.csv
```

| arm/run | prefill tok/s | decode tok/s | steady decode tok/s |
|---|---:|---:|---:|
| rollback 1 | 808.86 | 17.66 | 17.78 |
| default 1 | 829.89 | 17.85 | 17.97 |
| default 2 | 829.70 | 17.84 | 17.96 |
| rollback 2 | 805.29 | 17.48 | 17.60 |
| **rollback mean** | **807.075** | **17.570** | **17.690** |
| **default mean** | **829.795** | **17.845** | **17.965** |
| **gain** | **+2.82%** | **+1.57%** | **+1.55%** |

Isolated screens agree with the attribution:

- Direct prefill: 830.6275 versus 810.6825 tok/s over four interleaved runs,
  **+2.46%**; decode was neutral (+0.06%).
- Compressor/store: 17.935 versus 17.765 decode tok/s and 18.055 versus 17.885
  steady tok/s over a 512-token ABBA screen, **+0.96% / +0.95%**.

Author-local raw artifacts are under `/tmp/ds4-opt/`, notably `final-abba/`,
`final-correct/`, `abba/`, and `compstore/abba/`; these paths are provenance
notes, not required to reproduce the checked-in CSVs.

## Exactness and safety validation

Frontier hashes alone were not used as the decode criterion.

- A 2,782-token prompt with a 128-token greedy decode limit (56 tokens
  emitted through EOS) produced byte-identical stdout and full per-token
  logprob JSON with both defaults on versus both rollbacks. SHA-256:
  `6519499391d81b625344a8335ba10fde02d1a2c4e414eb65e6ee711fa3b37d14`.
- The isolated compressor A/B generated-token/logprob artifact is
  byte-identical with SHA-256
  `9b76a6b7579a192d0aeb04bb48616944313cee805f7cbba158c99655ad657868`.
- The direct prefill frontier JSON was fully parsed and byte-identical,
  including every logit and selected ID, with SHA-256
  `e7b419e8ebbcb6c40a5eccfe8784645d910dbf203df7960500d4d8d133f061a8`.
- `DS4_MMQ_YIND_VERIFY=1` checked all 43 routed layers at 2,048 tokens; every
  layer reported `bad=0/393216` Q8 staging values.
- Small/ragged batches fall back normally; disabling D2R and ordered pair MMQ
  together also completed without an aligned-path error.
- `compute-sanitizer --tool memcheck` completed the fused decode path with
  `ERROR SUMMARY: 0 errors`.
- `make cuda-regression` passed its long-context/top-k smoke.
- `make test CUDA_ARCH=sm_121` passed in full. This included a 30,474-token
  long-context run (direct batches at 4,096 and the final ragged tail), five
  logprob vectors, exact tensor-equivalence vectors, local golden vectors,
  server tests, sampling tests, and all ordinary unit tests.
- Two independent source/safety reviews found no release blocker after the
  scratch-overflow, tri-state, debug, target-admission, and fallback-contract
  guards were applied.

Nsight Compute hardware counters were unavailable on this system because
`RmProfilingAdminOnly=1`; performance conclusions therefore use controlled
wall-clock ABBA measurements plus exact output and sanitizer validation.

## Decode phase: approximately 20 steady tok/s

A second profiling/optimization pass targeted the remaining decode path. Nsight
Systems showed that the token was dominated by exact, bandwidth-bound Q8 and
compressor projections. Four changes were promoted:

1. **No CUDA mid-token split synchronization.** Metal's four-layer split
   asynchronously commits a command buffer, but the CUDA implementation of the
   same flush calls `cudaDeviceSynchronize()`. CUDA now defaults the split to
   zero; Apple, ROCm, and CPU builds retain four. The existing
   `DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS` override remains available.
2. **Coalesced exact F16 compressor pairs.** On first use, each validated
   compressor KV/score weight pair is repacked into an interleaved chunk-32
   layout. Every lane still accumulates its original contiguous 128 values and
   lane 0 performs the same ordered sum, but weights at a given iteration are
   coalesced. An eight-value load prefetch preserves the arithmetic chain.
   The transpose is enabled only on the validated single-GB10 path. The cache
   is device-qualified, released on map change/cleanup, refuses graph capture
   allocation, and permanently falls back after an allocation failure.
3. **Aligned Q8 fused consumers.** Decode Q8 pair, HC-expand, and grouped
   attention-A kernels now consume the already-built aligned Q8 artifacts
   directly instead of returning to misaligned raw 34-byte blocks. DP4A term
   order, per-lane accumulation, warp reduction, and all epilogues are
   unchanged. Admission is limited to one validated GB10, exact aligned
   artifacts, full tensors, and dimensions divisible by the proven tile sizes;
   multi-GPU and unsupported shapes fall back to the raw kernels.
4. **Persistent aligned Q8 projections.** The K=1024 Q-b projection and K=4096
   vocabulary projection use eight row warps per persistent CTA on one GB10. The
   K=1024 kernel hoists the immutable activation into registers. Lane/block
   assignment and the float warp tree remain identical; integer dot regrouping
   is overflow-safe. The generic aligned kernel remains the fallback.

New CUDA rollback switches are:

```text
DS4_CUDA_NO_F16_PAIR_COMPRESSOR_TRANSPOSE=1
DS4_CUDA_NO_F16_PAIR_COMPRESSOR_TRANSPOSE_PREFETCH8=1
DS4_CUDA_NO_Q8_FUSED_ALIGNED=1
DS4_CUDA_NO_Q8_ALIGNED_PERSISTENT=1
```

`DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=4` is the existing, Metal-named token-split
control; setting it to `4` restores the old blocking split behavior on CUDA.

### Initial ABBA result

The final clean-build comparison used the same 2,048-token prompt frontier and
512 generated tokens. The rollback arm set all five switches above; the default
arm set none.

| arm/run | prefill tok/s | decode tok/s | steady decode tok/s |
|---|---:|---:|---:|
| rollback 1 | 832.25 | 17.98 | 18.10 |
| default 1 | 830.41 | 19.85 | 20.01 |
| default 2 | 829.24 | 19.81 | 19.97 |
| rollback 2 | 832.36 | 17.94 | 18.06 |
| **rollback mean** | **832.305** | **17.960** | **18.080** |
| **default mean** | **829.825** | **19.830** | **19.990** |
| **gain** | **-0.30%** | **+10.41%** | **+10.56%** |

A post-audit clean run measured **19.92 decode / 20.08 steady tok/s**. Using
the ABBA default means (19.830 decode and 19.990 steady) relative to the
original `b030961` mean, the complete port is **+11.88% decode** and
**+11.79% steady decode**, while retaining a **+2.36% prefill** gain. Two
1,024-generation runs measured 19.73/19.88 and 19.71/19.86 decode/steady as the
attention context grew from 2,048 to 3,072 tokens.

Profiler attribution matched the wall-clock result. The ordered F16 compressor
family fell from about 5.14 ms/token to about 2.96 ms/token. The persistent
K=1024 Q8 kernel fell from roughly 165 us to 150 us per layer, while the aligned
fused Q8 kernels removed the raw-block penalty from pair, HC-expand, and
attention-A projections.

### Pre-target-21 exactness and validation

- A 7,000-byte *Promessi sposi* prompt plus up to 256 greedy generated tokens
  produced byte-identical text and full per-token logprob JSON with all defaults
  versus all five decode rollbacks. SHA-256:
  `128362d060d18e38ebadc7649c18ca8db625f53b5ee3b12762db9859d508d174`.
- A post-audit 64-token default/rollback repeat was also byte-identical, SHA-256
  `3f3d8890e13d4118b6a8964ed9ab3e2d11c8f0e3f8ab718d85a574bcf8cad6e5`.
- `compute-sanitizer --tool memcheck` completed the promoted path with
  `ERROR SUMMARY: 0 errors`.
- A clean `make cuda-spark`, `make cuda-regression CUDA_ARCH=sm_121`, and
  `make test CUDA_ARCH=sm_121` all passed after the final
  safety hardening. The full test again included the 30,474-token context and
  exact tensor-equivalence suite.
- Final source audit added single-device derived-artifact admission, active
  device validation, OOM negative caching, capture and divisibility guards,
  persistent-grid overflow guards, signed-zero preservation, init/reset
  hardening, and direct-MMQ scratch-size overflow admission.

Neutral or regressive experiments were not promoted: grouped top-6 MoE CTAs,
stream-0 graph replay, extending graph island A through Q/KV, small-output F16,
exact-score split LDG/vector variants, compressor CTA grouping, persistent-grid
retuning, and HC RMS-fold continuation.

## README speed-table replication

The upstream README's GB10 sweep was repeated with the final target-only build
(no DSpark support model or speculative flags), using the same 2,048-token
frontiers and 128 greedy generation tokens:

```sh
./ds4-bench -m ds4flash.gguf --cuda \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 \
  --gen-tokens 128 \
  --csv speed-bench/gb10.csv
```

| Context | README prefill | Final prefill | README generation | Final generation | Final steady |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 825.76 | 832.86 | 18.05 | 20.58 | 20.69 |
| 16,384 | 872.44 | 883.81 | 15.10 | 16.80 | 16.81 |
| 32,768 | 855.94 | 865.40 | 14.43 | 15.99 | 16.00 |
| 65,536 | 822.98 | 833.44 | 13.84 | 15.27 | 15.28 |

Generation gains at those four table rows are respectively **+14.02%**,
**+11.26%**, **+10.81%**, and **+10.33%**. The complete 32-frontier sweep is in
`speed-bench/gb10.csv`; benchmark binary SHA-256 was
`04d5321402dc073b1f0300a19063e95262a81cc20831fa4f083cf9147fcc145f`.

## Target-21 follow-up

A review of `eugr/spark-vllm-docker` at commit
`e5f3cf9e5320d9a424966a801570bf452405d122` led to the referenced B12X SM121
kernel package at `7cecbb2c4819636ae7f05f8b116f2c45ee2cff7b`. The applicable
ideas were stable caller-owned scratch (no per-token asynchronous allocation),
the split/parallel MHC decode structure, and measured GB10 launch geometry.
Its tensor-core FP4/FP8, sparse-MLA, and MoE scheduler machinery is not a
drop-in for this IQ2_XXS/Q2_K/F32-exact path.

Two exact changes were promoted:

1. The dense aligned-Q8 wrapper now prefers the already-owned 256 KiB aligned
   Q8_1 scratch allocation instead of recording a per-token `cudaMallocAsync`
   pool node. Rollback: `DS4_CUDA_NO_Q8_ALIGNED_DENSE_SCRATCH=1`.
2. The single-row 4K HC weighted-sum + RMS kernel now uses 16 partial CTAs,
   a one-CTA exact reduction replay, and 16 store CTAs. The reduction retains
   the original ascending-column FMA chain and 256-lane shared tree. Rollback:
   `DS4_CUDA_NO_HC_SPLIT_NORM_SPLIT4096=1`.

Matched 512-generation AB results at 2,048 context:

| Variant | Steady runs | Median steady |
|---|---:|---:|
| Both target-21 rollbacks | 19.78, 19.89 | 19.835 t/s |
| Aligned scratch only | 20.68, 20.69 | 20.685 t/s |
| Final 16-CTA HC split | 21.01, 21.06 | 21.035 t/s |

A 1,024-generation run measured 20.73 decode and 20.89 steady t/s as context
grew from 2,048 to 3,072 tokens. The full 32-frontier sweep in
`gb10.csv` was rerun with this final build.

The 7,000-byte prompt plus 256 greedy generated tokens remained byte-identical
to the HC rollback, including full per-token logprob JSON. SHA-256:
`952799babb7f421cb0e2e75e6ede9de73c40304da279ef1d8d99042ef684be62`.
`make test CUDA_ARCH=sm_121`, `make cuda-regression CUDA_ARCH=sm_121`, and
`compute-sanitizer --tool memcheck` all passed; sanitizer reported
`ERROR SUMMARY: 0 errors`.

## Post-rebase validation

The branch was rebased onto `upstream/main` commit
`84cc882352757baf628a1776badf7cc54d584e28` and retested. The rebased CUDA
build passed `make cuda-spark`, `make test CUDA_ARCH=sm_121`, and
`make cuda-regression CUDA_ARCH=sm_121`. Compute Sanitizer reported
`ERROR SUMMARY: 0 errors`. The 7,000-byte prompt plus 256 greedy
tokens remained byte-identical to the rollback logprobs, SHA-256
`952799babb7f421cb0e2e75e6ede9de73c40304da279ef1d8d99042ef684be62`.

Same-day 512-generation runs after the rebase measured 20.69, 20.65, and 20.74
steady tok/s at 2,048 context. A preserved pre-rebase worktree measured 20.67
and 20.67 steady tok/s in the same thermal window, so the rebase itself is
performance-neutral within noise. The full sweep and README table above were
refreshed with the rebased build.
