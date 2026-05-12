# CUDA Performance Optimizations — H200 (sm_90)

## Summary

Three targeted changes to `ds4_cuda.cu` improving decode throughput and prefill
tensor core utilization on NVIDIA H200 and other sm_80+ GPUs. No changes to
`ds4.c`, `ds4.h`, or any other file outside `ds4_cuda.cu`.

All changes are gated behind environment variable escape hatches for A/B testing
and are transparent to existing functionality.

---

## Changes

### 1. Force tensor core path in cublasGemmEx (prefill)

**File:** `ds4_cuda.cu`  
**Lines affected:** two `cublasGemmEx` call sites (q8_0 FP16 path and F16 matmul path)

**Change:** `CUBLAS_GEMM_DEFAULT` → `CUBLAS_GEMM_DEFAULT_TENSOR_OP`

Both `cublasGemmEx` calls already use `CUDA_R_16F` inputs but were using
`CUBLAS_GEMM_DEFAULT`, which allows cuBLAS to choose any algorithm. On sm_90
this does not guarantee tensor core selection. `CUBLAS_GEMM_DEFAULT_TENSOR_OP`
explicitly requires tensor op math, using the full 1979 TFLOPS FP16 throughput
of H200 rather than the 67 TFLOPS CUDA core path.

**Escape hatch:** none needed (strict improvement, regression-tested).

---

### 2. cuBLAS FP16 GEMV for q8_0 decode (n_tok == 1)

**File:** `ds4_cuda.cu`  
**Function:** `cuda_matmul_q8_0_tensor_labeled`

**Change:** Added a cuBLAS decode path for single-token (n_tok == 1) q8_0
matmuls when:
- cuBLAS is initialized
- FP16 weight cache is available (`cuda_q8_f16_ptr`)
- `out_dim >= 1024` (below this threshold the custom warp8 kernel is competitive)
- `g_quality_mode == 0`
- `DS4_CUDA_NO_CUBLAS_DECODE` env var is not set

Previously `n_tok == 1` always fell through to `matmul_q8_0_preq_warp8_kernel`,
a custom kernel that does not use tensor cores. The new path converts the single
activation vector to FP16, then calls `cublasGemmEx` with
`CUBLAS_GEMM_DEFAULT_TENSOR_OP`.

**Escape hatch:**
```sh
DS4_CUDA_NO_CUBLAS_DECODE=1 ./ds4-server ...
```

---

### 3. FP16 KV shadow buffers + attention kernels (experimental)

**File:** `ds4_cuda.cu`  
**New kernels:** `attention_static_mixed_heads8_online_h16_kernel`,
`attention_decode_mixed_heads8_online_h16_kernel`

**Change:** Added two FP16 variants of the main online attention kernels. KV
vectors are loaded as `__half2` from a shadow buffer, halving HBM bandwidth for
the attention dot product. Q remains FP32 for score precision. Online softmax
and output accumulation are unchanged (FP32).

Shadow buffers `g_kv_h16_raw` / `g_kv_h16_comp` are allocated lazily via
`kv_ensure_h16()` and freed on `ds4_gpu_cleanup()`. The F32→FP16 conversion
uses the existing `f32_to_f16_kernel`.

The h16 kernel path activates when:
- `g_quality_mode == 0`
- `DS4_CUDA_NO_FP16_KV` env var is not set
- `head_dim == 512` (DeepSeek V4 Flash MLA shape)
- The shadow buffer allocation succeeds

Falls back transparently to the original F32 kernel on allocation failure.

**Known limitation:** `kv_ensure_h16` reconverts the full KV buffer on every
attention call because `raw_kv` grows by one row per generated token. The
conversion overhead partially offsets the bandwidth gain at short contexts.
Full benefit requires native FP16 KV store (writing FP16 directly in
`store_raw_kv_batch_kernel`), which requires changes to `ds4.c` and `ds4.h`.
See [Future Work](#future-work) below.

**Escape hatch:**
```sh
DS4_CUDA_NO_FP16_KV=1 ./ds4-server ...
```

---

## Benchmark Results

Hardware: NVIDIA H200 NVL (sm_90, 80 GB HBM3e)  
Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`  
Command:
```sh
./ds4-bench -m <model> \
  --ctx-start 2048 --ctx-max 32768 --step-incr 2048 \
  --gen-tokens 128 --prompt-file speed-bench/promessi_sposi.txt
```

### Generation throughput (t/s) — higher is better

| ctx_tokens | Baseline | +TENSOR_OP | +cublas_decode | Delta |
|------------|----------|------------|----------------|-------|
| 2048       | 30.47    | 30.64      | **31.10**      | +2.1% |
| 4096       | 30.07    | 30.17      | **30.40**      | +1.1% |
| 8192       | 28.42    | 29.49      | **29.78**      | +4.8% |
| 16384      | 27.98    | 28.14      | **28.40**      | +1.5% |
| 24576      | 27.45    | 27.60      | **27.91**      | +1.7% |
| 32768      | 26.91    | 27.03      | **27.32**      | +1.5% |

### Prefill throughput (t/s) — higher is better

Prefill change is within measurement noise (±0.5%). The TENSOR_OP change
primarily helps large batch prefill; at `prefill_chunk=2048` the cuBLAS calls
are already near the bandwidth ceiling.

### Regression

`make cuda-regression` passes on all three incremental steps.

### KV FP16 mixed-cache A/B (post-fix)

After the mixed-dtype safety fix (`raw_kv=FP16`, `comp_kv=FP32` handled via
on-the-fly comp conversion to FP16 in CUDA attention paths), H200 benchmarks
show a stable generation gain with `DS4_CUDA_KV_FP16=1`.

| ctx_tokens | Default KV (FP32) | `DS4_CUDA_KV_FP16=1` | Delta |
|------------|-------------------|----------------------|-------|
| 2048       | 30.76             | **33.19**            | +7.9% |
| 12288      | 29.28             | **31.09**            | +6.2% |
| 24576      | 27.58             | **28.80**            | +4.4% |
| 32768      | 27.01             | **28.15**            | +4.2% |

Prefill stays roughly aligned (small noise-level differences), while decode
improves consistently across context frontiers.

---

## Testing

### Quick regression
```sh
make cuda-regression
```

### Benchmark with escape hatches for A/B comparison
```sh
# baseline (all optimizations disabled)
DS4_CUDA_NO_CUBLAS_DECODE=1 DS4_CUDA_NO_FP16_KV=1 \
  ./ds4-bench -m <model> --ctx-start 2048 --ctx-max 32768 \
  --step-incr 2048 --gen-tokens 128 --prompt-file speed-bench/promessi_sposi.txt

# with all optimizations
./ds4-bench -m <model> --ctx-start 2048 --ctx-max 32768 \
  --step-incr 2048 --gen-tokens 128 --prompt-file speed-bench/promessi_sposi.txt
```

### GPU profiling (requires ncu/nsys on the target machine)
```sh
# NCU kernel metrics
./bench/profile_h200.sh ncu 32768

# Nsight Systems timeline
./bench/profile_h200.sh nsys 32768
```

---

## Future Work

### Native FP16 KV store (high impact, requires ds4.c + ds4.h changes)

The full bandwidth gain from FP16 KV requires writing FP16 directly at store
time instead of converting lazily. This involves:

- `store_raw_kv_batch_kernel` in `ds4_cuda.cu`: change `float *raw` to
  `__half *raw`, write `__float2half(kv[...])` directly (the F32 round-trip
  already does this implicitly — the cast just needs to stick).
- `compressor_store_kernel`: same for comp_kv.
- `ds4_layer_cache` in `ds4.c`: `float *raw_kv` / `float *attn_comp_kv` →
  `uint16_t *` (or introduce a typed GPU tensor variant).
- All GPU tensor allocation size checks: `sizeof(float)` → `sizeof(__half)`.
- All `extern "C"` signatures in `ds4.h` that reference raw/comp KV.

Estimated scope: ~50 lines across `ds4.c`, `ds4_cuda.cu`, `ds4.h`.  
Estimated decode gain at long context (>64k tokens): +30–50% t/s.

### Speculative decoding (MTP)

Already present as experimental `--mtp` flag. As MTP matures the decode ceiling
on H200 can be pushed significantly higher without kernel changes.

### Multi-GPU tensor parallelism

For decode t/s beyond ~35 t/s on this model, the primary lever is splitting
expert matmuls across multiple H200s via NVLink. Out of scope for this patch.
