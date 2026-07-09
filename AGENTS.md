# CPU Backend Optimization: Multithreading & SIMD/AVX512

## Codebase Structure

The inference engine is a single-file C99 implementation (`ds4.c`, ~28K lines) with a CPU backend compiled via `-DDS4_NO_GPU`. All compute kernels, model loading, and threading primitives live in this file.

## Key Dimensions (Flash / Pro)

| Param | Flash | Pro |
|-------|-------|-----|
| `DS4_N_EMBD` | 4096 | 7168 |
| `DS4_N_HEAD` | 64 | 128 |
| `DS4_N_HEAD_DIM` | 512 | 512 |
| `DS4_N_FF_EXP` | 2048 | 3072 |
| `DS4_N_ROT` | 64 | 64 |
| `DS4_N_HC` | 4 | 4 |
| `QK_K` | 256 | 256 |
| `DS4_N_HEAD_KV` | 1 | 1 |
| Layers | 43 | 61 |
| Experts | 256 | 384 |
| Experts used | 6 | 6 |

## Existing Multithreading

A custom pthread-based thread pool exists (`ds4_thread_pool` at ds4.c:1317) with up to 32 workers. Key API:
- `ds4_parallel_for(n_rows, fn, ctx)` — parallel dispatch with 512-row minimum
- `ds4_parallel_for_min_rows(n_rows, fn, ctx, min)` — configurable minimum

**Problems:**
1. Simple block partitioning — no work-stealing, leading to load imbalance
2. `g_parallel_depth` (per-thread TLS) prevents nested parallelism entirely
3. Many hot paths bypass the pool entirely (see below)

## Existing SIMD

**ARM NEON only** — no x86 SIMD at all. All x86 code paths use plain scalar C loops.

NEON is used in these hot functions:
- `dot_f32`, `axpy_f32`, `scale_f32` — float vector ops
- `dot_f16_row` — F16 dot product
- `dot_q8_0_row`, `dot_q8_0_row_2`, `dot_q8_0_row_pair` — Q8_0 quantized dot products
- `dot_i8_32` — int8 dot product (used as fallback for non-32-byte-aligned sizes)
- `dot_iq2_pair_16`, `dot_q2_16` — IQ2/Q2 quantized dot products
- `ds4_vec_dot_q4_K_q8_K`, `ds4_vec_dot_q2_K_q8_K` — K-quant dot products
- Quantize/Softmax helpers

## Serial Bottlenecks (Not Parallelized)

### Attention — Head Loop (line ~8930)
```c
for (uint32_t h = 0; h < DS4_N_HEAD; h++) {  // 64 or 128 iterations
    // dot_f32(qh, kv_row, 512) for each KV row
    // softmax (serial)
    // axpy_f32 weighted sum
}
```
This runs **serially** for single-token decode. With 64 heads × 512 dim, this is ~64 × 512 × n_kv FMA ops done one head at a time — a prime candidate for parallelization across heads.

### SwiGLU Activation (line ~7170)
```c
for (uint64_t i = 0; i < n; i++) {
    out[i] = silu(gate[i]) * up[i];  // element-wise, no vectorization
}
```
Runs serially on x86; trivial to vectorize with AVX512.

### K-Quant Dot Products (Q2_K, Q4_K)
Scalar fallback on x86; heavy inner loops over QK_K=256 elements.

## AVX512 Availability

The build server supports:
- AVX512F (foundation)
- AVX512BW (byte/word)
- AVX512DQ (double-word/quad-word)
- AVX512VL (vector length)
- AVX512VNNI (neural network int8 dot product)
- AVX512VBMI (bit manipulation)
- AVX512BF16 (bfloat16)

All available via `-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni` flags.

## TBB Availability

Intel TBB (oneAPI) is available at `/opt/intel/oneapi/tbb/latest/`. Includes:
- `oneapi/tbb/parallel_for.h` — parallel iteration with automatic work-stealing
- `oneapi/tbb/task_arena.h` — thread binding and concurrency control
- `oneapi/tbb/global_control.h` — thread count control

## Optimization Strategy

### 1. Add AVX512 SIMD Intrinsics
Float vector ops, int8 dot products, F16 conversion/dot, quantized block dot products.

### 2. Replace Thread Pool with TBB
Better work-stealing, automatic nesting support, hardware-aware partitioning.

### 3. Parallelize Attention Heads
Split the decode-time attention head loop across threads.

### 4. Vectorize Element-Wise Ops
SwiGLU activation, RMS norm, softmax reduction.

## Makefile Changes

Add to `CFLAGS`:
```
-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni
```

Add to `LDLIBS` and link:
```
-I/opt/intel/oneapi/tbb/latest/include
-L/opt/intel/oneapi/tbb/latest/lib
-ltbb
```
