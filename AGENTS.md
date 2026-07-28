# CPU Backend Optimization: Multithreading & SIMD/AVX512

## Codebase Structure

The inference engine is a single-file C99 implementation (`ds4.c`, ~28K lines) with a CPU backend compiled via `-DDS4_NO_GPU`. All compute kernels, model loading, and threading primitives live in this file.

TBB wrapper lives in `ds4_tbb.cpp` / `ds4_tbb.h` (C++ shim to expose TBB's `parallel_for` as a C-callable function).

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

## Optimization State (Commits on `cpu-optimization`)

### Done
1. **AVX512 float vector ops** — `dot_f32`, `axpy_f32`, `scale_f32`
2. **AVX512 quantized dot products** — `dot_q8_0_row`, `dot_q8_0_row_2`, `dot_q8_0_row_pair`, `dot_i8_32`, `dot_f16_row`
3. **AVX512 IQ2/Q2 pairs** — `dot_iq2_pair_16`, `dot_q2_16`
4. **TBB integration** — replaces pthread thread pool; C++ shim in `ds4_tbb.cpp`/`.h`, conditional on `-DDS4_USE_TBB`
5. **Parallel attention heads** — `layer_attention_rows_one`, `layer_attention_mixed_one`, `layer_attention_mixed_one_decode_scratch` dispatch head iterations via `ds4_parallel_for_min_rows(DS4_N_HEAD, ...)`
6. **Build system** — `Makefile` adds `-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni`, `-DDS4_USE_TBB`, TBB include/lib paths, compiles `ds4_tbb.cpp` with `g++`

### Not Done (Remaining)
- **SwiGLU activation vectorization** — `silu(gate[i]) * up[i]` loop, trivial AVX512
- **RMS norm / softmax reduction** — parallel reductions
- **K-quant dot products** — `ds4_vec_dot_q4_K_q8_K`, `ds4_vec_dot_q2_K_q8_K` (scalar fallback on x86)
- **Benchmark verification** — no perf testing done yet

## Key Technical Details

- `ds4_parallel_for_min_rows` still guards with `min_parallel_rows` threshold; TBB path drops the `g_parallel_depth` nesting guard since TBB handles nesting natively.
- Attention workers use C99 VLAs for per-thread score arrays (no heap allocation).
- `dot_q2_16` AVX512 path uses `_mm256_cvtepi16_epi8` (AVX512VBMI) for pack+shift+narrow of 2-bit values.
- `f16_to_f32` maps to `_mm512_cvtph_ps` (AVX512F) — no need for separate F16 conversion intrinsics.
- TBB thread count controlled by `DS4_THREADS` env var or `--n-threads` CLI option.

## Makefile

CPU target:
- `AVX512_FLAGS ?= -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni`
- `TBB_HOME ?= /opt/intel/oneapi/tbb/latest`
- `ds4_tbb.o` compiled with `$(CXX) -std=c++17 $(TBB_FLAGS)` (TBB include + `-DDS4_USE_TBB`)
- `ds4_cpu.o` compiled with `$(CC) $(CFLAGS) $(AVX512_FLAGS) -DDS4_NO_GPU -DDS4_USE_TBB` (must match `ds4_tbb.cpp` define)
- Link adds `$(TBB_LDLIBS)` (`-L$(TBB_HOME)/lib -ltbb -lstdc++`)
