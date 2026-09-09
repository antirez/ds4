# Vendored llama.cpp mmq kernels

This directory contains source files copied verbatim from
[llama.cpp's `ggml-cuda` backend](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda),
plus a thin ds4-side adapter (`ds4_ggml_stubs.{h,cu}` and `ds4_mmq.{h,cu}`)
that lets the templated CUDA kernels compile and link without the full ggml
runtime.

## Why these files are vendored, not submoduled

ds4 is a flat, self-contained C/CUDA codebase with no third-party build
dependencies. A submodule would force consumers to clone, configure, and
build all of llama.cpp just to use a few quantized-matmul kernels. Vendoring
keeps ds4 self-contained at the cost of a periodic re-sync.

## Upstream pin

| Field         | Value                                                                                                              |
|---------------|--------------------------------------------------------------------------------------------------------------------|
| Source        | https://github.com/ggml-org/llama.cpp                                                                              |
| Commit        | `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`                                                                         |
| Commit date   | 2026-05-14                                                                                                         |
| Commit title  | `ggml-hexagon: cpy: add contiguous fast-path in reshape copy (#23076)`                                             |
| License       | MIT (`LICENSE` at the repository root, copyright "2023-2026 The ggml authors"). Compatible with ds4's MIT license. |

## File inventory

| File                  | Origin in llama.cpp                          | Status                                                                   | Lines |
|-----------------------|----------------------------------------------|--------------------------------------------------------------------------|-------|
| `mmq.cuh`             | `ggml/src/ggml-cuda/mmq.cuh`                 | verbatim                                                                 |  4176 |
| `mma.cuh`             | `ggml/src/ggml-cuda/mma.cuh`                 | verbatim                                                                 |  1456 |
| `vecdotq.cuh`         | `ggml/src/ggml-cuda/vecdotq.cuh`             | verbatim                                                                 |  1317 |
| `quantize.cuh`        | `ggml/src/ggml-cuda/quantize.cuh`            | verbatim                                                                 |    41 |
| `quantize.cu`         | `ggml/src/ggml-cuda/quantize.cu`             | verbatim                                                                 |   443 |
| `mmid.cuh`            | `ggml/src/ggml-cuda/mmid.cuh`                | verbatim                                                                 |     5 |
| `mmid.cu`             | `ggml/src/ggml-cuda/mmid.cu`                 | verbatim                                                                 |   164 |
| `mmvq.cuh`            | `ggml/src/ggml-cuda/mmvq.cuh`                | patched (Step 6): `mul_mat_vec_q_switch_type` proto exposed; ggml-tensor entries gated on `DS4_MMVQ_INCLUDE_GGML_ENTRIES` | ~36 |
| `mmvq.cu`             | `ggml/src/ggml-cuda/mmvq.cu`                 | patched: `mul_mat_vec_q_switch_type` promoted from `static`; `ggml_cuda_mul_mat_vec_q` + `ggml_cuda_op_mul_mat_vec_q` gated on `DS4_MMVQ_INCLUDE_GGML_ENTRIES` | 1163 |
| `unary.cuh`           | `ggml/src/ggml-cuda/unary.cuh`               | verbatim (needed by `mmvq.cu` for inline GLU epilogues)                  |   114 |
| `common.cuh`          | `ggml/src/ggml-cuda/common.cuh`              | verbatim                                                                 |  1489 |
| `ggml-common.h`       | `ggml/src/ggml-common.h`                     | verbatim                                                                 |  1900 |
| `vendors/cuda.h`      | `ggml/src/ggml-cuda/vendors/cuda.h`          | verbatim                                                                 |    28 |
| `ggml.h`              | (new)                                        | redirect to `ds4_ggml_stubs.h`                                           |     5 |
| `ggml-impl.h`         | (new)                                        | redirect to `ds4_ggml_stubs.h`                                           |     5 |
| `ggml-cuda.h`         | (new)                                        | redirect to `ds4_ggml_stubs.h`                                           |     5 |
| `ds4_ggml_stubs.h`    | (new)                                        | shim: ggml_type enum, macros, info struct, type_size lookups             | ~280 |
| `ds4_ggml_stubs.cu`   | (new)                                        | shim impls: `ggml_cuda_info`, naive pool, `ggml_backend_cuda_context::*` | ~110 |
| `ds4_mmq.h`           | (new)                                        | public C ABI for ds4 to call                                             |  ~70 |
| `ds4_mmq.cu`          | (new)                                        | host wrappers; Phase 0 instantiates `mul_mat_q_case<Q8_0>` only          | ~120 |

**Total vendored:** ~11,000 lines of CUDA. **Total shim/adapter:** ~600 lines.

## What llama.cpp's `mmq.cu` does that we don't vendor

The upstream `ggml/src/ggml-cuda/mmq.cu` (372 lines) is the ggml-backend
dispatch entry. It talks to `ggml_tensor` / `ggml_backend_buffer_get_usage`,
does its own activation Q8_1 quantization, and wires into the ggml op graph.
We **don't vendor it.** Instead `ds4_mmq.cu` provides ds4-style entries
that bypass the ggml graph and call the per-type `mul_mat_q_case<T>` directly.

## Symbol-resolution table

Symbols the vendored files reference, and how they resolve in this directory:

| Symbol category                                | Resolution                                                                                       |
|------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `GGML_ASSERT`, `GGML_ABORT`, `GGML_UNUSED`     | Macros in `ds4_ggml_stubs.h`                                                                     |
| `GGML_PAD`, `GGML_UNUSED_VARS`                 | Macros in `ds4_ggml_stubs.h`                                                                     |
| `GGML_CUDA_CC_*` constants                     | Defined in vendored `common.cuh`                                                                 |
| `GGML_TYPE_*` enum values                      | Defined in `ds4_ggml_stubs.h::enum ggml_type`                                                    |
| `GGML_CUDA_MAX_DEVICES`, `GGML_CUDA_NAME`      | Macros in `ds4_ggml_stubs.h`                                                                     |
| `GGML_MAX_DIMS`, `GGML_MAX_SRC`                | Macros in `ds4_ggml_stubs.h`                                                                     |
| `block_q8_0`, `block_q2_K`, `block_iq2_xxs`    | Defined in vendored `ggml-common.h` (gated by `GGML_COMMON_IMPL_CUDA`)                           |
| `ggml_half`, `ggml_half2`                      | `uint16_t` typedefs in `ds4_ggml_stubs.h`                                                        |
| `ggml_type_size`, `ggml_blck_size`             | Inline implementations in `ds4_ggml_stubs.h`                                                     |
| `ggml_cuda_info()`                             | Singleton in `ds4_ggml_stubs.cu`, populated via `cudaGetDeviceProperties`                        |
| `ggml_cuda_get_device`, `ggml_cuda_set_device` | Thin wrappers in `ds4_ggml_stubs.cu`                                                             |
| `ggml_time_us`                                 | `std::chrono::steady_clock` in `ds4_ggml_stubs.cu`                                               |
| `ggml_cuda_pool`, `ggml_cuda_pool_alloc`       | Defined in vendored `common.cuh`. Concrete `ds4_naive_pool` subclass in `ds4_ggml_stubs.cu`      |
| `ggml_backend_cuda_context`                    | Defined in vendored `common.cuh`. `new_pool_for_device` and `~dtor` provided in `ds4_ggml_stubs.cu` |
| `ggml_tensor`                                  | Forward declaration in `ds4_ggml_stubs.h`. Never dereferenced - only held as pointer.            |
| `ggml_glu_op`                                  | Enum stub in `ds4_ggml_stubs.h` (fusion args present in common.cuh but unused by ds4)            |
| `CUDA_CHECK`, `CUBLAS_CHECK`                   | Macros in `ds4_ggml_stubs.h`                                                                     |
| `ggml_cuda_launch_mm_ids_helper`               | Defined in vendored `mmid.cu`                                                                    |
| `ggml_cuda_should_use_mmq`                     | Defined in vendored `mmq.cu` - **we re-implement in `ds4_mmq.cu`** since we don't vendor mmq.cu  |
| `ggml_cuda_mul_mat_q*`                         | Upstream's host entry. **Replaced** by `ds4_mmq_q8_0_dense` and family.                          |

## Things we deliberately do NOT support

- **CUDA graphs.** ds4 manages its own streams; `USE_CUDA_GRAPH` is undefined.
- **HIP / MUSA / AMD backends.** vendored code's HIP/MUSA `#ifdef` branches are dead in our build but kept for upstream diff-cleanliness.
- **The full `ggml_tensor` type.** No tensor introspection - shapes and strides come in via raw arguments to `ds4_mmq_*`.
- **`ggml_op` graph evaluation.** We call kernels directly.

## Re-syncing with upstream

When upstream lands a bugfix or perf improvement we want, the procedure is:

```sh
cd /tmp/llama-research && git fetch && git checkout <NEW_COMMIT>
cd /Users/ent/code/ds4-mmq-lift/cuda/mmq
cp /tmp/llama-research/ggml/src/ggml-cuda/{mmq.cuh,mma.cuh,vecdotq.cuh,quantize.cuh,quantize.cu,mmid.cuh,mmid.cu,common.cuh} .
cp /tmp/llama-research/ggml/src/ggml-common.h .
cp /tmp/llama-research/ggml/src/ggml-cuda/vendors/cuda.h vendors/
# Verify the shim still covers all referenced symbols:
grep -hoE 'GGML_[A-Z_]+|ggml_[a-z_]+' *.cuh *.cu *.h | sort -u > /tmp/symbols.new
diff /tmp/symbols.last /tmp/symbols.new  # check for newly-introduced names
# Update this VENDOR.md's commit pin and run `make cuda CUDA_ARCH=sm_120`.
```

If the new symbols are minor (e.g., new `GGML_CUDA_CC_*` constants), they
likely come from `common.cuh` which we vendor and don't need any shim
changes. If they're new `ggml_*` host functions (rare, but possible if
upstream adds a new helper), extend `ds4_ggml_stubs.h`.

## Testing matrix

| Test                                          | Status                       | Phase     |
|-----------------------------------------------|------------------------------|-----------|
| `nvcc -c cuda/mmq/ds4_mmq.cu` builds cleanly  | **passes** (sm_120, nvcc 13) | Phase 0   |
| Q8_0 dense parity vs CPU reference            | **passes** (4 shapes)        | Phase 2   |
| Q2_K dense parity                             | **passes** (4 shapes)        | Phase 3   |
| IQ2_XXS dense parity                          | **passes** (4 shapes)        | Phase 3   |
| MoE `_id` Q8_0 parity                         | **passes** (3 shapes)        | Phase 4   |
| MoE `_id` Q2_K parity                         | **passes** (3 shapes)        | Phase 4   |
| MoE `_id` IQ2_XXS parity                      | **passes** (4 shapes)        | Phase 4   |
| `make ds4-bench` with mmq integration         | **builds and runs**          | Phase 5/6 |
| Frontier sweep, ctx 2k-16k, V4 Flash IQ2XXS   | **see results below**        | Phase 7   |

## Validated performance

PRO 6000 Blackwell (sm_120), CUDA 13.0, V4 Flash IQ2_XXS GGUF (86.7 GB),
default `DS4_CUDA_MMQ_MOE_MIN_TOKENS=2` (legacy decode preserved):

| ctx    | baseline pf t/s | mmq pf t/s | speedup    | baseline gen t/s | mmq gen t/s | gen ratio |
|--------|-----------------|------------|------------|------------------|-------------|-----------|
|  2048  | 373.21          | 1033.42    | **2.77x**  | 40.48            | 39.33       | 0.972x    |
|  4096  | 366.25          | 1041.39    | **2.84x**  | 39.64            | 38.64       | 0.975x    |
|  6144  | 364.81          | 1025.24    | **2.81x**  | 39.50            | 38.55       | 0.976x    |
|  8192  | 364.01          | 1026.71    | **2.82x**  | 38.81            | 37.88       | 0.976x    |
| 10240  | 361.75          | 1019.53    | **2.82x**  | 38.45            | 37.57       | 0.977x    |
| 12288  | 360.52          | 1013.17    | **2.81x**  | 38.31            | 37.22       | 0.972x    |
| 14336  | 359.15          | 1004.45    | **2.80x**  | 38.09            | 37.18       | 0.976x    |
| 16384  | 357.99          | 1001.29    | **2.80x**  | 38.71            | 37.86       | 0.978x    |

Sustained ~2.80x prefill speedup across the swept context range; gen
within 2.5% of baseline (run-to-run variance).  See `local/docs/`
(auto-round companion repo) for the full Phase 0-7 execution log,
parity-test output, and detailed plan.
