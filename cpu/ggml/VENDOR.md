# Selected GGML CPU primitives

Upstream source: https://github.com/ggml-org/llama.cpp

Selected GGML CPU operations are adapted to ds4's shared `qwen35moe`
text-inference path. Local wrappers use `ds4_qwen35moe_*` names; GGML block
layouts, lookup tables, and numerical operation order are retained.
The intrinsic-header guard supports AVX2-only builds as well as AVX2/FMA.

Pin: `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` (`v0.4.1`).

License: MIT, copyright (c) 2023-2026 The ggml authors. The selected
`llamafile/sgemm.cpp` reduction is copyright (c) 2024 Mozilla Foundation.
The full MIT notice and both attributions are retained in ds4's root LICENSE.

`ggml-common.h` is copied verbatim from `ggml/src/ggml-common.h`. It supplies quantized block definitions and lookup tables. `qwen35moe-quants.c` contains selected functions from:

- `ggml/src/ggml-quants.c`: Q2_K/Q4_K/Q5_K/Q6_K/Q8_0/IQ2_XXS/IQ2_XS row decoders, Q8_K/Q8_0 reference activation quantization, and their helpers.
- `ggml/src/ggml-cpu/quants.c`: generic quantized dot products for those formats.
- `ggml/src/ggml-cpu/arch/x86/quants.c`: AVX2 dot products, Q8_0 activation quantization, and associated helpers/tables.
- `ggml/src/ggml-cpu/vec.h`: AVX2 vector exponential and SiLU functions.
- `ggml/src/ggml-cpu/simd-mappings.h`: x86 F16-to-F32 lookup-table strategy;
  the local 65,536-entry table is initialized once from the existing IEEE
  converter before any quantized row uses it.
- `ggml/src/ggml-cpu/ops.cpp`: 64-key tiled F32 flash-attention precision and
  online softmax update, split-KV partial reduction, and descending expert
  argsort. The local C implementation retains their arithmetic and observed
  tie order without importing GGML's graph executor or libstdc++ code.
- `ggml/src/ggml-cpu/simd-gemm.h`: the AVX2 six-row F32 microkernel used by
  tiled flash attention, adapted to C with a scalar K-order FMA fallback.
- `ggml/src/ggml-cpu/iqp.cpp`: IQ2_XXS/IQ2_XS eight-row panel decode and
  four-activation AVX2 GEMM, including selected-expert tail-row duplication.
- `ggml/src/ggml-cpu/repack.cpp` and `ggml/src/ggml-cpu/arch/x86/repack.cpp`:
  Q4_K eight-row packing, Q8_K activation layout, and selected AVX2 GEMM/GEMV
  integer reductions. The local C path unpacks a bounded panel from mmap rows
  per call; single-query GEMV uses the same block reduction directly on the
  original row, with no persistent packed weight buffer.
- `ggml/src/ggml-cpu/llamafile/sgemm.cpp`: AVX2 F32 eight-lane accumulation
  and horizontal reduction used for multi-token control projections. A local
  four-output, two-activation tile reuses loads while retaining each output's
  K-order FMA and reduction; this
  source carries the Mozilla Foundation copyright above.

Adaptations: the selected functions compile against local definitions instead of the GGML runtime headers. Half conversion uses a local IEEE implementation, with F16C for writing halves when available. Local `ds4_qwen35moe_*` wrappers dispatch to generic or AVX2/FMA primitives. The float dot product uses GGML's AVX four-accumulator reduction; vector SiLU and softmax wrappers retain GGML's operation and reduction order. No GGML runtime or libllama is linked into ds4.

Build with strict aliasing disabled, matching the upstream AVX2 IQ2_XS type-punning assumption. Do not enable fast-math for these primitives. The imported primitives preserve rounded operations and use FMA where the reference uses it. Rounding changes can alter later activation quantization, so validate layer outputs and logits after changes.

The primitive object is linked into `CORE_OBJS` and `CPU_CORE_OBJS` for the
shared `qwen35moe` CPU text path. Production binaries do not link the GGML
runtime or `libllama`. `make test-qwen35moe-quants` runs standalone numerical
checks without a model or external library; `make test-qwen35moe-core` checks
model contracts and saved-state identity. The imported public header also
provides the selected F16, routing, and F32 helpers used by the engine.
