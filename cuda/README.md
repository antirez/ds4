# CUDA implementation units

`ds4_cuda.cu` includes these fragments exactly once and still compiles as one
CUDA translation unit. The split is source ownership only: it adds no kernel
base class, wrapper layer, dispatch table, or runtime boundary.

- `runtime.inc`: CUDA initialization, tensor storage, model caching, and
  multi-GPU plumbing.
- `deepseek_dense_attention.inc`: DeepSeek embedding, dense, normalization,
  RoPE, and attention kernels.
- `deepseek_control.inc`: DeepSeek HC, compressor, router, and indexer kernels.
- `common_dispatch.inc`: concrete launch wrappers and shared dense dispatch.
- `deepseek_moe.inc`: routed-expert device kernels and launch paths.
- `deepseek_hc.inc`: HC public launch wrappers.
- `runtime_services.inc`: device probing and streamed-expert cache loading.
- `glm.inc`: GLM-specific CUDA kernels and launch paths.

Future model integrations should add concrete model fragments and plug their
whole-model provider into the engine core. Duplication between tailored
kernels is acceptable.
