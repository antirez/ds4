# CUDA implementation units

This directory owns CUDA infrastructure and low-level paths reused by model
integrations:

- `runtime.inc`: CUDA initialization, tensor storage, model caching, and
  multi-GPU plumbing.
- `common_dispatch.inc`: concrete launch wrappers and shared dense dispatch.
- `runtime_services.inc`: device probing and streamed-expert cache loading.

Model-specific CUDA implementations live under `models/<model>/cuda/`.
`ds4_cuda.cu` includes the shared and model-owned fragments exactly once and
still compiles as one CUDA translation unit. This split adds no kernel base
class, wrapper layer, dispatch table, or hot-path runtime boundary.
