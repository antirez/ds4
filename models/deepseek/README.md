# DeepSeek V4 integration

This directory owns the DeepSeek V4 model provider and its tailored inference
implementation:

- `provider.c` exposes the whole-model lifecycle to the engine core.
- `cpu.inc` is the CPU reference and decode path.
- `graph.inc` owns GPU graph state, allocation, prefill, decode, checkpoint,
  and layer-slice orchestration.
- `cuda/`, `metal/`, and `rocm/` contain DeepSeek-specific host and device
  implementations.

The provider calls these concrete paths directly. They do not implement a
generic kernel interface.
