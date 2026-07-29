# GLM DSA integration

This directory owns the GLM DSA model provider and its tailored inference
implementation:

- `provider.c` exposes the whole-model lifecycle to the engine core.
- `cpu.inc` contains CPU reference kernels used by correctness diagnostics.
- `graph.inc` owns GPU graph state, allocation, prefill, decode, MTP, and
  checkpoint orchestration.
- `cuda/`, `metal/`, and `rocm/` contain GLM-specific host and device
  implementations.

The provider calls these concrete paths directly. They do not implement a
generic kernel interface.
