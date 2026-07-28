# Model implementation units

These files contain concrete, model-specific inference code:

- `deepseek_cpu.inc`: DeepSeek V4 CPU reference and decode pipeline.
- `deepseek_graph.inc`: DeepSeek V4 GPU graph state, allocation, prefill,
  decode, and diagnostics.
- `glm_cpu.inc`: GLM CPU reference kernels used by correctness diagnostics.
- `glm_graph.inc`: GLM DSA GPU graph state, allocation, prefill, decode, MTP,
  and diagnostics.

They are implementation fragments included exactly once by `ds4.c`. This is
deliberate: the existing pipelines share many private tensor, model, backend,
and session types. Keeping one translation unit preserves static linkage and
optimization while giving each model a clear source boundary.

The runtime boundary is `ds4_model_provider_v1`; these files do not implement a
generic kernel or operator layer.
