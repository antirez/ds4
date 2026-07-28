# Metal implementation units

The `.metal` files are the device shader library. The `.inc` files are the
Objective-C host implementation fragments included exactly once by
`ds4_metal.m`; they remain a single translation unit and introduce no runtime
abstraction.

- `runtime.inc`, `model_io.inc`, and `expert_streaming.inc`: backend lifetime,
  command submission, mapped weights, and streamed expert residency.
- `embedding.inc`, `dense_norm.inc`, and `elementwise.inc`: shared concrete
  kernel launch paths.
- `moe_dispatch.inc`: shared low-level MoE encoding helpers.
- `deepseek_indexer.inc`, `deepseek_attention.inc`, `deepseek_moe.inc`, and
  `deepseek_hc.inc`: DeepSeek-specific host launch paths.
- `glm.inc`: GLM-specific host launch paths.
- `compat.inc`: compatibility entry points required by the common GPU API.

Model-specific implementations may duplicate code when that keeps their
kernel paths direct and tunable.
