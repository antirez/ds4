# Metal implementation units

This directory owns Metal runtime code, shared host launch paths, and reusable
device primitives:

- `runtime.inc`, `model_io.inc`, and `expert_streaming.inc`: backend lifetime,
  command submission, mapped weights, and streamed expert residency.
- `embedding.inc`, `dense_norm.inc`, and `elementwise.inc`: shared concrete
  kernel launch paths.
- `moe_dispatch.inc`: shared low-level MoE encoding helpers.
- `compat.inc`: compatibility entry points required by the common GPU API.
- `*.metal`: shared device primitives. `model_abi.metal` defines the structs
  used while concatenating the model shader sources into the library.

Model-specific host launch paths and shaders live under
`models/<model>/metal/host/` and `models/<model>/metal/shaders/`. The runtime
concatenates the shared and selected model shader sources into one library;
`ds4_metal.m` likewise includes all host fragments into one Objective-C
translation unit.

The directory split is ownership only. Model implementations may duplicate
code when that keeps their kernel paths direct and tunable.
