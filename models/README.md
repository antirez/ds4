# Model integrations

Each directory owns one complete, tailored model integration:

```text
models/<model>/
├── provider.c / provider.h
├── cpu.inc
├── graph.inc
├── cuda/
├── metal/
│   ├── host/
│   └── shaders/
└── rocm/
```

The engine-facing boundary is the whole-model `ds4_model_provider_v1`
lifecycle. A provider owns its session orchestration and calls its custom
kernels directly; there is intentionally no generic kernel, operator, graph,
or tensor interface between them.

The `.inc` implementation fragments are still included exactly once by the
engine or backend entry point. This preserves the existing single translation
units, private types, static linkage, and compiler visibility. The directory
split expresses ownership without adding wrappers to hot paths.

Code belongs under `models/<model>/` when its semantics, tensor layout, or
launch sequence are specific to that model. Backend runtime, memory management,
and genuinely reused low-level primitives remain under `cuda/`, `metal/`,
`rocm/`, and `kernels/`.

To add a model, implement its provider and the backend paths it supports. It is
fine to duplicate kernels when separate implementations are easier to tune or
understand.
