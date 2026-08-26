# Graph diagnostics

The GPU graph has binary tensor-dump hooks for correctness work. They are
disabled unless a backend-specific prefix is set:

```text
DS4_ROCM_GRAPH_DUMP_PREFIX=/tmp/run
DS4_METAL_GRAPH_DUMP_PREFIX=/tmp/run
DS4_ROCM_GRAPH_DUMP_NAME=name1,name2
DS4_ROCM_GRAPH_DUMP_LAYER=all
DS4_ROCM_GRAPH_DUMP_POS=0
```

The ordinary hooks synchronize the GPU, read the selected tensor, and restart
the command batch at every hook. This is useful for isolated dumps but can
perturb command scheduling and must not be treated as transparent
instrumentation.

## Deferred one-token layer-state capture

For a one-token DeepSeek V4 Flash prefill at position zero, four replicated
layer-state hooks can instead be copied device-to-device in stream order and
written only after layer execution and the output head complete:

```text
DS4_ROCM_GRAPH_DUMP_DEFER_LAYER_STATE=1
DS4_ROCM_GRAPH_DUMP_NAME=hc_attn_post,ffn_moe_topk,ffn_moe_weights_scaled,hc_ffn_post,result_output
```

Use the corresponding `DS4_METAL_*` variables on Metal. Deferred mode:

- supports exactly one token at position zero on one local GPU;
- supports Flash HC graphs and these four layer-state names, plus the ordinary
  `result_output` hook;
- preserves the existing graph-dump kernel-selection mode. In particular,
  `DS4_ROCM_GRAPH_DUMP_NONINVASIVE` remains an independent choice;
- allocates one graph-owned arena, queues ordered D2D snapshots, and performs
  host reads only after the prefill/output command sequence;
- supports full graphs and explicit layer-slice ranges;
- rejects follow-on decode, multi-token, unsupported-name, and multi-local-GPU
  use rather than silently falling back to synchronous intermediate dumps;
- quarantines the arena, and on graph teardown all graph-owned GPU resources,
  instead of freeing copy destinations or sources if synchronization fails.

Float captures use `${PREFIX}_${NAME}-${LAYER}_pos${POS}.bin`; top-k IDs use
`.i32`. Deferred layer files are logged after `result_output` because host I/O
is intentionally delayed.

Run the backend regression with:

```sh
make test-graph-deferred-dump-rocm HIPCC=/path/to/hipcc
```

## Exact-span routed-expert overlap probe

Set `DS4_DIST_SPEC_EXACT_ROUTE_PROBE=1` alongside the experimental exact-span
switches to snapshot every selected routed-expert ID for each row and layer of
a 2..5-row exact span. The probe queues D2D snapshots at the router boundary,
uses the span's existing end-of-command synchronization, then emits one line
per layer containing raw row selections, adjacent intersections/Jaccards, and
the whole-layer union. It works for both the ordinary exact loop and the
`DS4_DIST_SPEC_EXACT_SHARED_ROWS` helper. It is otherwise allocation-, copy-,
and log-free.

`DS4_DIST_SPEC_EXACT_ROUTE_PROBE_LIMIT=N` limits successful probe spans per
session; `N` must be in `1..UINT32_MAX`. Allocation, copy, read, formatting, or
invalid-limit failures fail the requested span instead of omitting records.
