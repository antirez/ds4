# Expert Hotlist And Locality Profile

[README](../README.md)

Two Metal-only outputs describe how a run routed MoE experts: a plain-text
hotlist that the SSD streaming cache can be seeded from, and a JSON locality
profile for analysis. Both are written when the engine closes. This page is
the format contract for tools that read them; the writer is
`ds4_expert_profile_write_hotlist_file()` and `ds4_expert_profile_close()` in
`ds4.c`, the hotlist reader is `metal_graph_streaming_expert_hotlist_load_file()`.

## Producing the files

```sh
# Both outputs, for any prompt or session mix:
DS4_EXPERT_HOTLIST=/tmp/hot.txt ./ds4 -m ds4flash.gguf --expert-profile /tmp/profile.json -p "..."

# Hotlist only:
DS4_EXPERT_HOTLIST=/tmp/hot.txt ./ds4-server -m ds4flash.gguf
```

`--expert-profile FILE` (or `DS4_EXPERT_PROFILE=FILE`) selects the JSON path.
`DS4_EXPERT_HOTLIST=FILE` selects the hotlist path. Either enables the
profiler; on CUDA and ROCm both are ignored with a message. Counting covers
every routed layer of every token the engine processes, prefill and decode
alike, for the whole life of the process.

## Consuming a hotlist

```sh
DS4_METAL_STREAMING_EXPERT_HOTLIST=/tmp/hot.txt ./ds4 -m ds4flash.gguf --ssd-streaming
```

`DS4_METAL_STREAMING_EXPERT_HOTLIST` (`DS4_ROCM_STREAMING_EXPERT_HOTLIST` on
ROCm) replaces the built-in preload list with the file. Preloading is only
active with `--ssd-streaming`, not with `--ssd-streaming-cold`, and can be
disabled with `DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST`. The built-in
lists in `ds4_streaming_hotlist.inc` and `ds4_streaming_hotlist_glm52.inc`
were generated from hotlists of this format.

## Hotlist format, version 1

Shape of the file (the counts are illustrative):

```
# ds4 expert hotlist v1
# model DeepSeek V4 Flash
# layers 43
# experts 256
# layer_records 178235
# selections 1069410
# columns: layer expert hits weight
17 203 4102 3151.2270183801655
17 90 3988 2410.8815417154012
...
```

- The first line is exactly `# ds4 expert hotlist v1`. Readers should check
  it. The number is bumped when a column changes meaning, is removed, or the
  ordering rule changes; adding a trailing column or a `#` header line does
  not bump it.
- Header lines are `# key value`. `model` is the model shape name (may
  contain spaces), `layers` and `experts` are the shape's routed layer and
  expert counts, `layer_records` is the number of (layer, token) routing
  events, `selections` is the number of expert slots those events filled
  (records times the model's experts-per-token, minus rejected slots).
- Data lines are `layer expert hits weight`, whitespace separated:
  `layer` and `expert` are zero-based, `hits` is how many tokens selected
  that expert in that layer, `weight` is the sum of the router weights of
  those selections, printed with `%.17g`.
- Every (layer, expert) pair of the model appears exactly once, including
  pairs with zero hits, so a v1 file has `layers * experts` data lines.
- Lines are sorted by `hits` descending, then `weight` descending, then
  `layer`, then `expert`. File order is the preload priority.

The reader is more permissive than the writer:

- Blank lines and lines starting with `#` are skipped; the header values are
  not checked against the running model.
- Only the first three columns are parsed; anything after `hits` is ignored.
- Lines with `hits` of 0 are skipped; a nonzero `hits` becomes the entry's
  priority, saturated at 2^32-1.
- Entries whose `layer` or `expert` is out of range for the running model,
  and repeated (layer, expert) pairs, are skipped silently. Feeding a hotlist
  from a different model shape therefore preloads only the pairs that happen
  to exist, without a warning.
- Reading stops at the preload count (`--ssd-streaming-preload-experts N` or
  the automatic budget), so only the head of the file matters.
- A line that does not start with two unsigned integers and a third unsigned
  integer is an error: the file is rejected and streaming startup fails.
- Lines are read in 255-byte chunks; keep lines short.

## Locality profile JSON

`--expert-profile` writes one object. Tools should identify the format by
`"source"` being exactly `"ds4 Metal expert locality profile"`; a change in
the meaning of any field below changes that string. Shape of the object
(counts illustrative, `...` elided):

```json
{
  "source": "ds4 Metal expert locality profile",
  "model": "DeepSeek V4 Flash",
  "layers": 43,
  "experts": 256,
  "expert_used": 6,
  "layer_records": 178235,
  "selections": 1069410,
  "cache_ns": [1,2,4,8,16,32,64,128,256],
  "cache_summary": [
    {"n":1,"hits":...,"selections":...,"hit_rate":0.0,"weighted_hit_rate":0.0},
    ...
  ],
  "layers_detail": [
    {"layer":0,"hash_router":false,"records":...,"selections":...,"unique_experts":...,
     "avg_adjacent_overlap":0.0,"avg_adjacent_jaccard":0.0,
      "top_experts": [{"id":203,"count":4102,"pct":0.0,"weight":0.0,"weight_pct":0.0}, ...],
      "cache": [{"n":1,"hits":...,"hit_rate":0.0,"weighted_hit_rate":0.0}, ...]},
    ...
  ]
}
```

- `model`, `layers`, `experts`, `layer_records`, `selections` match the
  hotlist header; `expert_used` is the model's experts per token.
- `cache_ns` lists the simulated per-layer LRU cache sizes, in experts; the
  candidates are 1, 2, 4, 8, 16, 32, 64, 128, 256, 384, trimmed to those not
  larger than `experts`. `cache_summary` has one entry per size with hits and
  hit rates over all layers; `hit_rate` is by selection count,
  `weighted_hit_rate` by router weight.
- `layers_detail` has one entry per routed layer, in layer order.
  `hash_router` is true when the layer routed by hash instead of the learned
  router. `records` and `selections` are the per-layer counts.
  `unique_experts` is the number of experts with at least one hit.
  `avg_adjacent_overlap` and `avg_adjacent_jaccard` compare the expert set of
  a token with the set of the previous position of the same session, averaged
  over adjacent pairs (overlap is the shared fraction of `expert_used`,
  Jaccard divides by the union).
- `top_experts` holds at most 16 entries, sorted like the hotlist;
  `pct` and `weight_pct` are percentages of the layer's selections and of
  its total weight. `cache` repeats the simulated cache results per layer.
