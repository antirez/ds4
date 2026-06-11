# SSD streaming + mixed-precision routed experts

This fork change makes `--ssd-streaming` serve GGUFs whose routed-expert tensors are
**not uniformly quantized across layers** — e.g. a 2-bit model with a few "boosted"
layers upcast to Q4_K (`deepseek4-quantize --tensor-type 'blk.30.ffn_*_exps=q4_k' …`,
the recipe pattern used by [forgequant](https://github.com/andreaborio/forgequant)).

Upstream behavior: such a model **fails every request** under streaming with

```
ds4: Metal model range 50.64..51.76 GiB is not covered by mapped model views
ds4: gpu layer 30 ffn batch encode failed
ds4: gpu layer-major prefill layer 30 encode failed
```

while the same GGUF works with full residency, and uniform GGUFs stream fine.

## Root cause (two compounding assumptions of expert-size uniformity)

1. **Prefill.** Under the streaming "batch selected addr" prefill mode
   (`metal_graph_stream_prefill_batch_selected_addr_enabled`, gated on `layer[0]`'s
   types), each layer is mapped with the *decode-static* span set
   (`model_map_span_vec_include_layer_decode_static`), which deliberately **omits
   `ffn_{gate,up,down}_exps`** — routed experts are expected to come from the SSD
   expert cache. But the Metal encoder's per-layer gate is IQ2-only
   (`gate_type == IQ2_XXS && down_type == Q2_K`), so a Q4_K layer falls through to the
   generic path that wraps the **full fused expert tensors** via
   `ds4_gpu_wrap_model_range` — a range no installed view covers. Encode fails, the
   request 500s.

2. **Decode (latent).** The streaming expert cache is a **single-size-class slab
   allocator**: `ds4_streaming_routed_expert_bytes()` derives the per-expert byte size
   from the *first* routed layer and the slab slot size is frozen on first allocation
   (`g_stream_expert_cache_slab_slot_bytes`). `…note_expert_size()` was
   last-writer-wins, so a boosted layer touching the cache would re-declare the global
   expert size: off-size experts can never use slab slots (size mismatch → one-off
   buffers), they distort the byte budget (a slot's accounting no longer matches its
   bytes), and after an mlock-driven budget cap the exact-size victim requirement in
   `entry_reusable()` can permanently starve loads.

## The change

Detection: a layer is *boosted* iff its per-expert bytes differ from the slab class
(= first routed layer's). New helper `weights_streaming_layer_experts_uniform()`
(ds4.c, next to `ds4_streaming_routed_expert_bytes`).

**Piece A — cover boosted layers with mapped views (prefill + decode reads).**
`model_map_span_vec_include_layer_decode()` wraps the decode-static builder and, for
boosted layers only, also includes the three exps tensors; used by the three decode-span
builders (`weights_model_map_decode_layer_spans`, `…_decode_static_spans`,
`…_decode_static_slice_spans`). The existing fallback paths
(`ds4_gpu_wrap_model_range` for prefill batches, `wrap_model_exact_range` per-expert
on decode) then find their ranges covered. Boosted-layer experts are read through
mmap'd no-copy views (paged from SSD on demand, no residency/warmup in streaming
mode) instead of the expert cache.

**Piece B — keep the expert cache single-class, deterministically.**
`ds4_gpu_set_streaming_expert_cache_expert_bytes()` (new export; no-op stubs for
CUDA/ROCm) pre-seeds the slab size class at engine startup from
`ds4_streaming_routed_expert_bytes()`, and
`ds4_gpu_stream_expert_cache_note_expert_size()` becomes **freeze + reject**: off-size
layers are refused (→ they use the Piece-A mapped path) instead of adopted. Startup
logs the split, e.g.:

```
ds4: SSD streaming mixed-precision model: 6/43 routed layers off the slab size class
     will bypass the expert cache and read experts via mapped model views
```

plus a loud warning if the *majority* of routed layers are off-class (i.e. the first
routed layer itself is the odd one — catastrophic cache hit rate, recipe smell).

### Behavior guarantees

- **Uniform models: byte-identical.** `weights_streaming_layer_experts_uniform()` is
  `true` for every layer → Piece A adds no spans; the startup pre-seed writes the same
  value the first `note_expert_size()` call would have written → Piece B's reject
  branch never triggers. Verified empirically (below).
- **Full-residency (non-streaming) paths: untouched.** All changed call sites are on
  streaming-only paths; the whole-file model map continues to cover everything.
- Memory: a boosted layer adds ~3 tensor-sized mmap views (≈3.4 GiB for Q4_K
  DeepSeek-V4-Flash layers; ~20 GiB for a 6-layer boost). These are pageable, not
  wired; recommend funding them by lowering `--ssd-streaming-cache-experts`
  (40 GB → 24 GB on a 64 GB machine worked well). Note `weights_streaming_non_routed_bytes()`
  (auto cache budget) now counts boosted exps as mapped bytes — the automatic budget
  shrinks accordingly, which is the conservative direction.

## Validation performed (2026-06-11, M5 Pro 64 GB, DeepSeek-V4-Flash)

| check | result |
|---|---|
| Diagnosis end-to-end: **unpatched** binary + `DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR=1` (forces legacy full-layer maps, which include exps) | q4boost GGUF (6×Q4_K-boosted layers, 97 GB) answers correctly → root cause confirmed |
| Uniform-model regression: patched binary, uniform IQ2 GGUF, greedy temp-0 replay of recorded benchmark Q&A | **byte-identical** outputs (3/3) |
| Mixed-precision GGUF on patched binary | serves correctly; MBPP+ canary 25/30 (83.3%), 0 transport errors, ~14 s/task; full benchmark suite (LCB/SuperGPQA/MMLU-Pro/BCB + 24k-token thinking runs) executed against it |
| Startup detection | logs `6/43 routed layers off the slab size class` as expected |

### Known limits / what a reviewer should look at before upstreaming

- **Not compiled here**: CUDA / ROCm targets (the new export has 3-line no-op stubs in
  `ds4_cuda.cu` and `rocm/ds4_rocm_current_api_compat.cuh`, mirroring
  `…_set_streaming_expert_cache_budget`); distributed slice serving with boosted
  layers is unexercised.
- **PRO-shaped models** (384-expert, ≥2 GiB exps tensors): boosted tensors there take
  the `q4_isolated`/grouped-view path in `model_map_span_vec_include_one` — logic is
  shared, but untested on real PRO weights.
- **Perf**: boosted-layer experts are read via page cache instead of the wired expert
  cache — per-token latency on those 6 layers depends on SSD/page-cache pressure. On
  the validation machine decode held ~10-14 s/task on MBPP+-sized generations
  (vs ~10.6 s/task for the uniform sibling model).
- The legacy env escape (`DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR=1`)
  remains available and is the zero-code workaround (at the cost of full-layer
  streaming for *all* layers and the unresolved decode-cache hazard).
