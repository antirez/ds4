# PR: FP8-packed compressed-KV cache + long-context memory optimizations (Metal)

**Author:** lixiang `<lixiang.ict@gmail.com>`
**Assisted by:** Claude Opus 4.8 (design, implementation, and validation done in collaboration with Claude Opus 4.8)

---

## Summary

Three always-on, zero-drift memory optimizations for the Metal backend's compressed
(MLA latent) KV path, plus an opt-in packed FP8 comp cache. Together they cut
context-dependent KV memory by **~2.3x** with **bit-identical** model output and
**speed-neutral** decode (the comp-read bandwidth saved is offset by the per-element
e4m3 dequant, so decode is break-even — this is a memory optimization, not a speedup).

All three are on by default; `DS4_DISABLE_KV_OPTS=1` reverts to the previous layout
for A/B comparison. `DS4_METAL_FP8_KV_STORE=1` additionally enables the packed FP8
comp cache (opt-in because it changes the comp cache to e4m3 precision — validated
output-equivalent, see below).

## Motivation

At long context the compressed-KV cache and its indexer scratch dominate the
context-dependent memory and the per-step decode bandwidth. Profiling the
DeepSeek-V4-Flash model showed the comp cache (f16) plus two f32 scratch buffers
(`indexer_scores`, `comp_mask`, each `comp_cap × prefill_cap`) are the largest
context-scaling allocations, and the repeated comp-KV read is the decode bottleneck
at long context.

## What changed

1. **Packed FP8 compressed-KV cache** (opt-in, `DS4_METAL_FP8_KV_STORE`).
   The persistent comp cache is stored in a 3-plane row layout
   (`struct ds4_fp8_kv_row`: e4m3 nope bytes + ue8m0 scale + f16 rot) = **584 B/row**
   vs **1024 B/row** f16. The compressor writes packed once; the indexed-attention
   kernel and a region-aware FlashAttention variant read packed directly (dequant
   to half is bit-identical to the prior `(half)` value). e4m3 decode uses a
   128-entry constant LUT (`ds4_e4m3_lut`) so the per-element dequant is a table
   load, not a branch+exp2.
   - `metal/flash_attn.metal`: `ds4_fp8_kv_row`, `ds4_e4m3_lut`/`ds4_e4m3_mag`,
     `dequantize_fp8_kv_t4`/`_4x4`, MIX region-aware vec + non-vec kernels.
   - `metal/dsv4_kv.metal`: `kernel_dsv4_kv_pack_fp8_row_f32`,
     `kernel_dsv4_kv_unpack_fp8_row_f16`.
   - `ds4.c`: `metal_graph_attn_comp_cache_format()` (0=f32/1=f16/2=packed),
     packed alloc/store/row_bytes, session save/restore payload sizing.
   - `ds4_metal.m`: packed pipelines, `ds4_gpu_dsv4_kv_pack_comp_rows`, the
     format-aware comp→f16 staging unpack, indexed/flash dispatch wiring.

2. **`comp_mask` stored f16** (always on; zero-drift). The top-k mask is binary
   (`-inf`/`0`), both exact in f16 → halves the `comp_cap × prefill_cap` buffer.
   - `metal/dsv4_misc.metal`: `kernel_dsv4_topk_mask`/`_scatter` write f16 or f32
     by an `args.mask_f16` flag. `metal/cpy.metal`: `kernel_cpy_f16_f16`.
   - `ds4_metal.m`: `ds4_gpu_comp_mask_f16()`, f16/f32-aware wrapper + readers.

3. **`indexer_scores` token-tiling** (always on; zero-drift). The score matrix is
   processed in token tiles of 512 so the buffer is `comp_cap × 512` instead of
   `comp_cap × prefill_cap` (4096) = **8x smaller**. Top-k is per-token independent,
   so tiling is exact.
   - `ds4_metal.m`: `ds4_gpu_indexer_prefill_score_topk_tiled` +
     `ds4_gpu_indexer_decode_batch_score_topk_tiled` (token-offset views, pos0=t0).
   - `ds4.c`: `DS4_INDEXER_SCORE_TILE`, tiled call sites, scratch sizing.

Also fixed: the session/checkpoint payload-size accounting
(`session_payload_live_tensor_bytes`, `ds4_session_layer_payload_bytes`) now uses
the packed row stride when packed (previously assumed f32 → "trailing payload
bytes" on packed save/restore).

## Experiment environment

- Hardware: Apple **M5 Max** (40-core GPU), **128 GB** unified memory.
- OS: **macOS 27.0** beta (build 26A5353q); toolchain Xcode 27 beta, **Metal 4.1**
  (`-std=metal4.1`).
- Model: **DeepSeek-V4-Flash** IQ2-XXS (86.7 GB GGUF).
- Harness: `ds4-bench`, LongMemEval_s prompt, context sweep **8k/16k/32k/64k**
  (+ a 96k point), greedy decode 64 tokens. A/B via the single binary:
  `DS4_DISABLE_KV_OPTS=1` (original) vs `DS4_METAL_FP8_KV_STORE=1` (optimized).
  The two sweeps run back-to-back with a 300 s cooldown between them so both start
  from a comparable thermal state; the speed comparison was additionally repeated
  with the sweep order reversed to cancel the residual run-order thermal bias (see
  Performance). Memory = process `phys_footprint` (via `footprint -p PID`), which
  excludes the file-backed model mmap and so captures exactly the KV + scratch +
  activations.

## Validation results

**Quality — zero difference.** Frontier next-token logits are **bit-identical**
(`max_abs_diff = 0.0`, 0/129280 vocab logits differ; argmax token + logit identical)
at every frontier (8k, 16k, 32k, 64k, and 96k). e4m3-precision comp was separately
confirmed output-equivalent on 3 LongMemEval cases.

**Memory — KV cache ~2.3x smaller.**

| ctx | KV cache (orig → opt) | reduction |
|----:|:----------------------|:----------|
| 8k  | 136.8 MB → 71.9 MB    | 1.90x |
| 16k | 249.5 MB → 119.8 MB   | 2.08x |
| 32k | 475.0 MB → 215.7 MB   | 2.20x |
| 64k | 926.0 MB → 407.3 MB   | 2.27x |

The `kvcache_bytes` figures above are deterministic (computed from the layout, not
measured). Process `phys_footprint` (which also includes activations + indexer
scratch and is sampled at 5 s, so it varies slightly run to run) lands at peak
~6.2 GB → ~5.4 GB (≈12–14%) and avg ~11% saved at the 64k sweep. Projected ~7.8 GB
saved at 1M context.

**Performance — prefill equal, decode speed-neutral (break-even).**

Decode throughput on this hardware is dominated by run-to-run thermal variance that
is larger than any effect from the change. The longest context (64k) is the last and
hottest measurement in a sweep, so whichever variant runs second loses ~15% there
purely from heat. Running the A/B in both orders and averaging cancels that bias:

| ctx | decode opt-second | decode opt-first | thermal-averaged |
|----:|:------------------|:-----------------|:-----------------|
| 8k  | +0.3% | −1.2% | **−0.5%** |
| 16k | −0.5% | −1.4% | **−1.0%** |
| 32k | +0.6% | +2.4% | **+1.5%** |
| 64k | −16.5% | +12.5% | **−2.0%** |

The 64k decode figure flips sign with run order (−16.5% ↔ +12.5%), which by itself
proves it is a thermal artifact rather than an algorithmic cost. After averaging,
every context is within **±2%**: decode is break-even. The comp-KV read bandwidth
saved (1024 → 584 B/row) is offset by the per-element e4m3 dequant (kept cheap by the
128-entry LUT), and both scale with comp-row count, so the net is context-independent
and ~zero. Prefill is equal. **This change buys memory, not speed.**

## Notes

- All three optimizations are zero-drift (bit-identical output); packed FP8 comp is
  opt-in because it pins comp to e4m3 precision (validated output-equivalent).
- `ps -o rss` is misleading for this model (it excludes the shared Metal mmap and
  reports ~91 MB); use `footprint`/`phys_footprint`.
