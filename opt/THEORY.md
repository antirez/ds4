# Kolibri-1 on M3 Ultra — Theoretical Ceilings

Hardware: Mac Studio M3 Ultra, 60-core GPU, 256 GB unified LPDDR5X.
Spec bandwidth 819 GB/s, FP32 ~30 TFLOPS (to be replaced by measured values — see §Bandwidth).

## Architecture (from gguf-tools/kolibri1_quantize.py, ds4.c DS4_SHAPE_KOLIBRI1)

- 50 layers, every 5th full attention (10 full), other 40 sliding window 513.
- hidden 2560; heads 48 x head_dim 128 (attn dim 6144); kv heads 4 x 128 (=512).
- MoE: 384 routed experts, top-6 per token (norm_topk_prob=false, sigmoid+bias router,
  unnormalized weights), 1 shared expert, expert width 512. Activation silu.
- vocab 128000, untied head. Rope base 10000, NeoX pairs over full head dim,
  sliding layers only. RMSNorm eps 1e-6; four norms/layer (sandwich).

## Active bytes per emitted token (weights actually read)

Per layer attention: q 2560x6144 + k/v 2x(2560x512) + o 6144x2560 = 34.08 M params
Shared expert: 3 x 2560x512 = 3.93 M/layer
Routed: 6 x 3.93 M = 23.59 M/layer
Router: 2560x384 = 0.98 M/layer (F32!)
LM head: 128000x2560 = 327.7 M
Active params = 50 x 62.58 M + 0.328 B + embed lookup = 3.457 B (matches spec 3.46B)

| recipe | bytes/token (active) | breakdown |
| --- | ---: | --- |
| Q8   | 3.82 GB | attn 1.81 + shared 0.21 + routed 1.25 (Q8_0 8.5bpw) + router F32 0.20 + head 0.35 |
| Q4   | 3.23 GB | routed experts Q4_K 4.5bpw: 0.66; rest same |
| Q2   | 2.89 GB | routed gate/up IQ2_XXS ~2.06bpw 0.34 + down Q2_K 2.56bpw 0.21; rest same |

## KV read per token (f32 caches, 4 KB/layer/token K+V)

- 40 sliding layers: 513 rows -> 82 MB (constant with context)
- 10 full layers: L x 4 KB -> 80 MB @2K, 0.64 GB @16K, 1.28 GB @32K
- Observed amplification today: every K/V row is read by all 4 simdgroups of a
  threadgroup (x4), plus f32 stores. Ceilings below assume NO amplification.

## Decode ceiling (c1, single user)

t/token = (weights + KV read) / BW. Using spec 819 GB/s; practical ~90%:

| ctx | Q8 theoretical | Q8 practical (~90%) | measured today |
| --- | ---: | ---: | ---: |
| 2K  | 4.78 ms -> 209 t/s | 5.3 ms -> 188 t/s | 58 t/s (17.2 ms) = 31% |
| 8K  | 5.09 ms -> 196 t/s | 5.7 ms -> 176 t/s | 45 t/s (22 ms) = 26% |
| 16K | 5.42 ms -> 184 t/s | 6.0 ms -> 166 t/s | 35 t/s (28.8 ms) = 21% |
| 32K | 6.33 ms -> 158 t/s | 7.0 ms -> 142 t/s | ~24 t/s extrapolated = 17% |

Interpretation: even at 2K, decode runs at ~31% of the pure weight-streaming floor.
The gap is either CPU-side dispatch overhead (~850 kernel encodes/token), GPU idle
gaps between tiny kernels, or kernel inefficiency. Must be attributed by measurement
(DS4_METAL_ENCODER_TIMELINE / DS4_METAL_CB_TIMES) before picking kernels.

## Prefill ceiling

FLOPs/token = 2 x 3.457 B = 6.91 GFLOP. At 50% MFU (~15 TFLOPS effective): 2170 t/s.
Compute-optimal grouped MoE would read each expert's weights once per ~32-token tile:
routed weight traffic at 8K-token chunk ~= 384 experts x 4.17 MB x ceil(~128/32) = 6.4 GB/layer
is wrong — per layer: 384 x 4.17 MB x (avg tokens/expert / 32) = 384 x 4.17 MB x 4 = 6.4 GB for ALL
experts (16 tile-passes total per expert with 128 tokens each -> 384x4 passes? recalc below).

Correct per-chunk (8192 tokens, 6 picks => 49152 picks, ~128 tokens/expert):
- each expert: ceil(128/32)=4 tile passes x 4.17 MB = 16.7 MB -> 384 experts = 6.4 GB/layer
  ... that is 6.4 GB x 50 layers = 320 GB per 8192-token chunk = 39 MB/token.
- today's per-(token,slot) matvec: 8192 x 6 x 4.17 MB x 50 = 10.2 TB = 1250 MB/token.
- attention+dense weights: (34.08M attn + 3.93M shared) x 50 x 1.0625 B + router F32 0.197 GB
  + head 0.35 GB ~= 2.02 GB once per chunk.
- => today is ~32x more weight traffic than grouped MM at 8K chunk.
  (At 2048-chunk: today 313 MB/token vs grouped ~10 MB/token: 30x.)

| metric | measured @2K | practical ceiling |
| --- | ---: | ---: |
| prefill 2048 chunk | 412 t/s (2.4 ms/tok = 20 MB/tok effective) | ~1900 t/s (compute) |
| Qwen3.8 on same machine (reference) | 1331 t/s | — |

Prefill is the widest gap: ~3.2x below a sibling model on identical hardware, ~4.7x
below the practical compute ceiling. The grouped tensor-core MM kernels that Qwen uses
(QWEN4_K_MOE_MM_*, specialized on M3 Ultra by default) already accept q8_0 (type 8).

## Startup
- mmap + residency: ~1.3 s for 79 GB mapping (from log: views 2 ms, residency 834 ms,
  warmup 7 ms). Not a bottleneck for batch benches; low priority.

## Constraint to respect
The engine compares full-vocab logits bit-exactly against the CPU reference for
changes claiming losslessness. Any kernel change must preserve reduction order or be
validated as a labelled trade (compare greedy outputs on eval + needle tests).

## Priority of levers by (gap x likelihood)

1. Prefill grouped MoE MM (reuse Qwen4 MM path; q8_0 supported): expect 2-4x prefill.
2. Decode overhead attribution -> fix dominant cause (dispatch count vs kernel time).
3. Decode attention: threadgroup K/V tiling kills 4x read amp (matters most >8K ctx).
4. Kernel-count reduction via fusion (norms/adds, qkv single GEMV) if timeline shows gaps.
5. Splits/keys tuning for full-attn layers at long ctx.
