# DESIGN — REAP25 → IQ2XXS GGUF: why this approach

Decision trail for building a 2-bit ds4 GGUF of the REAP25-pruned
DeepSeek-V4-Flash-0731. Written 2026-08-02, after execution. Read RUNBOOK.md
for the how; this is the why, so a future agent doesn't re-litigate it.

## Goal

A ds4-loadable IQ2XXS GGUF of the 0731 REAP25 (25% of routed experts pruned),
smaller than GaelicThunder's 86.7 GB stock-0731 build. Target ≤ ~67 GB.

## The recipe (source of truth)

`gguf-tools/deepseek4-quantize` from the ds4 repo (NOT llama.cpp): routed MoE
experts → IQ2_XXS (gate/up) + Q2_K (down), everything else Q8_0 (attn, shared
experts, output), F16 (HC/compressor/indexer), driven by a **template GGUF**
(metadata + shapes + quant types) + an **imatrix .dat** (per-expert importance).
Reference builds: GaelicThunder/Rednalreden (86.7 GB, 256 experts).

## The two candidate paths

**Path A — re-quantize the REAP25 MLX weights** (`pipenetwork/DeepSeek-V4-Flash-MLX-REAP25`,
129 GB): dequant MLX 8/4-bit → FP8/FP4 safetensors → deepseek4-quantize with a
REAP-shaped template. Rejected as primary because it is **triple quantization**
(official FP8/FP4 → MLX affine → IQ2XXS) and ~1 day of compute, and the
quantizer is not REAP-aware (shape validation vs template).

**Path B — GGUF-level prune of the stock 0731 GGUF** (CHOSEN): take the
86.7 GB `-imatrix-0731.gguf` (already at GaelicThunder quality — single
quantization from real FP8/FP4 0731 with a real imatrix), drop the 64 pruned
experts' quantized chunks per layer, recompact. **No re-quantization at all**:
the kept bytes are byte-identical to the stock file. Justified by:
- pipenetwork's own argument: expert subsetting runs along the expert axis,
  quant groups along the input dim — independent.
- eouya2's existing compact IQ2XXS GGUF (68.6 GB) proves it works in practice.
- GGUF layout: expert dim is the slowest (ne[2]); each expert's data is a
  contiguous, block-aligned chunk; blocks never straddle expert boundaries.

## The missing piece: which experts did pipenetwork drop?

The REAP25 MLX weights are compacted (ids 0..191) — original identity is not
stored. The saliency `usage.npz` is unpublished. Sources considered:

1. **B-exact (CHOSEN): weight matching.** Match each REAP25 expert's weights
   against all 256 stock experts (both derive from the same original weights,
   under different quantization noise: MLX 4-bit affine vs IQ2XXS/Q2K). 4
   independent parts (gate, up, down, router rows) must all agree. Result:
   **40/40 layers, 4-part agreement 192/192, decisive margins** — this IS
   pipenetwork's map.
2. Imatrix proxy (per-expert importance from antirez's .dat): only ~77% overlap
   with the true map — the marginal band is flat (all experts get similar
   signal; expected for a well-trained MoE). Retired as a map source; the
   overlap validated B-exact's plausibility.
3. Reproduce saliency with pipenetwork's `profile_experts.py`: infeasible on
   macbook2 — the unpruned model (165 GB) doesn't fit 96 GB RAM.
4. 0xSero's gated observations dataset: real REAP saliency for the pre-0731
   revision, identity-transferred to 0731 — viable backup map source, not
   needed after B-exact succeeded.

The `usage.npz` ask (posted in the HF comments) remains a free byte-level
confirmation of the recovered map.

## Key facts learned (ground truth)

- REAP25: layers 0–2 hash-routed, keep all 256; layers 3–42 keep 192
  (config `reap: {kept_experts:192, ratio_pct:25, hash_layers_keep:256,
  scored_layers:40}`). **Tensor shapes are authoritative** — the ds4 REAP
  runtime infers per-layer counts from `blk.N.ffn_gate_inp.weight dim[1]`;
  `reap.*` metadata is optional but kept for tooling compatibility.
- ds4 GGUF variant differs from the llama.cpp spec (u64 string lengths, no
  size field, sequential offsets, 32-byte data alignment) — see
  `ds4-gguf-format.md`.
- The stock 0731 GGUF's router has no bias tensor; it uses `exp_probs_b.bias`
  (per-expert, must be pruned too). The MLX port folds this into the gate bias.
- Size math: pruning only 40/43 layers of the ~98% expert-share gives
  ×0.791, not ×0.75 → **68.58 GB**, not 67. eouya2's equivalent compact is
  also 68.58 GB. "≤67 GB" was optimistic; the honest number is ~68.6 GB.

## Validation methodology

- Structural: gguf-py parse + byte-compare of unpruned tensors and kept chunks.
- Runtime: `ds4 --inspect` (per-layer counts 256/192), smoke generation.
- Quality: **δ vs the stock 0731 GGUF on the same corpus/protocol** (131,040
  teacher-forced wikitext-2-raw-v1 tokens, same flags) — both 2-bit, same
  runtime, so the δ isolates map+prune cost from the 2-bit penalty.
  (Comparing against pipenetwork's published 6.4025 would conflate the 2-bit
  penalty with map quality — that number is 4-bit MLX on a different corpus.)

## Known caveats

- Triple-quantization is NOT a concern for Path B (no re-quantization).
- The pruned GGUF's router semantics: `exp_probs_b.bias` pruned to the kept
  experts; matches the REAP runtime's expectations (verified by load + smoke).
- The stock model OOMs Metal at `--prefill-chunk 4096`; use 512 for ppl.
- DSpark/MTP were dropped by pipenetwork; the GGUF carries no drafter (the
  stock file excludes it too).

## What remains (post-documentation)

- ds4-server + speed-trial loop (skill `ds4-speed-trial` protocol).
- Rebase `reap-compact-support` onto upstream `54b36ed` (git-history op; in
  progress by another session).
- Optional: pipenetwork npz confirmation; 0xSero observations as backup map.
