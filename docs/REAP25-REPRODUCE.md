# REAP25-REPRODUCE — rebuilding the 0731 REAP25 2-bit GGUF

Reproduces `DeepSeek-V4-Flash-0731-REAP25-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf`
(68.58 GB, 63.87 GiB), the 25%-expert-pruned 2-bit GGUF of DeepSeek-V4-Flash-0731 for
the ds4 (DwarfStar) REAP-compact runtime on 96 GB Macs.

- **DESIGN.md** — why this approach (GGUF-level prune, no re-quantization).
- **RUNBOOK.md** — step-by-step pipeline with commands and acceptance gates.
- **scripts** — `extract_tables.py`, `bexact.py` (expert-map recovery),
  `prune_gguf.py` (GGUF surgery), plus artifacts `keep_map_bexact.json`,
  `bexact_votes.json`, `iq2_tables.npz`.

## Source chain (see DESIGN.md for details)

1. `deepseek-ai/DeepSeek-V4-Flash-0731` — base model (MIT).
2. `pipenetwork/DeepSeek-V4-Flash-MLX-REAP25` — REAP25 pruning (25% experts:
   layers 3-42 keep 192/256, layers 0-2 keep 256). No published deletion map.
3. `Rednalreden/DeepSeek-V4-Flash-0731-dwarfstar-q2-gguf` — stock 86.72 GB
   IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix GGUF (prune source).
4. Expert map recovered by B-exact weight matching (4-part agreement, 40/40 layers).
5. `prune_gguf.py` drops the 64 pruned experts' chunks per layer and recompacts
   (kept bytes byte-identical to stock; adds `reap.*` metadata).

## Runtime

Load with the ds4 REAP-compact build (`make` in this branch), then e.g.:

    ./ds4-server --metal -m <pruned.gguf> -c 1048576 --port 8001

Optional DSpark drafter (third-party):
`--mtp DeepSeek-V4-Flash-DSpark-support.gguf --dspark` (antirez/deepseek-v4-gguf) or
`--mtp DSpark-drafter-Q2K-Q8-0731.gguf --dspark` (bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF).

## Validation summary

- ppl (131,040 wikitext-2 tokens, ds4, same protocol): stock 4.934 -> REAP25 5.129
  (delta +0.195 = +3.95%).
- TG 14.7/11.7/11.2/11.5/11.7 t/s @ 1K/8K/32K/64K/128K; PP ~120-170 t/s.
