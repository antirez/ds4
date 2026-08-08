# RUNBOOK — DeepSeek-V4-Flash-0731 REAP25 → IQ2XXS GGUF for ds4

Reproduces: pruned 2-bit GGUF of the 0731 REAP25 model, loadable by the ds4
REAP runtime. Written 2026-08-02 after the first successful run; every command
below was executed once and verified. **Read DESIGN.md first** for the why.

## Final artifacts

| Artifact | Path | Size |
|---|---|---|
| Pruned GGUF (the deliverable) | `~/NVME_4TB_SSD_GRAUGEAR_Users_ljubomir/ds4/gguf/DeepSeek-V4-Flash-0731-REAP25-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf` | 68.58 GB (63.87 GiB) |
| Stock 0731 GGUF (prune source) | `~/NVME_4TB_SSD_GRAUGEAR_Users_ljubomir/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf` | 86.72 GB |
| REAP25 MLX source (map input) | `~/NVME_4TB_SSD_GRAUGEAR_Users_ljubomir/DeepSeek-V4-Flash-MLX-REAP25/` | 129 GB |
| Keep map (the recovered pruning) | `~/ds4/work/keep_map_bexact.json` | 65 KB |
| Per-part match votes | `~/ds4/work/bexact_votes.json` | 275 KB |
| IQ2 dequant tables | `~/ds4/work/iq2_tables.npz` | 3 KB |
| Wikitext-2 test corpus | `~/ds4/work/wikitext2-test.txt` (+ .parquet) | 1.3 MB |

Code lives in `~/ds4/work/` (git-tracked; NVME copy at
`~/NVME_4TB_SSD_GRAUGEAR_Users_ljubomir/ds4/work/` is byte-identical).

## Prerequisites

- macbook2, venv `torch313-metal` (has numpy, mlx 0.32, torch, safetensors, gguf-py, pandas):
  `source ~/python3-venv/torch313-metal/bin/activate`
- The three source files above on disk (REAP25 MLX + stock 0731 GGUF).
- ds4 REAP runtime binary: `~/ds4/github/worktrees/reap-compact-support/ds4`
  (build: `cd reap-compact-support && make`; REAP support is in ds4.c).
- IQ2_XXS/Q2_K dequant reference: `~/llama.cpp/worktrees/upstream-master/ggml/src/ggml-quants.c`
  (only needed if regenerating `iq2_tables.npz`).

## Pipeline (3 steps + validation)

### Step 1 — recover pipenetwork's keep map (B-exact weight matching)

The REAP25 MLX experts are compacted (ids 0..191) — the original expert ids are
lost. They are recovered by matching each REAP25 expert's weights against all
256 stock experts in the 0731 GGUF, per layer, 4 independent parts
(gate/up/down expert weights + router rows) that must all agree.

```bash
cd ~/ds4/work
source ~/python3-venv/torch313-metal/bin/activate

# 1a. Extract IQ2_XXS dequant tables from llama.cpp (only needed once):
python3 extract_tables.py            # -> iq2_tables.npz

# 1b. Match all 40 scored layers (layers 3..42; 0-2 are hash-routed, keep 256):
python3 bexact.py                    # -> bexact_votes.json  (~40 min)
# single layer test:  python3 bexact.py 3

# 1c. Build the final keep map:
python3 - <<'EOF'
import json
votes = json.load(open('bexact_votes.json'))['layers']
keep = {}
for L, v in votes.items():
    L = int(L)
    ids = set(v['gate'])
    assert ids == set(v['up']) == set(v['down']) == set(v['router']), L
    assert len(ids) == 192, L
    keep[L] = sorted(ids)
for L in (0, 1, 2):
    keep[L] = list(range(256))
json.dump({'n_experts': 256, 'kept_per_scored_layer': 192,
           'scored_layers': [3, 42], 'hash_layers_keep': 256,
           'source': 'b-exact weight matching (4-part agreement, 100%)',
           'keep_map': {str(L): keep[L] for L in sorted(keep)}},
          open('keep_map_bexact.json', 'w'), indent=1)
EOF
```

**Acceptance gate:** every layer prints `4-part full-agree 192/192`,
`unique ids 192` for all four parts, margins ≫ 0 (observed: gate 0.22–0.27,
up ~0.26, down 0.70–0.88, router 0.09–0.55). Any layer failing agreement means
the dequant or orientation is wrong — see Troubleshooting.

### Step 2 — GGUF surgery (prune the stock 0731 GGUF)

Drops the 64 dropped experts per layer (contiguous chunks), keeps the 192 in
ascending-id order (matching pipenetwork's compaction), adds reap metadata.

```bash
cd ~/ds4/work
source ~/python3-venv/torch313-metal/bin/activate
python3 prune_gguf.py                # -> 68.6 GB output, ~5 min
```

Pruned tensors per layer (40 layers × 5): `ffn_gate_inp.weight` (F16 router),
`ffn_gate_exps.weight` (IQ2_XXS), `ffn_up_exps.weight` (IQ2_XXS),
`ffn_down_exps.weight` (Q2_K), `exp_probs_b.bias` (F32, per-expert router bias).
Output adds metadata: `reap.enabled=true`, `reap.layout=ds4-compact-v1`,
`reap.layer.expert_count=[256×43]`, `reap.layer.keep_count=[256,256,256,192×40]`.

**Acceptance gate:** parse with gguf-py (all 1328 tensors, pruned shapes end in
192) + byte-compare kept chunks against the source:

```bash
python3 - <<'EOF'
import json, numpy as np
from gguf import GGUFReader
OUT='.../DeepSeek-V4-Flash-0731-REAP25-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf'
SRC='.../DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
r, rs = GGUFReader(OUT), GGUFReader(SRC)
to, ts = {t.name:t for t in r.tensors}, {t.name:t for t in rs.tensors}
for nm in ('token_embd.weight','blk.0.ffn_gate_exps.weight','output_norm.weight'):
    assert np.array_equal(np.asarray(ts[nm].data).reshape(-1), np.asarray(to[nm].data).reshape(-1)), nm
keep = json.load(open('keep_map_bexact.json'))['keep_map']
a = np.asarray(ts['blk.3.ffn_gate_exps.weight'].data); b = np.asarray(to['blk.3.ffn_gate_exps.weight'].data)
assert all(np.array_equal(a[k], b[j]) for j, k in enumerate(keep['3']))
print('OK')
EOF
```

### Step 3 — validate with the ds4 REAP runtime

```bash
cd ~/ds4/github/worktrees/reap-compact-support
# 3a. Inspect: expect "compact routed expert counts inferred: min=192 max=256" and no errors
./ds4 --inspect -m <PRUNED.gguf> --backend metal
# 3b. Smoke: expect a coherent completion (observed: "Hello! How can I help you today")
./ds4 -m <PRUNED.gguf> --backend metal --ctx 512 --nothink --temp 0 -n 8 -p hello
# 3c. Perplexity vs stock (same window/flags; ~2.5h per model):
./ds4 -m <MODEL.gguf> --backend metal --ctx 131072 --nothink --prefill-chunk 512 -n 200000 \
      --perplexity-file ~/ds4/work/wikitext2-test.txt
```

ppl comparison driver: `~/ds4/work/ppl_compare.sh` (stock then pruned, appends
`tokens=...` lines). The δ = ppl(pruned) − ppl(stock) on the same 131,040
tokens is the map-quality gate. Reference point: pipenetwork's REAP25 (4-bit
MLX) was +4.5% over its base; expect a similar or smaller δ at 2-bit.

## Expected outputs (first run)

- bexact: 40/40 layers OK, margins as above, ~40 min total.
- surgery: 68.6 GB in ~5 min (200 pruned tensors).
- inspect: clean; file size 63.87 GiB; tensor bytes 63.86 GiB.
- smoke: coherent text; generation ~24–29 t/s.
- ppl (measured 2026-08-03, 131,040 wikitext-2-raw-v1 tokens, ds4 --perplexity-file,
  `--ctx 131072 --prefill-chunk 512 --nothink`):

```
STOCK  (0731, 256 experts): tokens=287737 scored=131040 nll=209152.555808752 avg_nll=1.596097038 ppl=4.933738601
PRUNED (REAP25):            tokens=287737 scored=131040 nll=214233.554933795 avg_nll=1.634871451 ppl=5.128798653
δ = +0.195 ppl = +3.95%  (map quality: better than pipenetwork's published +4.5% at 4-bit)
```

Notes: the stock leg took ~8h, the pruned leg ~18h under memory pressure (80 GB resident
on a 96 GB box swaps; the ppl loop is single-token decode). Expect ~2.5h unloaded. Do NOT
compare absolute ppl against pipenetwork's 6.40 numbers — different protocol (their
1024-window teacher forcing vs ds4's 131k decode window).

## Benchmarks (REAP25 0731, measured 2026-08-03/04, ds4-server + llama-benchy 0.3.7)

5-depth sweep (pp 2048, tg 128, 3 runs):

| Depth | PP t/s | TG t/s |
|---|---|---|
| 1K | 169.97 ± 8.9 | 14.72 ± 0.8 |
| 8K | 142.33 ± 10.9 | 11.67 ± 1.2 |
| 32K | 120.35 ± 12.8 | 11.21 ± 0.9 |
| 64K | 124.99 ± 1.9 | 11.47 ± 0.3 |
| 128K | 121.62 ± 7.9 | 11.74 ± 1.1 |

Server ctx must exceed depth+pp+template overhead: use `-c 141000` for the 128K point
(135000 was rejected: prompt 138,616 tokens). Results: `~/llama.cpp/contrib/llama-benchy/results/reap25-0731-depth-sweep-*.md`.

## DSpark speculative decode (measured 2026-08-04)

Drafter: `antirez/deepseek-v4-gguf` → `DeepSeek-V4-Flash-DSpark-support.gguf` (5.6 GB,
pre-0731 format: `mtp.*` tensors, bare `dspark.*` keys). Plugs into the REAP25 GGUF:
`ds4-server --metal -m <REAP25> --mtp <drafter> --dspark -c 141000`. Loads clean
(missing=0 invalid=0 metadata_errors=0), runs fully on Metal (not NVIDIA-only).

**Measured: no speedup — net_saved = −91.9 ms (marginally slower).**
`DS4_DSPARK_STATS=1` output: cycles=47 proposed=8 accepted_draft=8 accept_rate=100%
no_draft=42 (scheduler_skips=35 tail_skips=8), propose=166.6ms verify=295.1ms
spec_total=300.9ms target=2553.9ms saved=375.6ms net_saved=-91.9ms.
Drafts (len 1–2) are 100% accepted but the scheduler gates most steps and the 3-layer
Metal verification overhead exceeds the saved decode. Observation knobs:
`DS4_DSPARK_STATS=1`, `DS4_DSPARK_SPEC_LOG=1`, `DS4_DSPARK_PROBE=1`; this fork has no
/metrics endpoint (404) — stats print at session/engine shutdown.

## Re-run checklist

1. Files present (REAP25 MLX, stock GGUF, venv, ds4 binary).
2. `extract_tables.py` (if `iq2_tables.npz` missing).
3. `bexact.py` → keep_map; **verify the acceptance gate** before proceeding.
4. `prune_gguf.py` → GGUF; verify parse + byte-compare.
5. ds4 inspect + smoke + ppl.

## Troubleshooting (all hit and fixed on the first run)

| Symptom | Root cause | Fix |
|---|---|---|
| bexact: `data type 'bfloat16' not understood` | safetensors-np can't read BF16 | read via torch: `safetensors.torch.safe_open(..., framework='pt')` + `get_tensor().float().numpy()` |
| bexact: weight in one shard, biases in another | shards split tensor triples | resolve each suffix's shard from `model.safetensors.index.json` separately |
| bexact: down-part matching fails (dup ids, margin 0) | `d`/`dmin` byte order swapped in Q2_K blocks | block layout is `scales[16], qs[64], d, dmin` (d BEFORE dmin) |
| GGUF reader: "cannot reshape array of size N into (256,2048,1056)" | written offsets wrong | ds4 GGUF uses **u64 key/name lengths, no tensor-size field, strictly sequential offsets** — see `ds4-gguf-format.md` |
| data copied from wrong region | tensor offsets are **relative to data_start** (aligned 32) | read from `align(header_end, 32) + offset`; pad output to 32 |
| ds4 inspect: `exp_probs_b.bias has dim[0]=256, expected 192` | per-expert router bias not in the prune list | it is a 5th pruned tensor per layer (F32 [256]) |
| ds4 ppl: `Metal command batch failed: Insufficient Memory` on the stock model | default `--prefill-chunk 4096` allocates too much on an 80 GB model | `--prefill-chunk 512` |
| bexact slow (~40 min) | per-expert numpy dequant of 8.4M blocks/layer | expected; layer-wise parallelism optional |
