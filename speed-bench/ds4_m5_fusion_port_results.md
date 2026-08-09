# DS4 M5 port results: decode fusion promotion (ds4f-q2, M5 Max 128 GiB)

Final results for the M5 (Metal 4) decode-fusion campaign on this machine.
Model: ds4f-q2 (IQ2XXS experts; the 156 GiB MXFP4 model does not fit
128 GiB RAM). All fusion gates are quant-agnostic shape gates, so the
port applies; the MXFP4-only pre-M5 extras (nsg=1 MoE decode,
fixed-route, pipeline fast lookup, split5/16 schedules) do not engage
with IQ2XXS experts and were not ported.

Code: commit e42d6dc (adds `ds4_gpu_device_is_m5_apple_silicon()` and
admits M5 at the seven gates below; `DS4_METAL_DISABLE_*` rollbacks
unchanged). `make test` fully green on this config.

## Mechanism, eligibility, and contract

Metal System Trace showed pre-M5 decode already ~97.5% GPU-busy, so the
lever is per-dispatch overhead inside a busy timeline (~3.9 us per
dispatch+wrapper on M3 Ultra; ~25 dispatches/layer). Each fusion removes
1–2 dispatches per layer while keeping bit-identical outputs. On M5 the
same lever holds when GPU busy stays high (~98% here); none of the fused
stages have NAX variants, so the fusion math matches the pre-M5 kernels.

The original four pre-M5 fusions live in
`metal_graph_encode_decode_layer_phase` (`ds4.c`). Each is default-ON for
pre-M5 Apple silicon, admitted on M5 after promotion, bit-exact by
construction, and rollback-gated:

| fusion | enable (pre-promotion) | rollback |
|---|---|---|
| F1 HC norm+mix | `DS4_METAL_ENABLE_HC_NORM_MIX_FUSE` | `DS4_METAL_DISABLE_PRE_M5_HC_NORM_MIX_FUSE` |
| F2 compressor quad store | `DS4_METAL_ENABLE_COMPRESSOR_QUAD_STORE` | `DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_QUAD_STORE` |
| F3 router + shared gate/up | `DS4_METAL_ENABLE_ROUTER_SHARED_FUSE` | `DS4_METAL_DISABLE_PRE_M5_ROUTER_SHARED_FUSE` |
| F4 qkv norm + KV RoPE + FP8 store | `DS4_METAL_ENABLE_QKV_NORM_KV_STORE_FUSE` | `DS4_METAL_DISABLE_PRE_M5_QKV_NORM_KV_STORE_FUSE` |

Later M5-only promotions add their own `DS4_METAL_DISABLE_M5_*` rollbacks
(documented in the round sections below).

Official benchmark contract used throughout:

```text
./ds4-bench -m MODEL --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 2048 --ctx-alloc 2081 --step-incr 2048 \
  --gen-tokens 32 --prefill-chunk 4096 --power 100 --warm-weights \
  --csv OUT/speed.csv --dump-frontier-logits-dir OUT/logits
replicas: independent processes; report median and min-max
```

This campaign uses ds4f-q2 (frontier SHA below). The MXFP4 canonical
frontier SHA `6f7edd6d7319d48270e3a5d34eb31f9f8f957ad7f54f4067afa9157c47708179`
does not apply on 128 GiB. Controlled decode A/B must keep
`exact_rows` / `exact_floats` / `exact_selected_ids` unanimous; a
close-but-not-exact result is a rejection. Useful diagnostics already in
tree: `DS4_METAL_GPU_BUSY_PROFILE=1` and
`DS4_METAL_DECODE_STAGE_PROFILE=all` (diagnostic only — serializes the GPU).

Do-not-repeat from the pre-M5 campaign (likely to transfer): wider n64
matvec prefill regressions; direct/bitwise MXFP4 decode (exact but much
slower); gate/up pair half-LUT constant-cache pressure; GPU-generated
indirect routed-work grids; fixed schedules losing to adaptive; sum6
row-widening regressions; flat indexer rope+QAT fusion. Full rejected
detail: `m3_ultra_mxfp4_attempts.csv`.

## Baseline (branch tip caf64d1, fusions off, official command, 3 replicas)

| metric | median | min-max |
|---|---:|---:|
| prefill | 784.13 tok/s | 783.38-784.72 |
| decode | 39.39 tok/s | 39.33-39.48 |
| steady decode | 39.60 tok/s | 39.51-39.63 |
| first token | 28.33 ms | 27.29-28.83 |

Local canonical frontier SHA-256 (ds4f-q2; the MXFP4 canonical
`6f7edd6d...` does not apply):
`eb794f497861d7d9e373665f6e115d8ebe4e17c13c431aa7e318282f16a0f21d`
— stable across every retained replica of every variant.

GPU busy during decode ~98% (DS4_METAL_GPU_BUSY_PROFILE), so the lever
is per-dispatch overhead inside a busy timeline, as on pre-M5.

## Promoted to M5 default (all bit-exact in every retained A/B block)

| feature | forward A/B (2+ processes) | inverse (disable) |
|---|---:|---:|
| F1 HC norm+mix | +0.13% / +0.10% | -0.61% |
| F2 compressor quad projection | +0.20% / +0.17% | -0.42% |
| F3 router + shared gate/up | +0.30% / +0.28% | -1.04% |
| F4 qkv norm + KV RoPE + FP8 store | +0.64% / +0.51% | -0.58% |
| attn inverse-RoPE into FA reduce | +0.28% / +0.14% | -0.29% |
| compressor ratio-4 decode pack | +0.31% / +0.39% | -0.42% |
| router transform+finalize+weights | +0.12% / +0.07% / +0.15% | -0.21% |

## Interleaved same-window official replicas (drift-controlled)

| arm | decode tok/s | steady | prefill | first token |
|---|---|---|---|---|
| all 7 ON | 40.59 / 40.41 / 40.54 (med 40.54) | 40.7 med | 787.6 med | 27.9 ms med |
| all 7 OFF | 39.59/39.55/39.48/39.48 (med 39.51) | 39.7 med | 786.9 med | 28.1 ms med |

Decode **+2.6%** same-window, **+2.9%** vs the baseline median; no
prefill regression. Machine drift between distant windows reached
~10%, so only interleaved/intra-process comparisons were trusted.

## Screened and rejected on M5 (flat or negative, all exact)

- 5/16 decode split schedule: -0.12% (4/0 already ~98% busy on M5)
- compressor exact reduction fusion: -0.16%
- head RMS+RoPE static pipeline: -0.06%
- raw zero attention mask: -0.06%
- output Q8 nr4 + HC sum/norm fusion + HC weights4 (combined): -0.9%
- Q8 decode exact views (alone): -0.6%


## Round 2: attempt to push decode past 41 tok/s (2026-08-07)

Final revalidated state (committed tree, 3 official replicas):
decode 40.64/40.56/40.63 (median 40.63), steady 40.82, prefill
753-779 (drift window), first token ~27.3-28.1 ms, all frontier
hashes canonical. **41 tok/s was not reached with exact methods.**

Everything below was measured and left OUT of the tree:

- Q8 r4 matvec (DS4_METAL_Q8_MV_ROWS=4): initially appeared to give
  decode 43.7/steady 44.1, but that was row-skipping corruption: the
  q_a/kv pair kernel and the hc_expand4 kernels have compile-time
  NR0=2 geometry and skip rows when the dispatch reports nr0=4.
  CAUTION: the official frontier hash covers only prefill logits;
  decode-token exactness is covered only by the A/B harness
  exact_rows and ds4_test --metal-tensor-equivalence greedy checks.
  With the pair/hc_expand dispatches pinned to nr0=2, r4 singles are
  bit-exact but flat (40.2-40.4 both ways). Reverted.
- IQ2 pair-SwiGLU / Q2_K sum6 nsg=1 and nsg=4 decode variants (new
  pipelines; NSG!=2 needs the collective grid/sign table load in
  kernel_mul_mv_id_iq2_xxs_pair_swiglu_f32 scaled as nval=8/NSG):
  bit-exact, flat (nsg1 -0.16%, nsg4 -0.07%). Reverted.
- DS4_METAL_Q8_DECODE_MPP (NAX matmul at n_tok=1): not bit-exact on
  M5 (different accumulation order) - rejected.
- Decode pipeline fast lookup extended to the IQ2XXS/Q2K tape
  (one-line probe): exact, +0.01% - host is not the bottleneck at
  ~98% GPU busy. Reverted.
- Split schedules 2/0, 8/0, 12/0, 4/16, 8/16, 2/16, 6/12: all within
  +/-0.15%; 4/0 remains optimal.
- DS4_METAL_ENABLE_HC_RMS_SCALE_PROJ / GATHERED_KV_STAGE /
  DECODE_NORM_EXACT_VIEWS / F32_DECODE_EXACT_VIEWS / SHARED_KV_PAD /
  UNRETAINED_COMMAND_BUFFERS: all flat (|delta| <= 0.15%).
- DS4_METAL_DISABLE_METAL4: prefill 784 -> 390 (NAX tensor matmuls are
  the prefill driver), decode slightly worse. Keep Metal 4 on.
- MTP speculative decode (Q4K MTP module, --mtp-draft 1..4, greedy
  exact verify): draft 1 ~= baseline (41.6 vs 41.5 t/s CLI short
  prompt), drafts 2-4 progressively slower (28.2, 23.5 t/s). Verify
  cost exceeds acceptance gains on this config.

Structural work that could plausibly beat 41 (out of session scope):
NAX/tensor-op IQ2XXS MoE kernels (prefill routed_moe is 40% of layer
time; decode pair_swiglu is the largest decode matvec), Metal
command-buffer replay for the fixed decode tape, or relaxing the
bit-exactness discipline (e.g. NAX Q8 decode matmuls).


## Round 3: two more exact M5 fusions — decode crosses 41 (steady)

Commits 7a3b9ca and 0d0a78b (ds4f-q2, M5 Max):

1. **q_a/kv pair + quad compressor store merge**: the Q8 q_a/kv pair
   projection and the four F16 compressor projections all read the same
   normalized attention input; one dispatch with two virtual NSG=4
   cohorts per threadgroup for the Q8 range and the verbatim quad
   ranges.  Controlled A/B +0.44%/+0.57%; official interleaved pairs
   +0.40%/+1.48%.
2. **Emit-path compressor finalize merge**: every 4th token, each
   layer's eleven tiny single-row dispatches (norm, rope, FP8
   round-trip + F16 commit for the attention row; norm, rope,
   Hadamard+FP4 QAT for the indexer row; both ratio-4 state shifts)
   become ONE dispatch (22 threadgroups).  Six controlled inverse A/B
   runs all positive (+0.36%..+0.59%), every block bit-exact.

Also measured and reverted: q_b + indexer q_b compound merge (exact,
flat: +0.03% — the indexer chain is idle below top_k=512 at the 2K
benchmark window, and cohort-packing the 24 MB q_b matvec offsets the
launch saving), unpacked cohort variants (flat).

### Final numbers (6 official replicas, drift-controlled window)

| metric | session baseline | final | delta |
|---|---:|---:|---:|
| prefill | 784.13 | 790.2 med | +0.8% |
| decode gen_tps | 39.39 | **40.85 med (40.37-41.05)** | **+3.7%** |
| steady decode | 39.60 | **41.05 med (40.54-41.25)** | **+3.7%** |
| first token | 28.33 ms | 27.0 ms med | -4.7% |

All frontier hashes canonical
(eb794f497861d7d9e373665f6e115d8ebe4e17c13c431aa7e318282f16a0f21d);
full `make test` green.

Remaining unfused emit-path work: the pool pipeline (pack + softmax +
product + 8-lane sum_rows).  The sum_rows 8-thread simd topology is
delicate to replicate exactly; expected value ~+0.2-0.3%.


### Round-3 final tip (09c53a5): 8 official replicas

Adds the M5 admission of the compressor exact softmax+product fusion
(+0.10-0.24% controlled; flat when first screened pre-finalize).

decode gen_tps: 40.38/40.66/40.94/40.95/40.96/40.99/41.01/41.04
    median 40.96 tok/s  (+4.0% vs the 39.39 session baseline)
steady decode:  median 41.19 tok/s (+4.0% vs 39.60)
prefill:        785-788 (no regression)
first token:    ~27.2 ms med
All frontier hashes canonical; focused tests and full make test green.

## Round 4: packed exact attention pushes official decode past 42

Code commits 8b16674, 43c137e, and 457437d (ds4f-q2, M5 Max):

1. **Exact ratio-4 compressor pool**: fuses pack, exact softmax/product,
   and the original active-eight-lane row sum into one dispatch while
   retaining device F32 materialization. Controlled gain +0.15-0.29%.
2. **Clustered HC norm/mix producer**: two independent eight-simdgroup
   NR0=2 clusters share a 512-thread group and retain the exact
   1024-virtual-thread RMS tree. Controlled gain +0.69-0.85%.
3. **Packed exact FlashAttention reduction**: eight physical simdgroups
   time-slice the 32 legacy split-K groups, materialize their exact F32
   weights and `(S,M)` statistics in threadgroup memory, form coalesced
   V partials through a padded 32x33 plane, run the original 32-lane
   reduction tree, and apply the shared inverse-RoPE helper. This removes
   the 4.2 MB/layer device partial round trip and one dispatch. The narrow
   default gate requires M5, F16 512-wide K/V, one unmasked query, 64
   heads, live inverse-RoPE fusion, and at most 1024 keys; all other shapes
   retain the legacy path. Rollback:
   `DS4_METAL_DISABLE_M5_FLASH_ATTN_PACKED32_REDUCE=1`.

The packed attention path was exact over forward and inverse 1024-token
A/B runs (`exact_rows=1041`, `exact_floats=134580480`,
`exact_selected_ids=1040`) and improved controlled decode by 3.95% in
both orders (short blocks +4.4-4.5%). The proposed IQ2/Q2 `sum8`
sidecar was also implemented and exact, but rejected: the producer-side
version was -0.32% and a standalone coalesced sidecar was effectively
flat.

### Final numbers (8 official replicas)

| metric | median | min-max |
|---|---:|---:|
| prefill | 784.46 tok/s | 782.14-787.59 |
| decode gen_tps | **42.76 tok/s** | 42.58-43.05 |
| steady decode | **42.97 tok/s** | 42.81-43.24 |
| first token | 26.16 ms | 25.61-26.48 |

This is +8.6% decode and +8.5% steady decode versus the session baseline,
with no prefill regression. All eight frontier hashes are the canonical
`eb794f497861d7d9e373665f6e115d8ebe4e17c13c431aa7e318282f16a0f21d`.
Full `make test`, focused Metal kernels, tensor equivalence, MXFP4 Metal,
and CLI smoke generation pass on the committed code.


## Round 5: exact concurrent FFN and HC continuation cross 45 tok/s

The M5 decode path now defaults to a narrowly admitted, same-command-buffer
`MTLDispatchTypeConcurrent` schedule for the resident ds4f-q2 FFN. Shared
Q8 gate/up and routed IQ2XXS pair-SwiGLU launch independently, explicit
resource barriers close each producer/consumer level, and the concurrent
encoder is ended before the final HC consumer. The gate is restricted to the
proven single-device IQ2XXS/Q2_K, top-6, one-token shape and declines custom
Q8 matvec geometry. Rollback:
`DS4_METAL_DISABLE_M5_PARALLEL_FULL_FFN=1`.

Additional exact M5 work in this round:

- a two-row IQ2XXS pair-SwiGLU packing/epilogue path;
- aligned vector HC expansion for the fixed Q8_0 shape;
- ratio-128 admission for the QKV pair/compressor compound kernel;
- persistent zero attention-mask reuse on M5;
- fused router projection/select with per-output cross-threadgroup completion;
- a compound HC norm/mix producer whose six legacy 512-thread producer
  groups continue into the pre/post/Sinkhorn split and exact 4096-wide
  RMSNorm. TG0 performs the legacy pre-collapse/RMS tree, TG1 the post
  transform, and TG2-5 use a fenced four-way completion protocol for the
  comb transform. Rollback:
  `DS4_METAL_DISABLE_M5_HC_PRODUCER_PRE_NORM_FUSE=1`.

Completion counters are cached with a bounded `NSCache`, retained with each
in-flight command batch, and invalidated (not CPU-reset in place) after a
Metal command-buffer error. This prevents stale partial counters from being
reused and avoids retaining graph buffers across session churn.

The proposed Q-head RMSNorm/forward-RoPE continuation into gathered F16 KV
staging was implemented and evaluated, but rejected and removed: both the
256-thread virtualized form and an exact-topology 128-thread form changed
full-vocabulary decode logits. The standalone Q kernel's compiled arithmetic
could not be reproduced bit-for-bit inside the compound kernel.

### Controlled exactness and performance

The HC producer/pre-norm continuation improved the 1024-prefix, 256-token
balanced harness from 45.11 to **45.66 tok/s** (+1.22%) with
`exact_rows=265`, `exact_floats=34259200`, and
`exact_selected_ids=264`. Earlier forward/inverse 2048-prefix runs measured
44.76 to **45.30 tok/s** and 44.74 to **45.24 tok/s**, also exact. Router and
persistent-mask inverse A/B checks were exact and independently positive in
the final tree.

### Official 2048-context result (8 cooled replicas)

| metric | median | min-max |
|---|---:|---:|
| prefill | 791.39 tok/s | 790.03-792.71 |
| decode gen_tps | **45.25 tok/s** | 44.90-45.33 |
| steady decode | **45.47 tok/s** | 45.12-45.58 |
| first token | 24.48 ms | 23.97-25.10 |

A no-environment-override replica reached **45.33 tok/s** overall and
**45.58 tok/s** steady. Two later system-loaded validation replicas measured
44.66/44.76 overall (44.93/45.03 steady) while retaining the canonical hash;
the median across all ten replicas remained **45.20 tok/s** overall and
**45.43 tok/s** steady. Every replica produced the canonical frontier
SHA-256
`eb794f497861d7d9e373665f6e115d8ebe4e17c13c431aa7e318282f16a0f21d`.
This is +14.9% decode versus the 39.39 tok/s session baseline and +5.8%
versus the committed Round-4 median.

Validation passed: full `make test`, Metal kernel numerics, five-case Metal
tensor equivalence with zero logit delta, MXFP4 Metal, multi-session Metal
batch exactness, CPU-only build, deterministic CLI generation, and the live
two-turn tool regression. Both fast and exact tool paths produced one tool
call on turn 1 and a normal `stop` with zero tool calls after replaying the
matching result on turn 2.
