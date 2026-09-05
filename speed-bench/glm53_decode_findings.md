# Where GLM 5.3 Flash decode time goes on an M3 Ultra

Apple M3 Ultra, 80 GPU cores, 512 GB, macOS 26.5.2, Metal backend.
Model: `GLM-5.3-Flash-Q4_K.gguf`, 177.8 GiB, fully resident, no SSD streaming.

The short version: the routed-expert kernels are already close to the hardware
ceiling, and the largest single consumer of decode bandwidth is not the experts
at all.  It is the KDA (linear attention) projections, which this artifact
stores as **BF16** while its experts are Q4_K.  Requantizing them to Q8_0 --
which is what `gguf-tools/glm53_quantize.py` already specifies for
`role == "linear_attention"` -- is worth **+13.4% decode** with no measurable
quality cost.

## Decode budget

Measured with `DS4_GLM_DECODE_ABLATE`, which removes a stage and reports the
resulting speed.  Baseline 21.19 tok/s, two baseline runs 0.38% apart.

**The original KDA row was never measured.**  `DS4_GLM_DECODE_ABLATE` had no
`kda` bit and the KDA path returned before the mask was read, so the 18.37 ms
and 38.9% recorded here first could not have come from this harness.  `kda`,
`kda_qkv`, `kda_gate`, `kda_recur`, `kda_out`, `hc` and `head` now exist, and
the whole budget below was re-measured through them.

Everything in the table is from **one** run of arms against **one** baseline of
22.375 tok/s = 44.693 ms/token (four interleaved baselines, 0.85% spread).
Do not mix it with the earlier 21.19 tok/s figures; those are superseded.

| component | ms/token | share | superseded figure |
|---|---:|---:|---|
| KDA attention (34 layers) | 15.99 | 35.8% | was 18.37 / 38.9% |
| DSA attention core (11 layers) | 8.02 | 18.0% | was 7.91 / 16.8% |
| routed MoE (42 layers) | 7.89 | 17.6% | was 8.08 / 17.1% |
| mHC producer chain (90 sites) | 3.99 | 8.9% | was inside the residual row |
| shared expert (42 layers) | 1.98 | 4.4% | was 2.13 / 4.5% |
| output head (norm + logits matvec) | 1.80 | 4.0% | was inside the residual row |
| attn_output projection (11 layers) | 1.33 | 3.0% | was 1.26 / 2.7% |
| q_path (11 layers) | 0.44 | 1.0% | was 0.59 / 1.3% |
| indexer (11 layers) | 0.21 | 0.5% | was 0.35 / 0.7% |
| remaining norms, residual, hc_expand | 3.04 | 6.8% | residual, not ablated |
| **total** | **44.69** | **100%** | |

Only KDA moved outside noise; every other carried-over row reproduced.

Bandwidth is deliberately **not** a column here.  It is only meaningful where
the byte count is exactly derivable, which is the dense projections; see "What
is left, priced" below for those, computed against 736.9 GB/s.  The routed-MoE
and shared-expert byte figures recorded in the first version of this document
(4.43 and 0.55 GiB) depend on which experts a token selects and were never
re-derived here, so they are omitted rather than restated.

**A per-token bandwidth figure computed over the whole decode step is
misleading.**  Dividing total bytes by total decode time gives roughly a fifth
of peak, but the bandwidth-bound kernels only occupy about a fifth of the step.
The kernels themselves are near the ceiling; the rest of the step is other
work.

**Ablation arms are destructive.**  A skipped stage leaves stale contents in
its output buffer, which is fine for timing the dispatches that remain but can
in principle change data-dependent routing downstream (expert selection, index
selection).  The arms agree with each other -- `hc,head` measures 5.77 ms
against 5.79 for the two separately, and the KDA substages sum to 15.44 against
15.99 for the whole stage -- so the effect is small here, but these are
skip-ablation estimates, not per-kernel timings.

## The BF16 KDA projections

`blk.N.kda_q`, `kda_k`, `kda_v` and `kda_output` are BF16 in this artifact, on
all 34 KDA layers.  They are dense -- every one is read on every decoded token:

| | bytes/token | share of decode traffic |
|---|---:|---:|
| KDA q/k/v/output (BF16) | 8.50 GiB | **60%** |
| routed experts (Q4_K) | 4.43 GiB | 31% |
| shared expert (Q8_0) | 0.55 GiB | 4% |
| attn_output (Q8_0) | 0.73 GiB | 5% |

The KDA projections alone read nearly twice what all routed experts read.

This is not what the repo's own quantizer produces.  `regular_qtype()` in
`gguf-tools/glm53_quantize.py` maps `role == "linear_attention"` to `Q8_0` for
its default `--artifact q4`, and embedding/output likewise.  This artifact has
all of them at BF16, so it was not produced by that path.

### Why BF16 is not simply a mistake

Metal has a fused three-way QKV matmul, `ds4_gpu_glm53_matmul_bf16_qkv`, which
requires all three of q/k/v to be BF16 and issues one dispatch instead of
three.  It is gated behind `DS4_METAL_DISABLE_M3_ULTRA_GLM53_DECODE`, so it was
added as an M3 Ultra optimisation.  Quantizing KDA forfeits it and falls back
to the generic per-tensor matmul.

Measured, so the trade is not a guess:

| | decode tok/s | Δ |
|---|---:|---:|
| baseline (fused BF16 QKV) | 21.12 | -- |
| `DS4_METAL_DISABLE_GLM53_BF16_QKV=1` | 20.97 | -0.7% |
| `DS4_METAL_DISABLE_M3_ULTRA_GLM53_DECODE=1` | 20.96 | -0.8% |

The fusion is worth **0.7%**.  The BF16 storage it requires costs an order of
magnitude more than that.  No fused Q8_0 QKV kernel is needed to capture the
win; the generic fallback is nearly free.

## Result of requantizing KDA to Q8_0

`gguf-tools/glm53-requant-bf16` converts the 136 KDA tensors from BF16 to Q8_0
into a **new file**, copying every other byte verbatim, through the same
`quants.c` facade the other tools use.  It refuses an output that resolves to
the input (same path, hard link or symlink), because the input stays mmapped
for the whole run; there is no in-place mode.

    make -C gguf-tools glm53-requant-bf16
    ./gguf-tools/glm53-requant-bf16 in.gguf out.gguf --type q8_0 --tensors kda

`--tensors` also takes `head` (`output.weight`), `embd` (`token_embd.weight`)
and `all`; it defaults to `kda`.

8.50 GiB of KDA weights become 4.52 GiB; the file goes 177.8 -> 173.8 GiB.

Speed, arms interleaved O-Q-Q-O with the same binary and only the model file
changing, 8 context frontiers:

| ctx | original | kda Q8_0 | Δ |
|---:|---:|---:|---:|
| 2,048 | 21.05 | 23.91 | +13.54% |
| 8,192 | 20.65 | 23.42 | +13.44% |
| 16,384 | 20.56 | 23.31 | +13.40% |
| **mean (8 ctx)** | | | **+13.37%** |

Prefill -0.25%.  Within-arm drift 0.15-0.19%, so the effect is far outside the
noise, and it is within 0.3 points at every context.

Quality, teacher-forced over 18,672 tokens of `promessi_sposi.txt`:

| | avg NLL | perplexity |
|---|---:|---:|
| original (BF16 KDA) | 1.838851 | 6.289309 |
| requantized (Q8_0 KDA) | 1.834773 | **6.263711** |

No degradation -- marginally better, which at this size is noise.  Greedy
generations from both are coherent and track word for word until a late
paraphrase.

### The projection was too optimistic, and why -- corrected

The earlier reading of this was wrong, and it mattered, because it set the
priority for the whole KDA path.

It went: scaling KDA's 497 GB/s by the byte reduction predicts +22%, we
measured +13.4%, therefore only ~62% of KDA was weight streaming and the
remaining **~7 ms/token** is conv1d, gating and the recurrent state update --
"the next thing to attack, and not a bandwidth problem."

Direct substage ablation says otherwise.  Splitting KDA on the original BF16
artifact:

| substage | ms/token | share of decode |
|---|---:|---:|
| qkv projections | 9.68 | 21.7% |
| output projection | 3.16 | 7.1% |
| gate/beta low-rank chain | 1.37 | 3.1% |
| recurrence kernel (conv1d + gating + state) | **1.23** | **2.8%** |
| unattributed (dispatch, interaction) | 0.55 | 1.2% |

The conv1d, the gating and the recurrent state update together are **1.23
ms/token**, not ~7.  KDA is about 90% weight streaming, not 62%.  Requantizing
the same tensors to Q8_0 and re-ablating confirms it directly -- qkv goes 9.68
-> 5.61 ms and the output projection 3.16 -> 1.85 ms against a pure-bandwidth
prediction of 5.14 and 1.68, so both are ~90% bandwidth-scaled:

| | original | Q8_0 KDA |
|---|---:|---:|
| decode | 22.375 tok/s | 25.545 tok/s (**+14.2%**) |
| KDA total | 15.99 ms | 10.40 ms |

The +13.4% headline reproduces (+14.2% here).  Only the explanation was wrong.

Where the original inference went astray: it assumed everything in KDA that
did not scale with weight bytes was recurrence work.  Most of it is instead the
projections failing to scale *perfectly* -- they are ~90% bandwidth-bound, not
100% -- plus fixed dispatch cost.  Attributing that gap to the recurrence
inflated a 1.23 ms stage into a 7 ms one.

The practical consequence: **the recurrence kernel is not where the time is.**
Work on `metal/glm53_kda.metal` is capped at 2.8% of decode no matter how good
it gets.  The qkv projections, at 21.7%, are the KDA target that matters.

## Splitting the old "norms, hyper-connections, residual, LM head" row

That row was a residual -- whatever the other arms did not account for -- and
at ~18% of decode it was the second largest line in the budget with nothing
measured inside it.  The `hc` and `head` ablation arms split it:

| | ms/token | share |
|---|---:|---:|
| mHC producer chain | 3.99 | 8.9% |
| output head | 1.80 | 4.0% |
| everything else in the row | ~3.0 | ~6.7% |

`hc,head` together measure 5.77 ms against 5.79 for the two separately, so the
split is additive and the arms are not interacting.

**The mHC producer was the largest unoptimised item in the decode step.**
`glm53_graph_hc_pre` issued four dispatches -- plain RMSNorm, the 16384->24 mix
matvec, the split/mix, and the weighted RMSNorm -- twice per layer over 45
layers, so 360 small dispatches per token for 3.99 ms of work.  DeepSeek V4
already fused the F16 equivalent. The initial GLM port reused its reduction
with BF16 mix weights, one dispatch per site instead of four. These historical
measurements predate the BF16 reduction correction described below:

| ctx | four dispatches | fused | delta |
|---:|---:|---:|---:|
| 2,048 | 22.282 | 23.545 | **+5.67%** |
| 4,096 | 21.94 | 23.195 | +5.72% |
| 16,384 | 21.795 | 23.00 | +5.53% |

The original bit-exact claim does not hold for actual decode. Frontier logits
match, but they exercise prefill rather than this fusion; actual decode logits
exposed the different BF16 reduction. The corrected fusion keeps GLM's one-SIMDgroup
scalar FMA sequence and the F32 normalization rounding boundary; DeepSeek's
F16 path retains its eight-SIMDgroup vector reduction. Both types are now
tested byte-for-byte against their own unfused four-dispatch chain, including
mix, split weights, collapse, and weighted normalization. The historical
2.41 ms saving is not a timing of the corrected implementation.

The corrected fusion was checked on M3 Ultra on 2026-09-04 with the original
`GLM-5.3-Flash-Q4_K.gguf`, `promessi_sposi.txt`, context allocation 16,384,
and 128 greedy decode tokens per frontier. The following are means of two
runs per arm in baseline / corrected / fusion-off / fusion-off / corrected /
baseline order. Baseline is `b0a147a`; both revisions include the same GLM
router device-memory fence. Timing uses uninstrumented `ds4-bench`.

| Context | Baseline, tok/s | Branch, HC fusion off | Branch, corrected fusion | Fusion gain | Branch gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 21.125 | 24.780 | 26.055 | +5.15% | +23.34% |
| 8,192 | 20.730 | 24.165 | 25.405 | +5.13% | +22.55% |

All six benchmark runs produce byte-identical greedy text. Separate full-F32
decode-logit captures match the historical baseline under teacher forcing,
and match fusion-off under greedy decoding in both default and `--quality`
modes. Each comparison includes 128 decode steps at each frontier and
continuation after snapshot restore (258 vectors including prefill).
`make test-glm53-kda` and `./ds4_test --metal-kernels` pass. The revised HC
fixture fails against the original BF16 shader and passes against the fix;
the F16 comparisons remain exact. These results supersede claims that the
uncorrected default branch's roughly 30-31% gain was bit-perfect.

**The output head is nearly all matvec.**  1.80 ms for a [4096 -> 154880]
BF16 matvec is close to what its 1.27 GB costs at this machine's measured
bandwidth, so there is no dispatch overhead worth chasing there.

### Why GPU-side argmax is not worth doing

A natural suggestion is to stop reading all 154,880 logits back and scanning
them on the CPU, and instead do a hierarchical argmax/top-k on the GPU and
return only the token.  Measured directly on this machine:

    logits readback (memcpy of 605 KiB)   0.0143 ms
    CPU argmax scan over 154,880 floats   0.1668 ms
    combined                              0.1811 ms

That is **0.40% of a 44.76 ms decode step, below the run-to-run spread**, so
the change could not be shown to work even if it were free.  It also would not
remove a synchronisation: the token is needed before the next step can start
either way.  Not worth the complexity.

## A trap in the stage profiler

`DS4_METAL_DECODE_STAGE_PROFILE` reports a stage named `attn_output` on all 45
layers, and on KDA layers it is the largest stage in the run.  It is **not**
measuring the output projection there.  The profile boundary sits after the
`glm53_attention_done:` label, and KDA layers reach that label by `goto`, so on
those layers the `attn_output` sample times the entire KDA attention.

The real output projection is 2.7% of decode, not 39%.  Ablation and the
profiler agree to within 0.3 points once the label is read correctly (2.7% vs
3.0% on the 11 DSA layers, where the label means what it says).

Two further cautions when using that profiler: it flushes the command buffer at
every boundary, which on this workload adds a uniform ~0.206 ms per boundary
and roughly triples the measured decode time; and because the floor is uniform,
stages that do little real work all read as roughly the floor.  Prefer
`DS4_GLM_DECODE_ABLATE` for attribution and use the stage profiler to localise.

## A non-destructive second instrument

The ablation arms are destructive: a skipped stage leaves stale contents, so
the run is timing-only and can in principle perturb data-dependent routing.
`DS4_GLM_DECODE_REPEAT` is the other half of the pincer.  Every stage it
accepts is a pure function of its inputs, so dispatching it one extra time per
site writes the same bytes; the whole-token delta is then one extra execution
of that stage and **the model output is unchanged**.  Verified: all six arms
dump logits identical to the baseline at max|delta| = 0.

Only idempotent stages get a bit.  The KDA recurrence advances conv and
recurrent state, and directional steering updates in place, so neither can be
repeated and neither is offered.

Where the two instruments agree, the number is trustworthy:

| stage | ablate | repeat | agreement |
|---|---:|---:|---|
| kda_qkv | 9.69 | 9.76 | 0.7% |
| head | 1.80 | 1.76 | 2% |
| kda_gate | 1.37 | 1.45 | 6% |
| **kda_out** | **3.33** | **2.47** | **35%** |

The `kda_out` disagreement is reproducible across rounds, and the exact byte
count settles it.  Both instruments agree that kda_qkv costs 9.69 ms for
6.845 GB, i.e. 706 GB/s; kda_output is 2.282 GB, which at that rate is 3.23 ms
-- next to the ablation figure, not the repeat one.  Repeat underestimates here
because the second dispatch re-reads a 67 MB per-layer weight set that is
partly still resident, while kda_qkv's 201 MB per layer does not survive.

**So: repeat is the right instrument for dispatch-bound stages and undercounts
cache-friendly bandwidth-bound ones; ablation is the reverse.**  Use both, and
let exact bytes arbitrate when they disagree.

### The remaining bucket, partly split

`hc_expand` measured **0.55 ms/token, 1.3% of decode** by repeat -- a
dispatch-bound stage, so that figure was the reliable one.  Of the 90 sites, 87 are
now folded into whatever produces their input, and between them they returned
0.37 ms of it:

| site | count | mechanism | gain |
|---|---:|---|---:|
| kda_output (BF16 matvec) | 34 | new epilogue kernel | +0.46% |
| attn_output (Q8_0 matvec) | 11 | DeepSeek's existing fused kernel | +0.11% |
| FFN tail (routed+shared add) | 42 | existing `has_add` path on the expand | +0.14% |

That is 87 of the 90 sites; the three leading dense FFN layers have no
routed/shared split to defer and keep the separate expand.

The last two together are +0.48% (t=9.65, n=8), 0.199 ms over 54 sites, or
3.7 us per site -- below the 4.6 us launch cost and the KDA epilogue's 5.6 us,
which fits: those two remove a cheap elementwise add and a Q8_0 matvec rather
than a BF16 matvec plus a 64 KiB round-trip.

Two of the three needed no new kernel at all.  `ds4_gpu_matmul_q8_0_hc_expand_tensor`
already existed for DeepSeek and reads post/comb from `hc_split` at the offsets
GLM uses; `ds4_gpu_hc_expand_add_tensor` already exposed the expand kernel's
`has_add` path.  Only the BF16 matvec needed an epilogue written.

That leaves roughly 2.5 ms in the residual row for the residual adds,
directional steering, the remaining norms and the final HC collapse, none of
which are separated yet.

With the mHC producer now fused, `hc_pre` measures 1.36 ms by repeat, down from
the 3.99 ms the four-dispatch chain cost.

## The budget, re-measured after the fusions

Everything above was measured before the mHC, gate-pairing and HC-expand work.
Re-run on the current tip, baseline 41.598 ms/token:

| stage | ms | share |
|---|---:|---:|
| KDA attention | 15.39 | 37.0% |
| routed MoE | 7.89 | 19.0% |
| DSA attention core | 7.86 | 18.9% |
| shared expert | 2.07 | 5.0% |
| output head | 1.68 | 4.0% |
| mHC producer | 1.44 | 3.5% |
| attn_output | 1.10 | 2.6% |
| q_path | 0.56 | 1.4% |
| indexer | 0.19 | 0.5% |
| **residual** | **3.42** | **8.2%** |

The mHC producer is down from 3.99 to 1.44 ms.  Note that `kda` now also
covers the HC expansion folded into `kda_output`, so its 15.39 is not directly
comparable with the earlier 15.99.

### How much launch overhead is left in total

Chasing the residual stage by stage has diminishing returns, so
`DS4_METAL_ENCODER_COUNT` counts compute-encoder acquisitions instead -- one
per dispatch for essentially every primitive here.  (It is acquisitions, not
encoder objects: inside a batch the same encoder is handed back for every
dispatch, so the count is a dispatch proxy.)  Differencing two runs of
different decode length removes prefill and setup:

    6,605 acquisitions over 8 decode tokens
    26,989 over 40
    (26989 - 6605) / 32 = **637 dispatches per decode token**

If the 4.6 us launch cost measured on the gate pairing transfers to the other
kernels and command-buffer arrangements -- which has not been checked, so treat
this as an estimate rather than a measured floor -- that is about **2.93
ms/token, 7% of the 41.31 ms step**, spread across every stage rather than
concentrated in the residual.  It is roughly what the remaining dispatch-count
work is competing for: no arrangement of the current graph gets under the
launch overhead without removing launches, whatever its exact size.

For scale, the fusions in this branch have already taken roughly 3.5 ms of
dispatch and intermediate-traffic cost out of the step, so what is left is
smaller than what was found.

### Splitting the residual

`DS4_GLM_DECODE_REPEAT` gained a `router` bit.  Repeat rather than ablate is
the only honest instrument for it: skipping the router leaves a stale expert
selection, which changes which experts the routed stage streams and therefore
changes the very cost being measured.  Verified non-destructive (identical
greedy output).

| | ms | share of decode | share of the residual |
|---|---:|---:|---:|
| router (logits + top-k, 84 dispatches) | 0.95 | 2.3% | 28% |
| remaining hc_expand (FFN tail, dense attn) | 0.33 | 0.8% | 10% |
| still unattributed | 2.13 | 5.1% | 62% |

The router reads `ffn_gate_inp`, which is **F32** at [4096, 288] over 42
layers: 198.3 MB/token, or 0.28 ms at the 707 GB/s the dense projections
achieve.  So about a third of the router is weight streaming and the other
~0.66 ms is the top-k select over 288 experts plus launch cost.  Requantizing
`ffn_gate_inp` is a model-artifact change and routing precision is the obvious
risk, but it is the only 200 MB/token F32 tensor left in the decode step.

### The shared expert is not the outlier it looked like

The original budget recorded the shared expert at 0.55 GiB/token and 279 GB/s,
38% of ceiling -- far below every other kernel, and an obvious target.  **That
byte count was under by about 2x.**  Summed from the tensor table, the shared
expert reads three Q8_0 [4096, 2048] tensors per layer over 42 layers:

    gate + up + down = 3 x 374.2 MB = 1.123 GB/token

Against the measured 2.07 ms that is **542 GB/s, 74% of ceiling** -- in the
same band as KDA overall (77%), not an outlier.  Its gate/up/SwiGLU is already
fused via `ds4_gpu_shared_mid_swiglu_q8_0_tensor`.  Closing the remaining gap
to 707 GB/s would be worth about 0.45 ms, 1.1%, not the large win the 38%
figure implied.

## The DSA attention core: corrected, then fixed

An earlier revision priced this stage at 135.4 MB/token and 2.4% of the memory
ceiling.  **That was wrong by 23x on bytes**, for three reasons each worth
recording:

- **`n_rot` is 0 for GLM 5.3**, so a compact cache row is the 512-wide lora
  part alone, 1024 B in f16 -- not 1152 B with a 64-wide rope tail.
- **There are 11 DSA layers in the trunk, not 12.**  `attn_v_b` appears 12
  times because the MTP layer has one, which is not in the decode path.
- **The cache is not read once per layer.**  The generic kernel dispatches one
  threadgroup per head -- 64 of them -- each independently walking all selected
  rows twice, once to score and once for the weighted sum.

Corrected traffic: 2051 rows x 1024 B x 2 passes x 64 heads x 11 layers is
2.96 GB, plus 0.20 GB of `attn_k_b`/`attn_v_b`, so 3.15 GB/token.  At 7.7 ms
that is 409 GB/s, **56% of ceiling** -- real headroom, but not the collapse the
old figure implied.  `qk_low` accounts for 0.55 ms of the stage, leaving 7.23
ms in the kernel proper.

### The split kernel was tried first, and is not what ships

`kernel_glm_attention_indexed_decode_split_group8_partial` is the obvious
candidate: 8 heads per threadgroup so a loaded cache row serves eight of them,
16 rows staged in threadgroup memory so scoring and the weighted sum read
device memory once, and row blocking so the work spreads over many more
threadgroups than the generic kernel's 64.  GLM 5.2 decode has been running it
all along; two guards kept GLM 5.3 out (`qk_rope != 64`, every rope path of
which is zero-trip at GLM 5.3's `n_rot = 0`, and a block-count check against
worst-case buffer sizing rather than the runtime count).  Relaxing both, at
ctx 2048, interleaved:

| | tok/s | ms/token |
|---|---:|---:|
| generic kernel | 24.232 | 41.27 |
| split group8 | **28.318** | **35.31** |
| | **+16.86%** | |

It is not bit-exact against the generic kernel, though: it scores with
lane-split dots and reduces with an online softmax across row blocks, and the
DSA attention outputs differ by 3.06e-05 of range.  That is float reordering,
and every quality measurement taken -- greedy generation identical over 128 to
256 tokens on six prompts, long-context NLL within 0.002% -- says it is
harmless.  It still fails the standard this branch holds itself to, which is
that a faster path must reproduce the path it replaces, so **on this branch
GLM 5.3 does not use it.**  It remains what it was before, GLM 5.2's kernel,
now selectable off under `--quality` (the generic kernel is the exact one) and
via `DS4_METAL_DISABLE_GLM53_DSA_SPLIT`, and covered by `tests/test_glm53_kda`
against a double-precision reference with out-of-range and `UINT32_MAX` rows
in the selection.

One thing about its call site was wrong and is fixed regardless: it passed
`selected_rows_valid = true`, selecting the kernel variant that skips the `row
< cache_cap` test.  GLM 5.2's selections are always in range.  GLM 5.3's are
not once more than the 4096-row full-attention window is visible (8192 under
SSD streaming): beyond it the pool selector supplies 2051 rows padded with
`UINT32_MAX` sentinels, and the unchecked variant reads those out of bounds.
On this machine those reads returned values whose effect stayed below the
greedy threshold -- a build with the check skipped was byte-identical over 128
tokens on prompts of 1,471, 3,841 and 10,352 tokens -- and the 1.04% deviation
an earlier revision attributed to them could not be reproduced.  An
out-of-bounds read is a bug whatever it returns, so the call site passes
`false` for every GLM model; on all-valid selections the two variants perform
the same arithmetic in the same order, which the test asserts bit for bit, so
GLM 5.2 is unchanged.

### The kernel that ships: the generic arithmetic, staged and shared

The generic kernel's cost is structural, not arithmetic: one threadgroup per
head, each walking every selected row twice.  The arithmetic can be kept to
the operation and reorganised around it.
`kernel_glm_attention_indexed_decode_exact_*` computes the same thing in four
phased dispatches:

- **scores**: one thread per (head, row) running the generic kernel's
  sequential 512-term dot, with 16 selected rows staged in threadgroup memory
  per threadgroup for all 64 heads at once (1024 threads), so each cache row
  is read from device memory once per token instead of 128 times;
- **weights**: one 256-thread threadgroup per head -- the generic kernel's
  threadgroup -- running its per-thread row partition and its 128/64/../1
  reduction tree for the max and the denominator, and turning scores into
  softmax weights in place;
- **lora**: one thread per (head, column pair) walking rows 0..n-1 in
  selection order with the generic kernel's `acc += w * kv` chain, over 8
  heads x 64 columns per threadgroup so a row slice is loaded once for eight
  heads.  Rows are consumed in stages of 32: every thread fetches 16 bytes and
  one weight three stages ahead into double-buffered threadgroup memory, so
  the scattered row reads are in flight while the fma chains run.  A row past
  `cache_cap` contributes `fma(0, kv[0], acc)`, which leaves `acc` unchanged
  bit for bit, where the generic kernel skips it;
- **value**: the generic kernel's quantised row dot from threadgroup memory,
  one thread per output element, over 256 threadgroups instead of 64.

Every floating-point operation, operand and ordering is the generic kernel's,
so the output is bit-identical to it, and that is asserted rather than
assumed:

- `tests/test_glm53_kda` runs the exact kernels and the generic kernel on one
  fixture at 8, 513, 1024, 2048, 2051 and 4096 selected rows, with rows at and
  past `cache_cap` and `UINT32_MAX` sentinels in the selection, and requires
  the outputs to match with `memcmp`;
- greedy generation from this tip is **byte-identical to main** (110afdd, and
  b0a147a after the branch was rebased onto the synced main) over 128
  tokens on a 1,471-token prompt at ctx 4096 (dense window, every row valid),
  a 3,841-token prompt at ctx 8192 (dense window, 3,841 rows) and a
  10,352-token prompt at ctx 16384 (pool selector, 2051 rows with sentinels).
  No switches and no quality mode: the default path reproduces the base
  commit.

What the phases cost at about 1,500 selected rows, each measured by dropping
its dispatch and reading the change in decode time:

| phase | ms/token |
|---|---:|
| scores | 0.33 |
| weights | 0.18 |
| lora, first version: row ids and weights read from device per row | 2.60 |
| lora, pipelined | 0.64 |
| value | 0.39 |

The first lora version was slower than the generic kernel it replaced -- two
serialised device loads per row instead of one.  Staging and prefetching is
what made the phase cheap; the arithmetic never changed.

Decode from the CLI, greedy, 128 tokens, single runs (the interleaved
benchmark below is the figure to quote):

| prompt | ctx | main | this tip | non-exact split kernel |
|---|---:|---:|---:|---:|
| 1,471 tokens | 4096 | 22.21 | **28.25** | 28.67 |
| 3,841 tokens | 8192 | 19.05 | **27.23** | 28.35 |
| 10,352 tokens | 16384 | 20.91 | **27.07** | 27.8 |

The exact kernels give back nearly all of what the split kernel offered --
within 1.5% at the short prompt -- while reproducing the base commit's output
bit for bit.  Three more dispatches per DSA layer (four instead of one) account
for about 0.15 ms/token of the gap.

`--quality` keeps the exact kernels, since they are exact;
`DS4_METAL_DISABLE_GLM53_DSA_EXACT` selects the generic kernel for A/B runs.
The two-host tensor-parallel head split keeps the generic kernel until that
configuration has been run.

Where the phased path starts to pay, decode from the CLI at ctx 4096, 128
greedy tokens, exact kernels against the generic kernel on the same prompt:

| selected rows | generic | exact | delta |
|---:|---:|---:|---:|
| ~36 (a one-line chat prompt) | 29.04 | 28.93 | -0.4% |
| ~134 | 28.79 | 28.89 | +0.3% |
| ~207 | 28.51 | 28.83 | +1.1% |
| ~308 | 28.25 | 28.87 | +2.2% |
| ~603 | 27.52 | 28.71 | +4.3% |
| ~992 | 26.57 | 28.57 | +7.5% |
| ~1,500 | 25.51 | 28.25 | +10.7% |

Below 128 rows the generic kernel's row traffic is a few megabytes per layer
and the three extra dispatches cost more than they save, so the exact path
engages from 128 selected rows.  Both kernels are exact, so crossing the
threshold as a generation grows changes nothing but speed.

## The shared-down fusion, after a second look

An earlier revision of this document said this could not be done without a
dedicated shared-expert mid buffer, because `glm_graph_routed_moe_one_dispatch`
takes `ffn_mid` as scratch and would clobber the shared mid.

That is true of only one of the two orderings.  `shared_first` is
`streaming_selected_cache`, so the shared expert runs first *only* on the
SSD-streaming path.  On the fully-resident path the routed stage has already
finished when the shared expert runs, `ffn_mid` still holds the shared mid and
`ffn_out` holds the routed result -- which is exactly what
`ds4_gpu_shared_down_hc_expand_q8_0_tensor` takes.  No extra buffer.

Fusing the shared down-projection, the routed add and the HC expand into that
one dispatch is worth **+0.77% (t=13.60), 0.320 ms over 42 sites** -- 7.6 us
per site, above both the 4.6 us launch cost and the 3.7 us the plain FFN-tail
fusion returned, because it also removes the `ffn_sum` round-trip.  The
streaming path is excluded and keeps the separate dispatches.

## Two tuning knobs that turn out not to matter

Both were expected to be worth something on an 80-core GPU and neither is.

**Decode command-buffer flush cadence.**  Indexed decode flushes every 4
layers, and `DS4_GLM_DECODE_FLUSH_INTERVAL` overrides it.  Sweeping 0, 2, 3, 4,
6, 8, 12, 16, 32 at ctx 2048: everything from 3 to 12 lands in 22.25-22.31
tok/s, inside the run-to-run spread.  Only the extremes lose -- 0 (never flush)
at 21.91 and 32 at 22.06.  Confirmed at ctx 16384, where 2/4/8 give
21.80/21.81/21.78.  **The default of 4 is already right.**

**DSA split-attention rows per block.**  The choice steps straight from 32 to
128 at 1024 selected rows and had never been swept, so
`DS4_GLM_DECODE_SPLIT_BLOCK_ROWS` was added to force one value:

| rows | ctx 2048 | ctx 16384 |
|---:|---:|---:|
| default (32/128) | 23.50 | 23.00 |
| 32 | 23.51 | 22.97 |
| 64 | 23.51 | 23.02 |
| 96 | -- | 23.01 |
| 128 | 23.51 | 23.02 |
| 256 | 23.49 | 23.02 |

Flat to within 0.2% at both contexts.  The selection count is capped by
`glm53_graph_indexer_selected_limit()`, which does not grow with context, so
this does not become interesting at longer contexts either.  The knob is kept
as instrumentation for other GPUs, not because it found anything here.

## Cumulative engine-only result

Individual commits report gains against whatever baseline was current when they
landed, which does not compose into a branch number.  This is the direct
measurement: the pre-series commit and the branch tip, each built in its own
tree so each reads its own `metal/*.metal`, run against the **same unchanged
GGUF** with the same harness, in the order main / branch / branch / main so
that drift lands on both arms alike.

    ds4-bench, promessi_sposi.txt, 128 greedy tokens per frontier,
    frontiers 2048, 4096, 8192, 16384; four runs, main / branch / branch / main

| frontier | main prefill | branch prefill | prefill | main decode | branch decode | decode |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 429.98 / 429.52 | 429.71 / 429.89 | +0.01% | 21.09 / 21.12 | 27.82 / 27.84 | **+31.86%** |
| 4096 | 390.14 / 390.08 | 389.93 / 390.03 | -0.03% | 20.75 / 20.76 | 27.12 / 27.13 | **+30.69%** |
| 8192 | 392.23 / 392.08 | 392.01 / 392.00 | -0.04% | 20.71 / 20.69 | 26.99 / 27.04 | **+30.51%** |
| 16384 | 389.49 / 389.54 | 389.33 / 389.45 | -0.03% | 20.65 / 20.58 | 26.88 / 26.98 | **+30.63%** |

Both runs of each arm are shown; the deltas compare the means.  Prefill is
untouched by this branch's decode work and measures as such.  At ctx 2048 the
base arm reproduces the 21.16 tok/s measured at the start of this series, so
machine conditions have not drifted.

Repeated after the branch was rebased onto the synced main (b0a147a, 24
upstream commits of Metal tensor-parallel and DSpark work), same protocol:

| frontier | main prefill | branch prefill | prefill | main decode | branch decode | decode |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 429.64 / 430.08 | 429.73 / 429.81 | -0.02% | 20.97 / 21.06 | 27.76 / 27.76 | **+32.10%** |
| 4096 | 390.20 / 390.55 | 390.10 / 390.14 | -0.07% | 20.67 / 20.73 | 27.06 / 27.05 | **+30.70%** |
| 8192 | 392.30 / 392.64 | 392.08 / 391.52 | -0.17% | 20.63 / 20.69 | 26.94 / 26.96 | **+30.45%** |
| 16384 | 389.56 / 389.97 | 389.43 / 389.52 | -0.07% | 20.55 / 20.59 | 26.93 / 26.81 | **+30.63%** |

The synced main decodes GLM 5.3 Flash at the same rate as 110afdd did and
produces the same greedy output on every prompt used here, so upstream's
changes did not touch this path; the rebase itself conflicted only in the
Makefile's test list and in the mHC producer kernel, which upstream had
refactored into a shared body that the branch now templates.

Contributions, each measured against the baseline current when it landed: the
mHC producer fusion +5.67%, the KDA gate pairing +0.74%, the three HC-expand
epilogues +0.46% / +0.11% / +0.14%, the shared-down/HC fusion +0.77%, the gate
trio +0.30%, and the exact phased DSA kernels (see above).  The widened BF16
loads (~+5.4%) and the split DSA kernel for GLM 5.3 (+16.86%) were measured on
the way and are not on this branch's default path, for the reason in the next
section.

Note the base reproduces the 21.19 tok/s of the original budget almost exactly,
which is a useful check that machine conditions have not drifted between the
first measurements in this document and the last.

Stacking the model-artifact changes on the engine, all at ctx 2048.  **This
table predates the exact DSA kernels**: it was taken at d5b7895, when the
engine-only tip measured 23.99 tok/s, and has not been re-measured since, so
its rows are not comparable with the figure above.  What it still shows is the
artifact effect on top of one engine state:

| model file | tok/s at d5b7895 | vs base engine + original artifact |
|---|---:|---:|
| GLM-5.3-Flash-Q4_K | 23.99 | +13.2% |
| GLM-5.3-Flash-Q4_K-kdaQ8 | 27.50 | +29.8% |
| GLM-5.3-Flash-Q4_K-kdaHeadQ8 | 28.23 | +33.2% |

Only the first row is an engine result.  The other two combine it with the
requantized artifacts and should never be quoted as engine tuning.

## Nothing on the default path is left that is not bit-exact

Two changes made on the way here were deterministic but not bit-identical to
the paths they replaced.  Both are off this branch's default path:

- **The widened BF16 matvec loads** (about +5.4%) repartitioned which lane
  accumulates which k.  An exact wide variant would have to redistribute every
  lane's strided elements with cross-lane shuffles, two per element, which
  costs roughly what the widening saved, so the scalar accumulation -- the
  pre-branch kernel -- is the only path.  The fused qkv/pair/trio/HC-expand
  kernels share its row helper unchanged, so they stay exact.
- **The grouped/split DSA kernel** (+16.86%) is replaced for GLM 5.3 by the
  exact phased kernels.  GLM 5.2 keeps it as before, with `--quality`
  selecting the generic kernel there.

### Checked end to end against the base commit

Greedy generation (`--raw-prompt --temp 0`, 128 tokens), the tip and main
(110afdd when first measured; repeated against b0a147a after the rebase, with
the same result and the same main output on every prompt)
each built in its own worktree, byte-compared, both in default mode:

| prompt | ctx | selection | result |
|---|---:|---|---|
| 1,471 tokens | 4096 | dense, 1,472+ rows | **byte-identical** |
| 3,841 tokens | 8192 | dense, 3,842+ rows | **byte-identical** |
| 10,352 tokens | 16384 | pool top-k, 2051 rows with sentinels | **byte-identical** |

Encoder counts confirm the exact path ran: it adds three dispatches per DSA
layer per token, 33 x 127 = 4,191 more acquisitions than the generic arm,
which is itself byte-identical to the base.  Under `--ssd-streaming`, same
prompt, 32 tokens, the exact and generic arms are byte-identical to each
other and to the resident run.  Tensor parallelism was not run; the exact
kernels stay off there.

## Other models: nothing broke, and one thing had slowed

Two models that take none of the GLM 5.3 Flash paths were run through the
full test suite and the same main / branch / branch / main `ds4-bench`
protocol, and byte-compared on greedy generation (128 tokens, 1,471- and
3,841-token prompts):

- **DeepSeek V4 Flash** (`MXFP4Experts-F16HC-...-chat-v2-mxfp4-0731`): every
  suite OK on the branch; output byte-identical to main; prefill and decode
  within 0.25% of main at every frontier, in both directions.  The only
  shared code it touches is the templated HC producer, whose f16
  instantiation is the kernel it always ran.

  | frontier | main decode | branch decode | decode | prefill |
  |---:|---:|---:|---:|---:|
  | 2048 | 42.67 / 42.58 | 42.61 / 42.54 | -0.12% | -0.01% |
  | 4096 | 38.87 / 38.73 | 38.69 / 38.72 | -0.24% | -0.04% |
  | 8192 | 38.19 / 38.26 | 38.17 / 38.30 | +0.03% | +0.05% |
  | 16384 | 37.30 / 37.40 | 37.31 / 37.45 | +0.08% | -0.13% |

- **GLM 5.3 (`glm-dsa`, `UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K`)**: the full
  model, 79 layers with a 64-wide RoPE tail, so it takes the GLM 5.2 path
  and the split DSA kernel, not the exact kernels.  Output byte-identical to
  main.  Five suites fail on the branch -- and fail identically on main, with
  the same 55 assertions and the same golden-vector statistics: they are
  DeepSeek official-vector fixtures and a 30k-token recall test this quant
  does not pass on either tree.  The benchmark found a real regression:

  | frontier | main decode | branch decode | decode | prefill |
  |---:|---:|---:|---:|---:|
  | 2048 | 16.32 / 16.26 | 15.96 / 15.95 | **-2.06%** | -0.01% |
  | 4096 | 16.24 / 16.22 | 15.91 / 15.90 | **-2.00%** | +0.04% |
  | 8192 | 16.02 / 16.03 | 15.68 / 15.70 | **-2.09%** | +0.04% |
  | 16384 | 15.64 / 15.62 | 15.32 / 15.33 | **-1.95%** | +0.07% |

  The cause is the split kernel's bounds-checked variant, which the branch had
  switched every GLM model to.  On Flash it cost 0.24% of decode, with DSA
  attention in 11 of 45 layers; here the split kernel runs in 76 of 79 layers
  and the same per-call cost is 2% of the step.  This model's selections are a
  dense range or a top-k over visible rows, always in range, so it goes back
  to the unchecked variant it always ran -- main's kernel, bit for bit, as
  the all-valid equivalence case in `tests/test_glm53_kda` asserts -- and
  only a GLM 5.3 Flash graph, which pads with sentinels, would pass `false`
  should it ever reach that call.  Re-measured with that change:

  | frontier | main decode | branch decode | decode | prefill |
  |---:|---:|---:|---:|---:|
  | 2048 | 16.42 / 16.33 | 16.32 / 16.30 | -0.40% | +0.03% |
  | 4096 | 16.32 / 16.21 | 16.20 / 16.19 | -0.43% | +0.08% |
  | 8192 | 16.01 / 16.01 | 16.01 / 15.93 | -0.25% | +0.01% |
  | 16384 | 15.66 / 15.66 | 15.64 / 15.62 | -0.19% | +0.04% |

  What remains is inside main's own run-to-run spread (its two ctx 2048 runs
  differ by 0.55%); if any of it is real it is at most 0.4%, against the 2%
  before the change.  Output stays byte-identical to main.

## Rollback switches, and what PR #954 does that this branch could use

antirez/ds4#954 (pre-M5 DeepSeek decode and prefill, bit-exact) puts every
optimisation behind its own `DS4_..._DISABLE_...` switch with an aggregate
that turns the whole set off, and measures each against its rollback.  This
branch has the same shape now.  Each switch restores the pre-branch path for
one change, and `DS4_METAL_DISABLE_GLM53_FLASH_TUNING` restores all of them:

| switch | restores |
|---|---|
| `DS4_METAL_DISABLE_GLM53_HC_PRODUCER_FUSE` | four dispatches per mHC producer site instead of the fused BF16 kernel |
| `DS4_METAL_DISABLE_GLM53_KDA_GATE_PAIR` | separate f_a / g_a and f_b / g_b projections |
| `DS4_METAL_DISABLE_GLM53_KDA_GATE_TRIO` | beta as its own projection beside the pair |
| `DS4_METAL_DISABLE_GLM53_KDA_OUT_HC_EXPAND` | a separate HC expand after kda_output |
| `DS4_METAL_DISABLE_GLM53_ATTN_OUT_HC_EXPAND` | a separate HC expand after attn_output |
| `DS4_METAL_DISABLE_GLM53_FFN_HC_EXPAND_ADD` | a separate routed+shared add and HC expand in the FFN tail |
| `DS4_METAL_DISABLE_GLM53_SHARED_DOWN_HC_EXPAND` | the shared down-projection without the routed add and expand |
| `DS4_METAL_DISABLE_GLM53_DSA_EXACT` | the generic DSA attention kernel |
| `DS4_METAL_DISABLE_GLM53_FLASH_TUNING` | every path above at once |

Not switchable: the KDA decay hoist, a kernel-internal cleanup that is
bit-identical (the KDA prefill/decode consistency test) and worth nothing
measurable, and the prefill constants, which are knobs with their defaults
unchanged.  `DS4_METAL_DISABLE_GLM53_DSA_SPLIT` belongs to the GLM 5.2 path.

With the aggregate set, greedy output is byte-identical to main and decodes the 1,471-token prompt at 22.28 tok/s against main's 22.21, with 146,046 encoder acquisitions over the run against 86,610 on the default path -- the unfused dispatch structure is back, so the switch restores the paths and not just the numbers.

Two of #954's pieces could in principle apply to GLM 5.3 Flash; neither
does in practice:

- **Greedy chain decode** keeps the token id on the GPU so the host's
  `waitUntilCompleted`, logits readback, argmax and re-encode leave the
  per-token critical path; #954 measures the boundary at about 0.5 ms of GPU
  idle per DeepSeek token and gains 1.75%.  Here `DS4_METAL_GPU_BUSY_PROFILE`
  over 16 decode tokens accumulates 35.2 ms of GPU time per 35.3 ms token:
  the GLM decode loop already flushes command buffers every four layers, so
  the GPU idles about 0.1 ms per token, a 0.3% ceiling.  Not worth the
  device-resident token ring, GPU argmax and session plumbing it takes.
- **Batch indexer-query pruning** skips the indexer query projection, RoPE,
  QAT and weight projection for prefill batches whose attention is entirely
  within the dense window; #954 gains 1.3-1.8% of prefill.  GLM's indexed
  prefill already does this: `use_causal_range_select` is true while the
  chunk's rows fit the 4096-row window, and the query projection is inside
  `if (!use_causal_range_select)`.

The rest of #954 is DeepSeek attention and MoE kernels (raw-layer gathered
attention, packed32, RB4-staged prefill rows, sum6/attn-out HC fusions) with
no GLM 5.3 Flash counterpart on the same shapes.

## A trap when verifying a decode-path change

`ds4-bench --dump-frontier-logits-dir` writes one file per **frontier**, which
is the logits at the end of prefill.  It does not exercise the single-token
decode graph at all.

This was found the hard way.  A change that skipped the FFN-side mHC producer
on every KDA layer -- catastrophic, garbage output after the first token --
produced frontier logits **bit-identical** to the baseline, because the bug was
entirely in the decode path the dump never touches.  A four-token greedy
generation caught it immediately.

For anything that touches decode, compare every F32 logit after actual
single-token evaluations under identical token inputs. Greedy text and argmax
are useful additional checks but do not establish bit-perfect output: the
original BF16 HC fusion changed logits throughout 128 teacher-forced steps at
2K and 8K while retaining every argmax. Greedy feedback also diverged at 2K,
but not at 8K. Cover continuation after prefill and snapshot restore. The
frontier dump remains useful for prefill checks.

## A trap when A/B-testing a shader change

`ds4_gpu_full_source()` reads `metal/*.metal` from disk at run time and there
is no embedded fallback, so building two binaries around a shader edit does
**not** compare two shaders -- both read whatever is on disk when they run.
Use the per-file overrides (`DS4_METAL_GLM53_KDA_SOURCE` and its siblings) with
a single binary instead.  A measurement in this file was wrong for exactly
this reason before it was caught.

## Scope and caveats

- This is a **model-file** change, not an engine change.  It does not speed up
  an artifact you already have; it produces a better one.
- Why the shipped artifact is BF16 is not established here.  It contradicts the
  repo's own quantizer, which suggests the artifact pipeline rather than a
  deliberate choice, but if it was deliberate the fix belongs upstream.
- Quality evidence is one perplexity run on one text plus a greedy generation.
  That is good evidence for a near-lossless type like Q8_0, not proof.
- `--artifact q4` also specifies Q8_0 for the embedding and output tensors,
  which are BF16 here too (~1.2 GiB more per token through the LM head).  This
  was subsequently done: `--tensors head,embd` covers them, and converting the
  head is worth a further +1.80% decode over a KDA-only artifact.  `token_embd`
  is deliberately not in the default -- it is a single-row lookup per token, so
  it saves resident memory rather than decode bandwidth.
- Neither the constants recorded below nor a Q4_K KDA variant were measured.
  The tool accepts `q4_K` as a target, which would take KDA to 2.39 GiB, but
  Q4_K on attention projections is a materially bigger quality question than
  Q8_0 and was not attempted.

## Prefill, measured

Prefill had never been swept.  Five constants that shape it were compile-time
`#define`s with no override, and two of them interact, so each is now
separately settable -- `DS4_GLM_PREFILL_CHUNK_TOKENS`,
`DS4_GLM_FULL_ATTN_LAYER_FLUSH_TOKENS`, `DS4_GLM_FULL_ATTN_CAP`,
`DS4_GLM_FULL_ATTN_STREAMING_CAP`, `DS4_GLM_PREFILL_SCORE_SCRATCH_MB`.
Defaults are unchanged: logits with the knobs unset and with them set to the
old constants match at max|delta| = 0.

### Chunk size, with layer flushing held constant

The document previously warned that the chunk (2048) and the layer-flush
threshold (2048) are the same number against a strict `>`, so raising the chunk
also switches per-layer flushing on -- two changes, not one.  Pinning the flush
threshold separates them.  Prefill tok/s at ctx 16384:

| chunk | flush off | flush on | default |
|---:|---:|---:|---:|
| 1024 | 354.40 | 355.00 | 354.34 |
| 2048 | 394.30 | 394.57 | 394.17 |
| 4096 | 393.98 | 394.45 | 394.12 |
| 8192 | 394.05 | 394.43 | 393.97 |

Two results.  **Layer flushing does not matter at all** -- every column agrees
to 0.2%, so the coupling the doc warned about is real in the code and
immaterial in practice.  And **the default chunk of 2048 is already optimal**:
1024 costs 10%, while 4096 and 8192 buy nothing.  Confirmed at ctx 32768, where
2048/4096/8192 give 388.68/388.31/388.50.

Raising the chunk is not free elsewhere, either.  Context buffers at ctx 4096
grow 1.62 -> 3.04 -> 5.88 GiB across chunk 1024/2048/4096, so 4096 would cost
nearly 2 GiB for no throughput.  The GLM 5.2 path's 4096 is not an argument for
changing this one.

### The full-attention cap asymmetry is backwards

`glm_graph_full_attention_cap` gives the SSD-streaming path 8192 and the
fully-resident path 4096, which looked like the memory-constrained machine
getting the larger window.  Forcing each value on the resident path, ctx 16384,
interleaved, n=6:

| cap | prefill | decode |
|---:|---:|---:|
| 4096 | 394.23 (sd 0.03) | 23.17 (sd 0.02) |
| 8192 | 379.23 (sd 0.10) | 23.15 (sd 0.02) |

**The larger window costs 3.81% of prefill and nothing on decode.**  So 4096 is
not the conservative choice, it is the fast one, and the resident default is
right.  Whether 8192 pays for itself on the streaming path by reducing
re-streaming is untested here -- `DS4_GLM_FULL_ATTN_STREAMING_CAP` exists to
try it.

### The 256 MiB score scratch does bind, but not where it hurts yet

`DS4_GLM_METAL_INDEXED_PREFILL_SCORE_SCRATCH_MB` clamps rows scored per
dispatch.  It is not dead: score columns are `compact_cap / 4`, so the budget
starts biting above 131072 allocated context.  Observed, chunk 2048 throughout:

| ctx_alloc | score_rows | scratch |
|---:|---:|---:|
| 65536 | 2048 | 128 MiB |
| 131072 | 2048 | 256 MiB |
| 262144 | **1024** | 256 MiB |
| 524288 | **512** | 256 MiB |

Raising the budget to 1024 MiB restores 2048 rows at a 524288 allocation.  The
model context limit is 1048576, so this is reachable, not theoretical.

It does not currently cost anything measurable, though: holding the allocation
at 524288 and varying only the budget, a 16384-token prefill runs at 393.84
tok/s with score_rows=512 against 394.30 with 2048 -- **0.12%, noise**.  A
prefill long enough for scoring to dominate was not measured; each run at ctx
65536 with that allocation exceeds ten minutes.  So: the clamp is real, the
knob to lift it exists, and nobody has yet shown it matters.

## Untested constants noticed while reading

Recorded so the next person does not re-derive them.  None were measured.

All four prefill entries that used to sit here have been measured; see
"Prefill, measured" above.  In summary: the chunk default of 2048 is optimal,
per-layer flushing does not matter, the 8192 full-attention cap is 3.81% slower
than 4096 rather than more generous, and the 256 MiB score scratch does clamp
above 131072 allocated context but costs nothing measurable at the prefill
lengths tested.

## Additional exact Metal campaign, 2026-09-05

This campaign starts from the corrected BF16 HC producer and fenced GLM
router described above. It uses the original resident
`GLM-5.3-Flash-Q4_K.gguf` with BF16 KDA projections on the 80-core M3 Ultra
with 512 GiB RAM. Model weights, precision choices, and compiler math flags
are unchanged. The earlier inexact HC speedup is not the reference.

Six additional policies are enabled only for the measured GLM shapes on
resident, single-device M3 Ultra execution:

| Policy | Work changed | Individual rollback environment variable |
| --- | --- | --- |
| Router top eight | One SIMDgroup keeps nine expert scores per lane; ordered selection replaces the finite-input 512-wire sort | `DS4_METAL_DISABLE_GLM53_ROUTER_TOP8` |
| QK-low decode | 32 independent output columns per threadgroup, using the same Q8_0 dot helper | `DS4_METAL_DISABLE_GLM53_DECODE_QK_LOW` |
| Short BF16 rows | Sixteen SIMDgroups for 128-to-8192 projections; one unchanged reduction per output row | `DS4_METAL_DISABLE_GLM53_BF16_SHORT_ROWS` |
| DSA score tiles | Two selected rows per tile up to 2,051 entries, including sentinels; sixteen above that | `DS4_METAL_DISABLE_GLM53_DSA_SCORE_TILE` |
| KDA decode rows | Interleave four independent state rows while retaining each row's arithmetic and SIMDgroup | `DS4_METAL_DISABLE_GLM53_DECODE_KDA_VALUES4` |
| KDA input projections | Combine Q/K/V, f_a/g_a and beta into one 24,896-row BF16 dispatch | `DS4_METAL_DISABLE_GLM53_KDA_INPUTS` |

Setting any listed variable restores that policy's reference path. The existing
`DS4_METAL_DISABLE_GLM53_FLASH_TUNING` disables all six as well as the branch's
earlier tuning. The broad decode rollback disables the new decode paths;
batched router selection retains its separate prefill policy. Existing
QKV/gate-pair/gate-trio rollbacks and substage ablation/repetition prevent the
combined input dispatch. Other devices, TP and streaming retain their previous
paths. No additional model-sized weight copies are allocated.

The router preserves lower-ID tie ordering, probability stores, the ordered
eight-term denominator, and normalization arithmetic. Integer-bit NaN detection
selects an emulation of the original sorting network, preserving exceptional
ordering. This fallback is slower; ordinary finite scores use the fast path.
The KDA row interleaving changes no cross-row reduction: normalization still
uses four SIMDgroups, and each recurrent row retains its original dot/FMA order.
The six input projections retain the BF16 row helper and write distinct graph
allocations; f_b/g_b still run after their low-rank inputs are complete.

Final uninstrumented measurement: three runs per arm, ordered reference /
candidate / candidate / reference / reference / candidate. Both binaries and
runtime Metal sources were frozen. Each run used `promessi_sposi.txt`, 256
greedy decode tokens at both 2,048 and 8,192 frontiers, and 16,384 allocated
context. All six generated texts match. Arithmetic means of the recorded
rates are:

| Context | Corrected reference decode | Final decode | Additional gain | Reference prefill | Final prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 26.050 tok/s | 27.040 tok/s | +3.80% | 510.237 tok/s | 510.870 tok/s |
| 8,192 | 25.400 tok/s | 26.367 tok/s | +3.81% | 459.803 tok/s | 460.483 tok/s |

Decode ranges are 26.01–26.08 versus 27.03–27.05 at 2K, and 25.39–25.41
versus 26.35–26.38 at 8K. The roughly 0.12%/0.15% prefill differences are too
small to claim a meaningful prefill gain. The 8K prefill column measures the
6,144-token continuation from 2K; fresh full-8K prefill is a different benchmark.
Results and per-run CSVs are in the artifact directory's `final-benchmark/`.
These are additional gains over the corrected branch, measured together;
they are not sums of the component screening percentages.

Validation of the combined production implementation:

- `make -j4 test-glm53-kda` and `./ds4_test --metal-kernels` pass. CPU-only
  syntax checking of `ds4.c` also passes.
- Regression checks compare output bits against individual and aggregate
  rollback, with explicit dispatch coverage. They cover router ties, signed
  zero, saturation, infinities and NaN payloads; signed/subnormal quantized
  scales; DSA sentinels and the first untuned count; all six BF16 projections;
  and independent persistent KDA histories across repeated updates and shapes.
- Both default and `--quality` greedy traces at 2K and 8K match the corrected
  reference byte-for-byte, including the restored-session continuation. Each
  trace has 258 frames and 159,840,288 bytes. SHA-256 values remain
  `3c600ac8994cf3327b2b19e7d2de0d8ad6b0dc261fe7c44eee2358974b0d1bf1`
  (default) and
  `fc83d2ab8b6bdb8dca7a4d259f03f3a35e70e77a9ad855d6520ec58c1250bc2a`
  (quality).
- A different segment of the novel at 16K, with 64 greedy updates and a 32K
  context allocation, also matches every logit and token bit: 65 frames,
  40,269,840 bytes, SHA-256
  `f49ed697561fdb2206b5414cddcadfcf9cd88b3ee0cd9af834347ec451607d22`.

Prefill experiments were measured with fresh 8K ABBA runs and full frontier
logit comparisons. Paired Q4_K gate/up/SwiGLU variants changed throughput by
-2.77% (64-row tiles), -0.34% (tail-culling 64-row tiles), and +0.14% (compact
tail-culling 32-row tiles). Explicit four-byte Q4_K dequant loads lost 2.48%.
KDA prefill recurrence with four values per SIMDgroup lost 0.55%; prepare
blocks of 8/16/64 versus 32 changed -0.16%/-0.002%/+0.03%. All were exact in
these tests, but none justified replacing the existing defaults.

Lossless BF16 weight packing was also exact but essentially flat, both for
large projections and the HC producer. Reducing the HC producer from sixteen
physical SIMDgroups to eight/four preserved its virtual reduction tree but
slowed the kernel from about 37 to 41/50 microseconds. The corrected producer
fusion remains enabled with its original weight layout.

Raw sources, harnesses, immutable reference/candidate trees, tests, traces and
benchmark logs are retained under
`/Users/jw/.cache/ds4-bench/glm53-exact-campaign-20260904-233959/`.
The untracked `glm53flash_astra_gen_ideas.md` records the experiments and
remaining proposals. MLX and llama.cpp were used only as read-only references.
These results establish exactness for the tested workloads and hardware;
they do not prove equivalence for every possible input or Metal compiler.
