## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

### DSpark speculation on M3 Ultra (measured, not yet profitable)

DSpark non-strict decode was measured end to end on M3 Ultra with the MXFP4
model and the 0731 support GGUF.  Three systemic costs keep it below plain
decode (~43.7 t/s) today; scheduler tuning alone cannot fix them:

1. The verify pass runs the speculative suffix through the generic batch
   prefill kernels (`metal_graph_encode_layer_batch`): ~46-60 ms per verify
   vs the ~29-33 ms memory floor for two rows (the extra routed-expert reads
   are inherent).  A commit-only stage decomposition showed a uniform
   1.5-2.5x per-stage excess (routed MoE 14.3 ms vs 5.9 decode, HC pre
   7.2 vs 1.7, output projection 7.1 vs 4.8, attention 4.4 vs 2.1).
   The fix is the N<=6 microbatch verifier on the decode-grade kernels that
   the verifier's own header comment calls out as "not yet" written.
2. The draft propose chain costs ~3-8 ms/cycle and its confidence gate
   (`sigmoid(confidence0) >= threshold`, Metal default 0.6) declines on
   45-75% of cycles; each such cycle still pays the propose before falling
   back to one plain decode.  Plain decode inside a DSpark session measures
   a normal ~23.3 ms (the hidden-state capture is not a decode tax).
3. Per-cycle bookkeeping (checkpoint, snapshot, commit, propose-fail
   waste) accounts for a further ~4-5 ms/cycle beyond the measured
   propose+verify+decode components.

Scheduler and confidence knobs were swept: `DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP=0`
(retry the draft every cycle) raised proposals from 91/179 to 125/147 cycles
and accepted drafts to 101 (80.8% accept); the best combination measured
39.5 t/s at `--dspark-confidence 0.75` on a code prompt, still below the
43.7 t/s plain-decode equilibrium.

Two dispatch-routing microbatch increments were also built, validated, and
measured negative before reverting: per-row routed MoE through the
single-token static-trip kernels (bit-identical output, verify_layer
unchanged at 1225 vs 1217 ms) and per-row HC pre through the decode fused
producer kernel (valid non-strict output, 39.3 vs 40.2 t/s).  Both show the
verify excess lives in the batch kernels' execution, not in dispatch
routing; the microbatch build must write genuine small-N batched kernels.  Strict mode (`--dspark-strict`)
measures 43.07 t/s, i.e. no gain, as accepted blocks are re-run through
one-token decode to stay byte-identical.

### Metal decode stage GPU counters

The end-and-wait stage profiler (`DS4_METAL_DECODE_STAGE_PROFILE=1`) adds a
synchronization per stage boundary and changes the schedule, so its numbers
are inflated by per-boundary waits.  The stage-counter diagnostic keeps the
production token mostly intact: every stage boundary commits the open batch
command buffer without waiting, so the GPU queue stays fed, and each stage's
GPU busy span is printed after the token:

```
DS4_METAL_DECODE_STAGE_PROFILE=1 DS4_METAL_STAGE_COUNTERS=1 ./ds4 -m ds4flash.gguf \
    -p "Write a short story." -c 8192 -n 24 --temp 0
```

Both env vars are required: the first arms the boundary macros, the second
switches them from end-and-wait to commit-only sampling.  The concurrent
shared-expert/routed-MoE overlap stays armed under counters (only the
serializing profiler disables it), but the per-stage command buffers queue in
order, so overlapped stages report their serialized costs.  The per-token
`total-cb-busy` line matches the production GPU-busy time (about 22.5 ms on
M3 Ultra at a short context), which is the check that the attribution is
faithful.  M3 Ultra decode at a short context attributes the token roughly as:
routed MoE 5.9 ms, attention output projections 4.8 ms, Q lora path 5.1 ms
(Q-A/KV/compressor quad kernel 41 us + Q-B matvec 59 us per layer), attention
core plus inverse RoPE 2.1 ms, router/shared gate-up 1.9 ms, and about 3.4 ms
of per-layer HC pre/post bookkeeping, with the remaining dense Q8_0 matvecs
streaming at 590-650 GB/s, i.e. at the memory wall.

### Metal decode schedule A/B

Build the balanced, same-engine Metal decode comparison with:

```
make metal-decode-schedule-bench
./speed-bench/metal_decode_schedule_bench \
  -m ds4flash.gguf \
  --include-selection
```

The harness prefills two sessions and alternates both variant order and
variant-to-session assignment. It aborts unless every full-vocabulary logit
row is bit-identical and, with `--include-selection`, both variants select the
same non-EOS token. Use `--candidate-env NAME` to measure a rollback control,
or `--help` to compare explicit split schedules.

To compare the default pre-M5 ratio-4 compressor pack/transpose fusion with the
legacy decode path, including token selection, use:

```
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION \
  --include-selection \
  --tokens 1024
```

The pre-M5 one-token Flash attention-output LOW projection also has an exact
fixed-shape Q8_0 kernel. Compare it with the generic rollback using:

```
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_ATTN_OUT_LOW_Q8_STATIC \
  --include-selection \
  --tokens 512
```

Balanced M3 Ultra A/B runs favored the fixed-shape kernel by 0.53%, 0.53%,
and 0.58% at a 2K-token prefix (43.03/42.80, 42.96/42.73, and 43.24/43.00
tok/s) and by 0.48% at an 8K-token prefix (38.44/38.26 tok/s). An independent
IQ2/Q2-model run at 2K gave 44.47/44.23 tok/s (+0.56%). All 1,909 compared
rows, 246,795,520 full-vocabulary logits, and 1,904 selected token IDs were
bit-identical.
Performance was measured on M3 Ultra; the exact host gate covers the shared
M1-M4 Flash shape and otherwise retains the generic kernel.

### Metal prefill variant A/B

Build the balanced prefill comparison. To compare the default resident pre-M5
MXFP4 pair tail-SIMDgroup cull against the original pair kernel, make the
rollback path the candidate:

```
make metal-prefill-variant-bench
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_PAIR_TAIL_SIMDGROUP_CULL
```

To isolate the default routed-down tail-SIMDgroup cull from the retained pair
default, use its down-specific rollback as the candidate:

```
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_DOWN_TAIL_SIMDGROUP_CULL
```

For resident pre-M5 MXFP4 prefills of 32 through 2047 tokens, the exact
scatter map, compact pair tile, pair/down tail culls, and down half-LUT now
use the same defaults as the established 2K+ path. Compare the complete
short-prefill extension with its aggregate rollback using:

```
./speed-bench/metal_prefill_variant_bench \
  --prefix-tokens 256 \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_SMALL_PREFILL
```

Balanced M3 Ultra A/B medians for the tuned path versus the rollback were
112.43/89.25 tok/s at 32 tokens (+26.0%), 111.95/89.30 at 33 (+25.4%),
180.48/142.59 at 64 (+26.6%), 275.72/219.98 at 128 (+25.3%),
389.27/316.93 at 256 (+22.8%), 512.94/439.39 at 512 (+16.7%),
173.94/157.08 at 1024 (+10.7%), and 632.30/593.70 at 2047 (+6.5%). Every
one of the 64 measured runs produced bit-identical full-vocabulary logits.
Performance was measured on M3 Ultra; the guarded default also covers the
shared resident M1-M4 path.

### Metal batch indexer-query pruning A/B

On zero-prefix ratio-4 layers, the indexer query and its per-head weights are
not consumed until the compressed cache grows beyond the 512-row top-k.  The
resident pre-M5 path now skips the otherwise dead Q projection, RoPE, QAT, and
weight projection for batches of at least 32 tokens while the final compressed
count remains at or below top-k.  The indexer compressor and its persistent
cache/state updates are unchanged.  Compare the pruned path with its rollback:

```
./speed-bench/metal_prefill_variant_bench \
  --prefix-tokens 2048 \
  --warmup-tokens 2048 \
  --repeats 4 \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_BATCH_INDEXER_QUERY_PRUNE
```

Balanced M3 Ultra A/B throughput for the pruned path versus rollback was
115.00/112.98 tok/s at 32 tokens (+1.79%), 277.42/273.81 at 128 (+1.32%),
509.19/502.28 at 512 (+1.38%), 606.13/597.65 at 1024 (+1.42%), and
660.17/651.29 at 2048 (+1.36%).  The last eligible prefix, 2051 tokens, gained
1.41%; 2052 tokens was flat, confirming that the query path remains enabled
when the 513th compressed row first makes top-k selection necessary.  All 56
prefill runs and 7,239,680 compared full-vocabulary logits were bit-identical.
A 2051-token prefix followed across the row-513 transition also matched three
full-vocabulary rows and two selected token IDs exactly.  Performance was
measured on M3 Ultra; the guarded path covers resident single-device M1-M4.

### Metal batch Q/KV finalizer A/B

The M3 resident Flash prefill path now follows vLLM's horizontal Q/KV
finalization schedule while retaining DS4's existing Q-head PSO. It dispatches
that exact Q RMSNorm+RoPE kernel concurrently with a KV-only kernel that folds
KV RoPE, the FP8 round trip, F16 rounding, and raw-ring insertion together.
The default covers 128 through 4096 tokens per dispatch; longer contexts use
the normal 4096-token chunks. A 32-token dispatch regressed and 64 tokens did
not clear the 0.3% acceptance threshold. Compare the retained path with its
serial rollback using:

```
./speed-bench/metal_prefill_variant_bench \
  --prefix-tokens 512 \
  --warmup-tokens 512 \
  --repeats 4 \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_BATCH_QKV_FINALIZE
```

Balanced M3 Ultra A/B throughput for the concurrent path versus rollback was
273.60/272.63 tok/s at 128 tokens (+0.36%), 502.44/500.37 at 512 (+0.41%),
598.07/595.87 at 1024 (+0.37%), 651.27/649.11 at 2048 (+0.33%), and
615.06/612.37 at 8192 (+0.44%). All 56 prefill runs and 7,239,680 compared
full-vocabulary logits were bit-identical.
A 512-token prefill followed by decode also matched 73 full-vocabulary rows and
72 selected token IDs exactly, covering the persisted raw-cache state. The
schedule is based on vLLM's
[fused DeepSeek V4 finalizer](https://github.com/vllm-project/vllm/blob/c8de519917ce549f72132952116185e38b37c95d/csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu#L382-L603),
but keeps Q and KV in separate Metal pipeline states to preserve DS4's exact Q
fast-math code generation.

### Metal prefill indexed-attention four-row staging A/B

The resident single-device pre-M5 indexed-attention prefill path now stages
four raw or F16-compressed K/V rows per threadgroup barrier while consuming
them in the original scalar-kernel row order. The specialization covers
non-quality batches of at least 32 tokens with 64 heads, 512-wide heads,
top-k 512, a 128-row raw window, ratio 4, and more than 512 compressed rows;
SSD streaming, TP2, decode, M5, and other shapes keep the original path.
Compare it with the
one-row staging rollback using:

```
./speed-bench/metal_prefill_variant_bench \
  --prefix-tokens 8192 \
  --warmup-tokens 4096 \
  --repeats 1 \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_INDEXED_ATTN_PREFILL_RB4
```

Balanced M3 Ultra A/B throughput for four-row staging versus rollback was
631.53/626.50 tok/s at 4100 tokens (+0.80%),
620.39/614.76 at 8192 (+0.92%), 594.11/589.36 at 16384 (+0.81%), and
557.91/553.71 at 32768 (+0.76%). The 2048-token boundary, where compressed
rows do not yet exceed top-k, was flat. All 16 active-path prefill runs and
2,068,480 compared full-vocabulary logits were bit-identical. A direct forced
kernel oracle also matched all 1,048,576 attention-output floats exactly while
covering raw-ring wrap, visibility stops, and one- through three-row tails.
Separate 8K and 32K prefix-to-decode checks matched 106 full-vocabulary
frontiers, 13,703,680 floats, and 104 selected token IDs. Eight-row staging was
also exact but was 0.17-0.21% slower at the first two screened sizes, so only
the four-row specialization is retained. Performance was measured on M3
Ultra; the guarded default covers the shared resident M1-M4 path.

The same guarded path now groups sixteen heads per 256-thread workgroup by
having each SIMDgroup update two heads from every staged four-row block. This
halves K/V staging and workgroup count without changing either head's row
order or online-softmax arithmetic. Compare it with the accepted eight-head
RB4 rollback using:

```
./speed-bench/metal_prefill_variant_bench \
  --prefix-tokens 8192 \
  --warmup-tokens 4096 \
  --repeats 1 \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_INDEXED_ATTN_PREFILL_HEADS16_DUAL_RB4
```

Balanced M3 Ultra A/B throughput for heads16 dual RB4 versus heads8 RB4 was
638.45/629.00 tok/s at 2052 tokens (+1.50%), 651.08/637.46 at 4100
(+2.14%), 640.33/626.12 at 8192 (+2.27%), and 613.72/600.81 at 16384
(+2.15%). The first matched 32768-token pair was 576.10/563.42 (+2.25%);
later slots in that process encountered severe system throttling and are not
used for the comparison. All 28 timed full-vocabulary rows were bit-identical.
The forced direct oracle matched all 1,048,576 attention-output floats and
proved both the dedicated rollback and new selector. Separate 8K and 32K
prefix-to-decode checks matched 94 full-vocabulary frontiers, 12,152,320
floats, and 92 selected token IDs.

### Metal batch MoE sum6-to-HC4 epilogue A/B

The resident single-device pre-M5 MXFP4 prefill path now consumes the six
routed expert-down rows directly in the HC4 expand/add/split epilogue. The
kernel preserves the original `s0 + s1 + ... + s5`, shared-expert add, and HC4
post/comb accumulation order while removing the standalone sum6 dispatch and
the routed-output F32 materialization. The default covers batches of 32 through
4096 tokens; longer prefixes use the normal 4096-token chunks. Debug, profiling,
steering, SSD, TP2, quality, decode, and other tensor shapes keep the original
path. Compare the fused path with its rollback using:

```
./speed-bench/metal_prefill_variant_bench \
  --prefix-tokens 8192 \
  --warmup-tokens 4096 \
  --repeats 4 \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_BATCH_MOE_SUM6_HC_FUSION
```

Balanced M3 Ultra A/B throughput for the fused path versus rollback was
663.24/660.81 tok/s at 2048 tokens (+0.37%), 621.83/619.49 at 8192
(+0.38%), 596.50/594.76 at 16384 (+0.29%), and 560.30/558.54 at 32768
(+0.32%). The 32-token point was flat (-0.03%); 128, 512, and 4096 tokens
gained 0.19%, 0.18%, and 0.28%, respectively. All 64 prefill runs and
8,273,920 compared full-vocabulary logits were bit-identical. A direct oracle
also matched all 527,372 HC output floats exactly for a tail shape and the
production 32-by-4096 shape. Separate 8K and 32K prefix-to-decode checks
matched 146 full-vocabulary frontiers, 18,874,880 floats, and 144 selected
token IDs exactly. Performance was measured on M3 Ultra; the guarded default
covers the shared resident M1-M4 shape.

### Metal batch Q8 attention-output-to-HC4 epilogue A/B

The resident single-device pre-M5 Q8 attention-output path now feeds the
aligned 8192-to-4096 output-B matmul tile directly into the HC4 expand/split
epilogue. The specialization preserves the legacy Q8_0 dequantization and
simdgroup-MMA order, stages every 64-by-32 result through an 8 KiB F32
threadgroup tile, and then repeats the original scalar HC4 post/comb
accumulation order. This removes the global F32 `attn_out` write/read and the
standalone HC dispatch without removing the materialized F32 rounding
boundary. The default is deliberately limited to aligned batches of 512
through 4096 tokens; shorter batches, tails, debug/profiling/steering,
quality, SSD, TP2, M5, and other shapes retain the original path. Compare the
fused path with rollback using:

```
./speed-bench/metal_prefill_variant_bench \
  --prefix-tokens 8192 \
  --warmup-tokens 4096 \
  --repeats 2 \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_BATCH_ATTN_OUT_HC_FUSION
```

Balanced M3 Ultra A/B throughput for the fused path versus rollback was
526.16/523.77 tok/s at 512 tokens (+0.46%), 668.43/663.66 at 2048
(+0.72%), 626.47/621.49 at 8192 (+0.80%), and 599.57/596.92 at 16384
(+0.44%). All 44 retained timing runs and 5,688,320 compared
full-vocabulary logits were bit-identical. A direct production-shape oracle
also matched all 262,144 low-rank values and 524,288 HC outputs exactly and
confirmed that the fused path did not touch the dead `attn_out` buffer.
Separate 8K and 32K prefix-to-decode checks matched 14 full-vocabulary
frontiers, 1,809,920 floats, and 12 selected token IDs exactly. A standalone
32K timing process was discarded because sustained thermal throttling changed
slot time from 58 to 142 seconds; the 32K run is correctness evidence only.
Performance was measured on M3 Ultra; the guarded default covers the shared
resident M1-M4 path.

The harness uses one Metal engine and fresh sessions for every run. It warms
both variants with at least 32 tokens, alternates control/candidate order in
ABBA and BAAB blocks, poisons host logit buffers before copying, and aborts
unless every final full-vocabulary logit row is bit-identical. Defaults are an
8192-token prefix, an automatically sized 8193-token context, and two repeats;
use `--help` to override them.
