# Pre-M5 Q2 decode port results (M3 Ultra)

This records the port of the exact M5 decode work to the resident pre-M5
Apple-Silicon path. Performance was measured on an Apple M3 Ultra (80 GPU
cores, 512 GiB) with:

```text
gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
```

The starting tree was `2ffda8d`. The local canonical 2,048-token frontier
SHA-256 is:

```text
62e69ac09a921f5bd88503b24d6b4b3d06fc983766db10f7af659ffabfc6016b
```

That frontier artifact covers prefill only. Decode correctness below comes
from the balanced per-token full-vocabulary A/B harness.

## Promoted pre-M5 defaults

Seven ordinary Metal kernel paths from the M5 campaign measured positive and
bit-exact on M3:

1. Q8 q_a/KV plus ratio-4 quad compressor projection/store.
2. The ratio-128 form of the Q8 q_a/KV plus compressor compound kernel.
3. Ratio-4 attention/indexer compressor emit finalization.
4. Exact ratio-4 compressor pool (pack, softmax/product, and legacy row sum).
5. Packed split-K FlashAttention reduction with fused inverse RoPE.
6. HC norm/mix producer continuation through split/Sinkhorn and RMSNorm.
7. Concurrent shared/routed full FFN for the narrow resident IQ2XXS/Q2_K,
   top-6, one-token shape.

The first six are independent of routed-expert quantization and also improved
the resident pre-M5 MXFP4 model; their attention/compressor shape gates still
apply. The concurrent FFN remains guarded by the exact Q2 shape. All existing
quality, SSD/cold, TP, debug/profile, resource-limit, and unsupported-shape
fallbacks remain in place.

The existing device policy admits these defaults on the standard M1-M4 Metal
path and retains the M5 policy independently. Performance and full decode
exactness were measured locally on M3 Ultra; other pre-M5 generations retain
all capability/resource fallbacks and the rollbacks below.

The exact resident IQ2XXS/Q2_K tape also defaults to the revalidated 2/32
command-buffer split instead of 4/0 between positions 128 and 2815. It requires
43 layers, 256 experts, top-6, and the exact 4096/2048 IQ2XXS/Q2_K row shapes.
Outside that admitted window it retains 4/0. Explicit split variables override
the automatic values: to force 4/0, set both the first split to 4 and the second
to 0.

Use this aggregate rollback for the whole port (including 2/32):

```text
DS4_METAL_DISABLE_PRE_M5_DECODE_PORTS=1
```

Individual rollbacks are:

```text
DS4_METAL_DISABLE_PRE_M5_QKV_PAIR_QUAD_FUSE=1
DS4_METAL_DISABLE_PRE_M5_QKV_PAIR_COMPRESSOR_FUSE=1
DS4_METAL_DISABLE_PRE_M5_COMP_FINALIZE_FUSE=1
DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_EXACT_POOL_RATIO4=1
DS4_METAL_DISABLE_PRE_M5_FLASH_ATTN_PACKED32_REDUCE=1
DS4_METAL_DISABLE_PRE_M5_HC_PRODUCER_PRE_NORM_FUSE=1
DS4_METAL_DISABLE_PRE_M5_PARALLEL_FULL_FFN=1
DS4_METAL_DISABLE_PRE_M5_Q2_DECODE_SPLIT2_32=1
```

Pre-M5 individual and aggregate rollbacks dominate the experimental force
enables. The original `DS4_METAL_DISABLE_M5_*` switches remain valid for M5
and are also dominant.

## Controlled decode evidence

The final bundle was compared with the aggregate rollback at a fixed 4/0
schedule over 1,024 measured tokens. The harness alternates variant order and
session assignment and includes exact non-EOS selection:

```sh
./speed-bench/metal_decode_schedule_bench \
  -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --prompt-file speed-bench/promessi_sposi.txt --prefix-tokens 2048 \
  --ctx 4096 --warmup 16 --tokens 1024 --include-selection \
  --control-first 4 --control-second 0 \
  --candidate-first 4 --candidate-second 0 \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_DECODE_PORTS
```

| arm | decode tok/s |
|---|---:|
| all ports ON | **44.064** |
| aggregate rollback | 40.650 |

This is **+8.40%**. Every frontier and selection was bit-identical:

```text
exact_rows=1041
exact_floats=134580480
exact_selected_ids=1040
```

Final-tree inverse screens at the production 2/32 schedule were also exact
(`273` rows, `35,293,440` floats, `272` selected IDs each):

| default removed | default ON | disabled | retained gain |
|---|---:|---:|---:|
| ratio-4 QKV/quad compound | 44.5167 | 44.2609 | +0.58% |
| ratio-128 QKV/compressor compound | 44.8930 | 44.6584 | +0.53% |
| compressor finalizer | 44.8763 | 44.4313 | +1.00% |
| exact ratio-4 pool | 45.0071 | 44.8732 | +0.30% |
| packed FlashAttention | 44.9735 | 44.5313 | +0.99% |
| HC producer/pre-norm continuation | 44.9322 | 43.6619 | +2.91% |
| concurrent full FFN | 44.8299 | 44.2904 | +1.22% |

Three 512-token forward/inverse processes measured a median 44.3956 tok/s for
2/32 versus 44.2425 for 4/0. Every paired result favored 2/32 (+0.29% to
+0.35%, median **+0.32%**) and each made all 68,389,120 logits and 528 selected
IDs exact. The comparison used control 4/0 and candidate 2/32; the inverse
process swapped the variants. A separate 512-token run from a 128-token prefix
measured the same +0.29% direction, covering early decode as well as the 2K
frontier.

As a cross-model safety/performance check, the pre-M5 MXFP4 model measured
42.6725 tok/s with the applicable ports versus 39.9685 with the aggregate
rollback (+6.76%), exact over 273 rows and 272 selected IDs.

## Official interleaved replicas

Command (three ABBA-style interleaved ON/OFF pairs):

```sh
./ds4-bench \
  -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 2048 --ctx-alloc 2081 --step-incr 2048 \
  --gen-tokens 32 --prefill-chunk 4096 --power 100 --warm-weights \
  --csv OUT/speed.csv --dump-frontier-logits-dir OUT/logits
```

The rollback arm set `DS4_METAL_DISABLE_PRE_M5_DECODE_PORTS=1`; the ON arm
unset it. All other decode experiment variables were cleared.

| arm | prefill tok/s | decode tok/s | steady decode | first token |
|---|---:|---:|---:|---:|
| aggregate rollback | 607.53 med | 41.08 med (40.98-41.34) | 41.44 med | 27.45 ms med |
| ports ON | 606.65 med | **44.18 med (44.13-44.23)** | **44.52 med** | 27.20 ms med |

Decode improved **+7.55%** and steady decode **+7.43%**, with no meaningful
prefill or first-token regression. All six frontier files have the canonical
local SHA above.

## Rejected or redundant M5 ports

- IQ2 pair-SwiGLU pack2 was exact but slower both alone (-0.25%) and with the
  concurrent FFN (-0.26%). It remains M5-only.
- The router projection/select continuation was not exact on M3: all 129,280
  logits differed at the first compared decode frontier. It remains M5-only.
- The aligned Q8 HC epilogue was only +0.16% before concurrent FFN and is
  bypassed by the retained concurrent path. It remains M5-only.
- HC cluster2 was exact and +0.55%, but the stronger HC producer/pre-norm
  continuation subsumes that kernel on the normal full-decode path.
- Persistent zero-mask reuse was already an M3 default and needed no port.
- A 4/16 Q2 split was effectively flat; the historical 2/32 schedule was
  revalidated instead.

## Validation

- Full 1,024-token balanced decode A/B: exact as reported above.
- Every individual inverse screen: exact as reported above.
- Three forward/inverse schedule processes plus the short-prefix process:
  exact as reported above.
- Aggregate rollback with all seven force-enable variables deliberately set:
  still disabled the full bundle and was exact over 81 rows/80 selected IDs.
- `tests/test_mxfp4_metal`: pass.
- `ds4_test --metal-kernels`: pass.
- Q2 `ds4_test --metal-tensor-equivalence`: five cases, `rms=0`,
  `max_abs=0`, identical greedy results.
- Q2 `tests/test_metal_session_batch`: pass, exact multi-session/mixed-batch
  logits.
- Full `make test` with the resident MXFP4 default model: pass, including
  long context, tool replay, SSD streaming cache pressure, Metal kernels, and
  tensor equivalence.
- CPU-only test-hook build is covered by `make test`.
