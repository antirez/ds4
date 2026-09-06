# Qwen3.8 MTP optimization, round 3

This round tunes MTP decode and shared prefill on Apple M3 Ultra (512 GiB). The baseline is `ecf1da3`, after the main rebase. Both builds use the Qwen3.8 Flash-Next Q4KImatrix model, the Q4_1 PLE sidecar, and default Metal settings. Weight formats and arithmetic order are unchanged.

## Performance

Three interleaved pairs per short prompt, following separate warmups. These are ratios of decode-rate medians; model loading and prefill are excluded.

| Prompt | Baseline t/s | Selected t/s | Change |
| --- | ---: | ---: | ---: |
| hamlet | 61.47 | 62.93 | +2.38% |
| fibonacci | 74.66 | 76.52 | +2.49% |
| explanation | 65.92 | 67.48 | +2.37% |

The two full sweeps run in opposite build orders. Each frontier generates 128 greedy MTP tokens with 262,401 context positions allocated. Values below average the two observations per build. Both builds force frontier snapshots and restore into fresh sessions. Prefill measures newly appended prompt tokens; snapshot/restore work is outside its timer. The first two frontiers use 4K batches; the large-prefill optimizations begin at the 16K frontier in this incremental sweep. These small samples do not establish statistical significance or a gain on every workload.

| Context | Prefill t/s, baseline → selected | Change | MTP decode t/s, baseline → selected | Change |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1152.67 → 1149.88 | -0.24% | 52.63 → 53.78 | +2.19% |
| 8,192 | 1124.92 → 1125.80 | +0.08% | 51.44 → 52.59 | +2.25% |
| 16,384 | 1176.91 → 1185.65 | +0.74% | 52.50 → 53.77 | +2.42% |
| 32,768 | 1180.06 → 1186.32 | +0.53% | 49.39 → 50.47 | +2.18% |
| 65,536 | 1173.74 → 1180.99 | +0.62% | 47.69 → 48.70 | +2.14% |
| 131,072 | 1158.98 → 1166.01 | +0.61% | 51.19 → 52.15 | +1.89% |
| 262,144 | 1128.63 → 1135.81 | +0.64% | 45.38 → 46.32 | +2.06% |

All measured MTP token sequences and short-prompt acceptance counts match. The [portable results](qwen38-mtp-round3-results.json) retain every observation, commands, frozen source/binary hashes, and intermediate screens. Diagnostic timings are excluded from these performance tables.

## Numerical checks

The final 262K diagnostic compares **1,828 full-vocabulary FP32 vectors** from target passes (both verifier rows), predictor passes, and prefill/replay. Every value is byte-identical to the baseline; maximum absolute difference is zero. It spans seven doubling frontiers from 4,096 through 262,144 and 128 MTP output tokens per frontier. Both runs use the same instrumented engine C object and their respective frozen Metal runtime objects and sources. This establishes parity on these finite histories, not general model-quality equivalence.

Exact sampling uses seed 123, temperature 0.7 and top-p 0.8. Its captured logits, output, acceptance counters and per-cycle traces match. Separate 1/2/3-token budget checks also match. The target-only frontier sweep preserves the serialized frontier logits and generated text; its timing observations are retained in the results.

The full Qwen Metal kernel suite and speculative planner pass. New GDN fixtures compare split and full batches at 8,191/8,192/8,193 tokens, including every output, final recurrent state and first-token snapshot, byte-for-byte. Existing tests cover Q4K NR1/NR2 and SIMD group counts 1–8, and HC normalization outputs and guarded tails. `make -j8 all` passes.

The single target-only sweep measured decode changes from -0.41% to -0.08%; it does not show a target-only decode gain.

After measurement, the comparison harness was tightened to fail on acceptance-counter drift even when generated output matches. A model-free end-to-end check verifies that rejection and raw-prompt/token-budget handling. The results retain both measured and final harness hashes; runtime and kernel sources remain identical to the measured build.

## Implementation

- Q4K gate/up uses NR1/NSG8 for two-token MTP on M3 Ultra, extending the existing one-token default.
- Large GDN prefill uses four SIMD groups per threadgroup on M3 Ultra at T≥8,192, 16 key heads, 48 value heads and head width 128. Each group owns independent value rows.
- HC normalization keeps separate reduction scratch for each of its eight chunks, removing eight overwrite barriers without combining partial sums or increasing scratch size.

The initial HC change alone measured only +0.07% whole-model prefill on a 17,623-token prompt. The subsequent GDN screen measured +0.57% over that intermediate build. These selection screens are not compounded into a claimed final gain. Smaller prefill batches and other devices retain their dispatch geometry. CUDA, CPU, SSD and distributed runtime sources are unchanged; physical validation in this round is limited to local Metal.

Detailed local artifacts are retained under the ignored `OUT/qwen38-mtp-round3` directory on the test machine. The first long diagnostic was stopped after discovering a missing doubling option; its partial baseline-only results are excluded. The corrected run explicitly passes `--step-mul 2`.
