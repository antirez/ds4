# Original Q4_K: main, previous tip and M3 Ultra merge

All performance figures on this page use only the unchanged `GLM-5.3-Flash-Q4_K.gguf` (190,875,526,464 bytes). KDA and embedding/output-head weights remain BF16. Neither `-kdaQ8` nor `-kdaHeadQ8` is used in this comparison.

Across these four frontiers, the merge adds **1.5–2.5% generation throughput** over the previous tip, with no resolved prefill change. Relative to current main, the branch gains **28.5–29.6% generation** and **14.9–18.9% prefill**. The 62,174-token new-arm result includes the opt-in DSA selector.

The integration merge is `b4f26df3c7238eb2d2b13f27a52b60c87b483b76`, with parents `224e7669abac9fa64580e71a9a1581fa0c852d2a` and `8ad81dd02c6dd10db33a04e4d657162de6a494e8`. Its complete tree equals the validated experimental tip. Subsequent integration documentation commits do not change inference sources.

## What changed

- Fused the router projection/selection with independent shared-expert gate/up work, preserving the existing arithmetic and reducing decode dispatch overhead. Enabled by default for the supported resident, serial M3 Ultra path; rollback: `DS4_METAL_DISABLE_GLM53_ROUTER_SHARED=1`.
- Added a guarded histogram/candidate DSA selector. Unsupported scores and shapes retain the original GPU sort fallback. This remains opt-in through `DS4_GLM_ENABLE_TOPK_FAST=1`, which is enabled for the new arm below. The 2K/8K/32K cases are below its threshold and measure router fusion; the 62,174 case exercises the combined path.
- Preserved exact GLM tool-call bytes through checkpoint replay and bounded checkpoint prefix lookup work. Fixed reuse of partial GPU completion/scratch state after command-buffer failure, non-finite host guards under fast-math, and invalid selected attention rows.
- Included the separately validated, opt-in mixed Q8/BF16 KDA input fusion. It is inactive for this original BF16-KDA model. The fork's expert-bank prefill scheduling was not integrated, and the untracked-model default was not changed.

Adapted from [IngeniousIdiocy/ds4 at 95eb218](https://github.com/IngeniousIdiocy/ds4/tree/95eb218614868284fb6a9350b4be9f32c0d57575), with local arithmetic-preserving guards, fallbacks and recovery fixes. The retained [experiment report](REPORT.md) contains the full development and validation history.

## Generation throughput (tokens/s)

| Context | Main `6289c51` | Previous `224e766` | New `b4f26df` + DSA | New vs main | New vs previous |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 20.98 | 26.64 | 27.10 | +29.14% | +1.73% |
| 8,192 | 20.84 | 26.42 | 26.87 | +28.91% | +1.68% |
| 32,768 | 20.59 | 26.06 | 26.45 | +28.48% | +1.50% |
| 62,174 | 20.40 | 25.78 | 26.43 | +29.56% | +2.52% |

## Prefill throughput (tokens/s)

| Context | Main `6289c51` | Previous `224e766` | New `b4f26df` + DSA | New vs main | New vs previous |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 429.40 | 510.54 | 510.43 | +18.87% | -0.02% |
| 8,192 | 378.57 | 435.85 | 435.87 | +15.13% | +0.00% |
| 32,768 | 371.76 | 427.93 | 427.97 | +15.12% | +0.01% |
| 62,174 | 358.87 | 412.31 | 412.35 | +14.90% | +0.01% |

Values are means of two runs per arm; percentage changes use unrounded means. The maximum within-arm spread across these frontiers is 0.38% for generation and 0.18% for prefill. Small differences should be read in that context; this is one machine and one prompt.

## Method

Apple M3 Ultra, 80 GPU cores, 512 GiB RAM, macOS 26.5.2 (25F84). Current `origin/main` was fetched and pinned to `6289c516273979173abbc062209a81dd3706b804`. Previous branch tip: `224e7669abac9fa64580e71a9a1581fa0c852d2a`. New merge: `b4f26df3c7238eb2d2b13f27a52b60c87b483b76`.

Six serial processes in main → previous → new → new → previous → main order, two runs per arm. Each arm has the same mean position in the sequence, balancing linear drift. Each binary runs from its own frozen source directory so Metal shaders match the measured commit. The model is fully resident; SSD streaming, tracing and dispatch statistics are off. No model conversion occurs.

Each process starts from a fresh session, prefills to 2,048, 8,192, 32,768 and 62,174 tokens, and generates 256 greedy tokens at each frontier. The first prefill row measures 2,048 tokens; subsequent rows measure incremental additions of 6,144, 24,576 and 29,406 tokens. They are not separate cold full-prompt prefill measurements. Internal session snapshots restore prompt state after generation; their save/restore cost is outside the timing windows. No previously saved prefix is loaded in this comparison. `gen_tps` covers all 256 generation steps.

Clear existing `DS4_`, `MTL_` and `ASTRA_` overrides, then run the following from each matching build directory. Only the new arm additionally sets `DS4_GLM_ENABLE_TOPK_FAST=1`:

```sh
DS4_BENCH_FORCE_SNAPSHOT=1 ./ds4-bench \
  -m /path/to/GLM-5.3-Flash-Q4_K.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 62174 --step-mul 4 \
  --ctx-alloc 65536 --gen-tokens 256 --csv /tmp/results.csv
```

Prompt SHA-256: `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`. [Manifest](evidence/q4-merge/manifest.json), [full commands and raw rows](evidence/q4-merge/results.json), [unrounded summary](evidence/q4-merge/summary.json), and the six adjacent CSV files preserve the measurements.

## Exactness and previous long-context confirmation

The merged inference sources equal `0f211542`, whose complete captured vocabulary logits match previous tip `224e766` byte for byte at 2K, 8K, 62,174 and 300,000 tokens on the original Q4_K model. This is the exactness comparison against the previous branch tip, not a blanket claim of byte equality to main. Kernel, GPU recovery, checkpoint/server and prefix-lookup oracle checks passed. No-mistakes run `01M24PAZH0HS08TZVK7WA4XD19` passed the final source review and lint at `8ad81dd`, after the earlier complete source reviews and fixes.

The earlier 300,000-token generation-only confirmation measured **23.300 → 24.770 tok/s (+6.31%)**, previous tip versus the same merged inference sources with router + DSA. It used restored identical prefixes, 256 greedy steps and two runs per arm in ABBA order, with allocation 301,000. It provides no measured prefill rate or current-main result at 300K and is not pooled with the fresh comparison above. See [the original timing records](evidence/recovery-timing.json); only entries naming `GLM-5.3-Flash-Q4_K.gguf` apply here.
