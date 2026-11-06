# DeepSeek V4.1 Flash router comparison

112 held-out prompts collected from the official DeepSeek API on September
19, 2026: 100 short prompts and 12 longer archive, C audit and Italian tasks.
API prompt lengths range from 16 to 23,454 tokens. Responses contain 6,922
output tokens in total. The prompts were fixed before examining local scores;
they do not reuse the earlier 100-prompt Flash set.

Model: `deepseek-flash`, identified as DeepSeek-V4.1-Flash by the
[official model table](https://api-docs.deepseek.com/quick_start/pricing/).
At collection time the old V4 Flash and Vision-Exp API names also redirected
to V4.1. Do not use these references to judge those older checkpoints.
All responses report fingerprint `aeb56401ca74e127821c4f9126dcb669`, also
present in the earlier V4.1 fixtures.

Thinking is disabled; no system message is sent. Temperature is 1, the output
limit is 64, and top-20 logprobs are retained. `top_p` uses the provider's
default of 1. These are sampled continuations, not greedy answers or complete
vocabulary logits. Per-case settings, lengths, groups and prompt hashes are
in `collection.json`; raw responses are retained.

## Scoring

On a dedicated Metal machine with at least 128 GiB RAM:

```sh
./gguf-tools/quality-testing/score_official MODEL.gguf \
  gguf-tools/quality-testing/deepseek-v4.1-flash-20260919-router/manifest.tsv \
  /tmp/router-scores.tsv 34816 \
  --ssd-streaming --ssd-streaming-cache-experts 48gb
```

`manifest-short.tsv` and `manifest-long.tsv` select the two groups. Compare
matched weights and settings, checking API/local token alignment before
interpreting probability metrics. Changing `--continued-prefill` or Metal
math mode is a separate numerical configuration, not an identical rerun.

Teacher-forced NLL measures the probability assigned to the reference tokens.
API top-token agreement compares each position after feeding the same prefix.
LCP is only an exact-prefix diagnostic here: the reference tokens were sampled.

`case_047` exposes a tokenizer discrepancy in the tested source: DwarfStar
encodes its continuation as 65 tokens, while both the API and DeepSeek's
published tokenizer give 64. The text contains curly quotation marks. Keep
the case as a reproducer, but exclude it from API probability comparisons;
the scorer does this automatically. Text NLL still includes it. The remaining
111 continuations align with the API, covering 6,858 positions. Do not treat
the discrepancy as a router effect or a damaged API response.

The router experiment also uses an isolated adaptation of Emilian Bold's
[overlap scorer](https://github.com/emilianbold/ds4/commit/f3e16a19d37f9c17d2f42869474742e9993c7098).
It sums `min(p_local, p_api)` over uniquely mapped API alternatives, without
renormalizing the truncated list. This is a lower bound on full distribution
overlap. The reported API probability mass bounds the missing contribution;
unmapped tokens are not treated as zero-probability API tokens. The comparison
used temporary shader overrides and a private scorer build, without changing
production source during the experiment.

## Router Comparison

Tested `aafc65b444f` against the two Metal shader changes from PR #1044,
using `DeepSeek-V4.1-Flash-IQ2_XXS-Q2_K-imatrix.gguf` on M5 Max. The full
112-prompt pair used a 48 GiB expert cache. Two additional pairs used
`manifest-sensitivity.tsv` (every fourth short prompt plus all long prompts,
37 cases), with a 64 GiB cache on the other M5 Max. Each pair kept its model,
host, context allocation and settings fixed.

| Setting | NLL, base / PR | Overlap lower bound, base / PR | API top-token agreement, base / PR | LCP, base / PR |
| --- | --- | --- | --- | --- |
| Default, 112 prompts | 0.396840 / 0.395353 | 0.907623 / 0.907639 | 90.347% / 90.245% | 5.563 / 6.071 |
| Continued prefill, last token separate | 0.301148 / 0.300111 | 0.924795 / 0.924768 | 91.876% / 91.697% | 9.568 / 9.811 |
| Safe Metal math | 0.306108 / 0.305066 | 0.924984 / 0.924797 | 91.697% / 91.921% | 9.270 / 8.270 |

The overall paired 95% prompt-bootstrap intervals include zero for NLL,
overlap and top-token agreement in all three settings. For the full pair,
the NLL difference is -0.001488, interval [-0.003328, +0.000199]; overlap
changes by +0.000015, interval [-0.000375, +0.000427]. This does not show a
clear regression, nor establish universal equivalence. No equivalence margin
was specified in advance.

LCP moves more sharply than the probability metrics. In safe math, just two
prompts account for its entire decline: case 096 changes from 0 to 13 and
case 102 from 64 to 14. That is not evidence of a one-token loss in general
answer quality.

Long-prompt overlap improves slightly in each setting; default long-prompt
NLL worsens by 0.000195 while the other two improve. These 12 prompts reuse
three task families, so their bootstrap intervals are only descriptive.
These results do not settle the earlier Vision-Exp checkpoint comparison.

Per-case score tables, probability bounds, paired intervals and source
provenance are in [results/comparison.json](results/comparison.json). The
full baseline was repeated at 48 and 64 GiB cache budgets with byte-identical
scores and per-position traces. All reported pairs completed; an interrupted
earlier candidate run was excluded. Trace arithmetic was independently
checked against the scorer totals.
