# DeepSeek V4.1 Flash long-prefill continuations

Nine held-out prompts at approximately 8k, 16k and 32k tokens. Each length
ends with one of three tasks: C bounds checking, network-server design or
Italian prose. Use these alongside the shorter batch and sparse-boundary sets.

Collected on September 11 locally (September 10 at 23:11 UTC) from the
official DeepSeek API using `deepseek-flash`, thinking
disabled, temperature 1, 64 output tokens and top-20 logprobs. The fingerprint
is `aeb56401ca74e127821c4f9126dcb669`, matching the other V4.1 sets and weight
revision `df42c109f1defefcbfcedbe7d905718a12266e40`.

These are sampled continuations, not greedy answers or full-vocabulary API
logits. Compare teacher-forced likelihood and API probabilities. Also check
continued prefills, restored sessions and real coding tasks.

```sh
./gguf-tools/quality-testing/score_official MODEL.gguf \
  gguf-tools/quality-testing/deepseek-v4.1-flash-20260911-long/manifest.tsv \
  scores.tsv 34816 --ssd-streaming --ssd-streaming-cache-experts 64gb
```
