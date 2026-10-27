# DeepSeek V4.1 Flash extended-context continuations

Six held-out prompts at approximately 64k and 96k tokens, covering C bounds
checking, network-server design and Italian prose. Use alongside the short,
sparse-boundary and 8k/16k/32k sets.

Collected September 11 from the official DeepSeek API using `deepseek-flash`,
thinking disabled, temperature 1, 64 output tokens and top-20 logprobs.
Fingerprint `aeb56401ca74e127821c4f9126dcb669` matches the earlier V4.1 sets
and weight revision `df42c109f1defefcbfcedbe7d905718a12266e40`.

These are sampled continuations, not greedy answers or full-vocabulary API
logits. Compare teacher-forced likelihood and API probabilities. Also test
continued prefills, restored sessions and real coding tasks separately.

```sh
./gguf-tools/quality-testing/score_official MODEL.gguf \
  gguf-tools/quality-testing/deepseek-v4.1-flash-20260911-extended/manifest.tsv \
  scores.tsv 102400 --ssd-streaming --ssd-streaming-cache-experts 64gb
```
