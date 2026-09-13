# DeepSeek V4.1 Flash batched-prefill continuations

Twelve held-out prompts with approximately 384, 768, 1536 and 4096 tokens,
covering C bounds checking, network-server design and Italian prose. The
archive background varies in length; each size is tested with all three tasks.
These prompts exercise matrix-prefill and sparse attention, unlike the much
shorter prompts in the general set.

Collected from the official DeepSeek API on 2026-09-10 using `deepseek-flash`,
thinking disabled, temperature 1, up to 64 output tokens and top-20 logprobs.
All responses have fingerprint `aeb56401ca74e127821c4f9126dcb669`, matching the
other V4.1 sets. The reference weight revision is
`df42c109f1defefcbfcedbe7d905718a12266e40`.

These are sampled continuations, not greedy answers or full-vocabulary API
logits. Compare teacher-forced likelihood and API token probabilities. Retain
the separate sparse-boundary, snapshot, cancellation and coding-session tests.

```sh
./gguf-tools/quality-testing/score_official MODEL.gguf \
  gguf-tools/quality-testing/deepseek-v4.1-flash-20260910-batch/manifest.tsv \
  scores.tsv 6144 --ssd-streaming --ssd-streaming-cache-experts 64gb
```
