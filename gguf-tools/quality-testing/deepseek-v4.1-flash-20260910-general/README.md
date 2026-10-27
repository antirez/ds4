# DeepSeek V4.1 Flash general continuations

The 100 prompts in `../prompts.jsonl`, collected from the official DeepSeek API
on 2026-09-10 using `deepseek-flash`. The provider's model table identified this
as V4.1 Flash. All responses have fingerprint
`aeb56401ca74e127821c4f9126dcb669`, matching the
[sparse-boundary set](../deepseek-v4.1-flash-20260910/README.md).
Reference weights: `deepseek-ai/DeepSeek-V4.1-Flash`, revision
`df42c109f1defefcbfcedbe7d905718a12266e40`.

Thinking is disabled, temperature is 1, and each response has at most 32 tokens
with top-20 logprobs. The set contains 2994 API output tokens. These are sampled
continuations, not greedy continuations or full-vocabulary logits. Use
teacher-forced likelihood and API top-token agreement, not greedy prefix length
alone. `collection.json` records the request settings.

```sh
./gguf-tools/quality-testing/score_official MODEL.gguf \
  gguf-tools/quality-testing/deepseek-v4.1-flash-20260910-general/manifest.tsv \
  scores.tsv 4096 --ssd-streaming --ssd-streaming-cache-experts 64gb
```

Use this set alongside the sparse-boundary cases when comparing quantizations.
It broadens the short-prompt comparison; it does not replace long-context,
session-state or actual coding-task tests. None of these prompts occurs verbatim
in the 8192-token corpus used to calibrate the first V4.1 mixed-Q2 candidate.
