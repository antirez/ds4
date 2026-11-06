# DeepSeek V4.1 Flash continuations

Collected from the official DeepSeek API on 2026-09-10, using `deepseek-flash`.
The [provider's model table](https://api-docs.deepseek.com/quick_start/pricing)
identified this as DeepSeek-V4.1-Flash. These are not V4 Flash vectors.
Reference weights: `deepseek-ai/DeepSeek-V4.1-Flash`, revision
`df42c109f1defefcbfcedbe7d905718a12266e40`.

Thinking is disabled. Temperature is **1**, because this endpoint reports
post-temperature probabilities: at temperature zero the logprobs collapse to
`0` and `-9999`. Responses retain the returned top-20 logprobs and fingerprint.
These are sampled continuations, not greedy continuations or full logits.
Use teacher-forced NLL and API top-token agreement; a short greedy prefix alone
is not a quality failure for this set.

Cases 000-004 cover prose, C, arithmetic and Italian. Cases 005-016 retrieve a
warehouse number from an archive at these exact rendered prompt lengths:
127/128/129, 511/512/513, 1023/1024/1025 and 16383/16384/16385 tokens.
The expected retrieval answer is `074`. This crosses the sliding-window,
compressed-key selection and hierarchical candidate boundaries.

Run from the repository root on a supported Metal host:

```sh
./gguf-tools/quality-testing/score_official MODEL.gguf \
  gguf-tools/quality-testing/deepseek-v4.1-flash-20260910/manifest.tsv \
  scores.tsv 17408 --ssd-streaming --ssd-streaming-cache-experts 32gb
```

Add `--max-cases 5` for the short subset. The hosted implementation uses bounded
decoder replay, which the technical report describes as approximate; API scores
complement, rather than replace, inference-graph and session-state tests.
