# Official Quality Fixtures

This directory contains curated hosted-model continuation fixtures that are
safe to commit and use in release QA.

- `glm52-openrouter-100`: 100 GLM 5.2 OpenRouter continuations with API
  top-logprob slices.
- `glm53-flash-openrouter-zai-fp8-100`: 100 deterministic GLM 5.3 Flash
  continuations from OpenRouter's pinned Z.AI FP8 endpoint. That endpoint did
  not return logprobs.
- `flash`: 100 DeepSeek V4 Flash 0731 continuations from the official DeepSeek
  API, with API top-logprob slices.
- `pro`: 100 DeepSeek V4 PRO preview continuations with API top-logprob slices.
- `pro-0813`: 100 DeepSeek V4 PRO 0813 continuations with API top-logprob
  slices.
- `qwen38-flash-alibaba-100`: 100 Qwen3.8 Flash continuations from Alibaba,
  pinned through OpenRouter, with top-five logprobs and no thinking.
- `qwen38-flash-alibaba-long`: 12 continuations from the same endpoint, with
  archive and code prompts from 2K to 24K tokens, crossing sparse attention.

Each fixture directory contains:

- `prompts/case_*.txt`: exact user prompts.
- `continuations/case_*.txt`: deterministic hosted-model continuations.
- `responses/case_*.json`: hosted responses, including logprob slices.
  Some curated sets omit unrelated response IDs and billing fields.
- `manifest.tsv`: paths consumed by `score_official`.

DeepSeek V4 Flash smoke vectors are also tracked in `tests/test-vectors/` and
are run by `./ds4_test --logprob-vectors`.

Qwen's `collection.json` records the explicit empty system message, provider,
template hashes and token-alignment checks. All prompt lengths match the
published tokenizer. Short cases 058 and 069 contain damaged emoji bytes in
the API logprob metadata: keep their text likelihood, but exclude their API
agreement. `manifest-aligned.tsv` selects the cases with intact token bytes.
Alibaba does not disclose this endpoint's precision, and hosted Flash is
based on Flash Next rather than guaranteed to use identical checkpoint weights.
