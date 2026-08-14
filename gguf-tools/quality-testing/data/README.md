# Official Quality Fixtures

This directory contains curated hosted-model continuation fixtures that are
safe to commit and use in release QA.

- `glm52-openrouter-100`: 100 GLM 5.2 OpenRouter continuations with API
  top-logprob slices.
- `laguna-openrouter-100`: 100 Laguna S 2.1 OpenRouter continuations. Poolside's
  endpoint does not expose output-token logprobs, so these support
  teacher-forced continuation scoring only.
- `flash`: 100 DeepSeek V4 Flash 0731 official continuations with API top-logprob
  slices.
- `pro`: 100 DeepSeek V4 PRO official continuations with API top-logprob
  slices.

Each fixture directory contains:

- `prompts/case_*.txt`: exact user prompts.
- `continuations/case_*.txt`: deterministic hosted-model continuations.
- `responses/case_*.json`: raw hosted responses, including logprob slices when
  the endpoint provides them.
- `manifest.tsv`: paths consumed by `score_official`; the optional raw-response
  path is present only for fixtures with API logprobs.

DeepSeek V4 Flash smoke vectors are also tracked in `tests/test-vectors/` and
are run by `./ds4_test --logprob-vectors`.
