# Official-Continuation Quality Testing

This directory contains the prompts, tracked official fixtures, and scripts used
to compare local GGUF variants against hosted-model continuations.

The metric is target-token negative log likelihood: collect a deterministic
official continuation, then ask each local GGUF how much probability it assigns
to that exact continuation token by token.  This avoids judging quality from one
sampled answer.

## 1. Tracked Fixture Sets

Curated fixtures are kept in the repository so release QA can run without
calling hosted APIs:

- `data/glm52-openrouter-100`: 100 GLM 5.2 continuations collected through
  OpenRouter `z-ai/glm-5.2` with `top_logprobs=20`.
- `data/flash`: 100 DeepSeek V4 Flash continuations collected from the official
  DeepSeek API with `top_logprobs=20`.
- `data/pro`: 100 DeepSeek V4 PRO continuations collected from the official
  DeepSeek API with `top_logprobs=20`.

DeepSeek V4 Flash also has tracked official smoke vectors in
`tests/test-vectors/`.  Those vectors drive `./ds4_test --logprob-vectors` and
include short prompts plus long-prompt attention cases.

The hosted APIs expose output-token logprobs and top-logprob alternatives, not
full vocabulary logits.

## 2. Collect Official Continuations

```sh
export DEEPSEEK_API_KEY=...
python3 gguf-tools/quality-testing/collect_official.py \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/flash \
  --count 100 \
  --max-tokens 24
```

For GLM 5.2 through OpenRouter:

```sh
export OPENROUTER_API_KEY=...
python3 gguf-tools/quality-testing/collect_official.py \
  --model z-ai/glm-5.2 \
  --endpoint https://openrouter.ai/api/v1/chat/completions \
  --api-key-env OPENROUTER_API_KEY \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/glm52-openrouter \
  --count 100 \
  --max-tokens 24 \
  --top-logprobs 20 \
  --token-limit-field max_tokens \
  --provider-order parasail/fp8 \
  --require-parameters \
  --thinking omit \
  --reasoning-effort none
```

Use one output directory per official model.  The default model is Flash, so
`data/flash` is the recommended path for Flash continuations.  For PRO:

```sh
python3 gguf-tools/quality-testing/collect_official.py \
  --model deepseek-v4-pro \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/pro \
  --count 100 \
  --max-tokens 24 \
  --top-logprobs 20
```

The script writes:

- `data/<model>/prompts/case_*.txt`
- `data/<model>/continuations/case_*.txt`
- `data/<model>/responses/case_*.json`
- `data/<model>/manifest.tsv`

The prompt list is tracked in `prompts.jsonl`.  Curated fixture directories are
also tracked after review; ad-hoc API collection directories should stay
untracked until they are intentionally promoted into the release QA set.

## 3. Build The Local Scorer

```sh
make -C gguf-tools quality-score
```

The scorer links against the DS4 runtime and uses Metal by default.

## 4. Score GGUF Variants

```sh
gguf-tools/quality-testing/score_official \
  ../deepseek-v4-quants/gguf/OLD.gguf \
  gguf-tools/quality-testing/data/pro/manifest.tsv \
  /tmp/old.tsv \
  4096

gguf-tools/quality-testing/score_official \
  ../deepseek-v4-quants/gguf/NEW.gguf \
  gguf-tools/quality-testing/data/pro/manifest.tsv \
  /tmp/new.tsv \
  4096
```

Use `data/flash/manifest.tsv` for Flash GGUFs,
`data/glm52-openrouter-100/manifest.tsv` for GLM 5.2 GGUFs, and
`data/pro/manifest.tsv` for PRO GGUFs.  The scorer and comparator do not care
which model produced the manifest; the manifest path selects the continuation
set.

For a full-residency vs SSD-streaming comparison, score the same model twice and
add the streaming flags to one run:

```sh
gguf-tools/quality-testing/score_official \
  /path/to/model.gguf \
  gguf-tools/quality-testing/data/glm52-openrouter-100/manifest.tsv \
  /tmp/streaming.tsv \
  4096 \
  --ssd-streaming
```

## 5. Compare

```sh
python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```

Output fields:

- `avg_nll`: average negative log likelihood; lower is better.
- `delta_new_minus_old`: negative means the new GGUF fits the official
  continuation better.
- `case_wins_new_old_ties`: per-prompt NLL wins.
- `first_token_matches`: how often the local greedy first token matches the
  official first token.
- `avg_greedy_lcp`: average greedy longest common prefix against the official
  continuation.
- `api_target_mae`: when the manifest includes `response_file`, absolute
  local-vs-API logprob delta for aligned official output tokens.
- `api_top_coverage`: fraction of API top-logprob alternatives that map exactly
  to one local tokenizer token.
- `api_top1_rate`: how often the API top alternative equals the local greedy
  token.
- `api_topn_recall`: fraction of mapped API top-N alternatives found in the
  local top-N for the same position.
- `api_top_mae`: local-vs-API logprob MAE over mapped API top alternatives.
- `api_pair_rate`: pairwise ordering agreement among mapped API alternatives.
