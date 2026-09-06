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
- `data/glm53-flash-openrouter-zai-fp8-100`: 100 GLM 5.3 Flash continuations
  from OpenRouter's pinned Z.AI FP8 endpoint. The endpoint does not expose
  logprobs, so these are deterministic continuation fixtures.
- `data/glm53-flash-openrouter-zai-fp8-long`: eight code-context continuations
  from the same FP8 endpoint, with 3K-7K prompts that exercise sparse attention.
  Use the rendered prompts below to preserve the API's reasoning prefix.
- `data/flash`: 100 DeepSeek V4 Flash 0731 continuations collected from the
  official DeepSeek API with `top_logprobs=20`.
- `data/pro`: 100 DeepSeek V4 PRO preview continuations collected from the
  official DeepSeek API with `top_logprobs=20`.
- `data/pro-0813`: 100 DeepSeek V4 PRO 0813 continuations collected from the
  official DeepSeek API with `top_logprobs=20`.

DeepSeek V4 Flash also has tracked official smoke vectors in
`tests/test-vectors/`.  Those vectors drive `./ds4_test --logprob-vectors` and
include short prompts plus long-prompt attention cases.

The hosted APIs expose output-token logprobs and top-logprob alternatives, not
full vocabulary logits.

## 2. Collect Official Continuations

For the tracked DeepSeek V4 Flash 0731 fixture:

```sh
export DEEPSEEK_API_KEY=...
python3 gguf-tools/quality-testing/collect_official.py \
  --model deepseek-v4-flash \
  --endpoint https://api.deepseek.com/chat/completions \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/flash \
  --count 100 \
  --max-tokens 24 \
  --top-logprobs 20 \
  --thinking disabled \
  --reasoning-effort omit
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

For GLM 5.3 Flash through the pinned Z.AI FP8 endpoint:

```sh
export OPENROUTER_API_KEY=...
python3 gguf-tools/quality-testing/collect_official.py \
  --model z-ai/glm-5.3-flash \
  --endpoint https://openrouter.ai/api/v1/chat/completions \
  --api-key-env OPENROUTER_API_KEY \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-100 \
  --count 100 \
  --max-tokens 128 \
  --top-logprobs 0 \
  --token-limit-field max_tokens \
  --provider-order z-ai/fp8 \
  --thinking omit \
  --reasoning-effort low
```

Use one output directory per checkpoint. For PRO 0813 through the official
DeepSeek API:

```sh
python3 gguf-tools/quality-testing/collect_official.py \
  --model deepseek-v4-pro \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/pro-0813 \
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

Build the optional llama.cpp control scorer with:

```sh
make -C gguf-tools quality-llama-score
```

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

Use `data/flash/manifest.tsv` for Flash GGUFs and
`data/glm52-openrouter-100/manifest.tsv` for GLM 5.2 GGUFs. Use
`data/glm53-flash-openrouter-zai-fp8-100/manifest.tsv` for GLM 5.3 Flash. Use
`data/pro/manifest.tsv` for the PRO preview checkpoint and
`data/pro-0813/manifest.tsv` for PRO 0813. The scorer and comparator do not
care which model produced the manifest; the manifest path selects the
continuation set.

Add `--quality` to disable DS4's speed-oriented numerical shortcuts. For an
independent llama.cpp comparison of a DeepSeek V4 GGUF, use the same manifest
and the token-identical DS4 prompt renderer:

```sh
gguf-tools/quality-testing/score_llama \
  /path/to/model.gguf \
  gguf-tools/quality-testing/data/flash/manifest.tsv \
  /tmp/llama.tsv \
  4096 \
  deepseek-ds4
```

GLM's hosted endpoint requires reasoning. A no-thinking local prompt does not
match a reply generated after reasoning, even at temperature zero. For the long
GLM fixtures, render the official `Reasoning Effort: Low` template and include
the returned reasoning before scoring the answer:

```sh
python3 gguf-tools/quality-testing/render_glm_references.py \
  gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-long
gguf-tools/quality-testing/score_official /path/to/GLM-5.3-Flash-Q2.gguf \
  gguf-tools/quality-testing/data/glm53-flash-openrouter-zai-fp8-long/manifest-rendered.tsv \
  /tmp/glm-long.tsv 8192 --rendered-prompt
```

`--rendered-prompt` reads the prompt file with its existing chat markers; it
does not add another chat template. The renderer can also be applied to the
older 100-case GLM 5.3 Flash fixture. Keep rendered and legacy no-thinking
scores separate: their prefixes differ. Z.AI supplies no token logprobs, so
judge these runs by continuation NLL and greedy agreement, not logprob parity.

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

## 6. Validate regression artifacts

Two standard-library Python tools check saved artifacts without running inference. For matched `score_official` runs with API logprob coverage, require the complete intended manifest and finite, consistent counts and scores:

```sh
python3 gguf-tools/quality-testing/validate_scores.py \
  --manifest gguf-tools/quality-testing/data/flash/manifest.tsv \
  --baseline /tmp/old.tsv --candidate /tmp/new.tsv --strict-identical > /tmp/scores.json
```

Exit 0 with `--strict-identical` requires every per-case value to match; omit it to report valid differences without deciding their acceptability. Unlike `compare_scores.py`, validation rejects missing/duplicate cases, malformed fields and changed coverage denominators. This stricter mode requires the current four-column manifest and scorer TSV format with nonzero API coverage; use the existing comparator for continuation-only references without API logprobs.

Compare complete vocabulary dumps from `ds4-bench --dump-frontier-logits-dir`:

```sh
python3 gguf-tools/quality-testing/compare_frontier_logits.py \
  /tmp/old-logits /tmp/new-logits --frontiers 2048 4096 --ctx 8192 \
  --model /models/model.gguf --backend rocm --quality false --quant-bits 2 --vocab 129280 \
  --output /tmp/frontiers.json
```

Supply the exact expected metadata: `ctx` is allocated capacity; each frontier is total context, and prefill is its increase from the previous frontier (initially zero). Each directory must contain exactly the requested dumps. Exit 0 requires complete finite float32 bit identity; exit 1 reports drift, invalid artifacts or an operational error. JSON distinguishes `identical`, `drift` and `invalid`, with full-vector hashes, differing counts and maximum absolute differences. The output path must be new. The parser follows the writer's nine-significant-digit float format and preserves signed zero.

Matching argmax is insufficient: a measured 8K–64K comparison retained all four argmax IDs while full vectors differed, with maximum absolute difference 7.45228. This is numerical drift, not by itself proof of worse generation quality. Record prompt/weight hashes, source/build identity, settings and successful run completion separately; these tools cannot recover that provenance from score tables or dumps, and do not replace model-quality checks.

Run the host-only positive and negative controls:

```sh
python3 -m unittest discover -s gguf-tools/quality-testing/tests -v
```
