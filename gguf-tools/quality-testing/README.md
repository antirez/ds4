# Official-Continuation Quality Testing

This directory contains the 100 prompts and scripts used to compare local GGUF
variants against official DeepSeek V4 Flash continuations.

The metric is target-token negative log likelihood: collect a deterministic
official continuation, then ask each local GGUF how much probability it assigns
to that exact continuation token by token.  This avoids judging quality from one
sampled answer.

## 1. Collect Official Continuations

```sh
export DEEPSEEK_API_KEY=...
python3 gguf-tools/quality-testing/collect_official.py \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data \
  --count 100 \
  --max-tokens 24
```

The script writes:

- `data/prompts/case_*.txt`
- `data/continuations/case_*.txt`
- `data/responses/case_*.json`
- `data/manifest.tsv`

The prompt list is tracked in `prompts.jsonl`; the official responses are not
tracked because they are derived from an external API.

## 2. Build The Local Scorer

```sh
make -C gguf-tools quality-score
```

The scorer links against the DS4 runtime and uses Metal by default.

## 3. Score GGUF Variants

```sh
gguf-tools/quality-testing/score_official \
  ../deepseek-v4-quants/gguf/OLD.gguf \
  gguf-tools/quality-testing/data/manifest.tsv \
  /tmp/old.tsv \
  4096

gguf-tools/quality-testing/score_official \
  ../deepseek-v4-quants/gguf/NEW.gguf \
  gguf-tools/quality-testing/data/manifest.tsv \
  /tmp/new.tsv \
  4096
```

## 4. Compare

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
