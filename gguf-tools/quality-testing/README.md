# Official-Continuation Quality Testing

This directory contains the prompts, tracked official fixtures, and scripts used
to compare local GGUF variants against hosted-model continuations.

The main metric is target-token negative log likelihood: collect an
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
- `data/qwen38-flash-alibaba-100`: 100 non-thinking Qwen3.8 Flash
  continuations from Alibaba through OpenRouter, with top-five logprobs.
- `data/qwen38-flash-alibaba-long`: 12 archive/code continuations from the
  same endpoint, with prompts from 2K to 24K tokens.
- [deepseek-v4.1-flash-20260919-router](deepseek-v4.1-flash-20260919-router/README.md):
  112 fresh official V4.1 Flash continuations, including 12 longer prompts up
  to 23K tokens. Temperature-1 samples with top-20 logprobs, intended for
  paired numerical comparisons. These are not V4 Flash or Vision-Exp vectors.

DeepSeek V4 Flash also has tracked official smoke vectors in
`tests/test-vectors/`.  Those vectors drive `./ds4_test --logprob-vectors` and
include short prompts plus long-prompt attention cases.

The hosted APIs expose output-token logprobs and top-logprob alternatives, not
full vocabulary logits.

The scorer verifies each API token's bytes against the local token boundaries,
not just the number of tokens. If they differ, it still scores the continuation
text but skips API logprob comparisons for that case. Alternatives containing
Unicode replacement characters are excluded because the original token bytes
may have been lost by the provider.

## 2. Collect Official Continuations

The Flash 0731 fixture is historical. The official API now redirects the old
Flash names to V4.1; do not overwrite `data/flash` with new calls. Collect
current V4.1 references in a separate directory:

```sh
export DEEPSEEK_API_KEY=...
python3 gguf-tools/quality-testing/collect_official.py \
  --model deepseek-flash \
  --endpoint https://api.deepseek.com/chat/completions \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out /tmp/deepseek-v4.1-flash-new \
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

Use `--system TEXT` when the reference needs an explicit system message.
`--system ''` sends an empty message, which is different from omitting it:
some providers insert a default system prompt when none is supplied. The
collector records this setting and rejects a changed system message on resume.
For a nonempty system message, render the matching local chat prompt and use
`--rendered-prompt`; the scorer's ordinary prompt mode adds no system message.

For Qwen, use the following settings in a new output directory:

```sh
python3 gguf-tools/quality-testing/collect_official.py \
  --out /tmp/qwen-official \
  --model qwen/qwen3.8-flash \
  --endpoint https://openrouter.ai/api/v1/chat/completions \
  --provider-order alibaba --require-parameters \
  --require-response-provider Alibaba --require-response-model qwen/qwen3.8-flash \
  --system '' --thinking omit --reasoning-effort none \
  --count 100 --max-tokens 64 --top-logprobs 5
```

This reads `OPENROUTER_API_KEY` from the environment and disables provider
fallbacks. Omitting `--system ''` caused Alibaba to add 13 prompt tokens in
the September 2026 checks. The empty message matches Qwen's published
no-thinking template and the scorer's ordinary prompt mode.

Alibaba does not disclose this endpoint's precision. The
[official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) describes
hosted Flash as based on Flash Next, not as a byte-identical checkpoint.
Agreement with this service is an external quality check, not proof of exact
native-weight inference.

## 3. Build The Local Scorer

```sh
make -C gguf-tools quality-score
```

The scorer links against the DS4 runtime, using Metal on macOS and CUDA on
Linux. On a DGX Spark, pass `CUDA_ARCH=sm_121` to the build command.

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

Add `--continued-prefill N` to process each prompt in two calls, leaving its
last `N` tokens for the second call. This checks continued-prefill quality
against the same official answers. Every prompt must contain more than `N`
tokens; otherwise the scorer fails instead of silently skipping the test.

Add `--session-batch N` (2 to 8) to score an official continuation alongside
unrelated, independently advancing sessions. The scorer rotates their row
order on every step. Compare against a run without this option using the same
manifest, model and context; budget memory for all `N` sessions.

Check session isolation separately on a dedicated Metal host:

```sh
DS4_TEST_MODEL=MODEL.gguf DS4_TEST_BATCH_ISOLATION=1 \
  DS4_TEST_SESSION_COUNT=4 DS4_TEST_DECODE_STEPS=32 \
  MTL_DEBUG_LAYER=1 ./tests/test_metal_session_batch
```

This requires exact logits when companion prompts and row order change, then
checks a continued prefill and resumed serial decoding. It also checks that
invalid batches leave the target unchanged. It does not replace serial/batch
quality comparisons: different arithmetic can round differently without
sessions contaminating one another. Use `DS4_TEST_PROMPT_FILE` and
`DS4_TEST_CONTEXT_SIZE` to repeat at longer prefixes. For physical TP, the test
also accepts `DS4_TEST_TP_RDMA_DEVICE` and `DS4_TEST_TP_GID_INDEX` alongside
its coordinator/worker settings. Throughput measurements must run separately
without Metal API validation.

For Metal V4.1 prefill scheduling changes, also run:

```sh
make tests/test_deepseek41_prefill
./tests/test_deepseek41_prefill --dispatch
MTL_DEBUG_LAYER=1 ./tests/test_deepseek41_prefill MODEL.gguf speed-bench/promessi_sposi.txt
```

The model test uses SSD streaming and two 128K sessions. Run it alone on a
dedicated host with at least 128 GiB RAM. It alternates small and large appends
through 113K context, checking dispatch, progress, unchanged-prefix reuse,
saved state and subsequent decoding against a control without decoder deferral.
Keep the official continuation checks too: this scheduling test does not judge
quality across different floating-point operation orders. Measure speed without
Metal API validation, separately for initial and continued prefills.

Run the same mixed-prefix test over physical RDMA on two dedicated Metal hosts.
Start the worker with the same model and a 131072-token context, then the test
on the coordinator (replace the host and RDMA device names):

```sh
# Worker
MTL_DEBUG_LAYER=1 ./ds4 -m MODEL.gguf --ctx 131072 \
  --tensor-parallel --role worker --coordinator COORDINATOR 19455 \
  --transport rdma --rdma-device WORKER_DEVICE --rdma-gid-index 1

# Coordinator
MTL_DEBUG_LAYER=1 ./tests/test_deepseek41_prefill --tensor-parallel \
  MODEL.gguf speed-bench/promessi_sposi.txt COORDINATOR 19455 COORDINATOR_DEVICE 1
```

This compares equal TP prefill partitions and queued versus synchronous decode,
including full logits and cache state. TP snapshots rebuild both ranks from
tokens, so restoration is compared with an equivalent fresh replay. Also run
the official scorer in TP mode for initial and continued prompts; scheduling
parity alone does not establish model quality.

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
  continuation. This measures exact agreement, not general answer quality;
  one early mismatch discards all later agreement. Do not use it alone as a
  quality gate, especially for sampled references. Compare paired NLL and
  API probability agreement under matching execution settings as well.
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

## 7. Qwen3.8 Spark Baseline

Measured on 2026-09-15, using one DGX Spark per quant. The short set has 100
cases and 5,696 target tokens; the long set has 12 cases and 766 target tokens,
with prefixes from 1,971 to 23,985 tokens. These are ordinary decoding and
prefill checks, without MTP or vision.

| Suite | GGUF | API top-token agreement | NLL, default | NLL, `--quality` |
| --- | --- | ---: | ---: | ---: |
| Short | Q2 | 88.90% | 0.352980 | 0.352915 |
| Short | Q4 | 92.08% | 0.290297 | 0.290538 |
| Long | Q2 | 93.99% | 0.154199 | 0.154106 |
| Long | Q4 | 97.26% | 0.124611 | 0.125546 |

Agreement is measured after feeding the same official continuation prefix,
not as a percentage of correct answers. The two short cases with damaged
emoji metadata contribute to NLL but not API agreement, which covers 5,568
short-set tokens. All 766 long-set tokens align. Every local prompt and target
token ID was also checked against Qwen's published tokenizer.

The default fast path shows no material loss relative to `--quality` in these
sets. The paired case-bootstrap intervals include zero for both quants and
both suites. Q4 is closer to the hosted reference than Q2 on average.
These measurements do not prove identical native weights or correctness at
every context length.

Leaving the last 1 or 256 prompt tokens for a separate continued prefill
changed aggregate NLL by less than 0.0008 for either quant. Independent CLI
answers at about 2K and 24K context also retrieved the correct archive owner
and revision, and gave the correct clamp outputs and boundary tests, for
both Q2 and Q4.

Raw per-case scores, weight/source hashes and comparison details are retained
in [short results](data/qwen38-flash-alibaba-100/cuda-spark-20260915/results.json)
and [long results](data/qwen38-flash-alibaba-long/cuda-spark-20260915/results.json).
