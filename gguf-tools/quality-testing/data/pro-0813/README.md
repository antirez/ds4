# DeepSeek V4 Pro 0813 Fixture

This fixture contains 100 official DeepSeek API continuations for the August
2026 DeepSeek V4 Pro checkpoint. It was collected between
2026-08-12T23:06:53Z and 2026-08-12T23:09:33Z while the official API pricing
page identified the served model as `DeepSeek-V4-Pro-0813`.

The API request used the public model alias `deepseek-v4-pro`. Every response
reports this system fingerprint:

```text
fp_v4pro_20260812_prod0820_fp8_kvcache_20260402
```

The older preview fixture in `../pro` reports
`fp_9954b31ca7_prod0820_fp8_kvcache_20260402`. Keep the two fixtures separate
and score each GGUF against the fixture for its own checkpoint.

Collection parameters:

- 100 prompts from `gguf-tools/quality-testing/prompts.jsonl`;
- at most 24 output tokens per prompt;
- top-20 API log probabilities for every returned token;
- thinking disabled and reasoning effort omitted.

The exact command was:

```sh
DEEPSEEK_API_KEY=... \
python3 gguf-tools/quality-testing/collect_official.py \
  --model deepseek-v4-pro \
  --endpoint https://api.deepseek.com/chat/completions \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/pro-0813 \
  --count 100 \
  --max-tokens 24 \
  --top-logprobs 20 \
  --thinking disabled \
  --reasoning-effort omit
```

Ninety-one responses reached 24 tokens. Nine ended normally before the token
limit. The complete fixture contains 2,275 output tokens.
