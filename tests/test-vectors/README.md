# DeepSeek V4 Flash Test Vectors

These vectors were captured from the official DeepSeek V4 Flash API using
`deepseek-v4-flash`, greedy decoding, thinking disabled, and
`top_logprobs=20`. The hosted API does not expose full logits, so these files
store the best logprob slice the API provides.

Files:

- `prompts/*.txt`: exact user prompts.
- `official/*.official.json`: official API continuations and top-logprobs.
- `official.vec`: compact C-test fixture generated from the official JSON.

Regenerate official vectors:

```sh
DEEPSEEK_API_KEY=... ./tests/test-vectors/fetch_official_vectors.py
```

Running the fetcher without `--only` also regenerates `official.vec`.

The C runner consumes `official.vec` directly:

```sh
./ds4_test --logprob-vectors
```

`official.vec` is intentionally trivial to parse from C: each case points to a
prompt file and each expected token is hex-encoded by bytes. The official JSON
files remain in the tree so the compact fixture can be audited against the raw
API response.

To inspect a local top-logprob dump manually:

```sh
./ds4 --metal --nothink -sys "" --temp 0 -n 4 --ctx 16384 \
  --prompt-file tests/test-vectors/prompts/long_code_audit.txt \
  --dump-logprobs /tmp/long_code_audit.ds4.json \
  --logprobs-top-k 20
```

## Local regression fixture (`local.vec`)

The cloud `official.vec` will drift across imatrix variants because the
hosted API output is fixed while local quantization changes. When you swap
the active model (e.g. a new imatrix build) and `official.vec` starts
failing in places where the inference *code* is fine, regenerate a
local-only fixture instead:

```sh
./tests/test-vectors/regen_local_vectors.py \
  -m /home/cghart/ds4/ds4flash.gguf \
  -o tests/test-vectors/local.vec
```

Then run the gate against the local fixture:

```sh
DS4_TEST_MODEL=/home/cghart/ds4/ds4flash.gguf \
DS4_TEST_VECTOR_FILE=tests/test-vectors/local.vec \
  ./ds4_test --logprob-vectors
```

`local.vec` is a strict argmax-only regression gate — it catches *changes
in inference behavior* between ds4 versions against the same model, but
it does not compare against the cloud API for quality. Keep
`official.vec` as the cloud-quality reference and use `local.vec` for
day-to-day code-regression CI when the active model is not the one
`official.vec` was calibrated against.
