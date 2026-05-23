# DeepSeek V4 Flash Test Vectors

The compact fixture consumed by `ds4_test` is generated from the local default
CyberNeurova abliterated GGUF using greedy decoding, thinking disabled, and
`top_logprobs=20`. It is a local regression fixture for the model currently
linked by `ds4flash.gguf`.

The raw `official/*.official.json` captures from the hosted DeepSeek V4 Flash
API are still kept for auditing and comparison, but they are not the default
C test fixture.

Files:

- `prompts/*.txt`: exact user prompts.
- `official/*.official.json`: official API continuations and top-logprobs.
- `official.vec`: compact C-test fixture generated from the local GGUF.

Regenerate the official API captures:

```sh
DEEPSEEK_API_KEY=... ./tests/test-vectors/fetch_official_vectors.py
```

The fetcher preserves the hosted API captures. Regenerate `official.vec` from a
local model dump when the default GGUF changes.

The C runner consumes `official.vec` directly:

```sh
./ds4_test --logprob-vectors
```

The runner opens the normal non-quality path with accelerator-specific fast
routes disabled and pins `DS4_METAL_PREFILL_CHUNK=2048` for this strict
official-vector check.

`official.vec` is intentionally trivial to parse from C: each case points to a
prompt file and each expected token is hex-encoded by bytes.

To inspect a local top-logprob dump manually:

```sh
./ds4 --metal --nothink -sys "" --temp 0 -n 4 --ctx 16384 \
  --prompt-file tests/test-vectors/prompts/long_code_audit.txt \
  --dump-logprobs /tmp/long_code_audit.ds4.json \
  --logprobs-top-k 20
```
