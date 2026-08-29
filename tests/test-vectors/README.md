# DeepSeek V4 Flash Test Vectors

These vectors were captured from the official DeepSeek V4 Flash API using
`deepseek-v4-flash`, greedy decoding, thinking disabled, and
`top_logprobs=20`. The hosted API does not expose full logits, so these files
store the best logprob slice the API provides.

Flash checkpoints are not interchangeable for this test. Each checkpoint has
its own directory:

- `flash-0731/`: current July 31 checkpoint and the default used by
  `ds4_test`.
- `flash-pre-0731/`: historical vectors and local golden from the checkpoint
  served before July 31.

A GGUF whose name contains `0731` must use `flash-0731`. The older undated
GGUF uses `flash-pre-0731`. A mismatch between checkpoint and fixture is an
invalid test, not a model-quality result.

Files:

- `flash-CHECKPOINT/prompts/*.txt`: exact user prompts.
- `flash-CHECKPOINT/official/*.official.json`: official API continuations and
  top-logprobs.
- `flash-CHECKPOINT/official.vec`: compact C-test fixture generated from the
  official JSON.
- `flash-CHECKPOINT/local-golden.vec`: local top-k/logit fixture captured from
  the matching GGUF. It catches substantial backend drift that can keep the
  same greedy token while damaging the logits distribution.

Regenerate official vectors:

```sh
DEEPSEEK_API_KEY=... ./tests/test-vectors/fetch_official_vectors.py \
  --checkpoint 0731
```

The checkpoint label is required. By default the fetcher creates
`tests/test-vectors/flash-CHECKPOINT`; a future API checkpoint must go into a
new directory instead of replacing an existing fixture. Confirm which
checkpoint the API is serving before assigning the label. Running the fetcher
without `--only` also regenerates `official.vec`.

The C runner defaults to the 0731 official and local-golden fixtures:

```sh
./ds4_test --logprob-vectors
./ds4_test --local-golden-vectors
```

The historical checkpoint remains available explicitly:

```sh
DS4_TEST_MODEL=/path/to/pre-0731.gguf \
DS4_TEST_VECTOR_FILE=tests/test-vectors/flash-pre-0731/official.vec \
  ./ds4_test --logprob-vectors

DS4_TEST_MODEL=/path/to/pre-0731.gguf \
DS4_TEST_LOCAL_GOLDEN_FILE=tests/test-vectors/flash-pre-0731/local-golden.vec \
  ./ds4_test --local-golden-vectors
```

GLM 5.2 OpenRouter vectors are kept in a separate directory:

```sh
OPENROUTER_API_KEY=... ./tests/test-vectors/fetch_openrouter_glm_vectors.py

DS4_TEST_MODEL=models/GLM-5.2-UD-Q4_K_XL.gguf \
DS4_TEST_VECTOR_FILE=tests/test-vectors/glm-openrouter/official.vec \
  ./ds4_test --logprob-vectors
```

The same fetcher also writes `tests/test-vectors/glm-openrouter/manifest.tsv`
for `gguf-tools/quality-testing/score_official`.  By default it routes to
OpenRouter `parasail/fp8` with strict parameter matching so top-logprob slices
are present in the fixture.

The Metal SSD-streaming cache-pressure repro for issue #384 is a focused
variant of the official-vector check. It forces a 16GiB routed-expert cache and
runs only the `short_code_completion` case that exposes wrong logits when
layer-batched decode reuses expert-cache buffers before the command buffer has
completed:

```sh
DS4_TEST_MODEL=gguf/DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  ./ds4_test --metal-ssd-streaming-cache-pressure
```

The runner opens the normal non-quality path with accelerator-specific fast
routes disabled and pins `DS4_METAL_PREFILL_CHUNK=2048` for this strict
official-vector check.

`official.vec` is intentionally trivial to parse from C: each case points to a
prompt file and each expected token is hex-encoded by bytes. The official JSON
files remain in the tree so the compact fixture can be audited against the API
response.

To inspect a local top-logprob dump manually:

```sh
./ds4 --metal --nothink -sys "" --temp 0 -n 4 --ctx 16384 \
  --prompt-file tests/test-vectors/flash-0731/prompts/long_code_audit.txt \
  --dump-logprobs /tmp/long_code_audit.ds4.json \
  --logprobs-top-k 20
```
