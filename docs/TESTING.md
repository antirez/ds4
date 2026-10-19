# Testing and Development

[README](../README.md)

Read [CONTRIBUTING.md](../CONTRIBUTING.md) before proposing an inference change.
The [release QA guide](../QA_BEFORE_RELEASES.md) defines the hardware matrix,
checkpoint-matched quality tests, long-context agent tasks, and speed checks.
It also records which checks were not completed. A smoke test is not full QA.

## Focused checks

Model-free checks include:

```sh
make ds4_test ds4_agent_test test-session-state
./ds4_test --server
./ds4_agent_test
```

On Metal, small GPU tensor tests are available without loading a full GGUF:

```sh
make tests/test_session_state_gpu tests/test_glm53_kda tests/test_mxfp4_metal
./tests/test_session_state_gpu
./tests/test_glm53_kda
./tests/test_mxfp4_metal
```

`make test` also includes model-backed tests. Select the right GGUF and ensure
that it fits before running it; do not accidentally load a large model on a
single device during multi-GPU QA. ROCm has `make test-rocm`.

Official-vector tests must use continuations from the same checkpoint as the
GGUF. Flash 0731 and Vision Experimental are not interchangeable fixtures.
See [test vectors](../tests/test-vectors/README.md) and
[quality scoring](../gguf-tools/quality-testing/README.md).

## Investigating output

```sh
./ds4 --dump-tokens -p "..."
./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./ds4 --dump-logits /tmp/logits.json --nothink --prompt-file prompt.txt
./ds4-server --trace /tmp/ds4-trace.txt
```

Token dumps catch template differences without inference. Logits and
continuations help distinguish sampling changes from graph errors. Server
traces include cache and tool-parser decisions. Keep traces private when they
contain real conversations.

## Template parity without a model

`tests/template_parity_ab.py` renders synthetic Qwen scenarios through ds4's C
renderer (`./ds4_test --qwen-render EFFORT PRESERVE MESSAGES.json [TOOLS.json]`)
and through one or more Jinja templates given locally, then reports framing
differences. No model, no transcript, no corpus: scenarios live in the script
and reference templates are arguments.

```sh
python3 tests/template_parity_ab.py --template shipped=/tmp/shipped.jinja \
    --template fixed=/tmp/fixed.jinja --verbose
```

`--strict` exits non-zero when a scenario diverges from the shipped framing
differently than recorded in its `expect` field. The gating regression stays in
`./ds4_test --server`, whose Qwen parity blocks assert the shipped template's
defect patterns are absent; run them against a deliberately reverted build to
see them fail first.

Every scenario also reports a `\u` escape-sequence count for ds4 and for each
reference, and `--strict` fails when they disagree. The templates render tool
schemas through `tojson`, which decodes whatever spelling the client used and
writes non-ASCII raw, so these counts must match even in scenarios where framing
legitimately differs. The `tool-schema-escapes` scenario sends a schema whose
nested descriptions spell an em dash, a surrogate-pair emoji and a bell on the
wire, which is the case that used to survive verbatim into the prompt.

## Tools and source references

- [GGUF conversion and quantization](../gguf-tools/README.md)
- [Imatrix collection](../gguf-tools/imatrix/README.md) and
  [calibration corpus](../gguf-tools/imatrix/dataset/README.md)
- [Steering vectors](../dir-steering/README.md)
- [Benchmark scripts and charts](../speed-bench/README.md)
- [Evaluation data and licenses](../EVAL_DATA.md)
- [Session payload implementation](../ds4.c) and
  [cache header definitions](../ds4_kvstore.h)
- [Pipeline protocol](../ds4_distributed.c) and [TP protocol](../ds4_tp.c)

For changes to state handling, cover rewind/replay, save/load, images, and
multiple sessions as well as a fresh prompt. For distributed changes, exercise
both ranks and failures; a local command-parser test is not physical TP QA.
