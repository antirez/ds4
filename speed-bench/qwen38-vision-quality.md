# Qwen3.8 IQ2 and Q4_K vision smoke test

Both recipes passed all six existing API vision fixtures on the M3 Ultra
512 GiB machine. Tested revision: `3bf8552`, September 7, 2026 (Europe/Rome).

| Case | IQ2_XXS imatrix | Q4_K imatrix |
| --- | --- | --- |
| Diagram labels and order | Pass | Pass |
| Exact OCR | Pass | Pass |
| Spatial relationships | Pass | Pass |
| Software screenshot | Pass | Pass |
| Earth photograph | Pass | Pass |
| Rejection of unrelated train-ticket premise | Pass | Pass |

The IQ2 recipe is the 50.34 GB mixed IQ2_XXS gate/up, MXFP4 down build from
the [text quality comparison](qwen38-iq2-quality.md). The Q4_K recipe is its
combined MTP template. Both use the same external Q4_1 PLE and downloaded
`mmproj-Qwen3.8-Flash-Next-Q8_0.gguf` vision encoder (616,703,104 bytes).

Servers ran serially on Metal, with 8,192 context positions, a 1,024-token
prefill chunk, a maximum of 1,024 image tokens, and MTP disabled. The existing
`tests/run_glm53_vision_quality.py` runner sent local OpenAI-compatible API
requests with temperature 0, `reasoning_effort=none`, and a 320-token output
limit. No retries or fixture changes were made. Both runner exits were zero;
both temporary servers were stopped after testing.

`make -j8 ds4-server` succeeded. To reproduce, launch one model at a time:

```sh
./ds4-server -m /path/to/model.gguf --ple /path/to/PLE-Q4_1.gguf \
  --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
  --metal --ctx 8192 --prefill-chunk 1024 --host 127.0.0.1 --port 18938
```

In another terminal:

```sh
DS4_VISION_ENDPOINT=http://127.0.0.1:18938/v1/chat/completions \
DS4_VISION_MODEL=qwen3.8-flash-next DS4_VISION_SUITE=qwen38 \
DS4_VISION_REASONING_EFFORT=none \
python3 tests/run_glm53_vision_quality.py
```

This is a small functional smoke suite using required/forbidden answer
substrings. It confirms basic image processing and response quality, but does
not establish general vision accuracy, IQ2/Q4_K equivalence, or numerical
agreement of the encoder with the original HF checkpoint. The separate
`make test-qwen4-vision` reference comparison was not run.

[Full answers and provenance](qwen38-vision-quality-results.json) include the
exact server commands, encoder hash, fixture hashes and per-case results.
Raw server logs remain locally under `OUT/qwen38-vision-quality/`.
