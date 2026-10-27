# Qwen3.8 Flash Next

[Back to README](../README.md)

Qwen3.8-Flash-Next uses the `qwen4exp` GGUF architecture and a dedicated
Metal/CUDA graph with gated delta-net, gated GQA, block-sparse attention,
hyper-connections, n-gram embeddings, MoE, and MTP.

## Download and run

```sh
make
./download_model.sh qwen38-q2
./ds4 --ctx 8192 --prefill-chunk 1024
```

The download installs one GGUF containing the main model, MTP and original
BF16 n-grams, and updates `ds4flash.gguf`:

| Target | File size | Resident weights |
| --- | ---: | ---: |
| `qwen38-q2` | 137.10 GiB | 41.73 GiB |
| `qwen38-q4k` | 165.11 GiB | 69.74 GiB |

The 95.37 GiB n-gram table stays on disk. The runtime reads selected rows
directly from the GGUF, without mapping or preloading the table. Keep the
file on a fast local SSD. No sidecar or n-gram option is needed.

The Q2 model uses IQ2_XXS gate/up experts and Q2_K down projections padded
from 640 to 768 columns, calibrated with an imatrix. Q4 uses calibrated
Q4_K gate/up and MXFP4 down experts. Context and runtime buffers still need
RAM beyond the resident weights; start Q2 with the context and chunk above
on a 64 GB machine. Q4 needs a larger machine.

On a DGX Spark, build with `make cuda-spark` instead of `make`. Both Q2 and
Q4 fit resident on an otherwise idle 128 GB Spark; the n-grams stay on SSD.
The same commands support text, vision, MTP, steering and text-session
checkpoints in `ds4`, `ds4-agent` and `ds4-server`. No CUDA-specific model
conversion or feature flags are needed.
Leave `--prefill-chunk` unset on the Spark to use the faster 8192-token
default. The 1024-token example above saves memory on smaller Macs.

Use the same model options with `ds4-agent` or `ds4-server`.
Add `--mtp` for speculative decoding using the built-in MTP weights.
`ds4-bench` benchmarks ordinary decoding; it does not accept `--mtp`.
`--nothink` disables thinking. The server exposes
`qwen3.8-flash-next`, `qwen3.8-flash-next-chat`, and
`qwen3.8-flash-next-reasoner` aliases. Tool calls use the native
`<tool_call><function=...><parameter=...>` format.

On Metal, `ds4-server --batched-session N` decodes the slots together. Work that
only reads weights runs once for the whole batch, while the delta-net
recurrence, the attention caches and the n-gram convolution stay per session.
The slots also share one prefill workspace, so an extra slot costs its caches
rather than another few GiB of transients. As with the other natively batched
models, grouping changes the order of floating point reductions, so a batched
reply can differ from the same prompt decoded alone.
Add `--mtp` to use batched speculative decoding as well. The server accepts
matching greedy drafts by default, including at nonzero temperature. With
`--mtp-exact-sampling`, sampled requests use ordinary batched decoding;
temperature-zero requests can still speculate. Default greedy acceptance also
means a seed need not reproduce a reply under a different batching schedule.
Speculative batches above 16 sessions, images and steering use the ordered
fallback. CUDA currently
decodes sessions in order.

Disk KV checkpoints include recurrent state. Rewinding to an earlier position
replays the retained prefix on the next evaluation. The native context is
262144 tokens; `DS4_QWEN4_YARN_FACTOR=2` or `=4` enables static YaRN for
longer contexts, with a possible quality cost on shorter prompts.

## DGX Spark performance

Measured on a single Spark with resident weights and disk-only n-grams:

| Model | First 1024 tokens | Next 7168 tokens | Decode at 8192 tokens |
| --- | ---: | ---: | ---: |
| Q2 | 516 t/s | 745 t/s | 22.6 t/s |
| Q4 | 513 t/s | 755 t/s | 21.2 t/s |

These are averages of two runs using `speed-bench/promessi_sposi.txt`,
8192-token prefill chunks and 128 teacher-forced decode tokens, without MTP.
Loading the model is excluded. Q2 also reached about 771 t/s on a fresh
32K-token prefix with `--prefill-chunk 32768`. Small continuations have lower
throughput: adding 32 tokens after an 8K Q4 prefix took about 273 ms.
MTP speed depends on how often drafts are accepted; the table measures
ordinary decoding.

## Conversion

See [GGUF conversion](../gguf-tools/README.md) for conversion from the HF
checkpoint and calibrated low-bit expert quantization.
`gguf-tools/qwen4_pack_to_qwen4exp.py` migrates old DS4 fast packs to the
active `qwen4exp` schema. Finish older main-only conversions with
`gguf-tools/qwen4_native_ngrams.py`, using the original HF n-gram shards.
Expanding an old quantized table to BF16 does not restore its lost precision.

## Vision

Images go through the model's Qwen3-VL vision tower. Download the encoder
with `./download_model.sh qwen38-vision`, then use
`--vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf`. The CLI accepts
`/read image.png`; the server accepts `image_url` parts. Each image is resized to
multiples of 32 pixels within 64 to 1024 tokens (`DS4_QWEN4_IMAGE_MAX_TOKENS`
raises the cap), encoded on the GPU, and takes the model's 3D rope positions;
live KV reuse keys on the image fingerprints. `make test-qwen4-vision`
compares the tower with the Hugging Face implementation using the same GGUF
weights, dequantized to float32. It separately reports GGUF-versus-original
checkpoint quality and GPU-versus-original agreement. Both comparisons use
the unchanged minimum per-token cosine threshold of 0.99; implementation
parity gates the default exit status. Add `--require-quality` to also fail on
GGUF-versus-original quality loss. A passing implementation check alone does
not mean the quantized encoder matches the original checkpoint.

For a small suite including OCR, diagrams, a photograph, and resizing:

```sh
make tests/test_qwen4_vision
uv run --with numpy --with torch --with torchvision --with pillow \
  --with safetensors --with transformers --with gguf tests/qwen4_vision_ref.py \
  --snapshot /path/to/HF-checkpoint \
  --mmproj gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
  --image tests/vision-fixtures/qwen38/orbit.png \
  --image tests/vision-fixtures/qwen38/maple.png \
  --image tests/vision-fixtures/glm53/diagram.png \
  --image tests/vision-fixtures/glm53/text.png \
  --image tests/vision-fixtures/glm53/earth.jpg \
  --image tests/vision-fixtures/glm53/screenshot.png \
  --json-report /tmp/qwen38-vision-parity.json
```

Transformers must include `qwen4_exp`; the reference runs on CPU. Repeat
`--image` to add cases. The original single-image `--out` embedding dump is
still supported. Metric checks without model weights run with
`uv run --with numpy python -m unittest discover -s tests -p test_qwen4_vision_ref.py`.
JPEG decoder agreement with Pillow, including progressive scans, subsampling,
and image edges, is checked separately with
`uv run --with numpy --with pillow python -m unittest discover -s tests -p test_jpeg_decode.py`.

To check CLI `/read` image turns and text follow-ups with ordinary and MTP
decode, use two different PNG or JPEG images with the model-backed regression:

```sh
make ds4
uv run tests/test_qwen4_cli_vision.py --model /path/to/main-with-mtp.gguf \
  --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf \
  --image /path/to/first.png --image /path/to/second.png
```

The test checks that all four turns complete in each mode, including errors
that the interactive CLI can report without a nonzero process exit. It saves
the responses and diagnostics for inspection; it does not grade image content.

The Metal and CUDA graphs accept Q8_0, Q4_0, F16, BF16 and F32
dense weights, Q8_0/MXFP4/Q4_0/Q4_K/Q2_K/IQ2_XXS experts, F16/F32/Q8_0
hyper-connection mixers and the original BF16 n-gram table.
Tensor parallelism, pipeline execution and SSD expert streaming are not
implemented for this model yet.
ROCm is not supported. CPU code is a correctness reference, not a general
inference backend.


## Validation

```sh
make test-qwen4-kernels test-qwen4-q2 test-qwen4-prefill-reuse test-q8-prefill-variants
make test-frontends
make test-qwen4-ngrams
make tests/test_qwen4_ngram_state
make -B -C gguf-tools quants-shared
python3 -m unittest discover -s gguf-tools/tests -p test_qwen4_pack.py
python3 -m unittest discover -s gguf-tools/tests -p test_qwen4_native_ngrams.py
```

On CUDA, use `make test-qwen4-cuda` for the kernel tests. They compare the
active kernels with independent CPU references, without model weights.
Vision and end-to-end checks additionally require the checkpoints above.
Run `tests/test_qwen4_ngram_state MODEL.gguf` on either backend to check
failed disk reads during prefill, decode and MTP, then exact recovery.

Official Alibaba continuations are tracked for 100 short prompts and 12
archive/code prompts from 2K to 24K tokens. Build the quality scorer, then run
from the repository root:

```sh
gguf-tools/quality-testing/score_official MODEL.gguf \
  gguf-tools/quality-testing/data/qwen38-flash-alibaba-100/manifest.tsv /tmp/qwen-short.tsv 4096
gguf-tools/quality-testing/score_official MODEL.gguf \
  gguf-tools/quality-testing/data/qwen38-flash-alibaba-long/manifest.tsv /tmp/qwen-long.tsv 32768
```

Repeat with `--quality` and, for the long set, `--continued-prefill 1` and
`--continued-prefill 256`. These fixtures match the no-thinking template;
no rendered-prompt flag is needed. See [quality testing](../gguf-tools/quality-testing/README.md)
for collection settings, measurements and the hosted-checkpoint limitations.

[Checkpoint-fix benchmark charts and measurements](../speed-bench/qwen38-checkpoints/README.md)
compare prefill, ordinary decode, and MTP decode against the preceding PR head.

Qwen prefill checkpoints include matching logits at each completed chunk,
including cancellation frontiers. To test save/restore and continuation:

```sh
DS4_TEST_MODEL=/path/to/main-with-mtp.gguf \
  DS4_TEST_GLM_MTP=1 DS4_TEST_MTP_EXACT=1 \
  ./ds4_test --qwen4-prefill-checkpoints --qwen4-restore-reuse \
    --session-snapshot --session-rewind --session-rewind-resample
python3 tests/test_qwen4_checkpoint_replay.py \
  --model /path/to/main-with-mtp.gguf
python3 tests/test_qwen4_mtp_limits.py \
  --model /path/to/main-with-mtp.gguf
python3 tests/test_qwen4_logit_dump.py \
  --model /path/to/main-with-mtp.gguf
```

The server regression checks continued disk saves, cache-budget eviction,
and text-answer history with and without tool schemas. It covers clients
that omit reasoning and clients that echo `reasoning_content`, including
server restart after a continuation. For tools-enabled histories, the first
answer keeps the exact disk key; after a client continues without reasoning,
the next checkpoint uses the visible history key for restart reuse.

The MTP tests cover small prefill buffers, rewinds, reused sessions, and
truncated checkpoints. The logit-dump test requires NumPy and compares all-row
prefill with teacher-forced decode across a chunk boundary. For official
continuation scoring, use `--rendered-prompt`
when a fixture already contains the complete model chat template.

`tests/test_qwen4_prefill MODEL PROMPT 8192` checks mixed prefill sizes,
progress callbacks and exact replay through the sparse-attention boundary.
It also reports differences against a fresh prefill followed by individual
decodes. Those schedules can round differently and exchange nearly tied
experts; their full logits need not match. Use model-quality checks as well,
not only state replay:

```sh
python3 tests/test_server_story.py --url http://127.0.0.1:8000 \
  --model qwen3.8-flash-next --output /tmp/qwen-story
```

Start the server with at least `--ctx 49152`. This checks all sixteen facts
in a 31K-token story, then a correction turn that must reuse the prefix.

Qwen directional steering and activation capture support all 48 trunk layers;
see [directional steering](../dir-steering/README.md). Model-backed regressions:

```sh
python3 tests/test_qwen4_steering.py --model /path/to/main-with-mtp.gguf
./ds4_agent_test --full-context-save /path/to/main-with-mtp.gguf
make test-web-recovery
```

The agent can save an already-full text transcript without a KV payload.
Restart with a larger `--ctx`, then `/switch` to the saved session and
`/compact` to recover room. Sessions containing images still use the existing
save restriction. Automatic compaction reserves summary space and continues
interrupted generation through the existing agent compaction path.
