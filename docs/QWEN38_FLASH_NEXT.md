# Qwen3.8 Flash Next

[Back to README](../README.md)

Qwen3.8-Flash-Next uses the `qwen4exp` GGUF architecture and a dedicated
Metal graph with gated delta-net, gated GQA, block-sparse attention,
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

On Metal, SSD streaming keeps a bounded cache of routed experts instead of
requiring all of them in RAM:

```sh
./ds4 --ssd-streaming --ctx 4096 --prefill-chunk 128
```

The Q2 and Q4 packs support streaming during prefill, decode and MTP. Dense
weights, shared experts, context and runtime buffers still need RAM; the
n-gram table continues to use its existing disk reads. Missing routed experts
are read into owned buffers. If a prefill batch needs more experts than the
cache can hold, or an MTP layer uses a different expert size, it uses explicit
temporary staging while retaining the normal kernel arithmetic. On devices
with GPU buffer addresses, this staging contains only the distinct selected
experts. Selecting every expert, or running without buffer-address support,
retains the full-layer fallback.

During single-token decode, cache misses overlap with gate/up computation
for cached experts and the shared expert. Once the reads finish, the missing
gate/up slots are computed, followed by one down pass and the original ordered
reduction. Private address snapshots keep the running GPU work independent of
cache updates. Fully cached selections use one gate/up pass. Batched prefill
and temporary staging retain their existing schedule; an MTP layer whose
expert size differs from the main model is staged again on each invocation,
without replacing the main model's cache entries.

A local M1 Max 32 GiB check compared this compact split schedule with
`85d37ae`, using the Q2 pack, automatic expert cache, `--ctx 4096`,
`--prefill-chunk 128`, `-n 100`, `--temp 0 --nothink`, and the prompt
`narrami la storia di roma`. Three alternating pairs produced:

| Decode | Runs (t/s) | Median (t/s) |
| --- | --- | --- |
| Before | 8.40, 8.25, 7.26 | 8.25 |
| Compact split | 8.44, 8.51, 8.14 | 8.44 |

All generated output was identical. The observed median gain was 2.3%; the
variation between runs does not establish a 10% improvement. Prefill medians
were 9.98 and 10.05 t/s; its execution path is unchanged. These are local
warm-file measurements, not a guarantee for other Macs or cold SSD reads.

Leave the expert-cache budget automatic initially. A plain count passed to
`--ssd-streaming-cache-experts` requests dynamic cache slots; an `NGB` budget
also includes the reserved layer-staging space. Startup accounts for the
context, prefill chunk, static weights, vision weights when present, and
staging before choosing the cache size. Reduce the context or prefill chunk
if those fixed requirements leave insufficient memory.

Qwen fills its expert cache on demand, so `--ssd-streaming-cold` does not
change its preload behavior. Popularity preloading and the GLM-specific
`--ssd-streaming-full-layers` option are not supported for Qwen.

Use the same model options with `ds4-agent`, `ds4-server`, or
`ds4-bench`. Add `--mtp` for speculative decoding using the built-in MTP
weights. `--nothink` disables thinking. The server exposes
`qwen3.8-flash-next`, `qwen3.8-flash-next-chat`, and
`qwen3.8-flash-next-reasoner` aliases. Tool calls use the native
`<tool_call><function=...><parameter=...>` format.

Disk KV checkpoints include recurrent state. Rewinding to an earlier position
replays the retained prefix on the next evaluation. The native context is
262144 tokens; `DS4_QWEN4_YARN_FACTOR=2` or `=4` enables static YaRN for
longer contexts, with a possible quality cost on shorter prompts.

## Conversion

See [GGUF conversion](../gguf-tools/README.md) for conversion from the HF
checkpoint and calibrated low-bit expert quantization.
`gguf-tools/qwen4_pack_to_qwen4exp.py` migrates old DS4 fast packs to the
active `qwen4exp` schema. Finish older main-only conversions with
`gguf-tools/qwen4_native_ngrams.py`, using the original HF n-gram shards.
Expanding an old quantized table to BF16 does not restore its lost precision.

## Vision

Images go through the model's Qwen3-VL vision tower. Download llama.cpp's
mmproj file with `./download_model.sh qwen38-vision` (it comes from
`ggml-org/Qwen3.8-Flash-Next-GGUF`; alternatively run
`convert_hf_to_gguf.py --mmproj --outtype f16` on the checkpoint) with
`--vision` and send `image_url` parts as usual. Each image is resized to
multiples of 32 pixels within 64 to 1024 tokens (`DS4_QWEN4_IMAGE_MAX_TOKENS`
raises the cap), encoded on the GPU, and takes the model's 3D rope positions;
live KV reuse keys on the image fingerprints. `make test-qwen4-vision`
compares the tower with the Hugging Face implementation using the same GGUF
weights, dequantized to float32. It separately reports GGUF-versus-original
checkpoint quality and Metal-versus-original agreement. Both comparisons use
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

Metal only for now. The Metal graph accepts Q8_0, Q4_0, F16, BF16 and F32
dense weights, Q8_0/MXFP4/Q4_0/Q4_K/Q2_K/IQ2_XXS experts, F16/F32/Q8_0
hyper-connection mixers and the original BF16 n-gram table.
Tensor parallelism and pipeline execution are not implemented for this model
yet. CPU code is a correctness reference, not a general inference backend.


## Validation

```sh
make test-qwen4-kernels test-qwen4-q2 test-qwen4-prefill-reuse test-q8-prefill-variants
make test-qwen4-ssd-experts test-qwen4-memory
make test-frontends
make test-qwen4-ngrams
make tests/test_qwen4_ngram_state
make -B -C gguf-tools quants-shared
python3 -m unittest discover -s gguf-tools/tests -p test_qwen4_pack.py
python3 -m unittest discover -s gguf-tools/tests -p test_qwen4_native_ngrams.py
```

The kernel tests exercise the active Metal API. Vision and end-to-end model
checks additionally require the checkpoints described above.
The SSD test needs no GGUF: a sparse temporary file uses a 512-expert table,
including indices 383, 384 and 511, with cold and warm selections, eviction,
prefill staging, different-size MTP layers and failed reads followed by exact recovery. It
compares routed intermediates, reduced outputs and hyper-connection residuals
bit for bit against the resident kernels, including the padded Q2 down rows.
Its test-only backend checks actual `pread` requests against selected expert
ranges and counts returned bytes: ten experts for off-size MTP, 24 for cache
overflow, zero reads for cache hits, and the full 512-expert fallback.
Single-token tests also cover partially cached selections, duplicate IDs,
optional shared experts and all 16 routed slots. A failed SSD read must leave
the completed cached/shared gate/up slots exact, preserve the untouched slots,
and allow an exact retry after draining GPU work.
The memory-estimate test runs without a GPU or model.
Run `tests/test_qwen4_ngram_state MODEL.gguf` under Metal validation to check
failed disk reads during prefill, decode and MTP, then exact recovery.
For a smaller memory footprint, run
`tests/test_qwen4_ngram_state MODEL.gguf --ssd-streaming`. This uses a
256-token context, an eight-token prefill chunk and a 1024-expert cache request
while keeping both comparison sessions and MTP enabled.

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
