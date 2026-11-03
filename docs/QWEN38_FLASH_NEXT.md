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
cache updates. Fully cached selections use one gate/up pass. Cached batched
prefill uses overlap on every Metal device: resident experts run while missing
weights load, then missing gate/up computation overlaps the down-weight reads.
Temporary staging retains its existing schedule; an MTP layer whose
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

Q4_K gate/up shares activation reads between the shared expert's Q8
projections and uses packed activation/scale loads while preserving the
original FP32 accumulation order. On M1 Max, the 640-to-2560 MXFP4 down
projection also enables prefetch by default. `DS4_QWEN4_MOE_DOWN_PREFETCH=0`
restores the plain down kernel for diagnostics. Safe math always retains that
plain kernel, including when prefetch is explicitly requested, because its
noncontracted arithmetic differs from the prefetched kernel's explicit FMA.

A local Q4 SSD check on M1 Max 32 GiB used `--ctx 32768`,
`--prefill-chunk 1024`, `--temp 0 --nothink`, `-n 100`, and the prompt
`Narrami in italiano la storia di Roma`. Three alternating pairs comparing
the previous kernels with these changes produced:

| Decode | Runs (t/s) | Median (t/s) |
| --- | --- | --- |
| Before | 6.02, 6.07, 6.26 | 6.07 |
| Q4 kernel updates | 6.42, 6.03, 7.27 | 6.42 |

All generated output was identical. The observed median gain was 5.8%; SSD
and system variation remained substantial. Prefill medians were 6.94 and
6.91 t/s. These changes retain the existing prefill and row-dispatch policies;
Q2's smaller sparse-prefill tiles did not show a stable benefit for Q4 in
local comparisons. The measurements do not establish gains on other devices.

MXFP4 prefill also decodes each 16-value half-block with one exponent read
and packed byte loads, retaining the original multiplication and final
half/float conversion. Compared with `46425263`, alternating resident-kernel
benchmarks on M1 Max at 2048 tokens and 10 routed slots reduced median
gate/up-plus-down time from 70.65 to 66.31 ms with 32 experts and from 81.57
to 77.34 ms with 256 experts (6.1% and 5.2% less time).
The staging oracle checks all exponent bytes, nibble codes and byte
alignments against two original 8-value decoders in both math modes.

On M1 Max, batched MXFP4 down rows also reuse activations across two output
rows. A function constant enables this only for multi-token NR2 dispatches;
single-token decode and NR1/NR4 retain the original row loop. A warmed
same-process comparison with 16 alternating samples per variant, 128 calls
per sample, width 640, 2560 output rows and 10 routed slots measured:

| Two-token down kernel | Before GPU time | After GPU time |
| --- | --- | --- |
| Q8 shared expert | 176.62 us | 150.70 us |
| MXFP4 shared expert | 216.20 us | 188.48 us |

Outputs were bit-identical, and single-token GPU timings did not regress.
Full Q4 SSD runs remain sensitive to I/O: two alternating pairs at context
32768 and chunk 1024 gave prefill medians of 22.94 and 23.41 t/s on a
201-token prompt, with all 100 generated tokens unchanged. With MTP and two
draft tokens on the Rome prompt, median generation remained 7.65 t/s despite
the faster down kernel. These checks compare against `46425263`; they do
not imply the same percentage gain for complete inference or other devices.

Leave the expert-cache budget automatic initially. A plain count passed to
`--ssd-streaming-cache-experts` requests dynamic cache slots; an `NGB` budget
also includes the reserved layer-staging space. Startup accounts for the
context, prefill chunk, static weights, vision weights when present, and
staging before choosing the cache size. Reduce the context or prefill chunk
if those fixed requirements leave insufficient memory.
The total budget is the smaller of 87.5% of physical RAM and Metal's
recommended working set, with a further 2 GiB reserved for runtime overhead.
The recommendation is not discounted again, and admitted static weights stay
eligible for locking in RAM as the expert cache grows.

Qwen fills its expert cache on demand, so `--ssd-streaming-cold` does not
change its preload behavior. Popularity preloading and the GLM-specific
`--ssd-streaming-full-layers` option are not supported for Qwen.

On M1 Max, the shared half-operand MoE prefill path requires IQ2_XXS gate/up,
Q2_K down and at least 8192 actual tokens in one streamed batch. The configured
`--prefill-chunk` is only an upper limit: a 5760-token prompt with chunk 8192
does not activate this path. Comparing that prompt at chunks 2048 and 8192
measures the change in batching, not an isolated half-operand kernel gain.

On a DGX Spark, build with `make cuda-spark` instead of `make`. Both Q2 and
Q4 fit resident on an otherwise idle 128 GB Spark; the n-grams stay on SSD.
The same commands support text, vision, MTP, steering and text-session
checkpoints in `ds4`, `ds4-agent` and `ds4-server`. No CUDA-specific model
conversion or feature flags are needed.
Leave `--prefill-chunk` unset on the Spark to use the faster 8192-token
default. The 1024-token example above saves memory on smaller Macs.

Use the same model options with `ds4-agent` or `ds4-server`.
Add `--mtp` for speculative decoding using the built-in MTP weights.
`DS4_QWEN4_MTP_PREFILL` controls preparation of the predictor's prompt history:

| Value | Predictor history for new prompt rows |
| --- | --- |
| `on` (default, also `1`) | Prepare it for both resident and SSD streaming runs |
| `off` (also `0`) | Skip historical projections and initialize those cache rows to zero |
| `auto` | Prepare it only with SSD streaming |

For example, this opts into streaming-only preparation:

```sh
DS4_QWEN4_MTP_PREFILL=auto ./ds4 -m gguf/Qwen3.8-Flash-Next-Q2.gguf --mtp --ctx 8192
```

Use `on` for resident workloads where the decode
benefit outweighs the prefill cost, and compare total wall time on your prompts.
`off` and resident `auto` keep MTP drafts and target verification enabled, but
the cold predictor history can change drafts and reduce their acceptance rate.
Zero keys still participate in predictor attention; this is not equivalent to
the fully prepared history. Target prefill caches and logits are unchanged.
Decode continues to prepare real predictor rows. Each prefill chunk also
retains its final row for real preparation at the next boundary, keeping
checkpoint restore consistent with uninterrupted prefill.
The setting applies to newly prefilled rows;
reused or restored checkpoints retain their existing predictor cache contents.
Without `--mtp`, this setting has no effect.

`ds4-bench` benchmarks ordinary decoding; it does not accept `--mtp`.
`--nothink` disables thinking. The server exposes
`qwen3.8-flash-next`, `qwen3.8-flash-next-chat`, and
`qwen3.8-flash-next-reasoner` aliases. Tool calls use the native
`<tool_call><function=...><parameter=...>` format.

With resident weights on Metal, `ds4-server --batched-session N` decodes
the slots together. Work that
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
fallback. Metal SSD expert streaming also uses the ordered fallback and
keeps each session's streaming workspace; it does not use the resident batch
arena. CUDA currently decodes sessions in order.

Disk KV checkpoints include recurrent state. Rewinding to an earlier position
replays the retained prefix on the next evaluation. The native context is
262144 tokens; `DS4_QWEN4_YARN_FACTOR=2` or `=4` enables static YaRN for
longer contexts, with a possible quality cost on shorter prompts.

## Metal tuning on M1 Max and M2-M4

The established M1 Max simdgroup optimizations also apply to Apple M2, M3
and M4, including their Pro, Max and Ultra variants, within the same tensor
shapes, batch sizes, math modes and diagnostic overrides:

- Wide Q8 decode projections, HC gate/mix reuse and short MTP Q8 projections.
- IQ2 gate/up, Q2_K down-row reuse and the Q4 model's MXFP4 down prefetch.
- Small HC-down and HC-normalization prefill batches, and large Q8 SSD prefill.
- Sparse IQ2/Q2 SSD tiles, Q4/MXFP4 K32 tiles, optional half-operand staging
  for large SSD batches and release of that scratch before Q4 decode.

M3 Ultra retains its dedicated MoE specialization, row/group geometry and
prefill tile policy; the other shared Q8, HC and MTP choices still apply.
M5/M6 retain their existing policy. The selection does not expand to M1
base/Pro/Ultra or to future generation names.

This widens device selection, not the kernel arithmetic or GGUF formats.
Local validation uses M1 Max. An independent M4 Max Q2 resident test reported
about 7% more decode throughput after widening the decode gates, with matching
outputs; it did not validate M2/M3, other M4 variants, or SSD performance.
See the [M4 Max test report](https://github.com/antirez/ds4/pull/1056#issuecomment-5922543210).
Device eligibility therefore does not imply a measured speedup on every chip.

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
Tensor parallelism and pipeline execution are not implemented for this model
yet. SSD expert streaming is supported on Metal only. ROCm is not supported.
CPU code is a correctness reference, not a general inference backend.


M6 uses the same Metal dispatch defaults as M5, including the Qwen indexer,
attention merge, HC mixers and MoE scheduling. Existing M5 diagnostic rollback
switches apply to both generations; tensor kernels still require runtime
capability support. The F16 HC mixers explicitly preserve their accumulation
order in both normal and safe shader math modes. This does not guarantee
bit-identical logits across GPU generations: tensor matmuls retain their own
accumulation order.

## Validation

```sh
make test-qwen4-kernels test-qwen4-q2 test-qwen4-prefill-reuse test-q8-prefill-variants
make test-qwen4-ssd-experts test-qwen4-memory
make test-qwen4-hc-math test-metal-device-policy
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

To compare ordinary one-shot generation with the session path at identical
prefill boundaries, including every vocabulary logit during continuation:

```sh
make tests/test_qwen4_generation
./tests/test_qwen4_generation gguf/Qwen3.8-Flash-Next-Q2.gguf --ssd-streaming
# Exercise sparse attention and many chunk boundaries with a longer fixture:
./tests/test_qwen4_generation gguf/Qwen3.8-Flash-Next-Q2.gguf \
  --ssd-streaming --chunk 128 --ctx 8192 --tokens 5760
```

Resident Metal prefill preserves main's FP32 small-batch projections and
split-K reductions. SSD streaming keeps its original half/unsplit prefill
reference and the parent attention policy across internal partitions. Applying
the SSD reference to resident GDN alpha/beta changed prompt state and later
greedy tokens; these policies are now scoped to the graph's streaming mode.
The decode tuning shared by M1-M4 is retained.

`make test-qwen4-resident-arithmetic` compares actual-shaped resident
projections and internal attention partitions with an independent main
`0aaea5a` dispatch reference, without a GGUF. It also checks the SSD reference
and ordinary decode. `test_qwen4_generation --prefill-reference` selects the
legacy unsplit SSD reference and requires `--ssd-streaming`.

Omit `--ssd-streaming` on a machine that can keep the model resident. The
test checks explicit chunk precedence, progress frontiers, greedy tokens and
full logits. Different chunk sizes can change reduction order and routed
expert choices; compare the actual chunk boundaries before diagnosing token
drift. In particular, upstream `8db1d1d` ignores `--prefill-chunk` in the
ordinary one-shot Qwen path, although its sessions honor the option. For an
A/B comparison with that revision, also set `DS4_QWEN4_PREFILL_CHUNK` to the
same value in both binaries. This branch uses an explicit chunk in preference
to that environment setting and clamps it to the context size.

On CUDA, use `make test-qwen4-cuda` for the kernel tests. They compare the
active kernels with independent CPU references, without model weights.
Vision and end-to-end checks additionally require the checkpoints above.
Run `tests/test_qwen4_ngram_state MODEL.gguf` on either backend to check
failed disk reads during prefill, decode and MTP, then exact recovery.
For a smaller memory footprint, run
`tests/test_qwen4_ngram_state MODEL.gguf --ssd-streaming`. This uses a
256-token context, an eight-token prefill chunk and a 1024-expert cache request
while keeping both comparison sessions and MTP enabled.

To compare the predictor's cache-only prefill with a complete MTP pass:

```sh
make tests/test_qwen4_mtp_prefill
./tests/test_qwen4_mtp_prefill gguf/Qwen3.8-Flash-Next-Q2.gguf --ssd-streaming
```

This checks one-token prompts, chunk boundaries, non-argmax continuation,
stale cache contents and payload restore, followed by greedy, depth-three
and exact-sampling continuation checks. It uses a 128-token context and a
1024-expert cache request. The small-buffer MTP CLI regression also accepts
`--ssd-streaming`.

For the sparse-attention boundary and a 17408-token prefix, use the longer
predictor regression:

```sh
./tests/test_qwen4_mtp_prefill gguf/Qwen3.8-Flash-Next-Q2.gguf \
  --ssd-streaming --long-context --prefill-chunk 2048 \
  --dump-prefix /tmp/qwen-mtp-long-2048
```

Repeat with `--prefill-chunk 128` and a different dump prefix. The fixture
uses repeated, fixed chat token IDs, one graph, a 17416-token context and a
1024-expert cache request. It runs the trunk once and probes actual residual
rows across the first sparse query and at the final prefix. Each probe
compares cache-only preparation, the optimized last-row predictor and a
recursive draft with the frozen full-layer reference, using identical input
rows and lookahead tokens. It checks complete logits, cache contents, EH
projections, residuals and unchanged trunk state for T1/T2/T3 and exact T3
verification. Destination cache rows are poisoned before each replay.
Bit-identical comparisons are between the candidate and frozen reference
replayed from the same saved predictor state within each run. Equality between
chunk sizes is checked separately: chunk-dependent grouping can change the
arithmetic used to build the historical predictor cache.

`--dump-prefix` writes the fixture IDs, target/trunk captures, reference and
candidate predictor logits, and a JSONL manifest for comparisons between
builds. `--prefix-tokens N` changes the fixture length; it must be a multiple
of four covering the sparse boundary. These are implementation checks on a
synthetic sequence, not a quality evaluation or an acceptance-rate benchmark.
Comparing MTP acceptance between commits additionally requires the same
prompt, sampling settings and draft-depth policy. A trunk arithmetic change
can alter predictor inputs even when the predictor itself remains correct.

On an M1 Max with 32 GiB and Q2 SSD streaming, the 17408-token fixture passed
at chunks 128 and 2048 on `11811b9`, and at chunk 2048 on `04c0867` and
`89986c3`. At chunk 2048, all 36 captured tensors from `89986c3` and `11811b9`
were byte-identical. Differences from `04c0867` already appeared in the trunk
residuals after the attention fix in `89986c3`; both predictor implementations
followed those changed inputs.

Between chunks 128 and 2048 on `11811b9`, the two captured trunk slices and
final target logits were byte-identical. Predictor logits differed by at most
0.0001266003, with no argmax change in the 16 full/recursive comparisons. A
separate replay of identical rows as T3, T1+1+1, T2+1 and T1+2 found arithmetic
differences beginning at the HC down projection; its per-stage differences
were identical on `04c0867` and `11811b9`. This identifies an existing source
of grouping-dependent predictor cache values, not a demonstrated quality or
acceptance regression. The full-model runs above do not exercise the
8192-row half-operand path.

The independent `test_metal_qwen4_moe_half` comparison against the shader
source from `89986c3` also passed 31 fixtures, including 8192 rows, with Metal
API validation. Full shader validation exceeded the M1 threadgroup-memory
limit for the NT8 gate/up stress case (36864 bytes reported, 32768 available).
A separate NT1/2/4 subset passed with shader validation, including the
8192-row fixture at mid NT4/down NT4. These kernel checks do not replace an
8192-chunk full-model acceptance comparison.

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
