# Original-precision V4.1 on one Mac

The experimental `native` recipe preserves released FP8/FP4/BF16/F32 weight
values. It does not run a weight quantizer or use an imatrix. It expands dense
weights losslessly to F32, repacks the original expert FP4 codes and UE8M0 scales
into MXFP4 blocks, and copies Engram codes/scales into disk-only rows. The runtime
also retains the release's FP8/FP4 activation rounding and BF16 boundaries.

This correctness graph supports single-Mac text inference with Metal. It uses
layer-major prefill up to its admitted row capacity (8192 at the local 262K
context), preserving the serial row arithmetic while reusing expert bytes.
Decode and short prefills use the six selected experts directly from the bounded
GPU cache. Indirect buffers are declared as Metal resources and held until their
command buffer completes, avoiding a second allocation and copy of the same
expert bytes. The uncached path retains its staging buffers.
Large prefills borrow the existing expert-cache budget for two
complete layer buffers: one for GPU work and one for the next layer's SSD read.
Original MXFP4 bytes are read directly into these buffers, without a per-token
copy or a CPU router read. Up to 32 rows share a GPU command buffer. Both buffers
must fit the admitted cache budget; it is restored afterward. The local buffers
total 13.45 GiB, replacing part of the admitted expert cache during this work.
For selected-expert streaming, the bounded cache retains unchanged MXFP4 bytes;
its reader starts after routing while the GPU executes the shared expert.
Native sparse attention uses eight SIMD groups per head to compute keys in
parallel. Each dot keeps its 32-lane reduction; value dimensions keep their key
order, 64-slot softmax tree and BF16 probability/output boundaries. The native
arithmetic stays the same. CUDA, tensor parallelism,
DSpark, vision and concurrent decode batching are not admitted for native weights.
The existing Q2/Q4 recipes retain their existing paths.
Native expert reads can be served by macOS's file cache; measured elapsed times
include cache effects and are not uncached-SSD benchmarks.

For the uncensored 1017 checkpoint, pin source revision
`c19ca5ca4eaeb79d29fdf08aea8327da95b1116a` from
`dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8` and verify all 48 source shard
SHA256 hashes. Check free storage for both the 510.3 GB source and approximately
524.8 GB artifact, plus reserve, before downloading. Engram and expert banks must
remain on disk; avoid full-model CPU inference on macOS.

```sh
make
make -C gguf-tools libds4quants.dylib
python3 gguf-tools/deepseek41_quantize.py --hf /path/to/source \
  --out /path/to/native.gguf --quant native \
  --source-repo dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8 \
  --source-revision c19ca5ca4eaeb79d29fdf08aea8327da95b1116a --threads 4
python3 gguf-tools/deepseek41_validate_gguf.py --hf /path/to/source \
  --gguf /path/to/native.gguf --quant native --payload \
  --source-revision c19ca5ca4eaeb79d29fdf08aea8327da95b1116a
./ds4 -m /path/to/native.gguf --ssd-streaming -c 1024 -p 'Hello' -n 8
```

The native payload audit checks every dense value, every expert and every Engram
row against the source recipe, rather than sampling expert banks or Engram rows.
It validates artifact fidelity, not model quality.

## Lossless dense storage on APFS

The optional packer makes a separate APFS clone of an audited native artifact.
For supported dense matrices, it checks that every F32 bit pattern is exactly
representable in BF16, stores the upper 16 bits, then reads back and expands every
value to verify bitwise equality. A matrix with any inexact or nonfinite value
stays in F32. No weight rounding is performed. The original file is read-only;
expert and Engram payloads retain their original clone extents.

```sh
python3 gguf-tools/deepseek41_pack_native.py --source /path/to/native.gguf \
  --output /path/to/native-lossless-bf16.gguf --report /path/to/packing-proof.json
```

This requires macOS and a destination on the same APFS volume. Existing output,
partial and report files are refused. Tensor offsets and logical file size stay
fixed: this reduces resident dense storage and GPU traffic, not the apparent
file size. The pinned 1017 artifact has 321 eligible matrices containing
7,142,113,280 values, saving 14,284,226,560 resident bytes (13.30 GiB).

The native BF16 matvec expands these values into F32 registers. The F32 and BF16
storage paths use the same arithmetic for each matrix. FP8 activation rounding
and BF16 output boundaries are unchanged. Run the payload
audit above on the original artifact before packing; the packer's report is the
proof for the derivative. Validate logits and memory use before changing a
server's model path or increasing its expert cache. Keep the original artifact
as the rollback copy.

Native prefill can widen one MXFP4 expert layer into a checked BF16 RAM cache.
Each decoded F32 bit pattern must survive the widening exactly before that layer
uses it; otherwise the original packed layer is used. The two raw read buffers
plus the widened layer must fit inside the configured expert-cache budget. Disk
weights, per-block accumulation and activation rounding stay unchanged. The
native router computes scores and searches for experts in parallel, keeping the
lowest expert ID for ties and the original serial normalization order.

With a checked BF16 layer, native prefill batches Engram and HC inputs, query/key
projections, normalization, and the final attention projection. Engram and
HC finish before Q reuses their storage; keys are still published and attended
in causal token order with the original per-row arithmetic. After the tile's
attention finishes, FFN work is batched. Independent HC projections and shared-expert
projections share dispatches. Routed token/slot pairs are grouped by expert in
tiles of four, reusing each weight load while retaining each token's 32-value
dot reductions and K-order accumulation. Outputs return to their original slots
before the existing expert-ID sum. Scratch views preserve the attention/FFN
alias lifetimes, and partially completed tiles remain invalid checkpoints.
Smaller prefills and packed-bank fallbacks keep the scalar path.

Wide native prefills reuse the admitted carry buffers to finish every compute
tile in a layer before reading the next layer's experts. Each compute tile stays
within the existing 8,192-row limit, and the sweep stays within the allocated
carry capacity. Residuals already have BF16 values, so their compact transfer
preserves their bits without another rounding step; F32 mixers and index
selections retain their original storage. The expert-cache budget and scratch
allocations do not increase. Cancellation leaves a partial sweep invalid, and
native sweeps always evaluate every layer without decoder-suffix pruning.

## Bounded graph verification

Torch, NumPy and tokenizers are required only for the independent test oracle.
Use the unchanged `inference/model.py`, `engram.py`, `kernel.py` and `config.json`
from that pinned source. The fixture generator checks their hashes. CUDA kernels
are translated to bounded CPU operations in `tests/deepseek41_native_reference.py`;
the reference model and its control flow stay unchanged.

Native FP8 and FP4 projections accumulate each 32-value scale block before
adding the next block, as the released kernels do. The CPU translation keeps
these boundaries too; one long dequantized BLAS dot is not the same FP32
calculation. HC expansion sums the residual products before adding the scaled
sublayer output. Native HC expansion and weighted sums also preserve the released
source's separate FP32 product and addition boundaries: Metal safe math alone
still permits contraction, so these expressions explicitly disable it. Focused
cancellation cases exercise scale blocks, sum order and product rounding through
the native APIs and the independent CPU translation.

Native key rotation also fixes the FMA order before its BF16 and FP8/FP4
rounding. This keeps cache values stable when moving between fused and separate
rotation passes; a production-width midpoint case is included in the Metal checks.

```sh
make tests/test_deepseek41_metal
./tests/test_deepseek41_metal --native-source
python3 tests/test_deepseek41_native_reference.py
```

These corrections change activations without changing weights. Native snapshot
arithmetic revision 3 rejects both earlier snapshot revisions before modifying
the live graph; the server must recompute those prefixes. The graph fixture also
checks that rejection of either older tag leaves the current checkpoint intact.

```sh
make tests/test_deepseek41_native_graph
python3 tests/make_deepseek41_native_fixture.py --reference-dir /path/to/reference \
  --out /tmp/native-fixture --tokens 520
python3 tests/make_deepseek41_native_fixture.py --reference-dir /path/to/reference \
  --out /tmp/native-control --tokens 520 --double-accumulation
./tests/test_deepseek41_native_graph /tmp/native-fixture/model.gguf \
  /tmp/native-fixture/tokens.i32 /tmp/native-fixture/metal.f32
./tests/test_deepseek41_native_graph /tmp/native-fixture/model.gguf \
  /tmp/native-fixture/tokens.i32 /tmp/native-fixture/metal-stream.f32 stream
python3 tests/check_deepseek41_native_logits.py /tmp/native-fixture /tmp/native-control
```

The fixture retains all 40 layers, 64 attention heads, 8 output groups, both
Engrams, both compression ratios and 32 index heads, with narrower hidden widths
and 8 rather than 384 experts. It crosses the raw ring wrap and the 512-entry
sparse-selection boundary. It does not cover production widths, all 384 experts
or candidate block pruning beyond 16K compressed entries. The C runner asserts
exact continuation from a freshly
restored serial or batched snapshot at position 129, including unfinished
compression pairs, exact prefill/decode continuation, and multi-tile continuation
through the sparse-selection boundary. A single 520-row prefill also matches the
differently partitioned appends. Canceled prefills cannot publish a partial
checkpoint; restoration of a prior checkpoint and cache use after cancellation
are checked. The Python check requires finite logits,
independent first-token agreement within 1e-6 and bit-identical resident/SSD logits.

Later reference-logit differences are reported without pretending they are zero:
changing only CPU matrix accumulation to F64 can itself change BF16/FP8/FP4
rounding decisions and subsequent logits. These fixtures do not establish
bit-identical CUDA output, intelligence preservation or full-checkpoint quality.

## Actual-checkpoint source logits

After all pinned source shards have been verified, the optional one-token source
oracle avoids the engine's large CPU model mapping. It constructs one released
Block at a time, leaves unselected experts and Engram tables on the meta device,
loads only selected experts, and reads Engram/head rows with pread. It records
peak CPU RSS and a trace of all 40 layers. Fixture generation first checks this
oracle against the unchanged fully loaded reference, including its F64 control.

```sh
python3 tests/deepseek41_native_checkpoint_reference.py --hf /path/to/source \
  --token-id 0 --out /tmp/source-first-token.f32
./ds4 -m /path/to/native.gguf --metal --ssd-streaming -p '<｜begin▁of▁sentence｜>' -c 1024 \
  --dump-logits /tmp/metal-first-token.json
```

Both use BOS token 0. The CLI recognizes the rendered special token without
`--raw`. Compare all 129,280 logits, record numerical differences
and the top choices, and check continuation separately. Original FP4 experts
account for the CLI diagnostic's routed `quant_bits: 4` field. The native recipe
preserves their codes and scales.

The earlier raw `a` comparison (token 67, no BOS) used a CPU translation that
collapsed FP8/FP4 scale blocks into one dot product. That comparison is superseded
by the corrected block accumulation and HC product rounding. For raw `a`, the
corrected CPU and Metal paths choose different next tokens (260 and 201), with
relative L2 0.734 and next-token KL divergence 2.752. This is not source parity.

A 38-token prompt produced by the released encoder for `What is 2 + 2?`, with
maximum reasoning, selects the same next token (`We`) on both paths. Its
source-to-Metal next-token KL divergence is 0.000650, total variation is 0.005103,
and raw-logit relative L2 is 0.233. These are results for one prompt, not a general
quality guarantee. Traces isolate tiny FP32 reduction differences that cross
BF16/FP8 rounding boundaries; even CPU batch and serial execution can differ. A
control using reference HC coefficients matches all residuals through layer 30,
then one attention value differs by one BF16 step with identical inputs. Its
FP32 dot-product reduction differs between the CPU and GPU. The native GPU path
is retained, without CPU fallbacks or compensated summation. CUDA parity remains
unverified; functional coding acceptance must be measured separately.

## Local server

Keep one large model process running. Stop the server before a separate CLI test.

```sh
./ds4-server --metal --ssd-streaming -m /path/to/native.gguf \
  --ssd-streaming-cache-experts 64GB \
  --host 127.0.0.1 --port 8000 --ctx 262144 --tokens 32768 --prefill-chunk 8192 \
  --kv-disk-dir /path/to/server-kv --kv-disk-space-mb 8192 \
  --kv-cache-min-tokens 1 --kv-cache-reject-different-quant
```

The server exposes the existing OpenAI-compatible API at `http://127.0.0.1:8000/v1`. Native weights support text input only. Large agent prompts still require substantial GPU work and SSD reads; measure them separately from small bounded requests.

The expert cache above stores original weights in RAM; it is independent of the
disk KV cache. Its budget is capped by the existing memory admission checks.
Caching can avoid repeated SSD reads, but it does not make the whole model fit in
RAM. Layer-major prefill improves expert locality and batches independent
projections while preserving each row's reductions and causal attention order;
it does not quantize the weights.
Reusing a saved prompt
prefix avoids recomputing that prefix on later turns. Measure cold and continued
requests separately; OS file caching also affects these timings.

For a serial scheduling comparison, set
`DS4_METAL_DISABLE_V41_LAYER_PREFILL=1` before starting the process. To disable
the early expert reads independently, set
`DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD=1`. These existing diagnostic
switches do not change the weights or native arithmetic. The early-read switch
applies to selected-expert streaming; whole-layer reads use their bounded reader.
A canceled partial tile
cannot be saved as a valid checkpoint because its layer frontiers differ.
