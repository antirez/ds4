# Qwen3.8-Flash-Next Q4 fast pack

This branch adds the versioned `ds4.qwen4.fast-pack` v3 Q4 format and the
Qwen3.8-Flash-Next model identity. The model architecture and pack manifest are
DS4-specific, but every quantized tensor uses a standard GGML/GGUF block type.
An arbitrary GGUF or an Unsloth quant is not accepted. Legacy custom v1/v2 packs are
deliberately rejected instead of being interpreted as the v3 block recipe.
The same loader also accepts the fixed v4 mixed-Q2 profile documented below;
the version is part of the quantization contract, not a generic format switch.

## Pack conversion

The converter reads the immutable official BF16 checkpoint, not a quantized
derivative. It writes one
`Qwen3.8-Flash-Next-Q4KExperts-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf`
base artifact, an external CPU-mapped
`Qwen3.8-Flash-Next-PLE-Q4_1.gguf`, and optional vision and MTP sidecars.

```sh
uv run --script gguf-tools/qwen4_pack.py \
  --src /path/to/Qwen3.8-Flash-Next-metadata \
  --out /path/to/qwen3.8-flash-next-q4 \
  --remote-repo Qwen/Qwen3.8-Flash-Next \
  --source-revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --tokenizer-template /path/to/qwen-tokenizer-template.gguf
```

The source directory may contain the whole official checkpoint or only its
`config.json` and `model.safetensors.index.json`. With `--remote-repo`, missing
official files are fetched from the pinned revision and one source shard at a
time is staged, converted into its final random-write offsets, fsynced, and
removed. The output directory therefore needs space for the final pack plus at
most the largest source shard, not the complete 360 GB BF16 checkpoint. Before
preallocation, conversion admits the exact selected artifact sizes, the largest
remote stage (when needed), and the same 32 GiB safety reserve used by the GLM
converter. A
conversion state file resumes automatically at the last completed source
shard. State v4 authenticates every completed base/vision/MTP tensor range,
PLE row range, and PLE auxiliary range before skipping it; a damaged or older
unauthenticated partial conversion is rejected. `--fresh` deliberately
replaces a partial conversion.

Write v3 to a new output directory. The converter refuses to overwrite a
directory containing a finalized manifest, even with `--fresh`, so a failed
migration cannot destroy an existing pack. Legacy v1/v2 output directories
must not be reused.

The converter records the source revision, exact model geometry, quantization
recipes, per-artifact SHA-256 digests, and a deterministic tensor-manifest
digest in `qwen3.8-flash-next-q4.manifest.json`.

The quantization layout follows the DeepSeek/GLM strategy:

- routed gate/up/down experts: standard `Q4_K`;
- shared experts, output, GDN, QSA, PLE projections, and ordinary dense
  projections: standard `Q8_0`;
- token embedding, router, hyper-connection, norms, convolutions, and other
  sensitive control tensors: source BF16/F32;
- PLE n-gram rows: standard `Q4_1`, five 32-value blocks per 160-value row.

The routed down projection has a logical input width of 640. Its physical
`Q4_K` row is zero-padded to 768 so each row contains three complete 256-value
blocks; the runtime validates both widths and never exposes the padded tail as
model state. The external PLE uses `Q4_1` instead of padding every 160-value row
to a 256-value `Q4_K` block. This keeps each sparse SSD row at 100 bytes while
removing the former custom weight/scale/bias planes.

### 64 GiB mixed-Q2 profile

Pack v4 is the smaller-backbone profile for a 64 GiB Mac. It quantizes routed
expert gate/up tensors as standard `IQ2_XXS` and routed down tensors as
standard `Q2_K`; embeddings, control tensors, Q8 dense projections, shared
experts, output, and the SSD-backed `Q4_1` PLE keep the v3 precision choices.
The base metadata must declare `routed=mixed-q2`,
`routed_gate_up=IQ2_XXS`, and `routed_down=Q2_K`. The loader checks those
values and every routed tensor type before allocating Metal caches.

```sh
uv run --script gguf-tools/qwen4_pack.py \
  --profile q2 \
  --src /path/to/Qwen3.8-Flash-Next-metadata \
  --out /path/to/qwen3.8-flash-next-q2 \
  --remote-repo Qwen/Qwen3.8-Flash-Next \
  --source-revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --tokenizer-template /path/to/qwen-tokenizer-template.gguf \
  --imatrix /path/to/imatrix_unsloth.gguf_file \
  --threads 4 \
  --no-mtp
```

The v4 artifacts and resumable state are distinct from v3:
`Qwen3.8-Flash-Next-IQ2XXSGateUp-Q2KDown-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf`,
`Qwen3.8-Flash-Next-Q2-PLE-Q4_1.gguf`,
`qwen3.8-flash-next-q2-vision.gguf`,
`qwen3.8-flash-next-q2.manifest.json`, and
`qwen3.8-flash-next-q2.conversion-state.json`. The current geometry projects
43.139 GB (40.176 GiB) of base tensor payload, 32.00 GB for the SSD-only PLE, and 0.48 GB
for optional vision. The PLE mapping is sparse and is not made resident as part
of the base mapping.

`IQ2_XXS` requires the pinned Qwen GGUF importance matrix from
`unsloth/Qwen3.8-Flash-Next-GGUF` revision
`c8b5954a88c2775c546b92593eda40ea041d3176`. The converter requires SHA-256
`a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154`, validates
all gate/up/down sum and count tensors for 48 layers by 512 experts, normalizes
every positive-count expert independently, and records the dataset, chunk
geometry, digest, revision, and fallback provenance in both conversion identity
and the final manifest. The verified matrix has 24 zero-count part-experts:
16 gate/up and eight down. Only those entries use the deterministic
per-expert input-column weight-energy fallback; missing or malformed entries,
negative/nonfinite statistics, and any other imatrix digest fail closed.

Routed down remains logically 640 columns and physically 768 columns. The
converter quantizes each expert independently with its normalized 640-value
importance vector, zero-pads both source rows and importance to 768, and emits
three standard `Q2_K` blocks per row. `--threads` controls a bounded ordered
expert executor; changing the worker count does not change artifact bytes or
the resumable conversion identity. Use a low value while another conversion is
active, then resume with more workers later.

Q2 MTP is a matching-profile
sidecar contract in the runtime. Conversion selects the deterministic
per-expert input-column weight-energy importance for every MTP routed
expert with `--mtp-imatrix weight-energy` (the pinned unsloth imatrix
covers only the 48 base layers), and `--rebuild-mtp` atomically adds the
v4 MTP sidecar to an already-finalized pack without rewriting the base or
PLE artifacts; `--tokenizer-template` is required only for full
conversions. A v3/Q4 MTP sidecar is still rejected with a v4/Q2 base.
The measured Q2+MTP acceptance under that selection is lower than the Q4
sidecar's (about 2.2 versus 2.65 accepted tokens per cycle) and Q2+MTP
currently nets below Q2 plain decode; the sidecar is provided for
experimentation, not as a speedup. Vision and PLE sidecars likewise carry
the v4 pack ID/version even though their tensor recipes are unchanged.

The small visual position grid remains BF16 because the patch kernel performs
direct four-row lookup and bilinear interpolation rather than a projection.
The official visual FC2 width is 4304, so conversion appends 16 zero columns
before `Q8_0` quantization and the Metal graph pads the matching activation
rows to 4320. This preserves the declared dense recipe without changing FC2.
The visual MLP uses Metal's precise `tanh` intrinsic for
`gelu_pytorch_tanh`; the fast intrinsic can produce non-finite values in the
negative tail reached by the official FC1 weights.
If a format correction affects only that sidecar, `--rebuild-vision` rebuilds
it atomically from the official visual source shard and merges the new tensor
records and checksum into an already-finalized pack; it does not rewrite the
base GGUF or the 32 GB PLE table.

Conversion is deterministic for a fixed checkpoint, tokenizer template, DS4
native quantizer source, NumPy version, and source revision. Q4_K pins the
upstream Apple reference accumulation order on every host and has a checked-in
exact-byte BF16 fixture. The converter uses the same local `libds4quants`
pattern as the GLM tooling; build it with
`make -C gguf-tools quants-shared` or select it with `--quants-library`.
`--dry-run` validates and prints the partition without writing the pack.
`--no-vision` and `--no-mtp` omit optional sidecars.

## Loading and diagnostics

The single base GGUF is passed with `--model`; the manifest locates and binds
the external PLE and any requested optional sidecars before Metal cache
allocation.

```sh
./ds4 --inspect \
  --model /pack/Qwen3.8-Flash-Next-Q4KExperts-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf \
  --ple /pack/Qwen3.8-Flash-Next-PLE-Q4_1.gguf \
  --vision /pack/qwen3.8-flash-next-q4-vision.gguf \
  --mtp-model /pack/qwen3.8-flash-next-q4-mtp.gguf \
  --prefill-chunk auto
```

Loading rejects legacy v1/v2 manifests, missing, duplicate, or extra tensors
and artifacts, geometry drift, wrong GGUF block recipes, unsafe artifact
names, a mismatched tensor-manifest digest, and sidecars from another pack.
Like the DeepSeek and GLM loaders, the trusted-local policy checks artifact
sizes, GGUF structure and metadata without scanning every payload byte. The v3
directories are exact: 1,211 base tensors, 333
vision tensors, 32 MTP tensors, and four PLE tensors. The summary reports the
admitted prefill cap, QSA microtile size, active Metal specializations, PLE
staging mode, partial-RoPE behavior, and sidecar state.

Single-base packaging reduces the number of files. Normal trusted-local startup
does not stream the 73.6 GiB base GGUF or external 29.8 GiB PLE sidecar merely
to recompute their converter-recorded hashes.

Automatic prefill admits 8192 tokens only for an uncached suffix of at least 8192
tokens with native kernels and sufficient transient memory. Prefix-cache
resumes use 2048. Explicit `2048`, `4096`, or `8192` requests are strict and
fail with a memory or geometry diagnostic instead of silently falling back.

PLE retains an SSD-backed CPU mapping and open descriptor; rows are read on
demand. For each graph call, DS4 hashes the two- and three-token histories and
submits the
first 16-row-per-token gather before zero-based transformer layer 0 starts.
Qwen consumes that staged data at zero-based layer 1 (the second transformer
layer), so layer 0 computation masks the first SSD read. Longer calls continue
through two alternating BF16 staging buffers while the GPU consumes the prior
512-token tile; the full 51B-parameter table is never made resident in unified
memory.

The production default remains the sparse `pread` gather. Set
`DS4_QWEN4_PLE_GATHER=mmap` to run the experimental demand-paged gather, or set
it explicitly to `pread` as the kill switch. The choice is parsed once when the
two-slot stager is created; both modes preserve row order, BF16 output, double
buffering, and the layer-0 overlap. `mmap` reuses the reference mapping decoder
and applies `POSIX_MADV_RANDOM` (or `MADV_RANDOM`) to the mapping. The startup
diagnostic, runtime summary, and cold-evidence record report the chosen mode and
whether that advice succeeded. Mapping the 29.8 GiB sidecar only reserves
virtual address space: selected pages are faulted on demand, and DS4 neither
locks nor scans the complete mapping.

The mapped pack is immutable for the lifetime of the server. Replacing,
truncating, or modifying the same inode violates the trusted-pack contract and
can deliver `SIGBUS` while an mmap gather touches the changed range. Keep the
artifact unchanged and switch back to `pread` if mmap page-fault or
memory-pressure behavior is unsuitable. This experiment deliberately adds no
row cache, deduplication, page coalescing, `DONTNEED`, or pack-format change.

The production `pread` gather runs one tile's rows across
`DS4_QWEN4_PLE_GATHER_THREADS` threads (default 32; `1` restores the former
serial loop; gathers under 512 rows — decode and MTP — always stay serial).
The rows of a prefill chunk are scattered 100-byte reads over the whole
sidecar; at queue depth one a fresh region cost ~60 us/row, which stalled the
layer-1 consumption points for 6.9-7.7 s per fresh 8192-token chunk (the
per-tile waits are effectively serial because each next tile's gather starts
only after the previous wait returns). Threads write disjoint output slices,
so the staged BF16 bytes are identical and full-model greedy dumps are
SHA-identical between thread settings; a paired cold A/B measured 637 ->
1054 tok/s on that chunk shape with warm throughput unchanged.

The retained PLE descriptor is not a snapshot: the pack must not be modified
while it is served. Startup, benchmark, and runtime evidence bind the
descriptor to its device, inode, size, modification time, change time, and the
converter-recorded manifest SHA-256 without re-reading the complete payload.

Text prefix-cache payloads use a Qwen-specific serializer. It persists the
token frontier and logits together with every FP32 Gated DeltaNet/PLE state,
the live BF16 QSA rows, pooled sparse-index rows, and matching optional MTP
history. Restore rejects a different Qwen geometry or MTP configuration before
touching the live frontier. Multimodal requests remain outside the disk prefix
cache because their image embeddings and three-axis position table are
request-owned.

## Kernel and pack tests

```sh
make test-qwen4
```

The Metal fixture checks `Q4_K` routed-expert and `Q8_0` projection parity,
Gated DeltaNet R4/R2/R1
outputs and FP32 states at 1, 5, 17, 2048, and 8192 tokens with masks, exact
single-query QSA scoring and ordered top-k for FP32/FP16/BF16 inputs through
32768 pooled rows, streaming QSA top-k without a full score sheet, multimodal
three-axis partial RoPE, and dense-causal QSA. The host fixture covers pack
manifests, checksums, prefill policy, Qwen image resizing/patch order, contracted
  M-RoPE tables, n-gram hashing, `Q4_1` PLE dequantization, bitwise pread/mmap parity for
duplicate, reordered, and VM-page-crossing rows, and the two-slot asynchronous
PLE stager lifecycle.

Before handing off or releasing this branch on Apple Metal, run the complete
model-independent regression gate:

```sh
make test-qwen4-release
```

This target runs, in order:

- `test-qwen4-release-core`: extractor and agent tests, model-free server tests
  (including batched-session scheduling, cache, cancellation, API, GLM/Qwen
  prompt, and tool-state fixtures), layer-pack and CPU placement tests, shared
  GPU argument/CLI tests, sampling, Q4_K/MXFP4 CPU dot products, and a compile
  check of the normal CLI/server/benchmark/agent binaries;
- the existing `test-qwen4` aggregate without substituting or omitting any of
  its host, Metal, pack-conversion, or acceptance-validator fixtures;
- the model-free Metal kernel group from `ds4_test`;
- the synthetic GLM 5.3 KDA/DSA primitive suite; and
- the non-Qwen Metal MXFP4 exactness suite, including the checked-in half-LUT
  verification.

Every input in this aggregate is a checked-in deterministic fixture or a small
synthetic tensor. It intentionally does not run the model-backed Metal session
oracle, GLM vision engine, 16K golden generation, 29.8 GiB PLE sidecar, network
server, or full-model performance/quality benchmarks. Those remain separate
acceptance gates because they require checkpoint artifacts or dedicated
hardware; this target must not turn their absence into a silent skip.

For GPU traces, capture `tests/test_qwen4_metal`; its command encoders are named
for the Qwen `Q4_K` MoE and `Q8_0` projection stages, `Qwen Gated DeltaNet R4`,
`Qwen QSA M=1 ... index scoring`, `Qwen QSA streaming top-k`, and the
`Qwen vision ...` encoder stages. Full-model acceptance runs should remain
uncaptured because capture changes residency and timing.

Set `DS4_QWEN4_PROFILE=1` on a small uncaptured run to print one consolidated
per-forward timing line for embedding, total PLE work, CPU gather time,
layer-1 blocking time, dense projections,
Gated DeltaNet recurrence, QSA, routed experts, cache/state updates, optional
MTP, and total latency. The Metal encoder labels above provide the matching GPU
trace intervals without forcing synchronization in normal runs.

Decode and prefill use standard `Q8_0` blocks for main projections and standard
`Q4_K` blocks for routed experts. Qwen-specific dispatch code retains the
model's 512-expert/top-10 routing, exact slot-order accumulation, GDN/QSA
fusions, and the existing reference fallbacks; sharing a qtype does not force
DeepSeek's top-6 kernel geometry onto Qwen.

Single-token decode fuses the HC up projection, sigmoid, and four-stream mix
after the low-rank SiLU pass. The normal performance profile derives Q8_0
copies of the 194 BF16 hyper-connection down/up matrices once at startup;
`--quality` or `DS4_QWEN4_HC_Q8=0` retains the source BF16 matrices. Set
`DS4_QWEN4_HC_UP_MIX=0` to restore the unfused diagnostic path for parity or
performance comparisons. The Q8_0 down projection also fuses its trailing
SiLU for single-token decode; `DS4_QWEN4_HC_DOWN_SILU=0` restores the split
path. A same-build three-run A/B measured about 27.71 versus 27.51 tok/s with
identical continuation hashes, so the fused path is the default.

Ordinary and MTP decode store the Gated DeltaNet recurrent matrix in BF16 in
the normal performance profile. Replay-free MTP capture stores those recurrent
slots directly as BF16 while keeping convolution and PLE state in FP32. Set
`DS4_QWEN4_GDN_STATE_BF16=0` to retain FP32 state globally,
`DS4_QWEN4_MTP_GDN_STATE_BF16=0` to retain it only for MTP, or use `--quality`
to select the FP32-state profile. Disk KV checkpoints remain canonical FP32 on
disk and convert at their graph boundary, so the runtime modes can restore each
other's checkpoints without changing the payload format.

The Gated DeltaNet decode recurrence keeps the R4 value-row tile on Apple
GPUs. An alternating M3 Ultra full-model sweep of otherwise identical
128-token runs measured R4 at 22.70/22.74 tok/s, versus R8 at 22.46, R16 at
22.27, and the CUDA/vLLM-style R32 (`BV=32`) tile at 22.33 tok/s. All variants
produced identical output. The larger experimental kernels were removed;
their additional register pressure did not transfer profitably to Metal.

The routed `Q4_K` expert down projection maps each of the ten selected experts
to its own work partition during decode, treats activation columns 640..767 as
zero, and applies route weights in the original slot order.

Layer zero remains its own early command buffer so the first transformer layer
can overlap the PLE read. Decode then groups eight layers per command buffer by
default; `DS4_QWEN4_LAYERS_PER_COMMAND_BUFFER` retains the diagnostic override.
A same-build depth sweep measured 27.62, 27.66, 27.72, 27.68, 27.70, 27.66,
and 27.43 tok/s for depths 2, 4, 8, 12, 16, 24, and 48 respectively, with
identical outputs, confirming eight as the retained setting.
Set `DS4_METAL_DECODE_STAGE_PROFILE=1` and optionally
`DS4_METAL_DECODE_STAGE_PROFILE_LAYER=N` for synchronization-heavy per-stage
timing that must not be used as a throughput result.

The stage profiler starts at the selected layer boundary; otherwise the first
reported interval would incorrectly include unsent work from all preceding
layers. Isolated M3 Ultra decode samples put GDN/QSA at roughly 0.49--0.58 ms,
routed MoE at 0.47--0.52 ms, attention HC read at 0.38--0.41 ms, MLP HC read at
0.36--0.40 ms, and the two small HC writes at about 0.03--0.05 and
0.26--0.29 ms. The major stages are therefore comparable rather than GDN alone
accounting for most of decode.

The standard Q8_0 M=1 projection keeps four SIMDgroups. An exact-output sweep
of 1, 2, 4, 8, 16, and 32 SIMDgroups measured approximately 34.15, 34.48,
34.50, 34.45, 34.29, and 33.84 tok/s respectively, so the existing four-group
geometry already sits at the optimum and no autotuned override is retained.

An Xcode 27 Metal System Trace of the fixture recorded 76 labeled encoders, 76
submissions, and 76 target GPU intervals with all command buffers completing.
The synthetic trace is dominated by the deliberately large 32K exact-top-k and
long R4/R2/R1 parity cases, so it is label/scheduling evidence only; hotspot
decisions use uncaptured full-graph stage timings.

## Tensor-core prefill and parallel QSA selection

Prefill-sized batches no longer run the scalar per-row dot kernels. Four
changes moved the dominant stages onto tensor cores and parallel selection
networks; each keeps an environment kill switch and deterministic output.

1. Batched `Q8_0` dense projections (GDN qkv/z/output, QSA q/k/v/index/output,
   shared experts, hyper-connections, and the output head) dispatch the tiled
   `kernel_mul_mm_q8_0_f32` shared with the GLM/DeepSeek dense path once a
   graph call carries at least `DS4_QWEN4_Q8_0_MUL_MM_MIN_ROWS` rows (default
   32; zero restores the scalar kernels). The tiled kernel dequantizes into
   F16 threadgroup tiles and accumulates F32, matching the batched-kernel
   parity policy; `tests/test_qwen4_metal.c` covers aligned and bounds-checked
   batches against the CPU reference.
2. Routed experts for prefill-sized batches run mapped grouped matmuls on the
   llama.cpp-style 32-row expert work tiles (the same
   `kernel_mul_mm_id` machinery the DeepSeek routed path uses) once a call
   carries at least `DS4_QWEN4_MOE_MUL_MM_ID_MIN_ROWS` rows (default 512,
   matching the deterministic route-map tile; zero restores the scalar
   per-assignment expert kernels, and the Q4_K rows-8 weight-reuse path
   remains the fallback below the threshold). The deterministic route map,
   the SwiGLU epilogue arithmetic, and the exact slot-order weighted sum are
   preserved; `kernel_mul_mm_id_q4_0_f32` was added for the experimental
   Q4_0 routed profile and the Q4_K instantiation already existed. The work
   tiles cover the WHOLE routed batch by default
   (`DS4_QWEN4_MOE_MUL_MM_TILE_ROWS`, default one tile per chunk, 8192-row
   cap; 512 restores the original 512-row tiling): small tiles re-read
   every touched expert's weights once per tile — at 8192 rows each 512-row
   tile touches essentially all 512 experts, so the 1.4 GB per-layer weight
   set is streamed 16 times — and pad ~10-row per-expert populations up to
   the 32-row MMA blocks (~3.2x padded FLOPs). One full-chunk tile reads
   the weights once and amortizes the padding (~160 rows per expert = five
   full blocks). Row regrouping cannot change any output element's dot
   products, accumulation order, or destination, so the MoE outputs are
   byte-identical across tile settings — pinned by the extended
   `test_moe_q4_0_mul_mm_id` fixture (512-row tiling vs one 600-row ragged
   tile) and by byte-identical full-model greedy generations. Measured on
   the reference M3 Ultra (exp2 pack): the standalone stage drops from
   125.2 to 42.2 ms/layer at the uniform worst-case route (19.1 TFLOP/s
   effective, near the dense mul_mm rate), the in-situ GPU stage from 86.2
   to 47.2 ms/layer (4136 -> 2265 ms per 8K chunk), and full-model prefill
   gains 11-15 percent across all 16 ds4-bench frontiers (a back-to-back
   pair at 16K measured 760 -> 936 tok/s) with decode unchanged. An
   expert-sorted activation-staging variant (gather x/mid into
   expert-major order plus a contiguous-B `kernel_mul_mm_id` mode) was
   implemented, validated byte-identical, measured NET-NEGATIVE at
   full-chunk tiles (48.0 vs 42.2 ms/layer standalone — the staging passes
   cost more than the residual gather latency) and removed.
3. The single-query QSA ordered top-k uses a bitonic selection network
   (`kernel_qwen4_qsa_bitonic_topk_f32`) instead of the one-thread heap
   controller: 2048-entry chunks sort in threadgroup memory, each chunk
   forwards its top-k, and a final ordered pass writes the output and count.
   The comparator is exactly the heap's (score descending, index ascending),
   so the ordered result is bit-identical; the fixtures at 512 through 32768
   pooled rows pin this for every input type.
4. The multi-query streaming top-k (verifier rows and 512-query prefill
   microtiles) merges each tile into the running candidates with one
   threadgroup per query (`kernel_qwen4_qsa_bitonic_stream_f32`) instead of
   one heap-sifting thread per query per tile. Candidates stay ordered, so
   the final tile has already produced the ordered output.
   `DS4_QWEN4_QSA_BITONIC_TOPK=0` restores both heap controllers.
5. The gathered QSA attention (`kernel_qwen4_qsa_attention_bf16_f32` and its
   legacy twin) computes QK, the online softmax, and the value accumulation
   in one pass per simdgroup with 128-bit K/V loads and register value
   accumulators, replacing the threadgroup score sheet and the serial
   per-token output pass. The memoized and legacy instantiations remain
   bit-identical, and the padded-query invariance and CPU-reference fixtures
   still pass at 7.5e-9 max error. The structure follows oMLX's exact QSA
   attention (jundot/omlx PR #3244) and mlx-serve's `msv_attn_qsa256`.
6. The BF16 control-tensor projections — the MoE router (48 calls per
   chunk, K=2560 N=512) and the GDN decay/beta controls (72 calls, N=48) —
   dispatch the tiled `kernel_mul_mm_bf16_f32` (the mul_mm template over
   64-byte BF16 blocks dequantized to F16 threadgroup tiles, F32
   accumulation) once a call carries at least
   `DS4_QWEN4_BF16_MUL_MM_MIN_ROWS` rows (default 32; zero restores the
   scalar per-row `kernel_qwen4_bf16_matmul_f32`, which remains the
   authority for M=1 decode and verifier rows). The tiled path rounds the
   F32 activations and the BF16 weights to F16 tiles exactly once — the
   same operand-rounding policy as every other mul_mm variant — and
   `test_model_bf16_matmul_rows` pins it against a CPU reference that
   models that rounding (bit-exact at fixture magnitudes, where every
   product stays exactly representable in F32) plus the below-threshold
   scalar path against the exact reference. Measured on the quality
   profile: the stage drops 767-801 -> 66 ms per 8K chunk (0.55 ms per
   dispatch, distinct profiler label "Qwen BF16 tensor matmul"), warm
   full-model prefill gains 8.2-8.3 percent on 8K chunks and 5.6-7.4
   percent on the 4K-interval sweep, and the exact-checkpoint fixture
   drift stays at 0.0440 versus the 0.0444 baseline with the path forced
   onto the fixture's prefill batches (see the quality section).
7. The multi-query streaming top-k runs two bit-exact kernels that keep
   every ordered score, index, and count byte-identical to the original
   grid while removing the scan's long-context inefficiency (item 22h:
   the scan was 57 percent of a 256K-frontier chunk).
   `kernel_qwen4_qsa_score_tile_batch_bf16`
   (`DS4_QWEN4_QSA_SCORE_BATCHED=0` restores the original one-block-per-
   simdgroup scorer) loads the four query vectors into registers once per
   threadgroup and scans eight blocks per simdgroup, preserving each
   (query, block) dot's fma-chain order, simdgroup reduction, head
   max/sum, scaling, and masking — the original grid re-read the 2 KB of
   query vectors per four-block group and ran at a tenth of the scalar
   ALU floor. `kernel_qwen4_qsa_merge_select_f32`
   (`DS4_QWEN4_QSA_MERGE_SELECT=0` restores the full-sort bitonic merge)
   drops every tile entry worse than the running 512th-best (provably
   outside the final top-512), compacts the survivors, sorts only the
   adaptive-width survivor span, and bitonic-merges with the sorted
   running set; first tiles and oversized survivor sets fall back to the
   in-kernel legacy full sort. Both are pinned by fixtures
   (`test_qsa_streaming_topk`, `test_qsa_streaming_topk_merge_select`)
   against per-capacity scalar baselines and the CPU reference, and the
   full-model greedy dump at defaults is SHA-identical to the pre-change
   archive. Measured: the stage at 65536 visible blocks drops 963 -> 177
   ms/layer (scoring 734 -> 138, merging 253 -> ~60; the ablation knob
   `DS4_QWEN4_QSA_TOPK_ABLATE=score|merge` splits the label); short-
   frontier sweeps gain 5.9-6.9 percent, and the whole-258048-token
   frontier went 625.9 -> 937-938 tok/s with decode unchanged. A GLM
   `nax_direct_rhs` variant probed on the dense Q8_0 dispatch measured
   exactly neutral (the standard mul_mm is at its ceiling for tall
   batches) and was removed.
8. The QSA index scorer itself now runs on the tensor cores
   (`kernel_qwen4_qsa_score_tile_mm_bf16`, DEFAULT ON;
   `DS4_QWEN4_QSA_SCORE_MM=0` restores the scalar batched scorer, and
   `DS4_QWEN4_QSA_SCORE_MM_MIN_QUERIES`, default 16, keeps decode's
   M=1 path and MTP verifier rows on the scalar kernel). The four
   index heads of a query fold into the rows of one GEMM
   `A[queries*4, 128] @ B^T[128, tile_blocks]` over the shared
   pooled-key rows; F32 query vectors and BF16 pooled keys stage to
   F16 threadgroup tiles exactly once (the mul_mm operand-rounding
   policy) and accumulate F32 on the simdgroup matrix units, with the
   production max/sum/rsqrt head reduction and causal masking applied
   in a threadgroup-tile epilogue (fragment elements are never
   indexed directly). This path is DRIFT-GATED, not byte-exact.
   Occupancy is the lever at this short K: chunked K staging plus a
   C-tile overlay on the dead staging keeps the threadgroup at 16 KB
   — the 32 KB variants measured 1.7x slower with identical math.
   `test_qsa_streaming_topk_score_mm` pins it against a CPU reference
   that models the F16 rounding across partial M-tiles, ragged tail
   tiles, masked-capacity invariance, the threshold guard, and an
   observable-engagement check. Measured on the quality profile: the
   stage at 65536 visible blocks drops 200 -> 89 ms/layer (scorer
   134-154 -> 46-48 at ~11.5 TFLOP/s effective; the fresh-chunk ramp
   16.1 -> 13.6); full-model 4K-interval sweeps gain +0.2 to +1.6
   percent (growing with frontier) and the whole-258048-token
   frontier went 963.8 -> 1017.7 tok/s pair means (+5.6 percent, both
   interleaved pairs positive) with decode unchanged. Drift vs the
   exact-checkpoint fixture: +0.0000082 target MAE (0.044404 against
   the 0.044396 anchor; every agreement metric identical); the 16K
   logits delta is 0.083 mean / 0.66 max against the accepted
   0.074/0.55 chunk-size envelope. The 16K/128 greedy anchor at
   defaults is now
   `35ac916d9472f2db303f069ee31bdcca69021227e39a24a8b903b73e9dd2e8ea`;
   the pre-session `f6c2a929...` remains reachable with the kill
   switch and reproduces exactly.

9. The M5 TensorOps route now covers the Qwen prefill projections.  On
   M5-class devices — and, for fixture coverage, on any Metal 4 GPU via
   `DS4_METAL_ENABLE_TENSOR=1` (on pre-M5 silicon this selects the portable
   fallback: kernel-fixture parity, not a throughput path) — the dense Q8_0
   prefill dispatch takes the retained direct-RHS NAX kernels shared with
   the DeepSeek/GLM dense path (`kernel_mul_mm_q8_0_f32_nax_direct_rhs{,_n64,
   _n128}`) under the DeepSeek split-prefix contract: the widest
   128/64/32-token tile dividing the aligned rows, an unaligned >=192-row
   tail keeping the boundary-checked tiled kernel, and in_dim/out_dim
   multiples of 64.  Threshold/kill switch: `DS4_QWEN4_Q8_0_MPP_MIN_ROWS`
   (default 32, 0 disables).  The grouped routed-expert tiles additionally
   swap gate/up/down onto `kernel_mul_mm_id_q4_0_f32_mpp` /
   `kernel_mul_mm_id_q4_K_f32_mpp` at the identical route map, work tiles,
   and dispatch geometry (`DS4_QWEN4_MOE_MUL_MM_ID_MPP=0` restores the
   simdgroup id kernels; the v4 mixed-Q2 pack keeps its scalar expert
   kernels).

   Both routes also have F32-staged quality twins selected by
   `DS4_QWEN4_Q8_0_MPP_F32STAGE=1` and
   `DS4_QWEN4_MOE_MUL_MM_ID_F32STAGE=1` (the latter also honors the shared
   `DS4_METAL_MPP_MOE_F32STAGE`): the operand tiles stage as fp32 instead
   of binary16 — the same technique as the GLM/DeepSeek routed-MoE
   f32stage arms on exp/m5-tensor-precision (`kernel_mul_mm_id_*_mpp_
   f32stage`, measured ~2.5x tighter rms there; 16 KiB dense / 12 KiB MoE
   tile budgets here).  Measured on the M3 Ultra portable fallback vs the
   exact CPU dequant reference: the dense NAX f32stage route is BIT-EXACT
   (binary16-staged: 4.2e-3 max); the grouped tiles of the STANDARD
   Q4_0-routed pack drop to 6.0e-8 mid and output (from 3.4e-4 / 2.9e-4)
   and the v3 Q4_K tiles to 1.9e-6 mid / 6.1e-5 output (from 2.0e-3 /
   3.9e-2).  Like the exp branch's arms these are quality routes, not
   speed routes; the binary16 tiles remain the defaults.  M5 HARDWARE
   CORRECTION (2026-09-03): the quality claim transfers ONLY to the
   grouped MoE twins — measured on the real M5 Max tensor units at the
   fixture geometry (K=512), the grouped tiles hold the near-exact class
   (5.96e-8 Q4_0, 1.91e-6/6.1e-5 Q4_K — same as the fallback), but the
   DENSE direct-RHS f32stage twin does NOT: the real units compute the
   device-memory F32 operand at ~binary16-class precision no matter how
   the weight tile is staged, so F32 staging does not tighten the dense
   route there (kernel-level rel_rms 4.0e-4 binary16 vs 6.9e-4 f32stage
   at K=512, growing to 5.2e-4 / 7.7e-4 at K=8192; max_abs 0.024 vs
   0.035; the tiled simdgroup kernel measures 1.1-3.2e-4 — on real M5
   the dense f32stage twin is ~1.7x LOOSER than the default binary16
   route, and `DS4_METAL_MATH_SAFE` is bit-identical, so compiler
   fast-math is not the source; the mechanism is that the MoE twins hand
   matmul2d threadgroup-staged cooperative tensors, which the driver
   lowers to exact fp32 FMA, while the direct-RHS dense path reads the
   F32 activation tile straight from device memory through the tensor
   units' reduced-precision operand path).  The dense f32stage fixture is
   recalibrated to that measured envelope (5e-2 abs / 1.5e-2 rel); the
   dense quality-route claim now rests on nothing on M5 hardware, and
   end-to-end the twins do not restore parity: greedy anchors at a 50K
   prompt diverge at token 3 across binary16/f32stage/kill-switch with
   top-k logit rms 1.55-1.87 and max delta 6.2-8.3 (same class as the
   equivalence gate below).

   An earlier `nax_direct_rhs` probe on this same dispatch had measured
   exactly neutral and was removed — pre-M5 devices map matmul2d to
   portable fallbacks, so the route is now gated to the M5 TensorOps
   enable rather than the shared mul_mm threshold.  Fixtures:
   `test_model_q8_0_mpp_rows` pins the dense route (all three token tiles,
   the split-prefix tail, and the below-threshold/unaligned fallbacks) and
   `test_model_q8_0_mpp_f32stage_rows` pins the F32-staged twin at the
   measured M5 envelope (see the correction above); `test_moe_q4_0_mul_mm_id` (the standard Q4_0-routed geometry)
   and `test_moe_q4_k_mul_mm_id` (the v3 Q4_K geometry at 600 rows) each
   end with their own f32stage pass; the whole metal fixture suite runs
   with the opt-in so every route stays parity-pinned on every machine
   (`DS4_QWEN4_TEST_DISABLE_TENSOR_ROUTE=1`
   keeps a run legacy-only).  On the M3 Ultra the forced route passes every Qwen fixture,
   and `DS4_METAL_ENABLE_TENSOR=1 ./ds4_test --metal-tensor-equivalence`
   reproduces the accumulate drift locally on the portable fallback as
   well — short-prompt cases are bit-exact, while the long-prompt case
   drifts (long_memory_archive: rms 1.25, max_abs 6.5, greedy token flip),
   matching the `OUT/drift-check` finding that the matmul2d accumulate
   diverges from the reference kernels at long prefills.  The kernel
   fixtures pin tolerance classes, not bit parity; until the driver
   accumulate matches the reference kernels, quality-sensitive M5 runs
   should keep the kill switches above (and the drift PR's auto-withhold)
   in mind; the F32-staged twins are NOT a mitigation for the dense route
   on real M5 silicon (the MoE twins are).

MEASURED ON REAL M5 HARDWARE (M5 Max, 128 GiB, 2026-09-03, standard
Q4_0-routed pack + Q4_1 PLE; ds4-bench 8192-row chunks, 32768-token
interval frontiers on tripled promessi_sposi text, 64-token greedy decode
each; config A = defaults, B = the two kill switches above, C = both
f32stage envs): the TensorOps route is a decisive prefill win and
decode-neutral, and it is what brings this 128 GiB machine into the M3
Ultra's prefill class — the M5 Max's simdgroup baseline sits far below
the M3 Ultra's (B measured 611-737 tok/s vs the M3's 1069-1135 warm
band), so without the tensor units this machine would be strictly
slower at prefill.

```text
frontier            32K    64K    96K   128K   160K   192K   224K  262078
prefill A  tok/s   1236   1060    992    984    957    937    928    891
prefill B  tok/s    737    680    666    650    639    633    620    611
prefill C  tok/s    891    788    766    741    735    716    707    688
A gain vs B        +68%   +56%   +49%   +51%   +50%   +48%   +50%   +46%
decode  A  tok/s   38.7   38.7   38.6   36.7   37.0   36.6   36.1   35.2
decode  B  tok/s   38.6   38.8   38.8   36.9   37.6   37.3   36.5   36.4
```

Acceptance-harness cross-check on the standard corpus (config A):
cold-PLE first request 1105-1203 tok/s (cold fraction 1.0 verified),
prefill-10k median 925, prefill-50k 879, prefill-100k 854 tok/s;
decode-code-500 38.8 tok/s median.  The M3 Ultra's whole-258048-token
frontier record (1010-1025 tok/s) still leads the M5 Max tensor route
(891): the M5 wins at short frontiers (1236 at 32K) but its long-context
prefill decay is steeper — the QSA scan and gathered attention are the
context-growing stages and both are bandwidth-hungry.  Stage attribution
at the fresh-chunk geometry (8192 rows, 2048 visible blocks; M3 Ultra
reference in parentheses): routed MoE Q4_0 29.0 ms/layer on the real
units vs 62.9 on the simdgroup id kernels (M3: 40.7 tensor / 43.2
fallback — the real units pay ~29 percent over the M3's and the M5's
own simdgroup path is far slower), gathered attention 158 (M3 125.6),
streaming top-k 13.2 (12.5), GDN R4 12.1 (9.9), dense q8 12288x2560
11.0 (21.7).  F32STAGE wall-clock cost: -23 percent prefill at the top
frontier (891 -> 688 tok/s; the MoE stage alone 29.0 -> 65.8 ms/layer,
i.e. the fp32-staged tiles are 2.3x slower than binary16 MPP and even
lose to the simdgroup kernels) — given the precision correction above,
config C is neither cheap nor tighter on M5.  MTP: decode-neutral like
plain decode; on the deterministic counting continuation (engaged, 3.95
of 4 accepted per cycle) depth-4 MTP delivered 60.1 tok/s (B: 56.9)
against ~39 plain — while on novel prose the scheduler bypassed every
cycle (0 accepted, 24-32 ms bypass cycles) exactly as on the M3.  The
Metal-4 tensor banner ("Metal 4 tensor API enabled for Tensor kernels")
is a global capability line and prints in B as well; route engagement is
proven by the ~2x prefill separation and the stage numbers, not by the
banner.  `DS4_TEST_MODEL=gguf/GLM-5.3-Flash-Q2.gguf ./ds4_test
--metal-tensor-equivalence` on this machine reproduces the drift issue
exactly (worst_rms 1.38592, worst_max_abs 7.26952, long-prompt greedy
flips at step 0, ERR as expected until the driver accumulate fix), and
the acceptance corpus itself now trips stop-token contracts on this
pack (decode-prose greedy `<|im_end|>` at step 0 with a 1.13-logit
margin under BOTH A and B; the mtp-10k prose case stops naturally at
130/500 tokens, 173/500 plain) — a pack/corpus interaction to keep in
mind when comparing acceptance documents across machines.

Measured on the reference M3 Ultra with the experimental
`q4dense-q40routed-exp2` profile (2048/8192-token chunks, 64-token greedy
decode); the last row is the session-fifteen full-chunk MoE tiling measured
in the same session as its 720-878 tok/s baseline:

```text
prefill  2048: 154 -> 858 tok/s     decode ctx 2048:  15.2 -> 33.8 tok/s
prefill  8192: 171 -> 781 tok/s     decode ctx 8192:  11.8 -> 33.1 tok/s
prefill 32768:       737 tok/s      decode ctx 32768:  8.65 -> 31.7 tok/s
v3 pack prefill 2048: 75.7 -> 568 tok/s (decode unchanged at short context)
full-chunk MoE tiles: 16-frontier 720-878 -> 803-997 tok/s; 8K chunks
                      715-804 -> 823-944; decode unchanged; greedy output
                      byte-identical (kill switch: the TILE_ROWS env above)
```

Decode no longer collapses with context: the QSA per-layer stage measured
5.09 ms at ctx 8192 before the parallel selection work and 1.01 ms after it,
in line with GDN and routed MoE. MTP at a 8.7K-token prompt keeps the
scheduler engaged (zero bypasses); a full-accept depth-3 cycle is ~91 ms for
four target tokens and a 500-token completion including the 8.7K prefill
completes in ~25 s wall. (Session twenty later showed this engagement is a
property of that prompt construction, not of long context per se: on fresh
~8K novel-prose slices every depth 2-5 request on the standard pack — and depth 4 on
the v3 pack — was correctly bypassed; see the MTP section below.) Quality
gates: the 100-case short-prompt NLL
fixture is unchanged (the new paths engage only at batch sizes the fixture
never reaches), a 60-case 1272-token-prompt NLL A/B measured -0.1 percent
(noise), and 128-token greedy continuations remain fluent with divergence
only at rounding-scale argmax boundaries, the same class the 2K/8K chunk
comparison already accepts. Note the frontier-state comparator's 2e-5 logits
tolerance does not hold across chunk sizes on this branch even with the new
paths disabled (the pre-existing scalar-path delta is larger than the tiled
one), so that gate needs re-derivation rather than attribution to this work.


For full-graph chunk parity, run `ds4-bench` twice with identical text and
`--prefill-chunk 2048`/`8192`, adding `--dump-frontier-state-dir DIR` to each
run. Compare the resulting payloads with
`uv run --script speed-bench/qwen4_state_compare.py LEFT RIGHT`. The comparator
checks token and BF16 QSA/index caches exactly and applies the established
floating-point tolerances to logits, FP32 recurrence, PLE, and optional MTP
state while ignoring only the expected scratch-chunk header field.

After parity is established, add `--gen-tokens 128` and
`--dump-frontier-generation-dir DIR` to the 16K run. The emitted JSON records
the exact fixed-length greedy token IDs and byte pieces (EOS is deliberately
excluded by the benchmark), providing the deterministic 16K+128 golden without
depending on terminal text decoding.

No v3 generation golden is checked in yet. The former vector authenticated an
legacy v1 pack and was removed with that format. After reference parity is
established on a real v3 pack, record a new portable 16K+128 vector keyed by
the source revision, DS4 pack ID, tensor-manifest digest, prompt-file digest,
and 16K frontier rather than by a machine-specific model path.

`speed-bench/qwen4_acceptance.py` validates the final DS4 result JSON.
It enforces an idle Apple M3 Ultra/512 GiB declaration, one discarded warmup,
odd sample counts of at least three, unique prompts, `cached_tokens=0`, batch
size one, MTP disabled for the core cases, median aggregation, complete
latency/memory reporting, and the automatic 8K-versus-2K regression rule.

Generate the cache-distinct benchmark corpus with exact tokenizer-level context
lengths before running DS4. The generator gives every
measured request (and the discarded warmup) a different stable first token, so
the live session cannot reuse even a short common prefix. Cold-PLE and MTP use
dedicated prompts rather than warmed copies of the end-to-end inputs.

```sh
uv run --script speed-bench/qwen4_prompts.py \
  --tokenizer /path/to/official/tokenizer.json \
  --corpus speed-bench/promessi_sposi.txt \
  --out /path/to/qwen4-acceptance-prompts
```

Run the persistent DS4 samples with the resumable harness:

```sh
uv run --script speed-bench/qwen4_benchmark.py \
  --ds4-server ./ds4-server \
  --ds4-model /pack/Qwen3.8-Flash-Next-Q4KExperts-BF16Emb-BF16Control-Q8GDN-Q8QSA-Q8Shared-Q8Out.gguf \
  --ple /pack/Qwen3.8-Flash-Next-PLE-Q4_1.gguf \
  --prompt-dir /path/to/qwen4-acceptance-prompts \
  --output /path/to/qwen4-acceptance.json \
  --phase core \
  --phase chunks
```

Before issuing any warmup, the harness starts a fresh DS4 process and sends
exactly one dedicated 10K PLE request through the strict 8192
prefill path, followed by one generated token. After DS4 has computed all
160,000 n-gram row IDs (100 bytes per five-block `Q4_1` row, or 16,000,000 logical bytes),
but before its first PLE staging submission, it
plans the exact unique file pages touched by the selected contiguous `Q4_1`
row ranges and samples those pages with Darwin `mincore`. In the default
`pread` mode these remain the exact pages addressed by its sparse row reads;
the evidence retains all existing `F_NOCACHE` and runtime read-ahead fields.
It additionally records the gather mode and mapping-advice result so an mmap
A/B cannot be mislabeled as pread. Acceptance recomputes
the cold fraction from the page counts and requires at least 95%; a `cold`
label alone is not evidence. The harness resolves the containing mount with
`df -P`, binds it to `diskutil info -plist`, and requires a local block device
with `SolidState=true`; external SSDs are valid, rotational and network-backed
volumes are not. Darwin `proc_pid_rusage` V2 snapshots retain the process start
time and require at least 16,000,000 bytes of disk reads from that same process
lifetime. The residency probe duration is recorded separately: the result keeps
the raw observed latency/rate and a derived diagnostic with probe time removed.
The harness also reads DS4's internal prefill/decode timers, samples process RSS, requires the
exact raw token count and fixed generation length, rejects any nonzero cache
reuse, persists after every sample, and resumes completed prompt IDs. Add
`--phase mtp --ds4-mtp /pack/qwen3.8-flash-next-q4-mtp.gguf` to measure the
matching Qwen3.8 MTP sidecar. The MTP phase starts DS4 with
`--mtp-draft 4 --mtp-timing`; the core phase does not load an MTP sidecar. For every successful
DS4 MTP request the harness parses the request-local Qwen timing lines and
records total cycles, drafted and accepted draft tokens, target tokens, cycle
time, maximum per-cycle draft depth, and verifier modes. Acceptance requires
nonzero cycles, drafts, and accepts, never permits accepts to exceed drafts or
the configured depth four, observes the block verifier, proves
`target_tokens = cycles + accepted`, and requires that target-token sum to
equal the API's fixed 500-token completion.
The dedicated cold-evidence invocation must retain the same PLE
device/inode/size/times, manifest SHA-256, pack ID,
tensor-manifest digest, PLE SHA-256, and SSD descriptor for the complete phase.
The `chunks` phase restarts DS4 at explicit 2K, explicit 8K, and automatic
admission, records three cache-distinct, uncached-prefix 10K samples for each,
and writes the automatic selection plus the 8K/2K regression comparison into
the same result document. These are not described as disk-cold PLE samples;
only the DS4-only first-request phase carries that claim.

## Prefill stage attribution and the opt-in GQA MMA attention

Two attribution tools exist and they measure different things.

`DS4_QWEN4_PROFILE=1` prints one CPU-side submission-accounting line per
graph forward (embedding, PLE, dense, recurrence, QSA, routed experts,
cache updates, MTP, total). Those buckets time dispatch submission, not
GPU execution, and cannot attribute wall time.

`DS4_METAL_GPU_STAGE_PROFILE=1` (usually together with
`DS4_QWEN4_PROFILE=1`) is the GPU-timeline view: every public dispatch is
flushed as its own command buffer tagged with the dispatch label, and each
buffer's `GPUStartTime`/`GPUEndTime` interval is harvested at wait time
into a per-label table printed by `ds4_gpu_stage_profile_report()` at the
next Qwen profile line. The per-op flush cadence measured ~0% wall overhead
(803.2 vs 804.6 tok/s on the same 8K chunk) and the full Metal fixture
suite passes under this mode, so the table is honest attribution rather
than a perturbation. One caveat: a label identifies the dispatch family,
not the kernel — the tiled and scalar Q8_0 paths share a label, so confirm
kernel selection in the dispatch code before reading a label as a kernel
identity.

The corrected 8192-row-chunk budget on the reference M3 Ultra (exp2 pack,
GPU ~100% busy, wall ~10.2 s before the full-chunk MoE tiling below):

```text
routed Q4_0 grouped MoE matmul  86.2 ms/layer x48 = 4136 ms  (40%)
tiled Q8_0 dense family         ~2.9 s aggregate (three labels)
gathered QSA attention          110   ms/layer x12 = 1321 ms
scalar BF16 matmul family         7.5 ms/avg  x120 =  903 ms  (router + GDN decay/beta)
GDN R4 prefill recurrence       10.0  ms/layer x36 =  357 ms
QSA streaming top-k              21    ms/layer x12 =  256 ms
HC mix/norm/write family                            ~316 ms
```

The earlier "~4-5 s unaccounted" mass was the MoE grouped matmul (never
benchmarked before this) plus an undercounted dense family; there is no
gap, PLE-wait, or submission component of significant size at prefill.
The scalar BF16 family named below as the next candidate was landed in
session sixteen: the router and GDN decay/beta now run
`kernel_mul_mm_bf16_f32` at prefill batch sizes (767-801 -> 66 ms per 8K
chunk). The same attribution on the QUALITY profile
(the standard Q4_0-routed pack) measured its 8K chunk at 8.42 s GPU busy —
exp2-class, with the dense Q8_0 family at 2.90 s exactly matching what
exp2 spends on its Q4_K dense (the tiled kernel is compute-bound at
~22 TFLOP/s effective, so dense precision is not the profile's prefill
cost at 8K chunks) — and exposed that the profile's quoted 421-510
tok/s was a cold-PLE first-pass number (see the PLE section above).
Post-change warm budget on the quality profile: dense Q8_0 ~2.95 s (at
the mul_mm machine ceiling), MoE 2.27 s, QSA attention 1.43 s
(exhausted), GDN R4 0.35 s, streaming top-k 0.26 s, HC family 0.34 s,
BF16 controls 0.065 s; GPU ~100 percent busy at 7.96 s. The remaining
prefill levers are the long-context top-k scan (below) and a
categorically different dense kernel; the dense family and the MoE are
at their measured ceilings.

Long-context behavior (reference M3 Ultra, exp2 pack, 8K chunks,
frontiers 32K through 256K in 32K steps, 64-token decode each; the model
context cap is exactly 262,144 so ds4-bench needs `--ctx-alloc 262144`
with a top frontier of 262,078, and `DS4_BENCH_SNAPSHOT_MAX_BYTES=
unlimited` avoids inter-frontier prefix replays):

```text
frontier   32K    64K    96K    128K   160K   192K   224K   256K
prefill    889    562    500    480    463    436    410    387 tok/s
decode     31.3   30.4   30.1   29.4   29.4   28.9   28.9   29.1 tok/s
```

Decode does not collapse with context (first token 62-73 ms; ~29.5 KB of
live state per context token, 6.64 GiB at 224K). The prefill degradation
was attributed (stage profiler on the final 8192-row chunk at cache_pos
253,952, GPU ~100% busy): the QSA streaming top-k alone cost 11.48 s of
the 20.1 s chunk (57%) because its candidate scan is linear in visible
pooled blocks — roughly 45x the fresh-chunk ramp average — while the
gathered attention grows only +23% (full 2048-token budget per query) and
the MoE, dense, BF16, and GDN families are context-flat. THE LEVER HAS
SINCE BEEN LANDED TWICE: session seventeen's register-batched scorer
plus threshold-filtered merge-select are bit-exact and cut the
65536-visible-block stage 5.4x (963 -> 177 ms/layer; scoring 734 -> 138,
merging 253 -> ~60), taking the whole-258048-token frontier from 625.9
to 937-938 tok/s (+50 percent) with decode unchanged and short-frontier
sweeps +5.9-6.9 percent; session eighteen's tensor-core scorer (item 8
above) then took the scoring term to 46-48 ms/layer on the matrix units
(drift-gated), leaving the whole scan at ~89 ms/layer (~1.0-1.1 s per
last-chunk at the 258048 frontier, which reached 963.8 -> 1017.7 tok/s
pair means, +5.6 percent). The scan is still linear in visible blocks
at a better constant. The exp2 table above predates the MoE full-chunk
tiling, the BF16 tiling, and these kernels.

`make qsa-prefill-bench` (speed-bench/qsa_prefill_bench.c) drives the
production dispatch entry points at the exact prefill geometry (24 query
heads / 2 KV heads / head_dim 256, 512-block top-k, ratio 4, 512-query
microtiles, the routed Q4_0 expert family at 512 experts / 640 FF /
top-10, and a representative dense Q8_0 projection) so per-stage GPU times
can be compared against whole-chunk wall time without perturbing the
graph. `DS4_BENCH_MOE_ROUTE=uniform|block` selects the synthetic route
distribution for the MoE stage (uniform lattice = worst-case 32-row-block
padding; block = contiguous per-expert rows), and
`DS4_BENCH_QSA_VISIBLE_BLOCKS=N` overrides the streaming top-k geometry
from the fresh-chunk causal ramp to a long-context last-chunk shape
(every query seeing N pooled blocks) — the regime where the scan
dominates. On the reference M3 Ultra at
8192 rows with every query selecting the full 2048-token budget:

```text
routed MoE Q4_0          45 ms/layer   x48 =  2170 ms per 8K chunk (full-chunk tile)
gathered QSA attention   116 ms/layer  x12 =  1393 ms
QSA streaming top-k       30 ms/layer  x12 =   355 ms
GDN R4 prefill recurrence 10 ms/layer  x36 =   376 ms
dense Q8_0 12288x2560     24 ms              (per-projection)
```

At the long-context geometry (65536 visible blocks, all queries) the
streaming top-k measured 963 ms/layer split 734 scoring + 253 merging
(the ablation knob `DS4_QWEN4_QSA_TOPK_ABLATE=score|merge` isolates the
two), which the session-seventeen kernels cut to 177 ms/layer (scoring
138, merging ~60) and the session-eighteen tensor-core scorer to ~89
ms/layer (scoring 46-48, merging ~57).

The gathered attention is the largest ATTENTION-side kernel stage (the
routed MoE grouped matmul and the dense Q8_0 family are larger overall; see
the budget above): the
per-(query, head) grid re-reads the KV group's selected rows through L2
for every one of the 12 sibling heads, about 412 GB of effective L2
traffic per layer against a 16.8 MB cache footprint (3.6 TB/s effective —
the kernel runs at the L2 bandwidth limit, which is why it and the
f32-ALU floor agree within 2x).

The tree also carries an opt-in matrix-core rewrite,
`kernel_qwen4_qsa_attention_gqa_mma_f32`: one 32-thread simdgroup owns
eight query-head rows of a (query, KV head) pair, stages each 64-token
gather tile once, and runs QK^T and PV on the simdgroup matrix units with
F16 operands and F32 accumulation (Q rounds F32->F16; the BF16 KV cache
converts losslessly to F16 outside the subnormal tail — the
batched-kernel parity policy). The online softmax, lazy running-max
rescale, and sigmoid gate stay F32 through threadgroup tiles; the
rank-to-token mapping and masking are identical to the scalar kernel.
`tests/test_qwen4_metal.c` pins it against both the CPU reference and the
scalar kernel at 1.3e-05 max error (`test_qsa_attention_gqa_mma`).

Enable it for study with `DS4_QWEN4_QSA_GQA_MMA=1` (plus
`DS4_QWEN4_QSA_GQA_MMA_MIN_ROWS`, default 8; `DS4_QWEN4_QSA_MMA_DEBUG=1`
dumps per-tile softmax stats readable via
`ds4_gpu_qwen4_qsa_mma_debug_stats`). It is deliberately NOT the default.
The kernel is the ROLE-SPLIT organization validated by
`speed-bench/gqa_mma_proto.metal` (one 192-thread group per (query, KV
head): two score simdgroups run the full-width products and softmax while
four PV simdgroups each own an 8x128 output block, halving per-thread
accumulator registers so the matrix units can be fed at a higher rate).

SESSION NINETEEN CORRECTION AND CLOSURE (this is the current state of
the record; the numbers above from session fourteen described a kernel
that was not computing attention — see below).  Re-attributed on the
leaner standard-pack chunk, gathered attention is 1.46 s of a 7.98 s
GPU-busy 8K chunk (18 percent), and the A/B re-run measured a REAL
end-to-end win: 4K-32K sweep pair means +3.4 percent (16/16 frontiers
positive), the whole-258048-token frontier +4.3 to +9.0 percent,
standalone 131.2 -> 91.1 ms/layer (1.44x), decode unchanged.  But the
drift gate failed, and root-causing (a proto magnitude sweep plus
staging/score/fragment dumps through unused output rows) found that
the sfrag -> s_tile store loop was never guarded by `is_score`: the
four PV simdgroups' UNINITIALIZED fragments clobbered the score
groups' freshly stored products every tile, so the kernel had
computed exactly-uniform softmax since it landed — every fixture
passed because all of them validate at tiny magnitudes where uniform
weights approximate true weights, and single-token boundary cases are
immune by construction.  The store is now guarded (production kernel
and proto), and `test_qsa_attention_gqa_mma` gained a
production-magnitude pass (x12 input scale, score spreads ~+-20) that
negative-tests the bug.

The corrected kernel is numerically honest and STILL FASTER — and
still outside the quality class: exact-checkpoint fixture target MAE
0.0581 against the 0.0444 anchor (+0.0137, 3-7x the accepted
0.002-0.005 budget; first-token 85 -> 71), 16K logits 1.21 mean /
11.06 max against the accepted 0.074-0.083 / 0.55-0.66 envelope,
greedy divergence at token 2.  An opt-in F16 hi/lo Q-split
instantiation (`kernel_qwen4_qsa_attention_gqa_mma_qsplit_f32`,
selected by `DS4_QWEN4_QSA_GQA_MMA_QSPLIT=1` on top of `GQA_MMA=1`)
restores Q to ~22 mantissa bits and leaves the drift UNCHANGED
(0.0582 vs 0.0581; kernel-level mean drift 2.63e-4 vs the plain
variant's 2.64e-4) while costing the entire speed win standalone
(158.3 vs the scalar 131.2 ms/layer): the binding residual is the F16
P-tile rounding plus MMA accumulation order versus the scalar
kernel's F32 serial semantics, and fixing that would cost as much
again.  CONCLUSION: F16-operand gathered attention cannot hold this
model's quality class at a speed that beats the scalar kernel; the
scalar path remains the default, the MMA path stays opt-in for study,
and attention-side prefill work is closed on honest numbers.  The
prototype bench keeps the magnitude sweep (exact, F16-modeled, and
Q-split-modeled CPU references), the ablation floor (~41-47
ms/layer), and the concrete backlog (F16 S tile, pf hoisting,
one-pass softmax, double-buffered staging) for whoever returns with a
precision budget that admits F16 weights.

THE CLOSURE IS REOPENED AND LANDED (2026-09-03, M5 Max, Metal-4
TensorOps).  The M5 precision correction (item 9) supplied the lever
the F16 closure lacked: `matmul2d` with THREADGROUP-STAGED fp32
cooperative tensors lowers to exact fp32 FMA, so the P matrix — the
binding failure — can stay fp32 while the matmul machinery feeds the
Neural Accelerator units.  `kernel_qwen4_qsa_attention_gqa_t2d_f32`
(metal/qwen4.metal, DEFAULT ON wherever the Metal-4 tensor route is
enabled at prefill batch sizes; `DS4_QWEN4_QSA_GQA_T2D=0` restores the
scalar kernel and `DS4_QWEN4_QSA_GQA_T2D_MIN_ROWS`, default 32, keeps
decode and verifier rows on the scalar exact-arithmetic authority)
runs one 128-thread threadgroup per (query, KV head): the 12 query
heads (padded to M=16) share every gathered K/V tile, QK^T and PV run
as cooperative matmuls over threadgroup-staged tiles, and the 12x
gather-traffic cut of tile sharing finally pays.  PRECISION DESIGN
(constrained by three probe findings in speed-bench/t2d_probe.metal:
with a cooperative-tensor destination both input types must MATCH, so
a mixed fp32-P / half-V PV pair is inexpressible; the fp32-staged
matmul is exact at these shapes — rel 3.6e-7 vs double on M5 — while
half/half measures 3.3e-4; and the destination store wants extents
(N, M) with strides {1, N} for plain [m][n] row-major): the softmax
probabilities and both PV operands are fp32 — exact products, exact
accumulate — and only the QK operands stage as half: K converts
losslessly BF16->F16 and Q rounds F16 once per group, the residual the
Q-split probe already measured non-binding.  The all-fp32 twin
(gqa_t2d_exact in speed-bench/gqa_mma_proto.metal) differs from the
shipped split by 0.0e+00 to 6.8e-07 across fixture magnitudes while
costing 2.3x (213.9 vs 87.1 ms/layer standalone), so the split is the
production choice.  Structure notes: 32-token tiles, the Q tile
(16x256 half) and K tile (32x256 half, gathered with 128-bit uint4
loads — the scalar-ushort gather cost 68 ms/layer by itself) stay
resident with the QK contraction as ONE K=256 matmul2d, S and P share
one buffer element-for-element (the softmax overwrites each scaled
score by its probability in place, 8 threads per row with
simd_shuffle row reductions), the fp32 V chunks (16 KB, [dim][token])
overlay the dead K tile, the O accumulators live in cooperative-tensor
registers across the whole tile loop with the rare running-max rescale
applied in registers through `get_multidimensional_index`
(ids[0] = n = dim, ids[1] = m = head), and the threadgroup totals
27008 B against the 32768 B M5 limit.  MEASURED (M5 Max, standard
pack, qsa-prefill-bench 8192 rows): gathered attention 158.1 -> 89.6
ms/layer (x12 = 1897 -> 1075 ms per 8K chunk), with the F16-MMA kernel
at 121.5-123.4 on this machine for reference (quality-failed class).
Fixtures: `test_qsa_attention_gqa_t2d` pins the kernel against the
exact CPU reference and the scalar kernel at fixture magnitudes
(7.3e-06) and production magnitudes (score spread ~+-20: max 0.0265,
MEAN drift 7.7e-05 — the failed F16-MMA variant measured 3.08e-04 mean
under the same fixture — plus an observable-engagement check).
End-to-end quality on the exact-checkpoint fixture (100 cases,
score_official, ctx 4096): target MAE 0.037750 -> 0.037878 (+0.000127,
far inside the 0.002-0.005 accepted budget; the failed F16 route was
+0.0137), avg NLL +0.000284, top-1 rate -0.0004 (one rounding-scale
argmax flip), top-logprob coverage identical.  16K logits delta vs the
kill switch: whole-vocab mean 0.298 / max 2.15, top-20 rms 0.558 —
well inside the accepted A/B/C route family (top-k logit rms
1.55-1.87, max delta 6.2-8.3); the 16K+128 greedy anchor diverges at
token 1 on a narrative near-tie with both continuations fluent
(accepted-class divergence; the OFF anchor reproduces the prior
defaults).  Decode unchanged (route gated off below 32 queries; the
M=1 path is untouched).

End-to-end ds4-bench sweep (same recipe as the config A/B/C table:
8192-row chunks, 32768-token interval frontiers on tripled
promessi_sposi, 64-token greedy decode, back-to-back ON-then-OFF pair
under identical conditions; single pass per arm):

```text
frontier            32K    64K    96K   128K   160K   192K   224K  262078
prefill ON  tok/s   1321   1257   1206   1178   1127   1122   1088    1048
prefill OFF tok/s   1034   1005    980    941    927    919    909     877
ON gain             +28%   +25%   +23%   +25%   +22%   +22%   +20%    +20%
decode  ON  tok/s  39.1   38.8   38.5   37.4   36.5   36.6   36.3    35.3
decode  OFF tok/s  38.5   37.7   39.0   36.3   36.6   36.1   35.5    34.6
```

The whole-262078-token frontier at 1048 tok/s moves this M5 Max past
the M3 Ultra's 1010-1025 record band: the long-context deficit was the
gathered-attention term, not the scan.  Post-change stage attribution
at the fresh-chunk geometry (qsa-prefill-bench 8192 rows, defaults):
routed MoE 29.1, gathered attention 89.6 (was 158.1), streaming top-k
13.2, GDN R4 12.1, dense q8 11.0 ms/layer — the gathered attention
drops from the largest attention-side stage to roughly the dense
family's per-layer weight, and the remaining prefill budget is again
MoE-dominated.

Two probes used by this work are kept as inert scaffolds:
`speed-bench/sg_layout_probe.metal` and `speed-bench/sg_layout_probe.m`
print the per-thread element mapping of `simdgroup_matrix` (it is not
stable across compilations, so kernels must not index
`thread_elements()` directly).  The matmul2d reopen added a third:
`speed-bench/t2d_probe.metal` and `speed-bench/t2d_probe.m` pin the
cooperative-tensor operand/store layout conventions at the attention
shapes (LEFT [m][k] and RIGHT [n][k] k-contiguous, destination store
through (N, M) extents with strides {1, N}), the operand-type-match
static_assert, the exactness classes (fp32-staged exact, half/half
binary16-class), and the raw chunked-matmul rates — rebuild with
`cc -O2 -std=c99 -framework Foundation -framework Metal -o
speed-bench/t2d_probe speed-bench/t2d_probe.m` and run from the repo
root.

THE PRE-M5 COUNTERPART LANDED (2026-09-03, M3 Ultra, scalar route):
`kernel_qwen4_qsa_attention_gqa_share_f32` (+ `_legacy` twin) cuts the
gathered-attention L2 traffic fourfold on the NON-tensor route with
BITWISE-IDENTICAL outputs — the T2D stage win without tensor units, the
F16 rounding, or the drift budget.  The lever is organizational only:
the selected-block list and the rank -> simdgroup mapping (rank = sg;
rank += nsg) depend on the query, never the head, so the row a
simdgroup gathers is the same for all 12 sibling query heads.  The
kernel dispatches four 256-thread threadgroups per (query, KV head)
(group.z slices the twelve siblings into groups of three) and fuses
each group's rank loops: the simdgroup loads its gathered K/V row once
into registers, converts BF16 -> F32 once (exact, so every head
multiplies the identical operand values the per-head device-load path
produced), and runs each sibling head's dot -> simdgroup-reduce ->
online-softmax -> PV-FMA with the incumbent's lane/dim slicing, rank
order, and reduction order verbatim; each head's eight partials still
combine inside one threadgroup.  No threadgroup staging tiles, no
barriers beyond the incumbent's combine, no matmul2d, no simdgroup
matrices — which is the lesson of the discarded variants below.
Default ON at prefill batch sizes on the non-tensor route (the M5 T2D
route keeps precedence); `DS4_QWEN4_QSA_GQA_SHARE=0` restores the
per-(query, head) kernel and `DS4_QWEN4_QSA_GQA_SHARE_MIN_ROWS`
(default 32) keeps decode and MTP-verify rows (M=1..8) on today's
kernel shape.  NUMERICS CONTRACT: ideal class, achieved — the fixture
A/B against the kill switch is 0/245760 mismatched floats at fixture
magnitudes AND at production magnitudes (score spread ~+-12x12),
through both pipeline names (probability-cache on and legacy); the
CPU-reference max error is 1.12e-08, identical to the incumbent's.  A
below-min-rows arm pins the small-batch shape against the CPU
reference and a non-12-head geometry pins the gate exclusion (a broken
gate would dispatch the constexpr-12 kernel on an 8-head map and
diverge).  Because the kernel writes byte-identical `out` buffers, the
model's logits and greedy generations are unchanged by construction —
no drift re-measurement applies.

HPERG (heads fused per group) is pinned to THREE as the measured
occupancy optimum on the M3 Ultra, and the sweep is the interesting
part (qsa-prefill-bench 8192 rows, gathered-attention ms/layer):
incumbent 132.6; HPERG=2 82.9; HPERG=3 74.6; HPERG=4 82.2; HPERG=6
155.5.  Wider fusion amortizes more traffic but the register-resident
Q slices (HPERG x 8) and accumulators grow past what keeps two
threadgroups resident per core, and the occupancy loss dominates the
traffic win from HPERG=4 up.  Three discarded organizations are
recorded as negative results: (a) full 12-head single-group fusion
with everything in registers measures 802 ms/layer — the ~224-register
demand spills catastrophically; (b) cooperative threadgroup staging of
16-rank K/V tiles with a shared q block (the T2D kernel's
organization, scalar arithmetic) measures 154.4 — the staging is pure
overhead here because each gathered row is consumed by exactly ONE
simdgroup in the preserved mapping, so threadgroup memory never
broadcasts anything a register cannot hold (a z-split staging variant
with six heads per group measured 197); (c) HPERG=2 keeps the best
occupancy but pays double the conversions and half the amortization
(82.9).  Also re-measured this session on the M3: forcing the T2D
route via DS4_METAL_ENABLE_TENSOR=1 runs the gathered stage at 69.4
ms/layer (the portable fallback profits from the same traffic cut
even through matmul2d) but regresses dense 23.4 -> 27.1 and routed MoE
42.8 -> 45.6 ms/layer, so the opt-in pre-M5 gating stands unchanged.

MEASURED (M3 Ultra, standard Q4_0-routed pack + Q4_1 PLE, same-recipe
back-to-back sweep — 8192-row chunks, 32768-token interval frontiers
on tripled promessi_sposi, 64-token greedy decode, single pass per
arm; the fresh baseline below sat a few percent under the sessions
16-18 record band, so the pair is internally consistent but absolute
records should use identical-session comparisons):

```text
frontier            32K    64K    96K   128K   160K   192K   224K  262078
prefill OFF tok/s   1047   1018   1001    985    970    950    933    911
prefill ON  tok/s   1161   1135   1116   1038   1007    970    964    953
ON gain             +11%   +11%   +12%    +5%    +4%    +2%    +3%    +5%
decode  OFF tok/s   32.0   31.7   31.5   30.3   30.6   30.4   30.4   30.2
decode  ON  tok/s   32.2   31.9   31.9   30.7   30.6   30.5   30.3   29.8
```

Stage attribution at the fresh-chunk geometry (qsa-prefill-bench 8192
rows, defaults): gathered attention 132.6 -> 74.1 ms/layer (-44%; x12
per 8K chunk 1591 -> 889 ms), routed MoE 42.6, streaming top-k 13.4,
GDN R4 10.1, dense q8 23.2 unchanged.  The gain tapers at long
frontiers because the gathered stage's per-chunk work is constant
while the context-growing scan/paging stages take the larger share of
each chunk.  Decode is unchanged (route gated off below 32 queries;
the M=1 exact-arithmetic authority is untouched).  Fixtures:
`test_qsa_attention_gqa_share` (bitwise A/B at both magnitudes, both
pipeline names, small-batch gate, non-12-head exclusion); the GQA-T2D
fixture isolates its scalar reference arm from the new default with
`DS4_QWEN4_QSA_GQA_SHARE=0` so both organizations stay pinned on every
machine.

The experimental Apple Neural Engine runtime that this branch once
carried (`ds4_ane.m`, its transport kernels, diagnostics, and the
vendored oMLX license) was REMOVED by owner decision: it was never wired
into the model graph, and its measured integration results were negative
(0.23-0.39x at decode/verify shapes against the GPU alone; the only wins
were prefill-shaped standalone benches). The removal history and all
ANE measurements remain recorded in QWEN38_PERF_HANDOFF.md items 3, 9,
and 16-17.

### Session twenty-two: long-context decay attribution, GDN prefill width, dense-projection audit (M3 Ultra)

THE LONG-CONTEXT DECAY IS NOW FULLY ATTRIBUTED (2026-09-03, standard
Q4_0-routed pack, DS4_METAL_GPU_STAGE_PROFILE + DS4_QWEN4_PROFILE on the
full 32K-through-262078 sweep, 31 per-chunk GPU stage tables plus the
qsa-prefill-bench long-context model).  The profiled sweep reproduced the
decay (1218 -> 1025 tok/s, -15.8 percent; decode 27-29 under the flush
cost as documented) and the GPU stays ~100 percent busy at every
frontier: per-chunk GPU busy grows 6698 -> 8049 ms while chunk wall grows
by the same ~1.3 s, so there is NO hidden PLE, I/O, scheduling, or paging
term (the PLE SSD pread wait is flat ~196 ms/chunk and fully hidden
behind layer 0 at every context).  Growth decomposition of the +1351 ms
per 8192-row chunk, first chunk (cache_pos=0) versus last (245760):

```text
stage                                fresh    last    delta   share
QSA streaming score+bitonic top-k     73.5  1046.7    +973    72%   linear in visible blocks
gathered QSA attention               735.6   895.0    +159    12%   causal ramp -> full 2048
dense Q8_0 family                   2500.0  2600.9    +101     7.5% \  synchronized ~+4% step at
routed Q4_0 grouped matmul           2113.4  2195.8     +82     6%   /  cache ~147K, oscillating
GDN R4 + prep + remainder              379     391      +12     1%  /  (see below)
```

The gathered-attention term saturates by ~64K (every query selects its
full 2048-token budget; the bench's default geometry already models it).
The dense/MoE/GDN step is NOT context-structural: all three families sit
flat (2490/2110/346 ms) through cache_pos 139264, then step together at
~147K — roughly 2.5 minutes into sustained full-power load — and
oscillate afterward (a dip back to 2582 at 196608); eight back-to-back
fixed-work qsa-prefill-bench runs measure dead-flat dense/MoE (21.78-21.81
and 40.70-40.74 ms).  The signature — synchronized across unrelated
kernel families, uniform ~3.5-4 percent per dispatch, non-monotone —
matches the sustained-load clock envelope, not a capacity or paging
effect; recorded as machine condition, not a kernel target.

The scan, the one kernel-level term, was attacked a third time and found
at its floor for this organization.  At the 65536-visible-block geometry
(DS4_BENCH_QSA_VISIBLE_BLOCKS=65000) the stage measures 91.6 ms/layer
split 46.3 scoring + 43.7 merging (DS4_QWEN4_QSA_TOPK_ABLATE); the scorer
runs the simdgroup-matrix MM kernel at ~12 TFLOP/s effective (near the
wall).  Two byte-identical-by-construction merge variants were built,
measured NEUTRAL within the +/-0.8 ms run noise at both the long and
fresh geometries, and reverted: (a) the initial threadgroup load of
[512,2048) is dead in both the fast path (overwritten before read) and
the full-sort fallback (rebuilt from global before read) — cutting it
halves the merge's tile-score global reads but measured 43.7 -> 42.9;
(b) replacing the ten-stage bitonic merge tail (eleven threadgroup
barriers per query tile) with a rank-select scatter (binary-search merge
ranks against the other sorted span, one barrier, unique-key total order
so the top-512 is algorithm-independent) measured 43.0 vs 43.2
alternating A/B.  Conclusion: the merge-select kernel's cost is its
fixed per-dispatch work (the 512-candidate global round-trip per query
tile, the survivor filter, atomics, launch overhead), not its loads or
its sort network; the next lever is organizational (cross-query
candidate sharing or a hierarchical prefilter), out of bounded scope.

TWO gqa_share MICRO-OPTS MEASURED NEGATIVE AND REVERTED (the
kernel_qwen4_qsa_attention_gqa_share_f32 probes): (1) exp-skip — in the
online softmax, whichever arm loses the max comparison subtracts the
running max from itself, so its exp() is exactly 1.0f and skipping the
multiplies is bit-identical (fixture-confirmed 0/245760 at both
magnitudes) — but the simdgroup-uniform branch measured 70.0 -> 85.6
ms/layer (+22 percent): the kernel sits on the HPERG=3 occupancy cliff
and the branch costs more than the saved SFU work.  (2) An explicit
software pipeline prefetching the next valid rank's K/V uint4 pair —
91.4 ms/layer (+31 percent, register pressure past the same cliff) AND
not bit-identical under fast math (30154/245760 mismatched at 3.7e-9
max: the restructured loop contracts the dot/FMA chain differently).
The lesson mirrors the HPERG sweep: the kernel is L2-bandwidth-bound
with its loads already overlapped by the compiler; any added register or
control-flow pressure loses 20-30 percent.  Do not retry without new
organization-level evidence.

GDN PREFILL-WIDTH SWEEP LANDED (commit 1a790da): the BF16-state
recurrence's R was tuned at decode shapes, so the rows-per-thread /
threadgroup-height space was swept at the prefill geometry
(qsa-prefill-bench 8192 rows, DS4_BENCH_GDN_ROWS and the existing
DS4_QWEN4_GDN_SIMDGROUPS):

```text
ms/layer     simdgroups=1   =2    =4    =8
R4               9.82      9.95  9.05  11.06
R2              14.04     13.93  12.93  12.43
R1              21.58     19.63  18.13  17.60
```

R4 wins at every height — at prefill lengths the t-loop's k/q load
amortization over R rows dominates the extra parallelism of narrower
variants, inverting the decode-tuned intuition (the R1/R2 BF16-state
kernels stay in-tree for study, matching the fp32 precedent).  The one
win is threadgroup HEIGHT: four simdgroups per threadgroup at R4
measures 9.95 -> 9.05 ms/layer (-9 percent) at 8192 rows and 2.71 ->
2.49 at 2048, going neutral-to-worse at verify-sized rows (0.29 -> 0.30
at 16), so the dispatch now row-gates the default: n_tokens >= 1024
uses 4 simdgroups, smaller batches keep 2, and
DS4_QWEN4_GDN_SIMDGROUPS still overrides both.  The kernel has no
threadgroup memory or barriers, so the height change only regroups
threads and outputs are byte-identical; test_gdn_length gained a
BF16-state cross-R parity block at lengths {1,5,17,2048,8192} (outputs
tolerance-pinned at the fp32 R-sweep class — fast-math contraction
differs per R instantiation — and final states byte-pinned).

DENSE-PROJECTION AUDIT (commit 9eede07): the per-chunk inventory of
every dense Q8_0 dispatch at prefill — the "Qwen model Q8_0 projection"
family is exactly 504 command buffers per chunk: GDN qkv and z (36
each), QSA query+gate / k / v / index (12 each; query+gate already one
dispatch), shared gate / up / down (48 each), HC up 4x48 and HC down
48; already-fused families cover GDN out+HC-write, QSA output+HC-write,
the shared router+gated add, and the SwiGLU epilogue.  The audit's one
actionable finding is a routing hole, not a fusion pair: the PLE
key/value projections (the only BF16-activation Q8_0 consumers, 32
dispatches per chunk over the 512-token SSD staging tiles) dispatch the
scalar kernel_qwen4_q8_0_bf16 at every row count — one simdgroup per
output row per TOKEN, so every token re-reads the whole weight matrix
(~13 GB per 512-row key tile against ~0.4 GB for the 32-row tiled
grid).  The stage measured 263.4 ms of a 6.7 s chunk (3.9 percent GPU
busy).  The fix lands OPT-IN:
DS4_QWEN4_Q8_0_BF16_MUL_MM=1 (default off) widens the BF16 activations
with a lossless kernel_qwen4_bf16_widen_f32 and dispatches the shared
tiled kernel_mul_mm_q8_0_f32 (memory-barriered through the possibly
concurrent batch encoder; DS4_QWEN4_Q8_0_BF16_MUL_MM_MIN_ROWS, default
32, keeps decode and verifier rows on the scalar kernel either way).
MEASURED with the route on: 263.4 -> 26.5 ms per 8K chunk (10x, ~237
ms, ~+3.4 percent fresh-chunk prefill).  It ships opt-in because the
drift gate FAILED this branch's stated budget: 16K-prefill
whole-vocab logits delta versus the scalar authority 0.130 mean / 0.957
max against the 0.074-0.083 / 0.55-0.66 envelope (for scale, the
accepted T2D route measured 0.298/2.15; the rejected F16-MMA attention
1.21/11.1); top-1 argmax unchanged and greedy identical for 128 tokens
at 16K context, and with the flag unset the 16K logits dump equals the
scalar baseline BYTE FOR BYTE (pinned by re-run).  Fixture:
test_model_q8_0_bf16_mul_mm_rows (rows 64/33 x out 128/67 through the
tiled path at the F16-rounding tolerance class, rows 8 on the scalar
path, toggled via the env in one process).  Owner can flip the default
after weighing a 0.13-mean logit drift against +3.4 percent prefill.
The remaining same-input pairs are REPORTED ONLY — GDN qkv+z, QSA
k/v/index, shared gate+up, and the GDN decay/beta BF16 controls each
read the same [rows x 2560] activation through separate dispatches, and
fusing any of them needs a two-weight-range tiled kernel (a new matmul
variant; the tiles themselves are at the 22 TFLOP/s wall and are
explicitly out of scope); the avoided re-reads total ~12 GB per 8K
chunk, a ~1.5-2 percent ceiling against the compute-bound family.

Clean verification sweep at HEAD (same recipe, nothing else on the
GPU): prefill 1217 -> 1066 tok/s across the frontiers, decode 31.3-33.5
— at or above the 1161 -> 953 / 30-32 reference band on every frontier
(the tail margin is mostly session variance plus the GDN height win;
same-session stage comparisons above are the honest deltas).

## Official-reference quality baseline

100 continuations collected from OpenRouter `qwen/qwen3.8-flash`
(provider: Alibaba, thinking disabled, `top_logprobs` capped at 5 by the
provider — requesting 20 fails with HTTP 400) over the tracked
`prompts.jsonl`, 24 tokens per case, give the packs a fixed official
anchor. The owner confirmed the hosted `qwen3.8-flash` IS this
checkpoint (the API name drops the open-weights "Next" label; the served
1,000,000-token window is a serving-side extension of the 262,144-token
config). The collected responses expose no internal fingerprint, so the
last identity check available is local: the BF16 reference below should
agree with the API logprobs to ~0.01-0.05 MAE once it runs.

| metric (100 cases, 2241 target tokens) | v3 pack | all-Q4_K clone | exp2 pack |
|---|---|---|---|
| avg target NLL | 0.32587 | 0.33792 | **0.32175** |
| greedy first-token match | 49 | 54 | **62** |
| avg greedy LCP | 6.13 | 5.85 | **6.77** |
| API logprob MAE (target tokens) | 0.1733 | 0.1891 | **0.1707** |
| API top-1 agreement | 89.9% | 89.1% | 89.6% |
| API top-5 recall | 84.4% | 82.7% | 82.6% |
| pairwise order agreement | 88.9% | 87.9% | 88.0% |

The official model's own NLL on these same continuations (from the
collected logprobs) is 0.21220, so every pack sits ~0.11 nats above the
same-checkpoint reference self-entropy. Read as a same-checkpoint
anchor, that gap IS the pack's end-to-end quality cost (quantization
plus serving-precision and template-edge effects; perplexity multiplier
~1.12 on these continuations) — meaningful, not noise. The same-fixture
ORDERING is equally clean: exp2 (Q4_0 routed + Q4_K dense) fits the
official continuations best, the v3 profile (Q4_K routed + Q8 dense) is
1.3 percent behind, and the all-Q4_K clone (Q4_K routed + Q4_K dense) is
worst on every primary metric (NLL +5.0 percent over exp2, MAE 0.189).
The comparator puts exp2 ahead of v3 by 1.264 percent NLL (56-44-0 case
wins); the earlier same-tokenizer fixture had the sign reversed (+0.835
percent for v3), so pack ORDERING at this effect size is
fixture-dependent while the absolute ~0.11-nat cost is consistent.
Regenerate the
fixture with `collect_official.py --model qwen/qwen3.8-flash --endpoint
https://openrouter.ai/api/v1/chat/completions --api-key-env
OPENROUTER_API_KEY --count 100 --max-tokens 24 --top-logprobs 5
--token-limit-field max_tokens --thinking disabled`.

The exact-checkpoint BF16 anchor is prepared but not finished: the
transformers `qwen4_exp` checkout at `/tmp/transformers-qwen4` (repaired
from a sparse checkout that dropped `transformers/utils`; install via
PYTHONPATH shadowing because its wheel build is incomplete) reproduces
DS4's segmented chat tokenization exactly (60/60 token counts on
`/tmp/qwen4-nll-long`), but loading the 354 GB BF16 snapshot to MPS exits
silently at ~4 percent — a follow-up needs streamed/CPU-layer placement
before its NLLs are usable.

The exact-checkpoint BF16 anchor IS now established: 100 greedy
continuations (24 tokens, top-20 logprobs) generated from the local BF16
snapshot with transformers `qwen4_exp` on CPU (batched — see the handoff
for the generator and fidelity notes), stored in the same manifest
schema as the API fixtures so `score_official` produces full drift
columns against the true reference. All packs score the SAME tokens:

| drift vs local BF16 (100 cases) | v3 (Q4_K routed + Q8 dense) | **standard (Q4_0 routed + Q8 dense)** | all-Q4_K | exp2 (Q4_0 routed + Q4_K dense) |
|---|---|---|---|---|
| target logprob MAE | 0.0352 | **0.0444** | 0.0696 | 0.0737 |
| mean delta | -0.0094 | -0.0145 | -0.0305 | -0.0340 |
| greedy first-token match | **88** | 85 | 85 | 82 |
| greedy LCP (/24) | **15.81** | 14.92 | 13.61 | 12.51 |
| top-1 agreement | **97.4%** | 96.7% | 95.8% | 94.9% |
| top-5 recall | **93.6%** | 92.8% | 88.6% | 88.5% |
| pairwise order agreement | **94.0%** | 93.1% | 89.4% | 89.3% |
| decode tok/s (ds4-bench) | ~29 | **32-34** | — | 31-34 |
| prefill tok/s | v3-class (~570) | **1069-1135 warm; 1010-1025 at the 258K frontier (sessions 16-18)** | — | 770-1000 |

The standard profile was BUILT and validated: it is
an APFS clone of the v3 pack with the routed experts repacked in place to
Q4_0 (`speed-bench/qwen4_q4_0_routed_repack.py`, ~5 minutes; Q4_0 shares
Q4_K byte extents so only the clone's diverged routed blocks cost disk).
The loader detects the variant from the pack's actual routed tensor types
and dispatches the Q4_0 kernels — Q4_0-routed is a fully standard,
unconditionally accepted variant of the v3 profile (the historical
`DS4_QWEN4_EXPERIMENTAL_Q4_0_ROUTED` opt-in/kill-switch env was removed
when the profile was standardized; the dense GDN/QSA stay the pack's
original Q8_0 either way). Its drift landed exactly where
the decomposition predicted (0.044 vs the additive 0.039 estimate):
v3-class quality while keeping the routed decode win (32-34 tok/s).
CORRECTION (session sixteen): the profile's original 421-510 tok/s
prefill quote was a cold-PLE first-pass artifact, not the Q8 dense
family's GPU cost — at 8K chunks the profile's GPU budget is
exp2-class, and after the parallel PLE gather plus the tiled BF16
control matmuls it measures 923-1053 tok/s warm with cold first
requests matching warm (paired cold A/B 637 -> 1054). Sessions
seventeen and eighteen then landed the bit-exact scan kernels and the
drift-gated tensor-core scorer, lifting the warm band to 1069-1135
tok/s and the whole-258048-token frontier to 1010-1025; the standard
pack is the prefill leader among the local packs as well as the
quality profile.

Decomposing the same-fixture deltas: the dense GDN/QSA Q8→Q4_K swap
costs +0.0344 MAE while the routed Q4_K→Q4_0 swap costs +0.0041 — the
routed experts are nearly drift-free at Q4_0 and the experimental Q4_K
dense GDN/QSA is what separates exp2 from v3 quality-wise. The
data-recommended profile is therefore **Q8 GDN/QSA dense + Q4_0
routed**: v3-class drift with the routed speed wins; it is an APFS clone
of v3 plus the in-place `qwen4_q4_0_routed_repack.py` (equal byte
extents, no extra disk beyond the clone).

## MTP decode budget on the M3 Ultra

A `--mtp-draft 4 --mtp-timing` run on the v3 Q4 pack at a ~128-token context
measures one speculative cycle as: one single-row target pass (the sampled
input token, about 36.8 ms), the MTP-layer history pass for that row
(~4.9 ms), the recursive draft chain (~3.4 ms per drafted token), the
proposed-row verify pass (~57 ms for four rows), and the commit-side
reconcile that re-runs the MTP layer over the accepted rows and drafts the
next first token (~8 ms including the direct commit path). Accepted-length
histograms are bimodal — about 45 percent of cycles accept all four drafts
and about a third accept at most one — averaging roughly 3.65 committed
tokens per cycle and 25-26 generated tok/s.

A draft-depth sweep on identical prompts measured 26.49, 26.35, 25.17,
25.92, and 25.33 tok/s at depths 2 through 6: shallow drafts win on this
workload because the verify pass prices every drafted row while the accept
distribution saturates at four. The acceptance harness's depth four remains
the documented configuration.

### Session-twenty measurement (the standard Q4_0-routed pack's answer)

The depth and margin knobs were swept on the standard profile for the first
time with five
interleaved `ds4-server` instances (plain plus depths 2-5, default
scheduler, margin 110, 256-token greedy requests; engagement proven per
instance via the `MoE-down=Q4_0-expert-split` graph line). No source change
resulted — the answer is configuration-level.

Plain decode re-baselined at 35.3-35.5 tok/s short-context (34.6-36.0 band)
and 33.1 tok/s at ~8K context. One speculative cycle on this pack at short
context costs: base target pass ~26-27 ms, MTP history 3.3-4.1 ms, draft
~2.6 ms per token, verify 34.5/41.0/48.4/54.5/60.8/67.2/73.7 ms at depths
2-8, commit 3.3-4.8 ms. The verify pass is 37-41 percent cheaper than the
v3 pack's (48.4 vs 77.6 ms short, 82.4-82.5 ms at 8K, depth 4): the Q4_0
routed substitution speeds the verify MoE exactly as it speeds decode, and
unlike on v3/exp2 the depth optimum on profitable workloads is 5-7 rather
than 3-4.

The decisive finding: MTP draft acceptance is a property of the prompt
family, not the pack. On novel-prose continuation the draft matches the
target argmax at only ~50-60 percent per position at any context length —
forced-depth MTP measures 32.6/31.3/29.1/28.3 tok/s at depths 2-5 on
the standard pack (below plain at every depth), and the ORIGINAL v3 pack regresses
identically on the same slices (22.92 vs 28.28 tok/s at depth 4) while
winning identically on deterministic continuations (35.90 vs 28.34). The
historic acceptance numbers above (3.65 committed, and the harness's 2.65
accepted at 10K) are properties of those prompt constructions. On
closed-form continuations acceptance reaches 3.4-5.9 of 4-8 and MTP wins
monotonically in depth: with the default scheduler fully engaged (zero
bypass events, engagement ratio ~0.75), depth 4 delivers 50.2 tok/s, depth
5 delivers 54.2 tok/s, and depth 7 delivers 56.3 tok/s (57.8 best run)
against plain 35.5-35.9 — +42 to +57 percent, the fastest decode measured
on this machine for this model; factual-list prompts land at 44.7-48.5
tok/s (+26 to +37 percent, saturating at depth 5-6), while the acceptance
harness's own code-continuation family is bypassed like prose (accepted
0.33-0.5).

The depth boundary is measured, not assumed. On the deterministic
continuation the optimum is exactly 7 (d5 54.25, d6 54.51, d7 56.27, d8
55.11): acceptance at d7 is 5.737/7 (82 percent per position), so the
eighth row's ~6.6 ms verify plus ~2.6 ms draft has turned EV-negative even
on the best family. On factual lists d8 is worse still — the deeper cycles
pushed the cumulative ratio past break-even and the scheduler bypassed
every d8 request (34.7 tok/s, ~plain), the cost controller correctly
refusing the bet. The v3-pack depth-8 rejection therefore transfers to the standard
pack, but for the opposite reason: there acceptance collapsed, here
acceptance holds and the marginal verify row simply is not worth its
latency past 7. One counterintuitive bypass-side note: the window cost on
unprofitable prompts is NOT monotone in depth — the 1.5x severe-loss
check trips after 4 cycles at d7 versus 10 at d6 and 16 at d5, so deeper
drafts can cost less when bypassed (corpus: 34.55-35.00 across d5-d7 vs
plain 35.92).

The request-local scheduler arbitrates all of this correctly at the
shipped margin: every unprofitable request was switched to target-only
after its 16-cycle window at cumulative actual/baseline ratios 1.15-1.9
(several severe-loss trips after 1-5 cycles), bounding the cost of a wrong
bet at 0.4-0.7 tok/s short-context and 0.4-3.5 percent at ~8K context
(plain 33.06 vs MTP 32.10-32.94 at depths 2-5, all bypassed). Margin 110
is validated from both sides — profitable families sit at ~0.75, so any
margin in [100,120] decides identically, and margins high enough to force
engagement (>=150) would only lock in the measured 8-21 percent losses.

Recommended configuration for this profile: enable the sidecar with
`--mtp-draft 5` as the balanced default (within one to two percent of the
best on every profitable family and cheapest on capture memory), `7` when
the workload is known to be highly predictable (counting, pattern, or
structured continuations — worth +4 percent over d5 there), and never 8
(measured worse on both engaged families); leave
`DS4_QWEN4_MTP_SCHEDULER_MARGIN` at its 110 default — the upside on
predictable continuations is +26 to +57 percent and the worst case on
entropic prose is bounded below four percent. For
pure prose completion MTP never engaged on any measured slice of either
pack; do not expect a win there. The sweep driver is archived at
`speed-bench/qwen4_mtp_sweep.py` (interleaved server lifecycle, timing/
histogram/bypass parsers, engagement assertions); concurrent servers
require distinct `DS4_LOCK_FILE` overrides.

The round's secondary re-attribution (relative-only numbers; the per-stage
flush invalidates absolute throughput) confirms the stage picture on the
current binary: the M=1 window remains seven comparable latency-bound
stages — QSA 21 percent, MoE 19, GDN 18, MLP-HC-read 15, attention-HC-read
14, MLP-HC-write 11, attention-HC-write 1 (synchronized means 0.588/0.524/
0.510/0.426/0.390/0.300/0.033 ms per layer), with verifier rows scaling
sub-linearly (tokens=4 sums to 3.35 ms/layer; MoE 1.40x, GDN 1.36x, QSA
1.21x for 4x rows). The dense family still holds ~95 percent of CPU
submission time (25.0-25.5 of 26.3-26.9 ms per rows=1 forward). No new
kernel direction is justified by the re-attribution.

Stage profiling attributes the single-row pass roughly as: routed MoE about
29 percent, the two hyper-connection reads plus the MLP write about 45
percent, GDN or QSA mixing the remainder — all far above their
weight-traffic floors because M=1 projections are latency-bound. The
four-row verify costs about 1.5x the single-row pass (MoE scales about 2.1x
by reading up to 40 experts). Against approximate 750 GB/s effective
bandwidth the decode pass sits about 6x above its floor and the verify pass
about 2.9x. Reaching 60 tok/s with MTP therefore requires roughly a 2x
improvement across the target passes — better M=1 GEMV occupancy, MoE
verify batching, and/or concurrent ANE offload of the dense projection
family — plus trimming the ~23 ms per cycle of MTP-layer work. That MTP
work is structural rather than redundant: the chain self-conditions on its
own recurrent state (target hiddens for drafted rows do not exist yet),
while the commit reconcile deliberately re-runs the layer teacher-forced on
the verify pass's target hiddens to rebuild accurate state and draft the
next first token, so chain states cannot be reused for it.

The greedy CLI path now routes Qwen through the session executor in
`ds4_engine_generate_argmax`; the legacy raw greedy graph is
DeepSeek/GLM-shaped and crashed on the Qwen weight directory (a
pre-existing gap that predates this branch's working tree).

### Session twenty-one: the M5 Max decode/MTP attribution (no kernel landed)

The M5 Max's own decode/MTP cost picture was measured for the first time
(2026-09-03, standard Q4_0-routed pack, `speed-bench/qwen4_m5_mtp_probe.py`
on top of the archived session-twenty sweep module; counting/factual/prose
prompt families defined in the probe).  Measurement discipline matters on
this host: sequential requests within one server decay 5-15 percent
(first-to-third) and sequential server waves drift by up to ~10 tok/s, so
every throughput comparison below comes from either fresh-server
first requests or an interleaved round-robin protocol where every config's
k-th request sees the same machine state; six co-resident servers cost
plain decode ~2-3 tok/s and MTP cycles ~7 percent, so absolute and
interleaved numbers are kept separate.

The MTP cycle split at short context (~1.1K-token prompts), per engaged
cycle medians: base single-row target pass ~25.6 ms (plain decode 37.5-39.4
tok/s across the probe bracket), snapshot ~0.5 ms, draft chain ~2.7 ms per
drafted token (the chain already encodes all depths into one command
buffer with a single host sync), verify 31.3/39.1/46.3/53.8/56.7/69.5/77.5
ms at depths 2-8 (~7.7 ms per marginal row), history reconcile 3.3-6.0 ms,
commit 3.3-6.1 ms.  Versus the M3 Ultra the M5 is at parity on the base
pass and draft chain, slightly cheaper at shallow verify (31.3 vs 34.5 at
depth 2) and slightly dearer at deep verify (77.5 vs 73.7 at depth 8):
the depth optimum therefore moved — on deterministic counting the
fresh-server first-request throughput rises through depth 8 (66.8 / 70.1 /
73.1 / 73.4 tok/s at depths 5-8, acceptance 5.0/5.86/7.0/7.73), while
factual-list prompts saturate at depth 5-6 (~48-50 tok/s fresh, acceptance
~3.2) and prose never engages (125 bypass cycles per request at
25.5-26.5 ms — within ~1 ms of plain per cycle).  Margins 90/110/130
decide identically at depth 6 on every family; the 110 default stands.

Verify-pass GPU attribution (DS4_METAL_GPU_STAGE_PROFILE at depths 4 and
7, relative view — the per-op flush costs decode ~8-10 percent throughput
so these tables rank kernels, they do not price the wall): the dense Q8_0
projection family dominates and scales LINEARLY with rows (the
"Qwen model Q8_0 projection" label alone is 20.9 ms/forward at rows=4 and
46.8 at rows=7 versus 6.3-6.6 at rows=1 — every token row re-dequantizes
and re-reads the same weight blocks, including the 2560x~152K output head
once per row), the routed Q4_0 experts scale ~4.5x for 7x rows (already
bandwidth-bound per (row, expert) pair with only ~10-15 percent cross-row
expert collisions to exploit), and the BF16 control matmuls are flat in
rows.  The plain M=1 window attributes as dense-matvec family ~60 percent
(BF16 matmul ~6.2 ms/forward, Q8 projections ~6.3 + 3.2, HC fused family
~7.3), routed Q4_0 ~4.2, sparse QSA attention ~1.7 flat in context — and
the whole 38.7 -> 35.2 tok/s long-context falloff lives in the QSA
indexer scan: the M=1 block scoring kernel grows 0.19 -> 1.18 ms/forward
and the bitonic ordered top-k 1.71 -> 2.34 ms across 65K -> 262K context
while gathered attention and every other label stay flat.

Two kernel attacks on the verify pass were built and measured, and both
LOST — recorded here so the direction is not re-tried blindly.  (1)
Routing the 2..8-row verify projections through the existing
weight-sharing `kernel_qwen4_q8_0_f32_rows8` (via
`DS4_QWEN4_Q8_0_EXACT_MIN_ROWS=4`) regressed engaged depth-4 by ~7 percent
and depth-7 by ~5 percent: one SIMDgroup per output row serially advancing
eight token rows sacrifices too much thread-level parallelism at small M.
(2) A purpose-built micro-batch shared-tile kernel (one 256-thread
threadgroup per output row, eight SIMDgroups split-K over in_dim, the
reference per-slice Q8_0 arithmetic, per-token accumulators predicated
over compile-time row indices, fixture-pinned to <= 9.2e-5 max error and
byte-identical verify commits via `--mtp-verify-depth`
worst_argmax_gap=0.000) was 13-15 percent slower end-to-end and 5.9x
slower on the Q8 label under the stage profiler (46.8 -> 275.5 ms/forward
at rows=7) — the strided eight-token x reload pattern and per-token
accumulator pressure defeat the weight-traffic saving on this GPU
generation.  Both were reverted; the per-row reference kernels remain the
verify authority.  What the attribution says could still pay: a fused
single-dispatch QSA indexer scorer+selector (scoring plus the multi-level
bitonic currently cost ~3.5 ms/forward at 262K and both grow linearly —
the top-k runs full 2048-wide bitonic sorts at every reduction level with
single-threadgroup grids at the tail), and the output head's per-row
weight re-reads if a sharing shape can be found that keeps the per-row
kernels' occupancy.  The M=1 dense-matvec family (~60 percent of plain
decode, all far above their weight-traffic floors) remains the big plain-
decode target but is a latency-bound occupancy campaign, not a single
kernel swap.

The concrete configuration answer for the M5 Max mirrors the M3 Ultra's
shape with the optimum a notch shallower in cost terms: `--mtp-draft 7`
for known-deterministic continuations (~72-74 tok/s fresh-server,
+85 percent over plain 39; the interleaved matrix confirms d7 and d8 tied
at the top there — 61.3 vs 61.9 mean tok/s under six-server co-residency —
but d8 buys nothing over 7 on counting and is clearly worse on factual
lists), `--mtp-draft 5` for mixed predictable workloads (49.0-49.8 tok/s
factual interleaved, the family's peak, with 58-60 counting — within a few
percent of the deeper drafts), and leave the scheduler margin at 110.
The verify pass prices every row at ~7.7 ms and the M5's acceptance on
counting is near-perfect at every depth, which is exactly why deeper
drafts keep paying here longer than on the M3.

ds4-bench can now drive the same comparison end to end (`--mtp-draft N`,
joining `--mtp-model`; the speculative loop and per-frontier session
snapshots were already in place for DSpark).  Measured on the M5 Max with
the standard sweep shape (8 windows, 32K..262078, 64 greedy tokens each):
on a 300K-number deterministic counting corpus, depth 7 delivers
60.9/59.1/58.4/55.3/56.8/51.6/53.2/51.8 tok/s against plain
39.5/39.0/38.6/37.6/37.4/36.9/35.2/35.6 — a +46..+56 percent end-to-end
win that holds across the whole context band, with the MTP arm's
context falloff proportional to plain's.  On prose (tripled
promessi_sposi) depth 4 measures 30.4..26.9 against plain 34.5..32.3 —
a 7-16 percent LOSS at this window size: 64 greedy tokens is ~16
speculative cycles, exactly the scheduler's window, so the unprofitable
speculative phase is paid in full every window and the bypass decision
never arrives.  Short greedy windows over entropic text are the worst
case for speculative decode here; longer runs bypass after the window
and land within ~1 ms/token of plain.

One harness boundary measured while closing the session: `ds4-eval`
generates by sampling each token (default temperature 1.0) through the
one-token-at-a-time eval loop, so the greedy speculative executor never
runs there — passing `--mtp-model/--mtp-draft` to `ds4-eval` fires zero
speculative cycles and only adds the per-token drafter-history pass
(measured 39.3 -> 44.1 s wall on two questions, +12 percent, identical
outputs; zero `Qwen MTP timing` lines in the log).  ds4-eval now prints a
note when it sees those flags; speculative decoding is a `ds4-server
--mtp-draft` feature.

## Integration status

Pack conversion, trusted-local loading, single-base Metal mapping, adaptive
admission, SSD-overlapped PLE staging, chat/token identities, the end-to-end
48-layer native graph, optional Q4 MTP, and the native Qwen vision/M-RoPE path
are implemented. Qwen MTP recursively drafts up to 16 tokens on device and
verifies the proposed suffix with one target-model graph pass. During that pass,
verifier-only Metal kernels capture the active-profile Gated DeltaNet recurrent
state, FP32 convolution state, and FP32 PLE state after every row that can
become a partial commit. A full accept retains the final live target state. A
partial accept selects slot
`accepted - 1`, restores it entirely on device, truncates the logical QSA length,
selects the matching verifier-logits row, and reconciles only MTP history from
the already-produced target hidden rows. It does not replay the target graph.
Commit-time reconciliation runs the draft layer once over a widened
[saved trunk, accepted target rows] block instead of an M=1 pass plus a
second batched pass (`DS4_QWEN4_MTP_MERGED_HISTORY=0` restores the split
path); its outputs are identical and the four tiny-verifier fused kernels
(SiLU, SwiGLU, HC up/mix, HC write) carry explicit multi-row parity
fixtures alongside their row-1 cases.
Rejected QSA KV/index rows may remain physically present but cannot become
visible; subsequent writes, including ratio-4 pooled-index boundary rows,
recompute them before use.

A request-local cost controller compares cumulative speculative-cycle time with
ordinary target-row cost. It switches permanently to target-only decode when
MTP exceeds break-even, or immediately when the loss exceeds 1.5x; this also
stops MTP history work for the remainder of that request. Set
`DS4_QWEN4_MTP_SCHEDULER=0` for fixed-depth diagnostics, or change the normal
evidence window with `DS4_QWEN4_MTP_SCHEDULER_WINDOW` (default 16). The
controller switches permanently to target-only decode only after cumulative
speculative time exceeds the plain-decode baseline by a clear margin,
`DS4_QWEN4_MTP_SCHEDULER_MARGIN` (default 110 percent); the baseline prices
one plain token at the base-eval wall time and therefore understates the
true per-token decode cost, so an exact comparison disabled profitable
speculative cycles whenever decode-side kernels improved.

The normal BF16-GDN depth-4 capture allocation adds 179.93 MiB per MTP session;
the FP32/quality profile adds 341.93 MiB. Both are sized from the configured
verification depth, not the 8192-row work capacity. Set
`DS4_QWEN4_MTP_CAPTURE_DEPTH` to cap it, or
`DS4_QWEN4_MTP_CAPTURE=0` to retain the old snapshot/restore/replay fallback.
Allocation failure, excessive verifier depth, a failed invariant, or a backend
without capture support also falls back to replay; failure of both paths
invalidates the session. `DS4_QWEN4_MTP_FORCE_CAPTURE_FAILURE=1` is available
for fallback tests. Native batched-server mode, which already disables
speculative MTP, does not allocate these optional per-session verifier tensors.

The timing report distinguishes `direct-full`, `direct-partial`, and
`restore-replay`, reports restore/select and MTP-history time, and periodically
prints the accepted-length histogram and path counters. Capture is encoded in
the existing verifier command buffers: it performs no state readback, CPU state
copy, or per-row synchronization. Metal fixtures compare PLE, GDN convolution,
and GDN recurrent capture slots 1 through 16 byte-for-byte against replay in
both supported recurrent precisions. Host
and integration coverage includes EOS n-gram splits, ratio-4 QSA partial-commit
boundaries, the context-capacity boundary, accepted lengths 1 through 4,
consecutive partial commits, and forced fallback. A 256-token temperature-zero
run produced the same authoritative output SHA-256
(`798d13ff6723546962698744f2d4cd02e97f8f83bebc7b31ff22c03b364b5490`)
through direct capture and forced replay fallback, with zero replay fallbacks in
the supported direct run.

On the retained Q4 pack with Q8 hyper-connections, three warm 256-token runs
averaged 28.12 generated tok/s for ordinary decode, 22.07 tok/s for forced
restore/replay MTP, and 26.99 tok/s for replay-free MTP. The direct
path is about 22% faster than restore/replay, and direct partial selection plus
restore is roughly 3.2--5.2 ms instead of a 38--69 ms replay. Captured and
uncaptured verifier timings overlap (about 76--79 ms), so capture overhead is
not measurable above run noise. Replay-free MTP nevertheless remains about 4%
slower than ordinary decode on this profile; it is not an end-to-end decode win
yet. Immediate MTP-history reconciliation accounts for roughly 3.1--5.1 ms per
speculative block. Deferring it has a best-case ceiling near 27.8 tok/s in this
workload, still below ordinary decode, so the more stateful deferred path was
investigated but not enabled.

Switching replay-free MTP from FP32 to BF16 GDN state on three cache-distinct
128-token Q4 prompts changed decode from 21.14/24.84/30.10 tok/s to
22.74/24.43/31.86 tok/s, about 3.9% faster on average, while reducing capture
memory from 341.93 MiB to 179.93 MiB. Snapshot time fell from roughly 0.7 ms to
0.5 ms and a four-row verifier commonly saved 6--8 ms. The BF16 MTP profile
passed the four-case evaluator 4/4 and is the performance default. Block
verification and token-at-a-time BF16 decode are numerically equivalent but
not guaranteed bitwise-identical; a tested prompt crossed an argmax boundary,
consistent with the existing trust-the-target-block speculative policy.

On the 40.19 GiB v4 Q2 pack, a same-load three-run A/B measured 24.85 tok/s with
source BF16 hyper-connections and 33.63--33.64 tok/s with the derived Q8 profile,
a 35.4% improvement under unified-memory contention. Grouping decode layers in
eight-command-buffer batches measured about 33.81 tok/s versus 33.63 tok/s with
one layer per buffer. BF16 GDN recurrent state then measured about 34.15 tok/s,
roughly another 1%, and the combined profile passed the four-case deterministic
evaluator 4/4. A clean 128-token server smoke test after promotion measured
34.01 and 34.04 tok/s. Decode-only GDN fusion, which keeps convolution and the
BF16 recurrence in one encoder and folds Q/K normalization plus decay/beta
transforms into the recurrent kernel, measured about 34.40 tok/s versus 34.08
tok/s in a same-build warm A/B. It produced the same deterministic continuation
hashes, passed a direct Metal state/output parity fixture, and passed the
four-case deterministic evaluator 4/4, so it is enabled for ordinary BF16-state
decode. A second fusion makes one 512-thread group own each value head and
folds the trailing RMS normalization and sigmoid output gate into the same
recurrent kernel. A same-binary three-prompt A/B measured 28.03/28.02/27.97
tok/s versus 27.20/27.25/27.08 tok/s, about 3% faster, with identical greedy
continuations and exact fixture parity for output, BF16 recurrent state, and
convolution state. Set `DS4_QWEN4_GDN_FULL_DECODE_FUSION=0` to retain the split
output-normalization kernel, or `DS4_QWEN4_GDN_DECODE_FUSION=0` to disable the
decode fusion as a whole. These quantized
performance defaults can change greedy generation; `--quality` restores the
BF16-HC/FP32-state path and therefore does not use this fusion.

A full-model Metal System Trace showed the GPU busy for about 98% of the decode
window. Its whole-buffer labels assigned about 70% to the 36 GDN-layer command
buffers and about 28% to the 12 QSA-layer buffers; those percentages describe
layer mix, not isolated recurrence cost. Corrected stage-boundary timing gives
the comparable per-stage costs above. PLE projection/gate/conv work remains
below 1%, and PLE `mmap` versus `pread` was neutral. The decode limit is
therefore GPU kernel and weight traffic, not a CPU n-gram lookup stall.
Source-derived Q4_K substitutions were also rejected:
Q4 shared gate/up measured 31.53 tok/s versus 33.63 tok/s for Q8, while broader
dense profiles measured about 25 tok/s and changed greedy output. The repacker
is consequently restricted to the still-explicit GDN/QSA experiment. Encoding
all 194 hyper-connection matrices as Q4_0 reduced that derived profile from
0.63 to 0.33 GiB but measured only 25.54--25.81 tok/s versus about 27.7 tok/s
for Q8_0 and changed continuation hashes. Quantizing only the output head to
Q4_0 reduced it from 0.629 to 0.333 GiB, but measured 27.12/27.24/27.05 tok/s
versus 27.20/27.25/27.08 tok/s for Q8_0 and also changed greedy continuations.
It therefore offers no decode benefit on this GPU. Encoding
the shared and routed experts concurrently was also rejected: it measured
33.65 tok/s versus 34.04 tok/s for the serial path. The current Q2 result
remains well below the 60 tok/s target.

Kernel and host fixtures pass on the target M3 Ultra. The bounded, resumable
official BF16 converter validates the complete pinned Hub tensor directory
before any output allocation. Remaining acceptance work is producing and
validating the pack, full-model reference/golden runs, and the native DS4
throughput measurements. New conversion emits v3 standard GGML blocks. Legacy
v1/v2 packs are rejected, and v3 acceptance is not claimed until a real pack
passes inspection and regenerates the model-backed golden identity.

## Follow-up: 64 GiB Q2 profile

The Q4 fast pack is the correctness and performance baseline. After it passes,
the next branch should test a 2-bit backbone aimed at a 64 GiB MacBook while
leaving the 51B-parameter n-gram table on SSD. The PLE table is sparse at
runtime: each token hashes its two- and three-token history to 16 small rows,
and those rows enrich the representation at the second transformer layer. It
must therefore remain an on-demand sidecar rather than consume unified memory.

The decisive constraint for that profile is the same overlap implemented here:
submit the 16-row lookup before layer 0 and consume it at layer 1. The Q2 work
should first preserve Q4 token/logit parity tolerances and measure SSD stalls,
quality loss, resident memory, and first-token latency against this branch. It
should not trade away the established Q4 acceptance surface merely to reach the
64 GiB footprint.

The Q4 base is not the 64 GiB deliverable even when PLE stays on disk. A Q2
profile should follow the established DeepSeek/GLM recipe family—typically
imatrix-guided `IQ2_XXS` gate/up with `Q2_K` down, retaining higher precision
for shared, attention, output, and control paths. Its converter must compute
the exact emitted size and the runtime must prove the complete context/scratch
admission under memory pressure.

PLE itself contains 320,001,536 rows of 160 values (about 51.2 billion scalar
values) and occupies 29.8 GiB as standard `Q4_1`. Each row is five contiguous
20-byte blocks, so one sparse read obtains its codes, scale, and minimum without
touching distant tensor planes. The Q2 follow-up should keep this on-demand GGUF
sidecar unchanged unless a separately versioned PLE format proves both quality
and SSD-latency gains.
