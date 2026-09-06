# Branch DSpark Runtime and Tester Notes

[Speculative decoding](SPECULATIVE_DECODING.md) | [Environment variables](../ENVIRONMENT_VARIABLES.md)

These branch-specific notes retain the detailed rollout and A/B controls from
the former README. Experimental paths remain opt-in where stated. Run commands
from the repository root.

On a single accelerator, the main model can instead stream its routed experts
from SSD while the DSpark support model remains mapped or device-cached
separately. On Metal the support mapping is file-backed and pageable; CUDA and
ROCm prepare a separate device cache. Select the backend with `--metal`,
`--cuda`, or `--rocm`:

```sh
./ds4 -m ds4flash.gguf \
  --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --dspark --metal --ssd-streaming \
  --ssd-streaming-cache-experts 16 --temp 0
```

Use `--cuda` in a CUDA build. On ROCm, use `--rocm` and a verification-safe
cache, for example `--ssd-streaming-cache-experts 32`.

For memory-constrained Metal systems, use a small graph workspace as well as a
small expert cache. A practical 16 GiB starting point is:

```sh
./ds4 -m ds4flash.gguf \
  --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --dspark --metal --ssd-streaming \
  --ssd-streaming-cache-experts 16 \
  --ctx 4096 --prefill-chunk 128 --temp 0
```

When DSpark+SSD runs on a Mac with at most 24 GiB and neither
`--prefill-chunk` nor `DS4_METAL_PREFILL_CHUNK` is set, the runtime selects 128
automatically. Set `DS4_DSPARK_LOW_MEMORY_PREFILL_CHUNK=0` to retain the normal
workspace policy, or set it to another row count.

The Metal SSD verifier already supports the checkpoint's full five-draft
speculative block. With top-6 routing it needs at least 30 effective target
expert-cache slots; `--ssd-streaming-cache-experts 32` is the practical
five-draft setting. Smaller caches automatically limit the verifier to the
number of complete top-k rows that fit (a 16-expert cache normally selects two
rows). The Metal proposer follows that effective verifier/cache cap, avoiding
work on a suffix that cannot be consumed. Set
`DS4_METAL_DSPARK_PROPOSER_BLOCK_MAX=0` to restore the checkpoint's native
five-row proposer for an A/B control, or set a positive value to cap it
explicitly. Override the verifier policy independently with
`DS4_DSPARK_SSD_VERIFY_BLOCK_MAX=N`.

Metal can experimentally mirror the final target-hidden prefill row from the
HC weighted-sum kernel itself, avoiding a separate 16 KiB blit and
compute/blit encoder transition on each captured target layer. Enable it with
`DS4_METAL_ENABLE_DSPARK_CAPTURE_FUSED_LAST=1`; the historical
weighted-sum-plus-blit sequence remains the default because short M1 Pro SSD
A/B runs were bit-identical but did not show a repeatable throughput win.
`DS4_METAL_DISABLE_DSPARK_CAPTURE_FUSED_LAST=1` is the dominant kill switch.

An experimental single-device Metal verifier can use the current target
logits for the first draft and evaluate only the remaining `N-1` target rows.
Enable it with `DS4_METAL_DSPARK_ACCEPTANCE_ONLY_VERIFY=1`; it remains opt-in
because the smaller batch did not improve throughput on the measured M1 Pro
SSD path. Two-draft blocks retain the legacy verifier
because its one-row SSD routed-FFN path does not yet use the tiny-batch expert
table. A five-draft block starting exactly on a ratio-4 compressor boundary
also retains the legacy path so acceptance arithmetic does not switch to the
aligned compressor kernel. After verification rolls back,
the exact replay also skips the output head and logits readback for accepted
prefix tokens whose logits would be discarded; set
`DS4_METAL_DSPARK_HEADLESS_REPLAY=0` to restore the legacy replay. With
`DS4_DSPARK_STATS=1`, `metal_accept_only`, `metal_verify_rows_saved`, and
`metal_replay_headless` show how often these paths were exercised.

On very small unified-memory Macs, an additional diagnostic can keep only the
stage-0 `main_norm` and `main_proj` support tensors resident. Set
`DS4_METAL_DSPARK_PIN_MAIN_PROJ=1`; the 0731 support file locks about 51 MiB,
not the full 5.6 GiB GGUF. A failed lock is non-fatal and leaves the existing
pageable path active. Keep this opt-in until a same-machine A/B shows lower
`prop_setup`/generation time without reducing target-only throughput.

An experimental two-draft Metal SSD verifier can commit a full accept without
the normal rollback/replay pass:

```sh
DS4_METAL_DSPARK_EXACT2=1 DS4_DSPARK_STATS=1 \
./ds4 -m ds4flash.gguf \
  --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --dspark --metal --ssd-streaming \
  --ssd-streaming-cache-experts 16 --ctx 4096 --prefill-chunk 128 \
  --temp 0
```

It uses the canonical one-row decode kernels in layer order, restores and
replays token zero on a partial accept, and defaults both proposal and verify
width to two. Keep it opt-in until a long same-machine run is byte-identical,
has `exact2_attempt>0` and `exact2_fallback=0`, and improves throughput. The
generic Metal verifier can already evaluate five drafts together with a
32-expert cache, but its batch state is not numerically interchangeable with
ordinary decode and therefore still requires rollback plus exact replay.

`DS4_METAL_DSPARK_EXACTN_UNION=1` enables a separate experimental Metal SSD
verifier for two through five draft tokens. It executes canonical one-row
target decode in layer order, loads the union of the rows' routed experts once
per layer, and commits its verifier state directly after a full accept. On a
partial match it restores the frontier once, skips the boundary oracle,
exact-two path, and legacy token-by-token verifier, then exactly replays only
the already verified prefix. With `DS4_DSPARK_STATS=1`,
`exactn_union_partial_replay` and `exactn_union_verify_skip` should advance
together; `exactn_union_partial_replay_ms` isolates the required commit replay.
Set `DS4_DSPARK_FIXTURE_REQUIRE_METAL_EXACTN_PARTIAL=1` to require at least one
such partial match, equal replay/skip counts, no exact-union error fallback,
and byte-identical fixture output. The model-backed
`test-metal-exactn-oracle` is byte-identical to sequential decode for N=2..5,
including every N=5 partial prefix, EOS in the first or a middle row, serialized
KV/compressor state, logits, and a four-token continuation. Five drafts plus
the target token already available at the start of the cycle cover the
six-token speculative-cycle limit. Exact-union remains opt-in: correctness
does not imply a throughput improvement on a particular memory configuration.

For an independent Q8 output-head A/B inside exact-union, set
`DS4_METAL_DSPARK_EXACTN_BATCH_HEAD=1`. HC collapse and normalization remain on
the canonical one-row kernels, while one bit-exact decode-row dispatch projects
all two through five verifier rows to vocabulary logits. Non-Q8 output weights
are ineligible and a dispatch failure falls back to the ordinary per-row heads.
`DS4_METAL_DISABLE_DSPARK_EXACTN_BATCH_HEAD=1` is the unconditional kill switch
and wins if both variables are set. The
`metal_exactn_batch_head_attempt`, `metal_exactn_batch_head_use`, and
`metal_exactn_batch_head_fallback` counters identify the selected path; set
`DS4_DSPARK_FIXTURE_REQUIRE_METAL_EXACTN_BATCH_HEAD=1` to require a nonzero,
fallback-free use with byte-identical output.

The generic model-backed exact-N oracle keeps this Q8-only experiment disabled,
so it remains valid for target models with another output quantization. To add
model-backed batch-head coverage, use an OutQ8 target explicitly:

```sh
DS4_TEST_METAL_EXACTN_BATCH_HEAD=1 \
DS4_TEST_MODEL=/path/to/target-OutQ8.gguf \
make test-metal-exactn-oracle
```

The Metal proposer also has an independent experiment for confidence/Markov
synchronization overhead. Set `DS4_METAL_DSPARK_DEVICE_PROPOSER=1` when the
final confidence projection and both Markov matrices are Q8_0. On an eligible
single-device, tier-zero run it keeps the previous token, confidence decisions,
and Markov argmax chain on Metal, reuses the first confidence already computed
by the proposer, and returns one result for the complete draft block. Unlike
the CUDA experiment, the Metal path is eligible with SSD streaming; tensor
placement and proposal-quality mode remain excluded. It stops at the first
rejected confidence row and preserves the smaller-token argmax tie break.
Unsupported layouts, an incomplete result, or a CPU sigmoid-policy mismatch
fall back to the existing per-row implementation.

`DS4_METAL_DSPARK_NO_DEVICE_PROPOSER=1` is the unconditional kill switch;
`DS4_DSPARK_NO_GPU_MARKOV=1` and `DS4_DSPARK_NO_MARKOV=1` also keep the path
disabled. This remains opt-in because Q8 confidence accumulation moves from the
host CPU to Metal and therefore needs a same-machine greedy oracle and A/B.
With `DS4_DSPARK_STATS=1`, require
`metal_device_proposer_attempt == metal_device_proposer_use > 0`,
`metal_device_proposer_fallback=0`, and
`metal_device_proposer_policy_mismatch=0`. The acceptance fixture enforces
those conditions and byte-identical output with
`DS4_DSPARK_FIXTURE_REQUIRE_METAL_DEVICE_PROPOSER=1`.

Exact-union normally waits for every layer's routed-tail command buffer before
releasing its private expert-address scope. For an isolated A/B,
`DS4_METAL_DSPARK_EXACT_ROWS_ASYNC_TAILS=1` commits that tail without the CPU
wait, retains all scope resources until command-buffer completion, and lets the
next layer's router boundary provide the required ordering. The switch has no
effect outside exact-union and is also opt-in; unset it for the synchronous
control. Validate serialized state and greedy output as well as verifier time,
because removing a host wait is not by itself evidence of an end-to-end gain.

The AProjQ4 Metal decode path has several exact dispatch fusions relevant to
this verifier:

- HC RMSNorm plus the narrow F16 HC mixer is the M1-M4 default, including SSD
  split phases such as `TO_ROUTER`. Use
  `DS4_METAL_DISABLE_PRE_M5_HC_NORM_MIX_FUSE=1` for the reference control;
  `DS4_METAL_ENABLE_HC_NORM_MIX_FUSE=1` is the explicit non-default gate on
  other Apple generations.
- The Q4 Q-A/KV projections can share a dispatch with eligible F16 compressor
  projection/store work. It remains opt-in in both exact-union and ordinary
  `FULL` decode via `DS4_METAL_ENABLE_Q4_QKV_COMPRESSOR_FUSE=1`; the first M1
  Pro SSD A/B reduced dispatch count but did not improve verifier time. In
  either scope,
  `DS4_METAL_DISABLE_Q4_QKV_COMPRESSOR_FUSE=1` selects the existing fallback.
- The eligible Q4 attention-output B projection can perform the following HC
  expansion in the same dispatch. Use
  `DS4_METAL_DISABLE_Q4_ATTN_OUT_HC_FUSE=1` as its isolated A/B control.
- For an AProjQ4 multi-row attention-output batch with either attention B in
  Q8 or Q4_K, the opt-in
  `DS4_METAL_ENABLE_Q4_ATTN_OUT_TINY_BATCH=1` evaluates two through five rows
  in two dispatches while retaining the canonical one-row reduction order for
  every row. This covers the generic suffix verifier; the exact-union tape is
  intentionally still row-by-row and does not select this helper. Other
  output formats, unsupported shapes, a disabled Q4 classic matvec, or a
  dispatch setup failure return to the existing row-wise path.
  `DS4_METAL_DISABLE_Q4_ATTN_OUT_TINY_BATCH=1` is the unconditional kill
  switch and wins when both variables are set. For a fail-closed model-backed
  generic-verifier test, `DS4_METAL_REQUIRE_Q4_ATTN_OUT_TINY_BATCH=1` implies
  the enable gate for N=2..5 and turns an ineligible shape, the kill switch, or
  a dispatch failure into a hard error instead of a silent row-wise fallback.

For standalone Q4_K Q-b micro-batches (`1024 -> 32768`, 2–8 tokens), Apple
M1-family GPUs now reuse each lane's packed weights and scale/min metadata
across two tokens. The default retains the classic per-token arithmetic and
reduction tree. Set `DS4_METAL_DISABLE_Q4_QB_TOKEN_PAIR=1` to restore the
one-token-per-threadgroup path; any defined value, including `0` or empty,
disables it. One-token decode, Q-A/KV pairs, larger prefills, quality mode,
tensor parallelism and other GPU families keep their existing dispatches.
Local M1 Max kernel A/B measurements reduced Q-b time by roughly 20–28%; this
is not an end-to-end model or SSD throughput result. See
[the measurements and tester commands](../speed-bench/metal_q4_qb_token_pair.md).

The AProjQ8 Q-A/KV plus compressor compound remains the M1-M5 default for
eligible resident `FULL` decode. For an SSD-streaming A/B, including the
exact-union `TO_ROUTER` collection prefix, set
`DS4_METAL_ENABLE_Q8_QKV_COMPRESSOR_FUSE=1`. Ratio-4 layers combine the Q8
Q-A/KV pair with both attention and indexer F16 compressor pairs; ratio-128
layers combine it with the attention pair. The kernel preserves the canonical
NSG=4 Q8 and NR0=2 F16 reduction trees. A diagnostic Q8 NSG override or the
experimental NR0=4 compressor schedule therefore selects the separate
dispatches. `DS4_METAL_REQUIRE_Q8_QKV_COMPRESSOR_FUSE=1` turns such a fallback
into a visible error for model-backed tests. The existing ratio-specific
pre-M5/M5 QKV compound disable variables remain authoritative. Keep the SSD
extension opt-in until warm and cold A/B runs show a gain: it removes a launch
per row and layer but reads the same model bytes, and a compound grid can
change the order in which distant GGUF pages are faulted.

Additional PR #755 ports keep their established kernels as shape/resource
fallbacks:

- On Apple M1 through M5, an eligible one-row HC producer combines the F16
  RMSNorm/mixer, HC split and Sinkhorn-weighted sum, and destination RMSNorm in
  one compound dispatch for both attention and FFN producers. The global
  rollback is `DS4_METAL_DISABLE_HC_PRODUCER_PRE_NORM_FUSE=1`; the narrower
  controls are `DS4_METAL_DISABLE_PRE_M5_HC_PRODUCER_PRE_NORM_FUSE=1` and
  `DS4_METAL_DISABLE_M5_HC_PRODUCER_PRE_NORM_FUSE=1`. The existing
  `DS4_METAL_DISABLE_PRE_M5_DECODE_PORTS=1` umbrella also disables it before
  M5. `DS4_METAL_ENABLE_HC_PRODUCER_PRE_NORM_FUSE=1` permits a focused trial
  on another eligible Metal device.
- For an eligible ratio-4 layer on Apple M1 through M5, the standalone F16 compressor path can
  project the attention and indexer KV/gate pairs and append both recurrent
  states in one quad dispatch. It is the default in ordinary `FULL` decode and
  the exact-union collection prefix when the larger Q4 compound dispatch did
  not already store those states. Use
  `DS4_METAL_DISABLE_COMPRESSOR_QUAD_STORE=1` for the reference path;
  `DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_QUAD_STORE=1` is an additional
  compatibility rollback. `DS4_METAL_ENABLE_COMPRESSOR_QUAD_STORE=1` permits
  a focused trial on another Metal device and widens the phase scope for
  diagnostics.
- The exact ratio-4, one-compressed-row pool specialization is the M1-M5
  default for supported 128- and 512-element head shapes. Disable it globally
  with `DS4_METAL_DISABLE_COMPRESSOR_EXACT_POOL_RATIO4=1`, or use the
  pre-M5/M5 controls
  `DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_EXACT_POOL_RATIO4=1` and
  `DS4_METAL_DISABLE_M5_COMPRESSOR_EXACT_POOL_RATIO4=1`. For a diagnostic run,
  `DS4_METAL_REQUIRE_COMPRESSOR_EXACT_POOL_RATIO4=1` turns an unavailable
  exact dispatch into a visible failure instead of silently selecting the
  legacy reduction sequence.

Metal FlashAttention pipeline selection also keeps a generation-aware
one-entry host memo for hot specializations. This changes pipeline lookup, not
kernel arithmetic. Disable the pad/block memo with
`DS4_METAL_DISABLE_PRE_M5_FLASH_ATTN_PAD_BLK_MEMO=1` and the batched/vector
memo with `DS4_METAL_DISABLE_PRE_M5_FLASH_ATTN_BATCHED_MEMO=1` when isolating
host-side dispatch overhead.

The former 512-column streaming Metal top-k specialization has been removed;
its ordering was not deterministic for every input. The regular deterministic
top-k implementation is now used instead and has no runtime re-enable switch.
Use a previous binary only as a performance control, and require identical
selected ids on tie-heavy inputs before comparing timing.

Apple M1 defaults to a specialized SSD-streaming decode path for the exact
IQ2_XXS/Q2_K routed-MoE shape with 256 experts, top-6 routing, and a
4096-to-2048 gate/up projection. It replaces the IQ2 address-table pair-SwiGLU
producer, including complementary resident/missing cache masks.
It preserves the canonical dot-product,
reduction, clamp, activation, and route-weight order but writes `mid` directly
instead of materializing the otherwise unused gate/up rows. Every other
device, shape, streaming mode, unsupported mask/accumulate mode, or unavailable
pipeline keeps the canonical producer. Set
`DS4_METAL_DISABLE_M1_IQ2_MID_ONLY=1` to restore the canonical producer.
For fail-closed model coverage, `DS4_METAL_REQUIRE_M1_IQ2_MID_ONLY=1` rejects
an ineligible supported address-table dispatch; the kill switch still takes
precedence. The former `DS4_METAL_ENABLE_M1_IQ2_MID_ONLY=1` opt-in is accepted
as a harmless compatibility setting because the path is now automatic.
`make test-metal-iq2-midonly` compares all 12,288 top-6
output words bitwise at full shape for both unmasked and complementary masked
address tables, verifies that the candidates leave gate/up sentinels untouched,
and checks output guards. The routed-MoE stage profiler reports
`iq2_stream_addr_mid_only_4096x2048` or
`iq2_stream_addr_mask_mid_only_4096x2048` when the model path is actually
covered.

These gates change dispatch and intermediate-memory traffic, not model
arithmetic. Compare byte-identical output, exact-union counters, stage timings,
and generation rate on the same machine; do not infer a speedup from a lower
dispatch count alone.

CPU greedy decoding and the verifier's excluding-argmax scan use an unrolled
eight-lane implementation by default, including scalar tail handling and
first-index tie semantics. Set `DS4_CPU_DISABLE_UNROLLED_ARGMAX=1` to restore
the scalar scan for an isolated A/B. `tests/test_sampling` compares both paths,
including cross-lane ties, excluded ids, and non-multiple-of-eight vocabulary
sizes.

Exact file views for the two token embedding rows and repeatedly used Q8
support tensors are automatic; the compatibility kill switches are
`DS4_METAL_DISABLE_TOKEN_EMBED_EXACT_VIEW=1` and
`DS4_METAL_DISABLE_SUPPORT_Q8_DECODE_EXACT_VIEWS=1`.

DSpark attempts a proposal on every eligible cycle. Proposal cadence is not
adaptively throttled, so the reported acceptance rate covers the full runtime
sample. Quality and strict DSpark modes remain target-only.

Tune the expert-cache count for the available accelerator memory. ROCm needs
enough slots for a whole verification block (30 for the 0731 model; use at
least 32), and currently supports the IQ2_XXS/Q2_K or all-Q2_K routed-expert
layouts. CUDA uses a transient selected-expert cache for each target block.
The DSpark support weights are included in the startup memory budget even when
the Metal file-backed mapping remains pageable. This combination is
single-device only; CPU, distributed or
multi-GPU placement, tensor parallelism, and legacy MTP support models remain
incompatible with DSpark plus SSD streaming.

Resident single-GPU CUDA skips verifier captures that rollback/replay cannot
consume, batches frontier snapshot/restore copies behind one device fence,
computes the output head only for the final replayed token, pads the five-row
Q8 proposer head to the tensor-core shape, and fuses proposer Q RMSNorm with
RoPE. CUDA and ROCm also avoid the Metal-only mid-token submission split: on
those backends the same flush is a device-wide synchronization and only drains
the launch pipeline. The two kernel-selection kill switches for before/after
measurements are
`DS4_CUDA_DSPARK_NO_PADDED_HEAD=1` and
`DS4_CUDA_DSPARK_NO_Q_NORM_ROPE_FUSION=1`.

CUDA fuses HC split, weighted sum, and RMSNorm across multiple batch rows,
including the DSpark proposer and verifier; use
`DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED=1` for an A/B fallback to the separate
kernels. AProjQ4 CUDA decode can also share the activation quantization for
the Q-A/KV dense pair; `DS4_CUDA_DISABLE_Q4_DENSE_PAIR=1` selects the two
standalone projections. The canonical Q4 path submits to the decode stream so
these projections and the attention-output tail can participate in CUDA
decode graphs.

Single-token resident CUDA can also experiment with folding the canonical
Q8_1 activation emitted by the 4096-wide HC split plus RMSNorm stage into its
next MMVQ consumer. This remains opt-in pending GB10/DGX validation: set
`DS4_CUDA_ENABLE_Q8_FOLD=1`; `DS4_CUDA_NO_Q8_FOLD=1` is the dominant kill
switch. The sidecar is one-shot and keyed by model map, physical device,
stream, source pointer, and session epoch. Capture, scratch growth, a model-map
or device transition, and every lookup mismatch reject or invalidate it and
fall back to the established quantizer. For a diagnostic run, disable decode
graphs and add `DS4_CUDA_Q8_FOLD_ORACLE=1`; the oracle compares canonical Q8_1
bytes and the reached aligned-Q8 or IQ2 MoE consumer output, then always keeps
the freshly quantized reference. Require nonzero fold hits, `byte_calls`, and
`output_calls`, with zero mismatches and skips before considering promotion.
The experiment supports ds4's serialized, single-inference-host-thread CUDA
runtime only; embeddings that submit concurrently to the same CUDA stream
must leave it disabled.

On a single DGX Spark/GB10, the AProjQ4 path also mirrors the safe parts of
the aligned-Q8 decode work while retaining canonical Q4_K MMVQ/Q8_1
arithmetic:

- dense and paired Q4 projections reuse the persistent 1-MiB Q8_1 scratch;
  `DS4_CUDA_NO_Q4_DENSE_SCRATCH=1` restores pool allocation;
- attention-output A evaluates all output groups through one channel-grouped
  MMVQ dispatch per token, preserving the one-row reduction tree of every
  group. For DSpark verification widths 2--8,
  `DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_BATCH=1` flattens `(token, group)` into
  MMVQ channels and replaces the per-token loop with one grouped MMVQ
  dispatch while keeping `ncols_dst=1`;
  `DS4_CUDA_NO_Q4_GROUPED_ATTN_A_BATCH=1` restores the per-token grouped loop
  and `DS4_CUDA_NO_Q4_GROUPED_ATTN_A=1` restores the per-group loop;
  within eligible grouped MMVQ calls, a default specialization now derives the
  weight group from the grid index and sanitizes at store, removing the ids
  producer, output clear and separate sanitizer. Set
  `DS4_CUDA_DISABLE_Q4_GROUPED_MMVQ_FUSION=1` to restore those three operations
  while keeping grouped dispatch. Any defined value, including `0` or empty,
  disables this specialization. It does not enable multi-token batching or
  affect grouped prefill above eight tokens. GPU validation and throughput
  measurements are pending; see [the grouped MMVQ test notes](../speed-bench/cuda_q4_grouped_mmvq_fusion.md);
- for prefill widths above eight, the default eligible GB10 path quantizes the
  strided `[token][group][K]` input in one launch and writes each group directly
  into `[token][group][rank]`. It removes the eight F32 pack/unpack copies while
  keeping one established stream-K MMQ reduction per group. On the production
  `groups=8`, `K=4096`, `rank=1024` shape, a fixed-layout eight-warp Q8_1
  producer is also the default: each warp emits two canonical 128-value DS4
  records, reducing quantizer CTA count by four while preserving every output
  byte. `DS4_CUDA_NO_Q4_GROUPED_ATTN_A_Q81=1` restores only the generic
  strided Q8_1 producer, and
  `DS4_CUDA_REQUIRE_Q4_GROUPED_ATTN_A_Q81=1` fails closed if the specialized
  producer is not selected. Set
  `DS4_CUDA_NO_Q4_GROUPED_ATTN_A_PREFILL=1` for the dominant local rollback;
  exact `DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_PREFILL=0` is also a compatibility
  opt-out. Add `DS4_CUDA_REQUIRE_Q4_GROUPED_ATTN_A_PREFILL=1` for fail-closed
  tests. The separate single-grid/grid.z submission remains opt-in. The
  unrelated 16-warp MMQ experiment also remains opt-in and is not used by
  this producer;
- attention-output B keeps its canonical MMVQ result and the ordinary HC
  epilogue inside the graph-compatible fused call.  The row-packed epilogue
  remains oracle-only until a GB10 device run proves it bit-exact;
- the exact Q-b shape `32768x1024` has an experimental persistent-CTA kernel
  behind `DS4_CUDA_ENABLE_Q4_K1024_PERSISTENT=1`, with
  `DS4_CUDA_NO_Q4_K1024_PERSISTENT=1` taking precedence. Tests can add
  `DS4_CUDA_REQUIRE_Q4_K1024_PERSISTENT=1` to fail instead of silently using
  canonical MMVQ when the persistent dispatch is unavailable; this admission
  now fails before quantization, output clearing, or any kernel enqueue. Set
  `DS4_CUDA_Q4_K1024_PERSISTENT_STATS=1` for host-dispatch candidate/use/
  fallback counters. For a bitwise model-backed check, run with
  `DS4_CUDA_DECODE_GRAPHS=0 DS4_CUDA_Q4_K1024_PERSISTENT_ORACLE=1`; the oracle
  forces the candidate, compares it with canonical MMVQ, and always retains
  canonical output.

`DS4_CUDA_NO_Q4_GB10_FAST=1` is the umbrella rollback for these new GB10
choices; it does not disable the older cross-CUDA Q-A/KV pair itself. For a
fail-closed grouped attention comparison, set
`DS4_CUDA_ENABLE_Q4_GROUPED_ATTN_A_BATCH=1`,
`DS4_CUDA_REQUIRE_Q4_GROUPED_ATTN_A_BATCH=1`, and
`DS4_CUDA_Q4_GROUPED_ATTN_A_ORACLE=1`. The oracle computes the established per-group
MMVQ reference (or the established per-token grouped loop for a multi-token
candidate), reports aggregate calls/mismatches/skips plus separate
batch_candidates/batch_calls/batch_mismatches/batch_skips, and retains
canonical output. A valid multi-token test has nonzero candidates/calls and
zero batch mismatches/skips. The
oracle disables decode-graph capture. For a multi-token candidate, if it
encounters another active capture or cannot allocate comparison scratch, it
directly enqueues the canonical reference instead of consuming an unchecked
candidate. The scratch, grouped, and persistent paths are also covered by
`make test-mmq-parity-cuda CUDA_ARCH=sm_121`.

CUDA Q4_K MMQ performs its non-finite output guard in the final write-back
(or after the final stream-K fixup) instead of launching a separate
full-output sanitizer. Finite results and the per-group reduction order are
unchanged; the resident CUDA prefill harness checks them bit-for-bit against
the former pack/MMQ/unpack path.

Two additional CUDA fusions remain experimental until a device oracle passes
on the target GPU. `DS4_CUDA_ENABLE_HC_NORM_MIX_FUSE=1` combines HC RMSNorm
with the narrow F16 mixer when the selected standalone kernels have the same
reduction order; with the normal one-token cuBLAS path, also set
`DS4_CUDA_NO_F16_CUBLAS_ONE=1` to exercise it. The controls are
`DS4_CUDA_DISABLE_HC_NORM_MIX_FUSE=1` and
`DS4_CUDA_DISABLE_Q4_ATTN_OUT_HC_FUSE=1`. The Q4 attention-output B plus HC
path is automatic when MMQ is disabled, where the existing one-dispatch Q8_K
implementation is bit-compatible with its fallback. With the normal
MMVQ/Q8_1 decode path, the GB10 graph-compatible call preserves both the
canonical MMVQ projection and the ordinary HC expansion. The specialized
row-packed epilogue is evaluated only by the oracle below and is never
consumed by normal decoding. The older, truly single-dispatch Q8_K experiment
is isolated behind
`DS4_CUDA_Q4_ATTN_OUT_HC_Q8K_EXPERIMENT=1` and may differ numerically from
MMVQ/Q8_1.

For a fail-closed hardware comparison, set
`DS4_CUDA_Q4_ATTN_OUT_HC_ORACLE=1`. It retains the canonical MMVQ/Q8_1 output,
compares both the row-packed epilogue and the Q8_K compound bit-for-bit, and
prints `epilogue_mismatches`, `q8k_mismatches`, and `skips` at exit. The
oracle avoids readback while CUDA graph capture is active; run a separate
non-captured diagnostic and require `calls>0`, `skips=0`, and
`epilogue_mismatches=0` before promoting the row-packed path back into normal
decoding. A zero-call summary is therefore an explicit failed coverage gate,
not a silent pass.

An experimental resident-CUDA path can run the existing aligned
IQ2_XXS/Q2_K vector MoE kernels for two-to-five-draft routed batches,
preserving the established fused-SoA path as an automatic fallback:

```sh
DS4_CUDA_DSPARK_TINY_ALIGNED_VEC=1 DS4_DSPARK_STATS=1 \
./ds4 --cuda -m ds4flash.gguf \
  --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --dspark --temp 0 -p 'Write a Python quicksort function with comments.'
```

Keep the aligned tiny-batch path opt-in until the same-machine acceptance,
decode-consistency, and throughput comparisons pass on CUDA hardware.

An experimental exact two-token resident-CUDA verifier is available for a
DGX Spark A/B test. It uses the ordinary decode kernels, commits a two-token
full accept without rollback/replay, and replays only the first token on a
partial accept. By default the switch runs both the proposer and verifier at
width two, instead of evaluating the checkpoint's native five-row proposal
when only two drafts can be consumed:

```sh
DS4_CUDA_DSPARK_EXACT2=1 DS4_DSPARK_STATS=1 \
./ds4 --cuda -m ds4flash.gguf \
  --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --dspark --temp 0 -p 'Write a Python quicksort function with comments.'
```

The support model uses non-causal attention across the proposal block, so a
two-row proposal is not guaranteed to be a prefix-identical version of its
native five-row proposal. The target verifier still protects the emitted
greedy continuation. For isolated A/B tests,
`DS4_CUDA_DSPARK_PROPOSER_BLOCK_MAX=0` preserves the native proposer and an
explicit value such as `2` caps it independently of exact-2.

CUDA also has an opt-in tiled online-softmax kernel for this non-causal support
attention. It shares each raw KV row across a group of attention heads and is
selected only for the DSpark raw-ring/head geometry it supports; every other
shape keeps the reference kernel. Enable it with
`DS4_CUDA_ENABLE_DSPARK_NONCAUSAL_ONLINE=1`. The emergency control
`DS4_CUDA_DISABLE_DSPARK_NONCAUSAL_ONLINE=1` wins when both variables are set.
The online reduction order can change draft floating-point results even though
the target verifier still protects greedy output. Use
`DS4_DSPARK_VERIFY_NONCAUSAL=1` to print the first three comparisons against a
host double-precision reference, and require the final target continuation to
remain byte-identical in the performance A/B.

Keep this path opt-in until the CUDA acceptance fixture is byte-identical and
the same-machine statistics show lower `propose`, `verify` plus `replay` time.
The stats line reports `prop_capped`, `prop_scheduled_rows`, `exact2_attempt`,
`exact2_full`, `exact2_partial`, and `exact2_fallback`; a valid run must
exercise exact-2 and leave its fallback counter at zero.

A separate resident exact-N CUDA experiment extends the same canonical
one-token tape to two through five draft rows. It leaves hidden rows and all
target weights on one GPU, submits the per-row ordinary decode kernels in one
stream, and reads back only the `N-1` acceptance ids plus final logits. A full
match therefore commits its already-exact KV/compressor state without replay;
a partial match restores the pre-cycle frontier once and replays the prefix
already proven by exact-N, without running the legacy verifier a second time.
Only a backend error retains the legacy verifier/replay fallback. Enable it
independently with:

```sh
DS4_CUDA_DSPARK_EXACTN=1 DS4_DSPARK_STATS=1 \
./ds4 --cuda -m ds4flash.gguf \
  --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
  --dspark --temp 0 -p 'Write a Python quicksort function with comments.'
```

The default verifier cap under this gate is five (or the available prefill
workspace when smaller). The proposer uses at most one fewer workspace row,
because its support stage also carries the current target row; the existing
explicit proposer and verifier cap variables still take precedence.
`DS4_CUDA_DISABLE_DSPARK_EXACTN=1` is the
kill switch and restores the previous path without changing the enable
variable. Track `cuda_exactn_attempt`, `cuda_exactn_full`,
`cuda_exactn_fallback`, its partial/error split, and `cuda_exactn_rows`. Keep
this experiment disabled by default until a real CUDA oracle and a long greedy
A/B show byte-identical output, no fallback errors, and a throughput win.

The layer tape can separately reuse its position-independent CUDA decode
islands with `DS4_CUDA_DSPARK_EXACTN_GRAPHS=1`. Graph keys use the stable
device address (including each batch-row offset), not the short-lived tensor
view wrapper. Four cache entries per layer/island remain reserved for ordinary
decode and five more are isolated for exact-N rows, so a width-five verifier
cannot evict the normal decode keys. The first encounter warms lazy allocators,
the second captures/instantiates, and only later encounters are pure replay;
benchmark at least 128--256 generated tokens rather than judging a short
capture-heavy run. `DS4_CUDA_DISABLE_DSPARK_EXACTN_GRAPHS=1` is the dedicated
kill switch, while `DS4_CUDA_DECODE_GRAPHS=0` still disables all decode graphs.
The stats fields `cuda_exactn_graph_attempt`, `..._use`, `..._warm`,
`..._capture`, `..._replay`, `..._no_slot`, and `..._failure` expose warmup,
reuse, capacity misses, and retired captures. This first rollout is for
serialized, single-session DGX testing; the graph cache and cuBLAS capture
state remain process-global. The fixture can require a clean post-warmup
replay (including zero no-slot/failure events) with
`DS4_DSPARK_FIXTURE_REQUIRE_CUDA_EXACTN_GRAPHS=1`.

For a separate output-head A/B, set
`DS4_CUDA_DSPARK_EXACTN_BATCH_HEAD=1`. The experiment keeps HC collapse and
normalization on the canonical one-row kernels, then runs the Q8 vocabulary
projection for all exact-N rows through the bit-exact decode-row kernel. It is
automatically ineligible for non-Q8 output weights and falls back to the
ordinary per-row heads on a dispatch failure. The emergency kill switch is
`DS4_CUDA_DISABLE_DSPARK_EXACTN_BATCH_HEAD=1` and wins when both variables are
present.

The CUDA proposer tail has a second, independent experiment for the fixed
confidence/Markov synchronization overhead:

```sh
DS4_CUDA_DSPARK_DEVICE_PROPOSER=1
```

When the final confidence projection and both Markov matrices are Q8_0, this
keeps the previous token, confidence decisions, and all Markov argmax steps on
the decode stream and reads one 64-byte result for the whole draft block.  It
stops at the first rejected confidence row, preserves the Markov smaller-token
tie break, and rechecks the returned confidence prefix with the established
CPU sigmoid policy.  Unsupported layouts, an incomplete result, or a policy
mismatch fall back to the per-row implementation.  The unconditional kill
switch is `DS4_CUDA_DSPARK_NO_DEVICE_PROPOSER=1`; the older
`DS4_DSPARK_NO_GPU_MARKOV` switch also keeps this path disabled.
The initial gate is intentionally limited to resident, single-GPU, non-quality
CUDA and reuses the already-computed first confidence value.

This remains opt-in because the Q8 confidence accumulation moves from the host
CPU to CUDA and must pass the DGX proposal/acceptance oracle before promotion.
With `DS4_DSPARK_STATS=1`, require
`cuda_device_proposer_attempt == cuda_device_proposer_use > 0`,
`cuda_device_proposer_fallback=0`, and
`cuda_device_proposer_policy_mismatch=0`.  The acceptance fixture can enforce
those conditions with
`DS4_DSPARK_FIXTURE_REQUIRE_CUDA_DEVICE_PROPOSER=1`.

The stats line separates `cuda_exactn_ms` into setup, layer, head, and read
components, and reports restore, legacy-error-fallback verification, and
partial replay time independently. `cuda_exactn_partial_replay` and
`cuda_exactn_verify_skip` should advance together on valid partial matches;
the batch-head attempt/use/fallback counters make its dispatch unambiguous.

Set `DS4_DSPARK_FIXTURE_REQUIRE_CUDA_EXACTN=1` on the candidate acceptance
fixture to require at least one `cuda_exactn_attempt` and zero
`cuda_exactn_error_fallback`. The aggregate `cuda_exactn_fallback` is reported
but is not required to be zero: it also includes valid partial draft matches,
which deliberately restore the frontier and use exact replay.

The acceptance fixture can exercise the same SSD path on both the target-only
baseline and the DSpark run. It also requires real proposals and accepted
draft tokens, so an unavailable verifier cannot pass as a silent no-op:

```sh
DS4_DSPARK_FIXTURE_BACKEND=cuda \
DS4_DSPARK_FIXTURE_SSD_STREAMING=1 \
DS4_DSPARK_FIXTURE_SSD_STREAMING_CACHE_EXPERTS=32 \
make dspark-acceptance
```

For ROCm, use `make rocm-dspark-acceptance` with the same model, support, and
SSD fixture environment variables. The ROCm-specific target preserves the HIP
object set and linker; the generic target selects CUDA objects on non-Apple
hosts. `make rocm-dspark-verify-depth` provides the corresponding verifier
invariant test.
