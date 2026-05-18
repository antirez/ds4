# CUDA MTP on DGX Spark / GB10

This note covers the CUDA speculative decoding path for DeepSeek V4 Flash MTP
on NVIDIA GB10 systems such as DGX Spark. The feature is experimental. The
optimized non-strict path is useful as a small greedy-decoding speedup on the
benchmarks below, but it should be validated against no-MTP output for each
prompt class. Use `--quality` / `DS4_MTP_STRICT=1` when byte-identical target
stream behavior matters more than speed.

## Hardware Target

The current CUDA MTP work was tuned against:

- NVIDIA GB10, `sm_121`.
- 128 GB memory class.
- Linux CUDA build with cuBLAS.
- Main q2-imatrix DeepSeek V4 Flash GGUF.
- Optional Q4_K MTP GGUF.

The optimization assumptions are specific to this shape: a very large target
model, an MTP draft block that is much smaller but still non-trivial, and a
two-token target verifier.

## Build

On Linux, the default build uses CUDA:

```sh
make clean
make -j8
```

The build uses `CUDA_ARCH=native` by default. Override it if the local CUDA
toolchain cannot infer the GPU architecture:

```sh
make clean
make -j8 CUDA_ARCH=sm_121
```

Run the CUDA regression target after changing CUDA kernels:

```sh
make cuda-regression
```

## Model Files

Download a supported main model and the MTP model:

```sh
./download_model.sh q2-imatrix
./download_model.sh mtp
```

The MTP model is not loaded automatically. Pass it explicitly with `--mtp`.

## Basic Run

No-MTP CUDA baseline:

```sh
./ds4 --cuda \
  -m ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --temp 0 --nothink -n 256 \
  -p "List 500 prime numbers, comma-separated, just numbers."
```

Optimized CUDA MTP, draft depth 2:

```sh
DS4_CUDA_MTP_TOP2=1 \
DS4_CUDA_MTP_VERIFY_TOP2=1 \
./ds4 --cuda \
  -m ./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --mtp ./gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --mtp-draft 2 \
  --temp 0 --nothink -n 256 \
  -p "List 500 prime numbers, comma-separated, just numbers."
```

Use greedy decoding (`--temp 0`) when evaluating MTP throughput. Speculation is
most meaningful when target verification can accept or reject deterministic
drafts.

## Optimization Flags

The currently useful CUDA MTP flags are:

- `DS4_CUDA_MTP_TOP2=1`: use the CUDA top-2 output path for MTP draft decisions.
- `DS4_CUDA_MTP_VERIFY_TOP2=1`: use top-2/top-1 verifier shortcuts where full
  logits are not needed.
- `DS4_CUDA_MTP_VERIFY_OPT_OUTPUT=1`: optional A/B switch that schedules row0
  top-2 and row1 full-output work together. It is exact, but current GB10
  measurements favor the simpler no-opt schedule, so it is not part of the
  recommended fast baseline.
- `DS4_MTP_CERT_LOGITS=1`: opt-in exact candidate-certified row0 output
  verifier for CUDA decode2. It proves the drafted row1 token is row0 argmax
  before skipping the row0 top-2 scan; uncertified rows fall back to exact
  top-2. Use `DS4_MTP_CERT_LOGITS_SHADOW=1` to validate the certificate without
  changing control flow and `DS4_MTP_OUTPUT_VERIFY_TIMING=1` for row0 output
  verifier timing. The certificate bound uses 8 output-vector groups by
  default; set `DS4_MTP_CERT_LOGITS_GROUPS=N` (`1..16`) for profiling.
- Pair output projection is default-on for exact CUDA decode2 when candidate
  certification is not active. It stages row0/row1 normalized output
  activations together, then runs one mixed Q8_0 output projection producing
  row0 top-2 and row1 full logits. Rejected row0 fallbacks reuse the staged
  row0 normalized activation for full logits. Use
  `DS4_MTP_NO_VERIFY_PAIR_OUTPUT=1` to restore the older separate row0 top-2
  plus continuation-logits schedule.
- `DS4_MTP_VERIFY_V2_SHADOW=1`: opt-in diagnostic path for the fixed-depth
  decode3 verifier primitive. It runs in addition to the authoritative verifier
  and should only be used for proof runs.
- `DS4_MTP_VERIFY_V2=1`: opt-in active decode3 suffix verifier for
  `--mtp-draft 3`. It exercises selected-row output and prefix-depth commit
  without changing the default `mtp-fast` path.
- `DS4_MTP_BATCH_FIRST=1`: experimental purpose-built verifier probe. It tries
  to reuse a previously seeded MTP prediction as the first token of a three-row
  target verifier and automatically falls back through the V2 verifier when the
  proof-safe margin gate fails. The default mode does not seed from batched
  target rows, preserving exact comparison against the current fast stream in
  the proof harness.
- `DS4_MTP_BATCH_FIRST_MIN_MARGIN=N`: margin gate for the batch-first probe.
  The default inherits the normal MTP margin. Higher values force more fallback
  to the current verifier.
- `DS4_MTP_BATCH_FIRST_SEED_BATCH_ROWS=1`: research-only mode that seeds the
  next MTP prediction from batched verifier hidden rows. This improves
  batch-first residency, but current GB10 proof runs show it can change greedy
  prose output, so it is not a verified optimization.

The structural CUDA optimizations are default-on after this work:

- Batched Q8 pair projections for two-token verifier passes. Long ordinary
  prefill chunks stay on the normal Q8/cuBLAS path because the pair kernel is
  tuned for `n_tokens <= 2`, not thousands-token prefill batches.
- Fused six-expert MoE down+sum for the two-token verifier.
- Paired decode Q/KV projections.
- Startup pre-cache of both the base model and MTP model tensor spans.

Rollback switches:

- `DS4_CUDA_NO_BATCH_Q8_PAIR=1`
- `DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6_N2=1`
- `DS4_CUDA_NO_DECODE_Q8_PAIR=1`

These are intended for A/B performance checks and debugging, not normal use.

## Benchmark Method

Use identical prompt, model, token count, and decoding settings across all
runs. Capture stderr separately because throughput and MTP timings are written
there.

```sh
BASE=./gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf
MTP=./gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
PROMPT="List 500 prime numbers, comma-separated, just numbers."

./ds4 --cuda -m "$BASE" --temp 0 --nothink -n 256 -p "$PROMPT" \
  > /tmp/ds4_nomtp.out 2> /tmp/ds4_nomtp.log

DS4_CUDA_MTP_TOP2=1 \
DS4_CUDA_MTP_VERIFY_TOP2=1 \
DS4_MTP_TIMING=1 \
./ds4 --cuda -m "$BASE" --mtp "$MTP" --mtp-draft 2 \
  --temp 0 --nothink -n 256 -p "$PROMPT" \
  > /tmp/ds4_mtp.out 2> /tmp/ds4_mtp.log
```

Compare output bytes before trusting a speed result:

```sh
cmp -s /tmp/ds4_nomtp.out /tmp/ds4_mtp.out && echo MATCH
```

For a broader proof-bar run across prompts and optimization variants, use:

```sh
DS4_PROOF_BASE="$BASE" \
DS4_PROOF_MTP="$MTP" \
tests/cuda_mtp_proof_matrix.py --budget candidate
```

The matrix compares generated bytes against no-MTP, compares variants against
the current fast MTP path, and parses shadow-verifier diagnostics such as
`DS4_CUDA_MTP_SHADOW_B_N2_Q8=1`. This is the preferred way to localize whether
drift comes from the current fast verifier, a rollbackable structural
optimization, or a candidate shadow path.

Use `--custom candidate:CANDIDATE_FLAG=1` to add a candidate MTP variant on top
of the fast-MTP baseline flags without editing the script.

For a rollback comparison:

```sh
DS4_CUDA_MTP_TOP2=1 \
DS4_CUDA_MTP_VERIFY_TOP2=1 \
DS4_CUDA_NO_BATCH_Q8_PAIR=1 \
DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6_N2=1 \
DS4_CUDA_NO_DECODE_Q8_PAIR=1 \
DS4_MTP_TIMING=1 \
./ds4 --cuda -m "$BASE" --mtp "$MTP" --mtp-draft 2 \
  --temp 0 --nothink -n 256 -p "$PROMPT" \
  > /tmp/ds4_mtp_rollback.out 2> /tmp/ds4_mtp_rollback.log
```

## Current GB10 Result

On the 256-token prime prompt above, a clean CUDA build on GB10 measured:

| Run | Prefill | Generation |
| --- | ---: | ---: |
| No MTP | 30.12 t/s | 15.46 t/s |
| Optimized MTP | 30.10 t/s | 16.21 t/s |
| Rollback structural opts | 27.00 t/s | 15.14 t/s |

The no-MTP, optimized MTP, and rollback MTP outputs were byte-identical for
that prime-list run. Other prompt classes can differ in the non-strict
optimized verifier path, so compare output bytes before treating a result as an
exact speedup.

MTP verifier timing from the same run:

| Run | Micro verify avg | Margin-skip verify avg |
| --- | ---: | ---: |
| Optimized MTP | 106.08 ms | 64.53 ms |
| Rollback structural opts | 118.81 ms | 65.75 ms |

The verified speedup is modest but real on this benchmark: optimized exact MTP
beat the no-MTP baseline by about 4.9%, and beat the structural rollback by
about 7.1%.

## Current Profiling Details

For stage attribution, use the graph stage profiler plus MTP timing:

```sh
DS4_CUDA_MTP_TOP2=1 \
DS4_CUDA_MTP_VERIFY_TOP2=1 \
DS4_MTP_TIMING=1 \
DS4_METAL_LAYER_STAGE_PROFILE=1 \
DS4_CUDA_MOE_PROFILE=1 \
./ds4 --cuda -m "$BASE" --mtp "$MTP" --mtp-draft 2 \
  --temp 0 --nothink -n 32 -p "$PROMPT" \
  > /tmp/ds4_mtp_profile.out 2> /tmp/ds4_mtp_profile.log
```

`DS4_METAL_LAYER_STAGE_PROFILE` is shared by the graph encoder despite the
name, so it also profiles the CUDA graph path. It synchronizes at each stage
boundary, which makes the run slower; use it for attribution, not headline
throughput.

On GB10, the current two-token target verifier profile measured about
`103.45 ms` of synchronized layer work per verifier cycle. Normal unsynchronized
MTP timing from the same code path measured about `106 ms` average verifier time,
so the stage profile accounts for almost all of the remaining verifier cost.

Per verifier cycle, two-token layer-stage attribution was:

| Stage | Time | Share |
| --- | ---: | ---: |
| routed MoE | 29.73 ms | 28.7% |
| attention output projection | 27.20 ms | 26.3% |
| Q path | 15.03 ms | 14.5% |
| compressor | 6.81 ms | 6.6% |
| shared expert gate/up | 4.91 ms | 4.7% |
| shared expert down | 4.05 ms | 3.9% |
| indexer setup | 4.04 ms | 3.9% |
| HC pre projections | 3.97 ms | 3.8% |
| attention proper | 2.71 ms | 2.6% |
| norms | 1.58 ms | 1.5% |
| router | 1.41 ms | 1.4% |
| HC post projections | 0.91 ms | 0.9% |
| KV path | 0.81 ms | 0.8% |
| inverse RoPE | 0.31 ms | 0.3% |

Grouped by layer half:

| Layer half | Time | Share |
| --- | ---: | ---: |
| Attention-side work | 60.18 ms | 58.2% |
| FFN/MoE-side work | 43.27 ms | 41.8% |

The CUDA MoE sub-profile for the same two-token verifier shape averaged:

| MoE substage | Time |
| --- | ---: |
| input Q8 quantize | 0.006 ms |
| pair sort | 0.040 ms |
| gate/up | 0.407 ms |
| mid Q8 quantize | 0.004 ms |
| down | 0.222 ms |
| sum | 0.001 ms |
| total per layer | 0.680 ms |

This is why the remaining optimization target is clear: most of the exact MTP
cost is still the full target verifier pass, especially routed MoE, attention
output projection, and the Q path.

## Why The Gain Is Modest

MTP is not free. Each speculative cycle does three expensive things:

1. Runs the MTP draft block to propose candidate tokens.
2. Runs the target model over the drafted suffix to verify exact correctness.
3. Produces full target logits for the token that becomes the new sampling
   frontier.

The verifier dominates the remaining cost. Even with draft depth 2, the target
model still has to execute all layers over a two-token suffix. The CUDA work in
this branch reduces avoidable overhead inside that verifier, but it does not
remove the target-model pass itself.

The most important current structural wins are therefore narrow:

- Avoid duplicate Q8 activation quantization for paired projections.
- Fuse verifier MoE down projection and expert summation for the two-token case.
- Avoid unnecessary full-vocab output work when top-1/top-2 is enough.
- Keep startup model and MTP tensor spans warm so first-use copies do not pollute
  decode timing.

Future work should continue to target the target-layer verifier pass, especially
the large attention/output and MoE stages, while preserving exact output parity.
