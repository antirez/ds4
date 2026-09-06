# ROCm Q4 q_b F32 wave32 epilogue

Candidate implementation, not a demonstrated prefill improvement. HIP
compilation, GPU parity, register usage and throughput remain to be checked
on gfx1151. No remote tester was contacted. The CPU oracle is not GPU proof.

Local checks: host oracle passes with strict FP flags, production-like
fast-math and ASan/UBSan; the native harness passes host C++17 syntax checks.
The existing Q4 LDS, WMMA-load and decode host tests also pass. Without
`hipcc`, the optional GPU target skips and the required-device target fails
as intended. The environment-inventory generator is blocked by pre-existing
stale source metadata (`HOME` still refers to `ds4_agent.c:4023`); the two
new controls are documented in the curated reference and metadata table.

## Scope and implementation

Only the F32-output epilogue after the Q4 `attn_q_b` cached or transient F16
GEMM changes. Default-on admission is restricted to gfx1151 wave32,
256..4096 tokens, 64 heads, 512 values/head and 64 rotary values, outside
quality and SSD modes. In the ordinary resident configuration the transient
GEMM starts at 4096 tokens, so that is the main model-backed target. Smaller
direct-Q4 WMMA calls bypass this new epilogue entirely.

The legacy kernel uses one 256-thread block per head with nine block
barriers and separate input passes for the sum and output. The candidate
uses one wave32 per head and eight heads per 256-thread block. Each lane
retains sixteen F32 inputs. Eight canonical `x[i]^2 + x[i+256]^2` partials
are combined in the legacy stride-128/64/32 tree in registers, then the
stride-16/8/4/2/1 stages use wave shuffles. The 64 rotary inputs are exchanged
through registers so the output stage requires no input reloads. There is
no LDS allocation and no block barrier in this kernel.

Local Clang FP controls prohibit reassociation of the RMS reduction even
under the production `-ffast-math` flag, while permitting contraction inside
the original multiply/add expressions. Without that restriction, source
parentheses do not protect the tree. The YaRN expressions and the F32
GEMM/output boundary are unchanged. Native bitwise tests must still establish
that the compiler preserves the intended arithmetic on AMD.

This does not change GEMM selection, Q4 dequantization, F16-output experiments,
weight caching, the untyped normalization API, Q8, decode, Metal or CUDA.
It creates no extra device allocation and keeps the existing stream ordering.
Do not combine it with a transient-F16 rollback or F16-output experiment
when measuring its effect: those controls bypass the target epilogue.

## Default and rollback

```sh
DS4_ROCM_DISABLE_Q4_QB_F32_EPILOGUE=1 ./ds4-bench [usual arguments]
```

Unset, `0`, `false`, `no` or `off` permits the candidate; empty and other
non-false values disable it. Unsupported shapes/devices keep the legacy
kernel. Use `DS4_ROCM_Q4_QB_F32_EPILOGUE_STATS=1` in a separate diagnostic
process: `wave32_calls` should be nonzero for the candidate and zero on
rollback. No report can mean that another path owns `q_b`. Counters attest
successful enqueue, not GPU completion, bitwise correctness or speed.

## Verification without a model or Python

```sh
make test-rocm-q4-qb-epilogue-host
make test-rocm-q4-qb-epilogue ROCM_ARCH=gfx1151 DS4_TEST_REQUIRE_ROCM_DEVICE=1
make bench-rocm-q4-qb-epilogue ROCM_ARCH=gfx1151 DS4_TEST_REQUIRE_ROCM_DEVICE=1
```

The host test covers 26,400 admission combinations, unique 512-value
read/write ownership, rotary-pair mapping, a symbolic reduction tree and
10,000 F32 reductions. The target runs both the strict-FP C++ build and a
Clang fast-math build (`ROCM_Q4_EPILOGUE_HOST_CLANG` overrides `clang++`).
An adversarial rounding case detects fast-math reassociation. To run just
the production-like host build manually:

```sh
clang++ -O3 -ffast-math -fno-finite-math-only -std=c++17 -I. \
  tests/test_rocm_q4_qb_epilogue_host.cpp -o /tmp/ds4-qb-epilogue-host-fast
/tmp/ds4-qb-epilogue-host-fast
```

The native oracle exercises the same private dispatcher as both production
Q4 call sites. It compares default/rollback results over guarded allocations,
including token counts 256/257/512/2048/4096, positive/negative zero,
subnormals, disparate magnitudes, YaRN settings, inverse RoPE, nonzero
positions and four pointer offsets. It checks decode/small/large-shape,
quality, SSD and generic-API exclusions, boolean control semantics, invalid
spans and dispatch counts. Finite results must match bitwise; for non-finite
fixtures only NaN payload differences are allowed. A visible unsupported
GPU is a failure, not a silent fallback benchmark. No HIP/device may skip
only when the device is not explicitly required.

The isolated benchmark defaults to 4096 tokens and 12 ABBA/BAAB samples.
Each arm is reset to identical input outside timing; it never measures
repeated in-place normalization of changing data. Eight warmup calls precede
timing, and each measured call verifies full output, guards and dispatch.
HIP events enclose the production epilogue dispatch/kernel, excluding input
reset and readback. They do not time dequantization, GEMM or full-model prefill.
For another supported token count:

```sh
make bench-rocm-q4-qb-epilogue ROCM_ARCH=gfx1151 \
  DS4_TEST_REQUIRE_ROCM_DEVICE=1 \
  ROCM_Q4_EPILOGUE_TEST_ARGS="--tokens 2048 --samples 12"
```

## Model-backed acceptance

Use an already validated resident AProjQ4 model that fits the tester's RAM;
do not download a new model or force allocations around memory safeguards.
Use identical binary, model, prompt, power policy and context for both arms.
Leave persistent-cache overrides unset, keep transient F16 enabled and
F16 output disabled. Run the two arms in separate processes:

```sh
DS4_ROCM_DISABLE_Q4_QB_F32_EPILOGUE=1 ./ds4-bench --rocm \
  -m /path/to/aprojq4.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 32768 --step-incr 8192 --prefill-chunk 4096 \
  --gen-tokens 128 --csv q4-epilogue-legacy.csv 2>q4-epilogue-legacy.log

DS4_ROCM_DISABLE_Q4_QB_F32_EPILOGUE=0 ./ds4-bench --rocm \
  -m /path/to/aprojq4.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 32768 --step-incr 8192 --prefill-chunk 4096 \
  --gen-tokens 128 --csv q4-epilogue-wave32.csv 2>q4-epilogue-wave32.log
```

Discard a warmup run and the first 8192 frontier; collect at least three
measured runs per arm, alternating/reversing order and retaining distinct
CSV/log names. Compare 16384/24576/32768 prefill TPS and steady decode TPS.
Keep a same-commit AProjQ8 baseline to measure the remaining gap. Confirm
dispatch separately and compare full frontier logits/greedy output on the
non-quality optimized path, not only `--quality` (which deliberately falls
back). Do not accept a kernel-only win, altered output, or a speedup caused
by different chunking/cache state as evidence of closing the reported gap.
