# ROCm AProjQ4 prefill gap: attribution before another kernel change

The [tester comment](https://github.com/antirez/ds4/pull/952#issuecomment-5558676670)
reports a remaining Q4 prefill regression on gfx1151 with ROCm 7.1.1.
These are individual runs, not a repeated A/B estimate with uncertainty.
`--step-incr 8192` selects benchmark frontiers; it does **not** select an
8192-token kernel batch. Flash defaults to a 4096-token prefill cap in
`ds4_prefill_cap_for_prompt()`. Set that cap explicitly when reproducing.
This checkout is on an M1 Max with 32 GB, without an AMD GPU or HIP runtime.
No AMD compilation, GPU correctness, or percentage improvement is claimed.

The aligned-LDS candidate described below is now implemented; see its
[kernel changes and runnable A/B tests](rocm_q4_prefill_lds_aligned.md).

## What to measure

At 4096 tokens, resident Q4 `q_b` normally uses transient Q4-to-F16 weights,
rocBLAS and an F32 epilogue. The synthetic `--case qb` benchmark calls the
generic quantized matmul API and measures TILE4/direct WMMA instead; it does
not time that complete production path. Output A normally uses direct WMMA,
while Q4 output B (K=8192, M=4096) retains Q8_K activation quantization and
TILE8 integer dots. Output B is therefore a plausible exact-arithmetic
optimization target, **not an established bottleneck**.

The paired Q-A/KV TILE8 path already shares activation quantization and one
launch. Direct WMMA bypasses that pair. A, B and `q_b` consume different
inputs, so their quantizations cannot simply share a cached result.

## Attribute Q4 versus Q8 with the existing stage profiler

Use the same binary and already available matched AProjQ4/AProjQ8 models.
Keep other quantizations, prompt, context, power and residency settings equal.
Run each command in a separate process; then repeat in reverse order:

```sh
DS4_ROCM_LAYER_STAGE_PROFILE=1 ./ds4-bench --rocm -m /path/aprojq4.gguf \
  --prompt-file speed-bench/promessi_sposi.txt --ctx-start 8192 \
  --ctx-max 32768 --step-incr 8192 --prefill-chunk 4096 \
  --gen-tokens 128 --csv q4-profile.csv 2>q4-profile.log
DS4_ROCM_LAYER_STAGE_PROFILE=1 ./ds4-bench --rocm -m /path/aprojq8.gguf \
  --prompt-file speed-bench/promessi_sposi.txt --ctx-start 8192 \
  --ctx-max 32768 --step-incr 8192 --prefill-chunk 4096 \
  --gen-tokens 128 --csv q8-profile.csv 2>q8-profile.log
python3 speed-bench/analyze_rocm_prefill.py q4-profile.log q8-profile.log \
  --target-tps-pct 8
```

Inspect `q_path`, `output_proj`, attention and FFN contributions by position.
The existing logs say `metal layer stage` even on ROCm. For a separate Q-path
breakdown, replace the layer flag with `DS4_ROCM_Q_STAGE_PROFILE=1`; this
reports `q_a`, `q_a_norm`, `q_b`, `head_norm` and `rope`. Fused Q-B projection,
normalization and RoPE may all appear under `q_b`; subsequent labels can be
nearly empty. `pre_q` includes setup preceding `q_path`; the Q profiler overlaps
attention stages and must not be added to their recorded time.
`DS4_ROCM_Q4_PREFILL_TILE8_STATS=1` in another diagnostic run establishes
which Q4 paths enqueue work. Counters alone establish neither time nor parity.
Stage profiling synchronizes at boundaries and changes scheduling; its CSV
throughput is unsuitable for accepting an optimization.

## Measure the requested improvement separately

For baseline and candidate **Q4** binaries, repeat the same benchmark arguments
with all profiling flags unset, distinct CSV/log names and the identical model.
Discard a warmup run and compare the 16384/24576/32768 frontiers across at
least three measured runs per arm, alternating A/B/B/A and reversing order.
Keep the Q8 control separate from the Q4 before/after comparison. Require
unchanged frontier logits and generated output for an exact optimization;
`--dump-frontier-logits-dir` and `--show-output` support a separate parity run.
Do not enable `--quality` as a substitute for testing the optimized path.

+8% tokens/s means a 7.407% reduction in elapsed prefill time. If a stage
accounts for fraction p of that time, it must lose at least 0.074074/p of
its latency when every other stage stays constant. Closing a 7.4% Q4 TPS
deficit against Q8 needs a 7.99% Q4 TPS gain, not merely 7.4%.

## Aligned-LDS implementation and a larger hypothesis

The TILE8 candidate stages Q8_K in an LDS-only 304-byte layout:
`d` at byte 0, 12 bytes of padding, `qs` at byte 16 and `bsums` at byte 272.
The packed GGUF/scratch format remains 292 bytes. Aligning each staged block
to 16 bytes permits explicit int4 activation loads in the dot loop, analogous
to the aligned Q8 staging already used by the ROCm MXFP4 MoE kernel. Preserve
every payload bit and the existing dot/FP32 operation order. LDS rises from
18,688 to 19,456 bytes per TILE8 workgroup; compiler registers, occupancy and
actual instructions still need checking. This targets TILE8 consumption,
whereas the tested K64 LOAD4 switch targets direct WMMA input loading.

An output-B WMMA kernel could multiply Q4 nibbles (0..15) and Q8_K int8
values represented exactly in F16, accumulating each 32-value group in F32.
The absolute group-dot bound is 32*15*128 = 61440, below 2^24. Convert each
group dot to int32 and preserve the baseline integer scale/min accumulation:
scaled sums across groups can exceed the exact-integer range of F32. Retain
the baseline Q8_K quantizer, block/lane FP32 operation order and reduction.
This may offer more than another activation-loader change, but its mapping,
register pressure and throughput require AMD measurements. The bound is a
mathematical prerequisite, not proof of emitted GPU arithmetic or speed.

## Validation for any future candidate

```sh
make
make test-rocm-q4-lds-host test-rocm-q4-wmma-load-host
# On gfx1151 with HIP; a missing device must fail rather than count as PASS:
make test-strix-rocm-q4-prefill ROCM_ARCH=gfx1151
make rocm-q4-prefill-bench ROCM_ARCH=gfx1151
./speed-bench/rocm_q4_prefill_bench --case output \
  --tokens 256,512,1024,2048,4096 --sets 4 --samples 16 --warmup 4
```

The existing tests cover existing kernels only. A new exact WMMA path needs
its own GPU bitwise oracle, guarded tails and dispatch assertions, followed
by model-backed parity and unprofiled throughput. Host tests cannot prove
HIP compilation, GPU arithmetic or the requested +8%.
