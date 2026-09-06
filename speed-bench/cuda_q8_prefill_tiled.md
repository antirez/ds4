# CUDA Q8_0 tiled prefill producer

Candidate optimization, not a measured speedup. This Mac has neither nvcc
nor a CUDA GPU: native compilation, parity, resource usage and TPS must be
verified on the tester's GB10. No remote tester was contacted.

Local validation: the host oracle and ASan/UBSan pass; the GPU target fails
explicitly without nvcc. The existing environment-inventory check remains
blocked by stale `external/system/HOME` source metadata. New controls are
documented in the curated reference and metadata; the generated inventory
was not manually rewritten.

## Scope

The raw Q8_0 producer now has a separate large-prefill kernel. Eight warps
process eight contiguous 32-value quantization blocks in one 256-thread CTA.
The kernel uses the original shared-memory 16/8/4/2/1 maximum tree and
retains each input value for the rounding/store stage. Dummy warps in a
partial tile participate in all barriers and write nothing; valid tails
still write zeros. Quantization scales, rounding, clamping and layouts are
unchanged. For K=7168, N=4096 the launch has 114688 CTAs instead of 917504.
That is an 8x reduction in CTA count, **not** a claimed 8x speedup.

Default-on admission is restricted to a single GB10, 256..8192 logical
tokens and K=256..16384, outside quality mode. Ordinary native Q8_0 dense,
pair, K-slice and contiguous grouped output-A prequantization can select it.
The output-A call passes logical tokens, not its token*group packed rows.
Exact-decode APIs, top-1, HC, group-slice and dual Q8_K producers retain their
existing dispatch. Other GPUs, multiple GPUs and smaller/larger batches
fall back. No scratch allocation, host/device transfer or stream change is
introduced. Both resident and SSD calls can qualify, without changing I/O.

These activation producers can run in both AProjQ4 and AProjQ8 models;
this does not change weight quantization or the MMQ Q8_1 producer. Paths
that bypass raw Q8_0 through MMQ/GEMM/caches still bypass it. In particular,
a kernel-only win need not change the model's Q4/Q8 gap.

The candidate targets the CUDA quantizer area implicated by the reported
`f309990f` prefill regression, while preserving that commit's decode
dispatch. It is not a claim that this particular kernel alone explains all
of the regression; the three-way test below is required to separate effects.

## Controls

- Unset `DS4_CUDA_DISABLE_Q8_PREFILL_TILED`: candidate on eligible calls.
- `DS4_CUDA_DISABLE_Q8_PREFILL_TILED=1`: previous dispatch, including decode
  shuffles. This is the narrow A/B control for this change.
- `DS4_CUDA_DISABLE_Q8_QUANT_WARP_REDUCE=1`: original 32-thread shared-memory
  producers, overriding the candidate and also changing decode.
- `DS4_CUDA_Q8_PREFILL_QUANT_STATS=1`: logs each selected candidate with
  logical token count and K, before launch. Use only in an untimed process;
  selection is not successful completion or a measurement.

These are presence-based, cached at CUDA initialization: **even `0`
activates a flag**. Unset a rollback to test default behavior. Quality mode
always selects the original producer and cannot test the new kernel.

## Native verification and timing, no model or Python

```sh
make test-q8-quantize-host
make test-cuda-q8-quantize CUDA_ARCH=sm_121
make bench-cuda-q8-prefill-quantize CUDA_ARCH=sm_121
```

Rebuild after changing architecture/compiler flags; a stale executable is
not a valid result. The host target checks 65536 reduction/rounding cases,
2288 admission combinations and 2673 tiled mapping/reduction cases. It is
not CUDA compilation or parity proof. The GPU target must fail without a
CUDA device. The host-only executable rejects `--bench`.

The GPU oracle compares the original shared producer, existing shuffle
producer and new tiled producer bitwise. It checks quantized bytes/scales,
guarded and offset views, small/partial tiles, zero padding, signed zeros,
ties, subnormals, input immutability, non-default streams and graph replay.
Large fixtures include 256/2048/4096/8192 logical tokens and contiguous
group packing. Existing group-slice and dual-Q8_K tests still run.

On GB10 the benchmark runs that oracle first, then measures four shapes
with CUDA events around graphs containing sixteen independent quantizations
of immutable input. Eight warmup launches and an untimed graph replay
precede samples. Twelve samples use all six permutations of the three
variants twice. Output buffers are poisoned outside timing before every
sample; byte/scale parity and guards are checked afterward. Median/min/max
microseconds per producer and CTA counts are printed. Input copies, reset
and readback are excluded; graph replay overhead is included. This is not
full-model throughput. The benchmark rejects unsupported GPUs.

Before accepting, also run the native oracle under Compute Sanitizer
`memcheck`, `racecheck` and `synccheck`, and inspect compiler register/shared
memory usage for spills. Do not approve only from the CPU mapping test.

## Model-backed acceptance

First confirm `DS4_CUDA_Q8_PREFILL_QUANT_STATS=1` produces selection lines
with normal model settings. Absence of lines is a bypass, not evidence
that the candidate and rollback have equal performance.

Run candidate and narrow rollback in separate processes with the same
resident AProjQ4 model, input, power policy, chunk size and other flags:

```sh
env -u DS4_CUDA_DISABLE_Q8_QUANT_WARP_REDUCE \
  -u DS4_CUDA_Q8_PREFILL_QUANT_STATS DS4_CUDA_DISABLE_Q8_PREFILL_TILED=1 \
  ./ds4-bench --cuda -m /path/to/aprojq4.gguf \
  --prompt-file speed-bench/promessi_sposi.txt --ctx-start 8192 --ctx-max 32768 \
  --step-incr 8192 --prefill-chunk 4096 --gen-tokens 128 --csv q4-old-prefill.csv

env -u DS4_CUDA_DISABLE_Q8_QUANT_WARP_REDUCE \
  -u DS4_CUDA_DISABLE_Q8_PREFILL_TILED -u DS4_CUDA_Q8_PREFILL_QUANT_STATS \
  ./ds4-bench --cuda -m /path/to/aprojq4.gguf \
  --prompt-file speed-bench/promessi_sposi.txt --ctx-start 8192 --ctx-max 32768 \
  --step-incr 8192 --prefill-chunk 4096 --gen-tokens 128 --csv q4-tiled-prefill.csv
```

Repeat the same pair with AProjQ8. For a third, original-producer arm, set
the global warp rollback to `1`; that arm changes decode too and must not
be confused with the narrow prefill rollback. Discard warmup and first
frontier, collect at least three measured runs per arm in alternating order
with unique CSV names, and compare steady frontier prefill/decode TPS.
Check non-quality logits/greedy output as well as throughput. Keep the same
internal chunk: `--step-incr 8192` need not be one 8192-token kernel call.
Only model-backed results with matching output can establish a benefit or
any reduction of the Q4/Q8 gap.
