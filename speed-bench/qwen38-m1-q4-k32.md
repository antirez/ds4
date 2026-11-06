# Qwen Q4 SSD prefill on M1 Max

Eight controlled long-prompt runs measure **+4.65% prefill** and
**-0.21% decode** throughput. The warm decode window changes
+2.28%. Decode is substantially unchanged in these samples; the earlier
−8.06% decrease is not reproduced by this controlled comparison. These results
do not guarantee a 5.5% gain or prove that scratch release fixed decode.

The runtime selects K32 staging and constant quantization specialization for
Q4_K/MXFP4 NT4 expert matrices in SSD streaming on M1 Max. Gate/up reuses each
Q4 header over eight staging steps without changing the 8-wide product order.
Resident inference, other token tiles and other devices keep K64 entry points.
At 8192 tokens and above, this Q4 pair uses half shadows while retaining its
public float intermediate. Dense Q8 matrices use unpack plus F16 GEMM on M1 Max
with SSD streaming at the same threshold, independent of expert quantization.
Allocation failure falls back to direct Q8; old scratch survives unretained
command buffers. The measured model's graph reserve covers the workspace
without reducing the expert cache.

At the synchronized Q4 row-kernel transition, half shadows and Q8 scratch are
released without adding a GPU wait. A later prefill converts fresh operands.
This releases 200 MiB in the measured long workload; its benefit to decode
speed was not established.

## Controlled complete-model measurements

Measured on 2026-09-27: Apple M1 Max, 32 GiB,
`Qwen3.8-Flash-Next-Q4.gguf`, Metal SSD streaming, context 32768, prefill cap
8192, temperature zero, thinking and MTP disabled. Every run uses 1908 cached
experts (4.82 GiB) and 22.96 GiB planned memory. The long prompt has 8631 tokens
(8192 + 439). With limit 128 it generates 100 tokens before EOS; the 102 forward
calls comprise two prefill calls and 100 decode calls.

The controlled comparison uses **one executable and one current Metal source**.
An ignored diagnostic copy toggles three defaults together: K32/specialization,
Q4 half shadows, and dense Q8 unpack. A disables all three; B enables them.
Both arms retain scratch-release code. The toggle and pipeline instrumentation
are absent from production. The source/patch chain was checked against the
production source; hashes and individual runs are in
[qwen38-m1-q4-k32.json](qwen38-m1-q4-k32.json).

An ABBA block was followed by one identical BAAB block to investigate decode;
all eight runs are included. Only these identical controlled blocks are pooled.
Rates use harmonic means, corresponding to equal-work elapsed time.

| Series | Prefill token/s | Prefill change | Decode token/s | Decode change |
|---|---:|---:|---:|---:|
| Long, controlled ABBA | 156.34 → 164.61 | +5.29% | 4.12 → 4.09 | -0.64% |
| Long, controlled BAAB | 152.33 → 158.48 | +4.03% | 4.01 → 4.02 | +0.21% |
| Long, all 8 controlled runs | 154.31 → 161.49 | +4.65% | 4.07 → 4.06 | -0.21% |

Individual decode rates span 3.99–4.22 token/s for A and
3.94–4.22 for B. The second 50-forward timing window averages
226.931 → 221.873 ms/forward (+2.28% throughput).
For the long prompt this covers decode forwards 49–98; the first window includes
prefill and is unsuitable as a warm-decode measurement. Its `encode` bucket
includes intermediate GPU and SSD waits; `gpu` is only the final wait.

Instrumented decode pipeline creation totals 0.691–2.795 ms per run,
far below a seconds-scale regression. Instrumentation covers generic and Qwen
MV/MM cache misses, not every specialized pipeline factory. Four runs per arm
still do not bound performance across devices, prompts or system conditions.

## Earlier diagnostics

These separate series explain the investigation; they are not pooled with the
controlled comparison. The historical baseline is `ee79f22b90ebd3d1c1219c3006ba2945a7184456`.

| Series | Prefill token/s | Prefill change | Decode token/s | Decode change |
|---|---:|---:|---:|---:|
| Long, initial ABBA | 157.53 → 163.32 | +3.68% | 4.10 → 3.89 | -5.12% |
| Short, initial ABBA, limit 32 | 8.10 → 8.08 | -0.24% | 5.09 → 4.84 | -4.95% |
| Long, scratch release BAAB | 159.54 → 161.39 | +1.16% | 4.21 → 3.87 | -8.06% |
| Short, shader-only ABBA, limit 128 | 8.01 → 7.95 | -0.76% | 5.68 → 5.50 | -3.06% |
| Short, host comparison AB, limit 128 | 8.15 → 8.14 | -0.12% | 5.75 → 5.95 | +3.48% |

Only the initial **long** candidate retains 140 MiB of half shadows and 60 MiB
of Q8 scratch after prefill. The short candidate allocates neither. Releasing
scratch brings cleanup memory back to baseline but the separate-binary BAAB
series still shows a decode decrease; memory retention alone does not explain
it. Shader-only runs keep the old executable and change only Qwen source.
The host comparison keeps the current shader and has just one run per arm;
its warm window is 158.344 → 158.724 ms/forward.
The earlier exploratory +5.44% used a different prototype, limit 32 and Q8
unpack on the 439-token tail too, so it is not a production result.

All 26 outputs are byte-identical within the same prompt and generation limit;
expert hit/miss counts and requested pread bytes match too. Long runs request
166.14 GiB of expert reads each. These are requested bytes, not measured physical
SSD traffic. Cleanup counters combine prefill and decode; they cannot isolate
phase I/O time. The row-kernel split counters cover 4786 of 4800 decode layers;
their measured intervals increase by 0.185 s while the earlier BAAB's total
decode time increases by 2.082 s. N-gram reads
use a separate descriptor with unchanged cache flags; the short warm runs have
the same 0.450 ms input-stage average, providing no evidence of a n-gram I/O
regression. Filesystem cache, storage latency, background macOS work and clocks
remain uncontrolled. Two-run historical arms are particularly unstable.

## Validation

- Forced full build with `make -B -j4 all`; runtime and SSD test rebuilt again
  after adding the synchronized scratch release. No compiler warnings.
- Independent frozen-reference Metal oracle, default and safe math: Q4/MXFP4
  passes 331,727,600 exact FP32 comparisons and 121,446,912 half comparisons per
  mode. It exercises K32 float/half, unchanged K64 NT1/2/4/8, partial rows,
  sparse routing, and token counts 439/8191/8192.
- IQ2/Q2 K64 oracle passes 19,998,720 exact FP32 and 7,428,096 half comparisons
  per mode. The Q4 resident K64 pair measures 293.15 → 290.34 ms (+0.97%) at
  8192 tokens and 512 experts: no observed resident regression. A full resident
  Q4 model run does not fit this device; this check uses resident tensors.
- `tests/test_qwen4_ssd_experts`: exact resident/SSD intermediates and outputs,
  missing-only reads, failed down reads after gate/up, 8191/8192 threshold,
  growth to 16384, reuse, decode release followed by a fresh 8192-token prefill,
  and explicit dispatch checks for K32 versus other token tiles.
- `tests/test_q8_prefill_variants`: exact outputs and guards, scratch growth in
  one batch with both retained and unretained command buffers, and SSD
  8191/8192 parity.

The kernel changes are Metal-only. No CUDA or distributed code changes are
included; measurements do not certify other devices or Q2 model throughput.

## Reproduction

To compare complete production revisions, build baseline and candidate in
separate checkouts. Run only one GPU workload
at a time. Each executable must use its corresponding `metal/` source tree.
Use the same GGUF file and generated prompt in both checkouts, with this command:

```sh
./ds4 -m "$QWEN_Q4_MODEL" --metal --ssd-streaming --ctx 32768 \
  --prefill-chunk 8192 --nothink --temp 0 -n 128 \
  --prompt-file /tmp/qwen-k32-long.txt
```

Generate the exact long prompt (newlines and trailing spaces are intentional):

```python
from pathlib import Path
paragraph = (
    "Roma nacque sulle rive del Tevere e divenne il centro di una vasta rete di scambi. "
    "Durante la repubblica le istituzioni cambiarono e la cittadinanza si estese gradualmente. "
    "Le conquiste portarono ricchezza e conflitti, mentre strade e acquedotti unirono territori lontani. "
    "Con Augusto iniziò una nuova fase politica, seguita da trasformazioni sociali e culturali. "
)
block = "Riassumi in italiano questi appunti storici in tre frasi.\n" + paragraph * 2 + "\n"
prompt = "Riassumi i seguenti appunti storici in tre frasi in italiano.\n\n" + (block * 6 + "\n") * 8
Path("/tmp/qwen-k32-long.txt").write_text(prompt)
Path("/tmp/qwen-k32-short.txt").write_text("Narrami in italiano la storia di Roma\n")
```

For short runs, use the short prompt and `-n 32` for the initial series
or `-n 128` for shader-only and warm diagnostics. Set
`DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1` to retain the same memory/I/O
counters. Keep other diagnostic environment overrides unset.

To repeat the controlled comparison, use a temporary source copy and a single
executable with a temporary boolean switching these three decisions together:
`qwen4_moe_mm_k32`, the Q4/MXFP4 branch of `half_prefill`, and the M1 Max/SSD
branch enabling Q8 unpack in `ds4_gpu_matmul_q8_0_tensor`. Keep scratch release,
all other host code, and `metal/qwen4.metal` identical. Run ABBA then BAAB;
set `DS4_QWEN4_TIMING=1` for the 50-forward windows. This requires a diagnostic
build: there is no production environment flag for that combined switch.
The JSON records the diagnostic binary/source hashes and instrumentation scope;
the ignored local diagnostic build is not included in this archive.

The independent correctness check accepts an explicit baseline source:

```sh
git show ee79f22b90ebd3d1c1219c3006ba2945a7184456:metal/qwen4.metal > /tmp/qwen4-ee79f22.metal
make -B tests/test_metal_qwen4_moe_half
./tests/test_metal_qwen4_moe_half --q4-k32 --baseline-source /tmp/qwen4-ee79f22.metal
./tests/test_metal_qwen4_moe_half --q4-k32 --safe-math --baseline-source /tmp/qwen4-ee79f22.metal
```
