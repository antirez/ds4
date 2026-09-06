# CUDA Q4 grouped attention-A: cyclic groups and fused sanitization

## Candidate and scope

The established grouped attention-A MMVQ entry submits an ids producer,
Q8_1 quantization, output zeroing, a grouped matvec, and an output sanitizer.
The new specialization needs only the quantizer and the grouped matvec:
weight group `grid.y % n_groups` replaces the generated ids, while nonfinite
outputs are mapped to +0 at their final store. This removes two kernel
launches and one memset submission per grouped call, not per group.

The canonical Q4 dot loop, K partition, launch geometry, warp count,
peer-warp additions and reduction tree are unchanged. No new matmul
arithmetic, repacking, allocation or synchronization is introduced. Quantized
activations and outputs retain their token-major layout. The scratch arena
still reserves the old ids space, so toggling rollback never needs a larger
allocation and short-scratch calls retain their pre-enqueue failure contract.

This is default-on **within** the existing GB10-gated grouped MMVQ path, on
NVIDIA Turing or newer, with positive M divisible by four, positive K divisible
by 256, 1--8 tokens and 1--16 groups. Admission also bounds the total signed
weight-block index and the unsigned Q8/output channel offsets. Other shapes
keep the established grouped implementation. General routed MoE ids, HIP,
Metal, dense/pair MMVQ and grouped MMQ prefill above eight tokens are untouched.
The existing 2--8-token grouped batch opt-in is not enabled by this change.
Model storage/residency policy is unchanged: SSD runs can benefit only if
they already reach this grouped entry; this is not an SSD I/O optimization.

## Rollback

Set `DS4_CUDA_DISABLE_Q4_GROUPED_MMVQ_FUSION=1` to restore the ids producer,
output clear and separate sanitizer **without** disabling grouped dispatch.
Any defined value, including `0` or the empty string, disables the candidate.
The flag is read at enqueue; changing it cannot rewrite a captured graph.
It is independent of `DS4_CUDA_DISABLE_Q4_MMVQ_EPILOGUE` for dense/pair calls
and of the grouped prefill flags. Existing grouped/global rollbacks still win.

## Correctness and measurement

```sh
make test-q4-epilogue-host
make test-cuda-q4-epilogue CUDA_ARCH=sm_121
compute-sanitizer --tool memcheck ./tests/test_cuda_q4_epilogue
```

Use the architecture of the tester's device. The CUDA target runs the
existing 40 dense/pair cases plus 20 new grouped cases. Grouped coverage
includes the production M=1024/K=4096/G=8 shape at 1 and 8 tokens, K=256
through 8192, non-power-of-two group counts, four-row and one-row dispatch,
odd-row fallback, distinct groups/tokens, signed-zero/changing activations,
and finite/NaN/Inf weights. Fixture-only extra physical weight rows protect
the legacy tail loader; this does not establish safety of unpadded odd rows.

Each grouped case compares every output bit with independent per-token,
per-group legacy dense calls. It also compares grouped legacy vs candidate
Q8_1 bytes, checks input/weight immutability and buffer guards, rejects short
scratch in all four arms (unset, `1`, `0`, empty), and requires the unused ids
space to remain poisoned in the candidate. CUDA Graph node counts must drop
by exactly two kernels and one memset for eligible cases, and stay unchanged
on fallback/rollback. Replays poison scratch and outputs before every launch,
change the environment after capture, and check each replay's output.

Local host tests cover 4,194,320 sanitizer patterns, 112,860 dense admission/
coverage cases, and 130,680 grouped admission/channel cases plus offset
bounds. These checks also pass with ASan/UBSan. Source comparison verifies
the unchanged dot/reduction body and legacy grouped dispatch arguments.

CUDA/nvcc and HIP are unavailable on the local Mac. GPU compilation, the
60-case oracle, compute-sanitizer and performance measurements are **pending**.
No remote machine was contacted and no speedup percentage is claimed.

For end-to-end A/B testing use fresh processes with this flag unset vs `1`,
keeping the binary, model, quantization, prompt, context, warmup, storage and
graph settings fixed. Alternate adjacent arms using the existing balanced
A/B harness. Do not disable grouped dispatch or compare Q4 vs Q8 to isolate
this candidate. Check correctness and actual path coverage before throughput.
