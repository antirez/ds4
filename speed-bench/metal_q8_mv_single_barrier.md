# Metal Q8 decode: single-barrier reductions

Local kernel-only measurements on Apple M1 Max, with the candidate changes
on top of `f309990f`. These are synthetic resident projection measurements,
not model, prefill, decode tokens/s, or SSD-streaming benchmarks. No remote
tester machine was used.

The candidate keeps the two-row tile, input traversal, dot products, reduction
tree, and simdgroup count. Simdgroup leaders own shared slots `[0, NSG)` while
simdgroup zero clears only `[NSG, 32)`, removing the zero-fill barrier. After
the remaining barrier, only simdgroup zero reduces/stores the final outputs.

## Method

Run `make bench-metal-q8-mv` from the repository root (no GGUF needed).
The harness compiles the checked-in production kernels. It first runs 84
bitwise cases, including single versus paired projections, odd/unequal output
extents, independent weight-row strides, nonzero buffer offsets, input hashes,
output guards, and repeated dispatches with changing inputs.

The timed shape is K=4096, M=1024+512, one token. The four arms are legacy
separate matvecs, candidate separate matvecs, legacy paired matvec, and candidate
paired matvec. Each command buffer executes 128 logical projection pairs
(256 dispatches for separate matvecs, 128 for paired). Two warmup rounds precede
12 samples per arm, with arm order reversed every round. Reported microseconds
are the median GPU command-buffer duration per logical projection pair; host
setup/compilation are not timed. Weights and activations are reused across
dispatches, so this is a warm resident/cache microbenchmark, not an estimate
of streaming memory traffic. Two separate process runs were collected.

## Results

Negative time deltas are improvements. NSG is the number of simdgroups, not a
proposed change to the launch geometry.

| Run | NSG | Path | Legacy us | Candidate us | Time delta |
| --- | ---: | --- | ---: | ---: | ---: |
| 1 | 4 | Separate | 47.576 | 37.035 | -22.16% |
| 1 | 4 | Paired | 40.611 | 33.082 | -18.54% |
| 2 | 4 | Separate | 49.014 | 38.787 | -20.87% |
| 2 | 4 | Paired | 35.864 | 32.702 | -8.82% |
| 1 | 8 | Separate | 64.183 | 58.596 | -8.70% |
| 1 | 8 | Paired | 51.761 | 47.122 | -8.96% |
| 2 | 8 | Separate | 59.607 | 58.000 | -2.70% |
| 2 | 8 | Paired | 37.417 | 45.659 | +22.03% |

Both runs passed all 84 bitwise GPU cases. The host CUDA quantizer topology
check also passed 65,536 cases; that is not CUDA GPU validation.

## Runtime policy and tester rollback

Only eligible **4-simdgroup** ordinary/paired Q8_0 decode calls default to the
candidate, outside quality and TP modes. Standalone matvecs additionally require
even output extents. The 8-simdgroup candidate remains in the oracle/benchmark,
but is not selected by production after the mixed measurements above.
The old kernels remain available, including automatic fallback if candidate
pipeline creation fails. Prefill, TP flag producers, and shared-expert SwiGLU
dispatch are unchanged by this change.

Set `DS4_METAL_DISABLE_Q8_MV_SINGLE_BARRIER=1` for the legacy arm and unset it
for the candidate arm. Any defined value, including `0`, enables rollback.
Keep other optimization flags unchanged so the A/B isolates this reduction.
For full-model testing, hold model, prompt, context, cache size, and warmup
constant and compare fresh processes, alternating their order. Dense Q8
projections can be present in both AProjQ4 and AProjQ8 models; the affected
share of total runtime depends on the model and dispatch. These measurements
do not establish a 2% end-to-end decode gain or improved SSD throughput.
