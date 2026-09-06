# Metal Q4 Q-b: reuse packed weights across two tokens

Local review on Apple M1 Max, 2026-09-05, based on `fce65048`.
No remote tester, model file, inference run or SSD I/O was used. Pre-existing
uncommitted CUDA changes are independent of this Metal work.

## Kernel and admission

`kernel_mul_mv_q4_K_dense_token_pair_f32` loads each lane's packed Q4_K
quants, scales and minima once, then evaluates two tokens with independent
accumulators. The lane-to-K mapping, scalar accumulation order and final
`simd_sum` tree match the classic matvec. The threadgroup still contains two
SIMDgroups, each owning two output rows. The token grid shrinks from N to
ceil(N/2); odd N executes only the final valid token. No shared-memory barrier,
activation conversion or temporary buffer is added.

The production default is deliberately limited to standalone Q4_K
`K=1024, M=32768, N=2..8` on the Apple M1 family, with eight-byte-aligned
weight offsets, quality mode off and no tensor-parallel split. Measurements
below are from **M1 Max only**; other M1 variants still need tester validation.
Other Apple generations, one-token decode, Q-A/KV pairs and N>8 prefill retain
their existing dispatch. Failure to create the new pipeline also falls back.
The companion `dense_pair_token_pair` entry is oracle/benchmark-only and is
not selected by the runtime.

The presence-based rollback is `DS4_METAL_DISABLE_Q4_QB_TOKEN_PAIR=1`.
Any defined value, including `0` or empty, disables the candidate. Unset it
to return to the default. `DS4_METAL_DISABLE_Q4_MV_CLASSIC` also disables it.
SSD mode does not change admission, but this only optimizes the dense Q-b
compute; it neither changes expert reads nor establishes faster SSD streaming.

## Local timing

The fixture compares four arms: two legacy standalone calls, two token-pair
standalone calls, a legacy paired call and a token-pair paired call. Each
logical invocation computes two independent `32768 x 1024` projections with
the same activation. Both standalone arms submit two dispatches, so their
comparison does not include a dispatch-count reduction. Two resident weight
sets total 36 MiB; they are reused across iterations.

Each arm has two warmup samples and eight measured samples of 64 logical
invocations. Order alternates `{legacy standalone, candidate standalone,
legacy pair, candidate pair}` and its reverse. Values are median GPU
command-buffer times divided by 64, in microseconds **per pair of projections**,
not per single Q-b call or token. Compilation and host encoding are excluded.

The final all-batch confirmation run measured the production-relevant
standalone comparison as follows:

| Tokens | Legacy (us) | Token pair (us) | Time change |
| --- | --- | --- | --- |
| 2 | 437.170 | 319.653 | -26.88% |
| 3 | 660.046 | 522.465 | -20.84% |
| 4 | 868.887 | 628.109 | -27.71% |
| 5 | 1211.181 | 936.088 | -22.71% |
| 6 | 1446.035 | 1058.203 | -26.82% |
| 7 | 1675.676 | 1263.175 | -24.62% |
| 8 | 1900.497 | 1387.586 | -26.99% |

An earlier process measured N=2/3/5/8 at -23.29%, -19.98%, -23.44% and
-26.22% respectively. Absolute times varied between processes; compare
within-run arms rather than combining their absolute baselines.

Earlier exploratory K4096 Q-A/KV runs were mixed. For example, one run's
paired arm regressed 13.30% at N=2, and its standalone arm regressed 6.91%
at N=5. Consequently no K4096 runtime path was promoted. The retained
benchmark option times only the Q-b shape, not those exploratory K4096 cases.
The earlier rejected single-token experiments in
[metal_q4_decode_review.md](metal_q4_decode_review.md) remain removed.

These are resident, repeated kernel-only measurements. They do not predict
whole-model tokens/s, cold page faults, SSD latency or a gain on other GPUs.

## Verification and tester commands

```sh
make test-metal-ssd-decode-kernels
make test-metal-q4-qb-token-pair
make bench-metal-q4-token-pair
```

- The kernel oracle passed 212 normal cases; the benchmark run passed 219
  including seven production-size Q-b cases. Q4 dense coverage includes all
  1..8 token counts, K=256..7168, unequal/reversed output widths, odd tails,
  padded row strides, nonzero offsets, widths above 65535, changing activations,
  input/weight hashes and output guards. All candidate outputs matched the
  classic standalone result bit for bit. Existing Q4 SSD and Q8 oracles passed
  in the same run. Physical weight rows are padded for the legacy two-row
  loader; this does not prove unpadded odd-row safety for that old kernel.
- The public-runtime oracle passed 144 cases: reference `1`, default unset,
  rollback `0` and rollback empty; N=1..9; quality mode on/off; registered
  model spans with SSD mode on/off; alternating streams; nonzero output
  offsets; guards and inactive tokens. All four arms matched bitwise within
  each mode/shape. It uses synthetic weights and skips on non-M1 devices;
  the lower-level kernel oracle remains available there.
- The complete macOS build passed with `make -j4`.

For model-backed A/B, keep the same binary, model, prompt, verification batch,
cache state and all other settings. Compare the normal environment against
`DS4_METAL_DISABLE_Q4_QB_TOKEN_PAIR=1`, alternating order. Validate greedy
output and quality alongside latency. A row-at-a-time verifier that only
calls matvec with N=1 cannot use this optimization. Model-backed checks and
end-to-end performance remain for testers.

The global environment-document checker currently fails on an unrelated
pre-existing stale `external/system/HOME` source location. This change adds
only its new Metal metadata/reference entry; it does not regenerate the
unrelated inventory.
