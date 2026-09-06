# Q4 dense decode: rejected kernel candidates

Local review on Apple M1 Max, 2026-09-05. Base checkout: `f309990f`, with
the pre-existing Q8 single-barrier changes still present. No remote machine,
model runtime, GGUF, or SSD I/O was involved. No new Q4 performance path or
environment flag was enabled by this review.

## Comparison

The existing `metal_q4_dense_pair_bench.m` was temporarily extended to compare
the current sequential **paired** kernel with each candidate. Both arms used
one dispatch per logical Q-A/KV call, not two standalone dispatches as the
baseline. Shape: K=4096, output widths 1024 and 512, one token, two SIMDgroups
per threadgroup, two rows per SIMDgroup. Each arm used 64 rotating resident
weight sets (216 MiB), 256 calls/sample, 16 samples/arm, GPU timestamps and
alternating ABBA/BAAB order. All 64 sets matched bitwise with intact canaries.

| Candidate | Run | Legacy median | Candidate median | Time change |
| --- | --- | --- | --- | --- |
| Shared activation loads and block sums for overlapping Q-A/KV rows | 1 | 33.757 us | 36.142 us | +7.06% |
| Same shared-activation candidate | 2 | 31.340 us | 39.531 us | +26.13% |
| Independent row ranges in a single dispatch, original matvec inner loop | 1 | 32.710 us | 35.073 us | +7.23% |

The independent-range variant used 384 threadgroups instead of 256, with the
same two SIMDgroups per threadgroup. It removed sequential projection work
inside each group but did not improve the measured latency. These are observed
kernel timings, not proof of a specific register/occupancy bottleneck or of
behavior on other Apple GPUs. Permission for a longer confirmation run was
declined; that run did not execute.

Both candidates and their temporary benchmark options were removed. The
existing benchmark remains available unchanged:

```sh
make metal-q4-dense-pair-bench
./speed-bench/metal_q4_dense_pair_bench
```

That command compares the original separate and paired kernels; it does not
reproduce the discarded candidates in the table.

## Retained correctness coverage

Added 48 Q4 dense-pair cases to `tests/test_metal_ssd_decode_kernels.m`, comparing
against the canonical standalone matvec. Coverage includes K=256 through 7168,
unequal/reversed widths, partial row cohorts, independent padded weight strides,
1/3/8 tokens, widths above 65535, changing activations and input/output guards.
Physical weight rows are padded in the fixture for the legacy two-row loader;
these tests do not establish that unpadded odd-row allocations are safe.

All 132 decode cases passed locally while the independent-range candidate was
also included as a third Q4 comparison arm. After removing that arm, the
retained tests compiled successfully and `git diff --check` passed; no additional
GPU run was requested. The pre-existing Q8 optimizations were not changed.
