# Persistent TP Gate Kernel — Baseline, A/B numbers and verdict

**Verdict (2026-09-03): negative result, reverted.** The resident gate
kernel made decode 14% slower (13.99 → 11.9 tok/s). A per-gate trace showed
the premise was wrong: the decode gate window is ~19–21 ms/token on *both*
paths and consists of NHI one-way latency (~217 µs/gate) plus MoE routing
skew between the ranks (~190 µs/gate) — neither of which a launch-free gate
kernel can touch — while the resident kernel cost ~15 ms/token in the
compute segments between gates. Code is preserved as
`persist-gate.patch` in this directory; only the harness
(`scripts/tp_ab.sh`) and this doc set stay in the tree.

Pair: `max2` (rank 0, leader, `10.99.0.2`) / `max` (rank 1, worker),
Strix Halo gfx1151, ROCm TheRock HIP 7.13, NHI over Thunderbolt
(`/run/ds4-tbstream/device`, port 18128). Model
`DeepSeek-V4-Flash-MXFP4Experts-…-mxfp4-0731.gguf` (145.26 GiB, 76.73 GiB
expert shard per rank, 50/50 split). Harness: `scripts/tp_ab.sh <persist>
197 128 <tag>` (`ds4-bench --prompt-file tp-ab-prompt.txt --ctx-start 197
--ctx-max 197 --gen-tokens 128 --tensor-parallel --transport nhi`). One
build for both columns; the flag-off path was byte-identical to the
pre-POC engine. Clocks not pinned; legacy runs agree to 0.1%.

## End-to-end (ds4-bench CSV, leader)

| run | path | prefill 197 tok/s | gen tok/s (128) | first-token ms | NHI | token-exact |
|---|---|---:|---:|---:|---|---|
| base02 | legacy (pre-POC binary) | 113.65 | 13.98 | 71.15 | failures=0 event_drops=0 crc=0 overrun=0 | ref |
| leg03 | legacy (flag off) | 113.62 | 14.00 | 70.91 | same | = |
| leg08 | legacy (flag off, traced) | 113.43 | 13.99 | 71.12 | same | = |
| per04 | persistent, scalar wave copies | 110.33 | 11.59 | 86.47 | same | = |
| per06 | persistent, vectorized wave copy | 110.26 | 11.98 | 83.62 | same | = |
| per07 | persistent, traced | 112.09 | 11.92 | 83.86 | same | = |

(per03 hung in `hipDeviceSynchronize` before the pause/resume wrappers
existed; per05 never ran — harness driver died with the saved session.)
First decoded text, temp 0, identical in every run: `ricamo al loro
valore. A me però, siccome l'umiltà mia non ha mai`. The persistent path
was bit-exact (R2) — it was just slower.

## Per-gate decomposition (trace, `DS4_TP_GATE_TRACE`, runs per07 / leg08)

The resident kernel stamped `wall_clock64` at TX release, RX spin start,
peer stamp first seen and RX copy done; the service thread stamped
`CLOCK_MONOTONIC` around the NHI submit ioctl. Joining the two ranks'
traces on the gate `seq` splits the arrival wait `X = seen − tx` into
one-way latency and skew with no cross-node clock sync: rank 0 waits
`L + s`, rank 1 waits `L − s`. The legacy trace has the equivalent
`hipEventElapsedTime(tx_ready, rx_consumed)` per gate. All figures are
decode row gates only (11008 = 128 tokens × 86; the 860 prefill big gates
are excluded — they average 2–3 ms each and had been inflating the old
`DS4_TP_GATE_STATS` means). GPU wall clock measured at 99.81 MHz against
the host clock (not 100), corrected.

| quantity (per gate) | legacy leg08 | persistent per07 |
|---|---:|---:|
| gate window, mean / p50 / p90 / p99 (leader) | 244 / 209 / 510 / 633 µs | 221 / 210 / 510 / 650 µs |
| gate window, mean / p50 / p90 / p99 (worker) | 253 / 209 / 536 / 751 µs | 238 / 212 / 572 / 753 µs |
| NHI one-way latency `L = (X0+X1)/2` (mean / p50 / p90 / p99) | — | **217 / 215 / 232 / 240 µs** |
| rank skew `|s| = |X0−X1|/2` (mean / p50 / p90 / p99) | — | **191 / 181 / 349 / 509 µs** |
| skew signed mean (+ = worker later) | — | 0.8 µs (symmetric — routing noise, not a slow rank) |
| host: TX-released → service thread noticed | — | ~1 µs (leader) / ~0.3 µs (worker) |
| host: NHI submit ioctl | 14.0 / 14.8 µs | 14.3 / 15.3 µs |
| RX copy (stamp seen → rx_done, 16 KiB from the NHI pool) | — | 8.7 µs |
| RX command already spinning before our TX release | — | yes, by 3.4 ms (host runs ~8 gates ahead) |
| gates where the peer's data arrived before we released | — | 22% |

| per token (86 gates) | legacy leg08 | persistent per07 |
|---|---:|---:|
| gate windows | 21.0 / 21.8 ms (leader / worker) | 19.0 / 20.4 ms |
| compute segments (rx_done[n] → tx_release[n+1]) | 50.5 / 49.7 ms (by subtraction) | **64.8 / 63.4 ms** (measured) |
| token | 71.5 ms | 83.9 ms |

So the persistent kernel did what it was designed to do — ~20 µs/gate off
the window (≈1.5 ms/token, 2%) — and lost ~15 ms/token somewhere in the
compute segments. Candidates (not investigated further; the ceiling was
already known to be ~2%): the resident block holding one CU permanently
(~2.5% of compute ≈ 1.3 ms), the two extra kernel boundaries per gate on
the compute stream (`wait_rx_done`, `mark_src_ready` — each an end-of-kernel
release/dispatch gap), the dispatcher/worker waves' continuous SYSTEM-scope
polling of host-coherent memory competing with compute kernels that already
stream at the LPDDR5X ceiling, and the once-per-token pause/resume around
`ds4_gpu_end_commands` (~0.1 ms).

## Why the premise was wrong

The plan's feasibility check attributed ~38 ms/token to "gate
choreography" from the `DS4_TP_GATE_STATS` mean (425 µs × 86). That mean
(a) included the prefill big gates and (b) on the persistent path dropped
the 22% of gates with a negative window. The decode-only window is ~245
µs, of which 217 µs is the NHI hop as seen GPU-to-GPU (host detect + ioctl
are ~15 µs of that; the rest is inside the tbstream driver / Thunderbolt
DMA / completion) and the remainder is waiting for the slower rank. Both
are outside the gate-kernel's reach: the transport is an explicit
non-goal, and the skew is inherent to splitting routed experts 50/50 by
index (per-token expert counts per rank are uneven; the signed skew
averages to zero, so no static rebalancing helps either). The
`tests/test_tp_nhi_live.c` "35 µs/exchange" figure that suggested a large
gap is a pipelined-throughput number, not one-way latency.

Compute is ~50 ms/token per rank, i.e. the 77 GiB shard's active bytes at
the ~200 GB/s DRAM wall measured in `6c090eb`. Remaining TP decode levers,
for the record: cut L (transport work — out of scope here), cut skew
(routing-aware expert placement, or exchanging at a coarser granularity
so per-layer imbalance averages out), or overlap the gate with
independent compute (the attention gate and the FFN gate of the same
layer are serialized today).

## Pipeline (layer split) vs tensor parallelism on the same pair

Same two nodes, same MXFP4 model, same Manzoni prompt text, run after the
POC was reverted. Pipeline numbers come from the deployed systemd units
(`ds4-mxfp4-server` on max2 / `ds4-mxfp4-worker` on max: `--layers 0:21` /
`22:output`, `--dist-transport auto`, which negotiates TCP over the
Thunderbolt IP link because no `--dist-nhi-device` is passed; build
`/opt/ds4-rebased-20260831`) driven through `/v1/completions` at temp 0
and read back from the server's `/metrics`. TP numbers are the
`scripts/tp_ab.sh` runs above (NHI, 50/50 expert split, `6c090eb`).

| workload | pipeline (deployed, TCP) | tensor-parallel (NHI) |
|---|---:|---:|
| prefill, ~4–5k-token prompt | **242 tok/s** (4859 tok, last chunk 279) | 166 tok/s (4096 tok) |
| prefill, ~200-token prompt | 123 tok/s | 113 tok/s |
| decode, 1024 tokens at small context | **15.08 tok/s** | 13.86 tok/s |
| decode, 128 tokens at ~4–5k context | 14.01 tok/s | 12.93 tok/s |

Pipeline wins on both axes on this hardware: ~1.45× on long prefill (the
chunk pipeline overlaps the two hosts) and ~8–9% on decode. That is the
same conclusion the gate decomposition above points to — TP decode pays
86 gates × (217 µs NHI latency + ~190 µs routing skew) per token, while
pipeline pays one activation hop per token — and matches the PR #861
guidance that pipeline remains the faster two-node deployment. Not a
perfectly controlled comparison: different binaries (`6c090eb` vs the
Aug 31 rebase) and the pipeline requests went through the chat path, so
the model "thought" about the text instead of continuing it.

## Loopback floors (single node, `tests/test_tp_persist_gate_rocm`, in the patch)

Host publish → `rx_done` through the ring, 600-gate burst: 41 µs p50 with
scalar wave copies, **11.5 µs p50** with the float4/8-in-flight copy;
26/26 checks, 10/10 runs green on max2. Kept for reference — it measures
the mechanism, which worked; the mechanism just had nothing to save.

## Notes

- 1609-token prefill point: not captured (prefill rode the legacy big
  gates on both paths, R8).
- Harmless in the logs: `failed to set Linux rocm backend
  oom_score_adj=1000: Permission denied` is the secure-exec side effect of
  the `cap_sys_rawio` file capability the NHI import needs; a relink drops
  the capability (`TBSTREAM_ZC_IMPORT: Operation not permitted`) and
  `scripts/tp_ab.sh` restores it in preflight.
- Raw trace CSVs (`/tmp/tpab/{per07,leg08}-{leader,worker}.trace.csv`,
  ~3.5 MB) were not committed; `tp_gate_trace.py` in this directory
  reproduces the tables from them.
