# MoE: compact scheduling, 30 September 2026

This experiment changes which independent tile a threadgroup executes. It does
not change arithmetic, weights, activation formats, the K8 MMA sequence, or the
public mid/part layouts. Production files are untouched.

`prepare.py` freezes the current working-tree shader (SHA256 `a83bfe82...`) and
produces the candidate shader and standalone oracle. `schedule-oracle.m` derives
from the September 30 frozen-oracle harness. Unlike that harness's deliberately
reversed active-ID stress order, timed runs use ascending active IDs and the
ordinary tile-major grid, matching the actual M1 SSD production wrapper.

The controls are `--schedule baseline` (same schedule, separate Metal library)
and `--schedule major` (only the already-existing expert-major grid). The new
variants are `compact` (ascending expert ID, then token tile) and
`compact-heavy` (descending frequency, stable expert ID ties, then token tile).
They append one uint32 descriptor `(tile << 9) | original_expert` per nonempty
32-token tile to the read-only lists buffer. The candidate grid is
`[ceil(rows/32), work_count, 1]`. Its unchanged K loop computes exactly one tile;
`tiles_per_launch=n_tokens` is greater than every possible next tile index.

Empty grid dimensions skip dispatch. Existing per-expert count guards remain.
Small correctness fixtures cover float and half operands, T1/7/9/30/33/67/131/
201/439, sparse active experts, and both original launch orders. Large timing
uses captured QWR2 lists in their actual GPU order. Intermediate FP32 outputs,
half conversion, poison, guards, and immutable inputs are checked. CPU worklist
construction time is added to whole-pair candidate timings; stage timings remain
GPU-only. Construction includes ordering, but allocation of a persistent lists
workspace is outside both timed GPU paths, as with their other input buffers.

## Why this is a distinct hypothesis

M1's current default order is not expert-major. Eight persistent groups for each
row-block/expert execute tile `z, z+8, ...`. The work can be uneven, and many
rectangular-grid groups return without useful work. Changing scratch banking,
packing, or dequantization does not test this scheduling hypothesis.

Captured 8192-token routes:

| Layer | Active experts | Compact token tiles | Empty tile/expert cells in rectangle | Longest persistent loop | Descriptor bytes |
|---|---:|---:|---:|---:|---:|
| 9 | 418 | 2792 | 1805 / 3344 | 13 | 11168 |
| 34 | 482 | 2828 | 2035 / 3856 | 10 | 11312 |
| 47 | 289 | 2740 | 1474 / 2312 | 32 | 10960 |

Counts above omit row-block multiplication (20 for mid, 80 for down). Compact
scheduling removes empty dispatches and distributes persistent loops, while the
major-order control isolates cache-locality effects. Extra descriptor storage is
about 11 KiB, rather than model-sized global packing buffers.

For layer 9, the kernel requests approximately 7.576 GB of compressed weights
across its token tiles, representing only 1.134 GB of distinct active expert
weights. This is **logical requested traffic**, not a DRAM measurement: current
caches already serve some reuse. Similarly, nominal X and mid reads are each
8.389 GB/layer because independent output row blocks reuse the operands. These
figures motivate locality work but do not predict a measured speedup.

The XL profile attributes about 12.66 s gross to the big MoE mid/down family;
whole prefill is about 55 s. A standalone +5% phase improvement would require
~2.62 s saved, or roughly +26% whole-MoE throughput before overlap. Small gains
may still combine with the already-tested +3.35% low-memory bundle, but their
percentages must not simply be added.

## Rejected before implementation

A single persistent threadgroup can compute all F640 gate/up rows, retain the
completed half mid in local memory, then produce down. TT8 uses 10 KiB for mid;
TT16 uses 20 KiB, leaving enough of M1's 32 KiB for A/B/C staging. TT32 needs
40 KiB for mid alone. Thus a straightforward exact fusion halves or quarters
weight reuse compared with the current TT32 kernel and reduces cross-row GPU
parallelism. Eliminating the small intermediate is not enough to justify the
increased dequantization and weight traffic. Keeping all D2560 down accumulators
live instead requires 320 KiB at TT32 and risks severe register spills. No
performance claim is made for this unbuilt design.

## Run sequence

Compile only after obtaining the root agent's GPU/compile slot. Run control and
candidate small oracles in fast/safe modes, then paired ABBA/BAAB complete MoE
screens on layer 9. Confirm any positive result on layers 34/47 before model
integration. No test result has been recorded at preparation time.

## Results: positive isolated scheduling screen

The exclusive GPU/compile slot was released after these four screens. Each used
four ABBA/BAAB rounds, eight samples per arm, all samples included, followed by
separate diagnostic stage timings. Full pair includes the existing conversion,
mid and down; compact timings also include CPU schedule construction. Every
screen passed 545,233,920 exact FP32 comparisons. Separate fast/safe small
oracles each passed 20,945,920 comparisons, guards and immutable inputs.

| Variant | Real routing layer | Baseline median ms | Candidate median ms | Throughput gain | CPU schedule ms |
|---|---:|---:|---:|---:|---:|
| Existing expert-major | 9 | 267.651 | 259.146 | +3.28% | none |
| Compact, expert ID order | 9 | 246.065 | 233.482 | +5.39% | 0.006 |
| Compact, heavy first | 9 | 271.556 | 250.269 | **+8.51%** | 0.023 |
| Compact, heavy first | 47 | 248.010 | 225.318 | **+10.07%** | 0.014 |

Absolute baselines differ between series; do not compare candidate milliseconds
across series as though they share a baseline. Means using all samples give
+3.19%, +6.10%, +9.58%, +10.24%, respectively. All four blocks are positive in
each series; heavy-first layer47 blocks give +8.55/+11.96/+10.14/+10.25%.
This is still a short isolated screen with synthetic weights and real recorded
routing, not a whole-model result. Safe math was tested on compact ID order;
heavy-first only reorders independent work items and passed fast numerical
fixtures, but does not yet have its own separate safe run.

The main saving is mid. For heavy-first layer9, separate mid medians are
156.797→146.988 ms, down85.507→83.663 ms. Those stage timings are not additive
with the independently measured full pair.

The prototype reserves a conservative 512 KiB descriptor tail in both arms to
keep all fixture shapes identical; only 11,168/10,960 bytes are written/read for
layers9/47. Thus the raw `extra_candidate_global_bytes=0` is a *paired fixture*
fact, not a claim that integrated descriptors require zero memory. A production
host can size this to the exact count or the tight bound
`4*(ceil(total_pairs/32)+active_experts)`, about12 KiB at T8192/topk10. It needs
explicit workspace lifetime and capacity checks before integration.

Numerical operation order is unchanged. `expert_major==2` and an appended lists
tail are private harness conventions, not a recommended public ABI. Any host
integration must restrict compact descriptors to NT4, force tail_base=0 (the
current tested M1 default), skip empty launches and ensure work_count fits the
Metal grid. Other NT variants, NAX and other devices keep their original path.

`results-summary.json` separates GPU-only samples from pair+CPU samples. The
original JSON field `candidate_gpu_ms` in raw screen files contains CPU worklist
time for compact whole pairs; this inherited name is clarified by the summary.
CPU construction has one short measured sample per fixture, negligible against
the 225–270 ms GPU pair, not a statistical CPU benchmark.

A conservative projection from the older 12.66 s gross MoE family is roughly
1.0–1.2 s saved for an8.5–10% whole-MoE speedup, or around2% of a55 s prefill.
It might combine usefully with the low-memory +3.35% bundle, but neither that
combination nor >=5% end-to-end is demonstrated here. No production patch,
commit, or push was made.

Control limitation: the measured `major` run used the candidate library with its
new compact branch inactive, rather than an independently compiled untouched
candidate library. Arithmetic and selected original block mapping are the same,
but this control does not exclude compiler lowering differences caused by the
extra branch. A future scheduling-only control should pass `baseline.metal` as
both source arguments with `--schedule major`. The positive compact screens
always compare against the untouched original baseline library.
