# Qwen Q4_K raw-load register reuse

Measured on 9 October 2026: warmed fresh 16K prefill improved **2.38%** on the
standard text fixture and **2.27%** on C source, with exact old/new output checks.

## Change and provenance

The NAX gate/up and down lookahead loops previously reloaded a Q4_K header and
quant chunk for every 32-weight group. The new next-step loader retains the
header within each 256-weight superblock and the quant words across each
low/high-nibble pair. Initial loads and non-Q4 loaders are unchanged. The
existing dequantizer, arithmetic order, barriers and dispatch geometry are
unchanged. No additional buffer or runtime flag is introduced.

Inspired by the Q4 portion of [StarForge 08ea4d7](https://github.com/Chida82/sf-q3-8flash/commit/08ea4d7eb2e780e2d31690766600880e8dea8718).
Only this register reuse is adopted; the donor's other optimizations are excluded.

The measured host baseline was patched-source commit
`b51ae1ce23a59e5e5f7031c0144e76ac35f13e6c`. It already included
[PLE prefetch #1154](https://github.com/antirez/ds4/pull/1154), common to both arms.
**These gains are incremental to #1154, not a measurement of that PR.** The PR
is based on upstream `fc80bd695da76ee14bcb86820bf900a5cdfda806`; its unmodified
shader, kernel test and Metal host source match the campaign baseline. The
complete upstream host integration was built separately, but was not rerun
through the full model campaign.

Shader SHA256:

- Control: `13224aeae23802defef678877146ae8047a45c2d1be824361efa2f4f5aa25b28`
- Candidate: `dff0b9e48e5d55b68a818d4ccc28ea3468ecd464b8eff83cf426dcb9a102b4cf`

## Measurement

M5 Max, 128 GiB, macOS 26.6.2, resident Qwen3.8 Flash Next Q4 GGUF, Q4_K routed
gate/up and MXFP4 routed down, 8192-token chunks, native context allocation
262144, no batching. Each process warmed weights and a disposable complete
prefill/decode session, freed that session, then measured fresh KV on the same
engine/source. This avoids first-use specialized Metal pipeline compilation.
Only one model process/session ran at a time. Timed runs did not add stage
synchronization or state dumps. Decode measurements use 64 ordinary greedy
tokens; they do not measure MTP throughput.

| Corpus | Runs / balanced blocks | Prefill throughput gain | Conditional 95% block-bootstrap interval | Ordinary decode change |
| --- | --- | --- | --- | --- |
| Standard text, 16384 tokens | 16 / 4 | +2.376% | +2.172 to +2.660% | -0.061% |
| C source, 16384 tokens | 8 / 2 | +2.274% | +2.147 to +2.401% | -0.622% |

Each block uses ABBA or BAAB (A=control, B=candidate). Its ratio is mean B time
/ mean A time; the summary is the geometric mean of block ratios, converted
with `gain = 100 * (1 / ratio - 1)`. The interval enumerates ordered block
resamples with replacement. It is conditional on these few blocks, not a
complete estimate of host uncertainty. A warmed A/A block showed a false
-1.83% difference, and each arm's text prefill times drifted about 6–7%
first-to-last. All four text blocks and both code blocks favored the candidate.
Cold-process results are excluded from this estimate. Do not extrapolate this
gain to every context length.

Historical model SHA256:
`680944460a8cbe93ba8b6d7b6107213ffb7e22320bd913000e563ca0a0f25a8a`.
The 177280286720-byte model was not rehashed for this campaign; its recorded
size, inode and modification time matched the previously verified file.

### Per-run durations

Seconds reconstructed from CSV throughput rounded to two decimal places;
printed precision below is not measurement precision. Each row measures
16384 prefill tokens and 64 ordinary decode tokens.

| Corpus | Run | Arm | Block | Prefill seconds | Decode seconds |
| --- | --- | --- | --- | --- | --- |
| Text | 1 | A | 1 | 10.618895 | 1.210974 |
| Text | 2 | B | 1 | 10.405314 | 1.201652 |
| Text | 3 | B | 1 | 10.576054 | 1.200075 |
| Text | 4 | A | 1 | 10.947627 | 1.197829 |
| Text | 5 | B | 2 | 11.428253 | 1.231954 |
| Text | 6 | A | 2 | 11.613763 | 1.234806 |
| Text | 7 | A | 2 | 11.543149 | 1.238151 |
| Text | 8 | B | 2 | 11.247340 | 1.234806 |
| Text | 9 | A | 3 | 11.488353 | 1.211892 |
| Text | 10 | B | 3 | 11.199748 | 1.226054 |
| Text | 11 | B | 3 | 11.201050 | 1.218351 |
| Text | 12 | A | 3 | 11.421640 | 1.221141 |
| Text | 13 | B | 4 | 11.157419 | 1.220675 |
| Text | 14 | A | 4 | 11.397010 | 1.217193 |
| Text | 15 | A | 4 | 11.355461 | 1.213730 |
| Text | 16 | B | 4 | 11.078879 | 1.218120 |
| Code | 1 | A | 1 | 10.207209 | 1.199850 |
| Code | 2 | B | 1 | 10.157848 | 1.220210 |
| Code | 3 | B | 1 | 10.268109 | 1.225115 |
| Code | 4 | A | 1 | 10.657300 | 1.221840 |
| Code | 5 | B | 2 | 10.232773 | 1.218351 |
| Code | 6 | A | 2 | 10.552621 | 1.219048 |
| Code | 7 | A | 2 | 10.505931 | 1.217193 |
| Code | 8 | B | 2 | 10.331954 | 1.224646 |

## Correctness and regression checks

Campaign checks passed on the patched host baseline and final local integration:

- The focused fixture directly byte-compared 87,492,020 old/new output bytes,
  covering both changed call sites, NAX levels 0–6, Q4/Q4, Q4/MXFP4, all-MXFP4
  and IQ2XXS/Q2_K, K512/2560, signed/zero finite Q4 headers, shuffled routes and
  counts around 8/16/32/64 boundaries. Live values must be finite and nonzero;
  reserved output slots and trailing write sentinels must remain untouched.
- All 24 warmed runs matched each corpus's full 248320-float frontier row.
- Complete persisted payloads at 16384 and 16896 tokens, save/load/resave,
  128 ordinary decode vocabulary rows, token IDs and positions matched exactly.
- Near-native retained prefixes at 131072, 245760 and 246272 tokens matched
  full frontier logits, including the final 512-token append. These are
  correctness observations, not statistically repeated speed measurements.
- `make`, `make test`, the full Qwen kernel suite, PLE rollback/prefetch/failure
  tests, checkpoint/restore/snapshot/rewind with opportunistic MTP, ordinary/MTP
  vision completion tests and synthetic Metal resident/SSD Q4 comparisons passed.
- An isolated 11-request sampled tool/history/restart fixture passed at
  temperature 0.6 with non-exact MTP: 102 verification cycles, 94 drafts accepted.
  This bounded smoke is not a multi-hour agent soak or an MTP speed comparison.

CUDA sources are untouched; no external CUDA or distributed runtime test is
claimed. The finite fixture is not exhaustive over all possible bit patterns.
Serialized state comparisons do not cover every ephemeral MTP scratch buffer.

The existing masked-row loaders can form inactive weight-row addresses past a
65-row logical tensor; this is present in both arms. Output sentinels check
writes, not logical weight-read bounds. Register reuse preserves/subsets those
read addresses; deployed row dimensions are multiples of 64. A bounds repair
needs separate validation and is not included in this patch.

### Reproduce the focused exact comparison

On M5, from the PR checkout, run the same test executable in separate processes
against the baseline and candidate shaders. A fixture pass alone is not the
exactness oracle; compare the complete dumps. Run GPU/model tests only when the
managed server is free and exclusively reserved.

```sh
make tests/test_qwen4_kernels
git show fc80bd695da76ee14bcb86820bf900a5cdfda806:metal/qwen4.metal > /tmp/qwen4-control.metal
DS4_METAL_QWEN4_SOURCE=/tmp/qwen4-control.metal DS4_TEST_QWEN4_RAW16_ONLY=1 DS4_TEST_QWEN4_RAW16_DUMP=/tmp/qwen4-control.raw ./tests/test_qwen4_kernels
DS4_METAL_QWEN4_SOURCE="$PWD/metal/qwen4.metal" DS4_TEST_QWEN4_RAW16_ONLY=1 DS4_TEST_QWEN4_RAW16_DUMP=/tmp/qwen4-candidate.raw ./tests/test_qwen4_kernels
cmp /tmp/qwen4-control.raw /tmp/qwen4-candidate.raw
```

Expected dump size: 87492020 bytes; expected SHA256 for each:
`6cd1882c9ede246f78cad98f0e6d13b63a045a5d0ba332e186e25e57e5dba554`.

Timing used a diagnostic warmed-session harness rather than the stock first
prefill invocation. Raw local manifests, shaders, dumps and generator scripts
are retained outside the PR; the durations above allow the reported timing
ratios to be independently recalculated.
