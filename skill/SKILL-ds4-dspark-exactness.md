---
name: ds4-dspark-exactness
description: "DSpark numerical exactness investigation, experimental evaluation, and benchmark methodology. Use for DSpark verifier correctness debugging, first-divergence probing, exact-row kernel design, accepted-token replay analysis, and speculative decoding performance comparison."
---

# DS4 DSpark Exactness Investigation & Evaluation

Skill for investigating and fixing the DSpark (Dynamic Speculative Decoding) batched verifier numerical drift bug, designing corrective kernels, and evaluating speculative decoding performance on target models.

## When to Use

- User reports wrong tokens / greedy identity issues with DSpark enabled
- Investigating why DSpark is slower than or equal to baseline decode
- Designing an exact-row Metal kernel for compressor pair projection
- Comparing DSpark vs normal decode vs other speculative decoding methods
- Profiling accepted-token replay cost and throughput impact
- Auditing verifier execution order against ordinary single-token decode
- Evaluating speculative acceptance rate and net speedup after fixes

## Core Problem: DSpark Numerical Drift

### The Bug

The batched target verifier in DSpark uses a different floating-point accumulation/tiling order than ordinary single-token decode. Although mathematically equivalent, the ordering difference causes:

1. **Compressor frontier drift** — small float differences compound across layers
2. **Greedy near-tie flips** — drift eventually flips which token wins at argmax
3. **Incorrect checkpoint commits** — committing non-canonical frontier state breaks future decoding

### Current Workaround (Slow but Correct)

Accepted draft tokens are replayed through ordinary single-token decode to restore canonical trajectory:

```
ds4_session_eval_dspark_speculative_argmax
  -> spec_frontier_snapshot        // save compressor state
  -> metal_graph_verify_suffix_tops[_impl]   // batched verifier (fast, non-canonical)
  -> determine commit_drafts
  -> spec_frontier_restore         // restore pre-verifier state
  -> for each accepted draft:
       metal_graph_eval_token_raw_swa(...)     // ordinary single-token decode (slow, canonical)
  -> install replay logits/state
```

Replay cost ≈ one full target decode per accepted token — essentially cancels speculative speedup.

### Empirical Evidence

| Platform | Plain | DSpark | Accept Rate | Replay Cost | Interpretation |
|----------|-------|--------|-------------|-------------|----------------|
| M4 Max / q2-q4 | 29.31 tok/s | 23.02 tok/s | 70.45% | ~34.3 ms/token (one decode) | Replay = baseline decode |
| M4 Max (diff drafter) | — | ~23.8% slower | 90.7% | — | Low acceptance not the main problem |
| M3 Ultra | saved: 3278 ms | replay: 3337 ms | — | replay/saved ≈ 1.018 | Replay cancels savings |

**Conclusion**: The key bottleneck is **not** low acceptance rate — it's that replay cost equals the entire speculative saving. Eliminating replay without breaking greedy identity is the path forward.

## Investigation Workflow

### P1: Stage Table + Pre-GGUF Static Audit

**目标**: GGUF 就绪前完成 ds4.c 静态分析。不依赖模型运行，纯代码审计。commit SHA: b030961。

#### Part A: Stage Comparison Table（P2 Runtime Probe 的前置）

Trace the verifier's encoding stages and compare with the exact N=2 verifier (reference oracle):

```bash
# Locate all DSpark/exactness symbols in ds4.c
grep -n 'symbol_name' ds4.c   # 对每个关键符号运行

# Generate context map (5-line window around each symbol)
python3 hy_extract_dspark_context.py ds4.c > /tmp/dspark_map.txt
```

> **最终目标**: 完成下表，每一格需要明确写出 generic verifier 和 exact N=2 verifier 在该阶段的实际行为差异。调查阶段结束时这张表必须填完。注意：任何算子都不能因为"理论上 stateless"就消失在第一分歧审计中。第一个偏离 canonical trajectory 的算子才是真正的问题所在，而不是预先假设 compressor 是问题。

| Stage                 | generic DSpark verifier          | exact N=2 verifier             | stateful? | 必须 exact? |
| --------------------- | -------------------------------- | ------------------------------ | --------- | ----------- |
| HC preparation        | TODO                             | TODO                           |           |             |
| attention norm        | TODO                             | TODO                           |           |             |
| Q/KV projection       | TODO                             | TODO                           |           |             |
| raw KV store          | TODO                             | TODO                           |           |             |
| compressor projection | TODO                             | TODO                           |           |             |
| compressor update     | TODO                             | TODO                           |           |             |
| indexer compressor    | TODO                             | TODO                           |           |             |
| attention output      | TODO                             | TODO                           |           |             |
| router                | TODO                             | TODO                           |           |             |
| routed MoE            | TODO                             | TODO                           |           |             |
| shared expert         | TODO                             | TODO                           |           |             |
| HC update             | TODO                             | TODO                           |           |             |
| output head           | TODO                             | TODO                           |           |             |

**如何填充**:
1. 在 ds4.c 中定位 `metal_graph_verify_decode2_exact()` — exact N=2 参考 oracle（MTP 用）
2. 在 ds4.c 中定位 `metal_graph_verify_suffix_tops_impl()` -> `metal_graph_encode_layer_batch()` — generic verifier
3. 逐阶段对比：exactifier 保留了哪些操作的原始顺序？batched 了哪些操作？两版本间 mutable state 的 advance 时序是否一致？

**核心问题**: 能否将 N=2 exactifier 的 "split" 策略（只保留关键 stateful 操作为 per-row，其余 batched）从 N=2 推广到 DSpark block size ≤5？

#### Part B: Pre-GGUF Tasks (TASK_BEFORE_GGUF)

在 GGUF 就绪前必须完成的 7 项工作：
1. Run symbol location tools → `/tmp/dspark_context.txt`（符号映射）
2. Read `metal_graph_verify_decode2_exact()` → one-page note（哪些步骤 canonical/rowwise，哪些 batched，row 0→1 mutable state advance）
3. Trace `metal_graph_verify_suffix_tops_impl()` → `metal_graph_encode_layer_batch()` → filled stage table
4. Inspect all existing `*_decode_rows_exact_tensor` implementations → 实现方式总结（each row launch one kernel / single dispatch + row dimension / explicit reduction order / alias batch kernel）
5. Find compressor path: `ds4_gpu_matmul_f16_pair_compressor_store_tensor()` (canonical) vs `ds4_gpu_matmul_f16_pair_tensor()` (batch) → 调用位置和差异分析
6. Write API proposal（不实现 kernel，只写 signature + input/output state/tensor）
7. Integrate `ds4_exactness_probe.[ch]` → zero behavior change to normal inference

Pre-GGUF deliverables: `/tmp/dspark_context.txt`, filled stage table, proposed probe insertion points, proposed exact compressor rows API signature, integrated probe (behavior-off)

### P2: Runtime First-Divergence Probe

**前提条件**: P1 全部完成 + DSpark support GGUF (~5.6 GiB) 就绪。

**目标**: 从同一 pre-speculation snapshot 对比 A（batched verifier）vs B（ordinary sequential decode），找到第一个分歧点。核心问题：**哪个 operator 第一个偏离 canonical trajectory?**

**Probe insertion points per layer/row** (in order — stop at first mismatch):
1. Compressor projected KV — First stateful output from verifier
2. Compressor projected score — Drives compression ratio decisions
3. Compressor mutable KV frontier — Accumulated over verifier rows
4. Compressor mutable score frontier — Same
5. Indexer compressed rows — Affects candidate selection
6. Emitted compressed row — Final output of each verifier row
7. Layer output hidden state — Can amplify earlier drift

**Probe utility**: `ds4_exactness_probe.[ch]` (in handoff bundle). Provides: CPU array comparison, GPU tensor chunked comparison (`ds4_gpu_tensor_read()` from ds4_gpu.h line 55), compact diagnostic output.

Output format (machine-readable):
```
ds4: exactness label=compressor_kv layer=3 row=0 exact=0 first=128 actual=0.123456789 expected=0.123456790 mismatches=42 max_abs=1.2e-6 max_rel=9.8e-6 max_ulp=3
```

**P2 gate** (必须全部满足才能进入 P3):
- stage table 填写完整
- one-page exact N=2 verifier note
- existing exact primitives audit summary
- proposed probe insertion points
- proposed exact compressor rows API signature
- probe helper 编译通过 + CPU comparator 单元测试通过
- source commit SHA（b030961）记录在案

### P3: Exact-Row Kernel Design (if compressor is confirmed)

Only implement if the probe confirms compressor pair projection as the **first** divergence point.

#### Candidate hypothesis only (NOT design principles — P1 audit determines truth)

Potentially batch-safe: operations proven bit-identical by P1 audit

Must remain canonical: any operation whose row-wise/batched implementation is shown
to alter committed or downstream-visible state

Do NOT classify HC / router / MoE / output / compressor before the stage audit is complete.

**Reference patterns** already in ds4 codebase — study before inventing a new API:

```
ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor   // Q8 pair, multi-row
ds4_gpu_matmul_q8_0_decode_rows_exact_tensor         // Q8, multi-row
ds4_gpu_matmul_f16_router_rows_exact_tensor           // F16 router, multi-row
metal_graph_encode_shared_rows_exact                   // shared expert, multi-row
```

**Naming convention**: Match existing `*_rows_exact_tensor` pattern. Candidate:

```c
ds4_gpu_matmul_f16_pair_compressor_decode_rows_exact_tensor(
    ds4_gpu_tensor *dst,           // output: projected KV/score rows
    const ds4_gpu_tensor *src,      // input: query/key pairs
    const ds4_gpu_tensor *weights,  // input: weight matrix
    uint32_t rows,                   // 1..5 (block size)
    uint32_t in_dim,                 // inner dimension
    uint32_t out_dim,                // output dimension
    ...);
```

**Contract**: Must be bit-for-bit equivalent to `rows` repeated canonical single-row calls for N=1..5. No tolerance-based equality.

### P4: Experimental Evaluation (requires DSpark support GGUF ~5.6 GiB)

**Setup**: Official DSpark support model is loaded via `--mtp <ds4flash-gguf> --dspark`, separate from the main model. The 24MB file in .download/ is partial — full download is ~5.6 GiB. Use `make dspark-acceptance` and `make dspark-verify-depth` fixtures (from QA_BEFORE_RELEASES.md).

**Design**: From the same pre-speculation snapshot, compare:

| Run | Configuration | What to measure |
|-----|--------------|-----------------|
| A | Normal DSpark batched verifier (current) | tokens/sec, acceptance rate, replay cost, divergence points |
| B | Ordinary sequential decode of same draft tokens | baseline logits/state reference |
| C | Exact-row verifier (post-fix) | bit-exact match with B, throughput improvement over A |

**Metrics**:
- Acceptance rate (%)
- Net speedup vs plain decode (tok/s ratio)
- Replay time per accepted token (ms)
- Total speculative saving (ms) = saved - replay
- GPU memory bandwidth utilization
- First-divergence layer/row/tensor (for A vs B)

**Benchmark commands**:

```sh
# Throughput comparison
./ds4-bench -m <model.gguf> --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128 \
  --csv /tmp/ds4-speed.csv

# Trace for debugging
./ds4-server --trace /tmp/ds4-trace.txt ...

# Correctness tests
make test                        # all regression tests
./ds4_test --metal-kernels       # Metal kernel numerics
./ds4_test --logprob-vectors     # official continuation comparison
```

**Model override**: `DS4_TEST_MODEL=/path/to/model.gguf ./ds4_test ...`

## Historical PR Context

### PR #590 — Replay-free prefix checkpoints (reference only)

| Concept | Status |
|---------|--------|
| Per-depth prefix checkpoints | Useful design reference |
| Direct checkpoint commit | **Unsafe** without canonical frontier |
| Inline Metal compute copies | Useful pattern |

**Do NOT port wholesale**: later correctness work showed the verifier's batched projection produces non-canonical frontier state. A checkpoint is only safe if its frontier is proven canonical.

### PR #659 — Greedy-identity correctness fix (current baseline)

| Concept | Status |
|---------|--------|
| Remove direct full-accept fast path | Done on current main |
| Force replay through ordinary decode | Current approach (correct but slow) |
| Frontier drift causes near-tie flips | Confirmed diagnosis |

### PR #677 — Byte-exact Metal verifier prototype

**Lesson**: Making the entire verifier serial/exact is too slow. Use it as evidence for a **narrow exact-operator approach**: exactify only the first divergent stateful operation(s), keep everything else batched.

## Do Not Do (Yet)

- ~~Scheduler redesign~~
- ~~Confidence tuning~~
- ~~Sampling / temperature > 0~~
- ~~SSD streaming~~
- ~~ROCm/CUDA work~~
- ~~Full exact verifier rewrite~~
- ~~Direct port of #590's replay-free commit without frontier exactness proven~~

## File Reference (Handoff Bundle)

| File | Purpose |
|------|---------|
| `ds4_exactness_probe.h` / `.c` | Model-independent tensor comparison helpers |
| `test_exact_rows_contract_template.c` | Exact-row kernel test skeleton |
| `hy_locate_dspark_paths.sh` | Locate DSpark symbols in ds4.c |
| `hy_extract_dspark_context.py` | Generate 5-line context map around each symbol |
| `HY_HANDOFF.md` | Full handoff instructions (what to implement now vs later) |
| `CURRENT_MAIN_REFERENCE.md` | Existing exact-row infrastructure reference |
| `EMPIRICAL_FACTS.md` | Prior benchmark data (M4 Max, M3 Ultra) |
| `TASK_BEFORE_GGUF.md` | Checklist of pre-GGUF tasks |
| `PR_REFERENCE_NOTES.md` | Summary of historical PRs |

## Critical Symbols to Know

| Function | Role |
|----------|------|
| `ds4_session_eval_dspark_speculative_argmax` | DSpark main entry point |
| `metal_graph_verify_suffix_tops[_impl]` | Batched target verifier |
| `metal_graph_verify_decode2_exact` | Exact N=2 reference oracle (MTP) |
| `metal_graph_eval_token_raw_swa` | Ordinary single-token decode |
| `spec_frontier_snapshot` / `_restore` | Frontiers save/restore |
| `ds4_gpu_matmul_f16_pair_compressor_store_tensor` | Canonical compressor store |
| `ds4_gpu_matmul_f16_pair_tensor` | Generic batch F16 projection |

## Key Questions to Answer

The central question is NOT "are final logits close?" but:

> **Which tensor/operator is the first place the DSpark verifier leaves the ordinary-decode numerical trajectory?**

Only if compressor pair projection/frontier is confirmed as first divergence should you implement an exact-row Metal kernel.

The test contract for any exact-row kernel:

```
candidate(rows=N) == N repeated canonical single-row calls
```

**byte-for-byte for N=1..5**. No tolerance-based equality accepted.

## Handoff Bundle — Original Files (Appendix)

以下为 handoff bundle 中所有原始文件的完整内容，作为 skill 的附录直接归档。

### HY_HANDOFF.md

```markdown
# HY handoff: DSpark work before the support GGUF finishes downloading

## What you can implement now

Add `ds4_exactness_probe.[ch]` to the repository and wire it into the build only if convenient.
The helper is deliberately model-independent.  It compares GPU float tensors in bounded chunks
and reports:

- first differing element
- first actual / expected value
- total mismatch count
- max absolute difference
- max relative difference
- max ULP difference
- strict bit-exact status

Enable future instrumentation with:

```sh
export DS4_DSPARK_EXACTNESS_PROBE=1
```

Do not enable any behavioral change yet.

## Static code audit to do now

On current `main`, locate and record the current names/line numbers for:

1. `ds4_session_eval_dspark_speculative_argmax`
2. the DSpark batched target verifier
3. ordinary single-token target decode
4. batched compressor KV/score projection
5. ordinary single-token compressor projection/store
6. compressor mutable frontier update
7. frontier snapshot/restore
8. accepted-token replay loop
9. existing prefix-1 checkpoint/capture helpers

Produce a short call graph.

## What the historical PRs established

### PR #590 — replay-free prefix checkpoints

Useful idea:
- capture compressor frontier after each verifier row
- partial accept of k tokens commits checkpoint slot k-1
- this removes one full target replay per accepted token
- it also introduced an inline compute-copy helper to avoid Metal encoder churn

Do NOT port #590 wholesale.

Why: later correctness work showed that the verifier's batched projection numerics can differ
from ordinary single-token decode.  A checkpoint is only safe to commit if the frontier stored
inside it is canonical.

### PR #659 / current correctness direction

This removed the old direct full-accept fast path and forced accepted tokens through ordinary
single-token decode.  The important diagnosis was:

- batched verifier projection and ordinary decode projection are mathematically equivalent
- but their floating-point accumulation/tiling order differs
- compressor frontier therefore drifts
- drift can eventually flip a greedy near-tie

This is why current replay is expensive but correct.

### PR #677 — exact verifier prototype

It demonstrated that byte-identical verifier execution is possible, but making too much of the
verifier decode-order exact was slower than baseline.  Lesson: do not serialize/rewrite the whole
verifier.  Find the first divergent stateful operator and exactify only what is necessary.

## The experiment once the GGUF arrives

Do not start with performance.

From the same pre-speculation snapshot, compare:

A. normal DSpark batched verifier
B. ordinary sequential decode of the same draft tokens

Instrument the earliest state checkpoints possible.

Recommended order per layer/row:

1. compressor projected KV
2. compressor projected score
3. compressor mutable KV frontier
4. compressor mutable score frontier
5. indexer equivalents when present
6. emitted compressed row
7. layer output hidden state

Stop at the first mismatch.

The key question is not "are final logits close?" but:

> Which tensor/operator is the first place the DSpark verifier leaves the ordinary-decode
> numerical trajectory?

Only if the compressor projection/frontier is confirmed as first divergence should you implement
an exact-row Metal projection kernel.

## Exact-row kernel contract

The candidate API may process rows 1..5 in one dispatch, but each row must preserve the same
within-row arithmetic order as the ordinary single-token kernel.

The test contract is strict:

```text
candidate(rows=N) == N repeated canonical single-row calls
```

byte-for-byte for N=1..5.

Do not accept tolerance-based equality for the state that will be committed.

## What not to do yet

- no scheduler redesign
- no confidence tuning
- no sampling / temp > 0
- no SSD streaming
- no ROCm/CUDA work
- no full exact verifier rewrite
- no direct port of #590's replay-free commit until frontier exactness is proven

## Files in this bundle

- `ds4_exactness_probe.h`
- `ds4_exactness_probe.c`
- `test_exact_rows_contract_template.c`
- this handoff

These files are preparation, not a finished DSpark fix.
```

### CURRENT_MAIN_REFERENCE.md

```markdown
# Supplemental offline notes for HY

## Important: use current main as the primary reference

The historical PRs are useful for ideas, but current main already contains
two pieces of infrastructure that are directly relevant to the proposed
DSpark experiment.

### 1. `metal_graph_verify_decode2_exact`

Current `ds4.c` has an exact N=2 target verifier used by MTP.

Its own source comment says the generic batch path is not a safe substitute
for autoregressive decode because row-wise differences in HC/MoE/output
kernels can flip future greedy tokens.  The exact verifier therefore keeps
the ordinary decode kernels and canonical cache-update order while arranging
two tokens layer-by-layer in one command stream.

Study this first.

Questions to answer:

- Which operations are kept rowwise?
- Which operations are safely shared/batched?
- Which mutable states are advanced between row 0 and row 1?
- Can the same split ("batch only stateless/safe work, preserve exact stateful
  operations") be generalized from N=2 to DSpark block size <=5?

Do not copy it wholesale into DSpark: a fully serial exact verifier is known
to be too expensive.  Use it as the correctness oracle / architectural guide.

### 2. Existing exact-row primitives

Current main also contains APIs with names such as:

- `ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor`
- `ds4_gpu_matmul_q8_0_decode_rows_exact_tensor`
- `ds4_gpu_matmul_f16_router_rows_exact_tensor`
- `metal_graph_encode_shared_rows_exact`

These are not the missing compressor solution by themselves.  They show,
however, that ds4 already has an architectural pattern for "multi-row
execution that must preserve decode arithmetic".

Before inventing a new API, inspect how these functions are implemented on
Metal and how their tests establish exactness.

A likely new primitive should match that naming/contract style rather than
introducing a DSpark-specific one-off API.

Candidate concept only:

    ds4_gpu_matmul_f16_pair_compressor_decode_rows_exact_tensor(...)

Do not implement this until the first-divergence probe confirms that the
compressor pair projection is actually the first bad operator.

## Current DSpark replay path to confirm locally

The current shape is:

    ds4_session_eval_dspark_speculative_argmax
      -> spec_frontier_snapshot
      -> metal_graph_verify_suffix_tops[_impl]
      -> determine commit_drafts
      -> spec_frontier_restore
      -> for each accepted draft:
             metal_graph_eval_token_raw_swa(...)
      -> install replay logits/state

The important performance fact is that the replay loop performs one ordinary
target decode per accepted token.

## Static question to answer before GGUF arrives

Trace `metal_graph_encode_layer_batch` from the generic DSpark verifier and
compare it with the exact N=2 verifier.

Make a table:

| Stage | generic suffix verifier | exact N=2 verifier | stateful? | exact required? |
|---|---|---|---|---|
| embedding / HC | | | | |
| attention norm | | | | |
| Q/KV projections | | | | |
| raw KV store | | | | |
| compressor pair projection | | | | |
| compressor update | | | | |
| indexer compressor | | | | |
| attention output | | | | |
| router | | | | |
| routed MoE | | | | |
| shared expert | | | | |
| HC update | | | | |
| output head | | | | |

This table is a better precursor to implementation than guessing the kernel.

## Do not confuse "exact row infrastructure exists" with "DSpark is fixed"

The current DSpark generic verifier still uses `metal_graph_encode_layer_batch`.
Accepted DSpark tokens are still replayed through ordinary decode for strict
greedy identity.

The existence of exact-row helpers elsewhere is useful because it gives us
reference code and conventions, not because replay has already been removed.
```

### EMPIRICAL_FACTS.md

```markdown
# Empirical facts HY can rely on without GitHub access

These are prior community measurements and should be treated as experiment
context, not new results from this branch.

## M4 Max / Metal / q2-q4 / DSpark

One report on M4 Max measured approximately:

- plain: 29.31 tok/s
- DSpark: 23.02 tok/s
- acceptance: 70.45%
- replay: 1063.1 ms for 31 accepted tokens ~= 34.3 ms/token
- baseline decode: ~= 34.1 ms/token

Interpretation: accepted-token replay cost was essentially one ordinary
target decode per accepted token.

A second M4 Max datapoint used a different drafter and reached 90.7%
acceptance, but DSpark was still ~23.8% slower than baseline.  This is strong
evidence that low acceptance alone is not the main problem.

## M3 Ultra

Another report measured:

- saved: 3278.481 ms
- replay: 3336.601 ms
- replay / saved ~= 1.018

Again, accepted-token replay approximately cancels speculative saving.

## Historical replay-free branch (#590)

A replay-free prefix-checkpoint experiment showed that eliminating replay can
recover substantial throughput.  On one mixed Q2/Q4 M5 Max run with a lower
confidence threshold, it measured roughly +27%.

However, later greedy-identity analysis showed that directly committing the
generic batch-verifier frontier is not a safe strict-greedy solution in
general.  Therefore #590 is performance evidence and a checkpoint-design
reference, not a correctness-complete implementation.

## Historical exact verifier experiment (#677)

A broad byte-exact verifier prototype proved exact verification is possible,
but was substantially slower than plain target decode.  This argues against
making the entire verifier serial/exact.

Desired middle ground:

- keep batch execution where numerically safe
- exactify only the earliest stateful divergent operation(s)
- commit canonical verifier state directly
- eliminate accepted-token full replay
```

### TASK_BEFORE_GGUF.md

```markdown
# HY task before DSpark support GGUF is ready

1. Run:
   ```sh
   bash hy_locate_dspark_paths.sh ds4.c
   python3 hy_extract_dspark_context.py ds4.c > /tmp/dspark_context.txt
   ```

2. Read `metal_graph_verify_decode2_exact()` carefully and produce a one-page
   note describing which steps remain canonical/rowwise and which are batched.

3. Trace `metal_graph_verify_suffix_tops_impl()` ->
   `metal_graph_encode_layer_batch()` and produce the stage comparison table
   in `CURRENT_MAIN_REFERENCE.md`.

4. Inspect implementations of all existing `*_decode_rows_exact_tensor`
   functions, especially Metal.  Identify whether they:
   - launch one kernel per row,
   - use a row dimension inside one dispatch,
   - preserve an explicit reduction order,
   - or simply alias a batch kernel known to be exact for that datatype.

5. Find the canonical single-token compressor path around
   `ds4_gpu_matmul_f16_pair_compressor_store_tensor()` and the generic batch
   compressor projection around `ds4_gpu_matmul_f16_pair_tensor()`.

6. Do not implement a DSpark compressor kernel yet.  Write an API proposal and
   identify the exact state/tensor that would be its input/output.

7. Integrate the previously supplied tensor-comparison helper if it builds
   cleanly; otherwise adapt it to local conventions, but keep it behavior-off
   by default.

Deliverables before the GGUF arrives:

- `/tmp/dspark_context.txt`
- exact N=2 vs generic verifier stage table
- proposed probe insertion points
- proposed exact compressor rows API signature
- zero behavior change to normal inference
```

### PR_REFERENCE_NOTES.md

```markdown
# Historical PR references

PR #590 — Replay-free partial accepts + encoder-batched captures
Key concepts: per-depth prefix checkpoints; direct checkpoint commit; inline Metal compute copies.
Do not port wholesale because later greedy-identity work showed non-canonical verifier frontier state.

PR #659 — Greedy-identity correctness fix
Key concept: remove direct commit of batch-verifier state; replay accepted tokens through ordinary
single-token decode to restore canonical compressor/KV trajectory.

PR #677 — Byte-exact Metal verifier prototype
Key lesson: exactness is attainable, but broad decode-order-exact verification was too slow.
Use it as evidence for a narrow exact-operator approach, not as a template to copy wholesale.
```

### ds4_exactness_probe.h (full source)

```c
#ifndef DS4_EXACTNESS_PROBE_H
#define DS4_EXACTNESS_PROBE_H

#include "ds4_gpu.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/*
 * Lightweight, model-independent helpers for DSpark numerical-exactness work.
 *
 * These helpers intentionally take an explicit float count instead of trying
 * to infer tensor shape/size.  That keeps them independent of internal graph
 * metadata and makes them usable at arbitrary verifier checkpoints.
 */

typedef struct ds4_probe_diff {
    bool bit_exact;
    uint64_t first_mismatch;
    uint64_t mismatch_count;

    float first_actual;
    float first_expected;

    float max_abs_diff;
    float max_rel_diff;
    uint32_t max_ulp_diff;
} ds4_probe_diff;

/* Enabled when DS4_DSPARK_EXACTNESS_PROBE is set to a non-empty, non-"0" value. */
bool ds4_exactness_probe_enabled(void);

/*
 * Compare two CPU float arrays.
 *
 * Returns true when the comparison itself completed successfully.
 * result->bit_exact says whether all float bit patterns were identical.
 */
bool ds4_probe_compare_f32_arrays(
        const float *actual,
        const float *expected,
        uint64_t count,
        ds4_probe_diff *result);

/*
 * Read two GPU tensors in bounded chunks and compare them as float32.
 *
 * actual_offset_bytes / expected_offset_bytes must be 4-byte aligned.
 * count is the number of float32 elements to compare.
 *
 * The function avoids allocating a buffer proportional to the whole tensor.
 */
bool ds4_probe_compare_f32_tensors(
        const ds4_gpu_tensor *actual,
        uint64_t actual_offset_bytes,
        const ds4_gpu_tensor *expected,
        uint64_t expected_offset_bytes,
        uint64_t count,
        ds4_probe_diff *result);

/* Print one compact machine-readable-ish diagnostic line. */
void ds4_probe_print_diff(
        FILE *fp,
        const char *label,
        uint32_t layer,
        uint32_t row,
        const ds4_probe_diff *diff);

#endif
```

### ds4_exactness_probe.c (full source)

```c
#include "ds4_exactness_probe.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DS4_PROBE_CHUNK_FLOATS 4096u

static uint32_t ds4_probe_float_bits(float v) {
    uint32_t u = 0;
    memcpy(&u, &v, sizeof(u));
    return u;
}

/*
 * Map IEEE-754 float bits to a monotonically ordered uint32 domain so that
 * integer distance corresponds to ULP distance across negative/positive
 * finite values as well as signed zero.
 */
static uint32_t ds4_probe_ordered_float_bits(float v) {
    uint32_t u = ds4_probe_float_bits(v);
    if (u & 0x80000000u) {
        return ~u;
    }
    return u | 0x80000000u;
}

static uint32_t ds4_probe_ulp_distance(float a, float b) {
    /* NaNs do not have a meaningful ULP distance for this diagnostic. */
    if (isnan(a) || isnan(b)) return UINT32_MAX;

    uint32_t oa = ds4_probe_ordered_float_bits(a);
    uint32_t ob = ds4_probe_ordered_float_bits(b);
    return oa >= ob ? oa - ob : ob - oa;
}

static void ds4_probe_diff_init(ds4_probe_diff *r) {
    memset(r, 0, sizeof(*r));
    r->bit_exact = true;
    r->first_mismatch = UINT64_MAX;
}

bool ds4_exactness_probe_enabled(void) {
    const char *env = getenv("DS4_DSPARK_EXACTNESS_PROBE");
    return env && env[0] && strcmp(env, "0") != 0;
}

static void ds4_probe_accumulate(
        const float *actual,
        const float *expected,
        uint64_t count,
        uint64_t global_offset,
        ds4_probe_diff *r) {

    for (uint64_t i = 0; i < count; i++) {
        uint32_t abits = ds4_probe_float_bits(actual[i]);
        uint32_t ebits = ds4_probe_float_bits(expected[i]);

        if (abits == ebits) continue;

        if (r->bit_exact) {
            r->bit_exact = false;
            r->first_mismatch = global_offset + i;
            r->first_actual = actual[i];
            r->first_expected = expected[i];
        }
        r->mismatch_count++;

        float abs_diff = fabsf(actual[i] - expected[i]);
        if (isfinite(abs_diff) && abs_diff > r->max_abs_diff) {
            r->max_abs_diff = abs_diff;
        } else if (!isfinite(abs_diff)) {
            r->max_abs_diff = INFINITY;
        }

        float denom = fmaxf(fmaxf(fabsf(actual[i]), fabsf(expected[i])), FLT_MIN);
        float rel_diff = abs_diff / denom;
        if (isfinite(rel_diff) && rel_diff > r->max_rel_diff) {
            r->max_rel_diff = rel_diff;
        } else if (!isfinite(rel_diff)) {
            r->max_rel_diff = INFINITY;
        }

        uint32_t ulp = ds4_probe_ulp_distance(actual[i], expected[i]);
        if (ulp > r->max_ulp_diff) r->max_ulp_diff = ulp;
    }
}

bool ds4_probe_compare_f32_arrays(
        const float *actual,
        const float *expected,
        uint64_t count,
        ds4_probe_diff *result) {

    if (!actual || !expected || !result) return false;

    ds4_probe_diff_init(result);
    ds4_probe_accumulate(actual, expected, count, 0, result);
    return true;
}

bool ds4_probe_compare_f32_tensors(
        const ds4_gpu_tensor *actual,
        uint64_t actual_offset_bytes,
        const ds4_gpu_tensor *expected,
        uint64_t expected_offset_bytes,
        uint64_t count,
        ds4_probe_diff *result) {

    if (!actual || !expected || !result) return false;
    if ((actual_offset_bytes | expected_offset_bytes) & 3u) return false;

    ds4_probe_diff_init(result);
    if (count == 0) return true;

    float *a = malloc(sizeof(float) * DS4_PROBE_CHUNK_FLOATS);
    float *e = malloc(sizeof(float) * DS4_PROBE_CHUNK_FLOATS);
    if (!a || !e) {
        free(a);
        free(e);
        return false;
    }

    uint64_t done = 0;
    while (done < count) {
        uint64_t left = count - done;
        uint64_t n = left < DS4_PROBE_CHUNK_FLOATS
                   ? left
                   : DS4_PROBE_CHUNK_FLOATS;
        uint64_t bytes = n * sizeof(float);

        if (!ds4_gpu_tensor_read(
                    actual,
                    actual_offset_bytes + done * sizeof(float),
                    a,
                    bytes) ||
            !ds4_gpu_tensor_read(
                expected,
                expected_offset_bytes + done * sizeof(float),
                e,
                bytes)) {
            free(a);
            free(e);
            return false;
        }

        ds4_probe_accumulate(a, e, n, done, result);
        done += n;
    }

    free(a);
    free(e);
    return true;
}

void ds4_probe_print_diff(
        FILE *fp,
        const char *label,
        uint32_t layer,
        uint32_t row,
        const ds4_probe_diff *d) {

    if (!fp || !d) return;
    if (!label) label = "unnamed";

    if (d->bit_exact) {
        fprintf(fp,
                "ds4: exactness label=%s layer=%u row=%u exact=1\n",
                label, layer, row);
        return;
    }

    fprintf(fp,
            "ds4: exactness label=%s layer=%u row=%u exact=0 "
            "first=%llu actual=%.9g expected=%.9g mismatches=%llu "
            "max_abs=%.9g max_rel=%.9g max_ulp=%u\n",
            label,
            layer,
            row,
            (unsigned long long)d->first_mismatch,
            d->first_actual,
            d->first_expected,
            (unsigned long long)d->mismatch_count,
            d->max_abs_diff,
            d->max_rel_diff,
            d->max_ulp_diff);
}
```

### test_exact_rows_contract_template.c (full source)

```c
/*
 * Exact-row kernel contract skeleton.
 *
 * This is intentionally NOT wired to a hypothetical DSpark kernel yet.
 * HY should replace the two adapter functions below after locating the
 * canonical single-row projection API and implementing the candidate
 * exact-row API on current main.
 *
 * Contract:
 *   candidate(n_rows=N) must equal N repeated canonical single-row executions
 *   bit-for-bit for N=1..5.
 */

#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ROWS 5u

static int run_reference_serial(
        ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *src,
        uint32_t rows,
        uint32_t in_dim,
        uint32_t out_dim) {
    (void)dst;
    (void)src;
    (void)rows;
    (void)in_dim;
    (void)out_dim;

    /*
     * TODO:
     * for row in [0, rows):
     *   create row views
     *   call the SAME canonical projection/store path used by ordinary
     *   single-token decode.
     *
     * Do not substitute a generic GEMM reference.
     */
    return 0;
}

static int run_candidate_exact_rows(
        ds4_gpu_tensor *dst,
        const ds4_gpu_tensor *src,
        uint32_t rows,
        uint32_t in_dim,
        uint32_t out_dim) {
    (void)dst;
    (void)src;
    (void)rows;
    (void)in_dim;
    (void)out_dim;

    /*
     * TODO:
     * one dispatch may process several independent rows, but arithmetic
     * WITHIN EACH ROW must preserve the canonical single-token reduction
     * order.
     */
    return 0;
}

static int compare_bit_exact_f32(
        const float *a,
        const float *b,
        uint64_t count) {
    for (uint64_t i = 0; i < count; i++) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            uint32_t ab = 0, bb = 0;
            memcpy(&ab, &a[i], sizeof(ab));
            memcpy(&bb, &b[i], sizeof(bb));
            fprintf(stderr,
                    "exact-row mismatch index=%llu actual=0x%08x expected=0x%08x\n",
                    (unsigned long long)i, ab, bb);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    /*
     * TODO:
     * 1. initialize GPU
     * 2. build deterministic synthetic input/weights
     * 3. allocate reference/candidate outputs
     * 4. for rows=1..5:
     *      clear outputs
     *      run_reference_serial(...)
     *      read result
     *      clear outputs
     *      run_candidate_exact_rows(...)
     *      read result
     *      require bit equality
     *
     * Prefer synthetic fixtures so this test remains independent of a
     * 90+ GiB GGUF and can run in CI.
     */
    fprintf(stderr,
            "test_exact_rows_contract: scaffold only; wire current-main APIs first\n");
    return 77; /* deliberate SKIP-style value until adapters are implemented */
}
```

### hy_locate_dspark_paths.sh (full source)

```bash
#!/usr/bin/env bash
set -euo pipefail

FILE="${1:-ds4.c}"

if [[ ! -f "$FILE" ]]; then
  echo "usage: $0 [path/to/ds4.c]" >&2
  exit 2
fi

symbols=(
  "ds4_session_eval_dspark_speculative_argmax"
  "metal_graph_verify_suffix_tops_impl"
  "metal_graph_verify_suffix_tops"
  "metal_graph_verify_decode2_exact"
  "metal_graph_eval_token_raw_swa"
  "spec_frontier_snapshot"
  "spec_frontier_restore"
  "spec_frontier_commit_prefix1"
  "ds4_gpu_matmul_f16_pair_compressor_store_tensor"
  "ds4_gpu_matmul_f16_pair_tensor"
  "ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor"
  "ds4_gpu_matmul_q8_0_decode_rows_exact_tensor"
  "ds4_gpu_matmul_f16_router_rows_exact_tensor"
  "metal_graph_encode_shared_rows_exact"
)

echo "# ds4 DSpark/exactness symbol map"
echo "# file: $FILE"
echo

for s in "${symbols[@]}"; do
  echo "## $s"
  grep -n -F "$s" "$FILE" || true
  echo
done
```

### hy_extract_dspark_context.py (full source)

```python
#!/usr/bin/env python3
"""
Generate a compact local code-map without any network access.

Usage:
    python3 hy_extract_dspark_context.py ds4.c > /tmp/dspark_map.txt
"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "ds4.c")
lines = path.read_text(errors="replace").splitlines()

symbols = {
    "DSpark main loop": "ds4_session_eval_dspark_speculative_argmax",
    "Batched verifier": "metal_graph_verify_suffix_tops_impl",
    "Exact N=2 verifier reference": "metal_graph_verify_decode2_exact",
    "Ordinary one-token decode": "metal_graph_eval_token_raw_swa",
    "Frontier snapshot": "spec_frontier_snapshot",
    "Frontier restore": "spec_frontier_restore",
    "Prefix-1 commit": "spec_frontier_commit_prefix1",
    "Canonical compressor projection/store": "ds4_gpu_matmul_f16_pair_compressor_store_tensor",
    "Generic F16 pair projection": "ds4_gpu_matmul_f16_pair_tensor",
    "Existing exact Q8 pair rows primitive": "ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor",
    "Existing exact Q8 rows primitive": "ds4_gpu_matmul_q8_0_decode_rows_exact_tensor",
    "Existing exact router rows primitive": "ds4_gpu_matmul_f16_router_rows_exact_tensor",
    "Existing exact shared rows helper": "metal_graph_encode_shared_rows_exact",
}

def hits(needle):
    return [i for i, line in enumerate(lines) if needle in line]

for title, sym in symbols.items():
    hs = hits(sym)
    print(f"\n=== {title}: {sym} ===")
    if not hs:
        print("NOT FOUND")
        continue
    for i in hs[:8]:
        lo = max(0, i - 5)
        hi = min(len(lines), i + 11)
        print(f"\n--- lines {lo+1}-{hi} ---")
        for j in range(lo, hi):
            print(f"{j+1:6d}: {lines[j]}")
    if len(hs) > 8:
        print(f"... {len(hs)-8} more hits omitted")
```

### handoff bundle 目录结构

```
hy_dspark_complete_handoff/
├── HY_HANDOFF.md                          (handoff 主文档 — 见上方 full source)
├── CURRENT_MAIN_REFERENCE.md              (补充参考: exact N=2 + existing exact primitives)
├── EMPIRICAL_FACTS.md                     (已有 benchmark 数据)
├── TASK_BEFORE_GGUF.md                    (调查前 7 项 checklist — 见上方 full source)
├── PR_REFERENCE_NOTES.md                  (历史 PR #590/#659/#677 摘要)
├── ds4_exactness_probe.h                  (tensor comparison API — 见上方 full source)
├── ds4_exactness_probe.c                  (probe implementation — 见上方 full source)
├── test_exact_rows_contract_template.c    (exact-row kernel 测试骨架 — 见上方 full source)
├── hy_locate_dspark_paths.sh              (符号定位脚本 — 见上方 full source)
├── hy_extract_dspark_context.py           (代码上下文提取脚本 — 见上方 full source)
└── hy_dspark_supplement/
    ├── CURRENT_MAIN_REFERENCE.md          (hy_locate_dspark_paths.sh 的引用副本)
    ├── TASK_BEFORE_GGUF.md                (hy_locate_dspark_context.py 的引用副本)
    ├── hy_locate_dspark_paths.sh
    └── hy_extract_dspark_context.py
```
