# Plan 004: Replace the unsafe GLM Q4 streaming cap

> **Executor instructions**: Run all gates and stop if target measurements do
> not support the policy below. Update Plan 004 in `plans/README.md` when done.
>
> **Drift check (run first)**:
> `git diff --stat bd89932..HEAD -- ds4.c ds4_ssd.c ds4_ssd.h tests/ds4_test.c README.md`

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: `plans/002-glm-verification-targets.md`
- **Category**: performance / bug
- **Planned at**: commit `bd89932`, 2026-07-09

## Why this matters

GLM Metal streaming unconditionally caps its automatic total expert budget at
12 GiB. For the routed Q4 layout, one full layer is about 5.0625 GiB and the
mandatory two-layer prefill reserve is about 10.125 GiB. Only about 1.875 GiB,
or roughly 94 expert entries, remains dynamic. One normal GLM token selects
8 experts across roughly 75 sparse layers (about 600 layer/expert entries), so
the code's own post-configuration warning predicts severe thrashing.

On a 512-GB M3 Ultra, full residency remains the default. If streaming is
explicitly requested, automatic policy must either produce a viable budget or
refuse with an actionable message; it must not silently select a pathological
configuration.

## Current state

- `ds4.c:38133-38144`:

  ```c
  const uint64_t glm_auto_cap_bytes = 12ull * 1024ull * 1024ull * 1024ull;
  if (glm_auto_cap && effective_cache_bytes > glm_auto_cap_bytes) {
      uint64_t capped_experts = glm_auto_cap_bytes / per_expert_bytes;
      ...
  }
  ```

- `ds4.c:3751-3796` reserves two maximum routed layers for prefill.
- `ds4.c:38309-38325` subtracts that reserve from the total expert budget.
- `ds4.c:38878-38893` only warns when the dynamic cache is below twice the
  per-token routed working set.
- The pure generic planner lives in `ds4_ssd.c:80-106` and targets 80% of
  Metal's recommended working set after non-routed weights.
- The fixed GLM shape uses 256 total experts, 8 per token, 79 layers, 3 leading
  dense layers, and one non-executed MTP layer.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Build | `make -j"$(sysctl -n hw.ncpu)"` | exit 0, no warnings |
| Unit tests | `make test-unit` | all pass |
| Streaming vectors | `DS4_TEST_SSD_STREAMING=1 make test-glm-vectors DS4_TEST_MODEL=/path/q4.gguf` | quality band unchanged |
| Streaming smoke | `./ds4 -m /path/q4.gguf --ssd-streaming --ctx 32768 --nothink --temp 0 -n 64 -p test` | viable budget or explicit refusal, no thrash warning |

## Scope

**In scope**:

- `ds4.c`
- `ds4_ssd.c`
- `ds4_ssd.h`
- `tests/ds4_test.c`
- `README.md`

**Out of scope**:

- Changing expert cache eviction algorithms or Metal kernels.
- Making streaming the default on a 512-GB machine.
- Reducing prefill headroom below two full routed layers.
- Hiding an inadequate budget by removing warnings.
- General CUDA/ROCm retuning.

## Git workflow

- Suggested branch: `codex/glm-q4-streaming-budget`
- Commit example: `Use viable GLM streaming cache budgets`
- Do not push unless explicitly instructed.

## Steps

### Step 1: Express viability as a pure cache-plan constraint

Add overflow-safe helper logic that calculates:

- prefill headroom bytes;
- one-token routed working set:
  `sparse_layers * experts_used * per_expert_bytes`;
- preferred dynamic cache: twice that working set;
- minimum total automatic expert budget:
  `prefill_headroom + preferred_dynamic_cache`.

For uniform GLM Q4 at the current fixed shape, the expected minimum is about
34.4 GiB. Assert exact integer byte arithmetic in tests; do not assert rounded
log strings.

### Step 2: Remove the unconditional 12-GiB cap

Use the general 80%-working-set plan, then apply model-specific viability:

- If the general plan can satisfy the viable minimum, use at least that minimum
  and no more than the general plan.
- If it cannot, refuse automatic streaming with required and available bytes
  plus guidance to use full residency, a smaller quant, or an explicit expert
  count for diagnostics.
- Do not silently expand past Metal's recommended budget.

Apply the same accounting after optional full-resident prefix-layer selection;
full layers must not consume the required dynamic minimum.

### Step 3: Turn the late warning into a planning invariant

Keep a warning for explicit expert-count budgets because the user intentionally
bypasses byte accounting. Automatic byte budgets must never reach inference
below the viability threshold. Distinguish in logs between total expert budget,
prefill reserve, full-layer bytes, dynamic experts, one-token working set, and
the selected safety multiple.

### Step 4: Add pure planner regression tests

Cover at least:

- current GLM Q4 sizes: auto plan is at least the viable minimum;
- Q2 and IQ2 sizes: no regression or overflow;
- recommended working set too small: clean refusal;
- explicit expert count below working set: accepted with warning;
- mixed-precision layers: viability uses only cache-compatible routed layers
  and reports bypass layers separately;
- multiplication/addition overflow.

### Step 5: Validate quality and throughput on the M3 Ultra

Compare full residency, old 12-GiB-equivalent explicit configuration, and the
new auto plan using identical model, prompt, context, greedy sampling, power,
and thermal state. Record hit rate, pread bytes/time, prefill t/s, generation
t/s, and first-token/logit equivalence. Preserve byte-identical output across
cache sizes.

## Test plan

- Pure planner tests with exact current-shape byte values.
- Existing streaming cache-pressure regression.
- GLM vector scorer full residency versus streaming.
- 100K long-context streaming smoke if the resulting budget is viable.
- Before/after benchmark artifacts consumed by Plan 005.

## Done criteria

- [ ] No unconditional 12-GiB GLM cap remains.
- [ ] Automatic Q4 streaming either provides at least the viable dynamic cache
      or fails before inference.
- [ ] Logs show all budget components and the working-set threshold.
- [ ] Explicit diagnostic counts preserve their existing semantics.
- [ ] Planner overflow and capacity tests pass.
- [ ] Full-residency and streaming GLM quality remain in the same accepted band.
- [ ] New auto Q4 streaming materially improves hit rate and throughput over the
      old 12-GiB-equivalent configuration, or the plan is marked BLOCKED with
      measurements showing that streaming is not worthwhile.
- [ ] `make` and `make test-unit` pass.
- [ ] Only in-scope files and `plans/README.md` are modified.

## STOP conditions

- A viable cache cannot fit below Metal's recommended working set on the target.
- Larger dynamic caches change logits rather than only performance.
- The two-layer prefill reserve is not simultaneously live in the current
  implementation; measure and report before changing the formula.
- Fixing the budget requires an eviction-policy or kernel rewrite.

## Maintenance notes

Any future GLM layer/expert shape or quant change must rerun the pure viability
tests and the target benchmark. A fixed GiB cap is not portable across quants.
