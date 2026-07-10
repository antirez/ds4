# Plan 003: Make GLM memory estimates match Metal allocations

> **Executor instructions**: Read the whole plan before editing. Verify each
> step and stop rather than weakening the guard if an assumption is false.
> Update Plan 003 in `plans/README.md` when complete.
>
> **Drift check (run first)**:
> `git diff --stat bd89932..HEAD -- ds4.c ds4.h ds4_gpu.h ds4_metal.m tests/ds4_test.c`

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: `plans/002-glm-verification-targets.md`
- **Category**: bug / performance
- **Planned at**: commit `bd89932`, 2026-07-09

## Why this matters

The GLM memory guard and startup report use a compact estimate that omits a
substantial part of the simultaneously live prefill graph. For the fixed GLM
shape at 32768 context, the current report is about 4.38 GiB while the graph
allocation list accounts for at least about 6.13 GiB before lazy global Metal
scratch. The guard also prefers `hw.memsize` and consults Metal's
`recommendedMaxWorkingSetSize` only if host RAM lookup fails, despite placing
the model views in a Metal residency set.

The result is optimistic capacity reporting exactly where a 404-GiB Q4 model
needs trustworthy headroom.

## Current state

- `ds4.c:26507-26512`:

  ```c
  const uint64_t host_bytes = glm_graph_host_memory_bytes();
  uint64_t budget_base = host_bytes;
  if (budget_base == 0) {
      budget_base = ds4_gpu_recommended_working_set_size();
  }
  ```

- `ds4.c:24052-24068` estimates one scratch expression plus compact caches.
- `ds4.c:27578-27668` allocates decode tensors, more than thirty batch tensors,
  and per-layer compact caches simultaneously.
- `ds4_metal.m:249-252` already tracks live and peak tensor allocation bytes
  under `g_tensor_mu`; `ds4_metal.m:2966-2969` snapshots them for reports.
- `ds4_metal.m:1515-1555` registers all non-streaming model views in a residency
  set.
- GLM's fixed shape is declared at `ds4.c:269-310`: 79 layers, 6144 embedding,
  64 heads, 256 experts, 8 active experts, and 1M model context.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Build | `make -j"$(sysctl -n hw.ncpu)"` | exit 0, no warnings |
| Unit tests | `make test-unit` | all pass |
| GLM vectors | `make test-glm-vectors DS4_TEST_MODEL=/path/model.gguf` | quality band unchanged |
| Memory probe | `DS4_GLM_MEMORY_GUARD_REPORT=1 ./ds4 -m /path/model.gguf --ctx 32768 --nothink -n 1 -p test` | report includes planned and actual tracked bytes |

## Scope

**In scope**:

- `ds4.c`
- `ds4.h` only if a public estimate structure must gain fields
- `ds4_gpu.h` and `ds4_metal.m` only for a narrow read-only allocation snapshot
- `tests/ds4_test.c`

**Out of scope**:

- Reducing graph memory through buffer aliasing.
- Changing GLM math, tensor precision, context semantics, or kernels.
- Disabling the memory guard or reducing its reserve.
- CUDA/ROCm policy changes except keeping shared stubs buildable.

## Git workflow

- Suggested branch: `codex/glm-memory-accounting`
- Commit example: `Correct GLM Metal memory accounting`
- Do not push unless explicitly instructed.

## Steps

### Step 1: Create one source of truth for GLM graph allocation sizes

Extract the byte calculations used by `glm_graph_alloc_slice()` into a pure
helper that returns a named allocation plan. The helper must include:

- persistent compact KV and indexer caches for the selected layer range;
- every decode tensor allocated at `27578-27619`;
- every batch tensor allocated at `27621-27654`;
- optional generic IQ2 routed buffers;
- per-layer arrays at `27656-27668`;
- overflow-safe totals and the working/full-attention caps.

Make the allocator consume that plan rather than recomputing independent byte
expressions. Do not create a second estimate that can drift again.

**Verify**: add a pure unit test for the fixed GLM shape at contexts 32768,
100000, 393216, and 1048576. Assert exact named component totals, not only a
loose final range.

### Step 2: Use the complete allocation plan in reports and guards

Replace `glm_graph_context_memory_estimate_for_compact_cap()`'s partial scratch
total with the complete graph allocation total. Preserve the public breakdown:
raw/full KV, compact DSA cache, and non-cache graph buffers. Ensure startup
reports do not double-count caches that are already included.

Use saturating arithmetic consistently. On overflow, refuse startup with a
clear message; never wrap or silently clamp to a plausible small value.

### Step 3: Respect both physical RAM and Metal's recommended working set

For a Metal graph backend, retrieve both `hw.memsize` and
`recommendedMaxWorkingSetSize`. Use the smaller nonzero capacity as the base for
GPU-resident model plus graph allocations, then apply the existing fraction and
reserve. Log both values and the selected base when
`DS4_GLM_MEMORY_GUARD_REPORT=1`.

Keep a documented exception only if measured M3 Ultra behavior proves that
file-backed no-copy model views should not count fully against Metal's working
set. Such an exception must be supported by recorded target-machine allocation
data; do not assume it.

### Step 4: Expose an actual tracked-allocation snapshot

Add a narrow API returning current and peak tracked Metal tensor bytes under the
existing mutex. Provide zero-valued stubs for non-Metal builds. Use it only for
diagnostics and test comparison; the pre-allocation guard must remain based on
the pure plan because actual bytes do not exist yet.

After graph creation, when guard reporting is enabled, print estimated graph
bytes and actual tracked tensor bytes. Define and document which global lazy
scratch buffers are outside the graph plan.

### Step 5: Add estimate-versus-allocation regression coverage

On Metal, allocate a small-context GLM graph through the existing test model
path and assert that tracked graph bytes match the plan within explicitly named
global scratch exclusions. The unit-level exact-size test must remain runnable
without a 434-GB model.

## Test plan

- Pure exact-size tests at four context sizes.
- Overflow tests with deliberately extreme dimensions/caps.
- Metal tracked-versus-planned test on the target model.
- Existing server, kernel, Q4_K, GLM vector, and long-context gates.
- Record the before/after startup report on the M3 Ultra in Plan 005.

## Done criteria

- [ ] The allocator and estimator consume one allocation plan.
- [ ] Every `DS4_GLM_GRAPH_ALLOC_TENSOR` call is represented exactly once.
- [ ] The guard uses the smaller usable Metal/host capacity when both exist, or
      a measured and documented alternative.
- [ ] Overflow refuses startup.
- [ ] Diagnostic output shows estimated and actual tracked graph bytes.
- [ ] Exact-size and overflow unit tests pass.
- [ ] Q2 and Q4 GLM vector quality is unchanged.
- [ ] `make` and `make test-unit` pass without warnings.
- [ ] Only in-scope files and `plans/README.md` are modified.

## STOP conditions

- Metal's reported working-set value on the M3 Ultra is below the resident model
  size even though a measured resident run succeeds; capture the full report and
  stop before choosing policy.
- Matching the estimate requires counting transient command-buffer allocations
  whose lifetime cannot be bounded from the current API.
- A memory-only refactor changes logits or greedy output.
- Shared header changes break CUDA/ROCm builds twice.

## Maintenance notes

Reviewers should compare every future GLM graph allocation addition with the
shared allocation plan. Treat unexplained estimate-versus-tracked drift as a
release blocker on capacity-bound models.
