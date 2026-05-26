# Implementation plan — ds4 + JACCL Distributed Expert Parallelism

**Date:** 2026-05-26
**Companion research:** `~/.claude/projects/-Users-ma-Projects-r1o/memory/research_ds4_jaccl_integration_2026_05_26.md`
**Working branch:** `feat/jaccl-distributed`
**Repo:** `~/opensource/ds4` (fork of antirez/ds4)

---

## Working protocol

Apply dependency-scanner framework before every multi-file edit. See Iron Laws for the non-negotiables this work obeys.

## Architecture

Expert parallelism across N nodes. Each node owns `256/N` experts per layer. All nodes run the full forward pass (attention is replicated, shared expert is replicated). After the routed expert down-projection, a single `all_sum()` over RDMA synchronizes the partial sums. 16KB per layer × 43 layers = ~700KB per token. At 11.7 GB/s RDMA, this adds ~0.06ms per token — negligible.

```
Node 0 (rank 0)                 Node 1 (rank 1)
┌─────────────────┐             ┌─────────────────┐
│ embed            │             │ embed            │
│ attention (full) │             │ attention (full) │
│ router (full)    │             │ router (full)    │
│ experts 0-127    │             │ experts 128-255  │
│ partial_sum      │             │ partial_sum      │
│       └──── JACCL all_sum() ────┘                │
│ shared_expert    │             │ shared_expert    │
│ moe + shared     │             │ moe + shared     │
│ output logits    │             │ output logits    │
└─────────────────┘             └─────────────────┘
```

Each node mmaps the full GGUF — the OS only faults in pages for accessed experts (lazy mmap). No model splitting needed.

## Tech stack

- **Language:** C99 (ds4) + C++20 (JACCL) + ~70 LOC C shim bridging them
- **JACCL:** Standalone lib from `ml-explore/mlx` @ main, pinned to commit `1322065f` (race fix, 2026-05-11)
- **Build:** CMake 3.24+ for JACCL, then link `libjaccl.a` into ds4's Makefile
- **SDK:** macOS 26.2+ (for `<infiniband/verbs.h>`)
- **Runtime:** `librdma.dylib` (dlopen'd), Thunderbolt 5 RDMA enabled

## Tasks

### Phase A — JACCL C Shim + Build Integration

#### Task A1 — Write the C shim header — 10 min

**Edit:** `~/opensource/ds4/jaccl_shim.h` (new file)

```c
#ifndef JACCL_SHIM_H
#define JACCL_SHIM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *jaccl_group_t;

// Dtype enum matching jaccl::Dtype
enum jaccl_dtype {
    JACCL_FLOAT32 = 11, // matches jaccl::Dtype::Float32
    JACCL_FLOAT16 = 9,  // matches jaccl::Dtype::Float16
};

bool     jaccl_is_available(void);
jaccl_group_t jaccl_init_from_env(bool strict);
void     jaccl_group_free(jaccl_group_t g);
int      jaccl_group_rank(jaccl_group_t g);
int      jaccl_group_size(jaccl_group_t g);
void     jaccl_group_all_sum(jaccl_group_t g, const void *in, void *out, size_t n_bytes, int dtype);
void     jaccl_group_barrier(jaccl_group_t g);
void     jaccl_group_send(jaccl_group_t g, const void *buf, size_t n_bytes, int dst);
void     jaccl_group_recv(jaccl_group_t g, void *buf, size_t n_bytes, int src);

#ifdef __cplusplus
}
#endif

#endif
```

**Verify:** File exists, compiles as C header: `cc -fsyntax-only -x c jaccl_shim.h`

**Commit:** `[jaccl] add C shim header for JACCL integration`

#### Task A2 — Write the C++ shim implementation — 15 min

**Edit:** `~/opensource/ds4/jaccl_shim.cpp` (new file)

```cpp
#include "jaccl_shim.h"
#include <jaccl/jaccl.h>
#include <memory>

static std::shared_ptr<jaccl::Group> unwrap(jaccl_group_t g) {
    return *reinterpret_cast<std::shared_ptr<jaccl::Group>*>(g);
}

extern "C" {

bool jaccl_is_available(void) {
    return jaccl::is_available();
}

jaccl_group_t jaccl_init_from_env(bool strict) {
    auto group = jaccl::init(strict);
    if (!group) return nullptr;
    auto *p = new std::shared_ptr<jaccl::Group>(std::move(group));
    return reinterpret_cast<jaccl_group_t>(p);
}

void jaccl_group_free(jaccl_group_t g) {
    if (!g) return;
    delete reinterpret_cast<std::shared_ptr<jaccl::Group>*>(g);
}

int jaccl_group_rank(jaccl_group_t g) { return unwrap(g)->rank(); }
int jaccl_group_size(jaccl_group_t g) { return unwrap(g)->size(); }

void jaccl_group_all_sum(jaccl_group_t g, const void *in, void *out, size_t n_bytes, int dtype) {
    unwrap(g)->all_sum(in, out, n_bytes, dtype);
}

void jaccl_group_barrier(jaccl_group_t g) { unwrap(g)->barrier(); }

void jaccl_group_send(jaccl_group_t g, const void *buf, size_t n_bytes, int dst) {
    unwrap(g)->send(buf, n_bytes, dst);
}

void jaccl_group_recv(jaccl_group_t g, void *buf, size_t n_bytes, int src) {
    unwrap(g)->recv(buf, n_bytes, src);
}

} // extern "C"
```

**Verify:** Compiles with JACCL headers: `c++ -std=c++20 -c jaccl_shim.cpp -I<jaccl-include-path> -fsyntax-only`

**Commit:** `[jaccl] add C++ shim implementation wrapping JACCL Group API`

#### Task A3 — Build JACCL as static lib + integrate Makefile — 15 min

**Pre-flight:** Verify macOS SDK >= 26.2: `xcrun --sdk macosx --show-sdk-version`

**Edit:** `~/opensource/ds4/Makefile` — add JACCL build targets

Add to Makefile (Darwin section):
- CMake configure + build of JACCL into `build/jaccl/`
- Compile `jaccl_shim.cpp` against JACCL headers
- Link `libjaccl.a` + `jaccl_shim.o` into all ds4 binaries
- Conditional: only when `JACCL=1` make flag is set (opt-in, doesn't break default build)

**Verify:** `make clean && make JACCL=1` — all 5 binaries link without error

**Commit:** `[jaccl] integrate JACCL static lib build into Makefile (opt-in JACCL=1)`

#### Task A4 — Standalone shim test — 10 min

**Edit:** `~/opensource/ds4/tests/test_jaccl_shim.c` (new file)

Simple single-process test: `jaccl_is_available()` returns true/false, if available init+rank+size+free cycle. Does not require multi-node — validates linkage.

**Verify:** `make JACCL=1 test_jaccl_shim && ./test_jaccl_shim` — prints availability and exits 0

**Commit:** `[jaccl] add shim linkage test`

### Phase B — Distributed State in ds4 Engine

#### Task B1 — Add distributed state to ds4_engine — 10 min

**Pre-flight:** Read ds4.c:15056-15073 (ds4_engine struct), ds4.h:62-76 (ds4_engine_options)

**Edit:** `ds4.h` — add to `ds4_engine_options`:

```c
bool distributed;      // enable JACCL distributed mode
```

**Edit:** `ds4.c` — add at file scope (conditionally compiled):

```c
#ifdef DS4_JACCL
#include "jaccl_shim.h"
#endif
```

**Edit:** `ds4.c` — add to `struct ds4_engine`:

```c
// Distributed (JACCL)
void *jaccl_group;     // jaccl_group_t, NULL when not distributed
int   world_size;      // total ranks (1 when single-node)
int   rank;            // this process's rank
int   expert_start;    // first expert owned by this rank
int   expert_end;      // one-past-last expert owned
```

**Verify:** `make JACCL=1` compiles

**Commit:** `[jaccl] add distributed state fields to ds4_engine`

#### Task B2 — Initialize JACCL in engine_open — 10 min

**Pre-flight:** Read ds4.c:17985 (ds4_engine_open)

**Edit:** `ds4.c` in `ds4_engine_open()` — after model load, before Metal init:

```c
if (opt->distributed) {
    e->jaccl_group = jaccl_init_from_env(/*strict=*/true);
    if (!e->jaccl_group) { fprintf(stderr, "ds4: JACCL init failed\n"); return -1; }
    e->world_size = jaccl_group_size(e->jaccl_group);
    e->rank = jaccl_group_rank(e->jaccl_group);
    int experts_per_rank = DS4_N_EXPERT / e->world_size;
    e->expert_start = e->rank * experts_per_rank;
    e->expert_end = (e->rank == e->world_size - 1) ? DS4_N_EXPERT : e->expert_start + experts_per_rank;
    fprintf(stderr, "ds4: distributed mode rank %d/%d experts [%d, %d)\n",
            e->rank, e->world_size, e->expert_start, e->expert_end);
} else {
    e->jaccl_group = NULL;
    e->world_size = 1;
    e->rank = 0;
    e->expert_start = 0;
    e->expert_end = DS4_N_EXPERT;
}
```

**Edit:** `ds4_engine_close()` — add `if (e->jaccl_group) jaccl_group_free(e->jaccl_group);`

**Verify:** Single-node: `./ds4 -p "test" --metal` works unchanged. Multi-node: `JACCL_RANK=0 ... ./ds4 --distributed -p "test" --cpu` initializes JACCL and prints rank info.

**Commit:** `[jaccl] init/teardown JACCL in engine lifecycle`

#### Task B3 — Wire --distributed flag through CLI/server — 5 min

**Edit:** `ds4_cli.c` — add `--distributed` flag parsing → `engine.distributed = true`
**Edit:** `ds4_server.c` — same
**Edit:** `ds4.h` — add `bool distributed` to `ds4_engine_options`

**Verify:** `./ds4 --help` shows `--distributed`

**Commit:** `[jaccl] add --distributed CLI flag`

### Phase C — Expert Dispatch Modification (CPU Path)

#### Task C1 — Modify expert accumulation to respect rank ownership — 20 min

**Pre-flight:** Read ds4.c:4375-4432 (`matvec_q2_k_experts_accum_prequant` and its worker)

This is the core change. The inner loop at ds4.c:4378-4386 currently iterates over all 6 selected experts. In distributed mode, each rank only computes experts it owns:

**Edit:** `ds4.c` — modify `matvec_q2_k_accum_worker()`:

```c
// Before: iterate over all n_expert selected
for (int i = 0; i < ctx->n_expert; i++) {
    int expert_id = ctx->selected[i].expert;
    // Skip experts not owned by this rank
    if (ctx->distributed && (expert_id < ctx->expert_start || expert_id >= ctx->expert_end))
        continue;
    float v = 0.0f;
    ds4_vec_dot_q2_K_q8_K(..., &v, ...);
    acc += v;
}
```

Then after the accumulation loop returns, if distributed:

```c
if (engine->jaccl_group) {
    float *tmp = alloca(DS4_N_EMBD * sizeof(float));
    memcpy(tmp, out, DS4_N_EMBD * sizeof(float));
    jaccl_group_all_sum(engine->jaccl_group, tmp, out, DS4_N_EMBD * sizeof(float), JACCL_FLOAT32);
}
```

**Verify:** 
1. Single-node: output unchanged (all experts owned, no all_sum called)
2. Two-node CPU: `JACCL_RANK=0 ... ./ds4 --distributed --cpu -p "hello"` on hub, `JACCL_RANK=1 ... ./ds4 --distributed --cpu -p "hello"` on m3u4 — both produce identical output

**Commit:** `[jaccl] distribute expert accumulation with all_sum (CPU path)`

#### Task C2 — Modify batch expert accumulation — 15 min

**Pre-flight:** Read ds4.c:5979-6003 (batch accumulation path)

Same pattern as C1 but for the batch (prefill) path. The `matvec_q2_k_batch_accum_rows_worker` needs the same rank-ownership filter, and the batch output needs `all_sum()` after accumulation.

**Verify:** Prefill of a multi-token prompt produces identical output across 2 nodes

**Commit:** `[jaccl] distribute batch expert accumulation (CPU prefill path)`

#### Task C3 — Modify gate/up projection to skip non-owned experts — 15 min

**Pre-flight:** Read ds4.c:5740-5768 (gate+up projection in `layer_routed_moe_one`)

The gate+up projection (`matvec_iq2_xxs_experts_mid_prequant`) currently computes all 6 selected experts. In distributed mode, skip experts not owned by this rank (saves compute, not just the reduction).

**Verify:** Same as C1 — identical output across ranks

**Commit:** `[jaccl] skip non-owned expert gate/up projections`

### Phase D — Metal Path (Fused Kernel + all_sum)

The fused Metal kernel `ds4_gpu_routed_moe_one_tensor` writes partial expert sums to `g->routed_out`. No kernel splitting needed — we mask non-owned experts before dispatch, then `all_sum()` the output buffer after.

#### Task D1 — Mask non-owned experts in router_selected buffer — 15 min

**Pre-flight:** Read ds4.c:10474-10492 (fused MoE kernel call) and ds4_gpu.h:639-664 (kernel signature). The kernel reads `router_selected` (int32 array of 6 expert IDs) and `router_weights` (float array of 6 weights).

**Edit:** `ds4.c` — after `ds4_gpu_router_select_tensor()` (line ~10466) and before `ds4_gpu_routed_moe_one_tensor()` (line 10474), add:

```c
// In distributed mode, zero weights for experts not owned by this rank.
// The fused kernel multiplies each expert's contribution by its weight,
// so weight=0 effectively skips it. No Metal shader changes needed.
if (engine->jaccl_group) {
    metal_graph_mask_non_owned_experts(g, engine->expert_start, engine->expert_end);
}
```

`metal_graph_mask_non_owned_experts` reads `g->router_selected` + `g->router_weights` from GPU→CPU (they're StorageModeShared, so zero-copy), zeros weights for expert IDs outside `[expert_start, expert_end)`.

**Note on GPU compute:** Weight masking zeroes the result but does NOT skip GPU dispatch — all 6 expert projections still execute for zeroed experts. This wastes ~50% GPU compute on 2-node. Correctness-first: accept this for now. Compute savings requires Metal shader changes to early-exit on weight=0 (future optimization, not in this plan).

**Verify:** Single-node: no masking (all experts owned). 2-node: output correctness matches single-node within float epsilon.

**Commit:** `[jaccl] mask non-owned experts before fused Metal MoE kernel`

#### Task D2 — Insert all_sum after fused MoE kernel output — 15 min

**Pre-flight:** Read ds4.c:10492-10568 (between MoE kernel and shared+routed addition)

**Edit:** `ds4.c` — after `ds4_gpu_routed_moe_one_tensor` returns (line ~10493), before the `ds4_gpu_add_tensor` or fused HC variant:

```c
// Distributed: synchronize partial expert sums across ranks via RDMA.
// g->routed_out is StorageModeShared (mmap-backed), RDMA-registerable.
// 16KB all_sum at 11.7 GB/s = ~1.4 microseconds — negligible.
if (engine->jaccl_group) {
    // No explicit GPU sync needed — ds4_gpu_routed_moe_one_tensor is synchronous
    // (calls ds4_gpu_finish_command_buffer at ds4_metal.m:14184).
    void *buf = ds4_gpu_tensor_contents(g->routed_out);
    jaccl_group_all_sum(engine->jaccl_group,
                        buf, buf,  // in-place (supported: mesh_impl.h:34 copies if in!=out)
                        DS4_N_EMBD * sizeof(float),
                        JACCL_FLOAT32);
}
```

**Key detail:** `ds4_gpu_routed_moe_one_tensor` already calls `ds4_gpu_finish_command_buffer()` at ds4_metal.m:14184 — it is synchronous. By the time it returns to ds4.c:10493, `g->routed_out` (StorageModeShared) is CPU-readable. No explicit sync needed before the all_sum.

**Verify:** 2-node Metal: output tokens match single-node Metal within float epsilon

**Commit:** `[jaccl] insert all_sum on routed_out after fused Metal MoE kernel`

#### Task D3 — Same pattern for batch (prefill) Metal path — 15 min

**Pre-flight:** Read ds4.c:13316 (`ds4_gpu_routed_moe_batch_tensor` call)

Same mask + all_sum pattern for the batch/prefill Metal kernel. The batch variant processes N tokens, so the all_sum is `N * DS4_N_EMBD * sizeof(float)`.

**Verify:** Multi-token prefill produces identical output across 2 nodes

**Commit:** `[jaccl] distribute batch Metal MoE path`

### Phase E — Launch Script + Integration Test

#### Task E1 — Write a JACCL launch script — 10 min

**Edit:** `~/opensource/ds4/distributed_launch.sh` (new file)

Reads the asmi hostfile format, sets `JACCL_RANK`, `JACCL_COORDINATOR`, `JACCL_IBV_DEVICES` per node, and SSHs to each node to start ds4-server with `--distributed`.

**Verify:** `./distributed_launch.sh --nodes hub,m3u4 --model gguf/ds4flash.gguf --ctx 32768` starts ds4-server on both nodes

**Commit:** `[jaccl] add distributed launch script`

#### Task E2 — End-to-end correctness test — 15 min

**Edit:** `~/opensource/ds4/tests/test_distributed_correctness.sh` (new file)

1. Run a prompt on single-node, capture output tokens + logits
2. Run same prompt on 2-node distributed, capture output tokens + logits
3. Compare: tokens must be identical, logits must match within float epsilon

**Verify:** Script passes on hub + m3u4

**Commit:** `[jaccl] add distributed correctness test`

#### Task E3 — Benchmark: single-node vs distributed — 10 min

Use `ds4-bench` to measure:
- Single-node q4-imatrix on hub (512GB): tok/s prefill + generation
- 2-node distributed q4-imatrix on hub + m3u4: tok/s prefill + generation

Record results, compute overhead percentage.

**Verify:** Distributed overhead < 5% for generation (16KB all_sum should be negligible)

**Commit:** `[jaccl] document distributed benchmark results`

## File touch matrix

| File | Lines added | Lines removed | Notes |
|---|---|---|---|
| `jaccl_shim.h` | ~35 | 0 | New — C shim header |
| `jaccl_shim.cpp` | ~50 | 0 | New — C++ shim impl |
| `Makefile` | ~25 | 0 | JACCL build integration (opt-in) |
| `ds4.h` | ~3 | 0 | distributed flag + engine accessors |
| `ds4.c` | ~80 | ~10 | Engine state, expert dispatch, all_sum calls |
| `ds4_cli.c` | ~5 | 0 | --distributed flag |
| `ds4_server.c` | ~5 | 0 | --distributed flag |
| `distributed_launch.sh` | ~60 | 0 | New — multi-node launcher |
| `tests/test_jaccl_shim.c` | ~30 | 0 | New — linkage test |
| `tests/test_distributed_correctness.sh` | ~40 | 0 | New — e2e correctness |

**Total:** ~420 LOC added, ~10 removed. 10 files touched (6 new, 4 modified). Phase D adds ~90 LOC for Metal path (mask + all_sum + batch).

**Time estimate:** 8-12 hours of focused work. Major time sinks: JACCL CMake extraction (~2h), RDMA debugging on live cluster (~2-3h), batch path complexity (expert-grouped histogram differs from single-token pattern, ~1.5h).

## Risk register

| Risk | Mitigation |
|---|---|
| Metal GPU compute waste | Weight masking zeroes output but doesn't skip dispatch. Accept 50% waste for correctness-first. Shader early-exit is future work. |
| PD exhaustion on repeated launch | One JACCL Group per process lifetime. Never teardown. Script enforces single launch. |
| Expert count not evenly divisible by world_size | expert_end for last rank = DS4_N_EXPERT (takes remainder). 256/2=128, 256/4=64 — clean. |
| KV cache divergence across ranks | Attention is replicated, same input → same KV. Disk KV cache is per-node — independent, no conflict. |
| antirez won't merge | We fork. Our changes are additive (behind --distributed flag). Default build is unchanged. |
| Q4 expert tensors are interleaved differently than Q2 | Both use `tensor_expert_bytes()` for slicing (ds4.c:4179-4180). Pattern is identical. |

## Rollback strategy

| Failure point | Action |
|---|---|
| JACCL build fails | `make clean && make` (no JACCL=1) — default build unaffected |
| Distributed produces wrong output | Remove `--distributed` flag — single-node path untouched |
| PD exhaustion | `shutdown -h` + 60s poweroff on all nodes (known recovery, [[pd-exhaustion-deep-dive-2026-05-14]]) |

## Acceptance criteria

1. `make JACCL=1` builds all 5 binaries with zero warnings
2. `make` (without JACCL=1) still builds — no regressions to default path
3. Single-node `./ds4 --metal -p "test"` output is bit-identical with and without JACCL compiled in
4. 2-node distributed CPU: output tokens match single-node CPU within float epsilon
5. 2-node distributed: `ds4-bench` generation overhead < 5% vs single-node
6. No PD leaks: `jaccl_group_free()` called in engine_close, verified via RDMA metric after run

## Out of scope

- **Metal shader early-exit optimization** — the fused kernel still dispatches all experts (weight=0 zeroes output but doesn't skip compute). Shader-level early-exit on weight=0 is future work.
- **4-node distributed** — 2-node proves the architecture. 4-node is config change only.
- **DeepSeek V4 Pro distributed** — Pro has 384 experts (different count). Test with Flash first.
- **Upstream PR to antirez/ds4** — build as fork first, demonstrate value, then propose.
- **TurboQuant KV cache** — PR #243 not merged yet. Orthogonal to distributed.
- **Ensemble approach** — antirez's preferred method. Complementary, not competing.
