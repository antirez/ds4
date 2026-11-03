# IQ1_M routed experts on gfx1151

Why this branch exists: the ISTA-DASLab GSQ-RCO 1017 release is the only
DeepSeek-V4.1-Flash-1017 quantisation small enough to be interesting on a
122.2 GiB unified pool — about 112.44 GiB for the IQ1_M main weights, against
246 GiB for the smallest alternative (mrfakename Q2_K) and 311.80 GiB for
unsloth UD-Q2_K_XL. Its routed experts are IQ1_M, which ds4 refuses to load.
This is upstream issue #1268.

## What already works, and what does not

The gap is narrower than it looks from the kernel symbol counts. Verified in
this tree:

| piece | state |
|---|---|
| `block_iq1_m` layout | present, `cuda/mmq/ggml-common.h:423` |
| `vec_dot_iq1_m_q8_1` | present and `__device__`, `cuda/mmq/vecdotq.cuh:1223` |
| `VDR_IQ1_M_Q8_1_MMVQ` / `_MMQ` | both defined, `cuda/mmq/vecdotq.cuh:1220` |
| generic ggml mat-vec | instantiated, `cuda/mmq/mmvq.cu:1018` |
| `ds4_mmq_vdr_mmvq_value<IQ1_M>` | **added on this branch** |
| `ds4_mmq_vec_dot_q8_1` IQ1_M arm | **added on this branch** |
| `ds4_mmq_iq1_m_moe_vec` entry point | **added on this branch** |
| `DS4_TENSOR_IQ1_M`, `block_iq1_m` in `ds4.c` | **added on this branch** |
| CPU `ds4_dequantize_iq1_m_block` | **added, tested** |
| CPU `ds4_vec_dot_iq1_m_q8_K` | **added, tested** |
| CPU `matvec_iq1_m_experts_{mid,accum}_prequant` | **added, wired into both dispatchers** |
| `load_tiles_iq1_m` for tiled MMQ | **missing, and hard — see below** |
| `tensor_is_routed_expert_type` admits IQ1_M | **deliberately not yet** |
| `ds41_stream_table` per-role typing | **not yet** |
| GPU graph dispatch to `ds4_mmq_iq1_m_moe_vec` | **not yet** |

So GPU *decode* for IQ1_M is reachable with small changes, and the CPU
reference path is complete. GPU *prefill* through the tiled MMQ path is not.

Nothing loads yet: `tensor_is_routed_expert_type` is the gate and it stays shut
until the paths behind it are all in place, so the tree is never in a state
where it accepts a model it cannot run.

## Why there is no `load_tiles_iq1_m`

`load_tiles_iq1_s` exists (`cuda/mmq/mmq.cuh:3308`) and shares the
`iq1s_grid_gpu` codebook, so the grid-lookup half of an IQ1_M loader is nearly
a copy. The scales are the problem. Compare the blocks:

```c
block_iq1_s { ggml_half d; uint8_t qs[32]; uint16_t qh[8];  }
block_iq1_m {              uint8_t qs[32]; uint8_t  qh[16]; uint8_t scales[8]; }
```

IQ1_M has **no `d` field**. Per `vec_dot_iq1_m_q8_1`, the superblock scale is
reassembled from one nibble in each of four `uint16`s:

```c
scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0)
          | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
```

and the sub-scales are two 3-bit fields per 32-element group, i.e. **one scale
per 16 values**:

```c
const int tmp = sc[iqs/2] >> (6*(iqs%2));
sc0 = 2*((tmp >> 0) & 0x07) + 1;   // first 16
sc1 = 2*((tmp >> 3) & 0x07) + 1;   // second 16
```

while the delta sign lives in bit 3 of each `qh` nibble, i.e. **one delta per 8
values**:

```c
delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f*IQ1M_DELTA/0x08);
```

The MMQ tile carries one `half2` per 32-element group — `(d, d*delta)` for
IQ1_S, where delta is uniform across the 32. IQ1_M needs scale granularity of
16 and delta granularity of 8, so one `half2` per 32 cannot represent it.

The obvious dodge does not work either: the sub-scales are odd integers
`2k+1` up to 15, and the grid values occupy the full 0..15 nibble range, so
folding the sub-scale into the stored values overflows the `int8` lanes that
`dp4a` consumes. A correct loader therefore needs a **new tile layout** with
finer scale storage, following the precedent of the K-quant loaders rather
than `load_tiles_iq1_s`. That is the one genuinely novel piece of work here,
and it is why upstream llama.cpp has an IQ1_S MMQ loader but not an IQ1_M one.

## What GSQ-RCO actually contains

Read from the GGUF header of shard 1 (a 48 MiB range request, no full
download). Expert dimensions match what ds4 expects for deepseek41 —
gate/up `[5120, 2304, 384]`, down `[2304, 5120, 384]`, i.e. `DS4_N_EMBD`
5120, `DS4_N_FF_EXP` 2304, `DS4_N_EXPERT` 384.

The important part is that **precision is mixed within a layer**:

| layer | gate | up | down |
|---|---|---|---|
| blk.0 - blk.6 | IQ1_M | IQ1_M | IQ1_M |
| blk.7, blk.8 | IQ1_M | IQ1_M | **IQ2_XXS** |
| blk.9 | IQ1_M | IQ1_M | IQ1_M |
| blk.10 | IQ2_XXS | IQ2_XXS | IQ2_XXS |
| blk.11 | IQ2_XXS | IQ2_XXS | **IQ1_M** |

So gate and up agree with each other in every layer seen, but `down` varies
independently in both directions. `ds41_stream_table` currently demands one
exact combination:

```c
if (t[i]->type != (i == 2 ? DS4_TENSOR_Q2_K : DS4_TENSOR_IQ2_XXS) || ...
```

which is wrong for GSQ-RCO twice over — `down` is never Q2_K in it, and
gate/up may be IQ1_M. The fix is to type the three roles independently against
a permitted set rather than against one expected pair. The surrounding
machinery already tolerates `down` differing from gate/up, since that is
exactly the V4.1 Q2 shape (IQ2_XXS gate/up, Q2_K down), so this is a widening
of an existing distinction rather than a new one.

## Upstreaming: where this can go

Tested by cherry-picking this branch onto `antirez/main` (`ee7e840`) in a
throwaway worktree. The first commit applies clean; the second conflicts on
`Makefile`. Of the fourteen files touched, nine are new and transfer as-is,
and exactly five are modifications to files that have diverged between
`antirez/main` and this branch's base (`d32febf`):

| file | main vs base | this branch adds |
|---|---|---|
| `ds4.c` | 3196+/2967- | 327+ |
| `Makefile` | 143+/365- | 20+ |
| `cuda/mmq/ds4_mmq.cu` | 113+/773- | 21+ |
| `cuda/mmq/ds4_mmq.h` | 22+/0- | 15+ |
| `.gitignore` | 11+/27- | 2+ |

So a PR against `antirez/main` is feasible but not a cherry-pick: 385 lines of
integration, all additions, re-applied by hand. They are small and localised —
the type enum, the block struct and its static assert, the
`routed_expert_block_bytes` case, two dispatcher arms, one template
specialisation, one `if constexpr` arm, one entry point and two Makefile
rules — with the bulk of `ds4.c`'s 327 being self-contained new functions.

Against a fork already carrying the ROCm work it applies as-is.

## Fatal: ds4 cannot open the GSQ-RCO file at all, and IQ1_M is not why

Everything below this section is still accurate about IQ1_M, but it is not the
binding constraint. GSQ-RCO fails at model open, before any expert type is
examined, because it packages Engram differently from the V4.1 Q2 release.

`model_unmap_engram()` runs unconditionally at `ds4.c:2950`, straight after
`parse_tensors`, with no resident/streaming and no CPU/GPU distinction:

```c
if (!model_get_string(m, "general.architecture", &arch) ||
    !ds4_streq(arch, "deepseek41")) return;              /* not us: skip */
if (!model_get_string(m, "deepseek41.engram.encoding", &encoding) ||
    !ds4_streq(encoding, "e4m3_e8m0_32_row264"))
    ds4_die("unsupported V4.1 Engram row encoding");     /* <- dies here */
```

The architecture *is* `deepseek41`, so the early return does not fire. Read
from shard 1's header, GSQ-RCO has these Engram keys and **no
`engram.encoding` among them**:

```
deepseek41.engram.file            = DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
deepseek41.engram.layer_ids       = [2 items]
deepseek41.engram.num_embeddings  = [2 items]
deepseek41.engram.present         = 1
deepseek41.engram.source          = 1bae0e1088993b9b82de08f59de5be5a34e42170
```

So the first `model_get_string` fails and ds4 dies. The mismatch is structural,
not a missing string:

| | V4.1 Q2 (works today) | GSQ-RCO 1017 |
|---|---|---|
| Engram location | inline in the model mapping (~189 GiB of its 341 GiB) | separate file, named by `engram.file` |
| row encoding | `e4m3_e8m0_32_row264` | IQ4_NL |
| index tensors | `blk.{1,14}.engram_embd.weight`, type 24 (I8), `dim[0] == 264` | `engram_token_map` (I32), `engram_primes`, `engram_offsets`, `engram_multipliers` (I64) |

That last row is the deep part. ds4 expects two dense I8 tables of 264-byte
rows living in the main model's tensor list; GSQ-RCO ships a hash-based scheme
(primes, offsets and multipliers over a compressed 99,092-entry token map) in a
separate IQ4_NL file. Supporting it means an external-Engram loader, an IQ4_NL
row decoder and a different lookup algorithm — a new Engram subsystem, not an
added case, and needed whether the model runs resident or streamed.

This also closes the resident idea. Residency would have bypassed the expert
cache's uniform-slot-size rule described below, but not this: the engram check
runs before any of that and does not care how the model will be executed.

### It is the whole 1017 generation, not one quantiser

Checked `mrfakename/DeepSeek-V4.1-Flash-1017-GGUF` `Q2_K` the same way, from
shard 1's header. It is a plain Q2_K requantisation with no exotic expert
types at all, and it **also** has no `deepseek41.engram.encoding`, so it dies
in `model_unmap_engram` just the same. Its Engram index is the same hash
scheme, carried as metadata arrays rather than tensors:

```
deepseek41.engram.primes         = [48 items]
deepseek41.engram.offsets        = [48 items]
deepseek41.engram.multipliers    = [8 items]
deepseek41.engram.token_map      = [129280 items]
deepseek41.engram.head_count     = 8
deepseek41.engram.key_length     = 256
deepseek41.engram.max_ngram_size = 4
```

It carries no `engram.file` key, so unlike GSQ-RCO its Engram weights are not
external; they are in a later shard. The two releases therefore differ from
each other in where the tables live, but agree in replacing the
`e4m3_e8m0_32_row264` encoding and the dense I8 264-byte-row index with a
hashed one.

So the conclusion is broader than this branch: **no 1017 checkpoint loads in
this engine**, whatever its quantisation, because the 1017 generation changed
the Engram metadata and indexing scheme and ds4 at `d32febf` understands only
the earlier one. That is presumably the substance of upstream issue #1224
("Support 10/17 update"), which is a larger change than a download-URL
refresh.

The flip side: the work is not GSQ-RCO-specific. Hashed Engram support would
unlock every 1017 variant at once, and the plain Q2_K one needs no new expert
kernels — only 246 GiB of disk against the 341 GiB in use today.

### Standing

The IQ1_M work on this branch (CPU dequantiser, q8_K dot product, both expert
matvecs, the `_moe_vec` entry point, 13/13 tests plus GPU parity) is correct
and is a usable contribution toward issue #1268, since IQ1_M support is wanted
independently. It does not make any 1017 checkpoint loadable, and nothing
short of the Engram work above will.

## Blocker: the expert cache wants one slot size for every layer

This is the finding that sets the real cost of the project, and it is not a
type-switch problem.

A 232 GiB model cannot be resident on a 122.2 GiB pool, so GSQ-RCO has to go
down the SSD streaming path. On ROCm that path is mandatory, not best-effort:
`ds41_stream_cache_configure` (`ds4.c:70312`, under
`DS4_ROCM_BUILD && DS4_HAS_DEEPSEEK41_GPU`) fails the engine open outright.

```c
if (!ds41_stream_table(&e->model, &e->weights.layer[il], il, &table)) {
    fprintf(stderr, "ds4: V4.1 ROCm SSD streaming requires uniform "
                    "IQ2_XXS/Q2_K experts (layer %u)\n", il);
    return false;
}
const uint64_t bytes = 2u * table.gate_expert_bytes + table.down_expert_bytes;
if (expert && expert != bytes) return false;
```

There is no CPU fallback here — the engine refuses to open — and the second
check requires **every layer to have the same expert slot size**. The slot
sizes GSQ-RCO actually presents, computed from the per-role types above (the
row-byte arithmetic agrees with `v41_mmq_topology`'s own hardcoded 1320 and
756, so these are sound):

| layers | gate/up, down | `2*gate + down` bytes |
|---|---|---|
| blk.0-6, blk.9 | IQ1_M, IQ1_M | 7,741,440 |
| blk.7, blk.8 | IQ1_M, IQ2_XXS | 8,202,240 |
| blk.10 | IQ2_XXS, IQ2_XXS | 9,123,840 |
| blk.11 | IQ2_XXS, IQ1_M | 8,663,040 |
| *(V4.1 Q2 today)* | *IQ2_XXS, Q2_K* | *9,953,280* |

Four distinct sizes where the allocator admits one. Supporting GSQ-RCO on the
streaming path therefore means reworking the expert cache to handle per-layer
variable slot sizes — a change to the slot allocator and its reservation logic
(`ds4_gpu_stream_expert_cache_reserve_layers`, the two-layer admission at
`ds4.c:73927`), not an added case.

There is also a second, separate gap: **the ROCm MoE does not use the
`cuda/mmq` `_moe_vec` entry points at all.** `routed_moe_build_plan`
(`rocm/ds4_rocm_moe_launch.cuh:537`) enumerates five `(gate, down)` pairs and
returns 0 for anything else:

```c
plan->q4k_path     = (gate_type == 12u && down_type == 12u);
plan->iq2_path     = (gate_type == 16u && down_type == 10u);
plan->iq2_iq2_path = (gate_type == 16u && down_type == 16u);
plan->q2k_path     = (gate_type == 10u && down_type == 10u);
plan->mxfp4_path   = (gate_type == 39u && down_type == 39u);
```

GSQ-RCO needs three more — (29,29), (29,16) and (16,29) — each backed by ROCm
kernels in `rocm/ds4_rocm_moe*.cuh`. So `ds4_mmq_iq1_m_moe_vec`, added earlier
on this branch, serves the CUDA path and the parity harness but is **not** what
the ROCm production MoE calls.

### What this means for the plan

The CPU work (dequantiser, q8_K dot, both matvecs) is correct and tested and
remains the foundation for anything else, but it does not by itself produce a
runnable GSQ-RCO: the streaming gate rejects the model before any expert
matvec is reached.

The cheaper remaining option is worth weighing against the slot-allocator
rework. GSQ-RCO's main weights are 112.44 GiB, which is **under** the 122.2 GiB
pool, so a resident configuration would bypass `ds41_stream_cache_configure`
entirely and with it the uniformity requirement. That trades a large allocator
change for a very tight memory budget, and it needs measuring rather than
assuming — see the DSpark arithmetic at the end of this file.

## Trap: two grid conventions

The codebook exists in two forms and they are off by one, so transcribing the
GPU arithmetic onto the CPU table silently biases every value by -1.0.

| table | value alphabet | paired delta |
|---|---|---|
| `iq1s_grid_gpu` (uint32, nibbles) | `{0, 1, 2}` | `-1 +/- IQ1M_DELTA` |
| `iq1s_grid` (uint64, bytes) | `{-1, 0, +1}` | `+/- IQ1M_DELTA` |

The `-1` in the GPU delta exists only to cancel the `+1` offset baked into
`iq1s_grid_gpu`. Entry 1 pins the relation: it is `0x00000002` in the GPU table
and `0xffffffffffffff01` in the CPU one, so `g_gpu = g_cpu + 1`.

This was caught by `tests/test_iq1_m_dequant.c` rather than by reading, and it
is worth noting that one of that test's cases initially agreed with the buggy
code because the error happened to cancel for `g_cpu = 0`. Any future IQ1_M
work should keep a case with `g_cpu = +1`, where nothing cancels.

## Plan

Staged so that each step is independently verifiable, cheapest first.

1. **Vector path in the mmq TU** — done on this branch. Four lines:
   `ds4_mmq_vdr_mmvq_value<GGML_TYPE_IQ1_M>` plus an `if constexpr` arm in
   `ds4_mmq_vec_dot_q8_1`. Six `ds4_mmq_moe_*_vec_impl` sites read the former,
   so this is what makes an IQ1_M expert instantiation possible at all. Note
   that template specialisations are only instantiated on use, so compiling
   this TU proves syntax and symbol resolution, not behaviour.

2. **`DS4_TENSOR_IQ1_M` plumbing in `ds4.c`** — add `= 29` to the enum at
   `ds4.c:2360`, extend `tensor_is_routed_expert_type()` (`ds4.c:5085`) and
   `routed_expert_block_bytes()` (`ds4.c:5096`, 56 bytes). The type table at
   `ds4.c:2351` already carries `{"iq1_m", 256, 56}`.

   Scope warning: **58 sites in `ds4.c` hardcode `DS4_TENSOR_IQ2_XXS`**, and
   they are not one switch. `scripts/map_iq2_sites.py` maps each to its
   enclosing function: 58 sites in 33 functions, grouping as follows. (It
   attributes the enum definition itself to `align_up`, which is a harmless
   artefact of scanning backwards for the nearest definition.)

   **Other architectures, out of scope** (10 functions): everything named
   `glm_*` (`8377`, `8479`, `16712`, `16720`, `16729`, `47669`, `47753`) and
   `qwen4_*` (`59254`, `59260`, `68736`).

   **On the deepseek41 ROCm path, in scope** (21 functions). Note that the
   `metal_graph_*` prefix is misleading and does **not** mean Metal-only: the
   family includes `metal_graph_cuda_tp_*` helpers that read `DS4_CUDA_TP_*`
   environment variables, and `metal_graph_decode_cuda_selected_slots_expected`
   is called from the CUDA/ROCm decode path at `23455`, `23528`, `26818` and
   `26852`. Treat it as the shared graph scheduler. The group covers
   `metal_graph_*` (`20202`, `20265`, `20301`, `22684`, `22759`, `22805`,
   `22853`, `24299`, `29391`, `32951`), the CPU `matvec_*_prequant` dispatchers
   (`10854`, `11729`, `11754`, `11776`, `11831`), `layer_routed_moe_batch`
   (`13626`, `13734`), `ds41_stream_table` (`41134`),
   `ds41_cuda_row_batch_supported` (`43268`) and `ds4_engine_open_internal`
   (`72574`, `72618`).

   **Deferred**: `ds4_session_dspark_seed_batch_enabled` (`61957`, `61967`).

   Most of the in-scope entries are `*_supported` / `*_expected` / `*_eligible`
   predicates that gate a fast path, so returning false for IQ1_M is the
   correct behaviour and they need no change — the generic path takes over.
   The ones that must be extended are the handful that otherwise reach a
   `ds4_die`: the `matvec_*_prequant` dispatchers, `layer_routed_moe_batch`,
   `ds41_stream_table` and `ds4_engine_open_internal`.

3. **Correctness before any tile work** — done for the CPU reference.
   `tests/test_iq1_m_dequant.c` checks the layout against hand-derived values
   and is wired into `test` and `test-rocm`.

   GPU parity is also covered now. `cuda/mmq/test/test_mmq_parity.cu` had
   **no Makefile target at all**, so nothing checked those kernels numerically
   on ROCm; `tests/test_mmq_parity_rocm` adds one, and an `IQ1_M/MOE_VEC` case
   exercises `ds4_mmq_iq1_m_moe_vec` against a host dequantiser.

   That host dequantiser needs its own copy of the codebook, because the GPU
   build of `ggml-common.h` declares the grid `__device__` — the same reason
   `iq2_host_tables.h` exists. `scripts/gen_iq1_grid.py` grew a namespace mode
   and emits `cuda/mmq/test/iq1_host_tables.h` from the same source as the
   engine-side table, so the two cannot drift apart.

   `test_mmq_soa_tiles.cu` still has no target; it covers the IQ2_XXS SoA
   tiles, which IQ1_M does not use.

4. **`load_tiles_iq1_m` plus a finer-grained tile layout** — the hard step
   above. Only worth starting once 1-3 hold.

5. **gfx1151 fast paths are out of scope.** The SoA repack, D2R fusion and the
   `cc == GGML_CUDA_CC_OFFSET_AMD + 0x1151` special case are gated on
   `type == GGML_TYPE_IQ2_XXS` (`cuda/mmq/ds4_mmq.cu:1402, 1512, 1520, 1727`).
   IQ1_M will take the generic path, so expect it to be slower per byte than
   IQ2_XXS despite moving ~13% fewer bytes. Matching those is a second project.

## What this does and does not buy

It makes GSQ-RCO loadable on the SSD-streaming path: roughly 232 GiB on disk
(112.44 main + 103.00 Engram-IQ4_NL) against the 341 GiB
`DeepSeek-V4.1-Flash-Q2.gguf` in use today, at better claimed quality.

It does **not** obviously unlock resident DSpark, and the arithmetic is worth
stating because it is closer than it first appears. `docs/STRIX_HALO.md:113`
gives 151.76 GiB of resident target spans plus 7.42 GiB of support model for
V4.1 Q2. Substituting the GSQ-RCO main weights gives 112.44 + 7.42 =
**119.86 GiB against a 122.2 GiB pool** — about 2.3 GiB left, and that figure
is explicitly "before context and runtime buffers". The DSpark example in that
doc uses `--ctx 8192`, which is small for a reason. Two things are unverified
and either would sink it: whether runtime buffers fit in what remains, and
whether the pinned MXFP4 support GGUF is a valid draft for a **1017** target
when it was released against V4.1. Treat resident DSpark as one measurement to
make, not as the goal.
