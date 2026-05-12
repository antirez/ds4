# MTP draft head silently produces NaN on CUDA backend — fd-cache reads MTP offsets from base gguf

**Affects:** `antirez/ds4` CUDA backend with both a base model gguf and the
MTP head gguf (`--mtp` / `ds4_engine_options.mtp_path`) loaded.

**Symptom:** `draft_accept_rate = 0.000` on every measurement, making MTP
appear useless on the CUDA backend. The draft head returns argmax token 0
on every cycle regardless of input, because the MTP transformer block's
intermediate activations contain NaN.

**Severity:** Silent correctness failure that defeats the entire MTP
speculative decoding feature on CUDA. The MTP path runs (no error), the
verifier rejects every draft, the user sees `α=0` and concludes
speculation is dead on this model.

## Repro

```sh
./ds4-bench --cuda \
  -m <base.gguf> \
  --mtp <DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf> \
  --mtp-draft 2 \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 \
  --gen-tokens 16
```

**Before fix:**

```
ds4-bench: mtp spec frontier=7047 draft=2 generated=16 cycles=16
  avg_accepted_per_cycle=1.000 draft_accept_rate=0.000
  full_accept_cycles=0 partial_accept_cycles=0 reject_cycles=15
```

**After the 8-line fix below, same command:**

```
ds4-bench: mtp spec frontier=7047 draft=2 generated=16 cycles=8
  avg_accepted_per_cycle=2.000 draft_accept_rate=0.571
  full_accept_cycles=3 partial_accept_cycles=2 reject_cycles=2
```

The DSv4-Flash MTP head accepts at the maximum rate for `mtp-draft=2`
once it can read its own weights.

## Root cause

The CUDA backend's fd-backed weight cache is pinned to **one** file
descriptor (`g_model_fd`) and **one** file. The engine calls
`ds4_gpu_set_model_fd(base_fd)` once at startup, then calls
`ds4_gpu_set_model_map_range` twice — once for the base map, then for
the MTP map.

After both calls, the state is:

- `g_model_fd` = base gguf's fd
- `g_model_host_base` = MTP gguf's mmap pointer (last winner)

When kernels request MTP weights via
`cuda_model_range_ptr(mtp_model->map, mtp_offset, ...)`, the function
eventually reaches `cuda_model_range_ptr_from_fd(model_map, offset, bytes, ...)`.
That function reads from `g_model_fd` at the requested `offset`,
**regardless of which `model_map` was passed in**. It pread()s the MTP
tensor's offset from the **base** gguf — undefined bytes that the
downstream Q4_K decode interprets as MTP weights, producing NaN.

Bisection trail that confirmed this:

1. `routed_moe_launch` rejects `(gate_type=12, down_type=12)` (Q4_K
   pair). Adding a Q4_K kernel still produced garbage output.
2. A device-side `printf` showed the Q8_K activation block fed to the
   Q4_K kernel had `d = NaN`, all `qs = 0`.
3. Host-side NaN probe at each step of
   `metal_graph_eval_mtp_draft_from_hc` pinpointed the **first** NaN at
   `mtp_enorm` — the RMS-norm output right after the embed.
   `mtp_enorm` had 18 NaN + 3 Inf out of 4096 elements, with values
   ranging from `1.8e-18` to `-3.6e+32` — random bits, not actual
   weights.
4. The RMS-norm reads `mtp->enorm` weight from `model->map + abs_offset`
   where `model->map = mtp_model->map`. The mmap and offset are correct,
   but `cuda_model_range_ptr` was returning a device pointer derived
   from the wrong file (the base gguf at the MTP offset).

## Fix (8 lines, host-base-aware fd-cache)

```c
// ds4_cuda.cu, near the other globals
static const void *g_model_fd_host_base;
```

```c
// ds4_cuda.cu, inside ds4_gpu_set_model_map, right after
//   g_model_host_base = model_map;
//   g_model_device_base = (const char *)model_map;
//   g_model_registered_size = model_size;
//   g_model_range_mapping_supported = 1;
//   g_model_hmm_direct = 0;
//   g_model_cache_full = 0;
if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
    g_model_fd_host_base = model_map;
}
```

```c
// ds4_cuda.cu, at the top of cuda_model_range_ptr_from_fd, after the
// existing `if (g_model_fd < 0 || bytes == 0) return NULL;`
if (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base) {
    return NULL;
}
```

With these in place, MTP reads bypass the fd-cache (which is fd-pinned
to the base gguf) and fall through to the per-tensor `cudaHostRegister`
/ `cudaMalloc` paths in `cuda_model_range_ptr`. Those paths resolve the
host pointer from the actual `model_map` argument, which is correct for
both base and MTP.

## What this fix does not do

It is **correctness-preserving** but not **performance-preserving** for
MTP weight reads. Without the fd-cache, MTP weights take the slower
`cudaMalloc + cudaMemcpy` path. For one-shot weight materialisation
this is fine; for a real production fix the cache should support
multiple fds keyed by host-base.

A complete fix would:

1. Replace `g_model_fd` / `g_model_fd_host_base` with a small
   `std::unordered_map<const void *, int> g_model_fds_by_host_base`.
2. Have `ds4_gpu_set_model_fd_for_map(fd, host_base)` populate it.
3. Have `cuda_model_range_ptr_from_fd(model_map, ...)` look up
   `g_model_fds_by_host_base[model_map]` and use that fd.
4. Have the engine call the new API for both base and MTP at startup.

That preserves the fd-cache performance for both models. The 8-line fix
above is the smaller correctness patch suitable for an immediate
upstream commit.

## Impact

This fix unblocks the **first real measurement of DSv4-Flash MTP
draft acceptance on CUDA**:

- `mtp_draft=2`: `draft_accept_rate = 0.571`,
  `avg_accepted_per_cycle = 2.000`, `full_accept_cycles ≈ 38%` of
  cycles. The drafter is healthy.
- The previously recorded "MTP is useless on DSv4-Flash, α=0" finding
  in the project's optimization plan was a symptom of this bug, not a
  property of the model. Any plan that rejects MTP-style speculation
  for DSv4-Flash on the CUDA backend should be revisited after this
  patch lands.

## Blame

The relevant lines were introduced by:

- `g_model_fd`, `cuda_model_range_ptr_from_fd`, and the cache release
  logic: commit `48beef81` ("Codex" CUDA-support commit).
- `ds4_gpu_set_model_fd`: commit `0ac5df3e` (antirez, "Different
  backends refactoring").

Both commits predate the MTP CUDA glue, so the bug surfaces only when
both a base model and an MTP head are loaded simultaneously on the
CUDA backend — a configuration that has likely been exercised mostly
through Metal (where the cache layer is different) until now.
