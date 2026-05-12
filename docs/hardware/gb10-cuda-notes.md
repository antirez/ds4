# GB10 CUDA Hardware Notes

This file is the durable notebook for DS4 CUDA work on the DGX Spark / GB10
machine. Keep hardware-specific observations here so future optimization work
does not need to rediscover them from benchmark logs.

## Machine Identity

- GPU reported by CUDA: `NVIDIA GB10`.
- CUDA compute capability: `sm_121`.
- CUDA-visible unified memory pool: about `121.63 GiB`.
- Prior design snapshot recorded driver `580.142`, CUDA `13.0`, and `ncu`
  installed. Re-check these before relying on profiler behavior.
- The q2 GGUF tensor staging footprint is about `80.76 GiB` upstream, and about
  `82,697.65 MiB` in the local CUDA branch layout.

## Profiling Safety

- Do not run full-model `ncu --set full` on this machine. A full-model Nsight
  Compute replay locked/OOMed the box and required a physical reboot.
- Use Nsight Systems (`nsys`) for full-model timing.
- Use Nsight Compute (`ncu`) only on small synthetic fixtures, or with a narrow
  metric set and `--launch-count 1`.
- If full-model profiler work is unavoidable, run it alone, with memory
  monitoring active, and with no parallel agent or benchmark load.

## CPU Topology

- The high-frequency CPU set observed for GB10 host work is:
  `5,6,7,8,9,15,16,17,18,19`.
- `DS4_CUDA_CPU_AFFINITY=auto` in the local branch pins the CUDA host thread to
  those CPUs.
- F1 CPU pinning was effectively performance-neutral for warm decode:
  pinned and disabled runs were both around `7.380 tok/s`.
- Treat CPU affinity as a stability/control feature, not a major throughput win.

## Memory And Page Cache Behavior

- Cold staging can push the system near direct reclaim if the model mmap and CUDA
  device allocations coexist without page dropping.
- E4/E6 showed that incremental `POSIX_FADV_DONTNEED` plus mapping page eviction
  is useful for stability and cold-staging speed. It keeps page cache from
  ballooning around the full model footprint during staging.
- `MADV_PAGEOUT` was useful in the local E6 branch to widen MemFree headroom.
- A pre-E6 run saw MemFree get OOM-adjacent. Later E6-style runs kept MemFree in
  a much safer range.
- Direct NVMe read floor measured with `dd if=ds4flash.gguf of=/dev/null bs=64M
  iflag=direct` was about `15.95s`, or roughly `5.4 GB/s`, for the model file.

## CUDA Mapping And Registration

- Whole-range `cudaHostRegister` failed on the full mapped tensor range with
  `operation not supported`.
- Direct host/mmap inference is correct enough to test, but too slow for default
  decode on this stack.
- Fast probe result for direct mmap:
  - `DS4_CUDA_STAGE=direct`
  - load/stage: about `0.322-0.329s`
  - decode: about `3.620 tok/s`, roughly `49%` of the then-current bounce
    baseline.
- Nsight Systems showed direct mmap regression is concentrated in MoE routed
  kernels:
  - MoE routed: `5.01x` slower
  - HC/projection: `1.73x` slower
  - Attention: `1.53x` slower
  - Indexer/compressor: `1.51x` slower
  - Output head: `0.43x` faster
- Interpretation: full direct mmap is not a default path. A hybrid path is only
  worth revisiting if it can avoid direct-mapped access for the regressing MoE
  weight class.

## Cold Staging Results

Historical local branch progression:

- Initial production staging: `314.283s` measured staging for `82,697.65 MiB`.
- Pinned 64 MiB bounce buffer default: `88.092s` for `82,697.65 MiB`.
- E6 tuned path with `DS4_CUDA_STAGE_PINNED_PARALLEL_FILL=1 DS4_THREADS=12`:
  about `23.9-24.2s`.
- E5a pinned-size sweep best observed value: `23.517s` with `512 MiB` pinned
  staging, not promoted at the time.

Current upstream `antirez/ds4` at `ae302c2`:

- Tiny cold probe:
  - command shape: `./ds4 --cuda -m ds4flash.gguf --ctx 4096 -n 16 ...`
  - staged `80.76 GiB` in `19.130s`
  - reported generation: `16.57 tok/s`
  - total wall: `25.01s`
- `ds4-bench` 7047-token frontier:
  - staged `80.76 GiB` in `19.787s`
  - prefill: `339.56 tok/s`
  - generation: `13.58 tok/s`
  - total wall: `52.12s`
- Same-day local tuned comparison:
  - staged `82,697.65 MiB` in `23.926s`
  - tiny probe generation: `8.27 tok/s`
  - total wall: `28.14s`

Conclusion: upstream's fd-backed device cache plus arena allocation and page
dropping supersedes the local E6 staging path.

## Decode And Prefill Observations

- Early local CUDA baseline: `1.73 tok/s` generation.
- Local warm decode after first CUDA optimization pass: about `7.2-7.4 tok/s`.
- Local short prompt prefill after that pass: about `11.7-13.2 tok/s`,
  depending on run and harness.
- Explicit small-batch prefill test showed partial amortization:
  - B1 decode: `7.420 tok/s`, `134.77 ms/token`
  - B2 prefill: `12.20 tok/s`, `81.97 ms/token`, `0.608x` B1 per-token cost
  - B4 prefill: `12.75 tok/s`, `78.43 ms/token`, `0.582x`
  - B8 prefill: `13.21 tok/s`, `75.70 ms/token`, `0.562x`
- This means batched prefill helps, but does not yet amortize enough by itself
  to make tree/speculative decode a guaranteed `45 tok/s` path.
- External GB10 note from a separate private branch reported about `12 tok/s`
  decode, memory-bandwidth-limited around `270 GB/s`, with prefill closer to M3
  Max at roughly `200 tok/s`.
- Upstream now reports and locally measured roughly `13.6 tok/s` generation at
  a 7047-token frontier.

## Decode Bandwidth Floor (2026-05-11 investigation)

Re-confirmed baseline on fresh upstream `ae302c2`, GB10, q2 model:

- `ds4-bench --ctx-start 7047 --ctx-max 7047 --gen-tokens 128`:
  cold staging `18.867s`, prefill `343.56 tok/s`, generation `13.68 tok/s`.
- Current-tree inertness check, same command with profiling env unset:
  cold staging `19.510s`, prefill `343.21 tok/s`, generation `13.63 tok/s`.
- `--ctx-start 64 --ctx-max 64 --gen-tokens 64`: generation `15.31 tok/s`.
  Long-ctx attention/indexer adds only `~8 ms/token`, so even free attention
  caps the upside at about `15.3 tok/s`.

`DS4_GPU_DECODE_STAGE_PROFILE=1 DS4_GPU_DECODE_STAGE_PROFILE_LIMIT=16`
breakdown at 7047 ctx, sampled across 16 decode tokens and summed across 43
layers per token:

| Stage             | ms/token | Notes                                       |
| ----------------- | -------- | ------------------------------------------- |
| routed_moe        | 15.72    | gate/up/down for 6 active experts (iq2_xxs) |
| attn_output       | 14.28    | output_a (grouped q8) + output_b + hc_expand|
| compressor_indexer| 10.51    | only ratio-4 layers, indexer fires at long ctx |
| q_path            | 9.56     | q_a + kv + q_b q8 matmuls + RMS + RoPE      |
| attention         | 6.24     | indexed/static/online mixed kernels         |
| attn_hc_pre+ffn_hc_pre | 6.80| pre-attn and pre-FFN HC split               |
| shared_gate_up+down    | 6.14| shared expert q8 fused gate/up swiglu       |
| router                 | 1.64| router select + softmax                     |
| output logits          | 2.41| final Q8 vocab projection                   |
| kv_path + norms + posts| ~2.4| FP8 KV quantize + raw store + small norms   |
| **sum**           | **~74.3**| matches 13.5 tok/s                          |

The aggregate profile reports `74.274 ms/token`, with `74.262 ms/token`
accounted by stage boundaries plus readback and only `0.012 ms/token`
residual. This profiler inserts sync boundaries, so it is an attribution tool,
not a headline throughput run. The non-profiled run above remains the speed
baseline.

The measured gap to `20 tok/s` is about `23.1 ms/token`:

- Current: `13.68 tok/s` = `73.1 ms/token`.
- Target: `20 tok/s` = `50.0 ms/token`.
- Required reduction: `~23.1 ms/token`, or `~32%` of decode wall time.

The seven dominant kernels in pure decode (`moe_gate_up_mid_decode_lut_qwarp32`,
`moe_down_sum6_qwarp32`, `grouped_q8_0_a_preq_warp8`, `matmul_q8_0_preq_warp8`,
`matmul_q8_0_hc_expand_preq_warp8`, `matmul_f16_pair_ordered_chunks`,
`matmul_q8_0_pair_preq_warp8`) are all already-specialized decode paths and
are bandwidth-bound on LPDDR5x. Toggle tests confirm:

- `DS4_CUDA_INDEXED_TWOPASS=1`: 13.64 tok/s (no change).
- `DS4_CUDA_DISABLE_QKV_RMS_FUSED=1`: 13.58 tok/s (~0.3% regression — the
  fused QKV RMS kernel is barely a lever).

Active params read per decode token across attention + shared + 6 routed
experts is roughly `4-5 GB`. At ~`70 GB/s` effective LPDDR5x bandwidth that
floor is ~`60-70 ms/token`, which matches measured `73.5 ms/token`. The
nominal `270 GB/s` figure is peak; effective achievable bandwidth is closer
to a quarter of that for this access pattern.

### Implications for future decode work

- Single-token decode is at the GB10 bandwidth wall. No micro-fusion of the
  remaining serial decode kernels has more than `~0.5%` upside, and most
  candidates (eliminate one `quantize_q8_0_f32` launch, pair `q_a`+`kv`,
  inline `low` quantize into output_b) are well under that.
- The top exact-decode candidates by measured cost are:
  - **MoE routed experts**: `15.7 ms/token`. Worth investigating only if a
    targeted rewrite can reduce bytes read or improve effective bandwidth by
    several ms/token; small launch fusions are not enough.
  - **Attention output + q_path**: `23.8 ms/token` combined. This is the
    largest non-MoE exact-decode surface, but much of it is Q8/Q8-HC bandwidth.
  - **Compressor/indexer**: `10.5 ms/token`, only present at long ctx. This is
    the best long-context-specific candidate and can move the 7047-token
    frontier without helping short-context decode.
- Practical paths above `~14 tok/s` require reducing data read per token:
  - **Speculative decoding / multi-token batched forward.** Existing batched
    prefill kernels at B=8 already show `~13.2 tok/s` effective per-token cost
    near the B=1 cost (`75.7 ms` vs `134.8 ms` historically). A speculator
    loop reusing those kernels is the highest-upside path, potentially
    `~1.5-2x` for low-temperature generation.
  - **Long-ctx-only**: indexer/compressor stage is `10.5 ms/token` at 7047
    ctx and decomposes mostly into `indexer_scores_wmma` (WMMA-shaped, n_tok=1
    underutilizes it) plus the f16 pair compressor projection. A decode-only
    indexer scoring kernel could recover a portion, but only at long ctx.
- CUDA Graph capture is not currently a primary lever on this path. The
  boundary profiler's residual is effectively zero; a separate Nsight Systems
  run can still bound launch overhead without using full-model `ncu`.

### Speculation probe

`ds4-bench --spec-probe` measures a no-risk prefix-retrieval drafter against
the actual greedy token stream and reports an upper-bound effective tok/s. It
does not change decode behavior or claim real speculative throughput.

Measured on the fixed 7047-token frontier:

| Probe | Generation | Upper-bound effective tok/s | Result |
| ----- | ---------- | --------------------------- | ------ |
| `--spec-ngram 4 --spec-draft 4` | `13.65 tok/s` | `13.65 tok/s` | no useful hits |
| `--spec-ngram 2 --spec-draft 6` | `13.69 tok/s` | `13.80 tok/s` | `~0.8%` upper-bound |

Conclusion: simple prompt-prefix retrieval is not the speculation path to
`20 tok/s` on this benchmark.

### MTP speculative benchmark

The optional MTP GGUF is now present on this machine:

```sh
/home/cghart/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
```

Antirez uploaded a replacement imatrix Q2 GGUF on 2026-05-11 with the same
upstream filename but different content:

```sh
DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

Hugging Face metadata from the files page, re-confirmed through the HF model
API on 2026-05-11:

- repo commit: `b0c3326275d2207e25e42bc8ac0704952466b5bb`
- file size: `86720111488` bytes
- SHA-256 / HF `x-linked-etag`:
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
- Xet hash: `38ccf413bd121a82b9023228e4f49e9557fca645a75c1628addcd9f9e2938430`
- commit note: "Replace imatrix GGUF with fixed routed-mid imatrix build"

Download target on this machine, kept separate from the existing baseline:

```sh
/home/cghart/ds4/gguf/fixed-imatrix-b0c3326/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

Do not repoint `/home/cghart/ds4/ds4flash.gguf` until the new imatrix file has
completed download and has its own baseline, exact replay, and MTP acceptance
measurements. The current symlink intentionally remains on the older
`chat-v2.gguf` baseline.

Fixed-imatrix measurements on 2026-05-11:

```sh
./ds4-bench --cuda \
  -m /home/cghart/ds4/gguf/fixed-imatrix-b0c3326/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 128
```

| Model | Context | Prefill | Generation | CUDA startup cache |
| ----- | ------- | ------- | ---------- | ------------------ |
| old `chat-v2.gguf` symlink baseline | 7047 | `~343 tok/s` | `~13.6 tok/s` | `~19s` |
| fixed imatrix Q2 | 7047 | `343.76 tok/s` | `13.43 tok/s` | `27.991s` first local run |
| fixed imatrix Q2, verified `b0c3326` | 7047 | `343.58 tok/s` | `13.45 tok/s` | `19.283s` |

Exact replay on fixed imatrix:

```sh
./ds4-bench --exact-replay-probe --cuda \
  -m /home/cghart/ds4/gguf/fixed-imatrix-b0c3326/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 32
```

| Model | Context | Tokens | Exact tok/s | Replay tok/s | Exact replay match | First mismatch |
| ----- | ------- | ------ | ----------- | ------------ | ------------------ | -------------- |
| fixed imatrix Q2 | 7047 | 32 | `13.36` | `13.39` | no | 0 |

First mismatch on fixed imatrix exact replay: first stream picked token `1526`,
replay picked token `1008`.

MTP on fixed imatrix:

```sh
DS4_MTP_STRICT=1 DS4_MTP_RESTORE_DRAFT_FRONTIER=1 ./ds4-bench --cuda \
  -m /home/cghart/ds4/gguf/fixed-imatrix-b0c3326/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --mtp /home/cghart/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --mtp-draft 2 \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start FRONTIER --ctx-max FRONTIER --gen-tokens N
```

| Context | Tokens | Draft | Effective tok/s | Exact tok/s | Avg accepted/cycle | Draft accept rate | Exact match | First mismatch |
| ------- | ------ | ----- | ---------------- | ----------- | ------------------ | ----------------- | ----------- | -------------- |
| 64 | 64 | 2 | `14.30` | `14.04` | `1.000` | `0.000` | yes | -1 |
| 7047 | 32 | 2 | `12.76` | `13.46` | `1.000` | `0.000` | no | 0 |

Conclusion for the fixed imatrix update: it is useful to keep as the likely
better-quality model artifact, but it did not change the early speculative
decoding go/no-go. These fixed-imatrix rows predate the later deterministic
attention/snapshot correction, so use the corrected symlink-baseline MTP sweep
below for the active go/no-go decision.

Benchmark support added to `ds4-bench`:

```sh
./ds4-bench --cuda -m /home/cghart/ds4/ds4flash.gguf \
  --mtp /home/cghart/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --mtp-draft N \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 128
```

The MTP mode reports target verifier cycles/sec, accepted tokens/sec, average
accepted tokens per verifier cycle, draft acceptance rate, rollback/partial
rate, exact-match status, and exact baseline tok/s. For the exactness oracle,
the benchmark generates an exact reference stream, invalidates the session,
full-prefills the same prefix again, then measures MTP from a clean frontier.

An `--exact-replay-probe` diagnostic was added to separate MTP effects from
plain CUDA replay determinism:

```sh
./ds4-bench --exact-replay-probe --cuda -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 128
```

Exact replay results:

| Context | Generated | Exact tok/s | Replay tok/s | Exact replay match | First mismatch |
| ------- | --------- | ----------- | ------------ | ------------------ | -------------- |
| 64      | 16        | `15.19`     | `15.20`      | yes                | -1 |
| 7047    | 128       | `13.49`     | `13.47`      | no                 | 19 |

The 7047 mismatch reproduced without loading or probing the MTP model:
`first=31833`, `replay=36127` at step 19. This makes the current MTP failure a
long-context CUDA replay/exactness blocker first, not primarily a drafter
quality result.

Follow-up top-logit diagnostics show the mismatch is not stable across replays:

| Run | First mismatch | First pass top/margin | Replay top/margin | Readout |
| --- | -------------- | --------------------- | ----------------- | ------- |
| top-1 | 2 | `269` over `3152` by `0.229324` | `3152` over `269` by `0.804474` | not a tiny replay tie |
| top-2 | 0 | `1526` over `1008` by `0.000948` | `1008` over `1526` by `0.193447` | first pass begins as a near tie, replay diverges immediately |

Interpretation: repeated full-prefill greedy decode at 7047 is not stable enough
to serve as an exact speculative-verification oracle. DDTree or any other
tree/speculative decoder remains blocked until this long-context replay
determinism issue is understood.

Component isolation with existing CUDA toggles:

| Probe | Env | Tokens | Replay match | First mismatch | Tok/s | Readout |
| ----- | --- | ------ | ------------ | -------------- | ----- | ------- |
| default | none | 32 | no | 19 | `13.40-13.44` | reproduces |
| no window attention | `DS4_CUDA_NO_WINDOW_ATTENTION=1` | 32 | no | 0 | `13.42-13.44` | not the fix |
| no cuBLAS attention | `DS4_CUDA_NO_CUBLAS_ATTENTION=1` | 32 | no | 5 | `13.39-13.46` | not the fix |
| slow indexer scoring only | `DS4_CUDA_NO_INDEXER_WMMA=1 DS4_CUDA_NO_INDEXER_DIRECT_ONE=1` | 32 | no | 22 | `11.88-11.92` | not sufficient |
| slow top-k only | `DS4_CUDA_NO_TOPK1024=1 DS4_CUDA_NO_TOPK2048=1 DS4_CUDA_NO_TOPK_CHUNKED=1` | 32 | no | 19 | `1.14-1.15` | not sufficient |
| slow indexer scoring + slow top-k | all five indexer/top-k disables above | 32 | yes | -1 | `1.13` | deterministic but too slow |
| slow indexer scoring + slow top-k | all five indexer/top-k disables above | 16 | yes | -1 | `1.14` | repeat confirmation |
| deterministic indexer switch | `DS4_CUDA_DETERMINISTIC_INDEXER=1` | 16 | yes | -1 | `1.13` | single-env equivalent |
| deterministic scorer + rank top-k | `DS4_CUDA_DETERMINISTIC_INDEXER=1 DS4_CUDA_DETERMINISTIC_TOPK_RANK=1` | 16 | yes | -1 | `11.82` | short pass only |
| deterministic scorer + rank top-k | same | 32 | no | 19 | `11.80-11.84` | rejected |

This narrows the exactness issue to the interaction of fast long-context indexer
score generation and fast top-k selection. The fallback path is deterministic,
but its `~1.1 tok/s` speed is not a production route. The next engineering task
is a deterministic CUDA indexer/top-k implementation or a focused fixture that
compares selected compressed rows across repeated full-prefill replays.

`DS4_CUDA_DETERMINISTIC_INDEXER=1` is now the one-shot correctness/debug switch
for this fallback. It disables the direct/WMMA indexer scorers and the fast
1024/2048/chunked top-k selectors, leaving the deterministic scalar scorer and
serial top-k selector. Keep it off by default.

Rejected candidate: `DS4_CUDA_DETERMINISTIC_TOPK_RANK=1` uses a parallel
rank-count top-k selector with the deterministic scorer. It improves the
16-token replay probe from `~1.13 tok/s` to `~11.82 tok/s`, but the 32-token
7047 replay still diverges at step 19. Do not use it as an exact verifier path.

Follow-up after adding `metal_graph_reset_mutable_state()` to
`ds4_session_invalidate()`:

| Probe | Env | Tokens | Replay match | First mismatch | Tok/s | Readout |
| ----- | --- | ------ | ------------ | -------------- | ----- | ------- |
| default after graph frontier reset | none | 32 | no | 31 | `13.38-13.40` | invalidation reset was a real improvement, but residual nondeterminism remains |
| deterministic indexer after reset | `DS4_CUDA_DETERMINISTIC_INDEXER=1` | 32 | no | 19 | `1.13` | still not a production oracle |
| quality after reset | `--quality` | 32 | no | 0 | `13.35-13.38` | quality mode is not a replay fix |

The invalidation bug was that `ds4_session_invalidate()` cleared the token
checkpoint but not the CUDA graph's logical mutable cache frontiers. A second
"full" replay prefill could therefore start with stale `layer_n_comp` /
`layer_n_index_comp` counters. Resetting those counters moves the default
7047-token replay mismatch later, but does not fully solve exact replay. The
remaining issue is likely nondeterminism or stale partial compressor/indexer
state, not only MTP model loading.

Rejected follow-up: also resetting the recurrent compressor state tensors
(`layer_attn_state_*` and `layer_index_state_*`) to their allocation defaults
on `ds4_session_invalidate()` did not fix replay. On the old Q2 baseline, the
7047-token 32-token exact replay failed at step `11` with exact/replay tok/s
`13.39/13.36`. This was worse than the counter-only reset probe, so keep only
the logical frontier reset for now.

Fresh-session exact replay diagnostic:

```sh
./ds4-bench --exact-replay-fresh-session --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 32
```

Result: exact replay still failed at step `19`, with exact/replay tok/s
`13.36/13.36`. The first stream picked token `36127`, while the fresh replay
session picked token `1337`.

Interpretation: the remaining 7047 replay blocker is not solely incomplete
`ds4_session_invalidate()` cleanup. Two independent graph sessions can still
diverge, so the next investigation should focus on nondeterministic CUDA
prefill/decode math or fast long-context indexer/top-k behavior across sessions.

Fixed-allocation fresh-session replay sweep:

```sh
./ds4-bench --exact-replay-fresh-session --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --ctx-alloc 7176 --gen-tokens N
```

Holding `ctx_alloc=7176` is important because otherwise changing `--gen-tokens`
also changes the allocated context shape and `compressed_kv_rows`, which
confounds replay-boundary probes.

| Tokens | Replay match | First mismatch | Notes |
| ------ | ------------ | -------------- | ----- |
| 1 | yes | -1 | fixed allocation |
| 2 | yes | -1 | fixed allocation |
| 4 | yes | -1 | fixed allocation |
| 5 | yes | -1 | fixed allocation |
| 6 | yes | -1 | fixed allocation |
| 8 | yes | -1 | fixed allocation |
| 9 | yes | -1 | fixed allocation rerun |
| 12 | yes | -1 | fixed allocation rerun |
| 16 | yes/no | -1 / 5 | one earlier fixed-allocation run failed at step 5, rerun passed |
| 32 | no | 11 | `first=88313`, `replay=68757`, first-stream margin `0.065052` |

Repeat fresh-session replay diagnostic, added to avoid restaging the 80 GiB
model for every sample:

```sh
./ds4-bench --exact-replay-runs 3 --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --ctx-alloc 7176 --gen-tokens 32
```

Result: `0/3` fresh replay runs matched the exact stream. First mismatches were
at steps `2`, `24`, and `24`; replay decode averaged `13.36 tok/s`.

Top-k overlap diagnostic:

```sh
./ds4-bench --exact-replay-runs 2 --exact-replay-topk 8 --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --ctx-alloc 7176 --gen-tokens 32
```

Result: `0/2` replay runs matched. Both runs first mismatched at step `11`
with the same two tokens swapped:

- exact stream: token `68757`, replay stream: token `88313`
- the exact token was replay rank `1` in both runs
- the replay token was exact rank `1`
- replay decode averaged `13.35 tok/s`

Interpretation: at least this failure mode is a top-two candidate swap, not a
case where the alternate token falls out of top-k. That points more toward
numerical replay instability around close logits than gross state corruption,
although it is still disqualifying for exact speculative verification.

Snapshot replay diagnostic:

```sh
./ds4-bench --exact-replay-snapshot --exact-replay-runs 2 \
  --exact-replay-topk 8 --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --ctx-alloc 7176 --gen-tokens 32
```

Result: `0/2` snapshot replays matched. First mismatches were at steps `22`
and `19`, with replay decode averaging `13.34 tok/s`. The top-k overlap again
showed the alternate token present in the exact stream's top-k and vice versa.

Interpretation: full prefill rebuild is not the only failure surface. Restoring
the saved pre-generation KV/session payload and decoding again can still pick a
different greedy stream. The remaining candidates are non-bit-exact snapshot
restore of GPU state or decode-time numerical nondeterminism after restore.

Fresh-session snapshot replay diagnostic:

```sh
./ds4-bench --exact-replay-snapshot-fresh-session --exact-replay-runs 2 \
  --exact-replay-topk 8 --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --ctx-alloc 7176 --gen-tokens 32
```

Result: `0/2` snapshot replays into fresh sessions matched. First mismatches
were at steps `19` and `11`, with replay decode averaging `13.29 tok/s`.
Top-k overlap still showed the alternate token present near the top.

Interpretation: same-session residue after snapshot load is not the main
explanation. The divergence follows the saved snapshot into fresh sessions, so
the likely boundary is either the snapshot payload is not a bit-identical GPU
frontier, or restored decode has numerically unstable reductions at the
7047-token frontier.

Short-context positive control for the same diagnostic:

```sh
./ds4-bench --exact-replay-snapshot-fresh-session --exact-replay-runs 2 \
  --exact-replay-topk 8 --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 64 --ctx-max 64 --ctx-alloc 128 --gen-tokens 32
```

Result: `2/2` snapshot replays into fresh sessions matched at ctx64. Exact
decode was `14.96 tok/s`; replay decode averaged `14.94 tok/s`.

Interpretation: the snapshot replay diagnostic itself can pass. The failure is
specific to the long-context frontier/state shape, not a universal bug in the
snapshot-fresh replay harness.

Snapshot-fresh frontier sweep with fixed allocation:

```sh
./ds4-bench --exact-replay-snapshot-fresh-session --exact-replay-runs 1 \
  --exact-replay-topk 8 --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start N --ctx-max M --ctx-alloc 7176 --gen-tokens 32
```

`ctx_alloc=7176` was held fixed so `raw_kv_rows=2304` and
`compressed_kv_rows=1796` stayed constant while only the prompt frontier
changed.

Observed boundary:

| Frontier | Replay match | First mismatch | Notes |
| -------- | ------------ | -------------- | ----- |
| 512 | yes | -1 | sweep pass |
| 1024 | yes | -1 | sweep pass |
| 1536 | yes | -1 | sweep pass |
| 1664 | yes | -1 | refined pass |
| 1792 | yes | -1 | refined pass |
| 1920 | yes | -1 | refined pass |
| 1952 | yes | -1 | refined pass |
| 1984 | yes | -1 | refined pass |
| 2016 | yes | -1 | refined pass |
| 2024 | yes | -1 | midpoint pass |
| 2032 | no | 20 | top-two swap, exact margin `0.007837` |
| 2047 | no | 10 | top-k overlap, alternate token exact rank `2` or `6` depending on run |
| 2048 | no | 12/25 | repeated failures, top-two swaps |

Interpretation: the first observed failure is between frontiers `2024` and
`2032`, very close to the `2048` prefill-chunk/raw-window transition. This is a
more actionable boundary than the original 7047 failure and should be the next
root-cause target before returning to MTP/DDTree.

Prefill-chunk sensitivity probe at the `2032` boundary:

```sh
DS4_METAL_PREFILL_CHUNK=1024 ./ds4-bench --exact-replay-snapshot-fresh-session \
  --exact-replay-runs 1 --exact-replay-topk 8 --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 2032 --ctx-max 2032 --ctx-alloc 7176 --gen-tokens 32

DS4_METAL_PREFILL_CHUNK=4096 ./ds4-bench --exact-replay-snapshot-fresh-session \
  --exact-replay-runs 1 --exact-replay-topk 8 --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 2032 --ctx-max 2032 --ctx-alloc 7176 --gen-tokens 32
```

| Frontier | Prefill chunk | Raw KV rows | Replay match | First mismatch |
| -------- | ------------- | ----------- | ------------ | -------------- |
| 2032 | default `2048` | `2304` | no | 20 |
| 2032 | `1024` | `1280` | no | 30 |
| 2032 | `4096` | `4352` | yes | -1 |

The same `4096` chunk does **not** fix the actual 7047 gate:

| Frontier | Prefill chunk | Replay match | First mismatch |
| -------- | ------------- | ------------ | -------------- |
| 7047 | `4096` | no | 2 |

Interpretation: prefill chunk/raw-cache allocation shape is part of the
replay-stability surface, but not the whole 7047 failure. Increasing the chunk
to `4096` can make the early `2032` boundary pass, while 7047 still diverges.

Forced-token logit-drift diagnostic:

```sh
./ds4-bench --exact-replay-snapshot-fresh-session --exact-replay-runs 1 \
  --exact-replay-topk 8 --exact-replay-forced-logit-diff --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start FRONTIER --ctx-max FRONTIER --ctx-alloc CTX --gen-tokens 32
```

This forces the replay session through the exact token sequence and compares
post-token logits, avoiding greedy-stream fork effects.

| Frontier | Ctx alloc | Greedy match | Forced max abs | Forced RMS | Max step | Notes |
| -------- | --------- | ------------ | -------------- | ---------- | -------- | ----- |
| 64 | 128 | yes | `0.000000000` | `0.000000000` | -1 | short-context positive control is bit-identical |
| 2024 | 7176 | yes | `6.669556618` | `1.147831321` | 31 | tokens match, logits already drift |
| 2032 | 7176 | yes when forced | `8.196508408` | `1.371109009` | 28 | greedy snapshot replay can fail here |

After adding first-drift reporting, repeated probes showed:

| Frontier | First drift step | First drift max abs | Worst max abs | Worst step | Notes |
| -------- | ---------------- | ------------------- | ------------- | ---------- | ----- |
| 2024 | 27 | `0.821135521` | `6.118864059` | 30 | drift appears late even while greedy stream matches |
| 2032 | 19 | `1.334746361` | `2.866692543` | 24 | this particular run matched greedily, but drift arrived earlier |
| 7047 | 0 | `0.803942442` | `4.131034851` | 19 | target frontier drifts immediately after first forced replay token |

Interpretation: token-stream equality is too weak as a replay correctness
signal. The snapshot-fresh diagnostic is bit-identical at ctx64, but by 2024
the restored path already produces materially different logits even when forced
through the same tokens. At 7047, forced replay drifts immediately after the
first token. MTP/DDTree exact verification needs a bit-stable verifier; this
logit drift is disqualifying even before it flips the greedy token.

Snapshot round-trip diagnostic:

```sh
./ds4-bench --exact-replay-snapshot-fresh-session --exact-replay-runs 1 \
  --exact-replay-topk 8 --exact-replay-forced-logit-diff \
  --exact-replay-snapshot-roundtrip --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --ctx-alloc 7176 --gen-tokens 32
```

Result:

- `snapshot_roundtrip_match=1`
- original snapshot bytes: `120948136`
- round-trip snapshot bytes: `120948136`
- forced replay still drifted immediately:
  `first_drift_step=0`, `first_drift_max_abs=1.573278189`,
  worst `max_abs=4.958981514` at step `19`

Interpretation: load-then-save preserves the serialized snapshot payload
byte-for-byte before decode. The long-context drift is therefore not a simple
snapshot payload read/write corruption. The remaining target is decode
execution after restore: either a non-payload graph tensor/state is read, or an
equivalent restored state takes a numerically different CUDA path.

Stage-localization dump:

```sh
DS4_METAL_GRAPH_DUMP_POS=7047 \
DS4_METAL_GRAPH_DUMP_LAYER=all \
DS4_METAL_GRAPH_DUMP_NAME='hc_ffn_post' \
./ds4-bench --exact-replay-snapshot-fresh-session --exact-replay-runs 1 \
  --exact-replay-topk 8 --exact-replay-forced-logit-diff \
  --exact-replay-snapshot-roundtrip \
  --exact-dump-prefix /tmp/ds4-exact-7047 \
  --replay-dump-prefix /tmp/ds4-replay-7047 \
  --cuda -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --ctx-alloc 7176 --gen-tokens 1
```

Result: `hc_ffn_post` is byte-identical through layers `0` and `1`; layer `2`
is the first byte difference. The layer-2 stage dump shows:

| Layer 2 stage | Exact vs replay |
| ------------- | --------------- |
| `hc_attn_pre` | byte-identical |
| `attn_norm` | byte-identical |
| `Qcur` | byte-identical |
| `KVcur` | byte-identical |
| `kqv_out` | first differing tensor, max abs `~1.67e-6`, RMS `~6.6e-8` |
| `attn_out` | max abs `~4.8e-7`, RMS `~1.2e-7` |

Repeating with `DS4_CUDA_DETERMINISTIC_INDEXER=1` still made `kqv_out` the
first differing layer-2 tensor, with similar tiny magnitude. That rules out the
fast indexer/top-k path as the sole source of first drift.

Additional attention-input probes:

- `DS4_CUDA_DETERMINISTIC_ATTENTION=1` preserves indexed compressed rows in
  top-k order instead of appending them with `atomicAdd`.
- Decode-time dumps of `decode_indexer_scores` and `decode_indexer_topk` showed
  byte-identical indexer scores and top-k selections for layer 2.
- `KVcompress`, the newly emitted attention compressed row, was byte-identical.
- Full `decode_attn_comp_cache` was byte-identical.
- Full `decode_raw_cache` differed outside the active window, but the active
  raw rows consumed by the 7047 decode attention call were byte-identical:
  raw physical rows `7..134`, `max_abs=0`, `rms=0`.

Attention double-run diagnostic:

```sh
DS4_CUDA_ATTENTION_DOUBLE_RUN_DIFF=1 \
DS4_METAL_GRAPH_DUMP_PREFIX=/tmp/ds4-double-run \
DS4_METAL_GRAPH_DUMP_POS=7047 \
DS4_METAL_GRAPH_DUMP_LAYER=2 \
DS4_METAL_GRAPH_DUMP_NAME='attention_double_run' \
./ds4-bench --exact-replay-snapshot-fresh-session --exact-replay-runs 1 \
  --exact-replay-topk 8 --exact-replay-forced-logit-diff \
  --exact-replay-snapshot-roundtrip --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --ctx-alloc 7176 --gen-tokens 1
```

Without deterministic attention, running the same layer-2 indexed attention
call twice on the same session inputs produced:

- exact pass: `max_abs=2.86102e-06`, `rms=6.45173e-08`
- replay pass: `max_abs=1.90735e-06`, `rms=5.9255e-08`

With `DS4_CUDA_DETERMINISTIC_ATTENTION=1`, both double-run diffs dropped to
zero and the one-token forced replay logit diff at 7047 also dropped to zero.

Stabilized exact replay gates:

| Env | Frontier | Tokens | Runs | Match | Forced drift | Exact tok/s | Replay tok/s |
| --- | -------- | ------ | ---- | ----- | ------------ | ----------- | ------------ |
| `DS4_CUDA_DETERMINISTIC_ATTENTION=1` | 7047 | 32 | 1 | yes | `max_abs=0`, `rms=0` | `12.61` | `12.63` |
| `DS4_CUDA_DETERMINISTIC_ATTENTION=1` | 7047 | 32 | 3 | `3/3` | not forced | `12.65` | `12.62` |
| `DS4_CUDA_DETERMINISTIC_ATTENTION=1` | 7047 | 128 | 1 | yes | not forced | `12.63` | `12.58` |

Interpretation: the long-context verifier replay instability was caused by
non-deterministic indexed attention row ordering, not by snapshot payload
corruption. The deterministic attention path is a correctness lever, but it
costs roughly `7-8%` decode throughput versus the old `~13.6 tok/s` baseline.
It should stay diagnostic/default-off until we decide whether exact
speculation needs this guarantee.

Layer 4 is where the small drift first becomes visibly amplified:

| Layer 4 stage | Max abs | RMS | Notes |
| ------------- | ------- | --- | ----- |
| `Qcur` | `~9.5e-7` | `~8.7e-8` | still tiny |
| `kqv_out` | `~1.4e-6` | `~4.8e-8` | still tiny |
| `ffn_moe_logits` | `~1.4e-6` | `~4.5e-7` | tiny |
| `ffn_moe_topk` | byte-identical | - | same routed experts |
| `ffn_moe_out` | `~1.37e-4` | `~3.8e-5` | first nontrivial amplification |
| `hc_ffn_post` | `~3.37e-5` | `~5.1e-6` | propagated layer output drift |

Interpretation: the immediate 7047 drift is not caused by different current
Q/KV, a corrupted serialized snapshot, or different routed expert IDs in the
early layer-4 MoE. The first observed byte difference is a tiny attention output
difference at layer 2, which routed expert math amplifies over later layers.
For exact speculation, this means "restore and replay" needs a stronger
bit-stability strategy than merely preserving more KV rows.

All-raw snapshot diagnostic:

```sh
DS4_SNAPSHOT_RAW_LIVE_ALL=1 \
./ds4-bench --exact-replay-snapshot-fresh-session --exact-replay-runs 1 \
  --exact-replay-topk 8 --exact-replay-forced-logit-diff --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start FRONTIER --ctx-max FRONTIER --ctx-alloc 7176 --gen-tokens 32
```

This diagnostic preserves every live row in the raw ring instead of only the
last `DS4_N_SWA` rows. It tests whether the default 128-row raw snapshot is too
narrow after long chunked prefill.

| Frontier | Replay match | First drift step | Forced max abs | Forced RMS | Max step | Notes |
| -------- | ------------ | ---------------- | -------------- | ---------- | -------- | ----- |
| 2024 | yes | not recorded | `6.603488922` | `0.793405890` | 31 | greedy still matches, forced logits drift |
| 2032 | yes | not recorded | `3.694396973` | `0.624182463` | 26 | single-run greedy symptom improved |
| 7047 | yes | not recorded | `4.829378128` | `0.843663752` | 26 | single-run greedy symptom improved |
| 7047 repeat | yes | 0 | `7.014596939` | `0.926338732` | 19 | full raw ring still drifts immediately |

Repeat check at the actual 7047 gate:

```sh
DS4_SNAPSHOT_RAW_LIVE_ALL=1 \
./ds4-bench --exact-replay-snapshot-fresh-session --exact-replay-runs 3 \
  --exact-replay-topk 8 --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --ctx-alloc 7176 --gen-tokens 32
```

Result: `0/3` replay runs matched. All three first mismatched at step `19`
with the exact stream selecting token `36127` and replay selecting token
`1337`; the alternate token remained near the top in both streams.

Interpretation: preserving more raw rows changes the single-run symptom and
reduces some forced-logit drift, so raw-ring coverage is part of the stability
surface. It is not sufficient to create a stable long-context verifier. Keep
`DS4_SNAPSHOT_RAW_LIVE_ALL=1` as a diagnostic only; do not promote it as an
MTP/DDTree correctness fix.

Interpretation: there is no stable short deterministic boundary after fixing
`ctx_alloc`. The same 16-token probe has produced both pass and fail results,
and the 32-token fixed-allocation probe still fails. Treat long-context replay
as nondeterministic or numerically unstable until proven otherwise; it is not a
valid exact speculative-verification oracle.

Rejected follow-up: `DS4_CUDA_ZERO_TENSOR_ALLOC=1` zero-fills every CUDA managed
tensor allocation before use. Fresh-session 7047 exact replay still failed at
step `11`, with exact/replay tok/s `13.36/13.41`. This makes simple
uninitialized graph tensors unlikely to be the primary cause. Keep the env as a
diagnostic hook, but do not treat it as a correctness or performance path.

Rejected follow-up: `DS4_CUDA_ZERO_TMP_ALLOC=1` zero-fills the reusable CUDA
temporary scratch buffer before each use. Fresh-session 7047 exact replay still
failed at step `19`, with exact/replay tok/s `13.30/13.27`. This makes stale
temporary scratch contents unlikely to be the primary cause.

Rejected follow-up: `DS4_CUDA_NO_TF32=1` disables cuBLAS TF32 tensor-op math.
Fresh-session 7047 exact replay still failed at step `19`, with exact/replay
tok/s `13.42/13.39`. This makes TF32 math mode unlikely to be the primary
cause.

Fresh-session with the slow deterministic indexer/top-k fallback also failed:

```sh
DS4_CUDA_NO_INDEXER_WMMA=1 \
DS4_CUDA_NO_INDEXER_DIRECT_ONE=1 \
DS4_CUDA_NO_TOPK1024=1 \
DS4_CUDA_NO_TOPK2048=1 \
DS4_CUDA_NO_TOPK_CHUNKED=1 \
./ds4-bench --exact-replay-fresh-session --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 32
```

Result: mismatch at step `19`, exact/replay tok/s `1.13/1.13`.
Interpretation: fresh-session nondeterminism is broader than the fast
indexer/top-k selectors. The old same-session "all slow" pass was therefore not
a sufficient proof of full graph determinism.

MTP under deterministic indexer:

| Env | Draft | Tokens | Effective tok/s | Exact tok/s | Exact match | First mismatch | Acceptance |
| --- | ----- | ------ | ---------------- | ----------- | ----------- | -------------- | ---------- |
| `DS4_CUDA_DETERMINISTIC_INDEXER=1` | 1 | 32 | `1.13` | `1.14` | no | 19 | `1.000` accepted/cycle |
| `DS4_CUDA_DETERMINISTIC_INDEXER=1` | 2 | 32 | `1.12` | `1.13` | no | 22 | `1.000` accepted/cycle |
| `DS4_CUDA_DETERMINISTIC_ATTENTION=1 DS4_MTP_STRICT=1 DS4_MTP_RESTORE_DRAFT_FRONTIER=1` | 1 | 32 | `12.72` | `12.74` | no | 19 | `1.000` accepted/cycle |
| `DS4_CUDA_DETERMINISTIC_ATTENTION=1 DS4_MTP_STRICT=1 DS4_MTP_RESTORE_DRAFT_FRONTIER=1` | 2 | 32 | `12.07` | `12.69` | no | 0 | `1.000` accepted/cycle, draft accept `0.000` |

This means the deterministic indexer is sufficient for plain exact replay, but
not sufficient for MTP-loaded long-context execution. Even `--mtp-draft 1`
diverges, where no extra MTP token should be accepted. The remaining MTP blocker
is therefore MTP-loaded graph/session state at long context, not only draft
quality.

Update after `DS4_CUDA_DETERMINISTIC_ATTENTION=1`: plain exact replay is now
stable through the fixed 7047-token, 128-generation gate. The original MTP
comparison was contaminated by full-prefill replay nondeterminism because
`ds4-bench` invalidated the session and rebuilt the prefix for the MTP side.
The benchmark now restores the saved pre-generation snapshot for the MTP
comparison, matching the exact-replay path.

Corrected MTP sweep at 7047 with the fixed benchmark generation length,
`gen_tokens=128`, using:

```sh
DS4_CUDA_DETERMINISTIC_ATTENTION=1 \
DS4_MTP_STRICT=1 \
DS4_MTP_RESTORE_DRAFT_FRONTIER=1 \
./ds4-bench --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --mtp /home/cghart/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --mtp-draft DRAFT \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --ctx-alloc 7176 --gen-tokens 128
```

| Draft | Exact match | Effective tok/s | Exact tok/s | Avg accepted/cycle | Draft accept rate | First mismatch |
| ----- | ----------- | ---------------- | ----------- | ------------------ | ----------------- | -------------- |
| 2 | yes | `12.28` | `12.77` | `1.000` | `0.000` | -1 |
| 3 | yes | `12.18` | `12.70` | `1.000` | `0.000` | -1 |
| 4 | yes | `12.19` | `12.70` | `1.000` | `0.000` | -1 |
| 6 | yes | `12.18` | `12.70` | `1.000` | `0.000` | -1 |

Corrected 32-token smoke sweep, retained for debugging:

| Draft | Exact match | Effective tok/s | Exact tok/s | Avg accepted/cycle | Draft accept rate | First mismatch |
| ----- | ----------- | ---------------- | ----------- | ------------------ | ----------------- | -------------- |
| 1 | yes | `12.79` | `12.77` | `1.000` | `0.000` | -1 |
| 2 | yes | `12.00` | `12.77` | `1.000` | `0.000` | -1 |
| 3 | yes | `12.04` | `12.68` | `1.000` | `0.000` | -1 |
| 4 | yes | `12.07` | `12.72` | `1.000` | `0.000` | -1 |
| 6 | yes | `11.96` | `12.70` | `1.000` | `0.000` | -1 |

Corrected interpretation: MTP/DDTree-style exact speculation is now
correctness-clean under the deterministic attention stabilizer and snapshot
comparison, but it supplies no extra accepted draft tokens at the 7047 frontier.
The effective rate is `<16 tok/s`, and drafts above 1 are slower than exact
decode. Per the performance gate, do not pursue this MTP path as the main route
to `20 tok/s`; pivot to exact-decode bandwidth work or a different drafter.

Short-context sanity gate:

| Context | Draft | Effective tok/s | Avg accepted/cycle | Draft accept rate | Exact match | Result |
| ------- | ----- | ---------------- | ------------------ | ----------------- | ----------- | ------ |
| 64      | 2     | `14.21`          | `1.000`            | `0.000`           | yes         | pass, no speculative gain |

Fixed 7047-token frontier sweep with
`DS4_MTP_STRICT=1 DS4_MTP_RESTORE_DRAFT_FRONTIER=1`:

| Draft | Effective tok/s | Exact tok/s | Avg accepted/cycle | Draft accept rate | Exact match | First mismatch |
| ----- | ---------------- | ----------- | ------------------ | ----------------- | ----------- | -------------- |
| 1     | `13.41`          | `13.46`     | `1.000`            | `0.000`           | no          | 19 |
| 2     | `12.94`          | `13.54`     | `1.000`            | `0.000`           | no          | 19 |
| 3     | `12.91`          | `13.49`     | `1.000`            | `0.000`           | no          | 19 |
| 4     | `12.86`          | `13.43`     | `1.000`            | `0.000`           | no          | 24 |
| 6     | `12.93`          | `13.47`     | `1.000`            | `0.000`           | no          | 2 |

These older rows are retained as historical data only. They predate the
deterministic attention stabilizer and the corrected snapshot-based MTP
comparison, so they should not be used for current go/no-go decisions.

`--quality` on the draft-1 diagnostic did not fix the long-context exactness
failure:

| Context | Draft | Mode      | Effective tok/s | Exact match | First mismatch |
| ------- | ----- | --------- | ---------------- | ----------- | -------------- |
| 7047    | 1     | `--quality` | `13.41`        | no          | 11 |

Historical interpretation before the deterministic-attention/snapshot fix:

- MTP model loading and short-context benchmarking work.
- The fixed 7047-token benchmark does not yet have a passing exactness gate,
  even for `--mtp-draft 1`, where no extra draft token is accepted.
- The independent `--exact-replay-probe` also fails at 7047 without MTP, so
  DDTree/MTP-style speculation cannot be judged safely until the replay
  determinism issue is isolated.
- The repeat fresh-session diagnostic `--exact-replay-runs 3` at fixed
  `ctx_alloc=7176` produced `0/3` matching 32-token replays after one exact
  decode, so the replay failure is not a one-off process-start artifact.
- Every measured draft depth accepted exactly one token per verifier cycle, so
  MTP supplied `0.000` extra draft acceptance and no path toward the required
  `~1.47` accepted tokens per verifier cycle.
- Effective MTP tok/s at 7047 was below the exact baseline (`~12.9` vs
  `~13.5 tok/s`) because MTP probing adds overhead without acceptance.
- Recommendation: do not pursue MTP speculation as the primary path to
  `20 tok/s` until the 7047-token exactness issue is understood. The next
  useful diagnostic is to build a deterministic fast-enough indexer/top-k path
  and verify `--exact-replay-probe` passes at 7047 before returning to DDTree or
  MTP acceptance work.

This interpretation is superseded by the corrected sweep above: with
`DS4_CUDA_DETERMINISTIC_ATTENTION=1` and snapshot-based MTP comparison, the
7047-token stream is exact for drafts `2,3,4,6`, but still accepts zero extra
draft tokens and remains below the `<16 tok/s` performance gate.

### Current go/no-go readout

- **Stop/reframe exact-decode-only 20 tok/s for now.** The measured `~23
  ms/token` gap is too large for launch cleanup or small fusions, and the
  dominant stages are bandwidth-heavy.
- **Go on targeted exact-decode experiments only where the measured surface is
  large enough:** `routed_moe` (`15.7 ms/token`), `attn_output+q_path`
  (`23.8 ms/token` combined), and long-context `compressor_indexer`
  (`10.5 ms/token`). Each experiment should have a measured byte-reduction or
  bandwidth-efficiency hypothesis before implementation.
- **No-go on simple prefix-retrieval speculation.** The measured upper bound is
  only `13.80 tok/s`.
- **No-go on current MTP speculation as the primary route.** The optional MTP
  GGUF is present and benchmarked. The corrected 7047-token snapshot comparison
  is exact under deterministic attention, but measured draft acceptance is zero
  and effective tok/s remains below the `<16` gate.

Near-term answer to "where can the missing `~23 ms/token` plausibly come from":
not from launch overhead, prefix retrieval, or the current MTP path. It would
need a combination of large exact-decode bandwidth wins in MoE/attention/indexer
plus a different speculation mechanism with demonstrated acceptance, or the
`20 tok/s` target should be reframed.

### Goal audit: MTP-style speculation toward 20 tok/s

Objective under audit: implement and measure a speculative decode path that can
plausibly move DS4 on GB10/CUDA from `~13.6 tok/s` toward `20 tok/s` at the
fixed 7047-token benchmark.

Checklist:

| Requirement | Evidence | Status |
| ----------- | -------- | ------ |
| Fresh upstream repo remains base | Work is in `/home/cghart/ds4-upstream-gb10`; baseline is recorded from fresh upstream `ae302c2`. | met |
| Fixed benchmark preserved | Baseline command and results are recorded above for `./ds4-bench --cuda -m /home/cghart/ds4/ds4flash.gguf --prompt-file bench/promessi_sposi.txt --ctx-start 7047 --ctx-max 7047 --gen-tokens 128`. | met |
| Baseline recorded | Old symlink baseline: cold staging `18.867-19.510s`, prefill `343.21-343.56 tok/s`, generation `13.63-13.68 tok/s`. | met |
| MTP GGUF available | `/home/cghart/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` exists and was used by `ds4-bench --mtp`. | met |
| MTP benchmark mode added | `ds4-bench` supports `--mtp`, `--mtp-draft`, `--mtp-margin`, and reports cycles/sec, accepted tokens/sec, average accepted/cycle, draft acceptance, rollback/partial rate, exact match, exact tok/s, and effective tok/s. | met |
| Exact verifier design | The main DS4 model is the verifier, draft tokens are accepted only through verifier comparison, rollback/frontier restore paths are wired, and corrected MTP comparison restores the saved target snapshot instead of re-prefilling. | met |
| Replay stability diagnostic | `ds4-bench` supports `--exact-replay-runs N`; historical full-prefill replay failed, but snapshot replay with `DS4_CUDA_DETERMINISTIC_ATTENTION=1` passed `3/3` at 7047/32 and `1/1` at 7047/128. | met |
| Exactness gate | Short context passes. Long 7047 snapshot replay passes under deterministic attention, and corrected MTP sweep produces exact greedy streams for drafts `2,3,4,6`. Full-prefill replay remains nondeterministic and is documented as a separate failure mode. | met with deterministic attention |
| Sweep coverage | Corrected MTP sweep covers draft `2,3,4,6` at 7047 with 128 generation tokens, plus a short-context draft-2 comparison. Historical draft-1 smoke is retained. | met |
| CUDA/source tests | `make ds4-bench ds4_test`, `./ds4_test --bench-mtp-spec-source --bench-exact-replay-source --cuda-indexed-decode-heads8-source`, and `git diff --check` pass. | met |
| Short logprob comparison | `DS4_TEST_MODEL=/home/cghart/ds4/ds4flash.gguf ./ds4_test --logprob-vectors` passes on CUDA after fixing the Linux test harness to select `DS4_BACKEND_CUDA` instead of hardcoded `DS4_BACKEND_METAL`. | met |
| Long 7047 MTP accepted token stream | Corrected MTP runs complete and match exact greedy stream under deterministic attention. | met |
| Performance gate | Corrected fixed-frontier MTP effective tok/s is `12.18-12.28`, below the deterministic exact baseline `~12.70` and below the `<16` gate, with `1.000` accepted token/cycle and `0.000` draft acceptance. | failed by design; go/no-go answered |
| Results recorded | Commands, baseline/speculative tok/s, acceptance metrics, failure modes, and go/no-go recommendation are recorded in this file. | met |

Audit conclusion:

- The near-term question is answered: current MTP-style exact speculation does
  **not** supply the missing `~1.47` accepted tokens per target verifier cycle.
  It supplies `0.000` extra draft acceptance on the corrected 7047-token path.
- The long-context verifier can be stabilized for snapshot replay with
  `DS4_CUDA_DETERMINISTIC_ATTENTION=1`, but this costs throughput and full
  prefill replay remains nondeterministic.
- The overall "toward 20 tok/s with current MTP" goal is **not achieved as a
  performance result**. The measurement goal is achieved: effective tok/s is
  `<16`, so do not pursue this MTP path as the main route.
- Next work should pivot to exact-decode bandwidth work in MoE, attention
  output, and compressor/indexer, or investigate a different drafter with
  demonstrated acceptance.

### Rejected experiment: indexed decode heads8 attention

Hypothesis: the existing `attention_indexed_mixed_heads8_online_kernel` might
be more GPU-cache-friendly for single-token indexed decode because it loads a
KV tile once and reuses it across eight heads. Upstream only used this path for
`n_tokens > 1`, so the experiment exposed it for `n_tokens == 1` behind:

```sh
DS4_CUDA_INDEXED_DECODE_HEADS8=1
```

Correctness smoke check:

- `--dump-logprobs` JSON for a 16-token CUDA greedy prompt matched byte-for-byte
  with and without the flag.

Performance at the fixed 7047-token frontier:

| Mode | Generation | Attention stage |
| ---- | ---------- | --------------- |
| default | `~13.63 tok/s` | `6.24 ms/token` |
| `DS4_CUDA_INDEXED_DECODE_HEADS8=1` | `12.96 tok/s` | `9.86 ms/token` |
| `DS4_CUDA_INDEXED_DECODE_HEADS8=1 DS4_CUDA_INDEXED_TWOPASS=1` | `12.58 tok/s` | not profiled; slower overall |

Conclusion: the apparent cache-locality improvement is outweighed by lower
single-token occupancy / synchronization overhead. For B=1 decode, the
per-head indexed kernel remains faster on GB10. Keep this path off by default;
do not pursue heads8 decode as the route to `20 tok/s`.

### Reproduction

```sh
DS4_GPU_DECODE_STAGE_PROFILE=1 \
DS4_GPU_DECODE_STAGE_PROFILE_LIMIT=16 \
./ds4-bench --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 128 \
  > /tmp/ds4-gb10-stage-profile.csv \
  2> /tmp/ds4-gb10-stage-profile.err

rg "gpu decode stage profile" /tmp/ds4-gb10-stage-profile.err
```

Measure prefix-retrieval speculation potential:

```sh
./ds4-bench --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 128 \
  --spec-probe --spec-ngram 2 --spec-draft 6 \
  > /tmp/ds4-gb10-spec-probe-ng2.csv \
  2> /tmp/ds4-gb10-spec-probe-ng2.err

rg "spec probe" /tmp/ds4-gb10-spec-probe-ng2.err
```

For kernel-level attribution use Nsight Systems (do not run
`ncu --set full`):

```sh
TMPDIR=$HOME/nsys-tmp nsys profile -o decode -f true -t cuda \
  --cuda-memory-usage=false --stats=false \
  ./ds4-bench --cuda -m ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 64 --ctx-max 64 --gen-tokens 32
nsys stats --report cuda_gpu_kern_sum --format table decode.nsys-rep
```

Use the short-ctx variant for "pure decode" attribution; long-ctx mixes
prefill and indexer kernels in.

### Decode baseline refresh

Performed after picking up the 2026-05-11 handoff plan, before any further
optimization. Repo snapshot:

- Branch: `main`
- HEAD: `ae302c2`
- Worktree dirty against HEAD: yes (carrying staged decode/test changes
  already present at handoff; baseline reflects this current built binary).

Baseline command:

```sh
./ds4-bench --cuda -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 128
```

| Metric              | Value           |
| ------------------- | --------------- |
| Cold staging        | `19.389 s`      |
| Prefill             | `340.16 tok/s`  |
| Generation          | `13.48 tok/s`   |
| KV cache bytes      | `120,948,136`   |

Stage profile (`DS4_METAL_DECODE_STAGE_PROFILE=1`, ctx 7047, 16 decode tokens,
totals divided by 16 to get per-token cost; aggregated to the handoff plan's
grouping):

| Stage                   | ms/token | Sub-stages summed                                   |
| ----------------------- | -------- | --------------------------------------------------- |
| attention               | `35.14`  | attn_hc_pre, attn_norm, q_path, kv_path, attention, attn_output, attn_hc_post |
| MoE                     | `17.44`  | routed_moe, router                                  |
| compressor/indexer      | `10.59`  | compressor_indexer                                  |
| shared FFN              | ` 9.93`  | ffn_hc_pre, ffn_norm, shared_gate_up, shared_down, ffn_hc_post |
| output/logits + residual| ` 1.16`  | wall (74.18) minus layer stages (73.10)             |
| **total per token**     | `~74.27` | matches measured 13.48 tok/s                        |

Decision: **baseline stable**. Generation `13.48 tok/s` is above the
`13.0 tok/s` abort threshold and within `1%` of the plan-recorded
`13.6 tok/s` baseline; stage attribution matches the plan's expected values
to within ~`3%`. Proceed with optimization tasks.

`DS4_CUDA_DECODE_PROFILE=1` from the handoff plan is not yet implemented in
the source (only `DS4_METAL_DECODE_STAGE_PROFILE` and
`DS4_DECODE_PROFILE_DETAIL` exist today); Task 2 of the plan introduces
`DS4_CUDA_DECODE_EVENT_PROFILE`. The host-stage attribution above is the
reference number set the event timing should agree with within `10%`.

### CUDA event timing for decode stages

Task 2 of the handoff plan: add a low-perturbation profile based on
`cudaEventRecord`/`cudaEventElapsedTime` alongside the existing host-side
`DS4_GPU_DECODE_STAGE_PROFILE`.

Implementation summary:

- New env flag `DS4_CUDA_DECODE_EVENT_PROFILE=1` activates a self-contained
  module in `ds4_cuda.cu`. Public API in `ds4_gpu.h`:
  `ds4_gpu_decode_event_profile_enabled`,
  `ds4_gpu_decode_event_profile_begin_token`,
  `ds4_gpu_decode_event_profile_mark`,
  `ds4_gpu_decode_event_profile_finish_token`.
- `ds4.c` wires `_begin_token` and `_finish_token` around the decode token in
  `metal_graph_eval_token_raw_swa`, and calls `_mark(stage)` from the two
  boundary macros (`DS4_METAL_PROFILE_DECODE_STAGE` in the layer body and
  `DS4_GPU_PROFILE_OUTPUT_STAGE` in the output head). The mark sites match
  the boundaries the host profile already uses, so the two profiles report
  the same stages and the same headline groups (`attention`, `moe`,
  `compressor_indexer`, `shared_ffn`, `output_logits`).
- The mark is a fast no-op when the env flag is unset (single load + branch).
- Events are reused across tokens: at end of token, `_finish_token` syncs
  on the final event (the logits readback already syncs the device), walks
  the event pairs with `cudaEventElapsedTime`, and accumulates per-group ms.
  An `atexit` handler prints one aggregate line.
- Source-hook test asserts the symbols are present, gated on
  `./ds4_test --decode-profile-source`.

Measured agreement with the existing host profile at the 7047-ctx benchmark
(`ctx=7047`, `gen-tokens=16`):

| Group              | Host `DS4_GPU_DECODE_STAGE_PROFILE` | CUDA event profile | Delta |
| ------------------ | ----------------------------------- | ------------------ | ----- |
| attention          | `34.81 ms/token`                    | `34.66 ms/token`   | `-0.4%` |
| moe                | `17.40 ms/token`                    | `17.31 ms/token`   | `-0.5%` |
| compressor_indexer | `10.55 ms/token`                    | `10.59 ms/token`   | `+0.4%` |
| shared_ffn         | ` 9.75 ms/token`                    | ` 9.66 ms/token`   | `-0.9%` |
| output_logits      | ` 2.61 ms/token`                    | ` 2.57 ms/token`   | `-1.5%` |
| **total reported** | `75.13 ms/token`                    | `74.80 ms/token`   | `-0.4%` |

All five group totals agree within `2%`, well inside the `10%` agreement gate
the handoff plan asked for. The CUDA event total is consistently `0.3 ms/token`
lower because the host profile inserts a `cudaDeviceSynchronize` at every
boundary; the event profile does not.

Throughput overhead, measured back-to-back at `ctx 7047`, `gen-tokens 32`:

| Run                                         | gen tok/s | Delta vs control |
| ------------------------------------------- | --------- | ---------------- |
| control                                     | `13.46`   | —                |
| `DS4_CUDA_DECODE_EVENT_PROFILE=1`           | `13.34`   | `-0.9%`          |

`<1%` overhead — safe to leave the mark sites unconditionally in the boundary
macros even when the host profile is also disabled. Aggregate line format
when active:

```text
ds4: CUDA decode event profile tokens=32 attention_ms_per_token=34.578 \
  moe_ms_per_token=17.188 compressor_indexer_ms_per_token=10.619 \
  shared_ffn_ms_per_token=9.593 output_logits_ms_per_token=2.546 \
  other_ms_per_token=0.000 total_ms_per_token=74.525 dropped_marks=0
```

Decision: **event timing in place**, agreement validated. The event profile
is now the cheaper default when only the headline group breakdown is needed,
and is what the next experiments (attention cache-friendly path, MoE
bandwidth microbench, indexer rewrite) should compare against to avoid the
per-boundary sync of the host profile contaminating measurements.

### Task 3 re-evaluation (attention cache-friendly)

Picked up the handoff plan's Task 3 after Task 2. The plan's first-priority
mechanism — "Reuse compressed KV tiles across head groups instead of
reloading per head" — corresponds to the `attention_indexed_mixed_heads8_online_kernel`
path already gated by `DS4_CUDA_INDEXED_DECODE_HEADS8=1`. That experiment is
already recorded in this notebook (see "Rejected experiment: indexed decode
heads8 attention") with the verdict:

> at GB10, B=1 decode loses more from low SM occupancy (8 head_groups → 8
> blocks) than it gains from KV-tile reuse. Generation drops `13.63 → 12.96
> tok/s`; attention stage rises `6.24 → 9.86 ms/token`.

The plan's other Task 3 mechanisms target the deterministic path
(`DS4_CUDA_DETERMINISTIC_ATTENTION`, `DS4_CUDA_DETERMINISTIC_INDEXER`,
`DS4_CUDA_DETERMINISTIC_TOPK_RANK`) which the project's safety constraints
already say "is useful for verifier exactness but costs roughly `7-8%` decode
throughput" — improvements there gate on whether someone needs deterministic
decode, which the active workstream (advance toward `20 tok/s`) does not.

Looking at the host stage profile, the attention *group* of 34.66 ms/token
decomposes as:

| Sub-stage   | ms/token | Notes |
| ----------- | -------- | ----- |
| attn_hc_pre | `3.41`   | small fused HC split |
| attn_norm   | `0.02`   | trivial |
| q_path      | `9.76`   | q_a/kv/q_b q8 matmuls + RMS + RoPE |
| kv_path     | `0.81`   | KV row append (`store_raw_kv_batch_kernel`) |
| attention   | `6.18`   | the actual flash kernel (`attention_indexed_mixed_kernel`) |
| attn_output | `14.62`  | output_a + output_b + hc_expand q8 matmuls |
| attn_hc_post| `0.02`   | trivial |

The two cost centers in the attention group are `attn_output` (14.62 ms;
already running at ~209 GB/s = 77% of peak per bandwidth check below) and
`q_path` (9.76 ms; ~184 GB/s = 67% of peak). The actual attention compute is
6.18 ms and is essentially dominated by KV reads which are not weight-bound,
so general "cache-friendly" reductions there have limited ceiling.

Bandwidth-floor check on `attn_output` and `q_path` (q8 weights, 1.0625 B/wt):

- `attn_output` per layer = output_a (4096 × 8192 q8) + output_b (8192 × 4096 q8)
  = `35.65 MB` + `35.65 MB` = `71.3 MB`. Across 43 layers per decode token =
  `~3.07 GB`. At measured `14.62 ms/token` that's `~210 GB/s` effective —
  77% of the GB10 `273 GB/s` peak. Little headroom.
- `q_path` per layer = q_a (`4.46 MB`) + kv (`2.23 MB`) + q_b (`35.65 MB`)
  = `42.34 MB`. Across 43 layers = `~1.82 GB`. At `9.76 ms/token` =
  `~187 GB/s` effective — 68% of peak. Some headroom but not large.

**Decision: skip Task 3.** Its primary mechanism is already rejected on this
machine, and the remaining attention sub-stages are already running at 68-77%
of peak LPDDR5x bandwidth. The largest mis-utilization is elsewhere — the
routed MoE stage (`17.31 ms/token`) sits near `106 GB/s` (`~38%` of peak) per
the earlier bandwidth-floor analysis. Pivot directly to the plan's Task 4
(MoE bandwidth microbench) — which the plan also says to pick when MoE is
unusually inefficient.

### MoE bandwidth event-profile microbench (Task 4)

Added `DS4_CUDA_MOE_EVENT_PROFILE=1` in `ds4_cuda.cu`. It shares the same
CUDA events as the existing per-call `DS4_CUDA_MOE_PROFILE` but accumulates
across the whole run and emits one aggregate line at exit. The aggregator
only counts `n_tokens == 1` MoE invocations (decode), so prefill MoE calls
don't dilute the per-decode-token cost. When the decode event profile is
also enabled, the per-token divisor is the actual decode-token count; when
the MoE profile is enabled alone, the divisor falls back to MoE invocations
(layer-calls) and the report includes `unit=moe_call` so the caller can
recover per-token by multiplying by their layer count.

Per-call weight bytes are computed from the kernel parameters already in
hand: `gate_up_bytes = 2 × pair_count × gate_expert_bytes` (gate + up over
the 6 active experts) and `down_bytes = pair_count × down_expert_bytes`.

Measured aggregate at `ctx 7047`, `gen-tokens 32`, both flags on:

| Field                         | Value             |
| ----------------------------- | ----------------- |
| decode tokens                 | `32`              |
| MoE invocations               | `1376` (43 layers × 32 tokens) |
| `gate_up_ms_per_token`        | `10.432 ms`       |
| `down_ms_per_token`           | ` 4.942 ms`       |
| `xq_ms_per_token`             | ` 0.231 ms`       |
| `midq_ms_per_token`           | ` 0.178 ms`       |
| `sort_ms_per_token`           | ` 0.038 ms`       |
| `sum_ms_per_token`            | ` 0.038 ms`       |
| `total_ms_per_token` (routed_moe wrapper internals) | `15.860 ms` |
| `gate_up_gib_per_token`       | ` 1.039 GiB`      |
| `down_gib_per_token`          | ` 0.661 GiB`      |
| **achieved_gbps**             | **`115.14 GB/s`** |
| decode event group "moe"      | `17.746 ms/token` |
| difference (router + glue)    | `1.886 ms/token`  |

Consistency check: 1.039 GiB + 0.661 GiB = 1.700 GiB ≈ 1.826 GB; at 15.860 ms
that's `1.826 / 0.015860 = 115.14 GB/s` ✓. The 1.886 ms/token gap between the
MoE-wrapper total and the decode event profile's `moe` group matches the
router select kernels measured at `~1.66 ms/token` in the host profile.

GB10 LPDDR5x peak is `273 GB/s`. **MoE achieves `115.14 GB/s` = `~42%` of
peak**, well inside the plan's `<150 GB/s → larger rewrite may be justified`
band.

Two corroborating measurements:

| Probe | Result |
| ----- | ------ |
| MoE profile alone (fallback to per moe_call) | `0.244 ms × 43 layers = 10.49 ms gate_up/token` → matches |
| achieved_gbps with MoE profile alone        | `115.05 GB/s` → matches per-token mode |

The kernels in scope at decode (all `n_tokens=1`) are:

- `moe_gate_up_mid_decode_lut_qwarp32_kernel` — the gate+up iq2_xxs path with
  shared-memory LUT for `cuda_iq2xxs_grid` and `cuda_ksigns_iq2xs`. Reads
  `2 × 6 × gate_expert_bytes` per layer.
- `moe_down_sum6_qwarp32_kernel` — the direct decode down kernel that fuses
  the 6-expert weighted sum (no separate `moe_sum_kernel` launch). Reads
  `6 × down_expert_bytes` per layer.

The gate/up kernel is the larger lever: `10.43 ms/token` against
`~1.04 GiB/token` of iq2_xxs weight traffic = `~107 GB/s` for the iq2 LUT
path alone. Even hitting the q8 matmul's measured `~210 GB/s` on the same
hardware would lift gate/up to `~5.3 ms/token` and the full MoE to
`~10.7 ms/token`, recovering **`~7 ms/token`** of decode time — the single
largest known available win toward `20 tok/s`.

**Decision: MoE iq2_xxs gate/up kernel rewrite is on the critical path.**
The 42% peak utilization is the worst observed across all decode kernels,
and the iq2_xxs LUT decode itself is the only reason it falls so far below
the bandwidth-bound q8 paths. Next workstream: prototype a TMA-staged
double-buffered variant of `moe_gate_up_mid_decode_lut_qwarp32_kernel` on a
synthetic fixture before touching the production dispatch.

Reproduction:

```sh
DS4_CUDA_DECODE_EVENT_PROFILE=1 DS4_CUDA_MOE_EVENT_PROFILE=1 \
  ./ds4-bench --cuda -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 32
```

### Compressor/indexer event probe (Task 5)

Added `DS4_CUDA_INDEXER_EVENT_PROFILE=1` in `ds4_cuda.cu`. The probe wraps
three public entry points with cudaEvent-bracketed timing, filtered to
single-token-decode invocations (`n_tok == 1`):

- `ds4_gpu_matmul_f16_pair_tensor` → bucket **compressor** (used by both the
  attention compressor and indexer compressor projections).
- `ds4_gpu_indexer_score_one_tensor` → bucket **indexer_scores**.
- `ds4_gpu_indexer_topk_tensor` → bucket **indexer_topk**.

The wrappers refactor their bodies into private `*_impl` helpers and the
public `extern "C"` entry points add the begin/end events. The events sync
per call (this is a diagnostic flag, sync cost is acceptable). The atexit
report uses the decode event profile's token count when available so the
per-token numbers are accurate. Source-hook test extended.

Measured at the fixed `7047`-token frontier, `32` decode tokens, both
profile flags enabled:

| Sub-kernel             | Calls/token | ms/token  | Notes |
| ---------------------- | ----------- | --------- | ----- |
| compressor (pair matmul) | `62`        | `5.611`   | 41 attn-compressor + 21 indexer-compressor projections |
| indexer_scores         | `21`        | `0.607`   | one per ratio-4 layer |
| indexer_topk           | `21`        | `0.697`   | one per ratio-4 layer (`indexer_topk_pow2_kernel<2048>`) |
| **indexer profile sum**| —           | `6.915`   | |
| decode event group `compressor_indexer` | — | `11.125` | from the host-comparable profile |
| **gap (other kernels)** | —          | **`4.21`** | `compressor_update`, indexer-q/indexer-weights f16 matmuls, fp8 KV quant — not yet instrumented |

Confirming run at `64`-ctx (no indexer fires) to verify ctx-sensitivity:

| Sub-kernel             | Calls/token | ms/token at 64-ctx | ms/token at 7047-ctx |
| ---------------------- | ----------- | ------------------ | -------------------- |
| compressor             | `62`        | `5.564`            | `5.611` |
| indexer_scores         | `0`         | `0.000`            | `0.607` |
| indexer_topk           | `0`         | `0.000`            | `0.697` |
| **sum**                | —           | `5.564`            | `6.915` |
| decode `compressor_indexer` | —      | `6.333`            | `11.125` |
| **ctx-sensitive overhead** | —        | —                  | **`+4.79 ms/token`** |

The compressor projection cost is **completely ctx-independent** — same
`~5.6 ms/token` at 64 and 7047 ctx — because the matmul shape is
`n_embd × comp_width`, not driven by KV history. The actual ctx scaling of
`compressor_indexer` (`+4.79 ms/token` from 64→7047 ctx) is split between
the indexer kernels in this profile (`+1.30 ms/token`) and the
not-yet-instrumented kernels in the gap (`+3.49 ms/token` — most likely the
indexer-q `matmul_f16_tensor`, the indexer-weights `matmul_f16_tensor`, and
`compressor_update_tensor`).

**Decision per plan thresholds:**

The plan's two variant candidates were:

- `DS4_CUDA_INDEXER_DECODE_VEC=1` — vectorized indexer scoring. Indexer
  scoring is only `0.607 ms/token`. Even a 2× rewrite saves `~0.3 ms/token`
  = `~0.4%`. **Skip — too small.**
- `DS4_CUDA_COMPRESSOR_CUBLASLT=1` — cuBLAS-Lt compressor matmul. Compressor
  is `5.611 ms/token` at any ctx. The current
  `matmul_f16_pair_ordered_chunks_kernel<<<out_dim, 32>>>` runs with one
  warp per output row — for `out_dim=1024` that's 1024 blocks of 32 threads,
  well utilized; for `out_dim=256` (indexer compressor) it's only 256 blocks,
  underutilizing GB10's ~60 SMs. **Moderately promising — `~1-2 ms/token`
  upside if cuBLAS-Lt FP16 handles the small n=1 case well**, but precedent
  on Ampere is that cuBLAS-Lt at n=1 is rarely a clean win over hand-tuned
  warp-per-row.

Neither variant is the highest-leverage next step. The current MoE rewrite
(Task 4 finding: `115 GB/s` vs `273 GB/s` peak, `~7 ms/token` headroom) is
both bigger and lower-uncertainty. **Queue compressor cuBLAS-Lt as a
secondary follow-up; primary next workstream remains the MoE iq2_xxs
kernel rewrite.**

A useful follow-on probe would be to additionally instrument
`ds4_gpu_compressor_update_tensor` and the two `matmul_f16_tensor` calls in
the indexer path (`indexer_attn_q_b` and `indexer_proj`) to close the
`4.21 ms/token` measurement gap.

Reproduction (long ctx):

```sh
DS4_CUDA_DECODE_EVENT_PROFILE=1 DS4_CUDA_INDEXER_EVENT_PROFILE=1 \
  ./ds4-bench --cuda -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 32
```

Reproduction (short ctx):

```sh
DS4_CUDA_DECODE_EVENT_PROFILE=1 DS4_CUDA_INDEXER_EVENT_PROFILE=1 \
  ./ds4-bench --cuda -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 64 --ctx-max 64 --gen-tokens 64
```

### Phase 0 free wins and characterization (2026-05-11)

After picking up the goal "implement the list above in the order present to
increase tok/s", started executing the calibrated plan derived from
SGLang/llama.cpp/vLLM precedents on GB10. Phase 0 is two free probes and a
characterization microbench.

**Phase 0a — Spin scheduling (`cudaDeviceScheduleSpin`).** Added
`cudaSetDeviceFlags(cudaDeviceScheduleSpin)` to `ds4_gpu_init` (gated by
`DS4_CUDA_NO_SPIN_SCHEDULE`). Matches the llama.cpp commit `5acd455`
Blackwell decode fix. Measured:

| Run | gen tok/s |
| --- | --------- |
| Baseline (Auto schedule) | `13.47` |
| `DS4_CUDA_NO_SPIN_SCHEDULE=1` | `13.44` |
| Spin schedule (default after change) | `13.44` |

Verdict: no measurable effect on GB10 with this workload — the runtime
heuristic already picks an effective schedule. Flag stays in place because
it costs nothing and matches upstream.

**Phase 0b — `-O2` sanity rebuild.** Rebuilt `ds4_cuda.cu` with
`nvcc -O2 --use_fast_math` to check for the known CUDA 13 / `sm_121` nvcc
`-O3` MUL_MAT codegen bug (llama.cpp issue 18331). Measured:

| Run | gen tok/s | logprob-vectors |
| --- | --------- | --------------- |
| `-O3` baseline | `13.47` | OK |
| `-O2 ds4_cuda.cu` | `13.51` | OK |

Delta is within noise; `logprob-vectors` passes both builds with identical
output. Confirms `-O3` is **not** silently miscompiling our kernels on
`sm_121`. Restored `-O3` as the default.

**Phase 0c — Memory-floor microbench.** Added
`ds4_gpu_memfloor_bench(bytes, iterations)` in `ds4_cuda.cu` and a test entry
`./ds4_test --memfloor-bench` (env-gated by `DS4_TEST_MEMFLOOR=1`). The
kernel does pure vectorized `uint4` reads with XOR-accumulation and a
sentinel-guarded write so the loads cannot be optimized away. Measured on
GB10:

| Working set | Achieved GB/s | Interpretation |
| ----------- | ------------- | -------------- |
| `4 MiB`     | `1014`        | L2-resident (irrelevant) |
| `26 MiB`    | `342`         | L2-partial (~MoE active gate+up size at decode) |
| `71 MiB`    | `274`         | Right at LPDDR5x nominal peak |
| `512 MiB`   | **`245`**     | **Sustained DRAM read on benign pattern (~90% of 273 peak)** |

Comparison against the kernels we already measure with the MoE/decode event
profiles:

| Kernel              | Bytes/token | Time/token | Achieved GB/s | % of GB10 245 GB/s sustained |
| ------------------- | ----------- | ---------- | ------------- | --------------------------- |
| q8 attn_output (a+b)| `~2.20 GB`  | `14.36 ms` | `~210`        | **`~86%`** ✓               |
| q8 q_path           | `~1.85 GB`  | ` 9.51 ms` | `~195`        | **`~80%`** ✓               |
| q8 shared FFN       | `~1.15 GB`  | ` 6.18 ms` | `~186`        | **`~76%`** ✓               |
| **iq2 routed MoE**  | `~1.70 GB`  | `15.78 ms` | **`~108`**    | **`~44%`** ✗               |

The q8 paths are already running at 76-86% of the GB10 sustained DRAM read
ceiling — modest headroom only. The routed MoE iq2_xxs path sits at **44%**
of the same ceiling, leaving roughly **`6-8 ms/token` of recoverable time**
if we can close even half of the gap.

**Phase 0 conclusion: routed MoE is bandwidth-utilization-limited, not at
the GB10 LPDDR5x wall.** Likely causes (from kernel structure): the 66-byte
iq2_xxs super-block layout interferes with coalescing; the 6-expert scatter
produces 6 concurrent DRAM streams; the per-block shared-memory LUT load
adds startup; and the LUT decode adds compute per byte. All of these are
targets for Phase 1 (tile/CTA sweep) and Phase 4 (TMA staging).

**Decision:** Phase 1 is justified. Proceed to MoE iq2 gate/up tile/CTA
sweep. Phase 4 (TMA) remains conditional on Phase 1 results and would
target the ~245 GB/s ceiling, which would save ~`8 ms/token` of MoE time
in the limit.

### Phase 1: MoE iq2 gate/up tile/CTA sweep (2026-05-11)

Per the calibrated plan: try the lower-risk CUTLASS-style tile/CTA tuning
before the TMA rewrite. Added a templated variant of
`moe_gate_up_mid_decode_lut_qwarp32_kernel` parameterised on the rows-per-CTA
count, gated by `DS4_CUDA_MOE_DECODE_LUT_TILE={32,64,128,256,512}`. The math
and shared-memory layout stay identical; only the grid x dimension and the
inner `rr` loop bound change.

Sweep results at `ctx 7047`, `gen-tokens 32`, both event profiles enabled:

| Tile | `gate_up_ms_per_token` | Achieved GB/s |
| ---- | ---------------------- | ------------- |
| 32   | `10.493`               | `115.00`      |
| 64   | `10.452`               | `115.06`      |
| **128 (baseline)** | **`10.343`** | **`116.00`** |
| 256  | `10.612`               | `114.00`      |
| 512  | `16.198`               | ` 84.48`      |

Baseline is essentially optimal within ±0.3% across the practical range and
collapses past 256 (likely register pressure or SMEM pressure pushing
occupancy down). **Pure tile/CTA shape is not the lever** for the routed
MoE on this kernel structure. SGLang's reported +6.3% from tile tuning was
on a fundamentally different kernel family (NVFP4 CUTLASS); it does not
transfer to our iq2_xxs `__shared__`-staged decode kernel.

### Phase 1 diagnostic: no-decode bandwidth probe

To characterise where the remaining time goes, added
`moe_gate_up_mid_decode_nodecode_kernel` (env-gated by
`DS4_CUDA_MOE_DECODE_NODECODE=1`) which keeps the production memory access
pattern exactly — same grid, same per-block xq cache load, same 6-expert
scatter, same per-row super-block stride — but replaces the iq2_xxs LUT
dequant/dot with a 64-bit `XOR`-fold. Produces meaningless math (so it can
never ship; writes a `-1.0e30f` sentinel into `mid_out`); useful purely to
measure the cost split between memory traffic and LUT compute.

Result at same workload:

| Variant   | `gate_up_ms_per_token` | Achieved GB/s |
| --------- | ---------------------- | ------------- |
| LUT decode (production) | `10.343` | `116` |
| nodecode  | ` 9.456`               | `122.69`      |

The LUT decode itself accounts for **`~0.9 ms/token`** (`~8.6%` of
`gate_up`). The remaining **`9.5 ms/token`** is memory traffic *at this
kernel's access pattern* — which caps out at `~122 GB/s`, less than half of
the `245 GB/s` benign-pattern sustained DRAM read measured by
`memfloor_bench` at the same working set.

The pattern-vs-floor gap (`122 GB/s` kernel vs `245 GB/s` benign) is the
real constraint, not the LUT decode. The kernel's 6-expert scatter across
disjoint DRAM regions, the `1056`-byte per-row stride, and the per-lane
inner-loop over 16 super-blocks add bank-conflict and request-coalescing
overheads that don't show up in benign sequential reads.

**Phase 1 conclusion:** Negative result for tile tuning; clear positive
diagnostic for *what the next lever has to attack*. Phase 4 (TMA staging
with double-buffered shared-memory tiles and per-expert pass reorder) is
the right intervention to close the `122 → 245 GB/s` gap. Phase 1's tile
variants and the no-decode probe stay env-gated in source as diagnostic
infrastructure; the production dispatch is unchanged.

The Phase 1 reproduction:

```sh
for T in 32 64 128 256 512; do
  echo "--- tile=$T ---"
  DS4_CUDA_DECODE_EVENT_PROFILE=1 DS4_CUDA_MOE_EVENT_PROFILE=1 \
  DS4_CUDA_MOE_DECODE_LUT_TILE=$T ./ds4-bench --cuda \
    -m /home/cghart/ds4/ds4flash.gguf \
    --prompt-file bench/promessi_sposi.txt \
    --ctx-start 7047 --ctx-max 7047 --gen-tokens 32 \
    | grep "CUDA moe event"
done
DS4_CUDA_DECODE_EVENT_PROFILE=1 DS4_CUDA_MOE_EVENT_PROFILE=1 \
DS4_CUDA_MOE_DECODE_NODECODE=1 ./ds4-bench --cuda \
  -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 32
```

### Phase 2: Compressor pair + HC fusion (deferred)

Attempted a `__half2`-vectorised variant of
`matmul_f16_pair_ordered_chunks_kernel` to halve the inner loop count and
expand ILP. The change preserves total bytes read and total flops; it only
restructures the inner sum from one `__half→float` mul-add per iteration to
two `__half2`-lane mul-adds per iteration. With `--use_fast_math` enabled,
the compiler reorders these into fused multiply-adds in a different order
than the original kernel, and the resulting rounding drift was enough to
flip top-1 tokens in the `logprob-vectors` test (`step 3 selected token
mismatch` on `long_memory_archive`). Reverted the variant.

The compressor stage observation that informed the abort:

- attn compressor (`in_dim=4096`, `out_dim=1024`): `~178 GB/s` per-call =
  `~73%` of GB10 sustained DRAM. Already well-utilised.
- indexer compressor (`in_dim=4096`, `out_dim=256`): only ~4 MiB per call;
  launch overhead dominates over BW. Would require batching across layers
  to materially improve, not a kernel rewrite.

Realistic combined Phase 2 upside on this codebase is `<1%`, paid for by
non-trivial engineering risk on a numerically sensitive matmul. Deferring
in favor of Phase 3 (speculative decoding) which the SGLang+EAGLE
precedent shows has the only documented large multiplier (`+70%`) on
this hardware class.

The fizzled variant lives in commit history but is not in source.

### Phase 3: Speculative decoding (this is the multiplier)

The earlier MTP probe in this repo accepted 0 extra draft tokens (recorded
above under "MTP speculative benchmark"). EAGLE-style self-speculation
needs a trained draft head we do not have. The remaining lower-effort
options are prompt-aware n-gram lookup and prompt-prefix retrieval; both
were dismissed earlier as "measured upper bound only about 13.80 tok/s"
in the "Rejected Or Low-Value Paths" section of the handoff plan.

**Phase 3 status: blocked on either a real drafter for DS4-Flash, or a
substantially different speculation design.** Concretely:

- n-gram / prefix-retrieval drafters: per the existing measurement in this
  repo, upper bound is `~13.80 tok/s`. Not enough headroom.
- EAGLE-style trained draft head: would require training data + GPU time;
  out of scope for this session's hands-on plan.
- A faster MTP variant: previously yielded zero acceptance; same model
  metadata, same problem.

The hands-on plan-to-20 ran into the same wall the original handoff plan
identified. Without a new speculation design, the kernel-only ceiling on
this stack remains the operative bound. Recorded Phases 0+1 lift the
ceiling slightly (and reveal the access-pattern is the real iq2 MoE
constraint), but the single-token decode wall on GB10 for DS4-Flash at
this quantization sits firmly in the `13.5-14 tok/s` range.

### Phase 1++: Static unroll of the iq2 decode kernel inner loop (the win)

After the half-warp variant proved coalescing wasn't the bottleneck, tried
specializing the kernel on the compile-time-known `xq_blocks=16` for the
DS4-Flash shape (`expert_in_dim=4096`, `CUDA_QK_K=256`). The production
kernel's inner loop is

```cuda
for (uint32_t b = lane; b < xq_blocks; b += 8u) { ... }
```

with `xq_blocks` a runtime `uint32_t`. With 8 lanes and `xq_blocks=16`, each
lane does exactly 2 iterations. By templating on `XQ_BLOCKS=16` the compiler
can fully unroll the loop and improve ILP — the two iq2 LUT-dot calls per
lane can issue back-to-back without the loop-test/increment overhead.

Added `moe_gate_up_mid_decode_lut_qwarp32_unroll_kernel<XQ_BLOCKS>` and
dispatched it automatically when `xq_blocks == 16`. Verified bit-equivalent
with `--logprob-vectors`. `DS4_CUDA_MOE_DECODE_LUT_NO_UNROLL=1` disables.

Measurements at `ctx 7047`, `gen-tokens 128`:

| Run                      | gen tok/s | gate_up_ms (per moe_call) | Achieved BW (gate+up+down) |
| ------------------------ | --------- | ------------------------- | -------------------------- |
| baseline run 1           | `13.45`   | `0.242`                   | `115.70` GB/s              |
| baseline run 2           | `13.46`   | `0.242`                   | `115.70` GB/s              |
| **unroll run 1**         | **`13.53`** | `0.226`                 | `121.06` GB/s              |
| **unroll run 2**         | **`13.63`** | `0.226`                 | `121.06` GB/s              |
| `NO_UNROLL` regression | `13.48`   | `0.242` | `115.70` GB/s |

Delta: **`+0.74%`** wall throughput (`13.455 → 13.580` avg) and **`+4.6%`**
on the MoE achieved bandwidth (`115.7 → 121.1` GB/s). Small but **reproducible
across runs and bit-equivalent under `--logprob-vectors`**. Per the handoff
plan promotion rule "improves memory materially" — kept default-on.

This is the single confirmed throughput improvement of the session. It does
not get to `20 tok/s`. But it validates that:

1. The kernel structure has real headroom (`+5%` BW from a pure compiler
   hint), and
2. The 245-GB/s memfloor really is achievable in principle; the question
   is how much engineering effort to recover the remaining `~125 GB/s`.

The remaining gap (`121 → 245` GB/s) requires `cp.async`-based software
pipelining or full TMA staging; both are multi-day novel CUDA work. The
`unroll` kernel keeps that work bounded by establishing what `+5%`
compiler-friendly tuning gets in isolation.

### Phase 1+++: float2 vectorization of attention KV reads (small win)

After studying llama.cpp's FA2 decode path (synchronous `int4` loads, no
`cp.async`, cooperative tile load + online softmax), applied the lowest-risk
piece — vectorized KV reads in the value-accumulate phase of two attention
kernels:

- `attention_indexed_mixed_kernel` (ratio-4 layers at long ctx)
- `attention_decode_mixed_kernel` (ratio-128 layers + short-ctx ratio-4)

Both had the same pattern at decode (`head_dim=512`, `blockDim=256`): thread
`tid` computed outputs `{tid, tid+256}` via two scalar 4-byte loads per row.
Remapped each thread to outputs `{2*tid, 2*tid+1}` and replaced the two
scalar loads with a single `float2` 8-byte load.

Bit-equivalence preserved (each output is summed in the same row order; only
the thread→d mapping changed). Verified via `--logprob-vectors`.

Measurements:

| Stage / metric | Pre | Post | Delta |
| -------------- | --- | ---- | ----- |
| attention sub-stage (host profile) | `6.18 ms/token` | `6.08 ms/token` | **`-0.10 ms`** |
| wall gen tok/s (4-run avg) | `13.575` | `13.625` | **`+0.05 tok/s` (+0.37%)** |

This was the simplest, lowest-risk piece of the FA2 borrow. I had predicted
`+0.2 tok/s` for the 2-hour effort and got `+0.05`. The honest reason:
attention KV reads were already well-coalesced through L2 — the warps were
issuing 4-byte loads but adjacent threads' loads merged into single
transactions at the cache, so halving the per-thread LDG count freed
instruction issue slots without freeing memory pipeline bandwidth. The
FMA chain (still scalar inside `acc.x += v.x * s; acc.y += v.y * s`)
remained the rate-limiter.

For larger attention wins we would need the full FA2 tile-load + online
softmax restructure (the +0.3-0.5 tok/s variant in the earlier plan).
`float2` alone was the small piece; the larger piece requires per-tile
shared memory staging + careful sync overhead bookkeeping.

### Cumulative session improvement (2026-05-11)

| Stage | gen tok/s @ 7047 ctx | Cumulative |
| ----- | -------------------- | ---------- |
| Session start baseline   | `13.48` | — |
| After Phase 0a (spin sched) | `13.48` | no change |
| After Phase 0c memfloor (no kernel change) | `13.48` | diagnostic only |
| After Phase 1 tile sweep | `13.48` | no change |
| **After MoE iq2 unroll** | **`13.58`** | **+0.74%** |
| **After float2 attention** | **`13.625`** | **+1.07%** total |

Two real, bit-equivalent, default-on improvements shipped this session.
Both are small. The path to further gains is documented and characterized:
the FA2-style attention rewrite (additional `+0.2-0.4 tok/s`), MoE
cp.async + iq2 repack (`+0.5-1.0`), and speculative decoding (the only
multi-tok/s lever). None fit in a single session.

### MTP α=0 root cause (2026-05-11)

The handoff plan recorded "MTP drafts tested 2,3,4,6 → Draft acceptance
0.000 extra accepted draft tokens" as the rationale for rejecting MTP
speculation. After challenging my own confidence rating on EAGLE-class
speculation, ran a 1-day investigation into what α=0 actually means.

**Reproduced** with `DS4_MTP_STRICT=1 DS4_MTP_RESTORE_DRAFT_FRONTIER=1`,
`--mtp-draft 2`, ctx 7047, gen-tokens 16: `draft_accept_rate=0.000`,
`reject_cycles=15` of 15 opportunities, `avg_accepted_per_cycle=1.000`
(only the verified token, no extras).

**Bisected** the failure with three layers of env-gated logging
(`DS4_MTP_VERIFY_LOG=1`, `DS4_MTP_CHAIN_LOG=1`):

1. `mtp_gate`: every cycle reports `mtp_ready=1 draft_valid=0 draft_tokens=2`.
   The speculation gate at `ds4_session_eval_speculative_argmax` returns
   early because `s->mtp_draft_valid` is never true.
2. `mtp draft_call`: `metal_graph_eval_mtp_draft` returns `ok=0` on every
   call. The draft head is loaded and the buffers exist, but the eval
   itself fails silently.
3. `mtp_chain`: bisection of `metal_graph_eval_mtp_draft_from_hc`
   identifies the failure at the `decode_layer` step (the per-token
   transformer block applied to the MTP block).
4. `encode_decode_layer first-fail`: inside the decode-layer fn, the
   failure occurs at `stage=routed_moe`.
5. CUDA dispatch reports
   `routed_moe gate_type=12 down_type=12 expert_in=4096 expert_mid=2048 out=4096 (reject: requires gate_type=16 iq2_xxs, down_type=10 q2_K)`.

**Root cause:** the MTP head's routed experts are quantized as **Q4_K
(type=12)**, but the CUDA `routed_moe_launch` dispatch hard-rejects
anything that isn't `gate_type=16 (IQ2_XXS)` + `down_type=10 (Q2_K)`. The
MTP block silently never runs. Every "α=0 cycle" was actually "no draft
ever produced," not "draft produced but verifier rejected."

This **invalidates the prior 'MTP not useful for DSv4-Flash' conclusion**.
The acceptance question is unanswered, not answered with 0. The
prior MTP probe was effectively measuring whether the speculation
plumbing was wired up — not the draft head's quality.

**What the fix requires:**

| Path | Effort | Risk | Outcome |
| --- | --- | --- | --- |
| Add Q4_K routed-MoE CUDA kernel (gate_type=12, down_type=12) | 3-5 days | low-medium (kernel work, no architectural change) | Finally enables MTP to run; first real α measurement |
| Re-quantize MTP head GGUF to IQ2_XXS gate/up + Q2_K down | 1 day quantization + day to validate | medium (further-quantizing the draft head may hurt acceptance) | Same plumbing, no kernel work; tests with lower-precision draft |

Either path unblocks the actual measurement that should drive the
speculation go/no-go decision. The earlier handoff-plan rejection of
MTP/EAGLE rested on a measurement that wasn't measuring what we thought.

**Diagnostic env vars added (left in source):**

- `DS4_MTP_VERIFY_LOG=1` — logs `mtp_gate`, `mtp_should_draft`,
  `mtp_draft_call`, `mtp_path`, and per-cycle verify state.
- `DS4_MTP_CHAIN_LOG=1` — logs per-step status inside
  `metal_graph_eval_mtp_draft_from_hc` and the routed_moe quant-type
  rejection.

These are zero-cost when disabled and were essential to catch a failure
mode that was previously silent.

### Switched to fixed-imatrix model (2026-05-11 evening)

Switched the working model from the older `chat-v2.gguf` to the new
fixed-imatrix gguf downloaded today. Active path going forward:

```sh
/home/cghart/ds4/gguf/fixed-imatrix-b0c3326/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

Recommended symlink update (run from a shell with permission to touch
`/home/cghart/ds4`):

```sh
ln -sfn gguf/fixed-imatrix-b0c3326/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf /home/cghart/ds4/ds4flash.gguf
```

(Until that swap, pass the explicit `-m` path on every bench invocation.)

Baseline on the new model with all session optimizations in place (MoE
iq2 unroll + float2 attention KV reads):

| Run | gen tok/s @ 7047 ctx |
| --- | --- |
| 1 | `13.63` |
| 2 | `13.64` |
| **avg** | **`13.635`** |

This matches our final number on the old `chat-v2.gguf` (`13.625` avg over
4 runs), confirming the optimizations transfer cleanly across models.
Slightly better than the previously recorded fixed-imatrix baseline of
`13.43-13.45` because that earlier measurement predates today's MoE unroll
and float2 attention work.

**Correctness gate status on the new model:**

- `--logprob-vectors`: **FAILS** as expected. The test fixture was
  generated against the old `chat-v2.gguf`; the fixed imatrix changes
  routed-mid weights enough to shift top-1 logits at some positions
  (`step 3 selected token mismatch` on `long_memory_archive`). This is
  a **model-different-than-fixture** failure, not a code regression.
  Action: regenerate the logprob-vectors fixture against the new model
  before relying on `--logprob-vectors` here.

**MTP root cause on the new model:** identical Q4_K MoE dispatch gap.
The MTP head gguf is the same artifact (`MTP-Q4K-Q8_0-F32.gguf`)
regardless of which base model loads it, so the
`gate_type=12 down_type=12 (reject: requires gate_type=16, down_type=10)`
log fires on both. Same fix paths apply.

### Three-item plan (2026-05-11 evening): logprob-vectors regen, Q4_K MoE, MTP α

User requested all three open items in sequence. Item 1 complete, item 2
scaffold complete with decode-math bug remaining, item 3 blocked until
item 2 lands cleanly.

#### Item 1: logprob-vectors fixture regenerated for new model — DONE

The cloud `official.vec` was calibrated against the older `chat-v2.gguf`.
Each imatrix variant produces slightly different outputs at some token
positions, so the strict argmax test fails on the new fixed-imatrix model
even when our inference code is correct.

Added `tests/test-vectors/regen_local_vectors.py` — runs `ds4 --cuda
--dump-logprobs` against each test prompt and emits the same `.vec`
format with **only the selected token per step** (matching the cloud
fixture's structure to avoid CUDA-non-determinism-induced false positives
in lower-rank tokens). Documented in `tests/test-vectors/README.md`.

Workflow now:

```sh
./tests/test-vectors/regen_local_vectors.py \
  -m /home/cghart/ds4/ds4flash.gguf \
  -o tests/test-vectors/local.vec
DS4_TEST_VECTOR_FILE=tests/test-vectors/local.vec ./ds4_test --logprob-vectors
```

Verified `logprob-vectors: OK` on the fixed-imatrix model with the new
`local.vec`. The cloud `official.vec` stays in tree as the quality
reference; `local.vec` is the code-regression gate.

#### Item 2: Q4_K routed-MoE CUDA kernel — SCAFFOLD DONE, MATH BUG REMAINING

The MTP draft head's routed experts are Q4_K (`gate_type=12, down_type=12`).
The CUDA routed_moe dispatch previously rejected anything not
iq2_xxs+q2_K, returning 0 — which is the bug that caused the prior
"α=0" measurement.

**Added to source (in `ds4_cuda.cu`):**

- `cuda_block_q4_K` struct (mirrors `block_q4_K` from `ds4.c`, 144 bytes).
- `dev_q4_K_get_scale_min` helper (canonical `get_scale_min_k4` port).
- `dev_dot_q4_K_q8_K_block` — scalar Q4_K × Q8_K dot product
  (correctness first, performance later).
- `moe_gate_up_mid_decode_q4K_qwarp32_kernel` — Q4_K analog of the
  IQ2_XXS LUT decode kernel.
- `moe_down_q4K_sum6_qwarp32_kernel` — Q4_K analog of the Q2_K
  direct-decode down kernel.
- Routed-MoE dispatch (`routed_moe_launch`) now accepts the Q4_K pair as
  a valid quant combination and routes to the new kernels.

**End-to-end status:**

The new dispatch path runs without error: `mtp_chain decode_layer ok=1`,
`mtp draft_call ok=1`. The MTP path is no longer silently rejected. But
the draft head's output is **garbage** — `mtp_top=0` on every cycle
regardless of input token, which means the routed-MoE in the MTP block
is producing near-zero activations and the vocab head argmaxes to token
0 by default.

The decode math was written against llama.cpp's canonical reference
(`get_scale_min_k4` from `ggml-quants.c` and the QK_K=256 sub-block
layout from `dequantize_row_q4_K`). On paper it matches; in practice it
returns zero. Likely cause is one of:

1. Wrong `qs` nibble unpack stride (low-vs-high nibble assignment per
   sub-block).
2. `get_scale_min_k4` index mismatch for `j ∈ [4,7]`.
3. Q8_K `bsums[]` alignment offset by one sub-block.
4. `__half` byte-order interpretation of `d` / `dmin` on `sm_121`.

**Next step:** add a CPU reference implementation of the Q4_K decode in
`tests/ds4_test.c`, compare CUDA output against it on a synthetic block
with known weights, and bisect from there. Estimated 4-8 more focused
hours.

**Routed-MoE dispatch flag:** `DS4_MTP_CHAIN_LOG=1` now prints
`routed_moe gate_type=X down_type=Y ... (reject: only iq2/q2 or q4_K
supported)` if a future model has yet another quant pairing. Kept as
durable diagnostic for the next "α=0" mystery.

**Performance note:** even when the decode bug is fixed, the Q4_K MoE
kernel is currently a slow scalar loop (~1 µs per block × 16 blocks × 12
gate+up calls × 43 layers ≈ enormous per-MTP-cycle cost). The bench
shows `effective_tps≈3.2` on the broken path vs the `13.4` baseline.
Once correctness lands, the next optimization pass would port the kernel
to use `dp4a`/vectorized loads — but that is separate from unblocking
the MTP acceptance measurement.

#### Item 3: First real DSv4-Flash MTP α measurement — BLOCKED

Blocked on item 2 producing correct output. Once the Q4_K decode is
fixed, run:

```sh
DS4_MTP_STRICT=1 DS4_MTP_RESTORE_DRAFT_FRONTIER=1 \
  ./ds4-bench --cuda -m /home/cghart/ds4/ds4flash.gguf \
  --mtp /home/cghart/ds4/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --mtp-draft 2 --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 128
```

and capture `draft_accept_rate` for the first time on this model. That
number is the actual go/no-go signal for any speculation track on
DSv4-Flash.

#### Final session totals

- Baseline at session start (old model, no optimizations): `13.48 tok/s`
- Baseline now (new model, MoE iq2 unroll + float2 attention): `13.60 tok/s`
- New regression gate working on new model: ✓
- MTP α measurement: still unmeasured, scaffold ready, decode bug to fix

All other gates (`--bench-mtp-spec-source`, `--bench-exact-replay-source`,
`--cuda-indexed-decode-heads8-source`, `--decode-profile-source`): pass.

### Session summary (handoff plan execution, 2026-05-11)

Worked through Tasks 1–5 of `docs/superpowers/plans/2026-05-11-gb10-decode-optimization-handoff.md`:

| Task | Status   | Outcome |
| ---- | -------- | ------- |
| 1    | done     | Baseline `13.48 tok/s` at ctx 7047. Stage attribution stable, matches plan. |
| 2    | done     | `DS4_CUDA_DECODE_EVENT_PROFILE` lands; agrees with host stage profile to within `2%`, `<1%` throughput overhead. |
| 3    | skipped  | Main mechanism (heads8 KV reuse) already rejected; remaining attention sub-stages already at 68-77% of peak. |
| 4    | done     | MoE achieves `115 GB/s` = `42%` of peak. Rewrite justified per plan thresholds. ~`7 ms/token` headroom. |
| 5    | done     | Compressor `5.6 ms/token` ctx-independent; indexer adds `1.3 ms/token` at long ctx. Variants assessed; deferred behind MoE rewrite. |

Verification gates (all passing post-changes):
- `./ds4_test --bench-mtp-spec-source --bench-exact-replay-source --cuda-indexed-decode-heads8-source --decode-profile-source` → OK
- `DS4_TEST_MODEL=... ./ds4_test --logprob-vectors` → OK
- baseline `13.49 tok/s` at ctx 7047 with all new instrumentation in place (no regression vs starting `13.48`).
- `git diff --check` → clean.

**Recommended next workstream:** prototype a tiled/pipelined variant of
`moe_gate_up_mid_decode_lut_qwarp32_kernel` on a synthetic 6-expert fixture
before touching the production dispatch, target `≥180 GB/s` to bring routed
MoE from `15.78 ms/token` toward `~9-10 ms/token` (single largest known
lever toward `20 tok/s`).

## Attention substage event profile (2026-05-12)

Added `DS4_CUDA_ATTENTION_EVENT_PROFILE=1` (ds4_cuda.cu, shares the same per-mark
events as `DS4_CUDA_DECODE_EVENT_PROFILE`). At-exit line breaks the existing
attention region marks (`attn_hc_pre`, `attn_norm`, `q_path`, `kv_path`,
`compressor_indexer`, `attention`, `attn_output`, `attn_hc_post`) into named
substage buckets. Cross-check: substage total equals
`attention_ms_per_token + compressor_indexer_ms_per_token` from the decode
profile, so the bucket mapping is consistent.

Bench at ctx 7047 / 128 gen-tokens against the fixed-imatrix model
(`fixed-imatrix-b0c3326/...-chat-v2-imatrix.gguf`), `gen_tps=13.47`:

| Substage              | ms/token | % attention block |
| ---                   | ---      | ---               |
| output_proj           | 14.57    | 32.4%             |
| compressor_indexer    | 10.53    | 23.4%             |
| q_proj                | 9.74     | 21.6%             |
| attn_kernel (fused)   | 5.98     | 13.3%             |
| attn_hc_pre           | 3.34     | 7.4%              |
| kv_proj               | 0.76     | 1.7%              |
| attn_pre_norm + post  | 0.08     | 0.2%              |
| **attn_total**        | **44.99**| 100%              |

Decode-profile cross-check: `attention=34.456 ms`, `compressor_indexer=10.532 ms`
→ sum `44.988 ms` matches `attn_total` to <1 µs.

**Headline finding (contradicts the SESSION_HANDOFF.md hypothesis that the MLA
expand inside the attention kernel is the lever):**

- The fused attention kernel (`ds4_gpu_attention_indexed_mixed_batch_heads_tensor`
  / `ds4_gpu_attention_decode_heads_tensor`) is only **5.98 ms** — 13% of the
  attention block. KV-from-compressed expand, scores+softmax, and V apply all
  live in this fused kernel and cannot be separated from the host without
  kernel-internal `cudaEventRecord` points; the headline number bounds them all.
- `output_proj` (14.57 ms) is the single biggest attention substage. It is
  `attention_output_a` (head_dim → n_groups·rank, q8_0 matmul) and
  `attention_output_b` (n_groups·rank → DS4_N_EMBD, q8_0 matmul, currently
  fused with HC expand via `matmul_q8_0_hc_expand_tensor`), plus an inverse RoPE.
- `q_proj` (9.74 ms) is `attn_q_b` (q_rank → DS4_N_HEAD·DS4_N_HEAD_DIM, q8_0)
  plus `attn_q_a` and head RMS. The q_b matmul is dominant.

**Implication for the path to 20 tok/s.** The two largest attention substages
are both q8_0 matmuls — same kernel family already touched for shared-FFN and
attention output. Targeting that kernel family (tile/pipeline like the proposed
MoE rework) compounds across `output_proj`, `q_proj`, and the shared-FFN
matmuls; the fused attention kernel itself is not the lever the handoff guessed.

Reproduction:

```sh
DS4_CUDA_ATTENTION_EVENT_PROFILE=1 DS4_CUDA_DECODE_EVENT_PROFILE=1 \
  ./ds4-bench --cuda \
  -m /home/cghart/ds4/gguf/fixed-imatrix-b0c3326/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 128
```

## MoE bandwidth floor (2026-05-12 — invalidates prior "tile to 180 GB/s" plan)

Prior notebook section ("Recommended next workstream") said: target ≥180 GB/s
by tiling/pipelining `moe_gate_up_mid_decode_lut_qwarp32_kernel`. **That plan
is unachievable through kernel-level tile tuning.** Evidence:

1. **Nodecode floor probe.** `DS4_CUDA_MOE_DECODE_NODECODE=1` runs an identical
   memory access pattern but replaces the iq2 LUT decode with a trivial
   XOR-fold. Result: **121.99 GB/s**, essentially identical to production's
   **119.54 GB/s** (`_qwarp32_unroll<16>`). Removing all decode compute does
   not lift the kernel above ~122 GB/s. The iq2 LUT decode is NOT on the
   critical path; the memory access pattern is.

2. **Variant sweep (ctx 7047, 32 gen tokens, achieved GB/s):**

   | Variant | achieved_gbps | gen_tps |
   | --- | --- | --- |
   | `_qwarp32_unroll<16>` (production default) | **121.99** | 13.54 |
   | `_qwarp32_tiled<1>` (TILE=32) | 113.96 | 13.38 |
   | `_qwarp32_tiled<2>` (TILE=64) | 114.53 | 13.42 |
   | `_qwarp32_kernel` (TILE=128) | similar to unroll | — |
   | `_qwarp32_tiled<8>` (TILE=256) | 113.08 | 13.36 |
   | `_qwarp32_tiled<16>` (TILE=512) | 83.87 | 12.45 |
   | `_hwarp16<16>` | 112.93 | 13.35 |
   | `_hwarp16<32>` | 114.04 | 13.37 |
   | nodecode probe | 121.99 | 13.54 |

   **Nothing breaks 122 GB/s.** All variants land in the 113–122 band.
   `TILE=512` is significantly worse (probably blowing shared-mem or
   register budgets).

3. **Why.** The 245 GB/s number from `ds4_gpu_memfloor_bench` is a sequential
   read of a single large device buffer. The MoE pattern is a per-CTA gather:
   each CTA loads a unique expert's weight slice (128 rows × 16 iq2 super-blocks
   = ~32 KB per gate or up). Different CTAs hit different expert tensors with
   no cross-CTA reuse on the decode (single-token) path. The DRAM controller
   sees a stream of unrelated small reads, not a sequential burst, and tops
   out at ~half the sequential-read ceiling on GB10's LPDDR5x.

**Implication for the 20 / 17 / 19 tps plan.** MoE rework via kernel tile
tuning is not the lever the prior notebook page assumed. Realistic MoE
headroom is whatever CUDA Graphs / persistent-kernel restructure can save in
launch and scheduling overhead (~1–2 ms/token at most, not 6–8). The
attention substage profile's `output_proj` (14.57 ms) and `q_proj` (9.74 ms)
become the highest-EV evidence-backed kernel-level levers — both are q8_0
matmuls of *streaming* shape (input × weight, weight read once per call), not
gathers like MoE, so the same memory-pattern floor doesn't necessarily apply.

Reproduction:

```sh
DS4_CUDA_MOE_DECODE_NODECODE=1 DS4_CUDA_MOE_EVENT_PROFILE=1 DS4_CUDA_DECODE_EVENT_PROFILE=1 \
  ./ds4-bench --cuda \
  -m /home/cghart/ds4/gguf/fixed-imatrix-b0c3326/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 64
```

## q_a/kv pair fusion delta (2026-05-12)

Implemented `ds4_gpu_matmul_q8_0_pair_tensor` use in the qkv_rms_fused decode
path of `metal_graph_encode_decode_layer`. Both q_a and kv consume `attn_norm`
as input; the existing pair kernel (`matmul_q8_0_pair_preq_warp8_kernel`) was
previously used only by `ds4_gpu_shared_gate_up_swiglu_q8_0_tensor`.

| Metric | Baseline | With pair fusion | Delta |
| --- | --- | --- | --- |
| `gen_tps` (ctx 7047, 128 tokens) | 13.47 | **13.56** | **+0.09** |
| `q_proj_ms_per_token` | 9.735 | **9.400** | **-0.335** |
| `total_ms_per_token` | 73.819 | 73.330 | -0.489 |
| `--logprob-vectors` local.vec | OK | **OK** | bit-equivalent |

**Reality vs estimate.** The substage-budget table estimated 1.2–2.7 ms savings
from q_a/kv pair fusion. Actual is ~0.5 ms (about 4–5× under). The pair kernel
only shares the input pre-quantize and saves one kernel launch — weight reads
are per-call and dominate. For DS4 decode at n_tok=1 the pre-quantize cost
is small, so the win is mostly from saved launches (≈60 layers × ~5 µs).

**Implication.** The "q8_0 fusion alone clears 15 tps" premise in the current
goal is at risk. Output_a/output_b are sequential (b consumes a's output), so
the pair pattern doesn't apply; a custom fused kernel could save ~0.3–0.5 ms
(one launch + the `low` intermediate buffer round-trip), still bounded by
the same launch-overhead floor. Q8_0 fusion's full ceiling in attention is
~1 ms total, not 3–6 ms.

Disable flag for the fusion: `DS4_CUDA_DISABLE_Q_KV_PAIR=1`.

## Upstream CUDA Lessons

The current upstream implementation should be treated as the new CUDA baseline
for GB10 work.

Relevant upstream mechanisms:

- fd-backed model range cache that reads tensor spans through the model fd.
- pinned staging pool with async host-to-device copies.
- direct I/O fallback when available.
- device arena allocation for cached model ranges.
- incremental file-page and mmap-page dropping while staging.
- cuBLAS/TF32-linked build path.
- fused and specialized decode kernels including QKV/RMS, Q8 HC expand, indexed
  top-k/WMMA paths, and improved MoE routing/down kernels.
- `ds4-bench`, which measures frontier prefill and generation separately.

## Recommended Re-Measurement Commands

Use upstream `ds4-bench` for GB10 baseline refresh:

```sh
./ds4-bench \
  --cuda \
  -m ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 \
  --ctx-max 7047 \
  --gen-tokens 128
```

Use a cache-dropped cold probe when comparing staging paths:

```sh
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
/usr/bin/time -v ./ds4 --cuda -m ds4flash.gguf --ctx 4096 \
  -n 16 --temp 0.0 -sys '' --nothink \
  -p 'Complete this sentence: CUDA accelerates'
```

Avoid profiler-heavy commands in normal measurement runs. If profiler data is
needed, prefer Nsight Systems first.

## F16 GEMV vectorization wins (2026-05-12, post-q_a/kv-pair-fusion)

Investigative-goal session. Baseline at session start (after q_a/kv pair
fusion, all prior optimizations): **13.66 tok/s** average over 3 runs (clean
bench, no profilers). Session result: **15.00 tok/s** (+1.34 tps; -6.09 ms
total_ms_per_token in the substage profile, 73.16 → 67.07).

Two evidence-backed kernel-level levers landed; one (CUDA Graphs) deferred.

### Lever 1: `output_a/output_b/HC-expand` sequential fuse — small, real

Eliminates the float `low` intermediate roundtrip + one `quantize_q8_0_f32`
launch between `attn_output_a` and `attn_output_b`. Added kernel
`grouped_q8_0_a_preq_to_q8_kernel` and wrapper
`ds4_gpu_attention_output_q8_fused_hc_tensor`; wired into the production
decode path (`fuse_attn_out_hc=true`) behind `DS4_CUDA_DISABLE_OUT_AB_FUSE=1`.

**Design note — first design regressed.** Initial CTA had 8 warps × 4-outputs-
per-warp sequential. That dropped CTA count to 1/4 of the original and added
an intra-warp sequential loop; bench went *down* by ~0.04 tps. Final design
uses 1024 threads (32 warps × 32 lanes) per CTA — each warp computes one
output, matching the original per-warp parallelism. CTA count drops from
1024 → 256 (one per Q8 block) but total threads-per-token stays the same.

| Metric | fuse-off | fuse-on (final) | Δ |
| --- | --- | --- | --- |
| gen_tps (clean) | 13.66 | 13.73 | +0.07 |
| output_proj_ms_per_token | 14.71 | 14.38 | -0.33 |
| total_ms_per_token | 73.77 | 73.16 | -0.61 |
| `--logprob-vectors` local.vec | OK | OK | bit-equivalent |

This matches the calibration note's expected ceiling (0.3-0.5 ms) almost
exactly, but the bench delta is small because the saving is mostly launch
overhead, which is partially overlapped with GPU work on the spin-scheduled
path.

### Lever 2: F16 GEMV uint4 vectorization — the big lever

`matmul_f16_pair_ordered_chunks_kernel` and `matmul_f16_ordered_chunks_kernel`
were doing scalar `__half2float(wr[i])` reads in a chunks-of-128 loop with
a shared-memory tree reduction. Both kernels were sitting at ~44% of the
GB10 sequential-read bandwidth ceiling.

**Before stage breakdown** (`DS4_CUDA_INDEXER_EVENT_PROFILE=1`, ctx 7047,
128 tokens, fuse-on, pre-vec8 baseline):

```
compressor_calls=7936 compressor_ms_per_token=5.511
indexer_scores_calls=2688 indexer_scores_ms_per_token=0.608
indexer_topk_calls=2688 indexer_topk_ms_per_token=0.673
                          total_ms_per_token=6.793
```

**Theoretical floor**: each pair call reads ~16 MB (attn compressor)
or ~4 MB (indexer compressor) of F16 weights from DRAM with no reuse across
CTAs. At GB10's 245 GB/s sequential ceiling, ~620 MB/token of compressor
weight reads should land at ~2.5 ms. Measured 5.51 ms was 44% efficient.
Scalar `__half2float` in a tight 128-iteration loop bottlenecks the load
path before saturating DRAM.

**Fix**: added `matmul_f16_pair_warp_vec8_kernel` and
`matmul_f16_warp_vec8_kernel`, gated on `in_dim % 8 == 0` and the
`DS4_CUDA_NO_F16_PAIR_VEC8` / `DS4_CUDA_NO_F16_VEC8` disable flags.
Mechanism:

- One `uint4` (16-byte) load reads 8 halves at a time → 8× per-LSU-op
  density vs scalar reads.
- `__half22float2` for the half2 → 2 floats unpack.
- `warp_sum_f32` (warp-shuffle) reduction instead of the shared-memory
  + serial-32-sum tree.

Same observable result as before (low-bit reduction-order diff well within
fp16-input rounding; `--logprob-vectors local.vec` stays OK).

**After stage breakdown** (vec8 ON, attention-event + decode-event profilers):

```
compressor_ms_per_token=2.936  (5.511 → 2.94, -2.57 ms)
indexer_scores_ms_per_token=0.606
indexer_topk_ms_per_token=0.686
total_ms_per_token=4.228       (6.793 → 4.23, -2.57 ms)

compressor_indexer_ms_per_token=8.424  (10.98 → 8.42 with profiler,
                                       6.48 without; the bare
                                       attention-event profile shows -4.5 ms)
shared_ffn_ms_per_token=8.79           (-0.83 ms — the single-output vec8
                                       also helps hc_ffn_fn)
attn_hc_pre_ms_per_token=2.51          (3.34 → 2.51, -0.83 ms — same path)
total_ms_per_token=67.07               (73.16 → 67.07, -6.09 ms)
```

Compressor pair achieves ~211 GB/s after vec8 (86% of the 245 GB/s ceiling).

**Why the single-F16-vec8 change moved more than just `indexer_proj`/`q_b`:**
the `ordered_router` dispatch branch is taken on most single-token F16
GEMV calls, including `hc_ffn_fn` (used by both `attn_hc_pre` and
`shared_ffn`'s mix path). The win compounds across every F16 GEMV in the
decode loop.

### Clean A/B (all changes on vs all changes off, same binary)

3 bench runs each, all profilers off:

| Config | Run 1 | Run 2 | Run 3 | Avg |
| --- | --- | --- | --- | --- |
| All-off (DISABLE_OUT_AB_FUSE + NO_F16_PAIR_VEC8 + NO_F16_VEC8) | 13.67 | 13.68 | 13.63 | **13.66** |
| All-on (defaults) | 15.02 | 14.99 | 14.99 | **15.00** |

Cumulative session delta: **+1.34 tps**, **-6.09 ms** total_ms_per_token.

### Falsification of the prior session's calibration note

The prior session's hand-off ("MoE 0.5-1.0 ms vs estimated 1.5-3.0; q8_0
fusion 1.0 ms vs estimated 3-6 — consistent 3-5× optimism bias on kernel-
side estimates") was a real and useful warning, but it was also conditioned
on the prior session's evidence set. **F16 GEMV vectorization went the
other way**: estimate (implicit) was "F16 path already well-tuned; no
notebook entry for it" — actual ~+1.3 tps. The prior pessimism didn't apply
because the F16 path was, in fact, near-untuned, and the ~244 GB/s sequential
ceiling on GB10 was the right floor to compare against (versus the 122 GB/s
gather-style MoE pattern, which was a separate ceiling).

Net lesson: the optimism bias was true for the specific kernels touched
(MoE, q8_0 fusion), not a universal multiplier. Each new probe should set
its own theoretical floor before measuring.

### Step 3 (CUDA Graphs) — deferred with rationale

Original plan: capture the per-token decode launch sequence into
`cudaGraph_t` and replay to eliminate per-launch overhead. Scoping outcome:

- Per-layer kernel count ≈ 20-25; total per-token launches ~1000.
- Per-launch overhead on spin-scheduled GB10 ≈ 3-5 µs ⇒ 3-5 ms theoretical
  CPU enqueue cost.
- Wall-clock impact is `max(CPU_enqueue, GPU_run)`. At 67 ms/token GPU vs
  ~5 ms CPU, the GPU is the bottleneck and the CPU enqueue is fully
  overlapped today.
- Real CUDA-Graph wins come from *gaps in the GPU schedule* where short
  kernels finish before the next launch arrives. Estimated such gaps:
  0.5-1.5 ms (~0.1-0.3 tps).
- Cost: a structural refactor — every per-token state (pos, KV write
  offsets, comp_row, top-k indices) is embedded in kernel args. The graph
  must either re-capture per-token (defeats the purpose) or be
  parameterized via `cudaGraphExecKernelNodeSetParams` / device-side
  pointer indirection.

Verdict: deferred. Concrete future-session entry point:

1. Add a per-layer event-mark probe that records both `cudaEventRecord`
   begin/end *and* a host-side `clock_gettime`-from-launch timestamp.
   Difference between consumer-side end and host-side last-launch gives
   the GPU-idle-while-waiting-for-CPU window. If <0.5 ms total, abandon
   CUDA Graphs.
2. If the window is meaningful, capture only the *inner* per-layer body
   (post-pos-dependent fanout, pre-residual-merge) into a graph, with
   `cudaGraphExecUpdate` for the offsets that do change.

### Updated reproduction commands

```sh
# Build
make ds4-bench ds4_test

# Baseline at session-start (all session changes disabled)
DS4_CUDA_DISABLE_OUT_AB_FUSE=1 DS4_CUDA_NO_F16_PAIR_VEC8=1 DS4_CUDA_NO_F16_VEC8=1 \
  ./ds4-bench --cuda \
  -m /home/cghart/ds4/gguf/fixed-imatrix-b0c3326/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 128
# Expect ~13.65 tok/s

# Full session state (defaults — all wins enabled)
./ds4-bench --cuda \
  -m /home/cghart/ds4/gguf/fixed-imatrix-b0c3326/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --prompt-file bench/promessi_sposi.txt \
  --ctx-start 7047 --ctx-max 7047 --gen-tokens 128
# Expect ~15.00 tok/s

# Correctness gate
DS4_TEST_MODEL=/home/cghart/ds4/gguf/fixed-imatrix-b0c3326/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
DS4_TEST_VECTOR_FILE=tests/test-vectors/local.vec \
  ./ds4_test --logprob-vectors

# Stage breakdown including new compressor split
DS4_CUDA_INDEXER_EVENT_PROFILE=1 DS4_CUDA_DECODE_EVENT_PROFILE=1 \
  ./ds4-bench --cuda -m /home/cghart/ds4/ds4flash.gguf \
  --prompt-file bench/promessi_sposi.txt --ctx-start 7047 \
  --ctx-max 7047 --gen-tokens 128
```

### Env-flag cheat sheet (new disables added this session)

| Flag | Purpose |
| --- | --- |
| `DS4_CUDA_DISABLE_OUT_AB_FUSE=1` | Disable the fused output_a/output_b/HC-expand path; restores the two-call sequence |
| `DS4_CUDA_NO_F16_PAIR_VEC8=1` | Force the scalar `matmul_f16_pair_ordered_chunks_kernel` instead of the new `_warp_vec8_kernel` |
| `DS4_CUDA_NO_F16_VEC8=1` | Same, but for the single-output F16 GEMV (ordered_router branch) |

### Open work, ordered by EV (post-session)

| Item | Effort | Expected tok/s | Notes |
| --- | --- | --- | --- |
| **GPU-idle-window probe + CUDA Graphs** | 1-2 sessions | +0.1–0.3 tps | See "Step 3" above; verify gap exists before refactoring |
| q8_0 GEMV vectorization audit (other Q8_0 paths) | 1 session | +0.0–0.3 tps | Same dp4a-via-larger-vector-load pattern as Q4_K had; check `matmul_q8_0_preq_warp8_kernel` family for analogous scalar loads |
| Compressor_update kernel review (uninstrumented 4 ms) | 1 session | +0.0–0.5 tps | APE + norm + EMA + FP8; not yet split into substages |
| MoE rework via launch-pattern restructure | weeks | +0.0–1.0 tps | Bandwidth floor (~122 GB/s gather) holds; only structural redesign moves it |
| MTP / verifier-bound speculation | weeks | unknown | Still gated by the verifier-sharing problem; see "MTP isn't the lever" in SESSION_HANDOFF.md |

