# Decode campaign — handoff (session of Aug 21, M3 Ultra, round 2)

Goal: push greedy decode past **45 tokens/s** on the MXFP4 `ds4flash.gguf`
model, bit-exact. **Reached: 45.5 t/s interleaved (45.47/45.55), all
transcripts md5 `db0c504c…`, `make test` 44/44, harness bit-identical.**
Round-1 state was 43.2–44.0 t/s; round 2 added +2.7% (attention parity) and
+1.75% (greedy chain decode), both validated per the campaign protocol.

## Round-2 changes (working tree; commit state at bottom)

1. **Raw-layer gathered attention** — `n_comp == 0` decode layers (ratio 0/128)
   now use the gathered path (fused staging + packed32 reduce + fused inverse
   RoPE) instead of the five-dispatch raw path. Found via commit-only stage
   counters: the raw path cost ~65 µs vs ~32 µs per layer, invisible in the
   averaged ledger ("attention core 44.3 µs flat"). ~0.7 ms/token at short
   context. Rollback envs: `DS4_METAL_DISABLE_DECODE_RAW_GATHERED_ATTN`,
   `DS4_METAL_DISABLE_DECODE_RAW_PACKED32`. Also fixed the staging wrapper's
   `n_comp == 0` guard in `ds4_gpu_encode_flash_kv_stage_f16`.
2. **Greedy chain decode** — `metal_graph_greedy_chain` (ds4.c, engaged from
   `generate_metal_graph_raw_swa` only): GPU argmax writes the next token id
   into a device ring; the next token's embedding gathers it from the ring
   (the by-value embed and the batched embed use the same get_rows/repeat
   kernels — bit-identical). Encode runs two tokens ahead of the host's
   confirm cursor; the host only lags (one MTLSharedEvent wait per token) to
   print and stop-check. Removes the per-token `waitUntilCompleted` + 517 KiB
   logits readback + CPU argmax boundary (~0.5 ms/token). Kill switch:
   `DS4_DISABLE_GREEDY_CHAIN=1`. Diagnostics: `DS4_GREEDY_CHAIN_DEBUG`,
   `DS4_GREEDY_CHAIN_DUMP_IDS`, `DS4_GREEDY_CHAIN_VERIFY`.
   **Hash-layer gotcha that cost an hour**: the first `DS4_N_HASH_LAYER`
   layers route experts by token id (`ffn_gate_tid2eid`); the select kernel
   already supports a device-resident token (`use_token_buffer` in
   `kernel_dsv4_router_finalize_one`), plumbed via
   `ds4_gpu_router_select_tensor_devtoken` + `g->chain_token_view`. The
   host-side `metal_graph_decode_set_hash_selected_override` is skipped in
   chain mode — the resident fixed-route MoE never consumes it (the override/
   readback blocks live under `use_selected_slots`, all gated on
   `g_ssd_streaming_mode`). Feeding token=0 instead was the one divergence:
   deterministic drift from layer 0, tipping an argmax 33 tokens in.
   Symptom signature if it regresses: transcripts match for N tokens then
   diverge deterministically.

## Verified numbers (interleaved CLI, `-c 8192 -n 128 --temp 0`, lighthouse prompt)

| variant | t/s | md5 |
|---|---|---|
| round-1 baseline | 43.40–43.52 | `db0c504c…` |
| + attention parity only | 44.59–44.67 | `db0c504c…` |
| + greedy chain (current) | **45.47–45.55** | `db0c504c…` |

Long-context (2K prompt): 44.12 chained vs 43.40 classic. 1024-token run and
an early-stop prompt: md5-identical. Harness (prefix 2048): 529 rows /
68,389,120 logits / 528 ids bit-identical, 43.18 vs 43.10 t/s (only layers
0–1 qualify at long prefix). SSD streaming decode smoke: OK (chain declines,
classic path taken).

## Remaining headroom (re-estimated)

GPU busy ≈ 21.85 ms/token now; wall ≈ 21.96 ms. The boundary is ~0.1–0.15 ms
(argmax→embed dependency + drain). Still open, expectations lowered by the
round-1 evidence: HC epilogue tail fusions (~0.3 ms), KV-staging direct read
(~0.25 ms; a prior attempt diverged — the addressing facts are in README's
stage-counter section and the code comments at cpy.metal:147), MXFP4
concurrent shared-expert stream (machinery exists hard-gated to IQ2,
ds4_metal.m:9286; medium confidence, and the "more parallelism throttles this
GPU" lesson applies). Each could add ~0.2–0.6 t/s; none is needed for 45.

## Watch item

The balanced harness failed twice at `step=0 variant=control` ("metal decode
failed") with an early round-2 binary, then passed 9+ consecutive runs with
semantically identical code. Unexplained; both failures immediately followed
sustained benching (the documented thermal-transient window). If it recurs,
reproduce with `--warmup 1 --tokens 2` and capture full stderr.

## Tools (unchanged from round 1)

- Commit-only stage profiler: `DS4_METAL_DECODE_STAGE_PROFILE=1
  DS4_METAL_STAGE_COUNTERS=1 ./ds4 …` (trustworthy; end-and-wait inflates).
- Balanced A/B harness: `make metal-decode-schedule-bench && ./speed-bench/
  metal_decode_schedule_bench -m ds4flash.gguf --candidate-env NAME
  --include-selection --tokens 512` (aborts unless bit-identical).
- Transcript md5 oracle: `./ds4 -m ds4flash.gguf -p "Write a short story
  about a lighthouse keeper." -c 8192 -n 128 --temp 0 | md5` → `db0c504c…`.
- Tensor dump bisect: `DS4_METAL_GRAPH_DUMP_PREFIX=/tmp/x
  DS4_METAL_GRAPH_DUMP_NAME=<stage> DS4_METAL_GRAPH_DUMP_LAYER=N
  DS4_METAL_GRAPH_DUMP_POS=P ./ds4 …` (synchronizes and dumps; comparing
  classic vs chain dumps located the hash-router divergence in one pass).
- One `ds4` instance at a time; idle ~60 s after heavy runs (transient
  2–10× slowdown recovers by itself).

## Round-1 closed avenues still stand

Eight bit-exact kernel variants (all ≤0.03%), DSpark strict/non-strict
(43.07 / peaks 40.2), three microbatch increments, split-schedule sweep
(default 4 optimal), readback-bubble premise (disproved). Details in
speed-bench/README.md. The round-1 verdict "bit-exact and >45 t/s are
jointly infeasible" was wrong: the ledger's averaged `attn_inv_rope` line
hid the 65/32 µs parity alternation, and the boundary's "wake/launch
latency" was recoverable by keeping the token id on-device.

## Round-3 addendum (Aug 22): session chain + headroom list exhausted

- **Session chain decode**: `ds4_session_eval_chain_greedy` (ds4.c) reuses
  `metal_graph_greedy_chain` for session API callers; `ds4-eval` decodes in
  bursts capped to stay out of the think-close controller window. Bit-exact
  (traces identical), eval decode **44.5 → 45.2 t/s**. Kill switch
  `DS4_DISABLE_GREEDY_CHAIN=1` covers both CLI and session paths.
- **MoE down-sum6+HC4 tail fusion**: landed bit-exact, speed-neutral,
  default on (rollback `DS4_METAL_DISABLE_DECODE_MOE_HC_FUSION`).
- **KV-staging direct read**: implemented bit-exact (wrap-safe) but −7%
  (per-head F32 re-read/re-convert amplification); landed gated OFF
  (opt-in `DS4_METAL_ENABLE_DECODE_RAW_DIRECT_KV`).
- The round-2 "remaining headroom" list is now closed out: both quantified
  items measured (neutral / negative), the concurrent shared-expert stream
  stays contraindicated by the throttling evidence. Wall−GPU gap is ~0.1
  ms/token; the next real gain must come from the MoE or attention core
  itself.
