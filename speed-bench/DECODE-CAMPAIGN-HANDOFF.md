# Decode >45 t/s campaign — handoff (session of Aug 21, M3 Ultra)

Goal: push greedy decode past **45 tokens/s** (towards 50) on the MXFP4
`ds4flash.gguf` model, bit-exact. Final verified state: **43.2–44.0 t/s**,
all paths in agreement. Goal not reached; every in-session path was built,
measured, and closed. This file is the cold-restart kit: state, evidence,
tools, closed avenues, and the decision the next session must make first.

## Where everything is

- Working tree: clean at `814a933`; tests `make test` PASS 44/44; bit-exact
  output md5 for the standard prompt unchanged (`db0c504c…`).
- Campaign commits (this session): `11689e1` (commit-only GPU stage profiler
  + docs), `b40f33c`/`5f9da0a`/`06ca424`/`c2084a1`/`814a933` (DSpark
  measurements, corrections, negative-result records, floor recalibration).
- `speed-bench/README.md` holds the full measurement record: per-stage decode
  ledger, every A/B protocol, and the DSpark-on-Metal work order. Read the
  sections "Metal decode stage GPU counters" and "DSpark speculation on M3
  Ultra" before doing anything.

## Verified numbers to trust (and how to reproduce)

| metric | value | command |
|---|---|---|
| CLI decode | 43.2–43.9 t/s | `./ds4 -m ds4flash.gguf -p "Write a short story about a lighthouse keeper." -c 8192 -n 128 --temp 0` |
| repo bench | 43.77 steady @ctx2048 | `./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 2048 --gen-tokens 96` |
| balanced harness | 43.38 t/s | `make metal-decode-schedule-bench && ./speed-bench/metal_decode_schedule_bench -m ds4flash.gguf --include-selection --tokens 512` |
| best DSpark | 39.5–40.2 t/s | `./ds4 -m ds4flash.gguf --mtp gguf/DeepSeek-V4-Flash-DSpark-support.gguf --dspark --dspark-confidence 0.75` (+`DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP=0`) |

Thermal envelope is ±2%: sustained runs sit ~43.2–43.5, first run on a cool
machine reaches 43.9–44.01. Always compare via the interleaved harness, and
let the machine idle ~60s after heavy runs (a transient 2–10× slowdown right
after sustained benching was observed repeatedly; it recovers by itself).

## The token ledger (22.6 ms GPU busy; encode 0.7 ms hidden by split-flush)

Per layer (µs, commit-only counters, short ctx): routed MoE 139 (~floor for
6×12.6 MB experts at ~550 GB/s effective), attention core 44.3 (flat vs
context; latency floor), attn output A+B 111 (645 GB/s ≈ wall), Q-lora path
120 (quad kernel 41 + q_b 59 + norm/rope 21), HC pre 2×19.4 (structural
floor), shared/router overlapped 44, KV staging 7.8. Weights memory floor
≈11.6 ms/token; the ~10 ms above it is distributed latency that resists
every single-kernel fix tried (see "Closed avenues"). Boundary tail:
CPU argmax 35.7 µs, loop ~0.05 ms recoverable, remainder wake/launch
latency. Bit-exact recoverable stack sums to ~0.55 ms < the ~0.7 ms needed
for 45 t/s.

## Closed avenues — do not redo (details + numbers in README)

1. Eight bit-exact kernel variants, all validated bit-exact, all ≤0.03%:
   HC tgstash, HC rows12 (10-TG sibling), quad NR1, packed attention sg16
   and sg32 (both +8 ms/token systemically — more parallelism throttles the
   whole token on this GPU), FP8 block one-pass amax, down r4 (slots6 and
   static paths), plus an earlier q8 nr0 tune (warm-cache artifact only).
2. DSpark strict: 43.07 t/s (no gain by design). Non-strict: peaks 40.2;
   knob space fully swept (confidence 0.75 optimal, scheduler pauses off).
3. Three microbatch increments: per-row routed MoE (bit-identical,
   verify_layer unchanged), per-row HC pre (slower), and a genuine dual-row
   HC-pre kernel — proven bit-exact via the strict oracle, no per-verify
   gain. Conclusion: the N=2 verify near 50 ms is close to its real floor;
   the "perfect sharing" 29–33 ms floor is likely undeliverable here, so
   speculation probably never beats plain decode on M3 Ultra.
4. Split schedule (`DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS`): default 4 optimal.
5. Readback-bubble premise: disproved — MXFP4 static-trip reads expert ids
   on-device; there is no CPU readback in decode.
6. Stale CSVs in speed-bench/ predate current code; regenerate before
   comparing historical numbers.

## Tools built this session (reuse them)

- **Commit-only stage profiler** (committed): 
  `DS4_METAL_DECODE_STAGE_PROFILE=1 DS4_METAL_STAGE_COUNTERS=1 ./ds4 …`
  prints per-stage GPU busy spans whose sum matches production GPU-busy
  (~22.5 ms/token) — trustworthy, unlike the end-and-wait profiler which
  inflates stages ~6× and serializes the schedule. For the batch/verify
  path also export `DS4_METAL_LAYER_STAGE_PROFILE=1` (and see the reset/
  report wrap pattern used around the verify call in the session log).
- **Strict-mode oracle** for any verify-kernel work: `--dspark-strict`
  output must match plain decode md5 exactly. It caught a real 5-argument
  kernel misbinding that ordinary testing missed.
- **Balanced A/B harness** (`speed-bench/metal_decode_schedule_bench`) with
  `--candidate-env NAME --include-selection`: interleaved, bit-exactness
  enforced over full-vocab logits. Acceptance threshold the project used:
  ≥0.3%.
- Microbench pattern (cold-cycling 6 weight buffers to defeat L2) for any
  standalone kernel work — note the lesson: warm-cache microbench wins
  (e.g. q8 nr0=8) evaporate in production cold streaming.

## Gotchas learned the hard way

- Adding Metal kernel parameters: insert new buffer args **after** existing
  ones or renumber the host bindings to match — a mid-signature insertion
  silently misbinds everything downstream (caught only by the strict oracle).
- The end-and-wait stage profiler changes the schedule (it disables the
  concurrent shared-expert overlap); the commit-only mode does not.
- `misc/` is gitignored; anything that must survive belongs in a tracked
  path like speed-bench/.
- One `ds4` instance at a time (instance lock is intentional; 145 GB
  resident model).
- DSpark stats: per-verify averages = verify_layer/(full+partial), not
  per-cycle; the two disagree wildly when no_draft is high.

## Decision needed before any work resumes

The session's two objectives — "bit exact" (first message) and ">45 t/s"
(goal) — are jointly infeasible on this hardware per the committed
arithmetic. Next session must pick first:

1. **go** — multi-day small-N batched-kernel DSpark verifier build, dropping
   bit-exactness. Fair warning: triple-confirmed evidence says it likely
   tops out below 45 on M3 Ultra anyway. If attempted, start from the
   per-stage work order in README's DSpark section; batched output
   projection and N≤2 KV-sharing attention are the only stages left
   untried, and expectations should be low.
2. **grind** — bit-exact persistent-kernel work (HC epilogue tail-fusions
   ~0.3 ms + KV-staging elimination ~0.25 ms + boundary ~0.05 ms ≈ 0.6 ms
   best case → ~44.5 t/s). The KV-staging direct-read kernel was started
   once (raw-rows-first layout, zero-mask semantics worked out) but
   diverged bit-wise and was cut; the addressing notes are in the session
   history — the layout facts in README are verified.
3. **accept** — 43.3–44.0 t/s is the M3 Ultra equilibrium; the campaign
   artifacts stand as the deliverable.
4. **different hardware** — M5-class parts change the latency-floor math
   (more L2, different power behavior); the profiler + ledger apply
   as-is there.

Quick first command next session:
`make && ./ds4 -m ds4flash.gguf -p "Write a short story about a lighthouse keeper." -c 8192 -n 128 --temp 0`
→ expect ~43.3–43.9 t/s on a cool machine; then decide.
