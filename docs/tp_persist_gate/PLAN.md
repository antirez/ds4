# Persistent TP Gate Kernel — Plan

Task checklist. `[REQ-x]` = REQUIREMENTS.md, `[DES-§n]` = DESIGN.md.
All new TP code goes behind `DS4_TP_PERSIST_GATE` (R1); every task's
acceptance criterion is listed.

**Outcome (2026-09-03, closed): NEGATIVE RESULT — REVERTED.** Everything
below was implemented and validated (loopback 26/26, pair runs bit-exact,
NHI clean), but the pair A/B is slower with the flag on: legacy 13.99
tok/s vs persistent 11.92 tok/s. A per-gate trace joined across the two
ranks showed the decode gate window is ~245 µs on *both* paths and is
made of NHI one-way latency (217 µs) + MoE routing skew (191 µs mean,
symmetric); the resident kernel trimmed ~20 µs/gate and cost ~15 ms/token
in the compute segments between gates. The "38 ms/token of gate
choreography" premise came from a biased `DS4_TP_GATE_STATS` mean (it
included prefill big gates and dropped negative-window gates). Full
numbers, decomposition and the remaining TP levers: BASELINE.md.

What is kept: this doc set, `persist-gate.patch` (the complete kernel,
engine wiring, guarded runtime wrappers, gate stats/trace and loopback
test — applies to `6c090eb`), `tp_gate_trace.py` (trace decomposition),
and `scripts/tp_ab.sh` (two-node A/B harness; the `persist` argument and
the `DS4_TP_GATE_STATS`/`DS4_TP_GATE_TRACE` env vars are no-ops without the
patch). Source changes were reverted from the tree and the nodes.

Run history: base02/leg03/leg08 legacy 13.98–14.00 tok/s; per03 hung
(device-wide runtime calls vs the resident kernel → guarded wrappers);
per04 11.59 (scalar copies); per05 never ran; per06 11.98 (vectorized
copy — loopback floor 41 → 11.5 µs, pair unchanged); per07 11.92 (traced).

## Feasibility check (why continue)

From commit `6c090eb` the decode kernels already stream at the LPDDR5X
ceiling (~202–222 GB/s), so per-rank compute cannot get faster. The TP
token (~72 ms at 13.98 tok/s) decomposes into ~34 ms per-rank compute plus
86 gates × ~0.44 ms ≈ 38 ms of gate choreography; the compute figure is
consistent with a ~77 GiB shard streaming at ~200 GB/s, so the 38 ms is
real overhead. The loopback burst test measures the persistent path's
own floor (host publish → rx_done, no wire) at **~41 µs/gate p50, 53 µs
p99**, i.e. ~3.5 ms/token if the wire adds nothing. With the Thunderbolt
hop the realistic band is 0.1–0.2 ms/gate → 43–51 ms/token → 19–23
tok/s, which is the R12 band. Kill-switch: if the pair A/B per-gate cost
stays ≥ 0.3 ms with the flag on, stop and write it up.

## Phase 0 — Baseline and harness

- [ ] **P0.1** Record the current per-gate cost and TP decode baseline on
  the pair (pinned clocks, new binary with the flag OFF — the flag-off
  path is unchanged and it keeps the A/B on one build): per-gate µs via
  `DS4_TP_GATE_STATS=1` (GPU-timeline stamp-released → RX-copy-done, same
  meaning on both paths; printed at shutdown), TP decode tok/s
  (expect ~13.9), and the 197/1609-token prefill numbers. Save as the
  reference table in this doc set (`docs/tp_persist_gate/BASELINE.md`).
  Scope note (from the full ds4.c read): the measured per-gate cost is the
  DECODE row-gate schedule (86/token, `ds4_engine_tp_gate_schedule`);
  prefill rides `ds4_gpu_tp_big_gate_encode` big-gate splits, which stay
  on the existing path during the POC (R8) and are measured only as the
  end-to-end prefill checkpoint numbers.
  Acceptance: numbers captured with the same launch commands the A/B uses.
  (R11.3)
- [x] **P0.2** Fix the TP A/B launch script (leader-first startup order,
  from the 09-03 NHI debugging session): `scripts/tp_ab.sh` that starts
  coordinator → waits for `waiting for worker` → starts worker, and
  runs the fixed benchmark requests. Acceptance: one clean end-to-end run
  on the pair + NHI stats line archived.
  **Done 2026-09-03** (`base02`: 13.98 tok/s decode, NHI `failures=0
  event_drops=0 crc=0 overrun=0`). Root causes were harness bugs, not
  node state: `cd && nohup CMD &` backgrounded the whole AND-list and held
  ssh's stdout; timeouts too short for a cold ~77 GiB shard load. Later
  addition: preflight restores `cap_sys_rawio` on the binaries (a relink
  drops the file capability → `TBSTREAM_ZC_IMPORT: Operation not
  permitted`) and `report` fails when no bench CSV line exists.

## Phase 1 — Control plane (no behavior change)

- [ ] **P1.1** Add the mapped control-plane alloc/dealloc helper
  (pinned, mapped host memory, like `g_tp_engine_state_host`) with the
  flag arrays + ring from [`DES-§3`]; no kernel uses it yet. Acceptance:
  builds clean; alloc/free under Valgrind/ASAN-logic review (no leaks on
  shutdown path). (R7)
  **Implemented 2026-09-03**: `DS4_ROCM_TP_PERSIST_*` layout + struct in
  `rocm/ds4_rocm_tp.cuh`; engine globals, `ds4_rocm_tp_persist_publish`
  (windowed, mutex-guarded, release publish) in `ds4_rocm.cu`. On-node
  build/valgrind still pending (no ROCm compiler on the Mac host).
- [ ] **P1.2** Engine init: when the env flag is set, allocate the
  control plane in `ds4_gpu_tp_nhi_init`; on any failure log + fall back
  to the legacy engine (flag-internal teardown mirrors `ds4_gpu_tp_shutdown`).
  Acceptance: flag set on both nodes still serves TP (legacy fallback
  path exercised by forcing an alloc failure). (R1, R6)
  **Implemented 2026-09-03**: opt-in under `DS4_TP_PERSIST_GATE=1`;
  alloc/mapped/stream/launch with log + legacy fallback on every failure
  point. Forced-failure drill pending.
- [ ] **P1.3** Teardown ordering for the new planes in
  `ds4_gpu_tp_shutdown` (after thread join, before NHI close).
  Acceptance: clean shutdown, no leaks, `tests/test_tp_combine_rocm`
  still 100%. (R7)
  **Implemented 2026-09-03**: builder-drain gate (`building == 0`) before
  the kernel shutdown word (prevents a late publish stranding on a dead
  kernel), then join → device sync → persist stream sync → destroy/free.
- [x] **P1.4** Pre-flight fixes found in review + on-node loopback
  (all 2026-09-03, gate P4.1):
  1. Init ordering: the resident kernel was launched before
     `g_tp_engine_state_dev` existed, so its dispatcher saw a NULL latch,
     computed `started=false` and exited on an empty ring; the first gate
     then spun forever on `tx_released`. Persist init now runs after the
     state word is mapped.
  2. Init-failure teardown never wrote the shutdown word but still waited
     on the persist stream. The word is now stored unconditionally in the
     persist teardown block.
  3. Wave size: the kernel is written for wave32; compile-time `#error`
     on `__AMDGCN_WAVEFRONT_SIZE__ != 32` plus a runtime
     `hipDeviceAttributeWarpSize` check that falls back to legacy.
  4. **Lost dispatcher→worker handoff (real hang, ~90% of burst runs):**
     every lane read the shared task word, and a 32-lane LDS read is
     serviced over several cycles, so the dispatcher's store could land
     mid-instruction and hand lanes different values. `if (task == 0)`
     then diverged: some lanes looped idle, the rest entered the task
     body without lane 0 and parked in the ack wait forever. All
     wave-wide control reads of concurrently-written shared words now go
     through `readfirstlane` (`dsv4_tp_wave_u32/u64`). Diagnosed via the
     debug words the kernel now exports after the ring.
  5. `s_tx_done/s_rx_done` were initialized to 0 == ring index 0, letting
     the dispatcher ack the first command before its worker ran; now `~0`.

## Phase 2 — Persistent kernel

- [ ] **P2.1** Write `persistent_gate_kernel` in `rocm/ds4_rocm_tp.cuh`
  per [`DES-§4`]: dispatcher + TX/RX worker warps, inlined copies of the
  three existing kernel bodies (`slot_copy`, `stamp_release`,
  `spin_copy`), per-command team barrier before stamp, ACQUIRE/RELEASE
  SYSTEM discipline, `max_spins` timeout that latches the mapped state
  word, shutdown-word exit. Acceptance: compiles; loopback unit test
  exercises producer+consumer against a host-written ring. (R2, R3, R6)
  **Implemented 2026-09-03**: `dsv4_tp_persistent_gate_kernel` — warp 0
  dispatcher, 4+4 worker warps, dispatch-time ring-entry snapshots into
  shared memory (host may reuse ring slots), in-order `consumed`
  accounting, latch-aware spins, drain-then-exit on shutdown or latch.
  Compile + loopback pending on-node.
- [ ] **P2.2** Loopback test `tests/test_tp_persist_gate_rocm.c` +
  Makefile rule: host fills a slab vector, publishes a TX command against
  a simulated RX slot, kernel copies+stamps, host validates payload
  equality and the exact stamp value; also the timeout path (missing
  stamp → state word latched). Acceptance: test passes; timeout path
  returns without hanging. (R2, R6, R11)
  **Implemented 2026-09-03** as `tests/test_tp_persist_gate_rocm.cpp` +
  `test-tp-persist-gate-rocm` target (hipcc-compiled, self-contained):
  round trip + payload/stamp/consumed checks, timeout latch, wait-kernel
  unlatch, ordered shutdown, plus a 600-gate burst through the ring
  (wraps the 512-entry ring and reuses flag slots; host-coherent slots
  like the NHI pool) that prints the host publish→rx_done latency
  distribution and fails above 500 µs p50. Bounded shutdown wait + kernel
  debug-word dump on failure. **Green 30/30 on max2 (gfx1151), ~41 µs
  p50.**
- [ ] **P2.3** The two 1-thread compute-stream kernels
  (`mark_src_ready`, `wait_rx_done`) per [`DES-§2`]; single-thread
  launches, mapped-flag store/spin with SYSTEM scope. Acceptance: covered
  by the loopback test and the A/B below. (DES-§2)
  **Implemented 2026-09-03**: `dsv4_tp_mark_src_ready_kernel` and
  `dsv4_tp_wait_rx_done_kernel` in the same cuh.

## Phase 3 — Engine integration

- [ ] **P3.1** Ring-writer path in `ds4_rocm_tp_enqueue_gate_tx/rx`
  (flag on): `acquire_tx` first (host), then ring write + `publish`;
  release ordering on entry stores; keep legacy branch flag-off. Encode
  failure paths return the same values as today. Acceptance: same code
  path still bit-serves flag-off requests; flag-on enqueue works in the
  loopback harness. (R4, R5, R8)
  **Implemented 2026-09-03**: `persist` flag on the request record;
  `ds4_gpu_tp_gate_encode` takes the ring path, `ds4_gpu_tp_big_gate_encode`
  passes 0 and stays legacy (R8); acquire-then-publish order kept; clean-
  shutdown publish drops don't latch failure.
- [ ] **P3.2** Service-thread poll mode [`DES-§5`]: pending commands wait
  `tx_released[seq]` then `submit_fn`; retire on `rx_done[seq]` then
  `consumed_fn`; condvar/shutdown/failure wakeups preserved. Acceptance:
  the retire loop's strict-FIFO sequence checks are exercised by the
  existing spin-exchange test plus a new out-of-order injection test if
  cheap (else the A/B). (R4, R6, R7)
  **Implemented 2026-09-03**: per-record `req->persist` branches in
  submit-wait and retire; event path untouched when off. Note: there is
  no spin-exchange unit test in the tree; strict-FIFO retire is covered by
  the persist loopback burst (in-order `consumed`), `test_tp_combine_rocm`
  and the pair A/B.
- [ ] **P3.3** Wire `mark_src_ready`/`wait_rx_done` launches around the
  graph-side gate fire points (ds4.c ATTN/FFN call sites) behind the env
  flag; ensure the persistent stream is non-blocking and the legacy
  launch-to-event path is untouched flag-off. Acceptance: diff of
  flag-off GPU trace shows zero new kernels/events vs baseline. (R1, R10)
  **Implemented 2026-09-03**: both one-thread kernels launch at the same
  enqueue points the fill/stamp/spin-copy used (stream-0 position
  preserved, no ds4.c changes needed); flag off runs the legacy kernels
  only.

## Phase 4 — Correctness validation (R11)

- [ ] **P4.1** Unit suites: `test_tp_combine_rocm` and the persist
  loopback (`make test-tp-persist-gate-rocm`); build both `strix-halo`
  binaries and restore `cap_sys_rawio` (preflight does it).
  Acceptance: all green on both nodes. Persist loopback: done (30/30).
- [ ] **P4.2** Pair smoke, flag ON: rank 0/1 bind; 197-token and
  ~1609-token prefills + decode via `scripts/tp_ab.sh`; NHI stats
  `failures=0 event_drops=0 crc=0 overrun=0`; teardown clean.
  Acceptance: two consecutive clean runs.
- [ ] **P4.3** Bit-exact A/B: identical session + prompt (temp 0),
  flag off vs on, compare full completion tokens and selected entries of
  the logits; assert equality. Acceptance: token-exact on the fixed
  benchmark prompts. (R2)
- [ ] **P4.4** Failure drills: spin-timeout abort (point RX at a
  nonexistent stamp — env/test hook), peer disconnect mid-decode, SIGINT
  during decode. Acceptance: each terminates with the existing failure
  logs and no hang. (R6, R7)

## Phase 5 — Performance A/B and write-up

- [ ] **P5.1** Per-gate timing under both paths (`DS4_TP_GATE_STATS=1`,
  on by default in `scripts/tp_ab.sh` via `TP_EXTRA_ENV`), decode tok/s
  (flag off vs on, pinned clocks), prefill regression check. Acceptance:
  numbers recorded; goal band from R12 reported honestly.
- [ ] **P5.2** Optional stretch if P5.1 ≥ ~2× per-gate improvement:
  fold the post-gate combine into the persistent kernel (eliminating
  `wait_rx_done`) per `DES-§2` phase 2 note; re-run P4.2/P4.3/P5.1.
  Acceptance: bit-exactness maintained or the fold is reverted. (R2)
- [ ] **P5.3** Update `docs/tp_persist_gate/BASELINE.md` with the
  after-numbers, and add a short PR comment summarizing POC results (no
  default change; feature stays `DS4_TP_PERSIST_GATE`-gated).
  Acceptance: PR comment posted; BASELINE.md committed with the repo.

## Out of scope (recorded, not tracked here)

- Transport tuning / module changes (explicit non-goal).
- Layer-compute megakernel (separate effort).
- Enabling the flag in production defaults regardless of results.