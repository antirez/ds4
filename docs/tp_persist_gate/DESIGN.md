# Persistent TP Gate Kernel — Design

One resident kernel per rank replaces the per-gate
launch/event/service-thread round trips on the NHI TP decode path, while
the host ioctl sequence (acquire → submit → consumed) and the transport
memory-model contract stay exactly as today. Correctness is "the same
bytes, the same stamps, in the same order" — the change is scheduling, not
semantics.

## 1. Current gate anatomy (the cost being removed)

For one gate the host (graph encode on the legacy stream) does:

```
acquire_tx(seq)                  host mutex + possible reap loop
dsv4_tp_slot_copy_f32_kernel     fill TX slot from slab out vector (grid ~8 CTAs)
dsv4_tp_stamp_release_kernel     1-thread stamp release
hipEventRecord(tx_ready)
dsv4_tp_spin_copy_f32_kernel     1x256 spin + copy into slab in vector
hipEventRecord(rx_consumed)
```

and the service thread, per gate:

```
hipEventSynchronize(tx_ready)   -> submit_fn (SUBMIT_TX ioctl)
hipEventQuery/Sync(rx_consumed) -> consumed_fn (reap + POST_RX)
```

Measured ~0.44 ms/gate × 86 gates/token ≈ 38 ms of the ~72 ms TP decode
token. The wire RTT itself is a few µs; the dominant terms are the tiny
launch latencies, the event synchronizations, and the thread wakeups.

## 2. Persistent execution model

```
                       compute stream (legacy)
   [producer kernels] -> mark_src_ready(pub) -> ... -> wait_rx_done(pub) -> [combine/consumer kernels]
                               |                                      ^
                        (mapped flag store)                   (mapped flag spin)
                               v                                      |
                       persistent stream (non-blocking, one resident block)
   +----------------------------------------------------------------------------+
   | persistent_gate_kernel: poll ring head                                     |
   |   TX op:  spin src_ready -> fill slot -> barrier -> lane0 stamp            |
   |           -> store tx_released[seq]                                        |
   |   RX op:  lane0 spin stamp(peer) -> barrier -> copy slot -> dst            |
   |           -> store rx_done[seq]                                            |
   +----------------------------------------------------------------------------+
                               |                                      ^
                    (mapped flags polled by host)                    |
   service thread:  poll tx_released -> submit_fn       poll rx_done -> consumed_fn
```

- The **persistent kernel** runs on its own non-blocking side stream, so
  it executes concurrently with layer compute instead of serializing
  through the legacy stream.
- The **only** per-gate GPU launches left on the compute stream are two
  1-thread kernels: `mark_src_ready` (published after the producer
  kernel, before the ring write) and `wait_rx_done` (spins on the mapped
  `rx_done` flag before the combine/consumer). All stream ordering is
  preserved through the mapped-flags + these two launches.
- The **host service thread** polls two mapped flag arrays instead of
  synchronizing two HIP events: `tx_released[seq]` (submit trigger) and
  `rx_done[seq]` (consumed/repost trigger). The ioctl cores
  (`ds4_tp_nhi_submit`, `ds4_tp_nhi_consumed`) retain their own strict
  FIFO re-checks.
- `acquire_tx(seq)` stays **host-side before the ring command is
  published** (rule 9: reserve before the GPU overwrites the rotating
  slot).
- Bit-exactness (R2/R3): the persistent kernel inlines the exact bodies
  of `dsv4_tp_slot_copy_f32_kernel`, `dsv4_tp_stamp_release_kernel`, and
  `dsv4_tp_spin_copy_f32_kernel` with per-command warp teams; the copy is
  a value-for-value write so its thread shape does not affect the bytes;
  the stamp is written by one lane after a workgroup-wide barrier; the
  spin uses the same ACQUIRE/SYSTEM stamp load and expects the same
  `ds4_rocm_tp_stamp(peer_rank, seq)` value. The combine kernel(s) after
  the gate are untouched, so the reduction order is untouched.

## 3. Ring and control-plane layout

All control words live in one pinned, mapped host allocation (same
mechanism as today's `g_tp_engine_state_dev`):

```
word 0        publish       host -> kernel   seq of the next command to execute
word 1        consumed      kernel -> host   seq of the last executed command
word 2        shutdown      host -> kernel   1 = exit kernel
word 3        failure/state host<->kernel    reused g_tp_engine_state semantics
words 4..      per-seq flags, capacity C: src_ready[C], tx_released[C],
               rx_done[C]                    flags are seq % C
ring           C entries { u32 kind [TX|RX], u32 reserved, u64 seq,
               u64 src_off, u64 dst_off, u32 n_floats, u32 expect_stamp }
```

- Capacity `C = 512` (matches `DS4_ROCM_TP_QUEUE`), far above the
  transport-ring outstanding bound (`msgs`, 64 at 4096 frames / 64
  frames-per-msg), so the existing windowing discipline (R5) is a strict
  subset of the ring's space.

  Production shapes (from ds4.c / gguf metadata, for sizing): n_embd is
  4096 floats, so one gate payload is 16 KB; NHI slot = 64 frames x
  4096 B = 256 KB; ring = 4096 frames = 64 messages; engine queue depth
  512; decode gate schedule = 2 gates x 43 layers = 86 gates per token.
- Host writes a command with release ordering then bumps `publish`
  (release store). Kernel advances `consumed` in seq order (release).
- Flags are writes-with-release by the producer side and polls
  with-acquire by the consumer side. Flag reuse across the ring wrap is
  safe because all three per-seq traffics are strictly seq-ordered and
  the outstanding window is bounded below C (documented invariant).
- Ring entries never alias slab or NHI pool memory; the slab offsets
  (src/dst) are resolved by the kernel against the slab base VA passed
  at init (like `ds4_gpu_tp_nhi_init` does today).
- The ring writer path replaces `ds4_rocm_tp_enqueue_gate_tx/rx` when the
  env flag is set; the legacy path remains fully intact for big gates
  (R8) and for the flag-off default (R1).

## 4. Kernel structure

One block of 256 threads (40 registers or fewer per thread target),
launched once per engine init on the persistent stream:

```
warps 0      dispatcher: polls publish; reads ring entry; assigns the
             command to the next free worker warp-pair; polls consumed
             completion to recycle assignments
warps 1..4   TX workers (4 concurrent): spin src_ready[seq], then copy
             src -> slot (stride = team size), __syncthreads-like team
             barrier, lane0 stamp store, tx_released[seq] store
warps 5..8   RX workers (4 concurrent): lane0 spins expect_stamp
             (ACQUIRE/SYSTEM), team barrier, copy slot -> dst,
             rx_done[seq] store by one lane
```

- 4+4 concurrency is ample: decode has at most a couple of gates in
  flight per rank once the windowing strip is applied; big gates do not
  enter the ring (R8).
- Timeout: the RX spin counts iterations against the engine's
  `max_spins`; on expiry the dispatcher stores `failed` to the mapped
  state word, exactly the current kernel's failure path (R6).
- Shutdown: dispatcher sees the `shutdown` word, drains queued commands,
  exits; the host synchronizes the persistent stream and then runs the
  existing teardown (R7).
- The `mark_src_ready` / `wait_rx_done` compute-stream kernels are
  1-thread, ~2–5 µs launches; phase 2 below folds additional work into
  the persistent kernel if measurements justify it.

## 5. Host-side integration

`ds4_rocm.cu`, engine region (same file/globals as today):

- `ds4_gpu_tp_nhi_init`: when `DS4_TP_PERSIST_GATE=1`, additionally
  allocate the mapped control plane, create the persistent stream
  (non-blocking), and launch the persistent kernel. Any failure logs and
  falls back to the legacy engine (R6 fail-closed).
- `ds4_rocm_tp_enqueue_gate_tx`: flag-on path = acquire `tx` ioctl
  (host), then write the TX ring command + `publish`; flag-off = existing
  launches unchanged.
- `ds4_rocm_tp_enqueue_gate_rx`: flag-on = write the RX ring command;
  no kernel launch, no event record.
- `ds4_rocm_tp_service_thread`: gains a poll mode — wait on
  `tx_released[seq]` (mapped, acquire) for pending commands
  (submit-ahead, same as today's never-wait-for-older-RX rule), retire on
  `rx_done[seq]` → `consumed_fn`. The condvar/wakeup discipline is
  preserved for shutdown and failure (R6/R7).
- `ds4_gpu_tp_gate_encode` and the `ds4.c` gate fire points: **no
  signature changes** — the two tiny compute-stream kernels are inserted
  at the two ends of the gate window between the graph call sites.
- `ds4_gpu_tp_shutdown`: shutdown word → stream sync → join thread →
  NHI quiesce/close → free mapped planes (R7).

Verified against the current ds4.c host side (full-file read):

- `ds4_engine_tp_bind()` already owns the NHI bind: zero-fill the
  `ds4_tp_slab_bytes` slab, `ds4_tp_nhi_open`, `ds4_gpu_tp_nhi_init` with
  slot stride equal to the full `ds4_tp_slab_*_offset` layout, then
  `ds4_gpu_tp_nhi_set_ring_msgs` + `ds4_tp_nhi_ready_barrier`. The
  passed callbacks (`tx_slot`, `rx_slot`, `acquire_tx`, `submit`,
  `consumed`, `failed`) are already the exact indirection points the
  persistent path swaps — nothing changes in ds4.c itself (R10).
- TP sessions allocate spec-verifier scratch and mirror leader→worker
  EVAL/SYNC frames with `++e->tp.eval_seq`; the persistent kernel only
  sees gate traffic, never control frames.
- Engine open maps only the rank's expert half via
  `weights_model_map_sharded_spans`; ROCm deliberately skips re-warming
  host pages for the shard (UMA repopulation cost).

## 6. Sequencing, failures, teardown

One gate, end to end:

```
encode: [producer] -> mark_src_ready          (compute stream)
host:   acquire_tx(seq)                       (order gate, ioctl)
host:   publish TX{seq,...}                   (ring)
kernel: poll publish -> spin src_ready -> fill -> stamp -> tx_released
host:   poll tx_released -> submit(seq)       (SUBMIT_TX ioctl)
peer:   (same play in mirror, one seq earlier or later by rank slot)
kernel: RX op: spin stamp -> copy -> rx_done
host:   poll rx_done -> consumed(seq)         (reap, POST_RX)
encode: wait_rx_done -> [combine/consumer]    (compute stream)
```

- Failure anywhere latches the mapped state word; the spin kernels, the
  ring writer, and the service thread all observe it and abort (same
  `ds4_rocm_tp_fail` latching).
- Prefill/big gates never enter the ring during the POC; the persistent
  kernel idles on the empty ring while those legacy launches run (R8).
- Teardown keeps the ordered close: stop-producing, drain, kernel exit,
  quiesce, stats, device, pools (R7).

## 7. Validation and measurement plan (mirrors R11)

- Loopback test (`tests/test_tp_spin_exchange_rocm` or its successor):
  producer command + mirrored RX command through the ring on one GPU —
  expects bit-identical payload and stamp from the writer side.
- Bit-exact A/B on the pair: identical session/prompt under flag off/on,
  temperature 0, compare completion tokens and selected logits.
- Per-gate timing: extend the existing gate trace instrumentation to
  timestamp host-side ring events; compare gate µs distributions.
- TP decode throughput on the pair with pinned clocks (baseline 13.94
  tok/s), and the 197/1609-token prefill matrix + NHI stats cleanliness
  (R11.2).
- Disconnect + SIGINT + spin-timeout abort smoke (R11.4).

## 8. Risks and mitigations

- **Persistent block occupancy:** 256 threads resident on one CU may cost
  decode occupancy. Mitigation: launched once; decode kernels already run
  far below full occupancy; measured A/B catches regressions.
- **Mapped-flag polling latency:** host polls add CPU spin per gate.
  Mitigation: poll with a tiny backoff similar to the existing reap
  loops; the numbers decide.
- **Ring/flag memory ordering bugs:** strict release/acquire discipline,
  the documented seq-ordered reuse invariant, and the existing transport
  contract rule set (R3). The loopback test covers the writer path
  without the NHI wire; the pair A/B covers the rest.
- **Cross-stream ordinaire bugs (compute vs persistent):** only the two
  1-thread kernels touch the mapped flags at stream boundaries; all
  payload memory is ordered through those flags, and bit-exactness
  against the flag-off run gives end-to-end coverage.
- **Interference with speculative verify spans (TP):** the verify path
  sends `EVAL_BATCH` frames and may issue batch gates; these stay on the
  legacy path during the POC; the ring is decode-gate-only (R8).
- **Bit-exactness slip (same bytes, different batching):** flag-off and
  flag-on runs both exist in one binary, so the POC's own A/B is the
  correctness gate and can be repeated by CI or reviewers.