# Persistent TP Gate Kernel — Requirements

Opt-in persistent-gate execution for the ROCm/NHI tensor-parallel decode
path. Today each of the 86 decode gates per token costs ~0.44 ms of
launch/event/ioctl choreography (3 kernel launches + 2 event records +
2 service-thread event synchronizations + 2 mutex-guarded ioctl paths).
A resident gate kernel that performs fill+stamp+spin+copy in one launch
targets that overhead, while proving the transport contract is honored
bit-for-bit.

References: gate kernels `rocm/ds4_rocm_tp.cuh`; engine + service thread
`ds4_rocm.cu` (the `ds4_rocm_tp_request`/`enqueue_gate_tx/rx` region and
`ds4_rocm_tp_service_thread`); NHI core `ds4_tp_nhi.c`; lockstep protocol
`ds4_tp.h`/`ds4_tp.c`; decode gate fire points `ds4.c`
(`ds4_gpu_tp_gate_encode`, ATTN at 24932, FFN at 26438).

## Functional requirements

- **R1 — Opt-in and reversible.** The persistent path activates only under
  the `DS4_TP_PERSIST_GATE=1` environment variable. With the variable unset
  (production default) the existing per-launch path is used unchanged: same
  kernels, same events, same thread behavior.
- **R2 — Bit-exact outputs.** For identical inputs and identical gate
  sequence, the persistent path must produce outputs identical to the
  launch-per-gate path:
  - the payload bytes copied into the TX slot are identical;
  - the in-band stamp value (`ds4_rocm_tp_stamp(rank, seq)`) is identical;
  - the RX slot poll uses the same `__hip_atomic_load(ACQUIRE, SYSTEM)`
    semantics and the same expected stamp;
  - payload stores complete before the stamp store (a workgroup-wide
    barrier between copy and stamp release, per contract rule 7);
  - the host-side SUBMIT/repost sequencing is unchanged in order and rule
    discipline (rules 9–13 in the transport contract).
- **R3 — Transport contract rules.** The persistent path must preserve
  every rule the current path relies on (see `docs/` note in
  `rocm/ds4_rocm_tp.cuh` and strix-rdma `TP_TRANSPORT_CONTRACT.md`):
  rules 1–3 (pools imported before ENABLE, dedicated uncached BOs),
  rule 5 (stamp in the slot's last word), rules 6–7 (poll/store pairs are
  ACQUIRE/RELEASE SYSTEM; payload-before-stamp), rule 8 (release is the
  NHI-visibility release), rule 9 (slots advance strictly in order),
  rule 11 (events reaped nonblocking; POST_RX after both the RX event and
  the GPU final-reader mark), rule 13 (the out-of-band barrier precedes
  the first submit), rules 4/16 (stats-first teardown order).
- **R4 — Host ioctl boundaries unchanged.** `ds4_tp_nhi` remains the sole
  owner of SUBMIT_TX / REAP / POST_RX:
  - `acquire_tx(seq)` still runs before the GPU overwrites the
    `seq % msgs` slot (host-side, before the ring command is published);
  - `submit(seq)` only after the GPU's stamp release is visible
    (a mapped `tx_released[seq]` flag replaces the `tx_ready` event);
  - `consumed(seq)` only after the GPU's final RX read is complete
    (a mapped `rx_done[seq]` flag replaces the `rx_consumed` event).
  Strict FIFO sequences are preserved on both flags (the ioctl cores
  reject out-of-order seqs).
- **R5 — Pipelining/window invariants.** The ring keeps the same
  bounded-outstanding discipline as today: never more unconsumed messages
  than the transport ring holds (`seq % msgs` safety), and the producer
  must not stall the graph encode behind RX completions it does not need
  yet. The existing pending/inflight bookkeeping invariants must hold in
  ring form.
- **R6 — Failure latching.** A spin timeout inside the persistent kernel
  must latch the mapped engine-state word to `failed` exactly like the
  current `dsv4_tp_spin_copy_f32_kernel` timeout path, wake the service
  thread, and make every subsequent ring write and ioctl path fail.
  A transport failure, peer close, or init failure must disable the
  persistent path loudly (log line) and fall back / abort under the same
  semantics as today — no silent hangs, no corrupted gates.
- **R7 — Shutdown discipline.** `ds4_gpu_tp_shutdown` keeps the current
  teardown order: stop new gates, drain in-flight commands, exit the
  persistent kernel (ring shutdown flag), quiesce NHI (`ds4_tp_nhi_quiesce`),
  stats-first close, then pool and mapped-memory teardown. Unit stops
  (SIGINT/SIGTERM) must not wedge the kernel or the service thread.
- **R8 — Coexistence with big gates.** Prefill big-gate exchanges
  (`ds4_gpu_tp_big_gate_encode`) and any verify/batch gates continue to run
  through the existing launch path during the POC. (In the NHI prefill
  path the chunked-batch big gates are split across the imported ring;
  decode row gates only are the POC target — ds4_session_sync_internal's
  `DS4_TP_NHI_TOKEN_PREFILL` branch and `ds4_gpu_tp_big_gate_encode`
  remain untouched.) The persistent kernel
  must be safe while those launches run (it idles on an empty ring), and
  both paths must never interleave messages for the same seq.
- **R9 — Schedule agnosticism.** The ring path is driven by
  (layer, gate, seq, n_floats), so DS4's identity schedule (86 gates/token)
  and GLM mask schedules work without model-specific logic. The schedule
  source is `ds4_engine_tp_gate_schedule()` (ds4.c): DS4 decode = layer 0,
  step 1, 2 gates x 43 layers = 86 per token; GLM resident = 2 per sparse
  layer; GLM streaming = FFN gates only; GLM-5.3 = masked ATTN+FFN. The
  ROCm/NHI bind is DS4-decode-only (`ds4_engine_tp_bind` rejects GLM on
  the NHI transport). Payload size
  up to `DS4_ROCM_TP_PAYLOAD_FLOATS_MAX` per message; for the production
  DS4 Flash shape the payload is exactly 4096 floats (16 KB) — n_embd from
  the GGUF metadata — and the NHI slot holds 64 frames x 4096 B = 256 KB.

## Non-functional requirements

- **R10 — No regression, no production default change.** `DS4_TP_PERSIST_GATE`
  unset keeps today's binaries behaviorally identical (kernel names, event
  counts, thread wakeups). With the flag set, the envelope must still
  respect the existing TP decode scheduling discipline: **one command
  buffer per token** (the encoder deliberately skips its mid-token flush
  for `tp_world == 2` because "a TP gate uses one monotonic shared event
  for the whole token"; the ring path must not reintroduce splits).
  The gate is a provable no-op when disabled.
- **R12 — Coexistence with session batching.** Under `e->tp.active` every
  native L1 batch fast path (Metal shared, CUDA pipeline, GLM-5.3 native)
  is already excluded by the ds4.c admission checks; NHI decode batching is
  per-session token encodes in one shared command epoch with mirrored EVAL
  frames. The persistent kernel must therefore handle N sessions x 86 gates
  interleaved in one epoch in submission order (its fill-n-spin loops do).
  The ROCm TP output head stays replicated (`e->tp.vocab_split == false`,
  metal-only vocab split), so no half-logits control frames are involved in
  the gate path.
- **R11 — Validation standard.** Evidence gates:
  1. Token-exactness: the same decode inputs under both paths produce
     identical outputs (temperature 0 comparison across the gate path),
     decisively the bit-exactness gate.
  2. Two-node NHI TP smoke: rank 0/1 bind, 197-token and ~1600-token
     prefills + decode, NHI stats `failures=0 event_drops=0 crc=0
     overrun=0`.
  3. Per-gate cost measurement: the baseline ~0.44 ms/gate must be
     re-measured before the change and re-measured after; decode
     tok/s reported for both paths on the same pair with pinned clocks.
  4. No hang: teardown, disconnect mid-request, and the spin-timeout
     abort path each terminate.
- **R12 — Performance goal (not a guarantee).** Reduce the per-gate
  choreography from ~0.44 ms toward ≤ 0.1 ms and lift TP decode from the
  measured ~13.9 tok/s toward the 18–22 tok/s band; if the measurement
  doesn't move, the feature stays experimental and default-off and the
  write-up says why.

## Explicit non-goals

- No transport tuning, no wire/protocol change, no kernel-module change
  (the NHI module is not in the gate-critical path for this work).
- No changes to the layer compute kernels (attention/MoE/etc.) — this is
  gate choreography only; folding the post-gate combine is a later,
  separately gated phase.
- No singleton MG model coverage claims (GLM schedules must not regress;
  GLM-specific tuning is not required).
- Not merging the feature into defaults regardless of results: the
  persistent path ships behind the env flag and the PR note reports
  measured numbers.