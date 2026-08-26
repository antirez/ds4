# DSpark distributed speculative decode — self-review

Branch: `perf-dspark-exact-krow`. The reviewed endurance candidate is
`ae762ae`; post-review transport/state hardening is code commit `5eccd5a`.
The single-PR base is `upstream/main` commit `c1d4597`; the distributed-DSpark
implementation starts at `93f75b9`, with `c200554` plus the follow-up commits
below.
The PR series also contains its required server metrics, descriptor-framed NHI
transport, tensor-parallel ROCm/NHI support, and exact-row kernel prerequisites.

## Scope summary

`--mtp FILE --dspark --mtp-draft N` now loads and fires on the distributed
coordinator (`--role coordinator`), with the worker cooperating through the
v3 protocol:

- The ds4.c "silent-ignore" gate (`role == DS4_DISTRIBUTED_NONE`) is removed
  from the `--mtp` load path, `ds4_engine_has_mtp`, and
  `ds4_engine_mtp_draft_tokens`, so the support model is initialized on a
  dist role and its draft block size drives the coordinator cycle.
- The coordinator runs `ds4_dist_session_mtp_spec_cycle`: a base decode pulls
  the worker's drafts (`F_OUTPUT_DRAFTS`), a multi-row verify span
  (`F_SPEC_VERIFY | F_OUTPUT_ALL_LOGITS`, optional `F_OUTPUT_DRAFTS` for
  continuation drafts) runs the target model over the pipeline, and the
  coordinator accepts only drafts whose continuation equals the target's
  greedy argmax, re-verified one token at a time. Partial accepts rewind the
  verify-span frontier (`F_SPEC_ROLLBACK`) and re-decode the accepted prefix
  through the single-token distributed route (`dist_mtp_spec_replay_accept`),
  so the accept/emit decision uses single-token logits and never the batched
  verify rows.
- The worker drafts through `ds4_session_dist_support_draft` -> legacy MTP or
  DSpark propose (`ds4_session_dist_dspark_draft`), and returns the new
  `DS4_DIST_RESULT_LOGITS_DRAFTS` / `..._NROWS` / `..._DECODE2` result kinds.

## The two interrupted-run fixes (verified, already present in c200554)

1. Worker result path only prepares the bulk TX plan for bulk kinds.
   - `dist_worker_process_work_message` computes `result_is_bulk`
     (`HIDDEN_STATE` or `LOGITS` only) and guards
     `dist_tx_bulk_plan_prepare(...)` with it. Spec result kinds (drafts,
     per-row logits, decode2) are control payloads and never reserve a bulk
     plan; `dist_tx_bulk_plan_desc(&result_tx_plan)` returns NULL for them, so
     `dist_send_work_result_prepared` sends them inline. Re-verified against
     the current tree; no further edit needed.
2. The shrunk `payload_bytes` is sent when drafts trail the logits.
   - `result_bytes` is the max (logits + 4 + 4*drafts_max) allocation, and
     `payload_bytes` is shrunk to `logits_bytes + 4 + 4*nd` before
     `dist_worker_upstream_send_work_result_prepared(...)`; the same shrunk
     value feeds `result_wire_bytes` and the drawn draft block. This avoids
     shipping uninitialized tail bytes for a count=0 or short draft cycle.
     Re-verified; no further edit needed.

## Dist code paths touched

Coordinator (ds4_distributed.c):
- `ds4_dist_session_mtp_spec_cycle` (new): fused-span / classic-cycle
  speculative decode with exact argmax acceptance and failure rebuild.
- `dist_coordinator_spec_commit_prefix` (new): direct per-prefix commit on
  both machines, falling back to rollback re-eval.
- `dist_route_plan_supports_spec` (new): all-hops v3 cap gate.
- `dist_coordinator_send_remote_work_on_fd`: added `spec_flags` and optional
  `tx_plan` passthrough.
- `dist_coordinator_eval_remote_on_fd`: handles the three new result kinds.
- `dist_coordinator_eval_span`: added `spec_flags`/`decode2_span`/draft
  outputs and the local slice + remote span split.
- `ds4_dist_session_print_spec_stats` (new): prints aggregate
  proposed/accepted counters; called from `ds4_session_free`.
- `struct ds4_dist_session`: added `spec_cycles/spec_proposed/spec_accepted`.

Worker (ds4_distributed.c):
- `dist_worker_process_work_message`: spec flag validation, drafting,
  result-kind construction, bulk-plan guard, shrunk payload send.
- `struct ds4_dist_worker_state`: added `spec_warned` one-shot warning.

ds4.c:
- `ds4_engine_open_internal`: `--mtp` load gate widened to dist roles; degenerate
  DSpark block warning.
- `ds4_engine_has_mtp` / `ds4_engine_mtp_draft_tokens`: dist-role DSpark
  block-size reporting.
- `ds4_session_eval_layer_slice` -> extracted `ds4_session_eval_layer_slice_span`
  (guarded by `DS4_NO_GPU`), plus new `ds4_session_eval_layer_slice_logits_all`
  and `ds4_session_eval_layer_slice_decode2`.
- `ds4_session_dist_mtp_draft` / `ds4_session_dist_dspark_draft` /
  `ds4_session_dist_support_draft` / `ds4_session_dist_spec_commit_prefix` /
  `ds4_session_dist_frontier_snapshot|restore` /
  `ds4_session_dist_timeline_truncate`.
- `DS4_SPEC_PREFIX_SLOTS` 4 -> 5 and `spec_prefix_rows` capture.
- `ds4_session_eval_speculative_argmax`: dist branch delegates to the spec cycle.

ds4.h: new slice/spec prototypes and `ds4_dist_mtp_frontier`.

ds4_dist_v3.h/.c: `DS4_DIST_V3_CAP_SPEC_DECODE_V1`; negotiation selects the bit.

ds4_server.c: widened the MTP spec gate for non-batched temperature-0 decode.

rocm/*.cuh: verify-batch Q8 direct kernel, MoE sorted-pairs threshold, and
runtime config env rollbacks (compile-checked by inspection only; not buildable
on macOS).

tests/test_dist_v3.c: negotiation coverage for the new capability bit.

## Wire/struct changes and protocol-version treatment

- New capability bit `DS4_DIST_V3_CAP_SPEC_DECODE_V1 = 0x00000008u`.
  Additive: `dist_v3_local_offer` advertises it; `ds4_dist_v3_negotiate`
  selects it only when both peers advertise it; `hello_ack_validate` requires
  selected bits to be a supported subset of the common offer. Old v3 peers
  simply never see the bit selected. `test_dist_v3` covers both directions.
- New capability bit `DS4_DIST_V3_CAP_SPEC_EXACT_V1 = 0x00000010u` gates the
  explicit bilateral exact-row requirement separately from base speculation.
  Mixed pre-flag peers therefore retain replay and never receive the new flag.
- New WORK flag bits 0x10/0x20/0x40/0x80/0x100/0x200 ride the existing
  32-bit `flags` field of the fixed WORK record (which already carried v2 flag
  bits). They are masked by the worker's `DS4_DIST_WORK_F_VALID_MASK`, and the
  coordinator only sets them when every hop selected the required capability,
  so an older peer never observes them. `0x200` is
  `F_SPEC_EXACT_REQUIRED`: a
  coordinator may retain exact-span state only after the worker positively
  executes its own exact-row gate; an environment mismatch fails the span
  instead of silently mixing exact and ordinary batch state.
- New result kinds 3/4/5 (`LOGITS_DRAFTS`, `LOGITS_NROWS`,
  `LOGITS_DECODE2`) ride the existing `result_kind` field. They are only
  produced in response to the gated spec WORK flags.
- `ds4_dist_mtp_frontier` is a host-only struct (not serialized); it is
  rebuilt per cycle and restored locally on both machines.

## Memory ownership / teardown

- `result` (worker) is `malloc(result_bytes)`; it is freed on every path
  (including the mapped-lease case where `result` is replaced by the lease
  host pointer, tracked by `result_mapped`). `result_tx_plan` is released on
  success and aborted on failure.
- `ds4_dist_session` counters are plain scalars (no ownership); the dist
  session object is intentionally process-lifetime, so the stats print is
  hooked onto `ds4_session_free` (owner session), which runs before
  `ds4_dist_session_free`.
- New coordinator buffers `logits0`/`rows`/`replay_logits`/`prefix_logits`
  are freed on all success/error returns; `dist_coordinator_eval_span` frees
  its `hidden` span buffer.
- The only new worker-state field (`spec_warned`) is a bool zeroed by the
  existing `memset` of the worker state.
- Engine teardown releases TP/GPU workspaces, retained model ranges, and host
  registrations before unmapping either the target or support GGUF. Inbound
  distributed RESULT headers are checked against request-specific kind and
  payload geometry before any payload allocation/read; telemetry is bounded
  by layer count. TP command frames are type/size-bounded before allocation,
  and the negotiated NHI payload ceiling is retained in transport state and
  enforced independently of ring capacity.

## Expected log lines

- (a) DSpark firing under dist (support model initialized):
  `ds4: DSpark support model detected: <path> (stages=.. block=.. markov_rank=.. tensors=.. missing=.. invalid=.. metadata_errors=..); use --dspark to enable experimental runtime decode`
  and, with `DS4_DSPARK_STATS=1` at session teardown after speculation ran:
  `ds4: dist spec stats cycles=<n> proposed=<p> accepted_draft=<a> accept_rate=<r>%`
- (b) init-unavailable warning (no silent no-op):
  `ds4: warning: --dspark cannot drive distributed speculative decode (block_size=<n>); falling back to per-token decode`
  (worker-side counterpart: `ds4: dist worker: speculative drafts requested but no enabled support model (--mtp/--dspark) is available; falling back to per-token decode`; existing hard errors remain: `ds4: --dspark requires --mtp FILE` and `ds4: unsupported --mtp support model ...`)
- (c) non-spec default (no flags, or `DS4_MTP_SPEC_DISABLE=1`): none of the
  above lines; ordinary decode path is unchanged.

## Unsure / caveats (explicit list)

1. Cannot build or execute ROCm on macOS; the `rocm/*.cuh` verify-batch
   kernel/dispatch changes were reviewed only for include order and call-site
   types. They need a ROCm host to compile/run.
2. The coordinator/worker protocol paths are validated by compile, link, and
   `tests/test_dist_v3` (92/92 passing).  The v21-window ROCm run on the
   NHI TP harness (`--mtp <DSpark GGUF> --dspark --mtp-draft 5`) is the
   first end-to-end execution evidence; see the v21-window fix section below.
3. Exactness: the original direct per-prefix snapshot commit and the batched
   rollback re-eval both inherited the batch-verifier floating-point order,
   which could flip a greedy argmax versus single-token decode.  As of the
   v21-window fix the dist spec cycle no longer commits batch-derived state:
   every accepted draft (and the fused base token) is re-decoded one token at
   a time through the ordinary route, so temperature=0 output is token-
   identical to the no-spec arm.  The remaining practical risk is the
   `rocm/*.cuh` verify-batch kernels still needing a ROCm compile check.
4. The DSpark propose pipeline requires the worker's layer slice to own the
   declared capture target layers. A coordinator-heavy split where no worker
   owns those layers would degrade to zero drafts; this is not detected at
   load time beyond the block_size<=1 warning.
5. `dist spec stats` counts "proposed" as drafts that entered a verify span;
   worker-proposed continuation drafts that are invalidated before a fused
   verify (e.g. next token changed) are intentionally not counted.
6. `--mtp-draft N` only governs the legacy MTP head; DSpark block size comes
   from model metadata (`deepseek4.dspark.block_size`), so DSpark ignores the
   numeric value of `--mtp-draft N` beyond acting as a load trigger.

## Review-fix commit (H1/H2/L3)

- H2 (exactness-of-feature): `ds4_distributed.c:10127` — worker base-decode
  drafts now call `ds4_session_dist_support_draft` with
  `draft_pos = work.pos0` (the position of the input token), matching the
  single-node `checkpoint.len - 1` convention instead of `work.pos0 + 1`.
  The fused continuation branch (`work.pos0 + work.n_tokens`) is unchanged.
- H1 (perf-guard): `ds4_distributed.c:6783` —
  `dist_route_plan_supports_spec` now requires `plan->count == 1u` plus the
  single entry's `DS4_DIST_ROUTE_F_OUTPUT_LOGITS` capability in addition to
  `DS4_DIST_V3_CAP_SPEC_DECODE_V1`, mirroring the worker-side
  "final logits span" validation.  The one-shot coordinator warning at
  `ds4_distributed.c:6830` now names the reason (multi-hop topology, missing
  output ownership, or missing capability) instead of only the capability
  case.
- L3 (hardening): `ds4_distributed.c:9833` — `result_bytes` is now computed
  in `size_t` arithmetic and rejected (worker error) when it exceeds
  `UINT32_MAX`, preventing the 32-bit `n_tokens*vocab*4 + drafts` products
  from wrapping.
## review-fix: exact acceptance

Deep review tightened the acceptance invariant: every accepted token must equal
`dist_logits_argmax` re-computed once on the single-token output-head path, and
no batched verify-row argmax may participate in the accept/emit decision.

- `ds4_distributed.c` — `dist_mtp_spec_replay_accept` now exact-verifies each
  replayed step: after decoding token `i` on the ordinary single-token route it
  compares `dist_logits_argmax(last_logits)` to `tokens[i+1]` and returns at
  the first divergence, emitting only the verified prefix.  The batched
  verify-row `accept_n` the caller passes in is demoted to a replay-length
  bound; `spec_accepted`, the emitted `accepted[]` prefix, and the fused/pending
  arming are all derived from the replay's returned exact count.
- Legacy MTP `draft_n==2` keeps its existing exact decode2 verifier; the
  replay re-verification is redundant-but-harmless there.
- No protocol/struct change; all recovery and WHY logging, memory freeing
  (including `replay_logits`), stats semantics (cycles/proposed/accepted_draft
  now post-exact), and the no-`--mtp`/`--dspark` default path are preserved.

## v21-window fix (dist DSpark hard-error + exactness)

Root cause (from the sealed NHI TP log pair + validation report):

- The spec cycle worked for 80 cycles (`cycles=80 proposed=16 accepted_draft=14`),
  then emitted `finish=error error="distributed speculative decode failed"` at
  gen=92.  The worker-side stats (`scheduler_skips=59`, `proposed=0` local spec
  cycle) show the worker only *drafts*; acceptance and commit live on the
  coordinator.
- The failure surface is the partial-accept commit cascade: the original code
  committed accepted batches by a direct per-prefix snapshot commit
  (`dist_coordinator_spec_commit_prefix` / `SPEC_COMMIT`) whose per-machine
  state (compressor/indexer row counters vs. the worker's kept session) can
  drift, then fell back to a *batched* `F_SPEC_ROLLBACK` re-eval, then to a
  full transcript rebuild.  When all three disagreed in one cycle the cycle
  returned the hard `fail_msg`, dropping the request; the coordinator's
  session-teardown stats (`cycles=80`) still showed the committed work.
- The same batch-verifier floating-point order also broke exactness: the
  captured validation report records `reasoning_content` differing from the
  no-spec arm, so the direct commit path produced non-greedy tokens.

Fix (minimal robust):

- Replaced the direct snapshot commit and batched rollback with
  `dist_mtp_spec_replay_accept`: after a verify span, the accepted prefix (and
  the fused base token) is re-decoded one token at a time through the ordinary
  distributed route.  Each span is the exact same single-token decode the
  no-spec arm uses, so temperature=0 output is token-identical, and the first
  replay token carries `F_SPEC_ROLLBACK` to rewind the worker's verify-span
  frontier while subsequent tokens are ordinary spans.
- Added `dist_mtp_spec_recover_after_failure`: on any failed speculative span,
  rewind to the committed prefix (`start`), rebuild the prefix, then re-eval
  the pending base token through the plain per-token decode.  Only if that
  also fails does the cycle return an error; every stage logs a one-line WHY
  (`ds4: dist spec span failed rc=.. why=..`, `.. prefix rebuild failed why=..`,
  `.. plain per-token fallback failed why=..`) so the next failing window pins
  the residual path.  This replaces the old `DIST_MTP_SPEC_REBUILD` macro.
- Exact replay now sets `owner` logits from the replay buffer
  (`ds4_session_set_logits`) and arms the fused continuation
  (`spec_pending_token`) from those exact logits instead of the batch rows.

Behavior is unchanged without `--mtp`/`--dspark`; the existing
`DS4_DSPARK_STATS` telemetry (cycles/proposed/accepted_draft/accept_rate) and
the `DS4_MTP_SPEC_LOG`/`DS4_DSPARK_SPEC_LOG` diagnostics are preserved.
Validated on this host by compiling `ds4_distributed.o` (Metal/CPU link path)
and `tests/test_dist_v3` (92/92); the ROCm executable warm-path must be
re-run on the NHI pair to close the residual `rocm/*.cuh` compile risk.

## Exact per-row verify span (perf-dspark-replay-opt)

Cost split (from the sealed v22b measurement and this code path):

- v22b measured spec wall 13.147s vs nospec 10.742s (99 cycles, 32 proposed,
  29 accepted).  The speculative cycle still paid the cost of the DSpark
  propose pipeline (negligible, hidden behind the base decode on the worker),
  a batched verify span per proposal group (one `F_SPEC_VERIFY |
  F_OUTPUT_ALL_LOGITS` span, cheap), and then `dist_mtp_spec_replay_accept`
  re-decoding every accepted draft one token at a time through the ordinary
  distributed route.  That serial replay is the dominant overhead: 29 accepted
  drafts become 29 extra single-token round trips (plus one more for each
  partially-accepted span), which is why decode t/s drops from 14.50 to 11.16.
- The optimization folds the replay's single-token decode compute back into
  the verify span itself.  Instead of running the multi-row layer batch and
  then re-decoding, the verify span decodes each row sequentially through the
  same `ds4_session_eval_layer_slice(..., n_tokens=1, ...)` call the replay
  and no-spec arms use, with KV advanced row by row.  The coordinator produces
  one hidden row per token, ships them once (`F_INPUT_HC`), and the worker
  consumes them and emits one logits row per token.  A full accept therefore
  needs exactly one round trip instead of 1 batch + N serial round trips.

What changed:

- `ds4.c` `ds4_session_eval_layer_slice_exact_rows`: per-token loop over the
  ordinary single-token layer-slice entry point.  Each row's logits/hidden are
  produced by the exact same kernel/accumulation order as sequential decode
  (`ds4.c:60654` path), so bit-identity is by construction, not by kernel
  tuning.
- `ds4_session_dist_spec_exact_verify`: gate.  True only on ROCm builds, for
  non-GLM/non-CPU sessions, and spans of <= 8 rows
  (`DS4_DIST_SPEC_EXACT=0` disables).  Metal/CPU and larger spans return
  false and keep the batched verify + exact replay arm unchanged.
- `ds4_distributed.c`:
  - `dist_coordinator_eval_span` runs its local slice through the per-row
    exact path for `F_SPEC_VERIFY` spans when the gate is true.
  - The worker eval dispatch uses `ds4_session_eval_layer_slice_exact_rows`
    only for `F_SPEC_VERIFY | F_OUTPUT_ALL_LOGITS |
    F_SPEC_EXACT_REQUIRED` spans after its local exact gate succeeds.
  - `ds4_dist_session_mtp_spec_cycle` trusts the span rows on a full accept
    (`exact_span && accept_n == draft_n`), keeping the span's commits and
    setting `owner` logits from the last row; partial accepts (or non-exact
    backends) still restore the frontier and run `dist_mtp_spec_replay_accept`.

Exactness argument:

- Each verify row is `ds4_session_eval_layer_slice(s, &tokens[i], 1, pos0+i,
  ..., input_hc_i, output_hc_i/logits_i, ...)`, the identical call the
  `dist_mtp_spec_replay_accept` arm issues one span at a time
  (`ds4_distributed.c:6731`) and the identical call the no-spec arm uses per
  token.  The only difference is that the rows are packed into one WORK span
  and one `F_INPUT_HC`/`LOGITS_NROWS` exchange instead of N spans, so the
  returned rows are bit-identical to sequential decode.  The worker's
  `F_SPEC_VERIFY` frontier snapshot is still taken before the span; a full
  accept leaves both machines' commits in place (they are already the accepted
  prefix), while a partial accept rewinds via the same replay path as before.
  `F_SPEC_EXACT_REQUIRED` makes the bilateral decision explicit. The worker
  hard-fails the span if its local exact gate is unavailable; without the flag
  it deliberately uses the ordinary batch path even if its environment alone
  enabled exact mode. Thus a successful response positively attests the same
  exact-row decision before the coordinator may retain span state.

Paths that still replay:

- Any non-ROCm build (Metal/CPU), GLM sessions, and spans > 8 rows use the
  existing batched verify + exact replay arm.  Legacy MTP `decode2` two-row
  verifies and partial accepts also retain the replay arm.  All existing
  `dist spec stats` / WHY logs and `F_SPEC_*` protocol surface are preserved.

Validation performed on this host (macOS): `make ds4 ds4-server`,
`make test-dist-v3` (117/117), `make test-transport` (126/126), and the ten
worker/safety source-contract checks pass. ROCm warm-path bit-identity and the
authoritative A/B for the post-review fixes remain to be measured on the NHI
harness after the active frozen-candidate endurance window completes.

## Trust-all research mode

`DS4_DIST_SPEC_TRUST_ALL=1` is a research-only measurement knob for the
distributed speculative cycle, never the default and not a correctness feature.
It is inert unless all of these hold:

- the env is set to a non-empty value other than `0` (the same convention as
  `DS4_DSPARK_STATS`);
- the backend offers drafts (`ds4_engine_mtp_draft_tokens(engine) > 1`,
  true for a loaded `--mtp --dspark` support model); and
- the existing `draft_cap < 2` fallback has not already demoted the cycle to
  the ordinary per-token decode (so `dist_route_plan_supports_spec` also held).

When active, `ds4_dist_session_mtp_spec_cycle` emits every drafted token
without comparing it to the target argmax. It still runs the ordinary base
decode (committing `first_token` and pulling `F_OUTPUT_DRAFTS`), then advances
every trusted draft through one ordinary batched target span. This keeps both
target timelines/KV states aligned with the emitted stream and samples the
next boundary from the final target logits, while deliberately skipping the
acceptance scan and replay. `spec_proposed` and `spec_accepted` both advance by
the emitted draft count, so `dist spec stats` reports 100% acceptance;
`spec_cycles` counts base decodes. Missing drafts return exactly the single
base token.

- The coordinator session prints once at creation (only when the env is set
  and the backend offers drafts):
  `ds4: WARNING: DS4_DIST_SPEC_TRUST_ALL=1 — target-vs-draft acceptance DISABLED (research mode); batched target-state advancement remains and emitted completion is not guaranteed to match greedy decode`
- Temperature is not special-cased here: the knob only lives in
  `ds4_dist_session_mtp_spec_cycle`, which the non-greedy distributed arm never
  reaches, so sampling semantics are unchanged.
- Live logits are set from the last trusted row after its batched target-state
  advancement. The emitted completion is explicitly not required to match
  greedy decode, but it is a coherent target timeline rather than stale state.
- No additional wire kind is needed. The existing exact/non-exact arms,
  `dist spec stats` format, and WHY logs are unchanged.  `DS4_MTP_SPEC_LOG`
  adds one gated diagnostic line (`ds4: dist mtp spec trust_all ...`).

## Fast verify span experiment (disabled by default after v25f)

Cost split (profile-by-inspection of the sealed v23c logs + code)

- v23c sealed numbers: nospec 10.434s / 14.49 t/s, exact 11.728s / 12.74 t/s
  (ratio 1.124), historical trust-all 7.651s / 21.41 t/s. `dist spec stats
  cycles=100 proposed=33 accepted_draft=28 accept_rate=84.85%` — 33 drafts
  enter verify spans, 28 survive. The historical trust-all result did not
  advance target state through its emitted drafts and is therefore not a valid
  end-to-end speed ceiling. It motivated the diagnostic, but must not be used
  as performance evidence for the corrected state-advancing mode; that mode
  needs a new measurement after this fix.
- On ROCm, `ds4_gpu_end_commands()` is `cudaDeviceSynchronize()`
  (rocm/ds4_rocm_runtime.cuh:6281) and `ds4_gpu_tensor_read_any()` is a
  synchronous `hipMemcpy` (rocm/ds4_rocm_runtime.cuh:6003).  The previous
  exact_row implementation (`ds4.c` pre-change) called
  `ds4_session_eval_layer_slice(..., n_tokens=1, ...)` once per verify row,
  so every row paid a full device sync plus a readback on *both* machines:
  - coordinator (layers 0:21, 22 layers): full layer-eval + 64 KiB hidden
    readback per row (hc = DS4_N_HC*DS4_N_EMBD = 16384 f32).
  - worker (layers 22:41, 20 layers + output head): full layer-eval + head
    GEMM (4096 x 129280, ~562 MiB Q8_0) + 517 KiB logits readback per row.
- On gfx1151 (Strix Halo: 2 MB L2, no MALL) the weights re-read per row all
  come back from DRAM.  The output-head weights dominate the worker row: a
  per-row head re-streams ~562 MiB, and a 5-row verify span re-streams it
  5x.  The layer weights (attn/MoE) are likewise re-streamed once per row on
  both machines because row-major means a layer's weights are long evicted
  from L2 before the next row reaches the same layer.
- Network framing / scheduling are NOT the dominant cost: hidden (320 KiB for
  5 rows) and logits (2.6 MiB for 5 rows) transfers amortize over one
  round trip, and the fused continuation already packs base commit + drafts.
  The dominant cost is (a) per-row device sync + readback, and (b) per-row
  head/layer weight re-stream — both are what `ds4_session_eval_layer_slice_exact_rows`
  was doing.

Experimental implementation (still opt-in)

- `ds4_session_eval_layer_slice_exact_span` (ds4.c, `#ifndef DS4_NO_GPU`)
  attempts to reuse the one-token layer kernels in one command buffer per
  *span* instead of per row:
  - 2..5 rows ride `metal_graph_encode_decode_layer` (ds4.c:25634) — the
    identical layer kernel the single-token decode path uses — interleaved
    per layer (row 0..n-1 within each layer) so each layer's weights are
    streamed once and reused by every row.  The interleave is the same
    weight-hot pattern the exact two-row verifier
    `ds4_session_eval_layer_slice_decode2` (ds4.c:61544) already used.
  - The output head runs its f16 `output_hc_fn`, `output_hc_weights`,
    `hc_weighted_sum` and output-norm stages one row at a time through the
    same single-token kernels (n_tok == 1 => the ordered-chunk f16 reduce)
    and then batches only the Q8 vocab matmul via
    `metal_graph_encode_output_head_exact_span` (ds4.c:26313).  The candidate
    now sends an unpadded 2..5-row span to
    `ds4_gpu_matmul_q8_0_decode_rows_exact_tensor`; Q8 output-TP shards use the
    same exact API.  Unsupported output weights or shapes fail closed rather
    than selecting a generic multi-row projection.
  - One `ds4_gpu_end_commands()` + one `ds4_gpu_tensor_read_any()` per span
    replaces 2*N syncs + 2*N readbacks.
- `ds4_session_eval_layer_slice_exact_rows` dispatches to the span only when
  `DS4_DIST_SPEC_EXACT_SPAN=1` explicitly opts into the experiment and the
  span is single-tier, 2..5 rows, `2*n <= prefill_cap`, and (for the worker)
  `spec_logits` is present.  Spans above five rows and the default path keep
  the original per-row loop.
  The outer exact gate (`DS4_DIST_SPEC_EXACT`, ROCm-only) is unchanged, and
  Metal/CPU/GLM/large spans still take the batched-verify + replay arm.
- A further ROCm-only nested experiment,
  `DS4_DIST_SPEC_EXACT_SHARED_ROWS=1`, changes only the admitted 2..5-row
  exact span. For each layer it runs
  `METAL_DECODE_LAYER_TO_SHARED_MID` sequentially for every verifier row,
  retaining causal KV/compressor/indexer/attention/router/routed-MoE order.
  It gathers each row's `ffn_norm`, routed output, post-attention HC, and HC
  split, issues one scalar-order Q8 pair-matmul launch followed by one SwiGLU
  launch, then runs the ordinary one-row shared-down/HC expansion,
  graph-pointer swap, and capture
  sequentially. It does not use the independent-session batching helpers.
- Admission is preflighted once before embeddings, cache writes, or capture
  reset. It requires resident prequant Q8 shared tensors, all row workspaces,
  one tier, no quality/TP/streaming/debug/profile/reference/steering/materialized-FFN
  mode, and no alternate shared-overlap/router path. Unsupported admission
  leaves the existing exact span untouched. Validation sets
  `DS4_DIST_SPEC_EXACT_SHARED_ROWS_REQUIRED=1`, which fails before graph/cache
  mutation unless the requested ROCm path is admitted; an admitted call emits
  `exact-span shared rows ACTIVE backend=ROCm` with its row and layer range.
  Once admitted, any failure invalidates the session; there is no mid-span
  fallback.

Original exactness argument and failed assumption

- Layer part: each row uses `metal_graph_encode_decode_layer`, as the
  single-token path (`ds4_session_eval_layer_slice` n_tokens==1 branch,
  ds4.c:60407/61008) and prior per-row exact loop do.  The interleave (layer
  il: row 0, row 1, ..., row n-1; then layer il+1) preserves the intended
  KV/compressed-KV write-before-read dependencies.  This remains a plausible
  weight-hot transformation, but v25f proves that the complete span cannot be
  called exact without differential hidden-state and logits evidence.
- Head part (f16/norm stages): `metal_graph_encode_output_head_exact_span`
  runs `output_hc_fn`, `output_hc_weights`, `hc_weighted_sum` and
  `output_norm` over per-row views with `n_tok == 1`, so
  `ds4_gpu_matmul_f16_tensor` selects the ordered-chunk single-token kernel
  (rocm/ds4_rocm_matmul.cuh:927-978) rather than the n_tok > 1 cublas/tree
  kernel; these stages are intended to retain the one-row arithmetic path.
  v25f showed that the old final Q8 path did not: it padded 2..7 rows to 8 and
  selected a generic ROCm direct-F32 batch kernel instead of the serial one-row
  Q8 prequant path.  The
  next candidate removes that padding and routes to the exact 2..5-row
  prequant kernel.  If prequant is disabled or its scratch is unavailable, the
  exact API issues ordinary one-row calls and never a generic multi-row call.
  This is an implementation fix, not yet end-to-end exactness or performance
  evidence.
- Capture ring: the interleaved loop swaps the graph's `cur_hc_by_tier` /
  `after_ffn_hc_by_tier` pointers alongside the local ping-pong swap before
  `metal_graph_dspark_capture_decode_layer`, mirroring the pointer swaps in
  `ds4_session_eval_layer_slice` immediately before each capture call
  (currently ds4.c:61115-61119 and 61141-61144), so the target-hidden ring
  records each layer's OUTPUT (last row wins) exactly as sequential decode does.
- Acceptance logic still accepts draft i iff
  `dist_logits_argmax(rows[i]) == drafts[i+1]`.  The release path then rewinds
  and runs `dist_mtp_spec_replay_accept` through ordinary one-token decode.
  The direct-commit experiment instead retains verifier KV/state and last-row
  logits after a full accept; therefore every layer, timeline, capture, and
  output-head side effect—not only each accepted argmax—must match serial
  replay before that shortcut can preserve the greedy stream.

ROCm validation result and containment

- v25f showed deterministic 512-token divergence in every direct-commit
  DSpark policy after a 505-character common decoded prefix. v26a disabled the
  padded fast verifier head and produced that same alternate completion, so
  the fast head was not the necessary cause.
- v27 forced restore plus serial one-token replay for every accepted prefix and
  still produced the identical alternate completion. v29 then set the actual
  server gate `DS4_MTP_SPEC_DISABLE=1`: it emitted no speculative WORK flags,
  proposals, verify spans, accepts, rollback, or replay, but merely loading the
  support model still produced the same bytes. This excluded all distributed
  speculative state machinery as the necessary identity seam.
- The root seam was ROCm's multi-model Q8 policy. Loading the support GGUF set
  `g_q8_f16_disabled_for_multi_model=1`; target prefill therefore fell back
  from its normal expanded-F16 Q8 path to different-arithmetic direct Q8
  kernels. v29's decode rate was unchanged while prefill fell from 82.14 to
  71.37 tokens/s, matching this mechanism.
- Sealed v30 kept DSpark target capture active, kept speculation disabled, and
  set `DS4_ROCM_Q8_F16_CACHE_MULTI_MODEL=1` on both ranks. It restored exact
  512-token identity (both reasoning payloads SHA256
  `5f38d237fa0622975c53d5affca6e86e66f18cba5294e5aa18475e5a436ccc71`),
  restored prefill (82.24 versus 82.10 tokens/s), and was wall-neutral
  (36.554s versus 36.519s). Restoration and dmesg gates passed.
- The runtime fix uses an explicitly marked primary target mapping. In a
  multi-model process, target Q8-to-F16 ranges remain eligible and are not
  discarded merely because a support mapping is loaded; support-model
  expansion remains disabled by default unless the existing explicit env opts
  it in. If an eligible primary preload cannot be created, startup fails rather
  than silently selecting different target arithmetic. Model handoffs
  synchronize before source-range release, and a support-cache failure disables
  only support expansion instead of discarding valid target ranges.
- `tests/test_q8_krow_rocm.c` includes a multi-model regression: it enters
  multi-model mode, requires a 2048x4096 primary Q8-to-F16 preload, evaluates
  eight rows, maps a second support image, and requires bitwise-identical target
  output afterward.
- The exact-head dispatch fix additionally removes the old 2..7-row padding
  and exercises the production 7168-wide input in both prequant-enabled and
  disabled modes. In enabled mode its oracle is the production one-row decode
  implementation (`matmul_q8_0_preq_rows_w32_kernel`) invoked separately for
  each row; the candidate is a different K-row kernel and launch. They
  intentionally share activation quantization and test data, so this proves
  the claimed property—bit-equivalence to ordinary target decode—not
  independent mathematical correctness of the Q8 quantizer or model quality.
  The disabled child instead proves that the exact API rolls back to separate
  ordinary non-prequant one-row calls, never a generic multi-row reduction.
  `DS4_ROCM_Q8_EXACT_KROW_REQUIRED=1` is test/benchmark-only and turns scratch
  or selector fallback into failure, so a passing production-head benchmark
  positively attests K-row dispatch.
- Sealed bilateral ROCm bundle `stage3-f7edfcf-krow-v2` validated commit
  `f7edfcf552363976a3f5b0ad50a1737d7dcb86ea` on both gfx1151 hosts. Every
  production-head row was byte-identical. Averaged across the two hosts, the
  4096x129280 K-row call took 2.461/2.474/2.481/2.487 ms for k=2/3/4/5 versus
  5.088/7.630/10.191/12.726 ms serial (2.07x/3.08x/4.11x/5.12x). The
  7168x129280 call took 4.264/4.271/4.274/4.280 ms versus
  8.933/13.343/17.736/22.163 ms (2.10x/3.12x/4.15x/5.18x). The 537 MiB and
  939 MiB weight streams therefore sustain about 229 and 231 GB/s; row time is
  nearly flat as k grows, so this kernel is already primarily memory-bandwidth
  limited. The bilateral build hashes, tests, restoration gates, and dmesg
  deltas all passed. This establishes exact Q8-head dispatch and isolated
  speedup, not complete hidden/KV or end-to-end inference identity.
- Sealed bilateral ROCm bundle `stage3-223b182-krow-v2` then validated the
  default-off shared gate/up exact-row implementation at commit
  `223b182e48c29b4cee22474a910262c1007227e0`. Both hosts produced identical
  production and test binary hashes, passed normal `strix-halo`, distributed,
  TP, MXFP4, graph, Q8 K=2..5, and production-head gates, and passed all four
  fail-closed shared-row static contracts. Restoration, zero dmesg deltas,
  and all 226 sealed evidence rows passed. This proves bilateral compilation,
  unchanged head arithmetic, and the source-level admission contract; it does
  not yet prove that production inference admits the shared path or preserves
  end-to-end identity.
- Sealed inference bundle `stage3-223b182-shared-inference-v3` closed that
  production admission and identity gap with one no-spec anchor and three
  alternating exact-base/exact-shared samples. All seven 512-token projections
  matched on role, finish reason, usage `(87,512,599)`, reasoning/content
  boundary, length 1374, and SHA256
  `5f38d237fa0622975c53d5affca6e86e66f18cba5294e5aa18475e5a436ccc71`.
  Every speculative sample had the identical work tuple `cycles=460`,
  `proposed=59`, `accepted_draft=53`, so the comparison is not explained by
  different scheduler density. Each shared sample admitted exactly 15 spans on
  both ranks, with row histogram `{3:3,4:6,5:6}`, worker layers `22:42`,
  coordinator layers `0:21`, and no shared-path rejection; base and no-spec
  logs had no shared-path contamination. Median wall time improved only from
  39.142 to 39.001 seconds (`0.9964x`), while median decode throughput improved
  from 13.45 to 13.50 tokens/s (`1.0037x`). This is a repeatable but insufficient
  approximately 0.14-second gain: shared exact mode still took 6.9% longer than
  no-spec (39.001 versus 36.482 seconds). The 15 admitted spans contained 63
  total rows, hence 48 rows beyond one row per span; the observed saving is
  consistent with eliminating those repeated shared gate/up weight streams at
  the independently measured roughly 230 GB/s bandwidth. It validates the
  batching mechanism and cost model, while showing that shared gate/up alone
  removes only a small fraction of the roughly 2.5-second remaining deficit.
  Restoration, the dmesg gate, exact launch receipts, and all 240 evidence rows
  passed; evidence manifest SHA256 is
  `e1c472d7164eba6b2495b50c43a223ecec2adac9f1ed1a7e7e422fc9e90f40eb`.
- Sealed no-env bundle `stage3-11a0c28-inference-v32e` also validated the
  default primary-target cache policy with a loaded support model and
  `DS4_MTP_SPEC_DISABLE=1`. Both 512-token payloads were byte-identical to each
  other and sealed v30 (SHA256
  `5f38d237fa0622975c53d5affca6e86e66f18cba5294e5aa18475e5a436ccc71`):
  14.45 versus 14.42 decode tokens/s and 36.487 versus 36.561 seconds wall
  (1.002x). The required primary-preserved/support-disabled policy line was
  present on both support-loaded roles, the multi-model override was absent,
  and restoration/dmesg/evidence gates passed. This proves the default policy
  no longer changes target arithmetic merely by loading support; speculation
  was deliberately disabled and no proposal correctness is implied.
- Sealed bundle `stage3-f7edfcf-inference-v3` then exercised real proposals at
  commit `f7edfcf552363976a3f5b0ad50a1737d7dcb86ea` with
  `DS4_DIST_SPEC_EXACT=1`, `DS4_DIST_SPEC_EXACT_SPAN=1`, and
  `DS4_ROCM_Q8_EXACT_KROW_REQUIRED=1` on both ranks. The no-spec and exact
  512-token payloads both had length 1374 and SHA256
  `5f38d237fa0622975c53d5affca6e86e66f18cba5294e5aa18475e5a436ccc71`.
  The coordinator recorded 460 cycles, 59 proposals, and 53 accepted drafts
  (89.83% acceptance). Required exact/span warnings, K-row receipt evidence,
  and the primary-preserved/support-disabled policy were present on both
  ranks; restoration, zero dmesg deltas, and all 130 sealed evidence rows
  passed. This closes end-to-end identity for this 512-token production-width
  run, but not every possible hidden/KV/capture state or span length.
  Performance is not yet a win: exact delivered 13.42 versus 14.45 decode
  tokens/s and 39.205 versus 36.487 seconds wall (7.1% lower throughput,
  7.45% longer wall); prefill was 81.80 versus 82.19 tokens/s. The report's
  screening decision therefore remains negative despite correctness PASS.
  Accepted drafts are not free target forwards in exact mode: they were
  computed in verifier spans. A useful first-order row count is 460 cycles +
  59 proposed/target-evaluated draft rows = 519 target rows versus 512 in the
  baseline; 53 accepted drafts compress the emitted cycle count but do not
  remove their target evaluation. Worker proposal accounting retained
  1.858 seconds (1.547 seconds in the support stage chain), about 68% of the
  2.718-second wall regression; rejected rows, partial replay, span setup,
  synchronization/readback, and transport account for the remainder. Thus an
  exact speedup requires broad K-row target-layer amortization and/or a much
  cheaper proposer, not acceptance accuracy alone.
- Sealed scheduler-density probe `stage3-f7edfcf-inference-v4` kept the same
  exact/span/K-row-required path but set `DS4_DSPARK_SCHEDULER=0`. Identity,
  restoration, dmesg, receipts, and all 130 evidence rows again passed. It
  increased proposals/accepts to 162/132 over 381 cycles, but proposal work
  rose to 6.342 seconds (5.429 seconds in the stage chain), exact decode fell
  to 11.19 versus 14.43 tokens/s, and wall rose to 46.820 versus 36.549
  seconds (+28.10%). The scheduler-enabled v3 exact arm was 7.615 seconds
  faster than this probe while the two no-spec controls differed by only
  0.062 seconds. More proposal coverage does not help until verification is
  substantially cheaper; the existing no-draft/confidence backoff remains the
  better measured policy.
- A first-order break-even model for the scheduler-enabled run is stricter
  than the acceptance rate suggests. Charging all 1.858 seconds of proposer
  work and about 0.300 seconds of remaining span overhead gives
  `59*c < 52*71.3ms - 1858ms - 300ms`, so the marginal target cost `c` of a
  drafted verifier row must fall below about 26 ms—roughly a 2.7x reduction—
  before exact speculation merely breaks even. If rejected rows retain a full
  standalone cost while only accepted rows receive the batched marginal cost,
  the bound is about 21 ms (3.4x). Current measurements cannot distinguish
  those two models because the marginal row still costs approximately a full
  row. Halving total proposer cost would relax the first bound to about 42 ms;
  therefore proposer reduction and routed-expert overlap/fraction measurements
  are prerequisites to deciding whether broad verifier batching is feasible.
- `DS4_DIST_SPEC_EXACT` remains explicit opt-in. The default verifies drafts,
  restores the speculative frontier, and serially replays accepted tokens.
  This is conservative containment of an unproven direct-state-retention path,
  not a claim that the fast path is correct: v25f disproved end-to-end identity
  for the then-current experiment, and the Q8-head repair has not yet proved
  all hidden/KV/capture state. `DS4_DIST_SPEC_EXACT_SPAN` therefore remains
  nested and experimental. `DS4_DIST_SPEC_EXACT_SHARED_ROWS` is nested one
  level further, defaults off, and prints a prominent warning; it has no
  bearing on the sealed results above.

## Endurance validation status

- Frozen bundle `stage3-ae762ae-endurance-v9` received independent review
  PASS for its exact six-phase AB/BA/AB protocol, evidence publication, and
  fail-closed checker. Its SHA256SUMS digest is
  `dfdd7d045a2779b706d15ad5b4fe8d8799071f8f16f7e41da0a5e679bf47d65e`.
- Its one authorized execution is active; there is no endurance runtime result
  yet, and the reviewed bundle must not be cited as performance or correctness
  evidence until its atomic SUCCESS receipt is published and independently
  checked. It targets code commit `ae762ae`. The later `5eccd5a` source fixes
  materially change transport/state handling and require their own compile and
  targeted node validation; the endurance result cannot be attributed to them.
- Until that run completes, the measured performance decision remains negative:
  exact/shared paths stay nested, experimental, and default-off. The existing
  512-token evidence establishes targeted correctness and mechanism behavior,
  not a general speedup or direct-commit default.

## Validation required before any direct-commit default

1. **Passed (v32e):** ROCm-build the primary-only multi-model cache fix and
   prove no-env support-loaded plain decode matches no-spec for 512 greedy
   tokens on both ranks. The sealed phase retained policy, cache, executable,
   output, restoration, and dmesg evidence.
2. **Passed (`f7edfcf` krow v2):** run the ROCm k-row test at the real
   7168-wide input for every 2..5-row exact-API span and its self-forced
   prequant-disabled rollback child, then run the fail-closed production-head
   benchmark. Both hosts passed every bitwise comparison and retained the
   timings above.
3. **Passed (`f7edfcf` inference v3):** require real proposals and accepts,
   required K-row dispatch, and exact 512-token identity for the direct exact
   span. This production-width run passed with 59 proposed and 53 accepted
   drafts, but its negative performance screening keeps the path experimental.
4. Repeat the replay-default speculative arm with real proposals and accepts;
   require exact token identity now that target arithmetic is held constant.
5. Compare fast-span and per-row hidden rows, KV/compressed-KV,
   compressor/indexer state, capture state, and final logits before considering
   direct verifier-state retention. For the shared-row sub-experiment, cover
   K=2,3,4,5 separately and compare every layer boundary against both the
   ordinary exact span and serial one-row decode, including routed/shared
   intermediates and the last-row capture ring.
6. **Passed for production admission, identity, and end-to-end timing
   (`223b182` shared-inference v3):** the required-mode `ACTIVE` ranges and
   identical row sequence were attested bilaterally with no rejection or
   baseline contamination. The measured gain was only about 0.14 seconds, so
   route-overlap and broader verifier/proposer diagnostics remain required.
   Kernel-level launch-count tracing remains a separate diagnostic rather than
   a prerequisite for the already sealed output-identity result.
7. Only after all state and token gates pass may verifier state bypass replay
   or the experimental exact span become a default.
