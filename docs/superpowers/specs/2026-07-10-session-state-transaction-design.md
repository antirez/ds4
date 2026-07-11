# Session-State Transaction Design

Status: Accepted for implementation on `server-enhancements`

## Objective

Make every admitted `ds4-server` request pass through one worker-owned
session-state transaction and produce exactly one immutable terminal outcome.
The transaction controls live-session mutation, continuation frontiers,
checkpoint settlement, output finalization, tracing, counters, and cleanup.

This is a state transaction, not a wire transaction. Bytes successfully handed
to the socket are irreversible and are never described as rolled back.

## Scope

This change includes:

- a worker-published immutable snapshot for `/stats` and configured context
  metadata;
- typed execution phase, terminal reason, session disposition, and wire
  disposition values;
- a sticky first-failure latch and one idempotent terminalizer;
- private production and scripted session, output, trace, and statistics
  adapters;
- deterministic failure tests for restore, sync, decode, cancellation, output,
  commit, rollback, trace, statistics, and cleanup;
- incremental migration of the current cache-selection, sync, decode,
  continuation settlement, checkpoint, output, trace, statistics, and cleanup
  blocks into the transaction.

This change does not redesign protocol events, continuation selection policy,
or the KVC payload/format. It does not add a public header or split unchanged
code into forwarding files.

## Chosen module shape

Three designs were compared:

1. A minimal `run -> outcome` module with a small internal phase sequence.
2. An explicit state machine whose legal transitions are represented in an
   enum and switch.
3. A common-path transaction centered on one cleanup label and a rollback
   journal.

The implementation combines their strongest properties: one narrow
worker-facing call, an explicit forward-only lifecycle, and a single idempotent
terminalizer with per-effect completion bits. The explicit lifecycle makes
failure injection and illegal transitions testable; the narrow entry point
keeps the worker and clients unaware of model, checkpoint, or protocol details.

The private worker-facing interface is conceptually:

```c
static server_txn_outcome server_txn_run(server *s, job *j);
```

`worker_main()` invokes it for every `ENQUEUE_OK` job, including a job whose
client disconnected while queued. The returned value is stored in the job
before `done` is signalled. Queue-full and stopping responses are pre-admission
and therefore outside the transaction.

## Typed lifecycle

The lifecycle is forward-only, except for the existing single bounded DSML
recovery retry:

```text
ADMITTED
  -> RESTORE
  -> SYNCHRONIZE
  -> EXTEND
  -> DECODE [contains the existing bounded recovery retry, at most once]
  -> SETTLE
  -> TERMINALIZE
  -> DONE
```

The implementation records:

- `server_txn_phase`: where execution is or where a result was decided;
- `server_txn_reason`: model stop, length, tool calls, continuation conflict,
  queued disconnect, cancellation, shutdown, restore/sync/decode/output/commit
  failure, or invariant failure;
- `server_session_disposition`: unchanged, valid prefix retained, committed,
  or invalidated;
- `server_wire_disposition`: untouched, started, irreversible, complete, or
  broken;
- secondary failure bits for rollback, checkpoint maintenance, output
  finalization, trace, statistics, and cleanup.

The terminal outcome owns all of its values; it contains no borrowed request,
session, trace, or output pointers.

## Ownership invariants

- The worker is the only thread that calls the production session adapter or
  mutates live token position, checkpoint scheduling, or continuation
  frontiers.
- The configured context size is copied into `server` during initialization.
  Client threads use that immutable value for parsing and model metadata.
- Shutdown checkpoint persistence runs in the worker epilogue. After joining,
  the main thread only waits for clients and destroys resources.
- `/stats` serializes one copied `server_stats_snapshot`. It never calls
  `ds4_session_pos()`, `ds4_session_ctx()`, or any adapter.

The snapshot contains all fields emitted by `/stats`, including counters,
queue depth, clients, busy state, live tokens, context size, and a monotonically
increasing version. Queue/client owners may update their own scalar fields
under `server.mu`; only the worker publishes live-session fields and terminal
statistics.

## Adapter boundaries

The module has four private adapter families, each with production and scripted
implementations:

- Session: plans/restores a usable frontier, synchronizes prompt state, performs
  decode/recovery, and settles commit or rollback as cohesive operations.
- Output: observes peer state, manages prefill/open/delta/terminal writes, and
  owns the monotonic wire ledger.
- Trace: consumes transaction observations and closes one request trace.
- Statistics: publishes progress snapshots and applies one terminal accounting
  observation.

These are not mirrors of `ds4.h` or the protocol helper set. Each production
operation owns a substantial existing lifecycle block. Scripted adapters use an
ordered in-memory operation log whose tests fail on an unexpected sequence;
they never open a socket, load a model, touch a KVC file, or inspect a live
session.

## Exactly-once terminalization

Every path reaches `server_txn_terminalize()`. It marks terminalization started
before invoking any effect and caches the final outcome. Calling it again
returns that cached outcome without repeating any effect.

Terminalization uses monotonic per-effect flags and attempts this controlled
sequence:

1. Freeze the first causal reason and intended state disposition.
2. Detach session callbacks and commit or roll back continuation/checkpoint
   state as appropriate.
3. Attempt the one allowed terminal wire projection, respecting the wire
   ledger; a broken wire forbids retries or an alternate HTTP response.
4. Close the typed trace record while transaction-owned trace data is live.
5. Release all transaction-owned resources once.
6. Apply terminal counters once and publish the final immutable snapshot, so
   `busy=false` means tracing and cleanup have also completed.
7. Cache and return the terminal outcome.

Frontier settlement remains before the final protocol event, preserving the
current continuation semantics and protocol event ordering. Trace, cleanup,
and counters cannot replace the primary outcome; their failures are secondary.

## Failure precedence

`server_txn_fail_once()` records the first causal execution failure. Later
failures cannot replace it.

- A session restore, sync, or decode failure beats failures encountered while
  reporting or cleaning up that failure.
- A stream write failure observed before cancellation remains primary; the
  cancellation is a consequence.
- If cancellation conditions are observed together, shutdown takes precedence
  over peer disconnect.
- A required in-memory continuation commit failure is primary when no earlier
  failure exists and leaves the session invalidated.
- An output-finalization failure is primary only when execution had not already
  failed. Once a write may have emitted bytes, the wire becomes broken and is
  never retried.
- Rollback, checkpoint maintenance/canonicalization, tracing, statistics, and
  cleanup failures are secondary. They are recorded and surfaced in the typed
  outcome but do not overwrite the primary result.

Successful model termination and delivery are orthogonal: for example, a model
may stop normally while the terminal wire write fails. The terminal reason is
then output failure, while the recorded model finish remains stop.

## Rollback and irreversible output

Rollback means settling the live session to the strongest valid state supported
by the engine contract. It is not an ACID snapshot restore:

- interrupted `ds4_session_sync()` retains its guaranteed valid prefix;
- a generic restore, sync, or decode failure may invalidate the live session;
- failed or cancelled execution does not publish newly parsed Responses,
  Anthropic, thinking, or tool-call continuation frontiers and does not run
  commit-only checkpoint canonicalization;
- a suppressed continued-store frontier and deferred disk checkpoint are
  restored or consumed using the existing rules;
- no full per-request KV snapshot is introduced, and token-only rewind is not
  treated as a complete backend rollback.

The output adapter marks the wire irreversible before any operation that may
partially write. Session rollback never changes that wire disposition and never
causes an alternate response or replay of a terminal event.

## Compatibility contract

Migration keeps the existing branch bodies and ordering until scripted and
socketpair tests pin equivalent behavior. Specifically, this change preserves:

- OpenAI chat/completions, Responses, and Anthropic endpoint status and bodies;
- streaming header, keepalive, delta, usage, finish, and `[DONE]` ordering;
- cache-source and continuation-frontier precedence;
- stop-sequence trimming, DSML repair/recovery limits, and tool-call IDs;
- checkpoint filenames, payloads, scheduling, consume/discard timing, and
  canonicalization behavior;
- current nonfatal treatment of checkpoint maintenance failures.

## Verification strategy

Deterministic scripted tests run the same phase driver as production and assert
typed restore, synchronization, decode, cancellation, commit, rollback, trace,
statistics, and cleanup failures. The shared output seam injects failures at
prefill, stream-open, incremental-update, flush, and terminal-finalization
operations. Tests also pin terminal-effect ordering, shutdown/output causal
precedence, settlement-time broken-wire behavior, repeated terminalizer
invocation, and checkpoint-failure disposition. A production
queued-disconnect test proves terminal accounting without a live session. The
`/stats` tests prove serialization works with `server.session == NULL` and
that queued work wins over a coalesced idle snapshot refresh.

Existing server socketpair/protocol, continuation, and KVC tests cover deferred
checkpoint handling, recovery behavior, protocol bytes, and cache compatibility.
Build verification uses `make`; server unit verification uses
`./ds4_test --server`. Live verification uses the user's local Metal server after the
replacement binary passes those gates.
