# Session-State Transaction Implementation Plan

> Execute on `server-enhancements`. Keep the existing local Metal server alive;
> compilation and unit tests must not launch a second model process.

## Goal

Every admitted request executes through one worker-owned transaction, produces
one typed terminal outcome, and publishes immutable statistics without changing
protocol, continuation, or checkpoint behavior.

## Task 1: Pin the immutable stats boundary

Files: `ds4_server.c`

1. Add a server test that gives `send_stats()` a populated published snapshot
   and no live session, then asserts the JSON values and lack of adapter access.
2. Run `make ds4_test && ./ds4_test --server`; confirm the new test fails against
   the direct session accessor.
3. Add `ctx_size`, `server_stats_snapshot`, and a mutex-protected snapshot copy
   to `server`.
4. Make model metadata, request parsing, and `/stats` use immutable copied
   values. Add worker publication points for startup, progress, decode, and
   terminal settlement.
5. Re-run the focused server suite and inspect all remaining
   `ds4_session_pos/ctx` call sites for worker ownership.

## Task 2: Add the typed lifecycle and terminalizer

Files: `ds4_server.c`

1. Add compile-failing scripted tests for typed phases, primary reason, session
   and wire dispositions, and secondary failure bits.
2. Add a transaction script that records ordered effects and injectable
   outcomes without sockets, model state, traces, or filesystem access.
3. Test repeated terminalization first: two calls must yield the same outcome
   and exactly one commit/rollback, output finalization, trace close, statistics
   application, and cleanup.
4. Implement `server_txn_fail_once()` and `server_txn_terminalize()` with
   terminalization-started and per-effect completion bits.
5. Add table-driven cases proving first-failure precedence and irreversible
   wire behavior.
6. Run `make ds4_test && ./ds4_test --server` after each red/green slice.

## Task 3: Add private production and scripted adapters

Files: `ds4_server.c`

1. Define private coarse session, output, trace, and statistics adapter
   interfaces. Keep them local to the server translation unit.
2. Implement the scripted adapters as an ordered operation program with
   failpoints for restore, sync, decode, cancellation, output, commit,
   rollback, trace, statistics, and cleanup.
3. Add strict tests for every required failure site, including compound cases
   where rollback/trace/cleanup fail after an earlier primary failure.
4. Implement production adapters by moving complete lifecycle blocks. Do not
   add wrappers that merely forward individual `ds4_session_*` or protocol
   helper calls.

## Task 4: Migrate request execution

Files: `ds4_server.c`

1. Give `job` an owned terminal outcome and readiness flag.
2. Move queued-disconnect handling, busy state, and admitted-request accounting
   into `server_txn_run()` so every admitted job terminalizes.
3. Convert the existing early returns in generation to typed failures leading
   to the single terminal path.
4. Migrate cache selection/restore, synchronization/prefill, extension,
   decoding/recovery, continuation settlement, final output, trace, counters,
   and cleanup into explicit phases while retaining branch bodies and ordering.
5. Delete `generate_job()` after its behavior has moved; do not leave a
   forwarding wrapper.
6. Move shutdown checkpoint persistence into the worker epilogue.
7. Run `make ds4_test && ./ds4_test --server`, then inspect golden protocol and
   continuation/KVC tests for exact compatibility.

## Task 5: Record the architecture

Files: `CONTEXT.md`, `docs/adr/0001-session-state-transaction.md`,
`tasks/todo.md`

1. Keep the glossary limited to Live Session, Session-State Transaction, and
   Terminal Outcome.
2. Record worker ownership, rollback versus wire semantics, exactly-once
   terminal accounting, failure precedence, adapter boundaries, and behavioral
   compatibility in the ADR.
3. Mark completed checklist items and add the concrete verification results and
   any deferred platform checks to `tasks/todo.md`.

## Task 6: Review and verify

Files: all scoped changes

1. Run `make` and `./ds4_test --server` without launching a second model.
2. Run any lightweight server-specific self-tests available in the tree.
3. Re-scan for client-thread session access and multiple terminalization paths.
4. Review production code, tests, documentation, and the complete diff using
   the configured guard/review skills; fix blocking findings.
5. Confirm the original server process still listens on port 8000 and inspect
   its new logs for failures.
6. Commit only scoped source, tests, context, ADR, plan, and task files. Exclude
   `CLAUDE.local.md` and `ds4_agent_test`; do not push.
