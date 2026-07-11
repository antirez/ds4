# Candidate 01 — Session-State Transaction

## Scope

- Implement exactly-once terminalization for admitted `ds4-server` requests.
- Keep live session/frontier mutation on the single worker.
- Publish immutable statistics snapshots; `/stats` must not inspect `ds4_session`.
- Preserve current wire ordering, continuation behavior, checkpoint compatibility, and the running local Metal process.
- Add deterministic scripted failure coverage and a concise ADR.
- Exclude protocol-adapter, continuation-policy, and KVC-format refactors.

## Plan

- [x] Compare three deep-module interfaces and record the approved design and implementation plan.
- [x] Add a worker-published immutable `/stats` snapshot test-first.
- [x] Add typed execution phases, reasons, dispositions, and one idempotent terminalizer test-first.
- [x] Add private production and scripted adapters with deterministic failure precedence tests.
- [x] Migrate restore, sync, decode, output, commit/rollback, tracing, statistics, and cleanup through the controlled lifecycle.
- [x] Add the domain glossary entry and ADR.
- [x] Route prefill/open/delta output and nonterminal trace/statistics observations through the private adapters.
- [x] Correct causal-failure, checkpoint-disposition, wire, and stats-publication edge cases found by guard review.
- [x] Add deterministic coverage for operational adapter failures, shutdown/output precedence, checkpoint disposition, and idle stats refresh.
- [x] Run safe verification, guard reviews, inspect the diff, and commit only scoped files.

## Acceptance criteria

- Every admitted request produces exactly one typed terminal outcome.
- Terminalization and cleanup are idempotent and run through one path.
- The primary failure survives rollback, trace, statistics, output-finalization, or cleanup failures.
- Streamed bytes are recorded as irreversible and are never described as rolled back.
- `/stats` reads one immutable snapshot and never calls a live-session accessor.
- Scripted tests cover restore, sync, decode, cancellation, output, commit, rollback, tracing, and cleanup failures.
- Existing server tests and the normal build pass without stopping the currently running server.

## Review

- Replaced the worker's request monolith with one shared, forward-only
  transaction driver and private production/scripted session, output, trace,
  and statistics adapters. All admitted jobs publish one typed outcome through
  an idempotent terminalizer.
- Routed prefill keepalives, stream open/update/flush/finalization, trace
  begin/event/piece/finalization, request counters, progress snapshots,
  settlement, cleanup, and checkpoint accounting through the controlled
  lifecycle. Rollback no longer publishes a newly parsed continuation frontier.
- `/stats` now copies only a worker-published immutable snapshot. Worker refresh
  coalescing prioritizes admitted jobs, and terminal publication occurs after
  trace and cleanup so `busy=false` is truthful.
- Added explicit session-validity inspection, sticky shutdown/output failure
  precedence, broken-wire no-retry behavior, checkpoint secondary failures,
  and actual-validity rollback disposition.
- Verification passed:
  - warning-free `make -B ds4-server ds4_test`;
  - `./ds4_test --server`;
  - `./ds4-eval --self-test-extractors`;
  - `git diff --check`;
  - independent spec, concurrency, test/documentation, and clean-code audits.
- Gracefully stopped PID 81030 after it persisted 101,224 live tokens to the
  disk cache, then restarted the exact requested command as PID 31807. Metal
  mapped about 95.1 GiB, allocated about 8.6 GiB for the 524,288-token context,
  reopened the 32 GiB cache, and listened on `127.0.0.1:8000`.
- Live health, model metadata, and immutable stats checks passed. A one-token
  chat smoke request completed in 2.7 seconds; the final snapshot reported
  `busy=false`, one request, nine live tokens, and no rejection/cancellation.
  Its trace closed with completed/length, committed session, complete wire, and
  no secondary failure.
- Supplemental macOS AddressSanitizer was not a passing gate: LeakSanitizer is
  unsupported, and the leak-disabled instrumented server suite was time-boxed
  and terminated when it remained unexpectedly CPU-heavy. CUDA, distributed,
  and SSD-streaming model modes were not exercised; the live Apple M5 Max Metal
  path and disk checkpoint cache were exercised.

---

# Prior: ds4 Server Architecture Review — 2026-07-10

## Scope

- Review `ds4-server` for architectural deepening opportunities.
- Preserve its role as a local DeepSeek V4 Flash Metal server on the Apple M5 Max with 128 GiB unified memory.
- Keep source code unchanged; write the visual report to the OS temp directory.
- Preserve the user's existing untracked files.

## Plan

- [x] Launch the requested server command and verify port 8000 is listening.
- [x] Monitor live server logs while the user exercises it during this review.
- [x] Read project guidance, server documentation, current implementation, tests, and recent server-enhancement history.
- [x] Identify candidates that pass the deletion test and validate each with file-level evidence.
- [x] Generate, validate, and open a self-contained HTML report with before/after diagrams.

## Acceptance criteria

- The requested server remains running on `127.0.0.1:8000` with tracing enabled.
- Every candidate names involved files, dependency category, architectural friction, deepening, locality, leverage, and test impact.
- No candidate re-proposes work already implemented on `server-enhancements`.
- The report ends with one top recommendation and proposes no concrete interfaces.

## Out of scope observations

- `ds4_server.c:11617-11663` lets a client thread read the worker-owned session position while the worker mutates it; the report treats this as architectural evidence, but this review does not change source code.
- `README.md:1039-1052` trails the current KVC header and extension flags in `ds4_kvstore.h:15-18,36-57`; documentation repair is separate from this architecture review.

## Review

- Started the exact requested command and verified the Metal-backed DeepSeek V4 Flash server on `127.0.0.1:8000`.
- Monitored the live tool loop through roughly 132K tokens of session depth. Requests completed with normal `stop` or `tool_calls` finishes; no queue drops, cancellations, socket failures, model failures, malformed DSML, or corrupt-cache warnings appeared.
- Normal disk-budget pressure evicted zero-hit checkpoints and successfully wrote cold/continued replacements.
- Produced and opened `/var/folders/fs/brn3wr_x4ns2km1cq1ph9zb80000gn/T/architecture-review-20260710-200425.html` with four deletion-test-backed candidates.
- Top recommendation: make request execution one terminal transaction while preserving the single local Metal worker.
- Verified the report in headless Chrome, including Tailwind and Mermaid rendering, and ran `./ds4_test --server` successfully.
- No production or test source was changed. Repo changes are limited to the task tracker and lesson required by the operating manual.

# Task: Post-review robustness fixes for the session transaction (2026-07-10)

## Plan

- [x] Two-axis review (standards + spec) of 5a91c33..a2815e5 → verify: findings cite file:line evidence.
- [x] Fix rollback retaining stale protocol live bindings → verify: new unit test fails before, passes after.
- [x] Consume the `kv_cache_store_current("shutdown")` result → verify: warning logged on failure path.
- [x] Compile `test_txn_event` only under `DS4_SERVER_TEST` → verify: warning-free production build.
- [x] Downgrade duplicate-terminal-publish `die()` to a loud log → verify: `./ds4_test --server` still passes.
- [x] Align spec text with the queued-disconnect encoding (client gone, restore phase) → verify: doc reads true against `production_restore`.

## Review

- Declined review item "wire mislabeled BROKEN on zero-byte failure": the spec's monotonic wire ledger deliberately treats any failed write attempt as may-have-emitted (spec lines 170-171, 195-196), scripted tests pin that contract, and the cited stream-failed early return is unreachable from `production_output_finish` because the wire is already BROKEN there.
- Kept the worker-ownership `die()` calls: they guard cross-thread session mutation, which is memory-unsafe to continue past.
- Deferred (out of scope): collapsing the adapter vtable layer to direct calls, deduplicating the two trace vsnprintf wrappers and the two failure latches, and whether docs/tasks files belong in the upstream PR.
- Verified: warning-free `make ds4-server ds4_test`; `./ds4_test --server` OK including new `test_production_settle_rollback_clears_live_bindings`.
