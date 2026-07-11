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
- [ ] Add a worker-published immutable `/stats` snapshot test-first.
- [ ] Add typed execution phases, reasons, dispositions, and one idempotent terminalizer test-first.
- [ ] Add private production and scripted adapters with deterministic failure precedence tests.
- [ ] Migrate restore, sync, decode, output, commit/rollback, tracing, statistics, and cleanup through the controlled lifecycle.
- [ ] Add the domain glossary entry and ADR.
- [ ] Run safe verification, guard reviews, inspect the diff, and commit only scoped files.

## Acceptance criteria

- Every admitted request produces exactly one typed terminal outcome.
- Terminalization and cleanup are idempotent and run through one path.
- The primary failure survives rollback, trace, statistics, output-finalization, or cleanup failures.
- Streamed bytes are recorded as irreversible and are never described as rolled back.
- `/stats` reads one immutable snapshot and never calls a live-session accessor.
- Scripted tests cover restore, sync, decode, cancellation, output, commit, rollback, tracing, and cleanup failures.
- Existing server tests and the normal build pass without stopping the currently running server.

## Review

Pending.

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
