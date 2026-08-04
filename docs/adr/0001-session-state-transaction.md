# ADR 0001: Worker-Owned Session-State Transactions

- Status: Accepted
- Date: 2026-07-10

## Context

`ds4-server` has one mutable inference session and one worker, but request
lifecycle effects are spread across queue handling, generation, progress
callbacks, output branches, tracing, counters, and cleanup. Client-thread
`/stats` and model metadata also read the live session directly. Early returns
can therefore bypass consistent tracing or accounting, and failure precedence
is implicit.

## Decision

Every admitted request runs through one private worker-owned session-state
transaction and returns exactly one typed terminal outcome.

- Only the worker reads or mutates live session position and mutates
  continuation frontiers. Client parsing may validate call IDs against
  mutex-protected frontier metadata, but `/stats` reads only a worker-published
  immutable snapshot and client code uses an immutable configured context size.
- One idempotent terminalizer settles session state, finalizes output, closes
  tracing, cleans up, applies counters, and publishes statistics. Each effect
  occurs at most once even if terminalization is invoked again; the final
  snapshot reports idle only after trace and cleanup complete.
- The first causal execution failure is primary. Rollback, checkpoint
  maintenance, terminal reporting, tracing, statistics, and cleanup failures
  are recorded as secondary and cannot overwrite it.
- Session rollback means retaining the strongest engine-guaranteed valid prefix
  or invalidating the live state. Failed and cancelled execution does not
  publish a newly parsed continuation frontier or canonicalize a new
  checkpoint. Rollback is not a full KV snapshot restore.
- Socket bytes are irreversible. Once a write may have emitted bytes, rollback
  cannot retract them and a broken stream is never retried as another HTTP or
  terminal response.
- Session, output, trace, and statistics behavior are hidden behind coarse
  private production and deterministic scripted adapters. They do not mirror
  individual engine or protocol functions.

## Compatibility requirements

The migration preserves endpoint responses, streaming event order,
continuation/cache precedence, DSML recovery limits, tool identifiers,
checkpoint scheduling and payload compatibility, and current nonfatal
checkpoint-maintenance behavior. Protocol-adapter, continuation-policy, and KVC
format refactors are separate decisions.

## Consequences

Failure ordering and state disposition become explicit and testable, `/stats`
cannot race the worker's live checkpoint vector, and every admitted job is
accounted for once. The server gains a larger private transaction context and a
small amount of snapshot staleness between worker publication points. It does
not gain atomic network output or cheap full-session rollback.
