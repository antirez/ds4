# Agent KV Context Hardening Fixes

This branch isolates a small set of hardening fixes for the agent context and
KV restore path. The goal is not to add more agent features here. The goal is
to make the existing context checkpoint, restore, and compaction behavior fail
more safely, report more honestly, and be easier for the model to call
correctly.

The fixes were driven by tests. In particular, the compaction canary e2e exposed
that a malformed compaction summary could be accepted as if it were useful
state. The runtime now rejects that case instead of rebuilding the live context
from bad memory.

## 1. Compaction Rejects Broken Summaries

This does not mean DS4 stops compacting.

It means:

- DS4 tries to compact normally.
- To compact, it asks the model for a useful summary of the current task state.
- If the summary is useful, DS4 uses it.
- If the summary is almost empty or clearly broken, DS4 rejects it.

Before:

```text
broken summary -> DS4 uses it anyway -> possible corrupted task memory
```

Now:

```text
broken summary -> DS4 aborts compaction -> corrupted memory is not used
```

How it is implemented:

- `agent_compact_summary_has_signal()` checks that the generated summary has a
  minimal amount of real text, not just a few tag-like words.
- The internal compaction prompt asks for plain text headings or bullets and
  explicitly rejects XML/HTML-like tool markup.
- If the check fails, compaction returns an error and invalidates the live KV
  session before any rebuilt context is accepted.

Verification:

- `make test-agent-context-compact-canary`
- The test requires DS4 to compact, then write five canary facts only after
  compaction.
- The harness checks that compaction really happened, reduced token count, kept
  the recent tail late enough, and preserved all canary values.

## 2. KV Restore Does Not Leave Half-Restored State

This does not make KV restore more limited.

It means:

- DS4 tries to load a saved KV cache.
- Then it checks that the loaded cache matches the context metadata.
- If the cache and metadata match, restore succeeds.
- If they do not match, DS4 avoids leaving the live session in an ambiguous
  half-restored state.

Before:

```text
partial restore -> mismatch detected later -> possible session/transcript drift
```

Now:

```text
invalid restore -> live KV is invalidated or resynced -> no half-restored state
```

How it is implemented:

- The restore path validates loaded token counts against checkpoint metadata.
- If validation fails after a KV load, the live session is invalidated so the
  next operation cannot accidentally continue from the bad KV state.

Verification:

- `make test`
- Context unit tests cover checkpoint metadata loading and incompatible restore
  handling.

## 3. Restore Metrics Are Explicit About Expected Versus Actual Savings

This does not change how the cache itself works.

It means:

- DS4 can estimate how many prefill tokens should be avoided by restoring KV.
- After restore, DS4 can also observe what actually happened during sync.
- If expected and actual behavior differ, the model-visible notice should not
  hide that difference.

Before:

```text
expected savings -> shown as if they definitely happened
```

Now:

```text
expected/actual savings -> reported more clearly -> less misleading feedback
```

How it is implemented:

- Restore bookkeeping tracks expected saved prefill tokens separately from
  actual cached-token observations after sync.
- The restore notice and trace avoid presenting estimates as stronger proof
  than they are.

Verification:

- `make test`
- `make test-kv-cache-benefit`
- The benchmark compares a full prefill against a restored-prefix run and
  reports `full_prefill_tokens`, `restored_prefill_tokens`, and
  `saved_prefill_tokens`.

## 4. Tool Schemas List The Allowed Actions

This does not reduce the tool's capability.

It means:

- The context tool accepts a fixed set of action names.
- The schema now tells the model exactly which actions are valid.
- The model has less room to invent plausible but wrong action names.

Before:

```text
action = any string -> model may invent an invalid action -> runtime error
```

Now:

```text
action = one of the allowed names -> fewer avoidable tool-call errors
```

How it is implemented:

- The context tool schema uses a JSON `enum` for action values such as
  `status`, `checkpoint`, `list`, `restore`, `compact`, and `drop`.

Verification:

- `make test`
- The schema is model-visible and the dispatch path still rejects unknown
  actions at runtime.

## 5. Metadata Parsing Is Key-Aware

This does not change the checkpoint metadata format.

It means:

- DS4 still reads the same JSON metadata files.
- It no longer finds fields by blindly searching for a word anywhere in the
  file.
- It distinguishes a real key from the same text appearing inside a value.

Before:

```text
search raw text -> possible confusion between key and value
```

Now:

```text
read the actual key -> metadata is interpreted more reliably
```

How it is implemented:

- The metadata reader now scans for JSON object keys instead of using a plain
  substring search.
- Tests cover pathological values that contain text resembling other keys.

Verification:

- `make test`
- The context unit test covers metadata roundtrip and key-aware lookup.

## 6. Benchmark Build Artifact Is Ignored

This is only repository hygiene.

It means:

- `make test-kv-cache-benefit` may build
  `tests/ds4_kv_cache_benefit_test`.
- That generated binary should not make `git status` look dirty after the test.

Before:

```text
run benchmark -> generated binary appears as untracked file
```

Now:

```text
run benchmark -> generated binary is ignored -> working tree remains clean
```

How it is implemented:

- `.gitignore` includes `/tests/ds4_kv_cache_benefit_test`.

Verification:

- `make test-kv-cache-benefit`
- `git status --short`

## 7. Adaptive Self-Improvement E2E Scope

This test demonstrates the agent loop, not a real DS4 code optimization.

It means:

- DS4 is given a temporary repository with a small failing Python project.
- DS4 must inspect repository state, fix the bug, run the tests, inspect the
  diff, checkpoint the context, restore it, and prove the tests still pass.
- If the native Git tool is available, the prompt asks DS4 to use it for
  `status` and `diff`.
- If the native Git tool is not available, the same test falls back to the
  existing `bash` path with `git status --short` and `git diff`.

Before:

```text
context tools work in isolated calls -> less proof of agent-level usefulness
```

Now:

```text
agent fixes a controlled project -> checkpoints -> restores -> verifies state
```

The limitation is intentional. This test does not claim that DS4 found and
optimized DS4's own C code. A stronger follow-up test should run against DS4
itself: ask the agent to inspect the repository, choose one small measurable
optimization, implement it, run the relevant benchmark or e2e check, inspect
the diff, checkpoint, restore, and record whether the metric improved.

That DS4-on-DS4 loop is the ideal product demonstration, but it is a slower and
less deterministic test than this PR should require by default. The controlled
temporary repository keeps this PR's regression signal clear while preserving a
direct path to the stronger self-optimization loop.

Verification:

- `make test-agent-context-self-improvement`
- The generated ledger records `git_status_mode`, `git_diff_mode`,
  `context_checkpoint_before`, `context_checkpoint_after`,
  `context_restore_used`, `tests_before_restore`, and `tests_after_restore`.

## Test Plan

Run:

```sh
make test
make test-agent-context-compact-canary
make test-kv-cache-benefit
make test-agent-context-self-improvement
git status --short
```

Expected result:

- default C tests pass,
- compaction canary e2e passes,
- context self-improvement e2e passes; it uses native Git tooling when that
  tool is present, and falls back to `bash`-run `git status` / `git diff` when
  this branch is tested without the Git-tool PR,
- KV benefit benchmark reports a large `saved_prefill_tokens` value,
- `git status --short` shows only intentional source changes before commit, and
  is clean after commit.
