# Agent KV Context Tools: Analysis And Implementation Plan

## Goal

Give `ds4-agent` a native tool for controlling its own context state without
exposing raw KV internals to the model.

The important distinction is that the agent should not read or write arbitrary
KV bytes. The useful feature is semantic control over checkpoints, restore
points, and context compaction. The tool should operate on transcript and
session checkpoints together, preserving the invariant that the visible
conversation and live `ds4_session` state describe the same timeline.

## Existing System Constraints

`ds4_session` is one mutable inference timeline. It owns the live KV cache and
logits, while callers provide full token prefixes to `ds4_session_sync()` so the
session can reuse, extend, or rebuild graph state.

`ds4-agent` already has user-facing slash commands for related operations:

- `/save` persists the current session under `~/.ds4/kvcache`.
- `/switch` loads a saved session and restores transcript plus KV payload.
- `/compact` asks the model for a durable summary and rebuilds the transcript.
- `/new` resets to the system/tool prompt.
- `/del` deletes a saved session.
- `/strip` removes a persisted KV payload while preserving rendered text.

Those commands are controlled by the user. The proposed feature gives the model
a narrower tool-level API so it can manage expensive context deliberately during
long autonomous work.

The relevant invariants are:

- A checkpoint is valid only if `ds4_session_tokens(w->session)` matches
  `w->transcript`.
- Restore must replace transcript and KV state as one operation.
- A context restore does not revert filesystem, process, network, or browser
  side effects.
- Active bash jobs are external state and must either block restore or be
  explicitly surfaced in the restored transcript.
- Compaction can temporarily put private compaction prompts into live KV; any
  failed compaction must invalidate live session state before continuing.
- For server/API usage, exact DSML replay must remain byte-for-byte compatible
  with the rendered history. For `ds4-agent`, sampled DSML is already preserved
  directly in the transcript, but the same principle applies: never rewrite a
  tool-call turn into a semantically similar but token-different form.

## Proposed Tool Surface

Use one DSML tool named `context` with an `action` parameter instead of many
separate tool names. This keeps the system prompt smaller and makes future
actions easier to add without teaching the model a large new catalog.

```json
{
  "type": "function",
  "function": {
    "name": "context",
    "description": "Inspect, checkpoint, restore, or compact the agent context.",
    "parameters": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": ["status", "checkpoint", "list", "restore", "compact", "drop"]
        },
        "id": {"type": "string"},
        "label": {"type": "string"},
        "reason": {"type": "string"},
        "allow_side_effect_mismatch": {"type": "boolean"},
        "dry_run": {"type": "boolean"}
      },
      "required": ["action"]
    }
  }
}
```

Initial actions:

- `status`: report transcript length, session position, context size, free
  tokens, dirty session state, side-effect epoch, active bash jobs, and known
  checkpoints.
- `checkpoint`: save a named restore point at the current stable transcript.
- `list`: list known checkpoints.
- `restore`: restore a checkpoint if side-effect rules allow it.
- `compact`: request the existing compaction path with an explicit reason.
- `drop`: delete checkpoint metadata and its associated payload when safe.
  In the first implementation, "safe" means the checkpoint id resolves
  unambiguously, paths remain inside the context directory, and no bash job is
  running. The tool does not understand semantic roles such as "best baseline";
  callers should use `dry_run=true` before deleting important checkpoints.

Phase 1 should be disk-backed and reuse the existing agent KV save/load path.
That avoids holding multiple huge KV payloads in RAM and keeps the first
experiment close to the existing `/save` and `/switch` implementation.

## Concrete Use Cases

1. Deep codebase exploration checkpoint.

An agent reads architecture files, traces call graphs, and builds a high-value
mental model. Before trying an implementation, it calls
`context action=checkpoint label="repo-map-before-fix"`. If the first
implementation path fails, restore avoids re-prefilling the whole exploration
history.

2. Alternative patch strategies.

Before changing a shared subsystem, the agent creates a checkpoint, implements
approach A, runs tests, then restores and tries approach B. This is useful when
both alternatives require long reasoning from the same inspected context.
Filesystem changes still need explicit version-control or file rollback, so
restore must warn when side effects happened after the checkpoint.

3. Compaction quality recovery.

The agent checkpoints before forced compaction. If the compacted summary loses
critical details, the agent can restore the pre-compaction checkpoint and retry
with a better compaction reason, smaller tool output, or a manual summary.

4. Long web research reuse.

The agent searches and visits several pages, creating large rendered Markdown
observations. A checkpoint lets it try different conclusions or implementation
plans without paying the same browser and prefill cost again.

5. Risky tool-call loop guard.

Before a sequence of generated `edit` and `bash` calls, the agent checkpoints.
If it starts following a wrong path, the user or model can restore the reasoning
state while separately deciding whether to keep, revert, or inspect filesystem
effects.

6. Parser and prompt experiments.

Developers working on DSML parsing, forced syntax, or tool visualization can
restart from the same prompt frontier and compare generated tool calls under
different prompt wording or sampling knobs.

7. Large session navigation.

The agent can preserve named frontiers such as "after reading tests",
"after reproducing bug", and "before final refactor". This gives long local
sessions an internal navigation model instead of relying only on `/save` and
manual `/switch`.

8. Bounded experimental loops.

The agent can run a disciplined optimization loop from one baseline context:
checkpoint the baseline, write an experiment ledger, propose a hypothesis,
materialize it in code, measure it, and either save the improved state or record
the failed attempt and restore to the baseline. The ledger survives restore, so
failed attempts do not need to remain in the model transcript to remain useful.
This is especially useful for prompt, parser, quality, and performance
experiments where many attempts share the same expensive codebase understanding.

Example flow:

```text
context checkpoint label=baseline-before-tool-parser-loop
write experiment.md with goal, metric, max_attempts, current_attempt=0
attempt 1: record hypothesis in experiment.md, edit code, run tests
if tests improve: record success, checkpoint label=best-attempt-1
if tests regress: record failure in experiment.md, restore baseline with reason
restore notice tells the model to reread experiment.md before attempt 2
stop when metric passes or current_attempt reaches max_attempts
```

## Safety Model

The implementation should add a monotonically increasing `world_epoch` owned by
the agent worker. Increment it for successful operations that may change
external state:

- `write`
- `edit`
- `bash`
- `bash_stop`
- future filesystem mutation tools

Read-only tools such as `read`, `search`, `list`, `google_search`, and
`visit_page` do not increment `world_epoch`.

Every context checkpoint stores the current `world_epoch`. A restore where the
current epoch differs from the checkpoint epoch should fail by default with a
clear message:

```text
Tool error: restore would rewind model context from world_epoch=7 to 4, but
external side effects may still exist. Revert or inspect those effects, or call
context restore with allow_side_effect_mismatch=true.
```

Even with `allow_side_effect_mismatch=true`, the tool result must say that only
model context was restored. It must not claim that files, commands, browser
state, or network side effects were reverted.

On agent startup, initialize `world_epoch` from the maximum epoch found in
existing checkpoint metadata. That keeps persisted checkpoints usable after a
restart while ensuring new side effects in the current process advance beyond
the restored baseline.

Restores should also fail while a bash job is running. A running process is a
live external dependency whose output may still arrive after the restored
transcript.

### Restore Notice

A model-initiated restore must not be silent. A raw restore of transcript plus
KV would move the model back to the checkpoint and erase the very reason it
decided to restore. The default tool behavior should therefore be:

```text
load checkpoint transcript + KV
append synthetic restore notice
continue from restored transcript plus notice
```

The restore notice becomes the first event after the restored checkpoint. It
should be inserted as a tool result or equivalent user-visible control message
after the restored transcript has been loaded. It must include:

- checkpoint id and label,
- restore reason supplied by the model or user,
- restored transcript token count,
- checkpoint `world_epoch` and current `world_epoch`,
- whether side-effect mismatch was allowed,
- a warning that files, subprocesses, browser state, network effects, and other
  external state were not reverted,
- a compact summary of known post-checkpoint side effects when available,
- an explicit warning when the in-memory side-effect history has been truncated
  and older post-checkpoint side effects may have been dropped.

Example:

```text
Context restored from checkpoint 7e1c2b1a label=after-repo-map.
Reason: approach A failed because parser regression test X still failed.
Restored model context to 18420 tokens. world_epoch restored=3 current=7.
External side effects were not reverted; inspect or revert files/processes
separately before assuming the workspace matches this checkpoint.
```

This means restore creates a coherent continuation, not a perfect time machine.
The agent retains the expensive pre-checkpoint context and receives a short
explanation of why the failed attempt was discarded.

## Critical Assessment: Agent And Server

The opportunity is real, but it is not the same feature in `ds4-agent` and
`ds4-server`.

In `ds4-agent`, the process has one user, one live worker, one transcript, and
one obvious owner of side effects. A context tool can be powerful because it
lets the model preserve expensive frontiers, checkpoint before risky work,
recover from weak compaction, and write durable notes for later reuse.

In `ds4-server`, the same surface becomes harder for two separate reasons. First,
API requests are stateless, may come from multiple clients, and are serialized
through one live backend session. Second, the server currently returns tool calls
to clients; it does not run native server-side tools in the way `ds4-agent` runs
`read`, `edit`, `bash`, or `context`. The server can reuse KV prefixes safely,
but a model-generated `restore` would be a mutation of the single global live
timeline. Without an explicit session owner, a checkpoint is just a global object
in a shared cache.

The feature therefore has two layers:

- Computational continuity: checkpoint/restore of transcript plus KV.
- Semantic continuity: structured memory files that record what the agent
  learned.

The first layer saves prefill. The second saves reasoning. Both are needed for
the tool to be genuinely useful.

### Opportunities

- Long local coding sessions can avoid repeated high-cost prefill after a repo
  exploration or web research phase.
- The agent can create named frontiers before risky edits, prompt experiments,
  tool loops, or compaction.
- Structured memory can preserve architecture facts, invariants, decisions, and
  open questions even after compaction or restart.

### Difficulties

- A KV checkpoint is not semantic memory. It preserves state, but not a compact
  object the model can inspect cheaply.
- Restore does not revert the world. Files, subprocesses, browser state,
  network effects, and external APIs remain changed.
- The current server has no authenticated tenant, owner, or session namespace.
  Adding one is a prerequisite for writeable multi-user context controls.
- Stateless clients may resend a history that disagrees with a server-side
  restore. The server must prefer explicit session-control semantics over
  implicit tool behavior.
- Exact DSML replay remains fragile if checkpoint movement loses the sampled
  tool-call bytes or maps them to the wrong request/session.
- Future concurrent or multi-slot serving can race on checkpoint metadata and
  memory files unless writes are serialized per namespace.

## Structured Memory Storage

Do not store structured memory inside the KV payload. Store it next to the
checkpoint as a separate, readable artifact:

```text
~/.ds4/kvcache/context/
  <checkpoint-id>.kv
  <checkpoint-id>.meta.json
  <checkpoint-id>.memory.md
```

`<checkpoint-id>.kv` stores transcript plus DS4 session payload.

`<checkpoint-id>.meta.json` stores machine-readable metadata:

```json
{
  "id": "7e1c2b1a...",
  "label": "repo-map-before-fix",
  "created_at": 1780000000,
  "world_epoch": 3,
  "transcript_tokens": 18420,
  "kv_path": "7e1c2b1a.kv",
  "memory_path": "7e1c2b1a.memory.md",
  "memory_sha1": "..."
}
```

`<checkpoint-id>.memory.md` stores model-readable semantic memory:

```md
# Context Memory

## Goal
## Files Inspected
## Architecture Facts
## Invariants
## Decisions
## Commands And Results
## Risks
## Open Questions
## Next Steps
```

This separation matters because memory can be regenerated, diffed, inspected,
loaded selectively, or retained after a KV payload is stripped.

For the first experiment, memory files should be created by the normal file
tools or by the existing compaction-style paths. They are useful artifacts, but
they are not required for checkpoint and restore to work.

## Experiment Ledgers

Long autonomous improvement loops need a durable record that is not rewound by
context restore. Store that record as a Markdown ledger outside the checkpoint
payload, either as a memory artifact associated with a checkpoint or as a named
experiment file referenced by checkpoint metadata.

The ledger should be append-oriented and machine-readable enough for the agent
to enforce its own budget:

```md
# Experiment Loop

## Goal
Reduce DSML tool-call failures without regressing server behavior.

## Success Metric
- `./ds4_test --tool-call-quality` improves or stays stable
- `./ds4_test --server` has no regressions

## Budget
max_attempts: 5
current_attempt: 2

## Baseline
checkpoint: 7e1c2b1a
score: ...

## Attempts

### Attempt 1
Prompt: ...
Hypothesis: ...
Patch: ...
Tests: ...
DS4 response: ...
Result: failed
Reason: parser regression X
Decision: discard

### Attempt 2
Hypothesis: ...
Status: in_progress
```

For DS4-generated loop tests, keep both levels of evidence:

- a compact model-written ledger with `ds4_prompt` and `ds4_response` fields;
- a harness-written report that preserves the exact prompt sent to DS4, the raw
  DS4 output, and the generated ledger.

The budget is not just a suggestion in prose. For the first experiment, the
agent should reread the ledger after each restore and stop when
`current_attempt >= max_attempts`, when the success metric is met, or when
restore safety checks fail. A later loop controller can enforce the same rule
programmatically. After a failed attempt, the expected flow is:

```text
append failure result to ledger
restore baseline checkpoint
append restore notice that points to the ledger
start the next hypothesis
```

After a successful attempt, the agent should update the ledger, save a new
checkpoint, and mark it as the new best state. This turns context checkpoints
into clean restart points and the Markdown ledger into the durable memory of the
search process.

## Verified Server Session Model

The current `ds4-server` does not have an explicit remote session or owner
concept. This was verified against the server implementation:

- `server` owns one `ds4_session *session`, one disk KV cache handle, one tool
  memory map, and a small set of live continuation bindings.
- HTTP client threads parse requests and enqueue stack-owned jobs. A single
  `worker_main()` dequeues jobs and mutates the one live session.
- `http_request` stores only method, path, body, and body length. Header parsing
  reads `Content-Length`; it does not keep `Authorization`, API key, tenant,
  session, organization, or user headers.
- `/v1/chat/completions`, `/v1/messages`, and `/v1/completions` parse protocol
  payload fields into rendered prompts and skip unknown JSON fields. OpenAI
  `user`, metadata, or similar caller fields are not retained as identity.
- `/v1/responses` explicitly rejects non-null `previous_response_id` and
  `conversation` because DS4 does not implement the durable Responses store.
- Disk KV cache lookup is keyed by rendered byte prefix plus compatibility
  checks such as quantization and context size. It is not keyed by user, owner,
  tenant, or application session.
- The live Responses, Anthropic, and thinking continuation structures bind
  recent tool call ids or visible transcript bytes to the current live token
  frontier. They are process-local accelerators, not durable session ownership.

Therefore, the server currently has stateless API semantics with one mutable
worker-owned timeline. The right server default is prefix-cache reuse, not
server-side conversation ownership.

## MVP Boundary

The first experiment is agent-only. `ds4-agent` is the only current runtime with
all required semantics in one place:

- one transcript owner,
- one worker-owned `ds4_session`,
- slash-command save/switch/compact precedents,
- side-effect visibility for `edit`, `write`, `bash`, and browser tools,
- active bash job tracking,
- a natural place to report restore warnings to the user.

The implementation should stay close to the existing local KV save/load path.
It may factor small helper functions for metadata, atomic writes, and
compatibility checks, but a general storage abstraction is not required before
the first working tool.

The first implementation should add only the state needed by the agent worker:
checkpoint metadata, a `world_epoch` counter, and enough recent side-effect
summary text to make restore notices useful. It should not introduce a new
global context subsystem before the native tool proves useful.

For `ds4-server`, the verified model above is enough guidance for the MVP: keep
automatic prefix-cache reuse as the server behavior, and do not add
model-visible restore semantics to stateless API traffic.

## Branch Boundary

This branch intentionally owns only agent context and KV checkpoint support.
Native Git support lives in `feature/agent-git-tools` and can be merged through
`feature/agent-kv-git-integration` when both feature lines need to work
together.

The context branch must not include or link `ds4_agent_git.*`. When an
integration branch combines both features, mutating Git actions should be
recorded as ordinary side effects in `world_epoch`, just like `write`, `edit`,
and `bash`.

## Implementation Plan

### Phase 1: Disk-backed context checkpoints

Add worker-owned checkpoint state and disk metadata:

```c
typedef struct agent_context_checkpoint {
    char id[41];
    char *label;
    char *path;
    uint64_t created_at;
    uint64_t world_epoch;
    int transcript_tokens;
    struct agent_context_checkpoint *next;
} agent_context_checkpoint;
```

Store checkpoint files below:

```text
~/.ds4/kvcache/context/<id>.kv
~/.ds4/kvcache/context/<id>.meta.json
```

Use existing save/load helpers where possible:

- Save with `agent_kv_save_path()`.
- Load with `agent_kv_load_path()`.
- Reuse `agent_worker_sync_tokens()` for stripped or text-only rebuild paths.
- Keep the worker thread as the only owner of `w->session` mutation.

`id` should be generated independently from the display label, for example from
random bytes plus checkpoint metadata. The label is user/model-facing display
text, not the stable identity.

### Phase 2: Tool dispatch

Add `context` to the tool schema prompt and dispatch in
`agent_execute_tool_call()`.

The handler should parse:

- `action`
- `id`
- `label`
- `reason`
- `allow_side_effect_mismatch`
- `dry_run`

The action handler should return compact machine-readable text. Example:

```text
context action=checkpoint id=7e1c2b1a label=before-parser-refactor tokens=18420 world_epoch=3
context action=compact status=ok old_tokens=28500 new_tokens=9400 removed_tokens=19100 reduction_percent=67.0 summary_tokens=2100 tail_tokens=7000
```

Restore appends a model-visible notice that includes KV reuse accounting:

```text
KV restore expected metrics: checkpoint_tokens=18420 expected_restore_notice_tokens=140 expected_restored_tokens=18560 expected_prefill_suffix_tokens=140 expected_full_prefill_tokens_without_kv=18560 expected_saved_prefill_tokens=18420.
```

This makes the benefit concrete for both the implementation and the model:
restoring the checkpoint loads the old prefix from KV, then only the synthetic
restore notice is expected to be prefetched. The word `expected` is intentional:
the notice is built before the final sync that appends it, while trace output
records the actual cached/suffix counts observed by `ds4_session_sync()`.

## Correctness Verification Measures

1. Transcript and session equality.

After every checkpoint and restore:

```text
agent_tokens_equal(ds4_session_tokens(w->session), &w->transcript) == true
ds4_session_pos(w->session) == w->transcript.len
```

2. Prefix reuse measurement.

For a restore from a disk KV payload, the next sync to the same transcript
should report zero prefill suffix. For stripped checkpoints, the suffix may be
non-zero, and the tool result must say it rebuilt from rendered text.

For model-initiated restores, the actual post-restore transcript should be the
checkpoint transcript plus the synthetic restore notice. Verification should
measure both values separately: zero prefill for loading the checkpoint payload,
then a small append for the notice.

If payload tokens and metadata tokens disagree after a KV load, restore must not
leave the live session at the loaded payload while the transcript still points
to the previous conversation. It should resynchronize the live session to the
current transcript or invalidate the session before returning the error.

The `context status` output should expose the live-cache view as
`cached_tokens` and `prefill_suffix_tokens`, so the agent can tell whether the
current transcript will reuse KV or force a rebuild.

3. Next-token equivalence.

Before checkpoint, copy logits with `ds4_session_copy_logits()`. After restore,
copy logits again and compare:

- exact token position equality,
- same argmax token,
- top-k ids match,
- float deltas are zero or within a backend-specific tolerance.

4. Side-effect epoch enforcement.

Create a checkpoint, run `edit` or `bash`, then attempt restore. Expected:

- restore fails without `allow_side_effect_mismatch=true`,
- restore succeeds with the override,
- restore notice explicitly warns that external effects were not reverted and
  names the epoch mismatch.

5. Active bash job guard.

Start a long-running bash job, checkpoint or restore depending on policy, and
verify that restore is denied while the job is running. After `bash_stop`, the
same restore should follow normal side-effect rules.

6. Compaction interaction.

Checkpoint before compaction, compact, then restore. Expected:

- transcript returns to the checkpoint token count,
- model-initiated restore appends a restore notice after that checkpoint,
- private compaction prompt text is absent,
- live session is synchronized to restored transcript,
- no stale compaction summary remains unless it was part of the checkpoint.

7. Corrupt or incompatible checkpoint handling.

Corrupt a checkpoint file or change quant/context metadata. Expected:

- restore fails,
- live session is invalidated only if load already touched it,
- transcript is not replaced with partial data,
- error text identifies the reason.

8. Persistence across restart.

Save a context checkpoint, exit `ds4-agent`, restart, list checkpoints, restore
the checkpoint, and verify token count plus next-token equivalence where the
same model/backend are available.

9. DS4-generated experiment loop.

Run the slow e2e target:

```sh
make test-agent-context-loop
```

This test is intentionally not part of default `make test`: it requires a real
model, a usable backend, and enough time for a short agent turn. The prompt in
`tests/ds4_agent_context_loop_prompt.md` requires DS4 itself to:

- create an experiment ledger with `write`,
- record the compact prompt and final DS4 response in that ledger,
- measure a DS4-owned helper test with `bash`,
- update the ledger with `edit`,
- create a model-visible `context checkpoint`,
- finish with `LOOP_DONE`.

The shell harness verifies the generated ledger, the prompt/response report,
and the checkpoint metadata. It also writes
`tests/generated/ds4_agent_context_loop_report.md` plus separate persisted
prompt, response, and ledger files with the exact expanded prompt, the raw DS4
output, and the generated ledger. It does not synthesize the loop in C; the
point is to test whether the model can operate the new tool surface in the
intended loop shape.

10. KV cache benefit benchmark.

Run the optional benchmark target:

```sh
make test-kv-cache-benefit
```

This target is intentionally separate from default `make test` because it opens
the real model and backend. It verifies:

- a saved KV payload reloads to the same token position,
- restored logits have the same argmax and near-zero delta versus the original
  checkpoint state,
- extending the restored session requires prefill only for the suffix,
- a fresh full prefill to the same extended transcript has the same top-1 next
  token as KV-restore-plus-suffix,
- the report prints `full_prefill_tokens`, `restored_prefill_tokens`,
  `saved_prefill_tokens`, payload bytes, and wall-clock timings.

The hallucination claim should be phrased conservatively: the deterministic
guard is model-state equivalence. If logits/argmax match after restore, the KV
path has not introduced state drift. Compaction can reduce context pressure,
but factual quality after compaction still depends on the summary and must be
tested with task-specific e2e prompts.

11. Compaction canary retention e2e.

Run the optional compaction-quality target:

```sh
make test-agent-context-compact-canary
```

This target is intentionally separate from default `make test`: it asks DS4 to
operate the `context compact` tool, places five canary facts before a long
irrelevant padding block, and then requires DS4 to write the canaries into a
ledger only after compaction. The harness verifies:

- the trace contains `compacted reason="canary-retention-test"`,
- the compaction trace reports a reduced token count and a late enough recent
  tail start,
- the post-compaction ledger exists,
- all five canary values survived,
- the final response marker is present.

This is still not a general hallucination benchmark. It is a focused task-level
guard that checks whether compaction preserves facts explicitly marked as
critical for the next action while those facts are pushed out of the recent
verbatim tail.

### Resume Point: 2026-05-25

The DS4-generated context loop was run successfully with:

```sh
make test-agent-context-loop
```

The first sandboxed attempt failed because the sandbox could not access Metal.
The successful run was executed outside the sandbox and produced:

- `tests/generated/ds4_agent_context_loop_report.md`
- `tests/generated/ds4_agent_context_loop_prompt.md`
- `tests/generated/ds4_agent_context_loop_output.txt`
- `tests/generated/ds4_agent_context_loop_ledger.md`

The generated ledger recorded:

```text
ds4_prompt=validate DS4's own agent context loop capability
ds4_response=LOOP_DONE
attempt=1 status=pass
attempt=1 metric=ds4_agent_context_test passed
```

Useful result: the loop proved that DS4 can operate the intended tool sequence:

```text
write -> bash -> edit -> context checkpoint -> final response
```

It also proved that the harness now captures the full evidence chain: expanded
prompt, raw model response, generated ledger, and checkpoint metadata.

Observed weakness: the prompt explicitly said `Do not explain the plan in
prose`, but DS4 still emitted conversational text such as:

```text
I'll execute the loop step by step.
The test succeeded (exit_status=0). Now I'll edit the file to mark success.
The attempt passed. Now I'll checkpoint the context.
```

This did not break the current harness because the final ledger and checkpoint
were correct, but it gives the next self-improvement loop a concrete target:
improve DS4's adherence to tool-only execution when the prompt requests no
prose.

Next loop to run from here:

1. Inspect baseline state with external version-control commands or with the
   independent Git branch after integration.
2. Ask DS4 to propose a small DS4-owned improvement for tool-only adherence.
3. Materialize the hypothesis in a Markdown experiment ledger.
4. Implement one minimal change.
5. Measure with a focused e2e check that fails when raw DS4 output contains
   unexpected prose before/between required tool calls.
6. If the metric improves, checkpoint and record the source diff in the ledger.
7. If it does not improve, record the failure and restore/retry from the saved
   context frontier.

## Exploration And Implementation Loop

Use this loop for each action before merging implementation:

1. Define a concrete agent scenario.

Write the starting transcript shape, tool calls involved, expected checkpoint
state, and external side effects.

2. Run the scenario against the current implementation.

Capture transcript token count, session position, world epoch, active bash job
state, and checkpoint id.

3. Assert invariants.

Check token equality, session position, side-effect policy, and tool result
clarity.

4. Measure cost.

Record save latency, restore latency, prefill suffix tokens, checkpoint payload
bytes, and whether restore avoided a cold rebuild.

5. Break it intentionally.

Try stale ids, corrupt files, active jobs, side-effect mismatch, stripped
payloads, and interrupted compaction.

6. Tighten the implementation.

Add the missing guard, simplify the API, or improve the tool result before
moving to the next action.

## First Loop Batch

The first implementation pass should cover these scenarios in order:

| Scenario | Purpose | Expected result |
| --- | --- | --- |
| `status` on fresh sysprompt | establish baseline | reports ctx, pos, transcript tokens, no checkpoints |
| `checkpoint` after one user turn | prove save path | checkpoint id returned, token/session equality holds |
| `restore` with no side effects | prove load path | checkpoint is loaded with zero prefill suffix, then restore notice is appended |
| `restore` after `edit` | prove guard | denied unless override is set |
| `compact` then `restore` | prove compaction safety | restored state has no leaked private summary |
| running `bash` then `restore` | prove live process guard | restore denied until job is stopped |
| failed-attempt retry after restore | prove model usability | model uses restore notice to abandon the failed attempt and try a different strategy |
| DS4-generated loop e2e | prove model tool use | DS4 writes a ledger, records prompt/response, runs a DS4 helper test, records pass/fail, checkpoints passing state |

## Open Design Decisions

- Whether `checkpoint` should be allowed while the session is dirty but idle.
  The likely answer is yes, after forcing `agent_worker_sync_tokens()`.
- Whether model-initiated `restore` should require user confirmation in
  interactive mode. For Phase 1, deny side-effect mismatch by default and do not
  prompt from inside the tool.
- Whether to expose an explicit `hard_restore` action for tests and manual
  debugging. The default model-visible `restore` should append a restore notice;
  hard restore should not be the autonomous path.
- Whether these controls should also be exposed through slash commands. The
  initial implementation can keep `/save` and `/switch` unchanged and expose
  only the DSML `context` tool.
- How experiment loops coordinate context restore with workspace rollback. The
  MVP can warn through `world_epoch` and require explicit cleanup before
  override.

## Non-goals

- No arbitrary KV byte editing.
- No filesystem rollback.
- No promise that browser state or network side effects are restored.
- No multiple live KV sessions in RAM in Phase 1.
- No prompt rewriting that changes sampled DSML history.
