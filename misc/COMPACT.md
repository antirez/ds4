# DS4 Agent Context Compaction

The agent should compact context before long sessions become brittle, and also
when a tool result would otherwise not fit.

## Triggers

- **Soft trigger:** compact before a new user turn when the current transcript
  is at least 85% of the configured context, or when no more than 8192 tokens
  remain. For small contexts the free-token threshold is capped to one quarter
  of the context size, so test runs with tiny `--ctx` values do not compact
  after every short turn.
- **Hard trigger:** if a tool result cannot be appended with enough room left
  for the model to answer, compact immediately and retry the append once.

## Model

Compaction uses the live session once, before rewriting the prompt:

1. Remember the current transcript length as `BOTTOM`.
2. Ask the model, internally, to emit a durable task-state summary of the
   conversation up to `BOTTOM`.
3. Roll the real transcript back to `BOTTOM`; the summary request and response
   are not part of the user-visible session.
4. Take the recent tail verbatim by scanning backward from `BOTTOM` up to 10%
   of the configured context, capped at 50000 tokens. Align the tail to a
   rendered `<｜User｜>` boundary when possible.
5. Rebuild the transcript as:
   - system prompt and tool contract
   - compacted durable-state summary
   - recent verbatim tail
6. Re-sync the live KV state to the rebuilt transcript.

This keeps the part of the conversation that is most likely to contain active
work exactly as sampled, while summarizing older task state instead of merely
dropping it.

## Summary Contract

The summary should preserve durable task state only:

- user goals and constraints
- files inspected or edited
- commands run and important results
- decisions, rejected approaches, and known bugs
- current next steps

It should not invent facts or summarize raw file contents unless the assistant
already used those contents to reach a conclusion. Reloadable bulky data should
be described as such, with enough path/range information for the model to fetch
it again using tools.

The summary prompt explicitly tells the model to stop after the summary, and not
to continue the user task. This matters because the summarizer is invoked from a
live agent transcript where "next step" often means "call the next tool"; during
compaction, that next step must be recorded, not executed.

The compaction loop treats `</think>` and DSML as hard control boundaries. If
the model tries to close thinking or emit a tool call while summarizing, the
summary stops before that token. Internal compaction must never execute tools or
store DSML markup in the rebuilt system state.

## UI

During compaction the terminal should make the state explicit:

- show a high-intensity `COMPACTING` marker
- stream the generated summary so the user can see what is retained
- then show the normal prefill progress while the compacted context is rebuilt

## Validation Log

- With `--ctx 12000`, read `README.md` lines 1-500, then let the model call
  `more count=500`. The second tool result did not fit, so hard compaction ran,
  rebuilt the context from a summary plus a 1200-token verbatim tail, appended
  the delayed tool result, and the model continued with the second chunk
  available.
- Saved the same near-full state as a reusable session, then asked a follow-up
  question. Soft compaction ran before appending the new user turn, preserved
  that only lines 1-500 of `README.md` had been read, and the model answered the
  follow-up correctly after the compacted context was rebuilt.
