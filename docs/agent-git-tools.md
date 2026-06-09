# Agent Git Tools

This branch keeps Git support independent from agent context/KV checkpointing.
The native `git` DSML tool is wired through `ds4_agent_git.c` and does not link
against `ds4_agent_context.c`.

## Goals

- Provide fast local repository inspection without shell parsing.
- Keep all argv construction direct through `fork`/`exec`, never through a shell.
- Bound output with `max_bytes` and reject unsafe paths, refs, remotes, and
  messages before invoking Git.
- Run Git non-interactively: stdin is `/dev/null`, pagers/prompts/editors are
  disabled, and every invocation has a bounded timeout.
- Allow guarded local mutations only when intent is explicit and validation
  happens before invoking Git.
- Require interactive user approval for every real Git mutation; `dry_run`
  calls and read-only inspection never prompt.
- Keep the riskiest local and remote mutations behind an additional
  model-visible `confirm=true` intent flag unless the call is a `dry_run`.
- Keep merge and rebase conservative: preview first, clean worktree required,
  and first real merge mode limited to `--ff-only`.

## Scope Status

The original MVP was read-only inspection. The implementation has deliberately
moved beyond that MVP into guarded local and remote operations because those
actions are needed for a useful branch-management loop.

This document is the authoritative scope for the Git branch. Older combined
planning text in the context/KV document is historical only after the split:
`feature/agent-git-tools` owns Git support, while
`feature/agent-kv-context-tools` owns context/KV checkpointing.

## Implemented Actions

The model-visible DSML schema enumerates the same action set so invalid action
names are rejected before the model has to infer command names from prose.

Read-only actions include `info`, `status`, `changed_files`, `diff`, `log`,
`show`, `ls_files`, `file_at_ref`, `blame`, `path_history`, `remote_list`,
`merge_base`, `merge_preview`, and `rebase_preview`.

Guarded local actions include `stage`, `unstage`, `commit`,
`worktree_restore`, `switch`, `stash_push`, `stash_apply`, `stash_pop`, and
`stash_drop`.

Guarded remote and integration actions include `fetch`, `push`, `merge`,
`merge_abort`, `rebase`, and `rebase_abort`.

## Guardrails

- Every Git subprocess defaults to `timeout_sec=30`; callers can request
  `timeout_sec` from 1 to 600 seconds. A timeout kills the Git process group and
  reports exit code 124 with a timeout notice in the tool output.
- Git runs with terminal prompts disabled (`GIT_TERMINAL_PROMPT=0`), askpass
  helpers disabled, merge auto-edit disabled, and `GIT_EDITOR=true`.
- Every real mutating action prompts the local user before invoking Git.
  In non-interactive mode, Git mutations are rejected.
- `stage` and `unstage` require either `path` or `all=true`.
- `commit` requires an explicit one-line `message`.
- `worktree_restore` requires either `path` or `all=true`, defaults `ref=HEAD`,
  supports `dry_run`, and requires `confirm=true` for a real restore.
- `switch` requires an explicit safe `ref` and `confirm=true` for a real branch
  switch; it does not create branches and does not use force, discard, or merge
  flags.
- `stash_push` requires an explicit one-line `message`; `all=true` includes
  untracked files.
- `stash_show`, `stash_apply`, `stash_pop`, and `stash_drop` accept only safe
  stash refs such as `stash@{0}`.
- Real `stash_pop` and `stash_drop` require `confirm=true`; `stash_apply`
  supports `dry_run` preview and leaves the stash entry intact.
- `fetch` requires an explicit `remote`; real fetch requires `confirm=true`.
  Optional fetch refs reject force and colon refspec syntax.
- `push` requires explicit `remote` and `ref`; real push requires
  `confirm=true`. Push rejects force, delete, and colon refspec syntax.
- `merge` requires an explicit target ref, `confirm=true` or `dry_run=true`,
  and a clean working tree. Real merge is `git merge --ff-only <target>`.
- `rebase` requires an explicit upstream ref, `confirm=true` or `dry_run=true`,
  and a clean working tree.
- `merge_abort` and `rebase_abort` require `confirm=true` or `dry_run=true`.
- Paths are passed after `--` where applicable.
- Refs, ranges, remotes, and push refs are rejected when they look like options
  or unsupported refspecs.

## Non-Goals

- No `reset`.
- No `clean`.
- No raw `checkout`; branch movement uses the guarded `switch` action only.
- No force push.
- No branch/tag deletion.
- No non-fast-forward merge strategy in the first implementation.
- No dependency on context checkpoint or KV-cache internals.
