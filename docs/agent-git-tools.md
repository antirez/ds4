# Agent Git Tools

This branch keeps Git support independent from agent context/KV checkpointing.
The native `git` DSML tool is wired through `ds4_agent_git.c` and does not link
against `ds4_agent_context.c`.

## Goals

- Provide fast local repository inspection without shell parsing.
- Keep all argv construction direct through `fork`/`exec`, never through a shell.
- Bound output with `max_bytes` and reject unsafe paths, refs, remotes, and
  messages before invoking Git.
- Allow guarded local mutations only when intent is explicit and validation
  happens before invoking Git.
- Keep remote mutations opt-in with `confirm=true` unless the call is a
  `dry_run`.
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

Read-only actions include `info`, `status`, `changed_files`, `diff`, `log`,
`show`, `ls_files`, `file_at_ref`, `blame`, `path_history`, `remote_list`,
`merge_base`, `merge_preview`, and `rebase_preview`.

Guarded local actions include `stage`, `unstage`, `commit`,
`worktree_restore`, `switch`, `stash_push`, `stash_apply`, `stash_pop`, and
`stash_drop`.

Guarded remote and integration actions include `fetch`, `push`, `merge`,
`merge_abort`, `rebase`, and `rebase_abort`.

## Guardrails

- `stage` and `unstage` require either `path` or `all=true`.
- `commit` requires an explicit one-line `message`.
- `worktree_restore` requires either `path` or `all=true`, defaults `ref=HEAD`,
  and supports `dry_run`.
- `switch` requires an explicit safe `ref`; it does not create branches and
  does not use force, discard, or merge flags.
- `stash_push` requires an explicit one-line `message`; `all=true` includes
  untracked files.
- `stash_show`, `stash_apply`, `stash_pop`, and `stash_drop` accept only safe
  stash refs such as `stash@{0}`.
- `fetch` requires an explicit `remote`; real fetch requires `confirm=true`.
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
