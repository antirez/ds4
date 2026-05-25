# Agent Git Tools

This branch keeps Git support independent from agent context/KV checkpointing.
The native `git` DSML tool is wired through `ds4_agent_git.c` and does not link
against `ds4_agent_context.c`.

## Goals

- Provide fast local repository inspection without shell parsing.
- Keep all argv construction direct through `fork`/`exec`, never through a shell.
- Bound output with `max_bytes` and reject unsafe paths, refs, remotes, and
  messages before invoking Git.
- Allow guarded local mutations only when intent is explicit.
- Keep remote operations opt-in with `confirm=true`.
- Keep merge and rebase conservative: preview first, clean worktree required,
  and first real merge mode limited to `--ff-only`.

## Supported Actions

Read-only actions include `info`, `status`, `changed_files`, `diff`, `log`,
`show`, `ls_files`, `file_at_ref`, `blame`, `path_history`, `remote_list`,
`merge_base`, `merge_preview`, and `rebase_preview`.

Guarded actions include `stage`, `unstage`, `commit`, `worktree_restore`,
`switch`, stash operations, `fetch`, `push`, `merge`, `merge_abort`, `rebase`,
and `rebase_abort`.

## Non-Goals

- No `reset`.
- No `clean`.
- No force push.
- No branch/tag deletion.
- No non-fast-forward merge strategy in the first implementation.
- No dependency on context checkpoint or KV-cache internals.
