You are validating DS4's own KV-backed self-improvement loop.

Do not explain the plan in prose. Use DSML tools.

Repository root: __REPO__
Ledger path: __LEDGER__

Task:
Fix the small Python project in the repository so its test suite passes. The
bug is intentionally simple and local to the repository.

Use absolute file paths under the repository root for read, edit, write, and
bash commands.

Repository inspection mode:

- If the available tool schemas include a native git DSML tool, prefer that tool
  for repository status and diff inspection.
- If the native git DSML tool is not available, use the bash tool to run git
  commands in the temporary repository.

Required sequence:

1. Use the context tool with action=checkpoint and label
   context-self-improvement-before. This context checkpoint call must be the
   only tool call in its DSML block.

2. Inspect repository status.

If native git DSML is available, use the git tool with action=status and repo
set to the repository root.

Otherwise, use the bash tool to run exactly:

cd __REPO__ && git status --short

3. Use read/edit/write/bash tools as needed to inspect, fix, and test the
   project. Run exactly this test command with the bash tool:

cd __REPO__ && python3 test_toy_math.py

4. Inspect the produced repository diff.

If native git DSML is available, use the git tool with action=diff, repo set to
the repository root, and path set to toy_math.py.

Otherwise, use the bash tool to run exactly:

cd __REPO__ && git diff -- toy_math.py

5. If the test passes, use the context tool with action=checkpoint and label
   context-self-improvement-after-pass. This context checkpoint call must be the
   only tool call in its DSML block. Save the returned checkpoint id for step 6.

6. Use the context tool with action=restore, id set to the checkpoint id from
   step 5, reason=context-self-improvement-restore-check, and
   allow_side_effect_mismatch=true. This context restore call must be the only
   tool call in its DSML block.

7. After restore, inspect repository status again.

If native git DSML is available, use the git tool with action=status and repo
set to the repository root.

Otherwise, use the bash tool to run exactly:

cd __REPO__ && git status --short

Then run exactly this test command again with the bash tool:

cd __REPO__ && python3 test_toy_math.py

After this restore, do not create any more context checkpoints and do not call
context restore again. Proceed directly to the ledger.

8. Use the write tool to create the ledger file at the ledger path. The ledger
   must contain these exact field names:

# DS4 Context Self Improvement Ledger
git_status_checked=yes
git_status_mode=<native or bash>
git_diff_checked=yes
git_diff_mode=<native or bash>
context_checkpoint_before=yes
context_checkpoint_after=yes
context_restore_used=yes
tests_before_restore=pass
tests_after_restore=pass
fixed_file=toy_math.py
final=CONTEXT_SELF_IMPROVEMENT_DONE

9. After the write tool result, answer exactly:
CONTEXT_SELF_IMPROVEMENT_DONE
