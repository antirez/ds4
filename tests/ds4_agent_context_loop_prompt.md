You are validating DS4's own agent context loop capability.

Do not explain the plan in prose. Use DSML tools to execute exactly this loop.

Repository root: __ROOT__
Ledger path: __LEDGER__

Loop:
1. Use the write tool to create __LEDGER__ with this exact starting content:

# DS4 Generated Context Loop
loop_limit=2
goal=validate DS4 agent context loop against DS4's context helper test
baseline=before-ds4-agent-context-test
ds4_prompt=validate DS4's own agent context loop capability
ds4_response=pending
attempt=1 hypothesis=the context helper test passes
attempt=1 command=cd __ROOT__ && ./tests/ds4_agent_context_test
attempt=1 status=pending
attempt=1 metric=pending

2. Use the bash tool to run exactly:
cd __ROOT__ && ./tests/ds4_agent_context_test

3. If the bash result reports success, use the edit tool to replace:
ds4_response=pending
attempt=1 status=pending
attempt=1 metric=pending

with:
ds4_response=LOOP_DONE
attempt=1 status=pass
attempt=1 metric=ds4_agent_context_test passed

If the bash result reports failure, replace the same status and metric lines
with status=fail and the observed failure summary, and replace
ds4_response=pending with the final response you will return.

4. If the attempt passed, use the context tool with action=checkpoint and label
ds4-generated-loop-after-pass. This context checkpoint call must be the only
tool call in its DSML block.

5. After the checkpoint tool result, answer exactly:
LOOP_DONE
