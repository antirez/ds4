You are validating whether DS4 context compaction preserves task-critical facts.

Do not explain the plan in prose. Use DSML tools.

Repository root: __ROOT__
Ledger path: __LEDGER__

The following five facts are task-critical canaries. They are the only source
of the values. Preserve them through compaction and use them after compaction:

canary_alpha=ORCHID-47
canary_beta=FJORD-932
canary_gamma=LEMMA-18
canary_delta=RUNE-604
canary_epsilon=VECTOR-251

The next block is intentionally irrelevant padding. Do not copy it to the
ledger. Its only purpose is to push the canary values out of the recent verbatim
tail, so the compacted summary must preserve them.

__PADDING__

Now execute exactly this sequence:

1. Use the context tool with action=compact and reason=canary-retention-test.
   This context compact call must be the only tool call in its DSML block.

2. After the compact tool result, use the write tool to create the ledger file
   at this exact path:

__LEDGER__

   The ledger must contain these exact field names and the exact canary values
   from above:

# DS4 Compact Canary Ledger
compaction=done
canary_alpha=<value from above>
canary_beta=<value from above>
canary_gamma=<value from above>
canary_delta=<value from above>
canary_epsilon=<value from above>
final=COMPACT_CANARY_DONE

3. After the write tool result, answer exactly:
COMPACT_CANARY_DONE
