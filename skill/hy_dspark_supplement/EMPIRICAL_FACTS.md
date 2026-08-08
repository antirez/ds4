# Empirical facts HY can rely on without GitHub access

These are prior community measurements and should be treated as experiment
context, not new results from this branch.

## M4 Max / Metal / q2-q4 / DSpark

One report on M4 Max measured approximately:

- plain: 29.31 tok/s
- DSpark: 23.02 tok/s
- acceptance: 70.45%
- replay: 1063.1 ms for 31 accepted tokens ~= 34.3 ms/token
- baseline decode: ~= 34.1 ms/token

Interpretation: accepted-token replay cost was essentially one ordinary
target decode per accepted token.

A second M4 Max datapoint used a different drafter and reached 90.7%
acceptance, but DSpark was still ~23.8% slower than baseline.  This is strong
evidence that low acceptance alone is not the main problem.

## M3 Ultra

Another report measured:

- saved: 3278.481 ms
- replay: 3336.601 ms
- replay / saved ~= 1.018

Again, accepted-token replay approximately cancels speculative saving.

## Historical replay-free branch (#590)

A replay-free prefix-checkpoint experiment showed that eliminating replay can
recover substantial throughput.  On one mixed Q2/Q4 M5 Max run with a lower
confidence threshold, it measured roughly +27%.

However, later greedy-identity analysis showed that directly committing the
generic batch-verifier frontier is not a safe strict-greedy solution in
general.  Therefore #590 is performance evidence and a checkpoint-design
reference, not a correctness-complete implementation.

## Historical exact verifier experiment (#677)

A broad byte-exact verifier prototype proved exact verification is possible,
but was substantially slower than plain target decode.  This argues against
making the entire verifier serial/exact.

Desired middle ground:

- keep batch execution where numerically safe
- exactify only the earliest stateful divergent operation(s)
- commit canonical verifier state directly
- eliminate accepted-token full replay
