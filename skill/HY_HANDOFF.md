# HY handoff: DSpark work before the support GGUF finishes downloading

## What you can implement now

Add `ds4_exactness_probe.[ch]` to the repository and wire it into the build only if convenient.
The helper is deliberately model-independent.  It compares GPU float tensors in bounded chunks
and reports:

- first differing element
- first actual / expected value
- total mismatch count
- max absolute difference
- max relative difference
- max ULP difference
- strict bit-exact status

Enable future instrumentation with:

```sh
export DS4_DSPARK_EXACTNESS_PROBE=1
```

Do not enable any behavioral change yet.

## Static code audit to do now

On current `main`, locate and record the current names/line numbers for:

1. `ds4_session_eval_dspark_speculative_argmax`
2. the DSpark batched target verifier
3. ordinary single-token target decode
4. batched compressor KV/score projection
5. ordinary single-token compressor projection/store
6. compressor mutable frontier update
7. frontier snapshot/restore
8. accepted-token replay loop
9. existing prefix-1 checkpoint/capture helpers

Produce a short call graph.

## What the historical PRs established

### PR #590 — replay-free prefix checkpoints

Useful idea:
- capture compressor frontier after each verifier row
- partial accept of k tokens commits checkpoint slot k-1
- this removes one full target replay per accepted token
- it also introduced an inline compute-copy helper to avoid Metal encoder churn

Do NOT port #590 wholesale.

Why: later correctness work showed that the verifier's batched projection numerics can differ
from ordinary single-token decode.  A checkpoint is only safe to commit if the frontier stored
inside it is canonical.

### PR #659 / current correctness direction

This removed the old direct full-accept fast path and forced accepted tokens through ordinary
single-token decode.  The important diagnosis was:

- batched verifier projection and ordinary decode projection are mathematically equivalent
- but their floating-point accumulation/tiling order differs
- compressor frontier therefore drifts
- drift can eventually flip a greedy near-tie

This is why current replay is expensive but correct.

### PR #677 — exact verifier prototype

It demonstrated that byte-identical verifier execution is possible, but making too much of the
verifier decode-order exact was slower than baseline.  Lesson: do not serialize/rewrite the whole
verifier.  Find the first divergent stateful operator and exactify only what is necessary.

## The experiment once the GGUF arrives

Do not start with performance.

From the same pre-speculation snapshot, compare:

A. normal DSpark batched verifier
B. ordinary sequential decode of the same draft tokens

Instrument the earliest state checkpoints possible.

Recommended order per layer/row:

1. compressor projected KV
2. compressor projected score
3. compressor mutable KV frontier
4. compressor mutable score frontier
5. indexer equivalents when present
6. emitted compressed row
7. layer output hidden state

Stop at the first mismatch.

The key question is not "are final logits close?" but:

> Which tensor/operator is the first place the DSpark verifier leaves the ordinary-decode
> numerical trajectory?

Only if the compressor projection/frontier is confirmed as first divergence should you implement
an exact-row Metal projection kernel.

## Exact-row kernel contract

The candidate API may process rows 1..5 in one dispatch, but each row must preserve the same
within-row arithmetic order as the ordinary single-token kernel.

The test contract is strict:

```text
candidate(rows=N) == N repeated canonical single-row calls
```

byte-for-byte for N=1..5.

Do not accept tolerance-based equality for the state that will be committed.

## What not to do yet

- no scheduler redesign
- no confidence tuning
- no sampling / temp > 0
- no SSD streaming
- no ROCm/CUDA work
- no full exact verifier rewrite
- no direct port of #590's replay-free commit until frontier exactness is proven

## Files in this bundle

- `ds4_exactness_probe.h`
- `ds4_exactness_probe.c`
- `test_exact_rows_contract_template.c`
- this handoff

These files are preparation, not a finished DSpark fix.
