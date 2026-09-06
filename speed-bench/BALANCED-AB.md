# Balanced A/B measurement

`balanced_ab.sh` runs two `ds4-bench` arms against each other in a design that
removes a bias we measured on real hardware, and `analyze_paired.py` reports the
result with the naive figure printed alongside so the size of that bias stays
visible.

## The problem this solves

Throughput drifts downward inside a benchmarking session. On a GB10 we measured
it directly: over a 24-value sweep, **every one of the 24 values was lower than
the one before it** (p = 2^-24 under the null of no trend).

If the script runs arm A for a while and then arm B for a while, the arm that
runs last is systematically penalised. In our windows that moved the A/B ratio
by **0.38 to 0.55 percentage points** - the same order of magnitude as the
effects we were trying to report. The bias does not average out with more
repetitions, because it is not noise. It grows with session length.

## The design

1. **The two arms are adjacent in time.** No third arm, no other work in
   between, so both see nearly the same machine state.
2. **The order alternates.** Round 1 runs `A, B`; round 2 runs `B, A`. A
   monotone drift enters the two rounds with opposite signs and cancels in the
   mean.
3. **Ratios are formed within a round, never across rounds.** Dividing a median
   of all A by a median of all B re-admits the drift, because the two medians
   can come from different points in the session.
4. **Clock and temperature are recorded before and after every run,** so a
   thermal explanation can be checked instead of assumed.
5. **The binary and the prompt are checksummed into the log.** A benchmark
   result whose binary cannot be identified afterwards cannot be reproduced or
   contested.

Does it work? On a 5-block run built this way, the drift sign test comes back at
p = 0.29 (decode) and p = 0.91 (prefill) - the trend that was overwhelming under
a fixed order is gone.

## Running it

```sh
./balanced_ab.sh --bench ./ds4-bench \
                 --a  /models/model-Q4.gguf --a-label Q4 \
                 --b  /models/model-Q8.gguf --b-label Q8 \
                 --prompt /tmp/bench-prompt.txt \
                 --blocks 5 --out results
```

If the machine also serves a model, do not run the benchmark alongside it. Stop
it for the duration of one block and start it again afterwards:

```sh
  --before-cmd 'systemctl --user stop my-server' \
  --after-cmd  'systemctl --user start my-server'
```

Written that way, the server is down for one block at a time rather than for the
whole session.

## Reading the result

```sh
./analyze_paired.py results/paired.csv --metric gen_steady_tps
```

```
metric: gen_steady_tps   arms: Q4 / Q8
paired samples: 40   incomplete cells dropped: 0

     ctx    n    paired A/B              95% CI   naive A/B
--------------------------------------------------------------
    2048   10       +16.99%  [+16.53%, +17.46%]     +16.77%
    4096   10       +13.92%  [+13.62%, +14.22%]     +13.84%
    6144   10       +13.84%  [+13.52%, +14.15%]     +13.86%
    8192   10       +13.80%  [+13.45%, +14.15%]     +13.63%
--------------------------------------------------------------
     all   40       +14.64%  [+14.17%, +15.11%]     +13.98%

drift sign test: 41/72 consecutive runs slower than the previous one, p = 0.289
```

The `naive A/B` column is what an unpaired script computes. When it sits close
to the paired column, as it does above, the design is doing its job. When the
two diverge by something comparable to the effect being reported, the naive
statistic has not established that effect.

Report **per context point**. The `all` row mixes context points with different
ratios and is a summary, not a measurement.

## Two traps worth knowing before you compare anything

**`prefill_tokens` is the step increment, not the context.** At each point of a
sweep `ds4-bench` prefills only the tokens added since the previous point,
reusing the KV cache already built. It equals `ctx_tokens` at the first point
and nowhere else:

```
--ctx-start 2048 --ctx-max 8192 --step-incr 2048
    ctx 2048  -> prefill 2048        ctx 4096  -> prefill 2048
    ctx 6144  -> prefill 2048        ctx 8192  -> prefill 2048

--ctx-start 8192 --ctx-max 65536 --step-mul 2
    ctx 8192  -> prefill 8192        ctx 16384 -> prefill 8192
    ctx 32768 -> prefill 16384       ctx 65536 -> prefill 32768
```

This is the trap: **"prefill_tps at ctx 8192" is not one quantity.** In the
first sweep it is a 2048-token increment onto a warm cache; in the second it is
a cold 8192-token prefill. Two people can report that number for the same
context, the same model and the same machine, and be measuring different
operations.

It is not a small effect. Measuring the same models at ctx 8192 both ways:

| what is actually measured | Q4 vs Q8 |
|---|---|
| cold 8192-token prefill | **+0.4%** |
| 2048-token increment onto a warm cache | **+2.3%** |

Both numbers are correct. They answer different questions.

So: only compare runs with the same `--ctx-start` and the same step, and quote
`prefill_tokens` whenever you quote `prefill_tps`. The summary CSV records them
side by side, and `analyze_paired.py` refuses to average across a mismatch.

**Check that two prompt files really are two prompts.** We spent a window
chasing a difference we had attributed to prompt length, before checking:

```
md5(prompt.txt)                     = 3c2d6993...   135 000 B
md5(head -c 135000 prompt-long.txt) = 3c2d6993...
```

The long file was the short one repeated three times. Every sweep short enough
to fit inside the first copy was reading a byte-identical prefix. Checksum the
prefix your sweep actually consumes, not the whole file.
