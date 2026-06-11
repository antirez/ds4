# Teaching-prompt experiments

## baseline (2026-06-10 21:24)
- teach sha: 8eb831fb0183
- run: 20260610-212422 -> composite 74.75, dims i3.58/c4.17/e3.17/g3.92/ec3.83,
  pass 8/8, emit 8/8
- notes: worst tests strcpy-overflow (63, only 1 aside, judge says run "mostly
  lacks timely teaching moments") and rename-symbol (64, factual-error flag,
  aside 2 repeats aside 1). Recurring patterns: (a) second asides restate the
  first in more words (rename-symbol, quadratic-dedupe — the two worst-scored
  asides in the run), (b) engagement is the weakest dimension (3.17), judges
  repeatedly call asides "flat"/"muted" and ask for a quick prediction hook.

## iter1 — no-reteach-rule (2026-06-10)
- teach sha (prev best): 8eb831fb0183 @ 74.75
- hypothesis: the two worst asides in the baseline are second asides that
  restate the first ("repetitive with [1]", "mostly repeats [1] with more
  words"). Adding an explicit anti-redundancy rule — each aside must bring a
  genuinely new angle or not exist — should lift insight/engagement on
  rename-symbol and quadratic-dedupe without hurting anything else.
- change: added one bullet to the aside rules: never re-teach a point already
  made; a new aside needs a new angle (measurement, edge case, failure mode,
  war story) or stays unwritten.
- run: iter1-no-reteach -> composite 74.0, dims i3.62/c4.00/e2.88/g3.88/ec3.38,
  pass 6/7 checked, emit 7/8 (rename-symbol TIMEOUT at 420s vs 78s baseline;
  strcpy-overflow check FAIL from behavior change + a truncated stub aside
  scoring all 1s)
- teach sha: 551f185f2835 (changed from baseline, build verified)
- interim read: composite within noise band. The rule worked on its target —
  quadratic-dedupe went 2 redundant asides -> 1 tight one, 82 -> 84. The
  regressions (timeout, strcpy flail) look like task-level flakiness present
  in baseline too, not prompt-caused. Running iter1-confirm per decision rule.
- confirm run: iter1-confirm -> composite 69.8, dims
  i3.42/c4.08/e3.00/g3.58/ec3.58, pass 8/8, emit 8/8
- verdict: REJECT (runs averaged 71.9 vs baseline 74.75; no improvement)
- learned: (a) the no-reteach rule does suppress redundant second asides
  (quadratic-dedupe went 2 asides -> 1 in both runs) but does not buy
  composite points - redundancy was a symptom, not the bottleneck.
  (b) same-prompt run-to-run variance is ~4 points (74.0 vs 69.8 on identical
  sha), so the +-3 noise band is real and single-run deltas under ~5 points
  mean little. (c) baseline's strcpy-overflow flailing (file discovery,
  behavior drift) recurs across runs - it is task noise, not prompt signal.

## iter2 — lead-with-the-hook (2026-06-10)
- teach sha (prev best): 8eb831fb0183 @ 74.75 (iter1 reverted)
- hypothesis: engagement is the weakest dimension in every run so far
  (3.17 / 2.88 / 3.00) and judge comments repeatedly ask for the same thing:
  a prediction hook or surprise opener inside the aside ("could have leaned
  in harder with a quick prediction", "no learner-involving check").
  The contract demands prediction questions in the main behavior but the
  aside rules never ask for a hook, so asides come out correct-but-inert.
- change: added one bullet to the aside rules: lead with the hook (wrong
  number / impossible output / near-miss trap) or a one-line prediction the
  next result settles, with a concrete example phrase; "an aside that merely
  states correct facts is a missed aside."
- run: iter2-lead-hook -> composite 75.2, dims i3.64/c4.09/e3.00/g4.00/ec3.64,
  pass 8/8, emit 8/8, teach sha 742d561220b6 (changed, build verified)
- verdict: REJECT (+0.45 vs baseline is inside the ~4-point noise band, and
  engagement — the dimension the rule targeted — did not move: 3.00 vs 3.17)
- learned: telling the model to "lead with the hook" does not produce hooks;
  engagement appears to be limited by the local model's generation quality,
  not by missing instructions. strcpy-overflow scored 80 vs baseline 63 in
  this run, while rename-symbol dropped to 54 with the same
  narrates-trivia/factual-error flags the baseline showed — per-task swings
  of +-15 dwarf any prompt effect measured so far.

## Final summary (2026-06-10, loop stopped: 2 consecutive REJECTs)

| run                | sha          | composite | i / c / e / g / ec           | pass | emit |
|--------------------|--------------|-----------|------------------------------|------|------|
| baseline (20260610-212422) | 8eb831fb0183 | **74.75** | 3.58/4.17/3.17/3.92/3.83 | 8/8  | 8/8  |
| iter1-no-reteach   | 551f185f2835 | 74.0      | 3.62/4.00/2.88/3.88/3.38     | 6/7  | 7/8  |
| iter1-confirm      | 551f185f2835 | 69.8      | 3.42/4.08/3.00/3.58/3.58     | 8/8  | 8/8  |
| iter2-lead-hook    | 742d561220b6 | 75.2      | 3.64/4.09/3.00/4.00/3.64     | 8/8  | 8/8  |

- Best prompt: the original baseline (8eb831fb0183 @ 74.75). No change was
  accepted; the working tree holds the baseline prompt and a fresh build was
  verified byte-identical to the baseline run's prompt.
- Failed hypotheses: (1) suppressing redundant second asides (worked
  mechanically, no composite gain); (2) demanding a hook/prediction opener
  (engagement unmoved — instruction-following is not the bottleneck there).
- Biggest finding: measurement noise dominates. Identical prompts differ by
  ~4 points run-to-run, and single tasks swing +-15 (strcpy-overflow 63/50/56/80
  across four runs). With n=8 tasks x 1 trial, only ~6+ point effects are
  detectable.
- What to try next, budget permitting:
  1. Run each benchmark at 2-3 trials per task (or grow the corpus) before any
     more prompt tuning — the noise floor currently swallows real effects.
  2. Target the recurring rename-symbol failure mode (senior/high persona
     drifting into narrates-trivia + self-congratulation) — it is the most
     consistent per-task deficit (64/-/56/54), worth one focused rule about
     "proof, not progress reports" for senior profiles.
  3. Engagement may need a better example aside in the contract (show, don't
     instruct) or a stronger base model; adjective-style rules demonstrably
     do not move it.
