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

## Interim summary after iter2 (2026-06-10, stop rule fired; loop later resumed at user request)

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

## iter3 — prediction-example (2026-06-11)
- teach sha (prev best): 8eb831fb0183 @ 74.75
- hypothesis: engagement is stuck at ~3.0-3.2 across all four runs and iter2
  proved adjective-style rules do not move it. The contract has exactly one
  example aside (the search-before-edit scar story) and the model's asides
  mimic its shape: declarative, no reader involvement. Adding a SECOND example
  that demonstrates a prediction/hook aside (open loop, stakes, evidence
  reading) should move behavior the way instructions could not — examples set
  the distribution the model samples from.
- change: added one example aside after the existing one, modeling "watch this
  run - I expect it to fail, and the way it fails is the lesson" with two
  competing theories the next result will arbitrate between. No rule text
  added or changed.
- expected: engagement mean up; insight possibly up (theory-arbitration shape
  invites mechanism talk); watch economy — prediction asides must stay short.
- run: iter3-prediction-example -> composite 71.4, dims
  i3.60/c4.00/e3.00/g3.90/ec3.60, pass 8/8, emit 7/8 (rename-symbol emitted
  zero asides), teach sha ab7be3f4fa74 (changed, build verified)
- verdict: REJECT (-3.35 vs baseline, engagement still 3.00, reliability dip)
- learned: the example DID transfer style — endian-magic's aside opens with
  "I'm betting the problem is..." — but the model predicts BEFORE it
  understands, so the aside scored i2/g2: confident speculation, not an open
  loop. With this base model, prediction-shaped examples trade grounding for
  theater. Engagement is now 3.17/2.88/3.00/3.00/3.00 across five runs —
  treat it as model-bound, stop chasing it via the prompt.

## iter4 — no-progress-reports (2026-06-11)
- teach sha (prev best): 8eb831fb0183 @ 74.75 (iter3 reverted)
- hypothesis: the senior/high tasks fail the same way every run:
  rename-symbol 64/56/54 (+1 timeout, +1 zero-aside run) and endian-magic
  iter3's 52, always flagged narrates-trivia or "restates the task; no
  method". The model fills asides with progress reports ("I'll rename the
  declaration, definition, and callers") instead of transferable knowledge.
  The aside rules say "teach decisions, not keystrokes" but never define the
  test for it. Adding a concrete acceptance test — would this still be true
  next week on different code? — should convert or kill narration asides.
- change: one bullet added to the aside rules: an aside must state something
  reusable on other code (a mechanism, an invariant, a failure mode, a way to
  verify); if it only describes what you did or are about to do, it is a
  progress report, not teaching - cut it.
- expected: insight up on senior tasks; rename-symbol/endian-magic floors
  lift; narrates-trivia flags disappear; watch emit reliability (the rule
  kills asides, so rename-symbol may emit zero - acceptable only if the
  composite floor rises elsewhere).
- run: iter4-no-progress-reports -> composite 74.7, dims
  i3.60/c4.00/e3.10/g4.10/ec3.60, pass 7/7 checked, emit 7/8 (rename-symbol
  TIMEOUT again, its 2nd of 6 runs), teach sha c24145f6b289 (changed,
  build verified)
- verdict: REJECT (dead even with baseline 74.75; target floor never lifted)
- learned: the rule killed narration as intended — zero narrates-trivia
  flags, junior tasks 84/84/80 — but the failure migrated instead of dying:
  makefile-header fell to 52 flagged overlong (the model pads asides to make
  them "reusable"), and rename-symbol timed out so the senior floor is
  untested. Constraining content shape on this base model squeezes the
  balloon; the composite never moves.

## Final summary (2026-06-11, loop stopped again: iter3+iter4 consecutive REJECTs after resume)

| run                       | sha          | composite | i / c / e / g / ec        | pass | emit |
|---------------------------|--------------|-----------|---------------------------|------|------|
| baseline (20260610-212422)| 8eb831fb0183 | **74.75** | 3.58/4.17/3.17/3.92/3.83  | 8/8  | 8/8  |
| iter1-no-reteach          | 551f185f2835 | 74.0      | 3.62/4.00/2.88/3.88/3.38  | 6/7  | 7/8  |
| iter1-confirm             | 551f185f2835 | 69.8      | 3.42/4.08/3.00/3.58/3.58  | 8/8  | 8/8  |
| iter2-lead-hook           | 742d561220b6 | 75.2      | 3.64/4.09/3.00/4.00/3.64  | 8/8  | 8/8  |
| iter3-prediction-example  | ab7be3f4fa74 | 71.4      | 3.60/4.00/3.00/3.90/3.60  | 8/8  | 7/8  |
| iter4-no-progress-reports | c24145f6b289 | 74.7      | 3.60/4.00/3.10/4.10/3.60  | 7/7  | 7/8  |

- Best prompt: still the original baseline (8eb831fb0183 @ 74.75). Four
  hypotheses tested, none accepted; working tree reverted and fresh build
  verified byte-identical to the baseline run's prompt.
- The four failures, compressed: anti-redundancy worked mechanically but
  bought nothing; hook instructions were ignored; a hook EXAMPLE transferred
  style but produced speculation-before-evidence; an anti-narration test
  killed narration but the failure mass moved to overlong padding. Every
  intervention reshaped the asides; none raised the composite.
- Conclusion after 6 runs: the composite is pinned at ~70-75 by (a) base-model
  generation quality (engagement never left 2.88-3.17 under three different
  attacks) and (b) benchmark noise (same-sha spread ~4 points, single-task
  spread +-15, rename-symbol fails structurally in 3 different ways across
  runs). Prompt wording is not the binding constraint at n=8x1.
- Recommendation: stop tuning the prompt until the harness can resolve
  smaller effects (2-3 trials/task or a bigger corpus) or the base model
  improves. Then retry iter4's narration test first - it had the cleanest
  mechanism-level win - ideally bundled with an economy guard, which the
  one-change-per-iteration rule forbade here.
