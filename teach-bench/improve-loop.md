# Teaching-prompt improvement loop

You are running an automated improvement loop on ds4-agent's **teaching
prompt**. ds4-agent is a terminal coding agent that doubles as a programming
mentor: while it works it emits short `<teach>` asides. The benchmark in
`teach-bench/` runs the agent over 8 small coding tasks and has an OpenAI
judge score every aside 1-5 on five dimensions (insight, calibration,
engagement, grounding, economy) against the mentor design contract. Your job
is to iteratively improve the judge composite (0-100) by editing the
teaching prompt, re-benchmarking, and keeping only changes that help.

## Where things are

- The teaching prompt is the C string `agent_teach_prompt` in
  `ds4_agent.c` (search for `static const char agent_teach_prompt`).
  This is the ONLY thing you may edit, plus a one-line entry per iteration
  in the experiment log. Do not touch the rest of the agent, the corpus
  (`teach-bench/prompts.json`), the judge rubric, or the benchmark tool —
  changing the measuring stick invalidates the comparison.
- Rebuild with `make ds4-agent` from the repo root.
- Benchmark: `cd teach-bench && ./teachbench.py bench --run-id <name>`.
  One full run takes 15-25 minutes and must not run concurrently with any
  other ds4 instance (global lock; the tool aborts loudly if locked).
- Results: `./teachbench.py history --json` for per-run aggregates;
  `teach-bench/results/<run-id>/run.json` for per-aside judge scores,
  comments, and flags; `results/<run-id>/<test>/stdout.txt` for transcripts.
- Experiment log: `teach-bench/experiments.md` (create it if missing).

## Before the first iteration

1. Verify `OPENAI_API_KEY` is set; if not, source `~/.zshrc` in the shell
   you use to run the benchmark. If still unset, stop and report.
2. Confirm the working tree is clean enough to experiment on; commit any
   unrelated pending changes first so each iteration is one clean commit.
3. If `history` already shows a run for the current teaching prompt (same
   `teach` hash after a fresh `make ds4-agent`), use it as the baseline.
   Otherwise run the baseline: `./teachbench.py bench --run-id baseline-<n>`.
4. Log the baseline in `experiments.md` (format below).

## The loop (repeat until the stop rule fires)

1. **Diagnose.** Read the latest `run.json`. Collect the worst-scoring
   asides and the judge's comments and flags (`generic-lecture`,
   `corporate-cheer`, `wrong-level`, `factual-error`, `overlong`,
   `buried-reveal`, ...). Look for a pattern, not a one-off: a flag or
   weak dimension that recurs across tasks. Read one or two transcripts to
   see the failure in context. Also note reliability (tests emitting zero
   asides) and check failures — a prompt change that costs task success or
   aside reliability is a regression no matter what the composite says.
2. **Propose ONE focused change** to `agent_teach_prompt` aimed at the
   diagnosed pattern. One hypothesis per iteration; never bundle. Keep the
   prompt static text (no dates, paths, or anything run-specific — it must
   stay byte-identical across runs so the sysprompt KV cache and the
   version hash keep working). Keep its overall length within ±25% of the
   original.
3. **Log it** in `experiments.md` BEFORE running: iteration number,
   hypothesis ("judge flags generic-lecture on 4/8 tasks; adding an
   explicit 'name the exact line you are looking at' rule"), the diff
   summary, expected effect.
4. **Rebuild and verify**: `make ds4-agent`, then run the benchmark with a
   descriptive id: `./teachbench.py bench --run-id iter<N>-<slug>`. After
   it finishes, confirm in `history` that the `teach` hash CHANGED from the
   previous run — an unchanged hash means a stale build or an ineffective
   edit; fix that before drawing any conclusion.
5. **Compare** against the current best run (not just the previous one):
   judge composite, the five dimension means, reliability, and task pass
   rate. Decision rules:
   - Composite within ±3 points: treat as noise. Re-run once
     (`--run-id iter<N>-confirm`) only if the change looked promising;
     otherwise count it as a non-improvement.
   - Improvement > 3 points with no regression in reliability or pass
     rate: ACCEPT. `git commit` the prompt change with the run ids and
     scores in the message. This becomes the new best.
   - Otherwise: REJECT. `git checkout ds4_agent.c` to revert, rebuild.
6. **Update the log** with the run id, scores, verdict, and one sentence on
   what was learned (rejected hypotheses are findings too — they prevent
   re-trying the same idea).

## Stop rule

Stop when 2 consecutive iterations end in REJECT or noise, or after 8
iterations total, whichever comes first. Then make sure the working tree
holds the best-scoring prompt (revert/rebuild if the last change was
rejected), run `make ds4-agent` one final time, and verify `history` shows
the best run's `teach` hash for a fresh build.

## Experiment log format (experiments.md)

```markdown
## iter3 — open-loops-rule (2026-06-12 09:15)
- teach sha: 3c1a99d20e44 (prev best: 8eb831fb0183 @ 74.8)
- hypothesis: engagement mean is 3.1, judge calls asides "flat"; add a
  concrete open-loop example to the contract instead of the abstract rule
- change: replaced the "Make them lean in" bullet with example-driven text
- run: iter3-open-loops -> composite 79.1, dims i3.9/c4.1/e3.8/g4.0/ec3.9,
  pass 8/8, emit 8/8
- verdict: ACCEPT (+4.3 vs best; engagement +0.7, no regressions)
- learned: examples in the contract move behavior more than adjectives
```

## Final report

When the loop stops, append a summary to `experiments.md` and report: the
full history table, baseline vs best composite, which changes were accepted
(with one-line rationale each), which hypotheses failed, and what you would
try next if the budget allowed.

## Mentor design intent (so proposals stay on-axis)

The mentor is meant to be the teacher students remember twenty years later:
Khan Academy patience plus slightly unhinged hacker glee. First principles,
no magic; open loops and prediction questions; failures treated as the
interesting part; opinions earned from scars; calibrated one level above
the learner profile; one concept per aside, tied to the exact code at hand;
no corporate cheer, no condescension, no generic lectures. Improvements
should make the asides MORE like that — sharper, better-calibrated, more
grounded — not safer and blander. If the composite rises while the asides
drift toward beige competence, flag it in the log: that is winning the
metric and losing the product.
