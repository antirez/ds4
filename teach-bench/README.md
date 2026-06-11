# teach-bench

Benchmark for ds4-agent's **teaching asides** — the `<teach>` paragraphs the
agent emits in teach mode, rendered as `📚 ...` in the terminal.

Teaching quality is subjective, so this benchmark triangulates it three ways:

1. **Objective harness** — a small corpus of fast, self-contained coding tasks
   that reliably produce teachable moments. Each task ships its own workspace
   files and a shell `check` that verifies the agent actually finished the job.
2. **LLM judge** — an OpenAI model scores every aside 1–5 against the agent's
   own teaching contract (the rubric is lifted from `agent_teach_prompt` in
   `ds4_agent.c`): *insight, calibration, engagement, grounding, economy*.
3. **Human ratings** — an interactive `rate` mode so you can score the same
   asides yourself; the report shows judge-vs-human agreement (Pearson r) so
   you can tell whether the judge is worth trusting.

Stdlib-only Python; no dependencies.

## Quick start

```sh
cd teach-bench

# The whole pipeline in one shot: run + judge + report
# (needs OPENAI_API_KEY in the environment; ~8 tasks x 1-3 min each on an M4 Max)
./teachbench.py bench

# Or a quick spin on three tasks
./teachbench.py bench --prompts off-by-one,mutable-default,float-compare

# All runs at a glance, with teaching-prompt version hashes
./teachbench.py history

# Rate the asides yourself afterwards
./teachbench.py rate
```

The steps are also available individually — `list` (show the corpus), `run`
(drive the agent), `eval` (judge a recorded run), `report` (summarize) — for
re-judging an old run with a different model, rating without re-running, and
so on. `eval`, `rate`, and `report` default to the latest run; pass
`--run <id>` for an older one. If `OPENAI_API_KEY` is unset, `bench` skips
the judge step and still prints the report; judge later with `eval`.

## What a run does

For every prompt × trial, the runner:

- creates a fresh workspace under `results/<run-id>/<prompt-id>/work/` and
  writes the task's seed files into it;
- creates a fresh `HOME` next to it and seeds `~/.ds4/learner.md` with the
  task's **persona** (a junior CS student or a staff systems engineer), so
  the agent's level-calibration can be evaluated against a known profile;
- invokes `ds4-agent --non-interactive --teach <level> --nothink -m <model>
  -p <prompt> --trace <trace.log>` with that HOME and workspace (real
  `~/.ds4` is never touched);
- extracts the `<teach>` asides from the trace log (exact spans, straight
  from the generated tokens; rendered-stdout `📚` parsing is the fallback),
  runs the task's `check` command, and appends everything to
  `results/<run-id>/run.json`;
- recovers the **teaching prompt** the agent actually executed with (the
  `# Teaching` section of the system prompt, from the trace's prompt-token
  dump) and stores it in `run.json` as `teach_prompt`, with a short
  `teach_prompt_sha` version hash.

The agent's Metal backend loads `metal/*.metal` relative to its cwd; the
runner sets the `DS4_METAL_*_SOURCE` overrides to the `metal/` directory
next to the agent binary so it can run inside the per-task workspaces.

Runs are sequential by design: ds4-agent holds a global instance lock
(`/tmp/ds4.lock`), so don't run the benchmark while another ds4 instance is
up. The per-run learner-profile update the agent performs on exit goes to the
throwaway HOME. `run.json` is rewritten after every task, so an interrupted
run keeps its partial results.

## Scoring

The judge returns, per aside, 1–5 on each dimension plus a blunt comment, and
per run a holistic 0–100 `overall`, a summary, and failure flags
(`generic-lecture`, `corporate-cheer`, `wrong-level`, ...). The report's
`judge` column is the **composite**: the dimension mean across asides scaled
to 0–100 — more controlled than the holistic number, so prefer it for
comparisons between runs.

Two numbers matter besides the scores:

- **reliability** — fraction of runs that emitted at least one aside (a
  teach-mode regression shows up here first);
- **task success** — fraction of `check` commands that passed (teaching is
  worthless if the agent stops doing the work).

Judge model defaults to `gpt-5.2`; override with `--eval-model` or
`$TEACHBENCH_EVAL_MODEL`. Any OpenAI-compatible endpoint works via
`--base-url` / `$OPENAI_BASE_URL`.

## Corpus format

`prompts.json` holds `personas` (name → learner.md text) and `prompts`:

```jsonc
{
  "id": "off-by-one",            // unique, used in --prompts and result dirs
  "title": "...",
  "persona": "junior",           // optional; empty profile if omitted
  "teach_level": "medium",       // off | low | medium | high
  "prompt": "...",               // what the "developer" asks the agent
  "notes": "...",                // orientation for the judge: what a good
                                 // mentor would plausibly teach here
  "check": "cc -o sum sum.c && [ \"$(./sum)\" = \"150\" ]",
  "files": [ {"path": "sum.c", "text": "..."},
             {"path": "header.bin", "base64": "..."} ]
}
```

Each prompt also carries a `fixes` field — the canonical minimal solution as
`{path, old, new}` replacements. `./teachbench.py selftest` validates the
whole corpus without touching the agent: every `check` must fail on the
seeded files and pass after `fixes` are applied. Run it after any corpus
edit.

Guidelines for new prompts: the task should be finishable by the agent in a
couple of tool calls (1–3 min wall clock), contain one genuinely teachable
idea, and have a `check` that fails on the seeded files and passes once the
task is done. Keep the corpus small enough that a full run stays under half
an hour.

## Comparing runs and the improvement loop

Each run is immutable under `results/<run-id>/`. `history` puts every run on
one line — date, teaching-prompt hash, check pass rate, aside reliability,
judge composite, human mean — so a prompt change shows up as a new `teach`
hash with new scores next to it:

```
run              created (utc)        teach         tests  pass  emit  asides  judge  human
20260610-212422  2026-06-11 02:24:22  8eb831fb0183  8      8/8   8/8   12      74.8   -
20260612-091500  2026-06-12 09:15:00  3c1a99d20e44  8      8/8   8/8   14      79.1   -
```

This is the intended shape of an automated self-improvement loop (e.g.
Claude Code driving it non-interactively):

1. read the best and worst judged asides from the latest `run.json` (the
   judge's per-aside comments say *why* something scored low);
2. edit `agent_teach_prompt` in `ds4_agent.c`, rebuild `ds4-agent`;
3. `./teachbench.py bench --run-id <experiment-name>`;
4. `./teachbench.py history --json` and compare the new composite against
   the previous teach-prompt hash; keep or revert the change.

The recorded `teach_prompt` is recovered from the agent's own trace, so it
is always the text the binary really ran with, not what the source claims —
a stale build shows up as an unchanged hash. `report --teach-prompt` prints
it; two runs' prompts can be diffed straight from their `run.json`. Use
`--seed` for fixed sampling seeds and `--trials 3` to measure variance;
judge scores move a few points between identical runs, so treat small deltas
as noise.
