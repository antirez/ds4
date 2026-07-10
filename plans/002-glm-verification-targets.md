# Plan 002: Add model-aware GLM verification targets

> **Executor instructions**: Follow every step and verification gate. Stop on
> any condition in the STOP section. Update Plan 002 in `plans/README.md` when
> complete.
>
> **Drift check (run first)**:
> `git diff --stat bd89932..HEAD -- Makefile tests/ds4_test.c tests/glm_long_context_smoke.sh tests/test-vectors/README.md QA_BEFORE_RELEASES.md CONTRIBUTING.md`

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: `plans/001-runnable-glm-downloads.md`
- **Category**: tests / dx / docs
- **Planned at**: commit `bd89932`, 2026-07-09

## Why this matters

The downloader replaces `ds4flash.gguf` with the selected GLM model, but plain
`make test` then runs DeepSeek default vectors and a DeepSeek-specific live tool
request. This creates false failures and gives no single command that certifies
GLM. The dedicated GLM long-context script is optional, absent from Make, and
defaults to a path the downloader never creates.

This plan preserves the current DeepSeek suite while creating explicit,
machine-checkable unit and GLM gates.

## Current state

- `Makefile:241-248`:

  ```make
  test: ds4_test ds4_agent_test ds4-eval q4k-dot-test
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	./ds4_test
  ```

- `tests/ds4_test.c:11-14` defaults the model to `ds4flash.gguf`.
- `tests/ds4_test.c:934-936` defaults vectors to
  `tests/test-vectors/official.vec`, the DeepSeek fixture.
- `tests/ds4_test.c:1791-1808` hard-codes the live tool request model to
  `deepseek-v4-flash`.
- `tests/glm_long_context_smoke.sh:23` defaults to the nonexistent
  `models/GLM-5.2-UD-Q4_K_XL.gguf`.
- `QA_BEFORE_RELEASES.md:72-79` documents GLM scoring against the same
  nonexistent `models/...` path.
- Repository test output convention is a named group followed by `OK`, `ERR`,
  and a nonzero exit on any assertion failure. Preserve it.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Build | `make -j"$(sysctl -n hw.ncpu)"` | exit 0, no warnings |
| Model-free tests | `make test-unit` | all model-free checks pass |
| GLM smoke vectors | `make test-glm-vectors DS4_TEST_MODEL=/path/model.gguf` | GLM vectors pass |
| GLM long context | `make test-glm-long DS4_TEST_MODEL=/path/model.gguf` | script prints PASS |
| Combined GLM gate | `make test-glm DS4_TEST_MODEL=/path/model.gguf` | both GLM gates pass |

## Scope

**In scope**:

- `Makefile`
- `tests/ds4_test.c`
- `tests/glm_long_context_smoke.sh`
- `tests/test-vectors/README.md`
- `QA_BEFORE_RELEASES.md`
- `CONTRIBUTING.md`

**Out of scope**:

- Changing model logits, tokenizer behavior, or fixture contents.
- Weakening existing DeepSeek quality thresholds.
- Making multi-hundred-GB tests run in generic CI.
- Server or agent runtime changes.

## Git workflow

- Suggested branch: `codex/glm-verification-targets`
- Commit example: `Add GLM verification targets`
- Do not push unless explicitly instructed.

## Steps

### Step 1: Separate model-free tests

Add `test-unit` to the Makefile. It should build and run:

- evaluator extractor self-tests;
- `ds4_agent_test`;
- server/parser tests (`./ds4_test --server`);
- isolated Metal kernels on a Metal host;
- Q4_K dot tests.

Do not run a full model, official vector, long-context, or tool-quality test in
`test-unit`. Keep the existing `test` meaning backward-compatible or rename it
only with explicit release-note documentation.

**Verify**: temporarily move `ds4flash.gguf` aside on a machine where it exists,
run `make test-unit`, then restore it. Expected: exit 0 without attempting model
open.

### Step 2: Add explicit GLM vector and long-context targets

Require `DS4_TEST_MODEL`; fail immediately with an example if it is unset or not
a file. Add:

- `test-glm-vectors`, setting
  `DS4_TEST_VECTOR_FILE=tests/test-vectors/glm-openrouter/official.vec`;
- `test-glm-long`, invoking `tests/glm_long_context_smoke.sh` with the exact
  model path;
- `test-glm`, depending on both.

Do not rely on whichever model `ds4flash.gguf` happens to reference.

**Verify**:

```sh
make test-glm
```

Expected without `DS4_TEST_MODEL`: fast nonzero exit with one actionable usage
message and no model load.

### Step 3: Make live tool-quality requests model-aware

Replace the hard-coded DeepSeek model alias in the test request builder with the
model family reported by the loaded engine. Preserve DeepSeek behavior and use a
GLM alias such as `glm-5.2` only when the engine is GLM. Do not simply change the
string globally.

Add a unit assertion for both family selections without requiring two huge
models.

### Step 4: Standardize model paths in documentation and scripts

Make the long-context script require an explicit argument or
`DS4_GLM_MODEL`. Do not retain a silently wrong `models/...` default. Update all
commands in the vector README, release QA, and contributing guide to use the
same `DS4_TEST_MODEL` contract and the model-specific manifest paths:

- `data/flash/manifest.tsv`
- `data/glm52-openrouter-100/manifest.tsv`
- `data/pro/manifest.tsv`

### Step 5: Make target-machine GLM gates release-blocking

In `QA_BEFORE_RELEASES.md`, add explicit sign-off items for:

- GLM Q2 and Q4 vector quality;
- 100K long-context GLM smoke;
- one CLI generation;
- OpenAI-compatible server streaming;
- one GLM agent tool/edit loop;
- recorded commit, model revision/hash, macOS version, context, peak memory,
  prefill t/s, and generation t/s.

Allow a release to record an explicit skip, but never silently satisfy GLM
sign-off with a DeepSeek Metal pass.

## Test plan

- Unit-test family-aware request aliases in `tests/ds4_test.c`.
- Run `make test-unit` with no model file.
- On the Studio, run `make test-glm` once with Q2 and once with Q4.
- Confirm the existing DeepSeek `make test` behavior remains intact.

## Done criteria

- [ ] `make test-unit` requires no model and passes.
- [ ] `make test-glm` refuses a missing model path before loading anything.
- [ ] GLM vector target selects only the GLM vector file.
- [ ] GLM long-context target uses the exact supplied model.
- [ ] DeepSeek live tool quality still uses its existing alias and rendering.
- [ ] All documented GLM commands refer to paths produced by Plan 001 or an
      explicit environment variable.
- [ ] Release sign-off cannot pass GLM based only on a Metal Flash run.
- [ ] `make` and `make test-unit` pass without warnings or failures.
- [ ] Only in-scope files and `plans/README.md` are modified.

## STOP conditions

- The GLM vectors are incompatible with the single-file antirez artifacts from
  Plan 001; record the observed mismatch instead of changing thresholds.
- Making tool requests family-aware requires changing public server behavior.
- A model-free test unexpectedly depends on production model state.
- Existing DeepSeek tests regress twice after a reasonable wiring fix.

## Maintenance notes

Future model families must receive an explicit fixture and test target; do not
let the default symlink determine test semantics. Keep expensive live gates out
of generic CI but make their absence visible in release sign-off.
