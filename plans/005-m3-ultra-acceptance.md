# Plan 005: Certify GLM 5.2 on one 512-GB M3 Ultra

> **Executor instructions**: This is the final target-machine acceptance plan.
> Do not edit inference code here. Execute the matrix, preserve raw evidence,
> update documentation with measured values, and stop on any failed gate.
> Update Plan 005 in `plans/README.md` when complete.
>
> **Drift check (run first)**:
> `git diff --stat bd89932..HEAD -- README.md QA_BEFORE_RELEASES.md speed-bench tests/test-vectors gguf-tools/quality-testing validation`

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW to code; MED operationally due multi-hundred-GB models
- **Depends on**: Plans 001, 002, 003, and 004
- **Category**: tests / docs
- **Planned at**: commit `bd89932`, 2026-07-09

## Why this matters

The branch builds and has GLM fixtures, but the repository contains no auditable
run proving GLM Q2 or Q4 on the requested M3 Ultra/512-GB machine. Existing M3
Ultra speed rows belong to the earlier DeepSeek model family, and release
sign-off explicitly requires Metal Flash rather than Metal GLM. This plan turns
"should fit" into a reproducible acceptance record.

The official hardware target is one M3 Ultra Mac Studio with 512 GB unified
memory. Validate Q4 for quality at 32K, 100K, and Think Max 393216; validate Q2
as the 1M-context capacity path.

## Current state

- GLM 5.2 shape: 79 layers, 754B-class model, 1M configured context.
- Single-file antirez model sizes at planning time: IQ2 about 211 GB, Q2 about
  262 GB, Q4 about 434 GB.
- `README.md:171-188` contains M3 Ultra numbers but does not identify any GLM
  artifact, revision, or GLM-specific result.
- `QA_BEFORE_RELEASES.md:72-79` records quality bands but no commit, hardware,
  model revision, residency mode, memory, or throughput.
- `tests/glm_long_context_smoke.sh` exercises the known long-context corruption
  boundary at 100K.
- Plans 001–004 must provide deterministic artifacts, explicit GLM tests,
  corrected memory reports, and viable streaming policy before certification.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Hardware | `system_profiler SPHardwareDataType SPSoftwareDataType` | M3 Ultra, 512 GB, macOS recorded |
| Disk | `df -h /path/to/gguf` | enough free space plus safety margin |
| Build | `make clean && make -j"$(sysctl -n hw.ncpu)"` | exit 0, no warnings |
| Unit tests | `make test-unit` | all pass |
| GLM gate | `make test-glm DS4_TEST_MODEL=/path/model.gguf` | vectors and long context pass |
| Benchmark | `./ds4-bench -m /path/model.gguf --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 32768 --step-incr 2048 --gen-tokens 128 --csv /tmp/glm.csv` | CSV complete |

## Scope

**In scope**:

- `README.md`
- `QA_BEFORE_RELEASES.md`
- `speed-bench/glm52_q2_m3_ultra.csv` (create)
- `speed-bench/glm52_q2_m3_ultra_ts.svg` (generated)
- `speed-bench/glm52_q4_m3_ultra.csv` (create)
- `speed-bench/glm52_q4_m3_ultra_ts.svg` (generated)
- `validation/glm52-m3-ultra-512.md` (create)

**Out of scope**:

- Any inference, kernel, cache, tokenizer, or server code change.
- Adjusting quality thresholds to make a failed artifact pass.
- CUDA, ROCm, or distributed certification.
- Publishing model files or sensitive local paths/credentials.

## Git workflow

- Suggested branch: `codex/glm-m3-ultra-acceptance`
- Commit example: `Record M3 Ultra GLM acceptance`
- Commit CSV/SVG/result Markdown, not multi-gigabyte raw traces.
- Do not push unless explicitly instructed.

## Steps

### Step 1: Freeze the environment and artifact identities

Create `validation/glm52-m3-ultra-512.md` with:

- source commit and dirty/clean status;
- Mac model identifier, M3 Ultra CPU/GPU core counts, 512-GB memory;
- macOS, Xcode/clang, and `hf` versions;
- internal/external storage type and free space;
- exact GGUF repository revision, filename, byte count, and available stable
  digest/Xet identity;
- power/thermal conditions and relevant environment variables.

Do not include access tokens, usernames in private paths, serial numbers,
hardware UUIDs, or other host identifiers.

**Verify**:

```sh
rg -n 'HF_TOKEN|Authorization:|Serial Number|Hardware UUID' validation/glm52-m3-ultra-512.md
```

Expected: no matches.

### Step 2: Run the clean build and model-free baseline

Run:

```sh
make clean
make -j"$(sysctl -n hw.ncpu)"
make test-unit
git diff --check
```

Record command, exit status, and concise result. Stop on warnings or failures.

### Step 3: Validate Q4 full residency

Use the pinned single-file antirez Q4 from Plan 001. Run and record:

1. `./ds4 --inspect` summary.
2. 32K greedy non-think one-shot generation.
3. `make test-glm-vectors`.
4. 100K GLM long-context smoke.
5. Think Max at exactly 393216 context with a bounded output.
6. OpenAI chat streaming through `ds4-server` using a GLM alias.
7. One real `ds4-agent` read/search/edit/build loop in a temporary project.
8. `ds4-bench` 2K–32K sweep into the tracked Q4 CSV.

For each run record planned memory, actual tracked/peak Metal tensors, process
peak resident memory if available, prefill t/s, generation t/s, and result.
Do not enable SSD streaming for the primary Q4 acceptance.

### Step 4: Validate Q2 including the 1M capacity path

Repeat inspect, vectors, 100K long context, server, agent, and benchmark for Q2.
Then test 1M context in stages rather than jumping directly:

- 393216;
- 786432;
- 1048576.

Use a bounded prompt/output at each stage. Stop immediately if the corrected
guard refuses, memory pressure enters the system danger zone, swap growth is
unbounded, or the machine becomes unresponsive. A clean guard refusal is a
valid recorded result; bypassing `DS4_GLM_MEMORY_GUARD` is prohibited for
acceptance.

### Step 5: Characterize Q4 streaming without making it the default

Only after full-resident Q4 passes, compare:

- full residency;
- the new automatic streaming plan;
- an explicit diagnostic budget equivalent to the former 12-GiB policy.

Use the same 32K greedy benchmark and record expert hit rate, pread bytes/time,
prefill t/s, generation t/s, and logit/vector equivalence. If automatic
streaming is not practically useful, document that outcome and keep full
residency as the 512-GB recommendation.

### Step 6: Publish the measured runbook and graphs

Generate SVGs using `speed-bench/plot_speed.py`. Add a dedicated GLM M3 Ultra
section to README containing:

- exact Q2/Q4 artifacts and revision;
- minimum free disk guidance;
- full-residency commands;
- validated context limits;
- measured memory and throughput;
- unsupported GLM flags (`--power`, `--mtp`, manual `--prefill-chunk`, steering)
  if they remain unsupported;
- explicit warning that sharded Unsloth files are not supported.

Update release QA to link the validation artifact.

## Test plan

- Clean build and model-free tests.
- Q2 and Q4 five-case vector fixtures plus 100-case quality scorer where time
  permits; preserve raw `summary` and `api_summary` lines.
- Q2/Q4 100K long-context smoke.
- Q4 Think Max 393216.
- Q2 staged 1M capacity test.
- Live OpenAI streaming and one agent tool loop per quant.
- Reproducible throughput CSVs using identical settings.

## Done criteria

- [ ] Validation artifact identifies commit, hardware class, software, exact
      models, contexts, flags, memory, quality, and throughput without secrets.
- [ ] Clean build and `make test-unit` pass on the M3 Ultra.
- [ ] Q4 full residency passes vectors, 100K long context, Think Max, server,
      and agent gates.
- [ ] Q2 passes vectors, 100K long context, server, and agent gates.
- [ ] Q2 1M is either demonstrated under the guard or documented as a clean
      measured limitation.
- [ ] Q4 streaming recommendation is based on measured hit rate and throughput.
- [ ] Q2/Q4 CSV and SVG files are committed.
- [ ] README commands are copy-pasteable on a clean clone.
- [ ] QA links the acceptance artifact and requires equivalent evidence for
      future GLM releases.
- [ ] Only in-scope files and `plans/README.md` are modified.

## STOP conditions

- Q4 full residency fails model mapping or Metal residency on 512 GB.
- Vector quality falls outside the existing documented band.
- Long-context output shows the known corruption markers.
- The corrected guard must be bypassed to reach any claimed result.
- Swap growth or system pressure threatens host stability.
- Server or agent failures require runtime code edits; report them as new
  findings rather than expanding this acceptance-only plan.

## Maintenance notes

Every future GLM artifact or runtime change affecting tokenizer, attention,
routing, cache, quantization, or prompt rendering must refresh the validation
record. Keep results tied to immutable model and source revisions; do not retain
unattributed throughput numbers.
