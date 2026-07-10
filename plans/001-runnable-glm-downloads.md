# Plan 001: Make GLM downloads deterministic and runnable

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If a STOP condition occurs, stop and report instead of
> improvising. When done, update Plan 001 in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat bd89932..HEAD -- download_model.sh README.md tests/download_model_test.sh`
> If any current-state excerpt below no longer matches, stop and reconcile the
> live behavior before editing.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW for disabling the broken target; MED for download validation
- **Depends on**: none
- **Category**: bug / dx
- **Planned at**: commit `bd89932`, 2026-07-09

## Why this matters

`glm-unsloth-q4` downloads eleven files and then links the default model to only
shard 1. DwarfStar opens and maps exactly one GGUF path and has no shard
discovery, so that route spends roughly 467 GB of bandwidth and disk before
failing on missing required tensors. The single-file antirez Q2 and Q4 files
match the loader's architecture, but the downloader follows mutable `main`,
accepts any nonempty local file, and does no disk-space preflight.

This plan makes every advertised GLM route runnable and reproducible without
adding high-risk multi-file model mappings.

## Current state

- `download_model.sh:149-158` builds the eleven-file Unsloth target:

  ```sh
  glm-unsloth-q4)
      REPO=$GLM_UNSLOTH_REPO
      MODEL_FILE=$GLM_UNSLOTH_Q4_FIRST_FILE
      MODEL_FILES=
      for part in 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010 00011; do
          MODEL_FILES="$MODEL_FILES $GLM_UNSLOTH_Q4_REMOTE_BASE-${part}-of-00011.gguf"
      done
  ```

- `download_model.sh:345-348` links only `MODEL_FILE`:

  ```sh
  ln -sfn "$OUT_DIR/$MODEL_FILE" ds4flash.gguf
  ```

- `ds4.c:2065-2111` calls `open(path)`, `fstat`, and one `mmap` for the supplied
  path. `weights_bind()` later requires all layers from that same model table.
- `download_model.sh:247-249` and `310-313` skip any existing nonempty file.
- `download_model.sh:273-275` invokes `hf download` without `--revision`.
- The compatible antirez repository revision observed during planning is
  `2638b3b878f5c6cc3ae7334b8dbea1275025f52e`; it contains the 211-GB IQ2,
  262-GB Q2, and 434-GB Q4 single-file artifacts. Confirm it still resolves
  before hard-coding it.
- Shell code uses POSIX `sh`, `set -e`, small functions, and explicit stderr
  messages. Match that style; do not introduce Bash-only syntax.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Shell syntax | `sh -n download_model.sh tests/download_model_test.sh` | exit 0 |
| Downloader tests | `./tests/download_model_test.sh` | all cases pass without network |
| Build inspect tool | `make -j"$(sysctl -n hw.ncpu)" ds4` | exit 0 |
| Full build | `make -j"$(sysctl -n hw.ncpu)"` | exit 0, no warnings |

## Scope

**In scope**:

- `download_model.sh`
- `README.md`
- `tests/download_model_test.sh` (create)

**Out of scope**:

- `ds4.c` and `ds4_metal.m`; do not add shard support.
- GGUF conversion or merge tooling.
- Changing DeepSeek download behavior except through shared preflight code with
  regression coverage.
- Downloading a real multi-hundred-GB model during automated tests.

## Git workflow

- Suggested branch: `codex/glm-runnable-downloads`
- Use concise imperative commits matching the repository, for example:
  `Make GLM downloads deterministic`
- Do not push unless the operator explicitly requests it.

## Steps

### Step 1: Disable the non-runnable sharded route

Remove `glm-unsloth-q4` from the advertised normal targets. If backward
compatibility for the target name matters, retain a case that exits before any
download with a precise message: DwarfStar currently loads one GGUF file; use
`glm-antirez-q4`; native split-GGUF support is not implemented.

Update `README.md` and downloader help in the same change. Do not claim that
the first shard is an entry point.

**Verify**:

```sh
./download_model.sh --help | grep -F 'glm-unsloth-q4'
```

Expected: no normal runnable target is advertised. If a compatibility error
target remains, the help text must label it unsupported.

### Step 2: Pin the antirez model revision

Add one named constant for the tested antirez revision and pass it to every
antirez `hf download` call using `--revision`. Keep DeepSeek behavior unchanged
unless a separately verified revision is available.

Before using the revision, confirm it resolves and contains all three exact
filenames. Record the revision in README's GLM model table.

**Verify**:

```sh
rg -n '2638b3b878f5c6cc3ae7334b8dbea1275025f52e|--revision' download_model.sh README.md
```

Expected: one named revision constant, the `hf download` argument, and user
documentation all agree.

### Step 3: Add dependency and disk-space preflights

Before starting a forced-HF GLM download:

1. Verify `hf` exists and print a tested installation command if absent.
2. Determine free bytes on the filesystem containing `OUT_DIR` using portable
   macOS/POSIX tooling.
3. Associate each GLM target with its expected minimum download bytes and add a
   conservative safety margin for temporary/cache activity.
4. Fail before transfer with required, available, and selected directory values.

Keep `DS4_GGUF_DIR` as the supported escape hatch to a larger volume. Handle
decimal provider sizes versus binary filesystem sizes explicitly.

**Verify**: the network-free test harness must simulate insufficient and
sufficient space and assert that the fake `hf` command is respectively not
called and called once.

### Step 4: Validate existing and newly downloaded files

Do not treat `-s` alone as proof of completeness. Store expected minimum byte
counts and reject undersized files. After a download, if `./ds4` exists and is
executable, run:

```sh
./ds4 --inspect -m "$out"
```

Only update `ds4flash.gguf` after inspection succeeds. If `./ds4` is not built,
emit a clear instruction to build and inspect before inference; do not silently
claim validation.

Do not add a home-grown digest loop over 434 GB unless the pinned repository
provides stable, reviewed digests that can be checked without duplicating the
entire file.

### Step 5: Add a network-free downloader regression harness

Create `tests/download_model_test.sh`. Put temporary fake `hf`, `df`, and, if
needed, `ds4` executables first on `PATH`. Cover:

- Q2 and Q4 select the exact antirez filenames and pinned revision.
- IQ2 remains either explicitly supported or explicitly experimental.
- `glm-unsloth-q4` fails before invoking `hf`.
- insufficient space fails before invoking `hf`.
- an undersized existing file is rejected rather than skipped.
- the symlink is updated only after validation succeeds.
- no token values or host credentials are printed.

Use `mktemp -d` plus a cleanup trap, following
`tests/glm_long_context_smoke.sh:28-29`.

## Test plan

- New file: `tests/download_model_test.sh`.
- All tests must be network-free and use byte-sized dummy files.
- Run `sh -n` before the harness.
- Manually run one real `--inspect` on the Studio after the Q2 or Q4 download;
  automated CI must not fetch a real model.

## Done criteria

- [ ] No documented target downloads eleven shards that the runtime cannot load.
- [ ] Antirez GLM downloads use a reviewed immutable revision.
- [ ] Disk and `hf` preflights happen before transfer.
- [ ] Existing undersized files are rejected.
- [ ] `ds4flash.gguf` is linked only after validation.
- [ ] `sh -n download_model.sh tests/download_model_test.sh` exits 0.
- [ ] `./tests/download_model_test.sh` passes every case without network access.
- [ ] `make` exits 0 without warnings.
- [ ] Only in-scope files and `plans/README.md` are modified.

## STOP conditions

- The pinned revision no longer resolves or lacks one of the exact antirez files.
- `hf download --revision` behaves differently from the currently documented
  Hugging Face CLI contract.
- A correct fix appears to require multi-file model mappings in `ds4.c`.
- Portable free-space detection cannot be made reliable on macOS and Linux;
  report the platform behavior instead of guessing.

## Maintenance notes

When a model file changes, update the revision, expected size, QA fixture result,
and acceptance artifact together. A filename alone is not a release identity.
Native sharded loading can be reconsidered later as a separate high-risk design.
