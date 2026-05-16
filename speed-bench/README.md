## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

For Metal Tensor prefill experiments, treat matmul as the first optimization
surface: profile routed-MoE stages and dense Q8_0 attention projections, then
compare the current standard path, current Tensor auto path, and a default-off
candidate env switch with:

```
python3 speed-bench/run_prefill_candidate_gate.py \
  --candidate-label moe-matmul-first \
  --set-env DS4_METAL_EXPERIMENTAL_MOE_MATMUL=1
```

### Metal Tensor helper map

The Metal Tensor work uses a small set of local tools so speed changes,
logprob drift, and diagnostic attribution stay tied to the same fixtures and
artifact format:

| Tool | Why it exists |
| --- | --- |
| `run_metal_tensor_bench.sh` | Regenerates the Standard Metal / Quality Metal / Tensor Metal chart for the current branch and keeps timestamped CSV/PNG artifacts under ignored `speed-bench/local-runs/`. Use this for PR performance evidence. |
| `run_quality_drift_gate.py` | Runs the five fixed prompt scenarios against `--quality`, `-mt off`, and `-mt auto`, then writes PR-ready `summary.md` and automation-friendly `summary.json`. Use this as the main logprob drift gate. |
| `run_prefill_candidate_gate.py` | Compares a default-off candidate against current Tensor and Standard speed first, then launches the drift gates only when the candidate is speed-positive enough to justify the cost. Use this before promoting any new prefill route. |
| `metal_tensor_presets.py` | Stores named environment profiles for measured default-off candidates so speed, drift, and comparator reruns use the same route settings without copying long env strings. |
| `run_chunked_prefill_drift_gate.py` | Adds resumed-prefill frontier coverage for candidates that depend on nonzero `pos=` route filters, because the five fixed prompts mostly validate cold `pos=0` prefill. |
| `run_mpp_compare_probe.py` and `summarize_mpp_compare.py` | Run and summarize local Tensor-vs-legacy projection comparisons for route attribution. Use them to decide which layer/route caused a drift breach before spending a full five-fixture gate. |
| `summarize_stage_profile.py` | Converts Metal stage-profiler stderr into Markdown/JSON tables so kernel targets are chosen from measured stage time instead of whole-layer timing alone. |
| `index_local_runs.py` | Builds a compact index over ignored local artifacts so candidate runs, drift gates, comparator probes, profiles, and chart runs are easy to find later. |

These tools intentionally write to ignored local directories by default. The
PR should include selected numbers or Markdown summaries, not the raw local
artifacts themselves.

The measured default-off profiles can also be selected with `--preset` to avoid
copying long environment strings by hand:

```
python3 speed-bench/run_prefill_candidate_gate.py \
  --preset mpp-fast-skip-down26-29-30 \
  --run-drift-gate
```

Add `--run-drift-gate` before promoting a candidate. The helper first evaluates
the speed screen; if the candidate fails the prefill or generation floor, it
records the skip reason and does not launch the five-fixture drift gate. When
the speed screen passes, it reuses the five-fixture `--quality` drift gate and
writes JSON plus Markdown summaries beside the benchmark CSVs. By default this
helper writes timestamped output under
`speed-bench/local-runs/<datetime>-<candidate-label>/`, which is ignored by git.
The candidate Markdown scorecard marks production promotion-safe only when every
measured context beats Tensor prefill by at least `--min-prefill-gain-pct`,
every repeat/context pair clears `--min-repeat-prefill-gain-pct`, the candidate
stays above the generation floor set by `--min-generation-gain-pct`, the drift
gate is green, and Tensor-vs-standard drift stays inside the configured
envelope (`--max-tensor-standard-rms` and
`--max-tensor-standard-top20-abs`). Candidates that use nonzero `pos=` route
filters need additional resumed-prefill coverage, because the existing five
fixtures mostly exercise cold `pos=0` prefill. When `--run-drift-gate` is set
and the speed screen passes, the helper now also runs the chunked frontier drift
gate for that class of candidate. Without that chunked gate artifact, nonzero
`pos=` candidates are marked not promotion-safe. With `--run-drift-gate`,
failed candidates still write artifacts before exiting non-zero; add `--no-fail`
for exploratory sweeps. Use `--reuse --out-dir=<existing-run>` to regenerate
summaries from saved CSVs, charts, and drift-gate dumps without rerunning
benchmarks. The gate refuses to use stale `ds4-bench` or nested `ds4` binaries
when core sources or `metal/*.metal` are newer than the executable; rebuild
first, or pass `--allow-stale-binary` only when intentionally summarizing old
artifacts. When nested drift gates are present, the candidate scorecard also
shows the Tensor-vs-standard fixtures or frontiers responsible for the worst
drift metrics. The Markdown scorecard also prints per-context repeat deltas, so
noisy median-only wins can be rejected without opening the JSON. Both JSON
reports record a `run_config` block with the command thresholds and resolved
paths used for the run, and the Markdown reports include a quoted replay
command.

To run only the five-fixture drift gate:

```
python3 speed-bench/run_quality_drift_gate.py
```

For default-off candidates, the drift gate accepts the same `--preset` names as
the candidate gate:

```
python3 speed-bench/run_quality_drift_gate.py \
  --preset mpp-fast-skip-down26-29-30 \
  --max-tensor-standard-rms 0.30 \
  --max-tensor-standard-top20-abs 0.60
```

By default the drift gate writes timestamped output under
`speed-bench/local-runs/<datetime>-quality-drift-gate/`. Set `--out-dir=...` to
override the destination. Each run writes both `summary.json` for automation and
`summary.md` for a persistent human-readable comparison table, including the
fixture responsible for each worst drift metric. Add
`--max-tensor-standard-rms` and `--max-tensor-standard-top20-abs` when the
standalone drift gate should enforce the production drift envelope. The drift
gate also refuses stale `ds4` binaries unless `--allow-stale-binary` is set.

To run the resumed-prefill frontier drift gate for candidates that depend on
nonzero `pos=` filters:

```
python3 speed-bench/run_chunked_prefill_drift_gate.py \
  --preset mpp-fast-continuation-chunks \
  --max-tensor-default-rms 0.30 \
  --max-tensor-default-top20-abs 0.60
```

This script uses `ds4-bench` to grow `speed-bench/promessi_sposi.txt` through
frontiers `512, 1024, 2048, 4096, 8192` by default, dumps one full-logit JSON
file after each resumed frontier, then compares quality, standard Metal, and
Tensor Metal. When a candidate preset or `--set-env` override is present, it
also captures the no-env Tensor baseline as `default_tensor` and reports
`tensor_vs_default_tensor`; the candidate gate uses that pair for resumed
coverage so candidates are judged against the current Tensor baseline instead
of an absolute chunked Tensor-vs-standard envelope. Output is timestamped under
`speed-bench/local-runs/<datetime>-<preset>-chunked-drift-gate/` and ignored by
git. The chunked gate also refuses stale `ds4-bench` binaries unless
`--allow-stale-binary` is set.

To regenerate the standard/quality/Tensor chart for the current branch:

```
OPEN_CHART=0 speed-bench/run_metal_tensor_bench.sh
```

By default the script writes timestamped output under
`speed-bench/local-runs/<datetime>-metal-tensor-bench/`. That folder is ignored
by git so multiple local comparison runs can be kept without pushing the CSVs or
charts. The generated CSV and PNG filenames are also prefixed with the same
datetime run id, so reruns stay distinct even when `OUT_DIR` is reused. The
script refuses stale `ds4-bench` binaries unless `ALLOW_STALE_BINARY=1` is set.
Set `OUT_DIR=...` or `RUN_ID=...` to override the destination.

To create a compact index of saved local benchmark charts, drift, comparator,
candidate-gate, and profile artifacts:

```
RUN_ID=$(date +%Y%m%d-%H%M%S)
OUT_DIR=speed-bench/local-runs/${RUN_ID}-local-run-index
python3 speed-bench/index_local_runs.py \
  --output ${OUT_DIR}/local-run-index.md \
  --json-output ${OUT_DIR}/local-run-index.json
```

The indexer only reads existing JSON summaries; it does not run the model. The
output directory is ignored by git, so it can be regenerated after local sweeps
without changing tracked artifacts. The prefill table includes both median and
repeat-level minimum candidate-vs-Tensor prefill deltas, matching the candidate
gate's speed-first promotion screen. It also reports five-fixture drift and
coverage/chunked drift separately, including the coverage pair used, so a
candidate that passes the normal drift gate but fails resumed-prefill coverage
is visible in the top-level table. Timestamped runs from
`run_metal_tensor_bench.sh` are indexed as chart runs with Tensor-vs-standard
prefill and generation ranges plus the PNG path. If the same `OUT_DIR` is
reused with multiple timestamped `RUN_ID` values, each complete CSV triplet is
indexed separately.

To summarize Metal stage-profile logs from runs with
`DS4_METAL_MOE_STAGE_PROFILE=1`, `DS4_METAL_Q8_PREFILL_PROFILE=1`,
`DS4_METAL_FLASH_ATTN_STAGE_PROFILE=1`, or layer profiling enabled:

```
python3 speed-bench/summarize_stage_profile.py \
  speed-bench/local-runs/<run>/long_code_audit_profile.stderr
```

Use `--output speed-bench/local-runs/<run>/stage-profile-summary.md` to keep a
timestamped local summary beside the raw profile log. When present, the report
also includes routed-MoE timing by Tensor mask, dense Q8_0 shape tables, and
FlashAttention shape tables, which helps separate kernel targets from per-layer
totals. Use `--json-output speed-bench/local-runs/<run>/stage-profile-summary.json`
when the profile should also be indexed by the local-run indexer.

To summarize local Tensor-vs-legacy comparator logs from runs with
`DS4_METAL_MPP_COMPARE_ROUTE=...`:

```
python3 speed-bench/summarize_mpp_compare.py \
  speed-bench/local-runs/<run>/<fixture>.stderr \
  --output speed-bench/local-runs/<run>/mpp-compare-summary.md \
  --json-output speed-bench/local-runs/<run>/mpp-compare-summary.json
```

This report ranks local projection deltas by max abs and RMS, shows comparator
target breaches, and keeps the largest-delta details needed for deciding whether
a fast prefill route should be narrowed before running the five-fixture drift
gate.

To run a targeted comparator probe and summarize it in one step:

```
python3 speed-bench/run_mpp_compare_probe.py \
  --preset mpp-fast-skip-down26-29-30 \
  --case long_memory_archive \
  --route moe_down
```

For dense Q8_0 prefill candidate work, use the same probe with the `q8` route
and a substring filter for the projection shape or module label you want to
inspect:

```
python3 speed-bench/run_mpp_compare_probe.py \
  --case short_code_completion \
  --route q8 \
  --q8-filter attn_q_b \
  --compare-max 3 \
  --verbose
```

For static-mixed FlashAttention candidate work, use the `flash_attn` route. The
probe enables `DS4_METAL_FLASH_ATTN_COMPARE=1` and replays the existing generic
static-mixed path into a reference head-output buffer:

```
python3 speed-bench/run_mpp_compare_probe.py \
  --case short_reasoning_plain \
  --route flash_attn \
  --flash-attn-filter static_mixed \
  --compare-max 1 \
  --verbose
```

By default this writes logs plus `mpp-compare-summary.md/json` under
`speed-bench/local-runs/<datetime>-<preset>-mpp-compare-probe/`. Use
`--all-cases` when a local comparator question needs the same five fixtures as
the logprob drift gate. `--route` is repeatable, and comma or pipe separated
route lists are split into separate probes. The comparator probe is only an
attribution tool; a candidate still needs `run_quality_drift_gate.py` before
promotion. It refuses stale `ds4` binaries unless `--allow-stale-binary` is
set. Add `--continue-after-breach` when the question is whether a route has one
isolated local breach or many; normal probes stop at the first target breach to
keep logs short.
