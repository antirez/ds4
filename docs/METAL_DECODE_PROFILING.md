# Reproducing the Metal F16 measurements and profiling real decode

The F16 microbenchmark measures kernel execution. The session profiler answers
which kernels the model actually dispatches and how much time their traced
passes take. Neither result by itself establishes an end-to-end speedup.

## Reproduce baseline/current F16 kernels

From the repository root on a Mac with the command-line developer tools:

```sh
make bench-metal-f16-decode METAL_F16_BENCH_ARGS="--output /tmp/ds4-f16-run1 --samples 8 --repeats 512 --warmup 2 --math both"
```

The default baseline is upstream `6289c516273979173abbc062209a81dd3706b804`.
Use `--baseline REF` to choose another local commit. The tool extracts the
actual shader functions from that Git object and the current working tree;
it has no dependency on a temporary file or a frozen kernel copy. A shallow
checkout missing the baseline fails with a diagnostic. Fetch the needed
history explicitly before rerunning; the tool never downloads code itself.

Cases cover K=4096 F16 pair NR2/NR4, attention/indexer quad widths 1024+256,
Q8 QKV plus the quad, and ratio 128 with no second compressor. The driver
allocates identical inputs once, specializes both libraries, alternates arm
order, and checks complete projection/state buffers and guards bitwise after
each timed arm against the ordinary matvec plus separate state-write reference.
The respective baseline/current threadgroup-memory sizes are used.

The output prefix must be new. It produces:

- `.csv`: individual warmup/sample rounds, order, shapes and GPU times.
- `.json`: resolved baseline/current commits, dirty state, source and harness
  hashes, compiler/GPU/OS metadata, parameters, memory sizes and paired summaries.

Compilation, allocations and reference checks are outside the timed region.
The interval is GPU command-buffer time divided by repeated dispatch count,
with hot weights and unchanged inputs. These direct shader calls bypass host
eligibility: an NR4 or compound case can be measured even when that device's
runtime would choose a different kernel. Paired min/max are sample ranges,
not confidence intervals. Use enough repeats and check whether the sign stays
consistent; do not extrapolate to model token/s or across Apple generations.

## Identify dispatches in a real Q8 model

Build the dedicated driver:

```sh
make metal-decode-profile
```

Use the same Q8 GGUF and settings as the end-to-end comparison. For example:

```sh
./speed-bench/metal_decode_profile \
  -m /absolute/path/to/model-AProjQ8-SExpQ8-OutQ8.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --prefix-tokens 2048 --ctx 4096 --prefill-chunk 128 \
  --warmup 4 --tokens 16 \
  --timeline /tmp/ds4-decode-run1.timeline \
  --csv /tmp/ds4-decode-run1.csv 2>/tmp/ds4-decode-run1.log

python3 speed-bench/metal_decode_profile.py /tmp/ds4-decode-run1.timeline \
  --steps /tmp/ds4-decode-run1.csv --json /tmp/ds4-decode-run1.json
```

**Add `--ssd-streaming` on machines that run the model from SSD.** The default
is resident mode. SSD profiling currently uses the automatic cache budget;
the driver does not expose the CLI's explicit cache-size/expert-count options.
Use the same mode across compared driver runs and retain the effective cache
budget printed in each log. The local M1 Max validation used SSD streaming.
The prefix is the first N tokens of the supplied raw text, without a chat template. Greedy selection
excludes EOS so the requested number of steps completes. CSV token IDs allow
the traced and control runs to be compared directly.

Record the checkout (`git rev-parse HEAD` and `git status --short`), exact GGUF,
prompt, command and any inherited kernel/scheduling settings alongside the
log. Use a new output path per process: trace files are append-only in the
backend, and the driver refuses to reuse an existing output. It also rejects
the common stage-profile/graph-dump settings that change dispatch selection.

The driver uses the existing `DS4_METAL_ENCODER_TIMELINE` diagnostic. A small
annotation API snapshots phase and absolute KV position into each command
buffer, including buffers restarted after selected-expert readback. Its `C`
records accompany the existing `B` command-buffer and `E` encoder records.
Asynchronous completion order does not change the stored phase. Encoder
records belong to the actual traced encoder, so owned dispatches and the
background keepalive cannot contaminate the last traced record.

## Read the report conservatively

The default report selects only measured `decode`; `--phase prefill` or
`--phase warmup` selects those phases separately. It shows actual dispatch
counts, valid GPU time and the share of **summed traced command-buffer GPU
time**, with per-position detail in JSON. In particular:

- F16 pair-store and quad-store are exclusive compressor passes.
- Q8 QKV+F16 is a compound pass. Its full duration stays compound; the tool
  does not assign a guessed fraction to the F16 work.
- Ordinary F16 pair calls are reported separately as candidates. Generic F16
  matvecs are not assumed to be compressor calls.
- Concurrent or mixed-kernel passes cannot be divided into individual kernel
  times. Unresolved/truncated names remain ambiguous.
- Invalid or missing counter samples retain their observed dispatch counts;
  their time is unavailable. Dropped encoders and structural inconsistencies
  are reported, rather than interpreted as zero-cost work.

The timeline splits serial dispatch groups into timestamped compute passes.
It retains kernel-selection gates and command-buffer boundaries, but changes
encoder boundaries and can affect overlap, cache state and clocks. Owned
command buffers and keepalive GPU work are outside its coverage. The reported
denominator is neither complete GPU time nor CPU wall time, and is not an
interval union if command buffers overlap. Its percentages describe the
instrumented execution, not an uninstrumented end-to-end speedup.

## Compare with tracing disabled

Repeat the identical command with `--no-trace` instead of `--timeline PATH`
and a fresh CSV/log path. Keep the same model, prompt, prefill length, warmup,
token count and SSD/resident mode. Compare token IDs before comparing the
recorded selection/evaluation/wall durations. Alternate traced/control order
over several runs to assess instrumentation overhead and temporal drift;
one traced run followed by one control is only a functional check.

For the original PR speed comparison, benchmark the two binaries with tracing
disabled and a paired, order-alternated design. The profiler is for attribution,
not the timing source for that comparison.

## Validation

```sh
make test-metal-decode-profile       # parser fixtures, no GPU needed
make test-metal-timeline             # real backend, no GGUF needed
make test-metal-decode-fusions       # original bitwise shader oracles
```

The native timeline regression covers flush plus both selected-readback
restart mechanisms, immutable context, rejection of mid-batch phase changes,
valid timestamps, guarded F16 outputs, and an untraced owned dispatch that must
not modify an earlier record. The parser checks missing/duplicate/concatenated
records, out-of-order completion, invalid counters, compound/mixed attribution,
dropped encoders and CSV joins.

On M1 Max the reproducible benchmark passed all five cases in strict and fast
math smoke runs; the original 68+68 bitwise cases still pass after sharing the
fixture helpers. A short model run with the available AProjQ4 Vision GGUF,
SSD streaming, prefix 32 and two measured decode steps exercised 42 quad
compressor dispatches and 40 ordinary F16 pairs. This validates real-model
capture; it does not reproduce the tester's AProjQ8/M5 workload or establish a
performance result. The same run with tracing disabled generated identical
token IDs. That same-model M5 measurement remains to be run there.
