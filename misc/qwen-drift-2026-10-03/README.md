# Qwen prefill arithmetic and token drift

Start with [RISULTATI.md](RISULTATI.md). This archive records the October 3
investigation of [PR comment 5963229415](https://github.com/antirez/ds4/pull/1056#issuecomment-5963229415).
It identifies changed prefill arithmetic and reproduces logit and greedy token
differences with controlled ablations on M1 Max/Q4/SSD. The tester's exact
M4 Max/Q2 resident configuration and prompts were not reproduced.

The archive contains 10 full-vocabulary captures, four 64-step greedy captures,
their logs, numerical summaries, fixtures, diagnostic patch and independent
audits. Model weights, executables, object files and the generated source
snapshot are excluded. Production behavior is unchanged by this archive.

## Recreate the diagnostic sources

Run from the repository root. Python 3, Git, `tar` and `patch` are required.
The preparation script extracts 103 tracked files from the fixed commit
`0231312820fb0ba0e8953606976461644511bcab`, applies the diagnostic patch only
inside a fresh snapshot, and checks every source hash against the captured
snapshot. It refuses to overwrite an existing destination.

```sh
python3 misc/qwen-drift-2026-10-03/prepare-snapshot.py
make -C misc/qwen-drift-2026-10-03/head-rebuilt -j4 ds4
```

This uses committed source blobs; pending production edits are not copied.
The recorded executable hash is provenance for the original build. A rebuild
need not produce the same binary hash, because paths and toolchains can differ;
the 103 source hashes must match.

## Repeat the model comparisons

The default runners expect the same local Q4 GGUF at
`gguf/Qwen3.8-Flash-Next-Q4.gguf`, macOS with Metal, and enough memory for
SSD streaming. Run one model process at a time. Use a fresh output directory
for each series; the scripts reject existing destinations.

```sh
python3 misc/qwen-drift-2026-10-03/run-dense-ab.py \
  --snapshot head-rebuilt --output dense-repeat
python3 misc/qwen-drift-2026-10-03/run-greedy-ab.py \
  --snapshot head-rebuilt --output greedy-repeat
```

Both runners remove inherited `DS4_*` overrides and execute from the isolated
snapshot directory, so the Metal sources match that snapshot. Context is 16384,
prefill chunk is 2048, temperature is zero, `--nothink` is set and MTP is off.
The fixture token counts are 29, 575 and 5942. Full prompt token IDs and content
hashes are in `fixtures.json`.

The diagnostic flags are checked for **presence**: even `=0` enables them.
The control arm omits both flags, rather than assigning zero.

- `DS4_DIAG_BASE_PREFILL_DENSE`: restores the main-style F16/Q8 batch gates
  and permits split-K for F32 prompt projections.
- `DS4_DIAG_BASE_PREFILL_SPLIT_ONLY`: keeps HEAD's F16/Q8 policy and permits
  only split-K in the prefill dense entry point.
- `DS4_DIAG_TRACE_PREFILL_DENSE`: writes dense projection shape and split
  counts to stderr; it is common to the first-logit comparisons.

The graph remains in PREFILL phase. Attention partition policy, HC arithmetic
and ordinary T1 decode kernels retain HEAD behavior in both arms. An arm named
`base_dense` is a policy ablation on HEAD, not the whole `0aaea5a` executable.

## Recorded results and provenance

`dense-ab/results.json` records commands, explicit environment, hashes of the
original binary/host sources/shaders and statistics over all 248320 logits.
The dense runner checks these hashes after each run. Control repeats for 29
and 575 token prompts are byte-identical. The long 5942-token comparison has
one run per arm, without a repeated control.

`greedy-ab/results.json` records commands, environment, capture hashes and
selected token IDs. Its historical runner did not check source hashes per run;
the independent audit verified the source/binary hashes after the series.
The two arms diverge after common prefixes of 58 and 19 generated tokens.
No repeated greedy controls were run. These are numerical comparisons, not
quality scores or performance benchmarks.

Historical absolute paths, times and hashes in the recorded captures are
preserved. The runners now accept fresh snapshot/output paths; these argument
additions do not change the recorded model policy. `comment-source.json`
identifies the public comment and its body hash without copying its text.
`validation-audit.md` independently recomputes the capture statistics and checks
provenance. It documents the limits of extrapolating to M4/Q2 resident.
