# Qwen Q4 SSD research archive

Start with [RISULTATI.md](RISULTATI.md) for findings, measurements and limitations.
This directory records the September 30 experiments; it does not enable any
production optimization. The executable, model weights and unrelated working
tree changes are intentionally outside this archive.

## Reproduce the isolated MoE experiment

Run from the repository root on a Mac with Metal. The recorded performance is
specific to an M1 Max with 32 GiB. Run GPU benchmarks serially, without another
model process or compiler competing for resources.

The frozen sources and standalone oracle are self-contained. No model download
or files from earlier experiment directories are needed for this test:

```sh
clang -O2 -Wall -Wextra -fobjc-arc -framework Foundation -framework Metal \
  misc/qwen-radical-2026-09-30/moe/schedule-oracle.m \
  -o /tmp/qwen-radical-schedule-oracle

gzip -dc misc/qwen-radical-2026-09-30/moe/routes-layer9-47.qwr2.gz \
  > /tmp/qwen-radical-routes.qwr2

/tmp/qwen-radical-schedule-oracle \
  --baseline-source misc/qwen-radical-2026-09-30/moe/baseline.metal \
  --qwen-source misc/qwen-radical-2026-09-30/moe/compact.metal \
  --schedule compact-heavy --bench --reps 4 --stages \
  --routing-file /tmp/qwen-radical-routes.qwr2 --routing-layer 9 \
  --output /tmp/qwen-radical-layer9.json
```

Repeat the final command with `--routing-layer 47` and a different output file.
For the small correctness fixtures, omit `--bench`, `--stages` and both routing
arguments. `--safe-math` disables fast math; the recorded safe run used
`--schedule compact`. Recorded benchmark data remains unchanged in this archive.

`moe/manifest.json` preserves the hashes of the original four frozen source
files. `moe/routes-layer9-47.json` records the full capture hash and hashes of
the two extracted records; gzip is lossless and contains routing indices only.
The original complete capture also contained other layers used by the design
audit. That larger capture is not needed for these two timing fixtures.

`moe/prepare.py` is the historical generator, retained for provenance. It reads
the then-current production shader and two earlier ignored experiment files;
**do not run it to reproduce the frozen comparison**. Compile the committed
oracle directly as above. Regenerating would replace the frozen source hashes.
The old path in the oracle's opening compile comment is likewise historical;
the command above uses the actual archived source.

## Reproduce the model comparison

The local Q4 GGUF is required at `gguf/Qwen3.8-Flash-Next-Q4.gguf`. From the
repository root, build `ds4` with the normal Makefile and use a fresh result
directory name:

```sh
make ds4
python3 misc/qwen-radical-2026-09-30/model/run-mtp.py --name mtp-repeat
python3 misc/qwen-radical-2026-09-30/model/run-mtp.py \
  --name mtp-cli-repeat --prompts rome --user-cli
```

The runner removes inherited `DS4_*` overrides, records its explicit settings,
freezes source/binary hashes, checks output equality and stops on a failed or
mismatched run. The two prompts are now local to this archive. Historical JSON
commands retain the original Roma prompt path; its copied bytes are unchanged.

The recorded model runs used `b96a12be` plus the working-tree changes present
on September 30. Their exact source and binary hashes are in each results JSON.
Those prior production changes are not included by this research archive:
building a different checkout measures that checkout and does not recreate the
recorded binary automatically. Both compared variants within each series used
the same binary, and the two series recorded identical source/binary hashes.

`io/sample_lossless.py` performs only bounded CPU codec tests using the same
local GGUF and macOS libcompression. It rewrites `io/lossless-results.json`;
preserve the recorded file before deliberately repeating that experiment.

The architecture documents include source line references and links to earlier
local experiments as provenance. Those older archives may not exist in another
checkout. Their speculative memory budgets and speed projections are explicitly
separate from the committed raw results.
