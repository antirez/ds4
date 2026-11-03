# Independent validation of the arithmetic fix

The completed `fix-ab/results.json` has status `PASS`, 14 successful serial
CLI runs, and seven comparisons. I independently reread all raw captures
with Python's standard library, without importing the runner's validation
or comparison functions and without executing a build, model, or GPU test.
All seven pairs are byte-identical JSON files.

| Series | Prompt tokens | Runs | Independent result |
|---|---:|---|---|
| Resident first logits | 29 | base `0aaea5a` / root fix | 248,320 finite values per arm; zero differing FP32 values |
| Resident first logits | 575 | base `0aaea5a` / root fix | 248,320 finite values per arm; zero differing FP32 values |
| Resident first logits | 5,942 | base `0aaea5a` / root fix | 248,320 finite values per arm; zero differing FP32 values |
| Resident greedy | 29 | base `0aaea5a` / root fix | 64 steps per arm; all selected IDs, token bytes, and complete JSON identical |
| Resident greedy | 575 | base `0aaea5a` / root fix | 64 steps per arm; all selected IDs, token bytes, and complete JSON identical |
| Full Q4 SSD first logits | 29 | before fix / root fix | 248,320 finite values per arm; zero differing FP32 values |
| Full Q4 SSD first logits | 575 | before fix / root fix | 248,320 finite values per arm; zero differing FP32 values |

For logits I packed every parsed value as little-endian FP32 and compared
all 248,320 four-byte words. The resulting SHA-256 values match the report;
the raw JSON and log SHA-256 values also match. Argmax token IDs agree with
the independently computed maximum over the complete vocabulary. Every
dump reports the expected prompt count and context 16,384. All greedy step
indices are contiguous from 0 to 63, selected IDs are in range, token byte
arrays are valid, and all recorded top logit/logprob values are finite.

Both arms of every pair use the same model and prompt paths, `--metal`,
`--ctx 16384`, `--prefill-chunk 2048`, `--nothink`, and `--temp 0`. The greedy
series also uses `-n 64`. The only `DS4_` environment variable recorded for
every subprocess is `DS4_QWEN4_PREFILL_CHUNK=2048`; the runner constructs
that environment after removing inherited `DS4_` variables. Each executable
runs from its corresponding source directory. The 5,942-token resident log
confirms actual progress frontiers 2,048, 4,096, and 5,942, covering the second
chunk's attention boundary and the final 1,846-row projection tail. Logs
identify Apple M1 Max, 32 GiB, with `math_safe=off` for these CLI captures.

## Source and binary provenance

At audit time I independently hashed the actual binaries and every source
listed by the runner: 88 files for root fix, 87 for base, and 88 for before
fix. All match `results.json`. Coverage includes `ds4.c`, `ds4_cli.c`,
`ds4_metal.m`, `Makefile`, and every `.metal` shader in each arm. The runner
also checks binary and source invariants before and after every serial run.
The following are executable SHA-256 values, not Git commit IDs:

| Arm | Binary SHA-256 |
|---|---|
| Root fix | `230ea58642b1fad9e54eaca53df962a1cbc81512d2ab3c53c8f228bd62646f19` |
| Base `0aaea5a` | `bf5f0451723acbae85102ac645e5c7c21d611c25cb0bcd35d1285e514cbe2d8d` |
| Before fix | `b00a2eb96ad2f7a8a5ed745fb7563a6c8b247ca3240b66daf58b22cc40e71b37` |

I additionally verified every file in `source-snapshots-manifest.json`
against its recorded size and SHA-256: all 89 base files and all 91
before-fix files. For the base, each file's actual bytes also match
`git show 0aaea5a238fb41a35106a551e73c8409dfb751ac:<path>` independently of
the recorded hashes. The before-fix snapshot contains six backed-up files
and 85 files captured from the working tree. The six backed-up copies
still match their recorded physical backup files. This before-fix arm
represents that captured working tree, including pre-existing experiments;
it is not a pristine historical commit.

Prompt fixture SHA-256 values and model device/inode/size/mtime/ctime still
match the report. Model files are protected by identity metadata rather
than by hashing the entire large GGUF files; the runner states this limit.
The runner's own SHA-256 matches its recorded value. The source manifest
SHA-256 is `c654d601d546842a9efc176f311d3e185b586e4092334052fcb205883a8f9ac2`.
The audited final results file SHA-256 is
`b7e364d9155e20804704d76769a777292173081c644d48512626b0d1a630f220`.

## Interpretation boundaries

The resident tests use the reduced fixture assembled from repeated real
Q4 weights. Its own metadata states that it is not a quality model. These
captures establish arithmetic regression parity with main on that fixture,
including the long prefill boundary and two complete 64-step continuations.
The full Q4 SSD captures establish unchanged first-token logits on two
prompts against the captured before-fix implementation.

This does not reproduce the tester's full resident Q2 model on M4 Max, prove
all device/model/prompt combinations, or measure general generation quality.
The timing fields are individual serial process wall times with differing
cache history; they include initialization and dump I/O. They do not support
a speedup or throughput claim. No timing benefit is inferred in this audit.
