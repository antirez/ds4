# Independent audit of the staged release

The release snapshot was reconstructed from the Git index after staging
only the correction. Its completed `release-staged-ab/results.json` reports
14 successful serial CLI runs and seven passing comparisons. I independently
reread every raw capture using Python's standard library, without importing
the runner's comparison functions or executing any build or GPU process.

| Comparison | Independent result |
|---|---|
| Resident first logits, 29 tokens | 248,320 finite values per arm; zero differing FP32 words; identical JSON |
| Resident first logits, 575 tokens | 248,320 finite values per arm; zero differing FP32 words; identical JSON |
| Resident first logits, 5,942 tokens | 248,320 finite values per arm; zero differing FP32 words; identical JSON |
| Resident greedy, 29-token prompt | All 64 selected IDs and token bytes identical; complete JSON identical |
| Resident greedy, 575-token prompt | All 64 selected IDs and token bytes identical; complete JSON identical |
| Full Q4 SSD first logits, 29 tokens | 248,320 finite values per arm; zero differing FP32 words; identical JSON |
| Full Q4 SSD first logits, 575 tokens | 248,320 finite values per arm; zero differing FP32 words; identical JSON |

Resident comparisons use main `0aaea5a` as the reference and the repeated
real-weight fixture as the common model. SSD comparisons use the captured
before-fix implementation and the full Q4 GGUF. Each logits vector was
independently packed as 248,320 little-endian FP32 words. Its SHA-256 matches
the report; all raw JSON and log hashes also match. Argmax IDs match the
maximum over each full vocabulary. All greedy step indices are contiguous
from 0 to 63, IDs and byte arrays are valid, and recorded top logits and
logprobs are finite. Every dump has its expected prompt count and context
16,384. Commands use their binary's source directory as CWD and record only
`DS4_QWEN4_PREFILL_CHUNK=2048` in the cleaned DS4 environment.

## Sources actually staged

I independently hashed all 94 files listed in `release-staged-sources.json`.
Every hash matches the corresponding release snapshot file. I also compared
each file's actual bytes against `git show :<path>`: all 94 match the Git
index exactly. This check covers production sources, shaders, headers,
Makefile, and the three test sources in the manifest, including the new
resident arithmetic regression. It does not inspect or constrain the
research files under `misc/` that are added to the index separately.

Every binary and source SHA-256 recorded by the A/B runner matches the
actual files at audit time. The release arm's runner source hashes also
agree with the 94-file index manifest. The executable tested is therefore
the release snapshot's binary, rather than the earlier working-tree binary
that included separate local experiments.

Release executable SHA-256:
`78456c05cf820e6f8ef3e3f333388c22d6eff7654cd652f48d79622d1e43e7f4`.

Release results SHA-256:
`f04451af079f4b77abaac83423f8b842e24cd832c573fbec51ef2d6a4b43526d`.

Release source manifest SHA-256:
`b428f91cf7333e3e1059e47e8b456f15b38dc9ffef6e551315a5153d47f80e95`.

## Build and regression logs

`release-staged-build.log` contains compilation and linking of `ds4`,
`ds4-server`, the resident arithmetic test, the generation test, and the
memory test. It contains no warning or error diagnostics. The arithmetic
test log has four complete success messages and four Metal API Validation
startup messages. Its four command lines exercise math-safe off/on with
ordinary command buffers and forced unretained command buffers; the runtime
math flags confirm two off and two on runs. No Metal validation errors are
recorded.

`release-staged-memory.log` reports success for bounded workspace,
recurrent/MTP staging, budgets, and attention query planning. This is a
memory-planning regression result, not a measurement of process peak RAM.

These results confirm the correction as actually staged on the local M1
Max. The resident fixture is synthetic and does not replace testing the
complete Q2 model on the tester's M4 Max. The two greedy series compare
selected tokens and exported logprobs, not full-vocabulary logits at every
decode step. Individual serial wall times with differing cache history do
not establish a performance benefit; no speedup is inferred here.
