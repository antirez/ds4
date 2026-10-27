# Evaluation data

`ds4-eval` contains small, curated subsets of external benchmarks. They are
integration tests for DwarfStar, not official benchmark distributions or
leaderboard scores. Case IDs are retained so each prompt can be checked against
its source.

## Hard suite

The 50-case `hard` suite was assembled on 2026-09-03:

| Source | Cases | Source revision | License |
|---|---:|---|---|
| [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) | 30 | `f418b116db00b065c2aea046518d8fcf74d39872` | Apache-2.0 |
| [OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench) | 10 | `91184b52131e7fc9455fef848035173aea8cc01a` | Apache-2.0 |
| [LiveBench](https://huggingface.co/datasets/livebench/reasoning) | 5 | `6fc6498a5dfba553f69f4413feabade1f1a2d384` | Apache-2.0 |
| [NIST Juliet 1.3](https://samate.nist.gov/SARD/test-suites/112) | 5 | archive SHA-256 `ada9d7e1c323d283446df3f55bdee0d00bda1fed786785fe98764d58688f38eb` | CC0-1.0 / public domain |

Selected source IDs:

- MMLU-Pro: `86`, `671`, `899`, `1467`, `1995`, `2398`, `2826`, `3024`,
  `3546`, `3696`, `4674`, `4699`, `5078`, `5848`, `6034`, `6500`, `6844`,
  `6955`, `7738`, `7754`, `8771`, `8863`, `9055`, `9487`, `10360`, `10567`,
  `10775`, `11120`, `11292`, `11346`.
- OlympiadBench: `1606`, `1613`, `1614`, `1618`, `1697`, `1709`, `1716`,
  `1736`, `1750`, `1766`.
- LiveBench: `c833574b0ff6e8609bee19b36fabc48501cdcd9a64b99cd0422f54ada1acc53d`,
  `b7b7e331df4614775210ef5e524ed7f20c5fec6655fb0dd179c94e4d71170862`,
  `0daa7ca38beec4441b9d5c04d0b98912322926f0a3ac28a5097889d4ed83506f`,
  `9ee37b9a04ab050936c86b2c5bb7abbaa0bc0e737d59a7bff9ba11e9b4069c1d`,
  `d7071c9ff5d9779e7ab955366d0ae8db40f785aadfe7ff0b5a7ede98c05c44ea`.
- NIST Juliet reductions use the local IDs beginning with `juliet-` in
  `ds4_eval_cases.c`.

The MMLU-Pro choices and keys are unchanged. OlympiadBench cases are English,
text-only rows; only their published final answers are embedded. LiveBench's
benchmark-specific output wrapper was removed because `ds4-eval` supplies one
uniform final-answer instruction. The NIST cases are short, locally written
reductions of Juliet weakness patterns rather than copies of complete Juliet
test cases.

Multiple-choice order is retained for MMLU-Pro. Open answers declare their
grading type and, where needed, a small list of equivalent surface forms. The
12-case `hard-smoke` suite is a fixed subset of `hard`.

Apache-2.0 terms for redistributed benchmark material are included in
[`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt). DwarfStar's own source
remains under the repository's top-level MIT license.

## Core suite

The original 92-case `core` suite contains 25 GPQA Diamond, 25 SuperGPQA, 25
AIME 2025, and 17 reduced defensive code-review cases. Its existing source IDs,
ordering, prompts, and answer keys are unchanged by the hard-suite addition.
