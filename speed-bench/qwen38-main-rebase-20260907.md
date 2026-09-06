# Qwen3.8 rebase validation — 2026-09-07

Rebased `qwen3.8-flash-next` from `8f7c4fa65457c1f7d097a8fd5882e8e30c41575e`
onto main `c0a6119f363ef82125877142f13fb3fe491cba14` on an Apple M3 Ultra
with 512 GiB RAM. A local backup remains at
`backup/qwen38-before-main-rebase-8f7c4fa`.

## Integration fixes

- Preserve main's text-only multimodal handling and SSD expert-address binding
  while retaining Qwen support.
- Teach the new tool-state tracker Qwen's function/parameter syntax. Without
  this integration, valid calls were treated as orphan endings and the visible
  tool-turn KV checkpoint was not saved.
- Use main's common speculative-boundary rollback for Qwen too; remove the
  duplicate retained-token counter and the superseded extra Qwen rewind.
- Apply delimiter-specific tool-argument unescaping and the current agent
  tool contracts to Qwen. Advertise `view_image` when vision is loaded.
- Correct two test controls: fused MoE kernels need not populate unused gate/up
  scratch, and rewind comparisons must preserve the prefill/decode split.
  The original MoE failure reproduced on untouched main; the original IQ2
  rewind failure reproduced on the pre-rebase branch. Numeric tolerances were
  not widened. MTP rewind compares restored output with its original verifier
  output, while replay compares with an independent fresh session.
- Fix the snapshot diagnostic's integer format and replace obsolete Qwen test
  target documentation. `DS4_TEST_ARGS` can select model-compatible groups;
  an unset value preserves the complete default `make test` behavior.

## Completed checks

- Clean CPU and Metal builds; final builds have no compiler warnings.
- `make test DS4_TEST_ARGS="--server --session-snapshot --session-rewind --metal-kernels"`
  with the IQ2 model and matching PLE sidecar. All other `make test` recipes
  run normally, including sampling, agent, CLI arguments, session payloads,
  TP command fixtures, layer placement, dot products, and image checks.
- Qwen kernels, GLM KDA/attention, MXFP4 kernels, SSD cache, SSD expert mapping
  and eviction, and corrected MoE prefill controls.
- JPEG decoder, vision comparison metrics, GLM reference rendering and
  embedding comparison, DSpark window/EOS contracts, Qwen pack conversion,
  GLM quantization, and Qwen acceptance/benchmark Python tests.
- Server and agent unit tests under AddressSanitizer and UndefinedBehaviorSanitizer;
  terminal PTY and JSON schema checks for DSML, GLM, and Qwen.
- IQ2 snapshots and rewind with MTP disabled and enabled; the MTP snapshot
  check covered 16 cycles, including accepted two-token cycles.
- Q4 live tool continuations, streaming and non-streaming, with and without
  MTP: four KV hits per run. A real client executed `pwd` and fetched
  `https://example.com`, with two successful cached tool continuations.
- IQ2 and Q4 API vision: 6/6 fixture checks each. IQ2 CLI vision: two images
  and two text follow-ups completed in both ordinary and MTP modes.
- IQ2 ordinary/MTP greedy comparison: byte-identical 256-token output;
  120/135 drafts accepted (88.9%).
- DeepSeek Vision Exp MXFP4 resident versus SSD streaming with an 8 GB expert
  cache: both completed and returned byte-identical output for the short
  greedy smoke prompt.
- Native Q4 agent: executed shell commands, wrote a fresh marker file, fetched
  the GitHub repository API over HTTPS (HTTP 200), and parsed the result.
  Artifact modification times were checked against this run's start time.

## Performance

Same model, PLE sidecar, text, 1,024-token prefill chunks, 65,793-token allocated
context, and 128 generated tokens as the earlier IQ2 measurement:

| Context | Previous prefill tok/s | Rebased prefill tok/s | Previous decode tok/s | Rebased decode tok/s |
|---|---:|---:|---:|---:|
| 32,768 | 968.57 | 968.86 | 47.93 | 47.78 |
| 65,536 | 953.97 | 954.07 | 47.53 | 47.42 |

These are single sanity runs, not a statistical benchmark. Decode changed by
less than 0.4%. They do not establish physical 64 GB machine compatibility.

## Scope

Raw local commands, output, control-build evidence, and CSVs are under
`OUT/qwen38-main-rebase/`; API vision details are under
`OUT/qwen38-vision-quality/`. Model processes ran sequentially.

CUDA, ROCm, physical multi-machine TCP/RDMA, and M5-only kernels were not run.
DeepSeek 0731 official/local-golden vectors and checkpoint-matched DSpark tests
were not run because this host has Vision Exp targets instead. The full
checkpoint-specific default suite and full model-quality evaluation campaigns
are not claimed as passed by this rebase check.
