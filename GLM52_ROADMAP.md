# GLM-5.2 Roadmap

This branch is `glm-5.2`. It adds GLM-5.2 support and keeps the public path focused on correctness first, then parity and speed.

## Current status

- GLM-5.2 prompt/vocab handling is on the branch.
- Metal GLM MLA attention, KV norm/copy, router selection, top-eight MoE slots, and SSD streaming cache policy are on the branch.
- The GLM-5.2 IQ2XXS SSD-streaming fast decode collapse is fixed on Metal.
- The fix was validated with:
  - `./ds4_test --metal-kernels` -> `gtest: ok`
  - GLM-5.2 IQ2XXS SSD-streaming multi-token smoke -> coherent planet list output.
- The latest Metal fix depends on `ds4.c`, `ds4_gpu.h`, `ds4_metal.m`, `metal/flash_attn.metal`, `metal/norm.metal`, and `tests/ds4_test.c`.

## Immediate release gates

Run these from a clean checkout of `origin/glm-5.2` on a Metal machine with the GLM-5.2 IQ2XXS GGUF available:

```sh
make ds4 ds4_test ds4_agent_test
./ds4_agent_test
./ds4_test --metal-kernels
./ds4 -m /path/to/GLM-5.2-IQ2XXS-w2Q2K-Q8rest-ds4.gguf \
  --ssd-streaming --nothink --temp 0 --system '' \
  -p 'List the planets in order. Answer in one short line.' -n 64
```

Expected smoke output is a coherent one-line planet list. A one-token answer is not enough; the original failure appeared at token 2+.

## Next work

1. **Clean branch verification**
   - Rebuild from a clean checkout, not a dirty development tree.
   - Record exact model path, quant, backend, command, and output.
   - Confirm no Metal source override environment variables are set during release checks.

2. **Quality parity**
   - Compare Metal fast path against CPU/reference for first-token logits and decode token 2+.
   - Cover short no-think prompts, multi-token factual prompts, and long-prompt decode.
   - Keep diagnostics that localize drift by layer; remove or gate one-off probes.

3. **SSD streaming robustness**
   - Exercise low cache budgets, eviction, repeated prompts, and long prompt + decode.
   - Verify missing-expert load paths, selected-expert residency, and fallback behavior.
   - Keep streaming cache policy explicit and observable.

4. **CUDA/ROCm parity**
   - Bring GLM router and selected-expert paths into backend parity.
   - Treat CUDA and ROCm as separate validation tracks; do not infer correctness from Metal.
   - Add backend-specific notes when hardware is unavailable.

5. **Quantizer/tooling**
   - Finish `gguf-tools/glm-quantize` and template generation as a separate tooling pass.
   - Verify generated GGUF tensor names, tensor shapes, metadata, and loader compatibility.
   - Keep generated binaries out of commits unless they are intentionally tracked release artifacts.

6. **Performance pass**
   - Profile MLA attention, KV norm/copy, MoE slots8, and SSD expert loading.
   - Optimize only after parity checks are stable.
   - Track prefill and decode speed before/after each optimization.

## Resume checklist

- Fetch and checkout `origin/glm-5.2`.
- Read `AGENT.md` first for project rules.
- Read this file for branch state and next gates.
- Start with the release gates above before editing.
- Keep public roadmap notes in tracked markdown files; keep machine-local setup and private agent instructions out of the repository.
