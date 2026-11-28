# Inkling 2 Mini GPU parity scripts

They drive `./ds4` (run from the repository root, or from `DS4_DIR`) and need a few environment variables:

- `IK_MODEL`: path of the Inkling 2 Mini GGUF.
- `IK_VECTORS`: directory of llama.cpp reference vectors: `logits_bf16.jsonl` (one row per prompt: `name`, `prompt_ids`,
  `positions[k].forced_ids`, `positions[k].top`) and, for `gen_cmp.py` / `cpu_vs_ref.py`, `LABEL_logprobs.json`.
- `IK_TEXT_DIR`: directory of long plain-text files used as prompts (`moby.txt`, `twocities.txt`, `pride.txt`, `holmes.txt`,
  `alice.txt`).
- `IK_WORK`: scratch output directory (default `./ik-work`).
- `DS4_DIR`: directory that holds the `./ds4` binary (default `.`).
- `DS4_LOCK_FILE`: ds4's lock file (default `/tmp/ds4-inkling.lock`).

`LABEL` in the usages below (for example `bf16` or `nvfp4`) only names the output files; the model is always `IK_MODEL`.
See docs/INKLING2.md for the method.

- `par.py LABEL NAME CHUNK [--free]`: GPU vs CPU oracle with the GPU router picks replayed.
- `lout.py LABEL NAME`: per-slot `l_out` sums, GPU vs CPU with picks replayed.
- `chunk.py LABEL NAME 512,100,1,512t8`: same tokens, different chunk schedules, raw logits compared bit for bit.
- `gen_cmp.py LABEL`, `cpu_vs_ref.py LABEL NAME`: greedy generation against the 64-token llama.cpp references.
- `speed.py decode|prefill`: tokens per second through the CLI.
- `depth.py DEPTH[,DEPTH...]`: decode and prefill speed at a given context depth.
- `deep_par.py NTOK_APPROX`: GPU vs CPU oracle parity on a long prompt (tens of thousands of tokens).
- `tl_agg.py FILE`: aggregates a `DS4_METAL_ENCODER_TIMELINE` file per kernel.
- Session API self-test: `DS4_INKLING2_SESSION_TEST=1 DS4_INKLING2_FT_TOKENS=... ./ds4 -m FILE --metal --first-token-test --raw-prompt -p x`.
- `selftest.sh [GGUF]`: session self-test in default and EXACT mode (includes a known-answer check against the llama.cpp reference).
