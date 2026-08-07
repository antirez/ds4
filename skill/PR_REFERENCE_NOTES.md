# Historical PR references

PR #590 — Replay-free partial accepts + encoder-batched captures
Key concepts: per-depth prefix checkpoints; direct checkpoint commit; inline Metal compute copies.
Do not port wholesale because later greedy-identity work showed non-canonical verifier frontier state.

PR #659 — Greedy-identity correctness fix
Key concept: remove direct commit of batch-verifier state; replay accepted tokens through ordinary
single-token decode to restore canonical compressor/KV trajectory.

PR #677 — Byte-exact Metal verifier prototype
Key lesson: exactness is attainable, but broad decode-order-exact verification was too slow.
Use it as evidence for a narrow exact-operator approach, not as a template to copy wholesale.
