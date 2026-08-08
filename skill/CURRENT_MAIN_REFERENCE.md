# Supplemental offline notes for HY

## Important: use current main as the primary reference

The historical PRs are useful for ideas, but current main already contains
two pieces of infrastructure that are directly relevant to the proposed
DSpark experiment.

### 1. `metal_graph_verify_decode2_exact`

Current `ds4.c` has an exact N=2 target verifier used by MTP.

Its own source comment says the generic batch path is not a safe substitute
for autoregressive decode because row-wise differences in HC/MoE/output
kernels can flip future greedy tokens.  The exact verifier therefore keeps
the ordinary decode kernels and canonical cache-update order while arranging
two tokens layer-by-layer in one command stream.

Study this first.

Questions to answer:

- Which operations are kept rowwise?
- Which operations are safely shared/batched?
- Which mutable states are advanced between row 0 and row 1?
- Can the same split ("batch only stateless/safe work, preserve exact stateful
  operations") be generalized from N=2 to DSpark block size <=5?

Do not copy it wholesale into DSpark: a fully serial exact verifier is known
to be too expensive.  Use it as the correctness oracle / architectural guide.

### 2. Existing exact-row primitives

Current main also contains APIs with names such as:

- `ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor`
- `ds4_gpu_matmul_q8_0_decode_rows_exact_tensor`
- `ds4_gpu_matmul_f16_router_rows_exact_tensor`
- `metal_graph_encode_shared_rows_exact`

These are not the missing compressor solution by themselves.  They show,
however, that ds4 already has an architectural pattern for "multi-row
execution that must preserve decode arithmetic".

Before inventing a new API, inspect how these functions are implemented on
Metal and how their tests establish exactness.

A likely new primitive should match that naming/contract style rather than
introducing a DSpark-specific one-off API.

Candidate concept only:

    ds4_gpu_matmul_f16_pair_compressor_decode_rows_exact_tensor(...)

Do not implement this until the first-divergence probe confirms that the
compressor pair projection is actually the first bad operator.

## Current DSpark replay path to confirm locally

The current shape is:

    ds4_session_eval_dspark_speculative_argmax
      -> spec_frontier_snapshot
      -> metal_graph_verify_suffix_tops[_impl]
      -> determine commit_drafts
      -> spec_frontier_restore
      -> for each accepted draft:
             metal_graph_eval_token_raw_swa(...)
      -> install replay logits/state

The important performance fact is that the replay loop performs one ordinary
target decode per accepted token.

## Static question to answer before GGUF arrives

Trace `metal_graph_encode_layer_batch` from the generic DSpark verifier and
compare it with the exact N=2 verifier.

Make a table:

| Stage | generic suffix verifier | exact N=2 verifier | stateful? | exact required? |
|---|---|---|---|---|
| embedding / HC | | | | |
| attention norm | | | | |
| Q/KV projections | | | | |
| raw KV store | | | | |
| compressor pair projection | | | | |
| compressor update | | | | |
| indexer compressor | | | | |
| attention output | | | | |
| router | | | | |
| routed MoE | | | | |
| shared expert | | | | |
| HC update | | | | |
| output head | | | | |

This table is a better precursor to implementation than guessing the kernel.

## Do not confuse "exact row infrastructure exists" with "DSpark is fixed"

The current DSpark generic verifier still uses `metal_graph_encode_layer_batch`.
Accepted DSpark tokens are still replayed through ordinary decode for strict
greedy identity.

The existence of exact-row helpers elsewhere is useful because it gives us
reference code and conventions, not because replay has already been removed.
