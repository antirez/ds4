# Prompt-lookup speculative drafting — prototype results

Branch `prompt-lookup-draft` (from main). Drafts the greedy continuation from the
session's own token history (most recent earlier occurrence of the last 4 tokens,
up to 8 proposed tokens) and verifies it with the existing MTP batch-verify /
rollback machinery (`metal_graph_verify_suffix_tops` + `spec_frontier_*`).
No new GPU kernels; no second model. Env-gated `DS4_PROMPT_LOOKUP_DRAFT=1`,
greedy/temp-0 only. M5 Max, 128 GiB, `ds4flash.gguf`.

## Comparison (same probes, same machine, temp 0)

Two prompt-lookup shapes were measured: the initial prototype (MTP-style cycle:
sequential anchor eval + verify pass) and the **fused** one-pass cycle (anchor
joins the verify batch — possible because lookup drafts need only the token
history, not the hidden state).  Fused with default depth 7 (anchor + 7 drafts =
a batch-8 pass, staying on the small-batch mat-vec kernels — see below).

| workload | baseline t/s | lookup (prototype) | **lookup (fused)** | MTP (`--mtp-draft 2`) |
|---|---:|---:|---:|---:|
| copy (repeat a 492-word passage) | 38.20 | 79.10 (2.07×) | **87.37 (2.29×)** | 40.50 (1.06×) |
| code edit (rename a fn, reproduce file) | 38.19 | 63.47 (1.66×) | **71.55 (1.87×)** | 40.56 (1.06×) |
| prose essay (novel text) | 38.22 | 38.16 (1.00×) | 38.59 (1.01×) | 37.65 (0.99×) |
| story recall @30K ctx | 29.84 | 29.18 (1.00×) | 29.33 (1.00×) | 31.76 (1.06×) |

Fused acceptance: copy 97.3% (6.67 tok/pass), code 91.5% (6.32 tok/pass);
prose/recall zero matches with zero measurable overhead (scan 0.02–0.51 ms total
per run); fused output **byte-identical to the same greedy decode loop on all four**; `--long-context` gate OK.

**The batch-8 cliff (measured engine fact):** the batched verify pass costs ~80 ms
at n_tokens ≤ 8 but ~167 ms at 9 — the small-batch mat-vec kernels cover 2..8
tokens (`metal/dense.metal:789`); batch 9+ falls onto the full prefill matmul
path.  Hence the depth-7 default (`DS4_PROMPT_LOOKUP_MAX`, 1..15, tunable).
Depth 15 on the copy probe: 13.88 tok/pass but only 69.6 t/s — the slow-path pass
cost eats the extra depth, validating 7 as the default.  A batch-8 pass costs
≈2.3 decode passes, which is also why fusion is worth ~10–13%, not the 2× a
weights-only cost model predicts.

Acceptance metrics:

| workload | lookup | MTP |
|---|---|---|
| copy | 100% accept, 8.00 tok/pass (28 passes) | 95% accept, 2.90 incl. anchor/pass |
| code edit | 87.1% accept, 6.88 tok/pass (49 passes) | 95% accept, 2.90/pass |
| prose | 0 matches (255/255 no-match) | 67% accept, 2.34/pass |
| story recall | 0 matches (63/63 no-match) | 100% accept, 3.00/pass |

Output identity vs non-speculative baseline (byte-for-byte):
- **prompt-lookup: identical on all four** — including through a real edit in the
  code probe (the drafter misses at the renamed identifiers and re-locks after),
  and including partial-accept recommits.
- MTP: identical on three; **prose differs** — the documented near-tie
  batched-verify caveat of the shipped path, observed in practice.

Overhead when lookup cannot help: zero measurable (38.22→38.16 prose; scan cost
≈18 µs per attempt even at 30K context, 1.16 ms total over 63 tokens).

## Reading

1. **The website's MTP claim is confirmed empirically**: ~1.06× at best. The cause
   is structural, visible in the code: every MTP cycle pays a full sequential eval
   (the drafter needs the backbone hidden state) *plus* a verify pass — tokens per
   weight-stream ≤ (1+2)/2 at production depth 2 — and draft cost, snapshot blits,
   and per-position verify work eat most of that.
2. **Prompt-lookup beats shipped MTP by ~8–16× margin-of-improvement on the
   engine's own target workloads** (coding agents, structured/repetitive output):
   1.66–2.07× vs 1.06×, with no download, no resident drafter, ~150 lines of host
   code, and output byte-identical to the same greedy decode loop.
3. **They are complementary**: MTP's hidden-state drafter works on novel text
   (prose 2.34/pass, recall 3.00/pass) where lookup finds nothing. A hybrid
   (lookup when a context match exists, MTP otherwise) is a natural follow-up.
4. **The fused one-pass shape is implemented** and is the version of record
   (+10–13% over the prototype; the bigger predicted gain was capped by the
   batch-pass cost above).  Remaining headroom: longest-match-first n-gram
   selection (acceptance points on code edits), an MTP hybrid for the no-match
   residue (+~6% on novel text), and — as kernel work, out of scope — a 9..16
   token fast-batch path to push past the batch-8 cliff.

## Caveats / scope

- Greedy/temp-0 only (as scoped); sampling callers unaffected.
- Engages only via the CLI session path (`run_sampled_generation`); the
  `DS4_PROMPT_LOOKUP_DRAFT` env also routes the one-shot greedy CLI there.
  Server/agent not wired yet.
- Inherits the batched-verifier near-tie caveat in principle (same contract as
  `--mtp`); not observed in any lookup run (4/4 byte-identical to the same loop).
- Fixed `min_ngram=4`; depth default 7 (`DS4_PROMPT_LOOKUP_MAX` to tune); no
  incremental index yet (full backward scan measured cheap: ~18 µs @30K).
- Anchor-only misses (match found but the target rejects the first draft) cost
  one wasted batch pass; measured rare (3 of 65 attempts on the code probe).
- `--long-context` regression gate passes with the feature enabled.
