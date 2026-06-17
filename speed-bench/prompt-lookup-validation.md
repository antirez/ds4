# Prompt-lookup speculative decoding — falsification / validation report

Adversarial validation pass before any PR, per review. Branch
`prompt-lookup-draft` (commits `3f6e619` prototype, `da611f6` fused). M5 Max,
128 GiB, `ds4flash.gguf`, greedy temp-0, fused shape, default depth 7.
Every number below is from this session; identity = `cmp` byte equality of the
generated output, on vs off, same prompt/flags.

## 1. Workload measurements (off → on)

| workload | off t/s | on t/s | speedup | hit rate / accept | tok/pass | scan total | identity |
|---|---:|---:|---:|---|---:|---:|---|
| **agent turn** (explain fix + corrected file) | 37.82 | 55.57 | **1.47×** | 54/152 cycles matched; 87.6% accept | 6.08 | 0.05 ms | IDENTICAL |
| copy (repeat 492-word passage) | 37.37 | 83.63 | **2.24×** | 33/36 matched; 97.3% | 6.67 | 0.00 ms | IDENTICAL |
| code edit (rename fn, reproduce file) | 36.40 | 60.52 | **1.66×** | 56/65 matched; 91.5% | 6.32 | 0.02 ms | IDENTICAL |
| prose essay | 37.08 | 36.47 | 0.98× (noise) | 0/256 matched | — | 0.01 ms | IDENTICAL |
| story recall @30K ctx | 29.99 | 29.55 | 0.99× (noise) | 0/64 matched | — | 0.49 ms | IDENTICAL |
| random unique-token dump @9K ctx | 31.10 | 31.36 | 1.01× | 0/53 matched | — | 0.11 ms | IDENTICAL |

(The earlier same-day fused runs measured copy 2.29× and code 1.87×; ratios vary
±10% run-to-run with thermals — claims below use the conservative end.)

## 2. Depth ablation (code probe; justifies default 7)

| `DS4_PROMPT_LOOKUP_MAX` | gen t/s | tok/pass | identity | note |
|---:|---:|---:|---|---|
| 4 | 58.19 | 3.73 | IDENTICAL | more passes, shallower commits |
| **7 (default)** | 60.52 | 6.32 | IDENTICAL | anchor+7 = batch-8, fast-path ceiling |
| 8 | **40.58** | 6.88 | IDENTICAL | batch-9 → falls off the small-batch kernel cliff |
| 15 | 61.96 | 13.92 | IDENTICAL | deep slow-path ≈ shallow fast-path; no win |

The cliff: the batched verify costs ~80 ms at ≤8 tokens and ~167 ms at 9+
(`metal/dense.metal:789` small-batch mat-vec kernels cover 2..8 tokens).

## 3. Correctness stress

| stress | result |
|---|---|
| EOS right after an accepting stretch (repeat-3×-then-stop) | 1.46× and IDENTICAL; proposals truncate at EOS, no post-EOS commits |
| partial accepts (code probe: 3 anchor-only misses, multiple partial passes) | IDENTICAL |
| ambiguous repeated n-grams (same 4-gram → different continuations) | output IDENTICAL and correct; see worst case below |
| context wall (ctx 1100, output to the wall) | speculation byte-exact; a final-token ±1 vs the no-env run is a **pre-existing CLI route difference**, see §4 |
| sub-wall same probe (`-n 350`), and no-wall same length (`ctx 4096 -n 600`) | IDENTICAL |
| `--long-context` (env on) | OK |
| `--long-context-hard` (env on) | OK |

## 4. Findings from the falsification pass

**(a) Known bounded worst case — adversarial ambiguity (~0.8× on a tiny output).**
The ambiguous-n-gram probe (≈20 generated tokens) ran 2 misfiring verify passes
(28.6% accept): ~160 ms of wasted pass time dominates a 0.5 s decode → measured
0.80×. Output remains byte-identical to the same greedy decode loop, and correct. On realistic text the misfire
rate is low (3/65 cycles on the code probe); a miss-backoff (suspend lookup for N
tokens after a low-accept pass) is the known future mitigation. Not a blocker;
documented.

**(b) The wall "divergence" is not in the feature — root-caused and exonerated.**
At a context wall the on/off outputs differed by exactly the final token
(2390/2391 bytes identical). Discriminators:
- same prompt, same `-n 600`, no wall (ctx 4096): **IDENTICAL** → not mid-stream
  numerics;
- prose probe at a wall with **zero speculation possible** (182/182 no-match, 0
  verify passes): outputs **still differ by the same final-token ±1** → the cause
  cannot be prompt-lookup.

Cause: ds4's two CLI decode loops clamp the context wall differently — the
default greedy one-shot path (`ds4_engine_generate_argmax`) emits one final token
more than the session loop (`run_sampled_generation`, `max_tokens = room − 1`).
Enabling any session-loop feature — **including the existing `--mtp`** — selects
the second loop and inherits the same ±1 today. Within a given loop, prompt-lookup
is byte-identical to the same greedy decode loop in every test above. (Aligning the two loops' wall clamps would
be an upstream consistency fix, deliberately out of scope for this feature.)

## 5. Supported claim (PR wording)

> Env-gated prompt-lookup speculative decoding for greedy generation
> (`DS4_PROMPT_LOOKUP_DRAFT=1`): drafts the continuation from the session's own
> token history and verifies it with the existing speculative batch verifier,
> committing only target-agreed tokens. On copy-heavy / code-style / agent
> workloads it improves decode throughput by ~1.5–2.2×; on prose/recall/no-match
> workloads it falls back with negligible overhead (µs-scale scans, ~1.0×).
> Output is byte-identical to the same decode loop in all validation workloads;
> a known adversarial repetition pattern can cost up to ~20% on very short
> outputs (documented; backoff planned).

Not claimed: any speedup on novel text; sampling support; anything about the
pre-existing wall-clamp difference between the two CLI loops.

## 6. Reproduction

Probes: `/tmp/{copy,code,agent,prose,random,eos,ambig}_probe.txt` (generated by
the scripts in this session; copy/code/agent are deterministic renders),
`tests/long_context_story_prompt.txt`. Run shape:

```
./ds4 --prompt-file <probe> --ctx <C> -n <N> --temp 0 --nothink            # off
DS4_PROMPT_LOOKUP_DRAFT=1 ./ds4 --prompt-file <probe> ... (same)           # on
DS4_PROMPT_LOOKUP_DRAFT=1 ./ds4_test --long-context                        # gate
```
Per-session counters print at session close; `DS4_PROMPT_LOOKUP_LOG=1` adds
per-pass lines; `DS4_PROMPT_LOOKUP_MAX=N` tunes depth (1..15, default 7).
