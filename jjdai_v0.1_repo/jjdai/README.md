# JJ DAI v0.1 — Reference Prototype (M1–M5 complete)

All milestones of the v0.1 spec are implemented and proven by executable
acceptance tests: sandbox + execution canary (M1), witness chain + external
anchor (M2), two-node federation + router with per-topic champions (M3, folding
in M0), pull-by-choice champion upgrade with rollback (M4), and RAG-as-tool +
the Plane H grounding gate (M5). 40 acceptance checks green, 3 honestly skipped
(they require a strong isolation host).

First runnable code of the JJ DAI stack. It implements the verification loop from
whitepaper §7 — **verify the artifact by executing it**, instead of re-running the
model — and the witness layer from §13: a hash-chained, externally anchorable log.

## Quick start

```bash
cd jjdai_m1
python3 demo.py             # M1: verification loop (dev tier: weak)
python3 test_isolation.py   # M1: adversarial isolation tests
python3 test_witness.py     # M2: chain + anchor, adversarial
python3 test_federation.py  # M3: two-node federation + router
python3 test_upgrade.py     # M4: pull-by-choice upgrade + rollback
python3 test_grounding.py   # M5: RAG-as-tool + Plane H grounding gate
```

Expected: `good.py` passes the canary, `bad.py` goes red, an eval-store overlapping
the RAG store is rejected, and every verdict lands in a verifiable witness chain.

---

## M1 — Sandbox + Execution Canary

### Why
A verifier must not re-run the model to check the work (that would cost as much as
producing it). Instead, the producer returns an **artifact** (code, a script); the
verifier executes it in a deterministic, isolated sandbox and compares the output
hash against a canary with a known-correct answer. Verification cost collapses to
one sandbox run — the cost asymmetry of §7.1.

### Isolation tiers (fail-closed)

| Tier | Provided by | Guarantees |
|---|---|---|
| **strong** | Linux: `bubblewrap` (unshare-all, cap-drop, ro-rootfs, private tmpfs) **+ seccomp** (libseccomp); macOS: `sandbox-exec` deny-by-default + no-net | network / filesystem / syscall filter |
| **medium** | `bubblewrap` without seccomp | namespaces, no syscall filter |
| **weak** | rlimits + output cap only | resources only — NO network/fs isolation |

**Fail-closed:** `run(..., min_tier="strong")` (the default) **refuses to execute**
if the host cannot meet the tier, rather than silently running untrusted code
without isolation. A dev box must explicitly request `min_tier="weak"`.

To reach **strong** on Linux:
```bash
sudo apt-get install bubblewrap        # namespaces
pip install pyseccomp                   # syscall filter (otherwise tier = medium)
```

### Always-on limits (every tier)
CPU time, address space, file size (RLIMIT_FSIZE), no core dumps, **no-new-privs**,
wall-clock timeout, and an **output cap** enforced in the parent (contains an
output bomb in any tier). A fork bomb is contained only on **strong** (pids
cgroup) — which is exactly why fail-closed is the default.

### Canary hygiene
Canaries live in an **eval store** read by the verifier. They must never share a
path with a RAG namespace the model retrieves context from — otherwise the canary
becomes open-book. This invariant is a hard guard in `canary_store.py`
(raises `EvalRagOverlapError`), not a comment.

---

## M2 — Witness Chain + External Anchor

### Chain
Every record carries the `prev_hash` of the previous one. Two record kinds are
interleaved in ONE chain per node:

* **INFER** — provenance attestation of the config that produced an answer
  (node, model, quant, adapters, seed). Trust via **replication**.
* **SANDBOX** — replayable execution proof (env fingerprint, in/out hashes,
  exit code). Trust via **replay**.

`verify_chain()` re-hashes every record and walks the chain offline;
`replay(seq)` returns any record after verifying the chain up to it.
The witness **observes and never gates** (§12.1): appends are fire-and-forget
on the hot path; auditing is offline.

### What the chain alone catches — and what it cannot
Proven by the test suite (10/10):

| Attack | Bare chain | Anchor |
|---|---|---|
| Tamper with any record | ✅ caught (hash mismatch) | — |
| Reorder records | ✅ caught (seq/prev break) | — |
| Truncate the tail | ❌ passes (prefix is honest) | ✅ caught (anchored seq missing) |
| Full consistent rewrite | ❌ passes (internally consistent lie) | ✅ caught (head mismatch) |

That division is the point: **chain = internal integrity; anchor = the past
cannot be rewritten.**

### Anchor backends (pluggable port)
* `LocalAnchor` — dev/tests. Proves the mechanism, provides **zero external
  trust** (same operator controls both files).
* `OpenTimestampsAnchor` — production: `pip install opentimestamps-client`;
  commits the log head into Bitcoin via public calendar servers, free.
  Complete/verify later with `ots upgrade` / `ots verify`.
* RFC-3161 TSA — seam prepared (signed TimeStampToken from a public TSA).

Anchoring is **periodic** (every N records), not per-record: one anchor
transitively commits to the entire prefix, and the hot path never waits on the
network. Trade-off: the tail after the latest anchor is protected by the chain
only, until the next anchor fires.

---

## M3 — Two-Node Federation + Router (folds in M0)

### Node daemon (M0)
`NodeDaemon` wraps an `Engine` behind the narrow waist: a `/v1/messages`-shaped
`handle(request) -> response`. Every response carries `metadata.provenance`
(node, model, quant, adapters, policy version, seed) and a `witness_head`
pointer; every answer emits an INFER record into the node's own hash chain.

**Engine seam, honestly:** in production the engine is DwarfStar (or any runtime)
behind `/v1/messages`. In this repo `MockSpecialistEngine` stands in — there is no
live model here. Everything above the seam (daemon, router, registry, witness) is
the real logic and is what the tests exercise.

### Champion registry — personal, per-topic
`ChampionRegistry` builds a per-TOPIC leaderboard from VERIFIED verdicts. Every
entry records HOW it was verified: `execution` (sandbox canary, the strong
signal) or `replication` (N independent nodes agreed — NL topics with no
executable ground truth). Scores are Laplace-smoothed pass rates, so one lucky
pass does not beat a long verified record. There is deliberately **no global
champion**: a single throne recreates monoculture; per-topic keeps diversity.

### Router — explicit policy, requester choice
The router classifies the WHOLE request by an explicit, versioned rule table
(`topic-policy-v0.1`) — coarse-grained and auditable, unlike the token-level
learned gating inside the MoE model (both exist, nested, at different scales).
It proposes the per-topic champion; the requester may override, and the override
is recorded in `metadata.routing`. Unknown topics fall back explicitly with a
stated reason — never a silent guess. Seam: swap the rule table for an embedding
classifier (semantic-router) later; bump the policy version when you do.

### Proven by tests (9/9)
- legal query → legal node; code query → code node (RU keywords included)
- provenance + witness pointer present and correct in every response
- same registry picks DIFFERENT champions for different topics (personal champion)
- requester override honored and recorded
- code-topic scores come from REAL sandbox verdicts (M1 verifier), not fiction
- both nodes' witness chains verify after traffic

---

## M4 — Pull-by-Choice Champion Upgrade (+ one-press rollback)

### The one button
The network **proposes** a champion (`ChampionManifest`); the node **pulls by
choice** — nothing is ever pushed. Pressing "upgrade" runs a fixed pipeline:

1. **Package integrity, fail-closed** — the manifest's artifact hash must match
   the delivered bytes. A tampered package is rejected *before any execution*.
2. **Local acceptance** — the candidate runs against the node's OWN **private
   held-out canaries** in the sandbox (canaries the network has never seen).
3. **Delta** — pass rates of current vs candidate, per-canary regressions and
   improvements, shown to the operator.
4. **No-regression policy** — the candidate may not lose ANY canary the current
   adapter passes; otherwise rejected with the delta as the reason.
5. **Accept** pushes onto a local version stack; **rollback** is the mirror
   button — one press back to the previous version, instant, local.

Every decision (accepted / rejected / rollback) lands in the node's witness
chain as an `UPGRADE` record.

### Why local acceptance is the point (proven, not asserted)
The test suite implements the literal §7.5 Goodhart attack: a poisoned candidate
that **hard-codes the answer to the public gauntlet canary** (visible to any
attacker) and is wrong everywhere else. Its provenance is genuinely VALID — the
manifest hash matches its bytes — and it genuinely passes the public gauntlet.
The node's private held-out canaries catch it, and it is never installed. That
is the ТЗ acceptance criterion, demonstrated by execution:

| Candidate | Provenance | Public gauntlet | Private held-out | Installed |
|---|---|---|---|---|
| tampered package | ✗ hash mismatch | — (never executed) | — | ✗ fail-closed |
| poisoned (Goodhart) | ✓ valid | ✓ passes | ✗ regressions | ✗ rejected |
| good candidate | ✓ valid | ✓ passes | ✓ +1 canary, no regressions | ✓ accepted |

Delivery in v0.1 is **adapters only** (DIIP Class 1, decision #2): instant to
apply, instant to roll back. Full-base swaps are Class 2 and out of scope.

---

## M5 — RAG-as-Tool + Plane H Grounding Gate

### Where the RAG obligation lives (and where it doesn't)
The engine stays dumb and fast: `retrieve()` is a TOOL declared by the harness
(`harness.py`), which runs the Anthropic-shaped tool loop (tool_use → execute →
tool_result → continue) and records which chunks it ACTUALLY served this turn.
Enforcement is downstream in the verifier: **Plane H** (`grounding.py`) fails
any grounded-topic answer whose citations don't verify. Obligation in the
harness, enforcement in the verifier, engine untouched — as designed.

### Memory as checkable provenance
`rag_store.py` is a SQLite chunk store (decision #6) where every namespace
commits to its chunks via a REAL Merkle tree. `retrieve()` returns chunks WITH
inclusion proofs. A citation is valid only if the chunk (1) exists, (2) was
actually served this turn with a matching hash, and (3) its Merkle inclusion
proof verifies against the CURRENT namespace root. Memory stops being
trust-me text and becomes auditable provenance.

**Honest seam:** retrieval scoring is naive token overlap; production swaps in
sqlite-vec + a real embedder. Ranking quality is not what M5 proves —
cryptographic checkability of citations is, and that part is real.

### Proven against behaviors, not just the happy path (9/9)
| Engine behavior | Outcome |
|---|---|
| diligent: retrieves, cites served chunks | ✅ passes; Merkle proofs verify |
| lazy: answers grounded topic with no citations | ❌ fails (the ТЗ criterion) |
| fabricating: retrieves, then cites a never-served chunk | ❌ fails |
| store tamper: silent chunk edit after citing | caught by auditor re-hash |
| store tamper: consistent rewrite (text+hash) | shifts the Merkle root — old commitments die |
| non-grounded topic, no citations | ✅ passes (gate is scoped, not global) |
| RAG store placed inside the eval store | rejected (no open-book canaries) |

Every gate decision lands in the witness chain as a GROUNDING record.

---

## Files

| File | Role |
|---|---|
| `proto.py` | schemas: SandboxTrace (tier, truncated), Canary, Verdict + hashing |
| `sandbox.py` | tiered runner: detect_tier, fail-closed, isolation wrap, capped I/O |
| `seccomp_policy.py` | BPF denylist for bwrap `--seccomp` (Linux strong tier) |
| `canary_store.py` | eval store with a hard eval/RAG-overlap guard |
| `verifier.py` | runs a candidate through a canary in the sandbox; emits verdict |
| `witness.py` | **M2**: hash-chained append-only log, verify_chain, replay |
| `anchor.py` | **M2**: pluggable external anchor (Local / OpenTimestamps / RFC-3161) |
| `witness_stub.py` | adapter: emit_sandbox / emit_infer → chained log (verifier surface unchanged) |
| `demo.py` | M1 acceptance scenario |
| `test_isolation.py` | M1: adversarial isolation tests (escapes are contained) |
| `test_witness.py` | M2: acceptance — tamper / reorder / truncate / rewrite are caught |
| `node.py` | **M3/M0**: NodeDaemon + Engine seam (MockSpecialistEngine stand-in) |
| `registry.py` | **M3**: per-topic champion leaderboard from verified verdicts |
| `router.py` | **M3**: explicit versioned topic policy, personal champion, requester choice |
| `test_federation.py` | **M3**: acceptance — routing, provenance, per-topic champions |
| `updater.py` | **M4**: pull-by-choice upgrade, local acceptance, no-regression, rollback |
| `test_upgrade.py` | **M4**: acceptance — Goodhart candidate with valid provenance is rejected |
| `rag_store.py` | **M5**: SQLite chunk store, Merkle commitments + inclusion proofs, retrieve() |
| `harness.py` | **M5**: retrieve() tool, Anthropic-shaped tool loop, citation extraction |
| `grounding.py` | **M5**: Plane H — cryptographic citation verification, scoped by topic |
| `test_grounding.py` | **M5**: acceptance — lazy & fabricating answers fail, tampering caught |

## Verified in-container (weak tier)
- ✅ broken artifact reddens the canary (out_hash comparison); eval/RAG overlap rejected
- ✅ fail-closed triggers; mem / wall / output / fsize bombs contained; deterministic out_hash
- ✅ witness: 10/10 — chain intact, replay works, tamper & reorder break the chain,
  truncation & consistent rewrite are caught **by the anchor**
- ⏭ network egress / host-fs read / fork bomb: tests present, auto-skip on weak,
  green on a strong host

## v0.1 exit criteria vs. status
From the spec: *2+ heterogeneous nodes; requests routed to a personal champion;
responses carry provenance; executable tasks checked by a sandbox canary;
witness log replayable; nodes accept-or-reject champion upgrades with rollback.*
All demonstrated in-process with mock engines. What separates this prototype
from a live v0.1 deployment:

- Replace `MockSpecialistEngine` with DwarfStar behind `/v1/messages` (the seam
  is one class); add the thin HTTP transport around `NodeDaemon.handle()`
- Run `test_isolation.py` on a strong host (bubblewrap + pyseccomp) — clears the 3 skips
- Switch `LocalAnchor` → `OpenTimestampsAnchor` (network required)
- Port the security boundary (sandbox / verifier) to Rust per the spec
- Swap naive retrieval for sqlite-vec + real embeddings; rotate private canaries
- Producer signatures on ChampionManifest (provenance chain across nodes)
