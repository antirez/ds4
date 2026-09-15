# JJ DAI — Logic Block 1

**A verified-inference node: a native inference engine + the JJ DAI trust layer.**

This repository combines two things:

1. **The engine** (repo root) — a fork of [antirez/ds4 (DwarfStar)](https://github.com/antirez/ds4):
   a native C inference engine for DeepSeek V4 Flash/PRO (Metal / CUDA / ROCm),
   with on-disk KV-cache sessions, native tool-calling, and an
   OpenAI/Anthropic-compatible server (`/v1/chat/completions`, `/v1/responses`,
   `/v1/messages`). Engine documentation: see upstream README and `AGENT.md`.

2. **The trust layer** (`jjdai/`) — the JJ DAI v0.1 reference prototype:
   everything that turns a bare inference engine into a **verifiable network
   node**. Sandbox, execution canaries, hash-chained witness log with external
   anchoring, federation router with per-topic champions, pull-by-choice
   champion upgrades with rollback, and a cryptographic grounding gate for RAG.

The two are deliberately decoupled at a **narrow waist**: the engine is a
vendored "muscle" behind the `/v1/messages` wire contract and is never modified;
the trust layer treats any engine speaking that contract as interchangeable
(DwarfStar today, other runtimes — e.g. dedicated inference ASICs — tomorrow).

Whitepaper & manifesto: [jj-dai.org](https://jj-dai.org)

---

## Architecture

```
            ┌─────────────────────────── JJ DAI NODE ───────────────────────────┐
 network ⇆ L7 Registry/Updater ─ L5 Router ─ L1 Node Daemon ─ L6 RAG ─ L0 Engine
                 │                    │            │  harness    (tool)  (DwarfStar)
            L4 Witness ◀──────────────┴──── L3 Verifier ── L2 Sandbox ─────┘
                 (append-only, hash-chained, externally anchored)
```

Two execution environments, two trust modes:
- **inference** emits a provenance *attestation* into the witness chain —
  trusted via **replication** across independent nodes;
- **sandbox** emits a replayable execution *proof* — trusted via **replay**
  (anyone can re-run the artifact and contest the hash).

## Repository layout

| Path | What it is |
|---|---|
| `/` (root) | DwarfStar engine (vendored fork; not modified by the trust layer) |
| `jjdai/` | JJ DAI v0.1 trust-layer prototype (Python; security boundary to be ported to Rust) |
| `jjdai/README.md` | Detailed milestone-by-milestone documentation (M1–M5) |

## Quick start (trust layer)

```bash
cd jjdai
python3 demo.py             # M1: sandbox + execution canary loop
python3 test_isolation.py   # M1: adversarial isolation tests
python3 test_witness.py     # M2: hash chain + external anchor
python3 test_federation.py  # M3: two-node federation + router
python3 test_upgrade.py     # M4: pull-by-choice upgrade + rollback
python3 test_grounding.py   # M5: RAG-as-tool + Plane H grounding gate
```

No dependencies beyond the Python 3 standard library. For the **strong**
isolation tier on Linux: `apt install bubblewrap` + `pip install pyseccomp`.

## Status: v0.1 milestones

| Milestone | Scope | Status |
|---|---|---|
| M0/M3 | Node daemon, provenance in every response, 2-node federation, per-topic champion router, requester choice | ✅ 9/9 tests |
| M1 | Deterministic sandbox (fail-closed isolation tiers), execution canaries, eval/RAG separation guard | ✅ 6/6 (+3 skips needing a strong host) |
| M2 | Hash-chained witness log; tamper/reorder caught by the chain; truncation/rewrite caught by the external anchor (OpenTimestamps seam) | ✅ 10/10 |
| M4 | Pull-by-choice champion upgrade: package integrity fail-closed, local acceptance on private held-out canaries, no-regression policy, one-press rollback | ✅ 6/6 (incl. a literal Goodhart attack rejected) |
| M5 | `retrieve()` as a harness tool; Merkle-committed memory; grounding gate fails uncited or fabricated answers | ✅ 9/9 |

## Honest limits (read before relying on this)

- Engines in the prototype are **mocks** behind the `Engine` seam; wiring the
  live DwarfStar server in is the next step (one class + thin HTTP transport).
- The sandbox's **strong** tier requires bubblewrap+seccomp (Linux) or
  Seatbelt (macOS); in weaker environments it fail-closes rather than pretending.
- `LocalAnchor` proves the anchoring *mechanism* only; a live node must switch
  to OpenTimestamps / RFC-3161.
- Retrieval ranking is naive token overlap (seam for sqlite-vec + embeddings);
  what M5 actually proves is cryptographic checkability of citations.
- Champion manifests bind artifact bytes by hash but are not yet
  producer-signed.

## License / lineage

Engine: upstream DwarfStar license (see root). Trust layer (`jjdai/`): part of
the JJ DAI project. Design lineage: verification · execution sandbox · routing ·
witness — see the whitepaper at [jj-dai.org](https://jj-dai.org).
