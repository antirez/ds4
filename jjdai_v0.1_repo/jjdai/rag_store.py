"""JJ DAI v0.1 — M5 RAG store.

SQLite-backed chunk store (decision #6: sqlite as the embedded, serverless
store) with a REAL Merkle tree per namespace:

  * every chunk is content-hashed; the namespace commits to all chunks via a
    Merkle root; retrieve() returns chunks WITH inclusion proofs;
  * Plane H later verifies citations against these proofs — memory becomes
    checkable provenance, not trust-me text ("data-not-weights" made auditable);
  * the store enforces the eval/RAG separation guard from M1: it refuses to
    live inside (or contain) the canary eval store — otherwise canaries become
    open-book.

HONEST SEAM: retrieval scoring here is naive token overlap. In production swap
in sqlite-vec + a real embedder; the interface (retrieve -> chunks+proofs) and
all Merkle machinery stay unchanged. Ranking quality is not what M5 proves —
cryptographic checkability of citations is.
"""
from __future__ import annotations
import os
import re
import sqlite3

from proto import sha256_text
from canary_store import _overlaps, EvalRagOverlapError


def _tok(s: str) -> set[str]:
    return set(re.findall(r"[\w\u0400-\u04FF]{3,}", s.lower()))


class RagStore:
    def __init__(self, db_path: str, eval_dir: str | None = None):
        if eval_dir and _overlaps(os.path.dirname(os.path.abspath(db_path)) or ".",
                                  eval_dir):
            raise EvalRagOverlapError(
                f"RAG store {db_path!r} overlaps eval store {eval_dir!r}: "
                "canaries would become open-book. Separate them.")
        self.db = sqlite3.connect(db_path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            " id INTEGER PRIMARY KEY, ns TEXT, doc_id TEXT,"
            " text TEXT, hash TEXT)")
        self.db.commit()

    # ---------- write ----------
    def add(self, ns: str, doc_id: str, text: str) -> tuple[int, str]:
        h = sha256_text(text)
        cur = self.db.execute(
            "INSERT INTO chunks (ns, doc_id, text, hash) VALUES (?,?,?,?)",
            (ns, doc_id, text, h))
        self.db.commit()
        return cur.lastrowid, h

    # ---------- merkle ----------
    def _leaves(self, ns: str) -> list[tuple[int, str]]:
        return list(self.db.execute(
            "SELECT id, hash FROM chunks WHERE ns=? ORDER BY id", (ns,)))

    @staticmethod
    def _parent(a: str, b: str) -> str:
        return sha256_text(a + b)

    def merkle_root(self, ns: str) -> str:
        level = [h for _, h in self._leaves(ns)]
        if not level:
            return sha256_text("")
        while len(level) > 1:
            if len(level) % 2:
                level.append(level[-1])
            level = [self._parent(level[i], level[i + 1])
                     for i in range(0, len(level), 2)]
        return level[0]

    def prove(self, ns: str, chunk_id: int) -> dict | None:
        """Inclusion proof: sibling path from the chunk's leaf to the root."""
        leaves = self._leaves(ns)
        idx = next((i for i, (cid, _) in enumerate(leaves) if cid == chunk_id),
                   None)
        if idx is None:
            return None
        level = [h for _, h in leaves]
        path = []
        while len(level) > 1:
            if len(level) % 2:
                level.append(level[-1])
            sib = idx ^ 1
            path.append((level[sib], "L" if sib < idx else "R"))
            level = [self._parent(level[i], level[i + 1])
                     for i in range(0, len(level), 2)]
            idx //= 2
        return {"chunk_id": chunk_id, "leaf": leaves[
            next(i for i, (cid, _) in enumerate(leaves) if cid == chunk_id)][1],
            "path": path, "root": level[0]}

    @staticmethod
    def verify_proof(proof: dict) -> bool:
        h = proof["leaf"]
        for sib, side in proof["path"]:
            h = RagStore._parent(sib, h) if side == "L" else RagStore._parent(h, sib)
        return h == proof["root"]

    # ---------- read ----------
    def get(self, chunk_id: int) -> dict | None:
        row = self.db.execute(
            "SELECT id, ns, doc_id, text, hash FROM chunks WHERE id=?",
            (chunk_id,)).fetchone()
        if not row:
            return None
        return dict(zip(("id", "ns", "doc_id", "text", "hash"), row))

    def retrieve(self, ns: str, query: str, k: int = 3) -> list[dict]:
        """Top-k chunks by naive token overlap, each with an inclusion proof."""
        q = _tok(query)
        scored = []
        for cid, doc, text, h in self.db.execute(
                "SELECT id, doc_id, text, hash FROM chunks WHERE ns=?", (ns,)):
            s = len(q & _tok(text))
            if s:
                scored.append((s, cid, doc, text, h))
        scored.sort(key=lambda r: (-r[0], r[1]))
        out = []
        for s, cid, doc, text, h in scored[:k]:
            out.append({"chunk_id": cid, "doc_id": doc, "text": text,
                        "hash": h, "score": s,
                        "proof": self.prove(ns, cid)})
        return out
