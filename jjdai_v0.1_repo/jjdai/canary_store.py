"""JJ DAI v0.1 — M1 canary store.

Canaries live in an EVAL store that the verifier reads. They must NEVER share a
path with a RAG namespace the model retrieves context from — otherwise the
canary is open-book and the model can just look up the answer (whitepaper §6).

This module refuses to load if the eval path overlaps the RAG path, turning that
invariant into a hard, testable guard instead of a comment.
"""
from __future__ import annotations
import json
import os
import glob

from proto import Canary


class EvalRagOverlapError(RuntimeError):
    pass


def _resolve(p: str) -> str:
    return os.path.realpath(os.path.abspath(p))


def _overlaps(a: str, b: str) -> bool:
    a, b = _resolve(a), _resolve(b)
    return a == b or a.startswith(b + os.sep) or b.startswith(a + os.sep)


class CanaryStore:
    def __init__(self, eval_dir: str, rag_dir: str):
        if _overlaps(eval_dir, rag_dir):
            raise EvalRagOverlapError(
                f"eval store {eval_dir!r} overlaps RAG store {rag_dir!r}: "
                "canaries would become open-book. Separate them.")
        self.eval_dir = _resolve(eval_dir)
        self.rag_dir = _resolve(rag_dir)
        os.makedirs(self.eval_dir, exist_ok=True)

    def add(self, c: Canary) -> None:
        path = os.path.join(self.eval_dir, f"{c.id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(c.__dict__, f, ensure_ascii=False, indent=2)

    def all(self) -> list[Canary]:
        out = []
        for p in sorted(glob.glob(os.path.join(self.eval_dir, "*.json"))):
            with open(p, encoding="utf-8") as f:
                out.append(Canary(**json.load(f)))
        return out
