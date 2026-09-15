"""JJ DAI v0.1 — M5 harness: retrieve() as a tool.

The RAG obligation lives HERE, not in the engine (as designed): the harness
declares the `retrieve` tool, runs the Anthropic-shaped tool loop
(tool_use -> execute -> tool_result -> continue), and records which chunks it
ACTUALLY served. Enforcement is downstream: Plane H (grounding.py) fails any
grounded answer whose citations don't verify. Engine stays dumb and fast.

Citation convention: the model cites chunks inline as [[chunk:ID]]. The harness
extracts them into metadata.citations = [{chunk_id, hash}] using the hashes of
chunks it actually served this turn — so a model citing a chunk that was never
served produces a citation with hash=None, which Plane H rejects.
"""
from __future__ import annotations
import re

from rag_store import RagStore

RETRIEVE_TOOL = {
    "name": "retrieve",
    "description": "Search the node's grounded memory for relevant chunks.",
    "input_schema": {"type": "object",
                     "properties": {"query": {"type": "string"}},
                     "required": ["query"]},
}

CITE_RE = re.compile(r"\[\[chunk:(\d+)\]\]")


class Harness:
    def __init__(self, engine, rag: RagStore, namespace: str):
        self.engine = engine
        self.rag = rag
        self.ns = namespace

    def run(self, request: dict, max_rounds: int = 4) -> dict:
        messages = list(request["messages"])
        served: dict[int, str] = {}          # chunk_id -> hash, this turn only
        for _ in range(max_rounds):
            resp = self.engine.respond(messages, tools=[RETRIEVE_TOOL])
            tool_uses = [b for b in resp["content"] if b["type"] == "tool_use"]
            if not tool_uses:
                text = " ".join(b["text"] for b in resp["content"]
                                if b["type"] == "text")
                citations = []
                for cid in map(int, CITE_RE.findall(text)):
                    citations.append({"chunk_id": cid,
                                      "hash": served.get(cid)})  # None if never served
                resp.setdefault("metadata", {})["citations"] = citations
                resp["metadata"]["served_chunks"] = sorted(served)
                return resp
            # execute tools, feed results back (Anthropic wire shape)
            messages.append({"role": "assistant", "content": resp["content"]})
            results = []
            for tu in tool_uses:
                if tu["name"] == "retrieve":
                    chunks = self.rag.retrieve(self.ns, tu["input"]["query"])
                    for c in chunks:
                        served[c["chunk_id"]] = c["hash"]
                    results.append({
                        "type": "tool_result", "tool_use_id": tu["id"],
                        "content": [{"type": "text", "text": repr(
                            [{k: c[k] for k in ("chunk_id", "text")}
                             for c in chunks])}]})
            messages.append({"role": "user", "content": results})
        raise RuntimeError("tool loop did not terminate")


# ---------- three engine behaviors the gate must face --------------------------
class GroundedMockEngine:
    """Retrieves first, then answers citing the chunks it actually got."""
    model_id = "mock-grounded"

    def __init__(self):
        self._turn = 0

    def respond(self, messages, tools):
        self._turn += 1
        if self._turn == 1:
            q = messages[-1]["content"]
            return {"content": [{"type": "tool_use", "id": "tu_1",
                                 "name": "retrieve", "input": {"query": q}}],
                    "stop_reason": "tool_use"}
        # read chunk ids out of the tool_result we were fed
        blob = str(messages[-1]["content"])
        ids = re.findall(r"'chunk_id': (\d+)", blob)
        cites = " ".join(f"[[chunk:{i}]]" for i in ids[:2])
        return {"content": [{"type": "text",
                             "text": f"Grounded answer based on memory {cites}."}],
                "stop_reason": "end_turn"}


class LazyMockEngine:
    """Answers a grounded topic from thin air: no retrieve, no citations."""
    model_id = "mock-lazy"

    def respond(self, messages, tools):
        return {"content": [{"type": "text",
                             "text": "Confident answer with no memory behind it."}],
                "stop_reason": "end_turn"}


class FabricatingMockEngine:
    """Retrieves (looks diligent) but cites a chunk that was never served."""
    model_id = "mock-fabricating"

    def __init__(self):
        self._turn = 0

    def respond(self, messages, tools):
        self._turn += 1
        if self._turn == 1:
            return {"content": [{"type": "tool_use", "id": "tu_1",
                                 "name": "retrieve",
                                 "input": {"query": "anything"}}],
                    "stop_reason": "tool_use"}
        return {"content": [{"type": "text",
                             "text": "Per your files [[chunk:99999]] this is so."}],
                "stop_reason": "end_turn"}
