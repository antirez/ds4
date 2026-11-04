"""JJ DAI v0.1 — M3 champion registry.

Per-TOPIC leaderboard built from VERIFIED verdicts (decision #4: champions are
measured per topic, never as one global throne — a global champion recreates
monoculture; per-topic keeps diversity, whitepaper §11).

Every entry records HOW it was verified:
  * kind="execution"   — sandbox canary (M1), the strong signal;
  * kind="replication" — N independent nodes agreed (NL topics, where no
                         executable ground truth exists).
Champion score = Laplace-smoothed pass rate; smoothing keeps a node with one
lucky pass from beating a node with a long verified record.
"""
from __future__ import annotations
import json
import os
from collections import defaultdict


class ChampionRegistry:
    def __init__(self, path: str | None = None):
        self.path = path
        # stats[topic][node_id] = {"passed": int, "total": int, "kinds": {...}}
        self.stats: dict = defaultdict(lambda: defaultdict(
            lambda: {"passed": 0, "total": 0, "kinds": defaultdict(int)}))
        if path and os.path.exists(path):
            self._load()

    # ---------- persistence ----------
    def _load(self):
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    self._apply(e)

    def _apply(self, e: dict):
        s = self.stats[e["topic"]][e["node_id"]]
        s["total"] += 1
        s["passed"] += 1 if e["passed"] else 0
        s["kinds"][e["kind"]] += 1

    # ---------- write path ----------
    def record(self, node_id: str, topic: str, passed: bool,
               kind: str = "execution") -> None:
        e = {"node_id": node_id, "topic": topic, "passed": bool(passed),
             "kind": kind}
        self._apply(e)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(e, sort_keys=True) + "\n")

    # ---------- read path ----------
    @staticmethod
    def _score(s: dict) -> float:
        # Laplace smoothing: (passed+1)/(total+2)
        return (s["passed"] + 1) / (s["total"] + 2)

    def leaderboard(self, topic: str) -> list[tuple[str, float, int]]:
        rows = [(nid, self._score(s), s["total"])
                for nid, s in self.stats.get(topic, {}).items()]
        return sorted(rows, key=lambda r: (-r[1], -r[2], r[0]))

    def champion(self, topic: str) -> str | None:
        lb = self.leaderboard(topic)
        return lb[0][0] if lb else None
