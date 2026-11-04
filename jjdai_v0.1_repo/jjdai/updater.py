"""JJ DAI v0.1 — M4 pull-by-choice champion upgrade.

The "one button" (upgrade) and its mirror (rollback), exactly as decided:

  * the network PROPOSES a champion (ChampionManifest for the node's topic);
  * the node PULLS by choice — nothing is ever pushed;
  * before installing, the node runs the candidate against its OWN PRIVATE
    held-out canaries in the sandbox (local acceptance), computes the delta
    vs the current adapter, and only then accepts;
  * rollback to the previous version is one press, local, instant;
  * every decision (accept / reject / rollback) lands in the witness chain.

Why local acceptance matters (§7.5 "pressure to game the boundary"): a poisoned
candidate can carry a perfectly VALID provenance and pass the PUBLIC gauntlet —
by hard-coding the known public canary's answer. Only canaries the attacker has
never seen catch that. Hence: the node's private set is the last line, and the
acceptance policy is NO-REGRESSION per canary, not just an aggregate score.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field

from proto import ChampionManifest, Canary, sha256
from witness import WitnessLog
import verifier


class UpgradeRejected(RuntimeError):
    pass


@dataclass
class Delta:
    current_pass: int
    candidate_pass: int
    total: int
    regressions: list = field(default_factory=list)   # canary ids: current ✓, candidate ✗
    improvements: list = field(default_factory=list)  # canary ids: current ✗, candidate ✓

    def summary(self) -> str:
        return (f"current {self.current_pass}/{self.total} -> "
                f"candidate {self.candidate_pass}/{self.total}; "
                f"regressions={self.regressions or 'none'}, "
                f"improvements={self.improvements or 'none'}")


class NodeUpdater:
    """Owns a node's adapter version stack + private canaries + upgrade logic."""

    def __init__(self, node_id: str, private_canaries: list[Canary],
                 witness_path: str, current: ChampionManifest,
                 min_tier: str = "strong"):
        self.node_id = node_id
        self.private_canaries = private_canaries   # held-out, NEVER published
        self.witness = WitnessLog(witness_path)
        self.stack: list[ChampionManifest] = [current]   # version history
        self.min_tier = min_tier

    # ---------- state ----------
    @property
    def active(self) -> ChampionManifest:
        return self.stack[-1]

    # ---------- step 1: fail-closed provenance / package integrity ----------
    def verify_manifest(self, m: ChampionManifest) -> None:
        """Refuse BEFORE any execution if the package doesn't bind together."""
        if m.delivery_class != "adapter":
            raise UpgradeRejected(
                f"delivery_class {m.delivery_class!r} not allowed in v0.1 "
                "(adapters only, DIIP Class 1)")
        with open(m.artifact_path, "rb") as f:
            got = sha256(f.read())
        if got != m.artifact_hash:
            raise UpgradeRejected(
                "artifact hash mismatch: delivered bytes do not match the "
                "manifest (tampered in transit or forged provenance)")

    # ---------- step 2: local acceptance on PRIVATE held-out canaries ----------
    def _gauntlet(self, m: ChampionManifest) -> dict[str, bool]:
        out = {}
        for c in self.private_canaries:
            v = verifier.verify(["python3", m.artifact_path], c,
                                min_tier=self.min_tier)
            out[c.id] = v.passed
        return out

    def local_delta(self, candidate: ChampionManifest) -> Delta:
        cur = self._gauntlet(self.active)
        cand = self._gauntlet(candidate)
        return Delta(
            current_pass=sum(cur.values()),
            candidate_pass=sum(cand.values()),
            total=len(self.private_canaries),
            regressions=[cid for cid in cur if cur[cid] and not cand[cid]],
            improvements=[cid for cid in cur if not cur[cid] and cand[cid]],
        )

    # ---------- the ONE BUTTON ----------
    def try_upgrade(self, candidate: ChampionManifest) -> Delta:
        """Propose -> verify package -> local gauntlet -> delta -> accept/reject.

        Policy: NO-REGRESSION per canary (a candidate may not lose any canary
        the current adapter passes) AND net non-negative delta.
        """
        try:
            self.verify_manifest(candidate)
        except UpgradeRejected as e:
            self.witness.append("UPGRADE", {
                "action": "rejected", "stage": "package",
                "candidate": candidate.champion_id, "reason": str(e)})
            raise

        delta = self.local_delta(candidate)
        if delta.regressions or delta.candidate_pass < delta.current_pass:
            self.witness.append("UPGRADE", {
                "action": "rejected", "stage": "local-acceptance",
                "candidate": candidate.champion_id, "delta": delta.summary()})
            raise UpgradeRejected(
                f"failed local acceptance on private held-out canaries: "
                f"{delta.summary()}")

        self.stack.append(candidate)
        self.witness.append("UPGRADE", {
            "action": "accepted", "candidate": candidate.champion_id,
            "delta": delta.summary(), "stack_depth": len(self.stack)})
        return delta

    # ---------- the MIRROR BUTTON ----------
    def rollback(self) -> ChampionManifest:
        if len(self.stack) < 2:
            raise UpgradeRejected("nothing to roll back to")
        dropped = self.stack.pop()
        self.witness.append("UPGRADE", {
            "action": "rollback", "from": dropped.champion_id,
            "to": self.active.champion_id, "stack_depth": len(self.stack)})
        return self.active


def make_manifest(champion_id: str, topic: str, artifact_path: str,
                  producer: str, gauntlet: dict) -> ChampionManifest:
    with open(artifact_path, "rb") as f:
        h = sha256(f.read())
    return ChampionManifest(champion_id=champion_id, topic=topic,
                            artifact_path=artifact_path, artifact_hash=h,
                            producer_node=producer, gauntlet=gauntlet)
