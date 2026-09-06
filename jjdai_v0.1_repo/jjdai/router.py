"""JJ DAI v0.1 — M3 router.

Routes a WHOLE request to a WHOLE specialist node (coarse-grained, explicit,
auditable) — deliberately unlike the token-level learned gating inside the MoE
model. Three properties from the design conversation:

  * explicit, VERSIONED topic policy — a rule table you can read and diff,
    not an opaque learned gate (seam left for semantic-router when rules
    stop being enough; the policy version travels in provenance);
  * PERSONAL champion — the target is the per-topic champion from verified
    verdicts (decision #4), never one global throne;
  * REQUESTER CHOICE — the router proposes, the requester may override; the
    override and the reason are recorded in the routing decision.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from registry import ChampionRegistry

POLICY_VER = "topic-policy-v0.1"

# Explicit, auditable, versioned rule table. RU+EN keywords per topic.
# Seam: replace/augment with an embedding classifier (semantic-router) later;
# keep the version string moving when you do.
TOPIC_RULES: dict[str, list[str]] = {
    "legal": ["договор", "аренд", "закон", "юрид", "суд", "иск", "устав",
              "contract", "legal", "law", "clause", "liability", "sha", "poa"],
    "code": ["код", "функци", "багу", "скрипт", "python", "rust", "compile",
             "code", "function", "bug", "script", "api", "debug", "sort"],
    "energy": ["котел", "котёл", "boiler", "энерг", "тепло", "scada", "кв",
               "energy", "heat", "turbine", "grid"],
}


@dataclass
class RouteDecision:
    node_id: str
    topic: str
    policy_ver: str
    reason: str
    overridden: bool = False
    leaderboard: list = field(default_factory=list)


class Router:
    def __init__(self, registry: ChampionRegistry, nodes: dict,
                 default_node: str | None = None):
        """nodes: node_id -> NodeDaemon; default_node: fallback for unknown topics."""
        self.registry = registry
        self.nodes = nodes
        self.default_node = default_node or next(iter(nodes))

    # ---------- classification (explicit policy) ----------
    @staticmethod
    def classify(text: str) -> tuple[str, str]:
        low = text.lower()
        scores = {t: sum(1 for kw in kws if kw in low)
                  for t, kws in TOPIC_RULES.items()}
        topic, hits = max(scores.items(), key=lambda kv: kv[1])
        if hits == 0:
            return "general", "no rule matched"
        return topic, f"{hits} keyword hit(s)"

    # ---------- routing ----------
    def route(self, request: dict, override_node: str | None = None) -> RouteDecision:
        last = request["messages"][-1]["content"]
        if isinstance(last, list):
            last = " ".join(b.get("text", "") for b in last if isinstance(b, dict))
        topic, why = self.classify(last)

        if override_node:  # requester choice beats the proposal
            if override_node not in self.nodes:
                raise KeyError(f"override node {override_node!r} unknown")
            return RouteDecision(node_id=override_node, topic=topic,
                                 policy_ver=POLICY_VER, overridden=True,
                                 reason=f"requester override ({why})",
                                 leaderboard=self.registry.leaderboard(topic))

        champ = self.registry.champion(topic)
        if champ is None or champ not in self.nodes:
            return RouteDecision(node_id=self.default_node, topic=topic,
                                 policy_ver=POLICY_VER,
                                 reason=f"no verified champion for topic ({why}); fallback",
                                 leaderboard=[])
        return RouteDecision(node_id=champ, topic=topic, policy_ver=POLICY_VER,
                             reason=f"per-topic champion ({why})",
                             leaderboard=self.registry.leaderboard(topic))

    def dispatch(self, request: dict, override_node: str | None = None) -> tuple[dict, RouteDecision]:
        d = self.route(request, override_node)
        node = self.nodes[d.node_id]
        node.route_policy_ver = d.policy_ver   # travels into provenance
        resp = node.handle(request)
        resp["metadata"]["routing"] = {
            "topic": d.topic, "reason": d.reason, "policy_ver": d.policy_ver,
            "overridden": d.overridden,
        }
        return resp, d
