"""SPARDA query router — the project's core IP.

Heuristic-first (fast, free, deterministic) query classification with an LLM-classify
fallback for ambiguous cases, plus an ASIN-coverage guard so a query is never routed to a
path the data cannot serve. The decision is RETURNED as a :class:`RouteDecision` so the
pipeline can log it and the demo can show "why this path."

See SPARDA_BUILD_PLAN.md §6.1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class RouteDecision:
    """The routing decision, surfaced to the demo and logged for every query."""

    route: str                       # 'local' | 'global' | 'multi_hop'
    method: str                      # 'heuristic' | 'llm_fallback' | 'degraded'
    confidence: float                # 0-1 (heuristic match strength or LLM logprob proxy)
    matched_rule: str = ""           # which keyword/regex fired (for the demo)
    reason: str = ""                 # human-readable, shown in the UI
    available_paths: list = field(default_factory=list)  # from ASIN-coverage check


ROUTER_CLASSIFY_PROMPT = """Classify this e-commerce query into exactly ONE category:
- local: find/recommend specific products ("best wireless headphones under $300")
- global: market overview, comparison of categories/ecosystems, trends ("compare smart home ecosystems")
- multi_hop: relational — accessories/compatibility/goes-with tied to a named product or brand
            ("accessories from Sony that work with the WH-1000XM5")
Reply with ONLY one word: local, global, or multi_hop.
Query: "{query}"
Category:"""


class QueryRouter:
    """
    Route queries to the optimal retrieval strategy.
    Heuristic-first (fast, free, deterministic); LLM-classify fallback for ambiguous
    cases; ASIN-coverage check so we never route to a path the data can't serve.
    The decision is RETURNED (RouteDecision) so the pipeline can log it and the demo
    can show "why this path."
    """

    # Word-boundary anchored: bare substrings misrouted real queries — "vs" is a
    # substring of "tvs", so "best TVs under $500" went to global. \b-anchored
    # stems (trend, compatib, accessor, ecosystem) keep the old inflection reach
    # (trending, compatibility, accessory, ecosystems) without cross-word hits.
    GLOBAL_KEYWORDS = [
        r"\bcompar", r"\bvs\b", r"\bversus\b", r"\btrend", r"\boverview", r"\bpopular",
        r"best brands", r"\bmarket", r"\blandscape", r"\becosystem",
        r"how do .* compare", r"across brands", r"which brand", r"what brands",
    ]
    MULTI_HOP_KEYWORDS = [
        r"works with", r"\bcompatib", r"\baccessor", r"from same brand",
        r"bought together", r"pair with", r"goes with", r"along with",
        r"similar to .* but", r"alternative.*from", r"buy (together )?with",
        r"that fit", r"that go with",
    ]
    # Below this heuristic confidence we ask the LLM instead of trusting a weak keyword hit.
    LLM_FALLBACK_THRESHOLD = 0.5

    # Compiled once at import — _heuristic used to re-parse every pattern per query.
    _GLOBAL_RES = [re.compile(kw) for kw in GLOBAL_KEYWORDS]
    _MULTI_HOP_RES = [re.compile(kw) for kw in MULTI_HOP_KEYWORDS]

    def __init__(self, llm=None):
        self.llm = llm  # shared LLM handle; if None, fall back to heuristic-only

    def _heuristic(self, query: str) -> RouteDecision:
        q = query.lower()
        for rx in self._MULTI_HOP_RES:
            if rx.search(q):
                return RouteDecision("multi_hop", "heuristic", 0.85, rx.pattern,
                                     f"matched relational pattern '{rx.pattern}'")
        for rx in self._GLOBAL_RES:
            if rx.search(q):
                return RouteDecision("global", "heuristic", 0.8, rx.pattern,
                                     f"matched market/overview pattern '{rx.pattern}'")
        # No strong signal → weak 'local' default, low confidence triggers LLM fallback
        return RouteDecision("local", "heuristic", 0.4, "",
                             "no relational/market keyword — defaulting to product search")

    def _llm_classify(self, query: str) -> RouteDecision:
        raw = self.llm.generate(ROUTER_CLASSIFY_PROMPT.format(query=query),
                                max_tokens=4, temperature=0.0).strip().lower()
        route = next((r for r in ("multi_hop", "global", "local") if r in raw), "local")
        return RouteDecision(route, "llm_fallback", 0.7, "",
                             f"LLM classified ambiguous query as '{route}'")

    def classify(self, query: str, coverage: "CoverageReport | None" = None) -> RouteDecision:
        """
        Returns a RouteDecision (not just a string).
        1. Heuristic first. 2. If low confidence and an LLM is available, LLM-classify.
        3. Degrade gracefully against the ASIN-coverage check: if the chosen path's data
           is unavailable for this query, fall back to 'local' (DANTE always works).
        """
        decision = self._heuristic(query)
        if decision.confidence < self.LLM_FALLBACK_THRESHOLD and self.llm is not None:
            decision = self._llm_classify(query)

        # Graceful degradation: graph-dependent paths require ASIN coverage.
        if coverage is not None:
            decision.available_paths = coverage.available_paths
            if decision.route in ("global", "multi_hop") and decision.route not in coverage.available_paths:
                decision = RouteDecision(
                    "local", "degraded", decision.confidence, decision.matched_rule,
                    f"'{decision.route}' unavailable (graph coverage too low for this query) "
                    f"→ degraded to local search", coverage.available_paths,
                )
        return decision
