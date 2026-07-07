"""Top-level SPARDA pipeline: query → route → retrieve → generate → unified answer.

SPARDA returns ONE unified answer schema regardless of which path ran, so the demo, the
eval harness, and any downstream caller see the same shape: ``answer``, the
``RouteDecision``, a flat list of ``citations`` (product/community/path provenance), and
the raw ``context``. The pipeline also caches by ``(query, route)`` and logs every routing
decision.

See SPARDA_BUILD_PLAN.md §6.7.
"""

from __future__ import annotations

import logging

from engine.coverage import global_coverage, query_coverage  # noqa: F401 (query_coverage re-exported)
from engine.router import QueryRouter, RouteDecision
from engine.local_search import local_search
from engine.global_search import global_search
from engine.multi_hop_search import multi_hop_search
from engine.prompts import (
    LOCAL_PROMPT, GLOBAL_PROMPT, MULTI_HOP_PROMPT, ENTITY_EXTRACTION_PROMPT,
)

log = logging.getLogger("sparda")


class SpardaPipeline:
    def __init__(self, dante, vergil_graph, llm, communities, summary_embs, encoder,
                 linked_db=None):
        self.dante = dante
        self.graph = vergil_graph
        self.llm = llm                       # ONE shared Qwen instance (§6.5)
        self.communities = communities
        self.summary_embs = summary_embs
        self.encoder = encoder
        self.router = QueryRouter(llm=llm)    # router shares the same LLM for fallback
        # ASIN-link coverage computed ONCE at startup (§6.0)
        self.coverage = global_coverage(linked_db or self.dante.product_db, self.graph)
        log.info("ASIN join rate: %.1f%% | available paths: %s",
                 self.coverage.join_rate * 100, self.coverage.available_paths)
        self._cache: dict[tuple, dict] = {}

    def answer(self, query: str) -> dict:
        # 0. Routing (heuristic → LLM fallback → coverage-aware degrade), logged + cacheable
        decision = self.router.classify(query, coverage=self.coverage)
        cache_key = (query.strip().lower(), decision.route)
        if cache_key in self._cache:
            return self._cache[cache_key]
        log.info("ROUTE q=%r → %s (%s, conf=%.2f) %s",
                 query, decision.route, decision.method, decision.confidence, decision.reason)

        citations: list[dict] = []

        if decision.route == "local":
            context = local_search(query, self.dante, self.graph, self.dante.product_db)
            citations = self._cite_products(context["dante_results"]) + \
                        self._cite_graph(context["graph_context"])
            prompt = LOCAL_PROMPT.format(
                dante_results=self._fmt_products(context["dante_results"]),
                graph_context=self._fmt_graph(context["graph_context"]),
                query=query,
            )
        elif decision.route == "global":
            context = global_search(query, self.communities, self.summary_embs, self.encoder)
            citations = self._cite_communities(context["communities"])
            prompt = GLOBAL_PROMPT.format(
                community_summaries=self._fmt_communities(context["communities"]),
                query=query,
            )
        else:  # multi_hop
            entities = self._extract_entities(query)
            context = multi_hop_search(query, entities, self.graph, self.dante)
            # Graceful fallback: entity-linking found nothing → fall back to local search
            if not context.get("discovered"):
                log.info("multi_hop entity-link miss → falling back to local search")
                decision = RouteDecision("local", "degraded", decision.confidence,
                                         reason="multi-hop found no graph entities → local")
                return self._run_local(query, decision)
            citations = self._cite_products(context["discovered"]) + \
                        self._cite_paths(context["paths"])
            prompt = MULTI_HOP_PROMPT.format(
                source_entities=context["source_entities"],
                discovered_products=self._fmt_products(context["discovered"]),
                paths=self._fmt_paths(context["paths"]),
                query=query,
            )

        answer_text = self.llm.generate(prompt, max_tokens=600)

        result = {                                   # ── UNIFIED ANSWER SCHEMA ──
            "answer": answer_text,
            "route": decision.route,
            "routing": decision.__dict__,            # method, confidence, reason, available_paths
            "citations": citations,                  # flat, typed provenance list
            "join_rate": self.coverage.join_rate,
            "context": context,                      # raw, for debugging/eval
        }
        self._cache[cache_key] = result
        return result

    def _run_local(self, query, decision):
        context = local_search(query, self.dante, self.graph, self.dante.product_db)
        prompt = LOCAL_PROMPT.format(
            dante_results=self._fmt_products(context["dante_results"]),
            graph_context=self._fmt_graph(context["graph_context"]), query=query)
        return {
            "answer": self.llm.generate(prompt, max_tokens=600),
            "route": "local", "routing": decision.__dict__,
            "citations": self._cite_products(context["dante_results"]),
            "join_rate": self.coverage.join_rate, "context": context,
        }

    # ── citation builders (uniform {type, id, name, evidence}) ──
    def _cite_products(self, results):
        cites = []
        for r in results[:10]:
            p = r.get("product", r)
            cites.append({"type": "product", "id": p.get("product_id", ""),
                          "name": p.get("name", "?"), "evidence": "retrieved/reranked"})
        return cites

    def _cite_graph(self, graph_ctx):
        cites = []
        for ctx in graph_ctx[:5]:
            for etype, prods in ctx.get("related", {}).items():
                for p in prods[:3]:
                    cites.append({"type": "graph_edge", "id": p["id"], "name": p["name"],
                                  "evidence": f"{ctx['source_name']} --[{etype}]--> {p['name']}"})
        return cites

    def _cite_communities(self, communities):
        return [{"type": "community", "id": c["community_id"],
                 "name": f"cluster {c['community_id']}",
                 "evidence": f"{c['num_products']} products; brands {c['key_brands']}"}
                for c in communities[:5]]

    def _cite_paths(self, paths):
        return [{"type": "path", "id": p["product_id"], "name": p["product_id"],
                 "evidence": p["path"]} for p in paths[:10]]

    def _extract_entities(self, query: str) -> list[str]:
        """Always returns a flat list[str]. The LLM may emit a JSON list of strings, a list
        of dicts (e.g. [{"brand":"Sony"}]), a {"entities":[...]} object, or prose-wrapped
        JSON — all get flattened to strings (multi_hop_search does entity.lower(), so a dict
        here was crashing every multi-hop query)."""
        import json
        import re

        resp = self.llm.generate(ENTITY_EXTRACTION_PROMPT.format(query=query),
                                  max_tokens=100, temperature=0.0)

        def _flatten(obj) -> list[str]:
            if isinstance(obj, str):
                return [obj.strip()] if obj.strip() else []
            if isinstance(obj, dict):
                return [s for v in obj.values() for s in _flatten(v)]
            if isinstance(obj, (list, tuple)):
                return [s for v in obj for s in _flatten(v)]
            return []

        ents: list[str] = []
        try:
            m = re.search(r"\[.*\]|\{.*\}", resp, re.DOTALL)  # tolerate prose/markdown wrap
            ents = _flatten(json.loads(m.group(0)) if m else resp)
        except Exception:
            ents = []
        # dedupe (case-insensitive), keep order, cap
        seen, out = set(), []
        for e in ents:
            e = str(e).strip()
            if e and e.lower() not in seen:
                seen.add(e.lower())
                out.append(e)
        # fallback to salient query tokens if extraction yielded nothing
        return out[:10] or [w for w in query.split() if len(w) > 3]

    def _fmt_products(self, results):
        # format top 10 products as numbered list
        lines = []
        for i, r in enumerate(results[:10], 1):
            name = r.get("name") or r.get("product", {}).get("name", "?")
            lines.append(f"{i}. {name}")
        return "\n".join(lines)

    def _fmt_graph(self, graph_ctx):
        lines = []
        for ctx in graph_ctx[:3]:
            lines.append(f"\n{ctx['source_name']}:")
            for etype, products in ctx.get("related", {}).items():
                names = ", ".join(p["name"][:40] for p in products[:3])
                lines.append(f"  [{etype}]: {names}")
        return "\n".join(lines)

    def _fmt_communities(self, communities):
        return "\n\n".join(
            f"Cluster {c['community_id']} ({c['num_products']} products): {c['summary']}"
            for c in communities[:5]
        )

    def _fmt_paths(self, paths):
        return "\n".join(f"- {p['path']}" for p in paths[:10])
