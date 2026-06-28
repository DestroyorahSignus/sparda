"""ASIN-link coverage check — which retrieval paths are even available.

The whole "graph enriches retrieval" story rests on ESCI products actually existing in the
Amazon-2023 graph (linked by ASIN). That overlap is NOT guaranteed (see RISKS R1) — so
SPARDA measures it once at startup and re-checks per query. If a query's retrieved products
aren't in the graph, the graph-dependent paths are simply not offered for that query, and
the router degrades to local search.

See SPARDA_BUILD_PLAN.md §6.0.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoverageReport:
    join_rate: float          # global: fraction of ESCI ASINs present in the graph
    available_paths: list     # subset of ['local','global','multi_hop'] usable right now
    note: str = ""


def global_coverage(linked_db: dict, graph) -> CoverageReport:
    """Computed ONCE at startup from data/link_datasets.py output."""
    asins = list(linked_db.keys())
    if not asins:
        return CoverageReport(0.0, ["local"], "no linked products; graph paths disabled")
    in_graph = sum(1 for a in asins if graph.has_node(a))
    rate = in_graph / len(asins)
    paths = ["local"]                                  # DANTE always works
    if rate >= 0.05:                                   # graph has *some* coverage
        paths += ["multi_hop"]
    if graph.graph.get("num_communities", 0) > 0:      # communities were built
        paths += ["global"]
    note = (f"ASIN join rate {rate:.1%}; "
            + ("graph paths enabled" if rate >= 0.05
               else "graph too sparse — local-only, graph used only for opportunistic expansion"))
    return CoverageReport(rate, paths, note)


def query_coverage(retrieved_ids: list[str], graph, base: CoverageReport) -> CoverageReport:
    """Per-query: are *these* products in the graph? Controls whether expansion runs."""
    if not retrieved_ids:
        return base
    hits = sum(1 for pid in retrieved_ids if graph.has_node(pid))
    if hits == 0:
        # nothing this query touched is in the graph → strip graph paths for this query
        return CoverageReport(base.join_rate, ["local"],
                              "retrieved products absent from graph; local-only this query")
    return base
