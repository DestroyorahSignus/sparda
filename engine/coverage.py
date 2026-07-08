"""ASIN-link coverage check — which retrieval paths are even available.

The whole "graph enriches retrieval" story rests on ESCI products actually existing in the
Amazon-2023 graph (linked by ASIN). That overlap is NOT guaranteed (see RISKS R1) — so
SPARDA measures it once at startup: path availability is gated on the graph's own
properties (edges/communities), and the join rate is reported as an honest informational
metric. (A per-query re-check — strip graph paths when THIS query's retrieved products
are absent from the graph — was designed but never wired in; adding it is a routing
behavior change that needs an end-to-end eval first.)

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
    """Computed ONCE at startup.

    Path availability is gated on the GRAPH's own properties, NOT on the ESCI<->graph ASIN
    join rate. That join only matters for *opportunistic* DANTE-product -> graph-entry
    expansion; the two graph-native routes stand on their own:
      * ``multi_hop`` links the QUERY's entities to graph nodes (fuzzy match) and traverses
        VERGIL's graph — it never needs a DANTE product to be in the graph.
      * ``global`` reads the community summaries.
    An earlier version gated ``multi_hop`` on ``rate >= 0.05``; with the real 0.82% ASIN
    overlap that silently disabled multi-hop for every query (router degraded → local),
    even though VERGIL's graph has ~570K edges. ``join_rate`` is still reported HONESTLY as
    an informational metric.
    """
    asins = list(linked_db.keys())
    in_graph = sum(1 for a in asins if graph.has_node(a)) if asins else 0
    rate = (in_graph / len(asins)) if asins else 0.0
    n_edges = graph.number_of_edges()
    n_comm = graph.graph.get("num_communities", 0)
    paths = ["local"]                                  # DANTE always works
    if n_edges > 0:                                    # graph traversal is self-contained
        paths += ["multi_hop"]
    if n_comm > 0:                                     # communities were built
        paths += ["global"]
    note = (f"graph {graph.number_of_nodes():,} nodes / {n_edges:,} edges / {n_comm} "
            f"communities → paths {paths}; ESCI<->graph ASIN join {rate:.2%} "
            f"(informational — gates only opportunistic DANTE→graph expansion)")
    return CoverageReport(rate, paths, note)
