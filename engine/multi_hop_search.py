"""Multi-hop search — graph traversal + DANTE scoring.

See SPARDA_BUILD_PLAN.md §6.4.

The traversal walks the edges that EXIST on Amazon-2023 — brand → category/feature →
similar_to — PLUS SPARDA's synthesized complement_of ("goes-with") edges (§3.3). There is
NO bought_together edge to walk (empty on 2023, §3.2); complement_of carries the
accessory/bought-with relationship and is SPARDA's edge over plain VERGIL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dante import DanteSearchEngine as DanteRetriever


def multi_hop_search(query: str, entities: list[str], G: "nx.Graph",
                     dante: "DanteRetriever", max_hops: int = 2) -> dict:
    """
    For relational queries: "accessories from Sony that work with WH-1000XM5".
    1. Entity-link query to graph nodes
    2. BFS traverse up to max_hops (brand/category/feature/similar_to/complement_of)
    3. Score discovered products with DANTE's ColBERT reranker
    4. Return with reasoning paths (graph citations — e.g. "A --[complement_of]--> B")
    """
    # Step 1: Entity linking (fuzzy match)
    from rapidfuzz import fuzz
    matched_nodes = []
    for entity in entities:
        best_match, best_score = None, 0
        for node, data in G.nodes(data=True):
            name = data.get("name", "")
            score = fuzz.partial_ratio(entity.lower(), name.lower())
            if score > best_score and score > 70:
                best_match, best_score = node, score
        if best_match:
            matched_nodes.append(best_match)

    if not matched_nodes:
        return {"discovered": [], "paths": [], "retrieval_method": "multi_hop",
                "note": "No entities matched in the graph"}

    # Step 2: BFS traversal from matched nodes
    discovered = set()
    for start in matched_nodes:
        for node, dist in nx.single_source_shortest_path_length(G, start, cutoff=max_hops).items():
            if G.nodes.get(node, {}).get("type") == "product" and node not in [n for n in matched_nodes]:
                discovered.add(node)

    # Step 3: Score discovered products with DANTE's ColBERT
    candidates = []
    for pid in list(discovered)[:100]:  # cap at 100 for reranking
        node_data = G.nodes[pid]
        candidates.append({
            "product_id": pid,
            "product_text": node_data.get("description", node_data.get("name", "")),
            "name": node_data.get("name", ""),
        })

    if candidates:
        reranked = dante.colbert.rerank(query, candidates, top_k=20)
    else:
        reranked = []

    # Step 4: Build reasoning paths
    paths = []
    for result in reranked[:10]:
        pid = result.get("product", {}).get("product_id", "")
        for start in matched_nodes:
            try:
                path = nx.shortest_path(G, start, pid)
                path_str = _describe_path(G, path)
                paths.append({"product_id": pid, "path": path_str})
            except nx.NetworkXNoPath:
                continue

    return {
        "source_entities": [G.nodes[n].get("name", n) for n in matched_nodes],
        "discovered": reranked,
        "paths": paths,
        "retrieval_method": "multi_hop",
    }


def _describe_path(G, path) -> str:
    parts = []
    for i, node in enumerate(path):
        name = G.nodes.get(node, {}).get("name", str(node))[:40]
        if i < len(path) - 1:
            edge = G.edges.get((path[i], path[i+1]), {})
            etype = edge.get("type", "related")
            parts.append(f"{name} --[{etype}]-->")
        else:
            parts.append(name)
    return " ".join(parts)
