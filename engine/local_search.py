"""Local search — DANTE retrieval + VERGIL graph expansion.

See SPARDA_BUILD_PLAN.md §6.2.

`dante` is a ``dante.DanteSearchEngine`` instance (≡ ``DanteRetriever``, §2.1); `G` is the
VERGIL product graph (``networkx.Graph``). Both are passed in — neither library is imported
here, so this module parses without dante/vergil installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, not imported at runtime
    import networkx as nx
    from dante import DanteSearchEngine as DanteRetriever


def local_search(query: str, dante: "DanteRetriever", G: "nx.Graph",
                 product_db: dict, top_k: int = 20) -> dict:
    """
    1. DANTE retrieves top-K via 4-signal fusion + ColBERT rerank
    2. VERGIL expands each result with graph context (Amazon-2023 has NO co-purchase
       edges — §3.2 — so expansion walks the edges that actually exist):
       - complement_of products (ESCI-Complement "goes-with"/accessories — §3.3,
         SPARDA-only; this is what replaces the missing bought_together signal)
       - same-brand alternatives (has_brand)
       - shared-feature products (has_feature)
       - similar products (similar_to, cross-brand)
    3. Return products + graph context for the LLM
    """
    # DANTE retrieval
    dante_results = dante.search(query, top_k=top_k)

    # VERGIL graph expansion for each result
    graph_context = []
    for result in dante_results[:5]:  # expand top 5 only
        pid = result.get("product_id") or result.get("product", {}).get("product_id")
        if pid and G.has_node(pid):
            neighbors = {}
            for neighbor in G.neighbors(pid):
                edge = G.edges[pid, neighbor]
                etype = edge.get("type", "related")
                ndata = G.nodes[neighbor]
                if ndata.get("type") == "product":
                    neighbors.setdefault(etype, []).append({
                        "id": neighbor,
                        "name": ndata.get("name", ""),
                        "edge_type": etype,
                    })
            if neighbors:
                graph_context.append({
                    "source_product": pid,
                    "source_name": G.nodes[pid].get("name", ""),
                    "related": neighbors,
                })

    return {
        "dante_results": dante_results,
        "graph_context": graph_context,
        "retrieval_method": "local",
    }
