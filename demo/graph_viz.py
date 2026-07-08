"""Subgraph selection + JSON serialization for the SPARDA demo's neuron-graph canvas.

``subgraph_data`` powers ``GET /api/graph``: it picks the query-relevant slice of the
VERGIL graph (fuzzy entity-link → seeds → 1-hop BFS) and returns plain ``{nodes, edges}``
JSON — the frontend (web/index.html ``renderNeuronGraph``) does all the drawing. The demo
is a presentation layer: a viz error must never take down answering (§9 Step 6), so
callers wrap this in try/except (modal_demo.py does).

(The old pyvis ``render_subgraph`` HTML path was removed with the Gradio UI — the custom
frontend renders the graph itself from this JSON.)
"""

from __future__ import annotations


def subgraph_data(pipeline, query: str, max_nodes: int = 46) -> dict:
    """Return the relevant subgraph as JSON-able {nodes, edges} for the custom neuron
    canvas renderer (the frontend draws it — this just selects + serializes).
    Selection: fuzzy entity-link (BEST match per entity via the shared cached name
    index — the old inline loop took the FIRST >70 hit and re-scanned every node per
    entity) → seeds → 1-hop BFS."""
    from engine.graph_index import link_entity

    entities = pipeline._extract_entities(query)
    G = pipeline.graph

    seeds = []
    for entity in entities:
        node = link_entity(G, str(entity))
        if node is not None:
            seeds.append(node)

    relevant = set(seeds)
    for seed in seeds:
        for nb in list(G.neighbors(seed))[:10]:
            relevant.add(nb)
    if not relevant:
        return {"nodes": [], "edges": []}

    sub = G.subgraph(list(relevant)[:max_nodes])
    seedset = set(seeds)
    nodes = [{
        "id": str(n),
        "label": (str(d.get("name", n)) or str(n))[:34],
        "type": d.get("type", "product"),
        "deg": int(sub.degree(n)),
        "seed": n in seedset,
    } for n, d in sub.nodes(data=True)]
    edges = [{"s": str(u), "t": str(v), "type": d.get("type", "")}
             for u, v, d in sub.edges(data=True)]
    return {"nodes": nodes, "edges": edges}
