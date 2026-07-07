"""Pyvis interactive subgraph rendering for the SPARDA demo.

See SPARDA_BUILD_PLAN.md §7.

The ``pyvis`` import is deferred into ``render_subgraph`` so this module parses/imports even
when pyvis is not installed (scaffold sandbox / CI). The demo is a presentation layer — a viz
error must never take down answering (§9 Step 6), so callers should wrap this in try/except.
"""

from __future__ import annotations

import tempfile

NODE_COLORS = {   # match the dark UI accent palette (demo/app.py)
    "product": "#4d8ff0",   # steel
    "brand": "#35c58e",     # green
    "category": "#eda100",  # amber
    "feature": "#9d6be0",   # violet
}


def render_subgraph(pipeline, query: str, max_nodes: int = 40) -> str:
    """Build a pyvis visualization of the relevant subgraph."""
    from pyvis.network import Network

    entities = pipeline._extract_entities(query)
    G = pipeline.graph

    # Find matching nodes
    from rapidfuzz import fuzz
    seeds = []
    for entity in entities:
        for n, d in G.nodes(data=True):
            if fuzz.partial_ratio(entity.lower(), d.get("name", "").lower()) > 70:
                seeds.append(n)
                break

    # BFS expand to 1 hop
    relevant = set(seeds)
    for seed in seeds:
        for neighbor in list(G.neighbors(seed))[:10]:
            relevant.add(neighbor)

    if not relevant:
        return "<p>No matching graph nodes found.</p>"

    # Build pyvis
    net = Network(height="380px", width="100%", bgcolor="transparent",
                  font_color="white", notebook=False)
    net.toggle_physics(True)

    subgraph = G.subgraph(list(relevant)[:max_nodes])
    for n, d in subgraph.nodes(data=True):
        ntype = d.get("type", "product")
        color = NODE_COLORS.get(ntype, "#888")
        label = d.get("name", str(n))[:30]
        net.add_node(n, label=label, color=color, title=f"{ntype}: {d.get('name', n)}")

    for u, v, d in subgraph.edges(data=True):
        net.add_edge(u, v, title=d.get("type", ""), color="#4a5163")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        return open(f.name).read()
