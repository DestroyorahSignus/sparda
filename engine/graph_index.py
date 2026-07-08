"""Per-graph cached name index for fuzzy entity linking.

Both ``engine.multi_hop_search`` and ``demo.graph_viz`` used to fuzzy-scan EVERY node in
the ~65K-node graph per entity, per query (a Python-level ``fuzz.partial_ratio`` loop),
and the demo pays it twice per user action (/api/query + /api/graph). This module builds
the ``(node_ids, lowercased_names)`` index ONCE per graph object and matches with
``rapidfuzz.process`` (C level). The WeakKeyDictionary ties each cache entry's lifetime
to its graph, so a freshly loaded graph can never see a stale index.
"""

from __future__ import annotations

import weakref

_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def name_index(G) -> tuple[list, list[str]]:
    """``(node_ids, lowercased_names)`` for all named nodes, built once per graph."""
    cached = _CACHE.get(G)
    if cached is None:
        pairs = [(n, str(d.get("name", "")).lower())
                 for n, d in G.nodes(data=True) if d.get("name")]
        cached = ([p[0] for p in pairs], [p[1] for p in pairs])
        _CACHE[G] = cached
    return cached


def link_entity(G, entity: str, cutoff: float = 70):
    """BEST fuzzy match for one entity over all named nodes, or None.

    Mirrors the original inline semantics: ``fuzz.partial_ratio`` on lowercased
    names, strictly ``score > cutoff`` wins.
    """
    from rapidfuzz import fuzz, process

    node_ids, names = name_index(G)
    if not names:
        return None
    needle = str(entity).strip().lower()
    if not needle:
        return None
    hit = process.extractOne(needle, names, scorer=fuzz.partial_ratio, score_cutoff=cutoff)
    if hit is None or hit[1] <= cutoff:   # extractOne is >=; original required >
        return None
    return node_ids[hit[2]]
