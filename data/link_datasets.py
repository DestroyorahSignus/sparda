"""Dataset linking — and SPARDA's Complement-edge enrichment.

See SPARDA_BUILD_PLAN.md §3.3.

This is SPARDA's own (first-party) code. ``link_esci_to_amazon`` is underspecified in the
plan (the body is a ``pass`` placeholder) so it is left as a NotImplementedError stub with
the right signature + docstring. ``add_complement_edges`` IS fully specified in the plan and
is transcribed faithfully — it is the SPARDA-only advantage (restores the "goes-with"
accessory signal Amazon-2023 lacks).
"""

from __future__ import annotations


def _norm_asin(value) -> str:
    """Normalize an ASIN/product_id for joining: cast to str, strip, uppercase.

    Both datasets are Amazon-derived (DANTE's ESCI ``product_id`` and VERGIL's
    Amazon-2023 ``parent_asin`` are the SAME ASIN keyspace), but casing/whitespace
    can drift across mirrors, so normalize before comparing (§3.3 / RISKS R1).
    """
    return str(value).strip().upper()


def link_esci_to_amazon(dante_catalog_df, vergil_meta_df) -> dict:
    """
    Measure the ASIN overlap between DANTE's retrieval catalog and VERGIL's graph.

    Inner-joins DANTE ``catalog.parquet.product_id`` × VERGIL
    ``electronics_meta.parquet.parent_asin`` on the normalized ASIN key. This overlap
    is the headline "does the graph enrich retrieval" number (§6.0 / RISKS R1): where
    the two datasets overlap, a DANTE-retrieved product also lives in VERGIL's graph,
    so it can be expanded with brand/category/feature/``similar_to`` neighbours AND the
    SPARDA-only ``complement_of`` "goes-with" edges. Where they don't overlap, that's
    fine and expected — the graph still covers Amazon-only products and ESCI still
    covers search-only products (graceful degradation).

    Args:
        dante_catalog_df: DANTE ``catalog.parquet`` — must have a ``product_id`` column
            (these ARE ASINs; the catalog is ``[product_id, product_text]``).
        vergil_meta_df: VERGIL ``electronics_meta.parquet`` — must have a
            ``parent_asin`` column (the graph node id).

    Returns:
        A stats dict::

            {
              "n_dante":     <# distinct DANTE ASINs>,
              "n_vergil":    <# distinct VERGIL parent_asins>,
              "n_overlap":   <# ASINs present in BOTH>,
              "join_rate":   n_overlap / n_dante,   # fraction of DANTE catalog in the graph
              "overlap_ids": [<normalized overlapping ASINs, sorted>],
              "linked_db":   {asin: {"asin": asin}},  # convenience map keyed by overlap ASIN
                                                      # (feeds engine.coverage / add_complement_edges)
            }

        ``join_rate`` is reported HONESTLY — whatever the measured overlap is (§6.0/R1).
    """
    dante_asins = {
        _norm_asin(v) for v in dante_catalog_df["product_id"].tolist()
        if str(v).strip() and str(v).strip().lower() != "nan"
    }
    vergil_asins = {
        _norm_asin(v) for v in vergil_meta_df["parent_asin"].tolist()
        if str(v).strip() and str(v).strip().lower() != "nan"
    }

    overlap = dante_asins & vergil_asins
    n_dante, n_vergil, n_overlap = len(dante_asins), len(vergil_asins), len(overlap)
    join_rate = (n_overlap / n_dante) if n_dante else 0.0
    overlap_ids = sorted(overlap)

    return {
        "n_dante": n_dante,
        "n_vergil": n_vergil,
        "n_overlap": n_overlap,
        "join_rate": join_rate,
        "overlap_ids": overlap_ids,
        # keyed by ASIN so it plugs straight into engine.coverage.global_coverage
        # (graph.has_node(asin)) and into add_complement_edges' linked_db[p]["asin"].
        "linked_db": {a: {"asin": a} for a in overlap_ids},
    }


def add_complement_edges(G, esci_df, linked_db, encoder=None, min_pairs: int = 1):
    """
    Restore the accessory / 'goes-with' relationship Amazon-2023 lacks, by mining
    ESCI 'Complement' pairs and writing complement_of edges into VERGIL's graph G.

    Only fires where BOTH products linked to ASINs that are nodes in G — i.e. where
    the two datasets overlap. Elsewhere it is a no-op (graceful degradation, R1).

    `encoder` is the SAME injected DANTE bi-encoder used for similarity edges (§2.1);
    here it only OPTIONALLY weights an edge by query↔product similarity. CPU-only graph
    writes → ~0 marginal GPU.
    """
    # 1) For each ESCI query, collect its products grouped by label.
    added = 0
    for query, grp in esci_df.groupby("query"):
        comp = [p for p, lab in zip(grp["product_id"], grp["esci_label"])
                if lab == "Complement"]                       # full word, not 'C'
        anchors = [p for p, lab in zip(grp["product_id"], grp["esci_label"])
                   if lab in ("Exact", "Substitute")]         # what the query is "about"

        # 2) Resolve to graph nodes via the ASIN link; skip non-overlapping products.
        comp_nodes    = [linked_db[p]["asin"] for p in comp    if p in linked_db]
        anchor_nodes  = [linked_db[p]["asin"] for p in anchors if p in linked_db]
        comp_nodes    = [a for a in comp_nodes   if G.has_node(a)]
        anchor_nodes  = [a for a in anchor_nodes if G.has_node(a)]

        # 3) anchor --[complement_of]--> complement: "goes with what you searched for".
        for a in anchor_nodes:
            for c in comp_nodes:
                if a == c:
                    continue
                w = 1.0
                if encoder is not None:
                    # optional: weight by query↔complement-product similarity
                    import numpy as np
                    qe = encoder.encode([query], normalize_embeddings=True)
                    ce = encoder.encode([G.nodes[c].get("name", "")],
                                        normalize_embeddings=True)
                    w = float(np.dot(qe[0], ce[0]))
                G.add_edge(a, c, type="complement_of", weight=w, source="esci_complement")
                added += 1

        # 4) (optional) co-results of the SAME query also "go together" — weaker edge.
        #    for u, v in itertools.combinations(comp_nodes, 2):
        #        G.add_edge(u, v, type="complement_of", weight=0.5, source="esci_coresult")

    G.graph["complement_edges_added"] = added   # surface in README/demo footer
    return G
