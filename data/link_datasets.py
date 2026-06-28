"""Dataset linking — and SPARDA's Complement-edge enrichment.

See SPARDA_BUILD_PLAN.md §3.3.

This is SPARDA's own (first-party) code. ``link_esci_to_amazon`` is underspecified in the
plan (the body is a ``pass`` placeholder) so it is left as a NotImplementedError stub with
the right signature + docstring. ``add_complement_edges`` IS fully specified in the plan and
is transcribed faithfully — it is the SPARDA-only advantage (restores the "goes-with"
accessory signal Amazon-2023 lacks).
"""

from __future__ import annotations


def link_esci_to_amazon(esci_df, amazon_df):
    """
    Link ESCI products to Amazon Reviews metadata by ASIN/product_id.
    Where they overlap, products get BOTH retrieval labels AND graph edges.
    Where they don't overlap, that's fine — graph covers Amazon-only products,
    ESCI covers search-only products.

    The overlap enriches local search: DANTE retrieves a product, VERGIL knows
    what category/brand/feature neighbours and which ESCI-Complement "goes-with"
    products it connects to.

    Implementation notes (SPARDA_BUILD_PLAN.md §3.3 / §6.0 / RISKS R1):
    - Left join ESCI products onto Amazon metadata by product_id / parent_asin.
    - Normalize ASINs first (uppercase, strip) and try the variant ASIN too.
    - Not all will match — that's expected and OK (measure the join rate, §6.0/R1).

    Returns a ``linked_db`` mapping ``product_id -> {"asin": ..., <retrieval fields>}``.
    """
    raise NotImplementedError(
        "TODO: SPARDA_BUILD_PLAN.md §3.3 — implement the normalized ASIN left-join "
        "(uppercase/strip, try variant ASIN + parent_asin), returning linked_db; "
        "the plan provides only the docstring contract for this function."
    )


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
