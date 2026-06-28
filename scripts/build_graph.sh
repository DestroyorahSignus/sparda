#!/usr/bin/env bash
# Build KG + communities + summaries (Phase B fallback, §9 B2-B4).
# REUSE FIRST — no-op if the pickled graph / cached communities / summaries exist.
# Calls the INSTALLED vergil package; inject DANTE's bi-encoder:
#   vergil.add_similarity_edges(G, encoder=dante.biencoder)   (§2.1)
# then SPARDA's own data.link_datasets.add_complement_edges (§3.3) writes the
# complement_of "goes-with" edges. See SPARDA_BUILD_PLAN.md §9 B2-B4.
set -euo pipefail
echo "TODO: SPARDA_BUILD_PLAN.md §9 B2-B4 — bootstrap-only; build graph + Leiden + summaries + complement edges."
exit 1
