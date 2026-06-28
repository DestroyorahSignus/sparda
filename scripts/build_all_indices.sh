#!/usr/bin/env bash
# Build FAISS + BM25 + SPLADE + ColBERT indices.
# REUSE FIRST — Phase B fallback (§9 B1). No-op if indices already exist (artifacts block).
# Calls the INSTALLED dante package's index builders. See SPARDA_BUILD_PLAN.md §9.
set -euo pipefail
echo "TODO: SPARDA_BUILD_PLAN.md §9 B1 — bootstrap-only; build FAISS/BM25/SPLADE/ColBERT indices via dante."
exit 1
