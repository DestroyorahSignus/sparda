#!/usr/bin/env bash
# Export models/indices/graph for Kaggle T4 inference (offline).
# Bundle the artifacts (configs/default.yaml :: artifacts) + build dante/vergil wheels so
# inference needs no network (RISKS R2 — private-repo install fallback).
# See SPARDA_BUILD_PLAN.md §9 / §10 / RISKS R2.
set -euo pipefail
echo "TODO: SPARDA_BUILD_PLAN.md §9 — package artifacts + dante/vergil wheels for Kaggle export."
exit 1
