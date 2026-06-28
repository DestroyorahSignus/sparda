#!/usr/bin/env bash
# One-shot bootstrap: train bi-encoder + SPLADE + (optional) ColBERT.
# REUSE FIRST — this is the Phase B fallback (§9). If DANTE's checkpoints already exist
# (configs/default.yaml :: artifacts), SPARDA loads them and this is a no-op.
# It calls the INSTALLED dante package's training functions (dante.train_biencoder, ...);
# SPARDA does not re-implement training. Counts against DANTE's <=4h A100 budget, NOT new
# SPARDA spend. See SPARDA_BUILD_PLAN.md §9 Phase B / B1.
set -euo pipefail
echo "TODO: SPARDA_BUILD_PLAN.md §9 B1 — bootstrap-only; invoke dante.train_biencoder + index build."
exit 1
