"""SPARDA engine — the unified router + orchestration layer over DANTE and VERGIL.

Public surface (SPARDA_BUILD_PLAN.md §2 / §2.1). DANTE and VERGIL are NOT vendored here;
they are pip-installed and imported inside functions (see the individual modules), so this
package parses and imports even when those two deps are absent.
"""

from engine.router import QueryRouter, RouteDecision, ROUTER_CLASSIFY_PROMPT
from engine.coverage import CoverageReport, global_coverage
from engine.local_search import local_search
from engine.global_search import global_search
from engine.multi_hop_search import multi_hop_search
from engine.pipeline import SpardaPipeline

__all__ = [
    "SpardaPipeline",
    "QueryRouter",
    "RouteDecision",
    "ROUTER_CLASSIFY_PROMPT",
    "CoverageReport",
    "global_coverage",
    "local_search",
    "global_search",
    "multi_hop_search",
]
