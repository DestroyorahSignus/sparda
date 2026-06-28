"""Curated test queries (local + global + multi-hop), labeled with expected routes.

Used by the VERGIL RAG ablation (§8.2) and to report router accuracy (RISKS R4).
See SPARDA_BUILD_PLAN.md §8.3.
"""

TEST_QUERIES = [
    # LOCAL — DANTE's domain
    {"q": "best wireless noise cancelling headphones under $300", "type": "local"},
    {"q": "USB-C charger for MacBook Pro", "type": "local"},
    {"q": "4K webcam for streaming", "type": "local"},
    {"q": "mechanical keyboard for programming", "type": "local"},
    {"q": "portable bluetooth speaker waterproof", "type": "local"},

    # GLOBAL — VERGIL communities
    {"q": "What are the major smart home ecosystems and how do they compare?", "type": "global"},
    {"q": "Overview of the portable power bank market by brand", "type": "global"},
    {"q": "What trends exist in wireless audio products?", "type": "global"},
    {"q": "Compare noise cancelling technology across brands", "type": "global"},

    # MULTI-HOP — VERGIL graph traversal
    {"q": "What accessories from Sony are compatible with the WH-1000XM5?", "type": "multi_hop"},
    {"q": "Find chargers from Anker that people buy together with MacBooks", "type": "multi_hop"},
    {"q": "Products from the same brand as this keyboard that also have RGB", "type": "multi_hop"},
    {"q": "What do people usually buy together with a Canon EOS R camera?", "type": "multi_hop"},

    # COMPARISON
    {"q": "Sony vs Bose noise cancelling headphones", "type": "global"},
    {"q": "JBL vs Sonos — which has more accessories?", "type": "multi_hop"},
]
