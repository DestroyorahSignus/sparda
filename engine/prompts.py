"""All prompt templates used by the SPARDA engine.

See SPARDA_BUILD_PLAN.md §6.6 (and §6.1 for the router-classify prompt, which lives in
``engine/router.py``).
"""

LOCAL_PROMPT = """You are a helpful product search assistant. Answer using ONLY the provided information.

## Search results:
{dante_results}

## Related products (from knowledge graph):
{graph_context}

## Question: {query}

Provide a clear answer. Reference specific products by name. If graph context reveals
useful "goes-with"/accessory (complement) or "same brand" connections, mention them.

Answer:"""

GLOBAL_PROMPT = """You are a product market analyst. Answer using the provided community summaries.

## Relevant product clusters:
{community_summaries}

## Question: {query}

Provide a comprehensive answer. Reference specific clusters, brands, and trends.

Answer:"""

MULTI_HOP_PROMPT = """You are a product discovery assistant. The user's question required graph traversal
to find connected products. Here's what was discovered:

## Query entities found in graph:
{source_entities}

## Discovered products (scored by relevance):
{discovered_products}

## Graph reasoning paths:
{paths}

## Question: {query}

Explain how these products connect to the question using the graph paths.
Recommend the most relevant ones.

Answer:"""

ENTITY_EXTRACTION_PROMPT = """Extract product entities from this query.
Return a JSON list of entities (brand names, product names, features, categories).
Query: "{query}"
Entities:"""
