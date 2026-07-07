"""All prompt templates used by the SPARDA engine.

See SPARDA_BUILD_PLAN.md §6.6 (and §6.1 for the router-classify prompt, which lives in
``engine/router.py``).

GROUNDING: every answer prompt embeds ``_GROUNDING`` — a hard anti-fabrication rule. Without
it the 30B gap-fills from training knowledge (e.g. a "best upgrade from the Galaxy S20" query
recommended the "S20 FE 5G" with specs that were NOT in the retrieved/cited products). The
rule forbids inventing product names/models/specs/prices and requires answers to reference
only the numbered items shown.
"""

_GROUNDING = """RULES (follow strictly):
- Use ONLY the numbered items listed below. They are the entire product catalog for this answer.
- Do NOT invent, assume, or recall from memory any product name, model number, specification,
  price, rating, or availability that is not explicitly written in the items below.
- Recommend ONLY products that appear in the list, and refer to each by the exact name shown
  (you may cite it as "#N"). Never name a product that is not in the list.
- Do NOT mention or compare against any product that is not in the list — not even as a
  reference point, example, or "vs." comparison. If you would compare to an outside product,
  omit that comparison entirely.
- Only state a specification/feature if it is written in that item's text below; do not add
  numbers (zoom, battery, etc.) from memory.
- If the listed items do not actually answer the question, say so plainly and describe the
  closest options that ARE listed — do not fabricate a better-fitting product."""

LOCAL_PROMPT = """You are a product search assistant.

""" + _GROUNDING + """

## Search results (the ONLY products you may reference):
{dante_results}

## Related products from the knowledge graph (also citable):
{graph_context}

## Question: {query}

Answer clearly, referencing specific listed products by their exact name. If the graph context
shows useful "goes-with"/accessory (complement) or "same brand" connections among the listed
items, mention them. Do not introduce any product not shown above.

Answer:"""

GLOBAL_PROMPT = """You are a product market analyst.

""" + _GROUNDING + """
(Here the "items" are the numbered product clusters below — ground every claim about brands,
categories, and trends in these summaries; do not add brands or trends not present in them.)

## Relevant product clusters (the ONLY evidence you may use):
{community_summaries}

## Question: {query}

Give a comprehensive answer grounded in the clusters above, referencing specific clusters and
the brands/trends they actually mention. Do not invent brands, products, or market claims.

Answer:"""

MULTI_HOP_PROMPT = """You are a product discovery assistant. The user's question required graph
traversal to find connected products. Here is exactly what the graph returned:

""" + _GROUNDING + """

## Query entities found in the graph:
{source_entities}

## Discovered products (scored by relevance — the ONLY products you may recommend):
{discovered_products}

## Graph reasoning paths (how each product connects):
{paths}

## Question: {query}

Explain how the discovered products connect to the question using the graph paths, and
recommend the most relevant ones — but ONLY from the discovered list above. If none of the
discovered products truly fit, say so rather than naming a product that was not discovered.

Answer:"""

ENTITY_EXTRACTION_PROMPT = """Extract product entities from this query.
Return ONLY a JSON array of plain strings (brand names, product names, features,
categories) — e.g. ["Sony", "WH-1000XM5", "noise cancelling"]. No objects, no keys, no prose.
Query: "{query}"
Entities:"""
