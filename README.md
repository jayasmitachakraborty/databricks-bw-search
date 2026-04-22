# databricks-bw-search

Medallion-style data platform layout: bronze (Fivetran landing), silver (cleansed/conformed), gold (curated analytics), pipelines, RAG notebooks, and AI configuration.

See [docs/architecture.md](docs/architecture.md) for an overview.

## RAG retrieval behavior

In `ai/src/retrieval.py`, `hybrid_retrieve_top50()` runs retrieval in two passes:

- First pass: applies metadata filters (from query understanding + any `extra_filters`).
- Fallback pass: if the filtered pass returns **0 rows**, it retries **once** with **no filters** to avoid “over-filtering” returning empty results.
