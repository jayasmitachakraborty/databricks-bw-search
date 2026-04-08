# Architecture

## Medallion layers

1. **Bronze** — Raw replication from **Fivetran** into Unity Catalog; minimal transformation; grants and optional thin views.
   - Sources in this repo: **Azure Database for MySQL** + **Azure Cosmos DB** connectors landing into schemas like `_01_bronze_<source>`.
2. **Silver** — Cleansed keys, types, deduplication, and conformed entities. This repo builds:
   - `bw_ai_search.02_silver.bw_pitchbook_company_all_investors` (distinct investors per company)
   - `bw_ai_search.02_silver.bw_deals_investors_with_ranking` (deal→investor with `noa_ranking`)
   - `bw_ai_search.02_silver.bw_company_semantic_text` (one row per company with concatenated `semantic_text`)
   - `bw_ai_search.02_silver.bw_company_text_chunks` (chunked text for embeddings)
   - `bw_ai_search.02_silver.bw_company_text_chunk_embeddings` (embeddings per chunk; requires an embedding endpoint)
3. **Gold** — Business-ready marts and feature tables for BI, ML, and applications (not implemented yet in this repo).

## Pipelines

Orchestration can be implemented as a Databricks **Workflow** (jobs) that runs:

- **SQL tasks**: build silver staging tables from bronze
- **Notebook tasks**: build `bw_company_semantic_text`, chunk it, then embed it

The `pipelines/` folder contains templates; deployment wiring depends on whether you use Workspace notebooks, Git folders, or Databricks Asset Bundles.

## AI / RAG

- **Notebooks** (`notebooks/`) — Runnable Databricks notebooks for transformations and RAG:
  - `notebooks/transformations/create_semantic_text.py`
  - `notebooks/rag/chunk_documents.py`
  - `notebooks/rag/create_embeddings.py`
- **Config** (`ai/config/`) — Embedding sources, vector index, and Genie settings.
- **Library code** (`ai/src/`) — Reusable chunking, embedding, and retrieval logic.

Adjust names, catalogs, and job definitions to match your Databricks workspace and deployment tooling.
