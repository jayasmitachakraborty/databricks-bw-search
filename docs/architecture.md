# Architecture

## Medallion layers

1. **Bronze** — Raw replication from Fivetran into Unity Catalog; minimal transformation; grants and optional thin views.
2. **Silver** — Cleansed keys, types, deduplication, and conformed entities.
3. **Gold** — Business-ready marts and feature tables for BI, ML, and applications.

## Pipelines

Orchestration lives under `pipelines/`: bronze → silver, silver → gold, and a path from documents to the vector index for RAG.

## AI / RAG

- **Notebooks** (`notebooks/rag/`) — End-to-end experiments and scheduled jobs.
- **Config** (`ai/config/`) — Embedding sources, vector index, and Genie settings.
- **Library code** (`ai/src/`) — Reusable chunking, embedding, and retrieval logic.

Adjust names, catalogs, and job definitions to match your Databricks workspace and deployment tooling.
