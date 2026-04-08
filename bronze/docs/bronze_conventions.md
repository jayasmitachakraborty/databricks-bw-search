# Bronze conventions

## Layer intent

Bronze stores raw replicated data from **Fivetran** with minimal modification. In this repo, bronze sources come from:

- **Azure Database for MySQL**
- **Azure Cosmos DB**

The goal is that someone can **clone the repo**, point Fivetran at their own Azure sources, and land an equivalent bronze layer that downstream silver/gold SQL expects.

## Object model

Use Unity Catalog names:
- catalog = top-level container
- schema = source-aligned grouping
- table = landed object

Recommended:
- one bronze schema per source system (or per connector if you prefer connector-level isolation)
- schema naming pattern: `_01_bronze_<source>` (matches the rest of this repo’s SQL examples)

### Expected Unity Catalog layout (convention)

- **Catalog**: `bw_ai_search`
- **Bronze schemas** (examples):
  - `_01_bronze_pitchbook` (example: relational source like MySQL)
  - `_01_bronze_cosmos_websites` (example: Cosmos container exported as tables)
  - `_01_bronze_bw_dashboard`
  - `_01_bronze_taxonomy`

If you choose different names, standardize them and update the SQL under `silver/sql/` accordingly.

## Fivetran ingestion conventions

### Connector naming & traceability

- connector name: 01_bronze
- source type: Azure Database for MySQL
- expected sync frequency / schedule: every 6 hours
- destination catalog + schema: 
    bw_ai_search.01_bronze_pitchbook, 
    bw_ai_search.01_bronze_taxonomy, 
    bw_ai_search.01_bronze_bw_dashboard

### Table naming

Prefer **source-faithful** names:

- keep table names and primary keys as close to the source as possible
- avoid “analytics-friendly” renames in bronze (do that in silver)
- preserve Fivetran metadata columns (for example `_fivetran_deleted`, `_fivetran_synced`) if present

### Incremental behavior & deletes

- if Fivetran provides a tombstone column like `_fivetran_deleted`, bronze should keep it
- silver models should filter or interpret deletes consistently

## Source connectivity

To reproduce bronze in a new workspace/environment, you’ll need:

- **Fivetran** account with connectors for Azure MySQL and Azure Cosmos
- A **Databricks destination** configured for Fivetran (Unity Catalog / Delta)
- Network access (private endpoints / firewall rules) so Fivetran can reach the Azure sources
- Credentials stored in Fivetran

This repo intentionally does **not** store secrets.

## Allowed changes in bronze

Allowed:
- access-control SQL
- helper documentation
- operational views where necessary
- light technical annotations

Not allowed:
- business-rule transformations
- semantic renaming for analytics convenience
- cross-source joins
- conformed dimensions

## Schema drift

When source columns are added or types change:
- update `sources.yml` if the expected-table inventory changes materially
- document downstream impact in the PR
- handle type cleanup and conformance in silver

## Governance

All bronze schemas must have:
- named owner
- business domain
- lifecycle status
- downstream target schema

## Table inventory (recommended template)

For each bronze schema, keep (at minimum) a lightweight inventory somewhere visible (README, PR template, or a short section here):

- **table name**
- **primary key**
- **CDC / delete semantics** (tombstone column? hard deletes?)
- **PII / sensitive columns** (and masking/tagging approach)
- **expected row volume** (order of magnitude) and growth
- **data freshness SLA**

This reduces time-to-first-success for anyone cloning the repo and wiring their own connectors.