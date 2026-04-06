# Bronze conventions

## Layer intent

Bronze stores raw replicated data from Fivetran with minimal modification.

## Object model

Use Unity Catalog names:
- catalog = top-level container
- schema = source-aligned grouping
- table = landed object

Recommended:
- one bronze schema per source system
- schema naming pattern: `<source>_bronze`

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