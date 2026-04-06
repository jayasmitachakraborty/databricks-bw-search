-- Unity Catalog: bw_ai_search.02_silver.bw_company_semantic_text
-- Columns: company_id, semantic_text
--
-- Output grain: one row per DISTINCT company_id in bw_market_trends_anlzd only (no extra companies
-- from Pitchbook/Cosmos/taxonomy alone). All other attributes are LEFT JOINed onto that set.
-- Geography: distinct country/region pairs from bw_market_trends_anlzd aggregated per company_id.
--
-- Investor text: `all_investors` from bw_pitchbook_company_all_investors, plus tiered lists from
-- `bw_deals_investors_with_ranking` (distinct investor names per `noa_ranking`).
-- Ranked investors use `deal_id` joined to `bw_market_trends_anlzd` via `deal_company`. Column `Investor_Name`.
--
-- =============================================================================
-- PERFORMANCE (large volumes)
-- =============================================================================
-- 1. company_ids: single DISTINCT on bw_market_trends_anlzd.company_id — one scan, matches output grain.
--
-- 2. investors_ranked_by_company: ONE read of bw_deals_investors_with_ranking + ONE join to
--    deal_company, ONE shuffle/groupBy company_id with conditional collect_set per tier — replaces
--    four separate full scans and four joins to assembled.
--
-- 3. Source tables: On Delta, use liquid clustering or Z-order on join keys (e.g. company_id on
--    dimension tables, deal_id on ranking + market-trends) to cut scan and shuffle cost.
--
-- 4. deal_company: Small cardinality vs fact — Databricks typically broadcast-joins to the fact;
--    if deal_company is large, ensure stats/file compaction on both sides of r.deal_id = dc.deal_id.
--
-- 5. assembled: Many LEFT JOINs on company_id — if any dimension is known tiny (e.g. taxonomy),
--    optional /*+ BROADCAST(n) */ hints can help (validate in Spark UI; wrong hints hurt).
--
-- 6. Output: For huge semantic_text rows, run during off-peak; consider writing to Delta with
--    CLUSTER BY (company_id) or liquid clustering on company_id for downstream keyed reads.
--
-- 7. Refresh strategy: For incremental pipelines, replace this full CTAS with MERGE into a target
--    keyed by company_id using only changed sources (not implemented here).
-- =============================================================================

CREATE OR REPLACE TABLE bw_ai_search.`02_silver`.bw_company_semantic_text AS
WITH deal_company AS (
  SELECT
    deal_id,
    max(company_id) AS company_id
  FROM bw_ai_search.`_01_bronze_bw_dashboard`.bw_market_trends_anlzd
  WHERE deal_id IS NOT NULL AND company_id IS NOT NULL
  GROUP BY deal_id
),
company_ids AS (
  SELECT DISTINCT company_id
  FROM bw_ai_search.`_01_bronze_bw_dashboard`.bw_market_trends_anlzd
  WHERE company_id IS NOT NULL
),
geo_by_company AS (
  SELECT
    company_id,
    array_join(
      sort_array(
        collect_set(
          nullif(
            trim(concat_ws(', ', cast(country AS STRING), cast(region AS STRING))),
            ''
          )
        )
      ),
      '; '
    ) AS geography
  FROM bw_ai_search.`_01_bronze_bw_dashboard`.bw_market_trends_anlzd
  WHERE company_id IS NOT NULL
  GROUP BY company_id
),
-- Single pass over ranking × deal_company; conditional collect_set per noa_ranking (nulls dropped by collect_set).
investors_ranked_by_company AS (
  SELECT
    dc.company_id,
    array_join(
      sort_array(
        collect_set(
          CASE WHEN r.noa_ranking = 1 THEN nullif(trim(cast(r.`Investor_Name` AS STRING)), '') END
        )
      ),
      ', '
    ) AS investor_list_high,
    array_join(
      sort_array(
        collect_set(
          CASE WHEN r.noa_ranking = 2 THEN nullif(trim(cast(r.`Investor_Name` AS STRING)), '') END
        )
      ),
      ', '
    ) AS investor_list_medium,
    array_join(
      sort_array(
        collect_set(
          CASE WHEN r.noa_ranking = 3 THEN nullif(trim(cast(r.`Investor_Name` AS STRING)), '') END
        )
      ),
      ', '
    ) AS investor_list_low,
    array_join(
      sort_array(
        collect_set(
          CASE WHEN r.noa_ranking IS NULL THEN nullif(trim(cast(r.`Investor_Name` AS STRING)), '') END
        )
      ),
      ', '
    ) AS investor_list_untracked
  FROM bw_ai_search.`02_silver`.bw_deals_investors_with_ranking r
  INNER JOIN deal_company dc ON r.deal_id = dc.deal_id
  GROUP BY dc.company_id
),
assembled AS (
  SELECT
    c.company_id,
    trim(
      concat_ws(
        '\n\n',
        CASE
          WHEN coalesce(trim(cast(co.description AS STRING)), '') <> ''
          THEN concat('Description: ', trim(cast(co.description AS STRING)))
        END,
        CASE
          WHEN coalesce(trim(cast(w.website_parsed_text AS STRING)), '') <> ''
          THEN concat('Website content: ', trim(cast(w.website_parsed_text AS STRING)))
        END,
        CASE
          WHEN coalesce(trim(cast(w.website_summary AS STRING)), '') <> ''
          THEN concat('Website summary: ', trim(cast(w.website_summary AS STRING)))
        END,
        CASE
          WHEN coalesce(trim(cast(v.verticals AS STRING)), '') <> ''
          THEN concat('Verticals: ', trim(cast(v.verticals AS STRING)))
        END,
        CASE
          WHEN coalesce(trim(cast(k.keywords AS STRING)), '') <> ''
          THEN concat('Keywords: ', trim(cast(k.keywords AS STRING)))
        END,
        CASE
          WHEN coalesce(trim(cast(inv.all_investors AS STRING)), '') <> ''
          THEN concat('Investors: ', trim(cast(inv.all_investors AS STRING)))
        END,
        CASE
          WHEN coalesce(ir.investor_list_high, '') <> ''
          THEN concat('High-quality investors: ', ir.investor_list_high)
        END,
        CASE
          WHEN coalesce(ir.investor_list_medium, '') <> ''
          THEN concat('Medium-quality investors: ', ir.investor_list_medium)
        END,
        CASE
          WHEN coalesce(ir.investor_list_low, '') <> ''
          THEN concat('Low-quality investors: ', ir.investor_list_low)
        END,
        CASE
          WHEN coalesce(ir.investor_list_untracked, '') <> ''
          THEN concat('Untracked investors: ', ir.investor_list_untracked)
        END,
        CASE
          WHEN coalesce(trim(cast(n.theme AS STRING)), '') <> ''
          THEN concat('noa Taxonomy — theme: ', trim(cast(n.theme AS STRING)))
        END,
        CASE
          WHEN coalesce(trim(cast(n.main_category AS STRING)), '') <> ''
          THEN concat('noa Taxonomy — main category: ', trim(cast(n.main_category AS STRING)))
        END,
        CASE
          WHEN coalesce(trim(cast(n.subcategory AS STRING)), '') <> ''
          THEN concat('noa Taxonomy — subcategory: ', trim(cast(n.subcategory AS STRING)))
        END,
        CASE
          WHEN coalesce(trim(cast(n.noa_impact_framework AS STRING)), '') <> ''
          THEN concat('noa Taxonomy — NOA impact framework: ', trim(cast(n.noa_impact_framework AS STRING)))
        END,
        CASE
          WHEN coalesce(g.geography, '') <> ''
          THEN concat('Geography: ', g.geography)
        END
      )
    ) AS semantic_text
  FROM company_ids c
  LEFT JOIN bw_ai_search.`_01_bronze_pitchbook`.companies co
    ON c.company_id = co.company_id
  LEFT JOIN bw_ai_search.`_01_bronze_cosmos_websites`.parsed_text w
    ON c.company_id = w.id
  LEFT JOIN bw_ai_search.`_01_bronze_pitchbook`.company_verticals v
    ON c.company_id = v.company_id
  LEFT JOIN bw_ai_search.`_01_bronze_pitchbook`.company_keywords k
    ON c.company_id = k.company_id
  LEFT JOIN bw_ai_search.`02_silver`.bw_pitchbook_company_all_investors inv
    ON c.company_id = inv.company_id
  LEFT JOIN investors_ranked_by_company ir
    ON c.company_id = ir.company_id
  LEFT JOIN bw_ai_search.`_01_bronze_taxonomy`.noa_taxonomy n
    ON c.company_id = n.company_id
  LEFT JOIN geo_by_company g
    ON c.company_id = g.company_id
)
SELECT company_id, semantic_text FROM assembled;
