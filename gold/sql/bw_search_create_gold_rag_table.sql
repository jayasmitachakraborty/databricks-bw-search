create or replace table bw_ai_search.03_gold.gold_rag_company_chunks
using delta
tblproperties ('delta.enableChangeDataFeed' = 'true')
as

with verticals_agg as (
    select
        company_id,
        concat_ws(
            ' ',
            sort_array(collect_set(trim(verticals)))
        ) as verticals_text,
        sort_array(collect_set(trim(verticals))) as verticals_array
    from bw_ai_search._01_bronze_pitchbook.company_verticals
    where coalesce(_fivetran_deleted, false) = false
      and verticals is not null
      and trim(verticals) <> ''
    group by company_id
),

keywords_agg as (
    select
        company_id,
        concat_ws(
            ' ',
            sort_array(collect_set(trim(keywords)))
        ) as keywords_text,
        sort_array(collect_set(trim(keywords))) as keywords_array
    from bw_ai_search._01_bronze_pitchbook.company_keywords
    where coalesce(_fivetran_deleted, false) = false
      and keywords is not null
      and trim(keywords) <> ''
    group by company_id
),

investors_agg as (
    select
        Deal_ID as deal_id,
        concat_ws(
            ' | ',
            sort_array(collect_set(trim(Investor_Name)))
        ) as investor_names,
        sort_array(collect_set(trim(Investor_Name))) as investor_names_array,
        min(noa_ranking) as best_investor_rank,
        count(distinct Investor_Name) as investor_count
    from bw_ai_search.`02_silver`.bw_deals_investors_with_ranking
    where Investor_Name is not null
      and trim(Investor_Name) <> ''
    group by Deal_ID
)

select
    c.chunk_id,
    c.company_id,
    c.chunk_index,
    c.chunk_text,

    e.embedding,
    e.embedding_endpoint,
    e.embedded_at,

    m.company_name,
    m.website,
    m.built_world,
    m.climate_tech,
    m.theme,
    m.main_category,
    m.subcategory,
    m.noa_impact_framework,
    m.area,
    m.deal_id,
    m.deal_no,
    m.deal_date,
    m.deal_year,
    m.invested_equity,
    m.deal_size,
    m.total_new_debt,
    m.raised_to_date,
    m.noa_funding_round,
    m.deal_type,
    m.deal_type_2,
    m.year_founded,
    m.country,
    m.city,
    m.region,
    m.bay_area,
    m.dealcount,
    m.last_updated,
    m._fivetran_deleted,
    m._fivetran_synced,

    -- Added metadata from related tables
    v.verticals_text,
    v.verticals_array,
    k.keywords_text,
    k.keywords_array,
    i.investor_names,
    i.investor_names_array,
    i.best_investor_rank,
    i.investor_count,

    coalesce(
        m.company_name,
        m.website,
        m.theme,
        m.subcategory,
        'Unknown Company'
    ) as display_title,

    concat_ws(
      ' ',
      -- high importance (repeat 2-3x)
      coalesce(m.company_name, ''),

      -- medium-low
      coalesce(m.theme, ''),
      coalesce(m.main_category, ''),

      -- high importance (repeat 2-3x)
      coalesce(m.subcategory, ''),

      -- low
      coalesce(m.noa_impact_framework, ''),
      coalesce(m.area, ''),

      -- high importance (repeat 2-3x)
      coalesce(m.country, ''),
      coalesce(m.city, ''),
      coalesce(m.region, ''),
      coalesce(m.noa_funding_round, ''),

      -- medium
      coalesce(m.deal_type, ''),
      coalesce(m.deal_type_2, ''),
      coalesce(v.verticals_text, ''),
      coalesce(k.keywords_text, ''),

      -- base content
      coalesce(c.chunk_text, '')
    ) as search_text

from bw_ai_search.`02_silver`.bw_company_text_chunks c

left join bw_ai_search.`02_silver`.bw_company_text_chunk_embeddings e
  on c.chunk_id = e.chunk_id
 and c.company_id = e.company_id
 and c.chunk_index = e.chunk_index

left join bw_ai_search._01_bronze_bw_dashboard.bw_market_trends_anlzd m
  on c.company_id = m.company_id

left join verticals_agg v
  on c.company_id = v.company_id

left join keywords_agg k
  on c.company_id = k.company_id

left join investors_agg i
  on m.deal_id = i.deal_id

where coalesce(m._fivetran_deleted, false) = false;