-- materialized_views.sql
-- Pre-aggregated tables for fast dashboard queries.
-- Run after the raw data is loaded and periodically refreshed via the
-- Dagster pipeline or a manual invocation.
--
-- DuckDB does not support native materialized views, so these are
-- implemented as regular tables created with CREATE TABLE ... AS SELECT.
-- Idempotent: DROP IF EXISTS before each CREATE.

-- =========================================================================
-- mv_funnel_daily
-- Daily funnel step counts for the canonical conversion funnel.
-- Each row represents one calendar date with user counts at each step.
-- =========================================================================

drop table if exists mv_funnel_daily;

create table mv_funnel_daily as
select
    cast(e.timestamp as date)                                                as funnel_date,
    count(distinct case when e.event_name = 'view_homepage'      then e.user_id end) as step_1_view_homepage,
    count(distinct case when e.event_name = 'view_product'       then e.user_id end) as step_2_view_product,
    count(distinct case when e.event_name = 'click_add_to_cart'  then e.user_id end) as step_3_add_to_cart,
    count(distinct case when e.event_name = 'begin_checkout'     then e.user_id end) as step_4_begin_checkout,
    count(distinct case when e.event_name = 'complete_purchase'  then e.user_id end) as step_5_complete_purchase
from events e
group by cast(e.timestamp as date)
order by funnel_date;


-- =========================================================================
-- mv_session_daily
-- Daily session aggregates broken down by device type and UTM source.
-- Bounce is defined as a session with exactly one page view and no other events.
-- =========================================================================

drop table if exists mv_session_daily;

create table mv_session_daily as
select
    cast(s.started_at as date)                                               as session_date,
    s.device_type,
    s.utm_source,
    count(*)                                                                 as total_sessions,
    avg(s.duration_seconds)                                                  as avg_duration_seconds,
    avg(s.page_view_count)                                                   as avg_page_depth,
    sum(case when s.event_count = 1 and s.page_view_count = 1
             then 1 else 0 end) * 100.0 / count(*)                          as bounce_rate,
    sum(case when s.has_conversion then 1 else 0 end) * 100.0 / count(*)    as conversion_rate
from sessions s
group by
    cast(s.started_at as date),
    s.device_type,
    s.utm_source
order by session_date, s.device_type, s.utm_source;


-- =========================================================================
-- mv_user_segments
-- Snapshot of user dimension data joined with activity summaries.
-- Designed to be enriched with RFM scores and cluster labels after
-- the segmentation pipeline runs.
-- =========================================================================

drop table if exists mv_user_segments;

create table mv_user_segments as
with user_activity as (
    select
        e.user_id,
        count(*)                                           as total_events,
        count(distinct e.session_id)                       as total_sessions,
        coalesce(sum(e.revenue), 0)                        as total_revenue,
        max(e.timestamp)                                   as last_activity_at,
        count(distinct case when e.event_type = 'purchase'
                             then e.event_id end)          as purchase_count,
        count(distinct case when e.event_type = 'feature_use'
                             then e.event_name end)        as unique_features_used
    from events e
    group by e.user_id
)
select
    u.user_id,
    u.plan_type,
    u.signup_source,
    u.country,
    u.created_at,
    u.is_active,
    coalesce(ua.total_events, 0)          as total_events,
    coalesce(ua.total_sessions, 0)        as total_sessions,
    coalesce(ua.total_revenue, 0)         as total_revenue,
    ua.last_activity_at,
    coalesce(ua.purchase_count, 0)        as purchase_count,
    coalesce(ua.unique_features_used, 0)  as unique_features_used
from users u
left join user_activity ua on u.user_id = ua.user_id
order by u.user_id;
