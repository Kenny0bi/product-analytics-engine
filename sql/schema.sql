-- Product Analytics Engine — Database Schema
-- Target: DuckDB (analytical SQL engine)
-- All tables use varchar primary keys for portability and readability.

-- =============================================================================
-- events: Raw event stream capturing every user interaction.
-- Each row is a single action (page view, click, purchase, etc.) tied to
-- a user and session. The properties column holds event-specific JSON payload.
-- =============================================================================
create table if not exists events (
    event_id            varchar primary key,
    user_id             varchar not null,
    session_id          varchar not null,
    event_type          varchar not null,       -- 'page_view', 'click', 'signup', 'purchase', 'feature_use', 'form_submit', 'search'
    event_name          varchar not null,       -- specific event identifier, e.g. 'view_homepage', 'click_add_to_cart'
    timestamp           timestamp not null,
    properties          json,                   -- event-specific key-value payload
    page_url            varchar,
    referrer            varchar,
    utm_source          varchar,
    utm_medium          varchar,
    utm_campaign        varchar,
    device_type         varchar,                -- 'desktop', 'mobile', 'tablet'
    browser             varchar,
    os                  varchar,
    country             varchar,
    city                varchar,
    revenue             decimal(10,2)           -- non-null only for purchase events
);

create index if not exists idx_events_user_id on events(user_id);
create index if not exists idx_events_session_id on events(session_id);
create index if not exists idx_events_timestamp on events(timestamp);
create index if not exists idx_events_event_type on events(event_type);
create index if not exists idx_events_event_name on events(event_name);

-- =============================================================================
-- users: Dimension table for user profiles.
-- Tracks signup metadata, current plan, and lifetime aggregates.
-- =============================================================================
create table if not exists users (
    user_id             varchar primary key,
    created_at          timestamp not null,
    signup_source       varchar,                -- 'organic', 'paid_search', 'social', 'referral', 'email'
    plan_type           varchar,                -- 'free', 'starter', 'pro', 'enterprise'
    industry            varchar,                -- 'saas', 'ecommerce', 'fintech', 'healthcare', 'education', 'media', 'other'
    company_size        varchar,                -- 'solo', 'small', 'medium', 'large', 'enterprise'
    country             varchar,
    is_active           boolean default true,
    first_purchase_at   timestamp,
    total_revenue       decimal(10,2) default 0,
    lifetime_events     integer default 0
);

create index if not exists idx_users_created_at on users(created_at);
create index if not exists idx_users_signup_source on users(signup_source);
create index if not exists idx_users_plan_type on users(plan_type);

-- =============================================================================
-- sessions: Session-level aggregates derived from the event stream.
-- A session groups events by user with a 30-minute inactivity timeout.
-- =============================================================================
create table if not exists sessions (
    session_id          varchar primary key,
    user_id             varchar not null,
    started_at          timestamp not null,
    ended_at            timestamp,
    duration_seconds    integer,
    event_count         integer,
    page_view_count     integer,
    has_conversion      boolean default false,
    entry_page          varchar,
    exit_page           varchar,
    device_type         varchar,
    utm_source          varchar,
    utm_medium          varchar,
    utm_campaign        varchar
);

create index if not exists idx_sessions_user_id on sessions(user_id);
create index if not exists idx_sessions_started_at on sessions(started_at);

-- =============================================================================
-- experiments: A/B test definitions.
-- Each row describes one experiment with its target metric, variants,
-- and sample size requirements.
-- =============================================================================
create table if not exists experiments (
    experiment_id       varchar primary key,
    experiment_name     varchar not null,
    description         varchar,
    start_date          date not null,
    end_date            date,
    status              varchar default 'running',      -- 'running', 'completed', 'stopped'
    metric_name         varchar not null,                -- 'conversion_rate', 'revenue_per_user', 'session_duration'
    control_variant     varchar default 'control',
    treatment_variant   varchar default 'treatment',
    target_sample_size  integer,
    minimum_effect_size decimal(5,4)
);

-- =============================================================================
-- experiment_assignments: Maps users to experiment variants.
-- Tracks whether the user converted and the conversion value (if applicable).
-- =============================================================================
create table if not exists experiment_assignments (
    user_id             varchar not null,
    experiment_id       varchar not null,
    variant             varchar not null,                -- 'control' or 'treatment'
    assigned_at         timestamp not null,
    converted           boolean default false,
    conversion_value    decimal(10,2),
    primary key (user_id, experiment_id)
);

create index if not exists idx_exp_assignments_experiment on experiment_assignments(experiment_id);
create index if not exists idx_exp_assignments_variant on experiment_assignments(experiment_id, variant);

-- =============================================================================
-- daily_metrics: Precomputed daily aggregates for fast dashboard queries.
-- Stores one row per (date, metric, dimension) combination. The dimension
-- columns are null for overall (non-segmented) metrics.
-- =============================================================================
create table if not exists daily_metrics (
    metric_date         date not null,
    metric_name         varchar not null,
    metric_value        decimal(15,4),
    -- 'overall' sentinel for non-segmented metrics; otherwise 'device_type',
    -- 'plan_type', etc. A sentinel (not NULL) because DuckDB enforces
    -- NOT NULL on all primary key columns.
    dimension_name      varchar not null default 'overall',
    dimension_value     varchar not null default 'overall',
    primary key (metric_date, metric_name, dimension_name, dimension_value)
);

create index if not exists idx_daily_metrics_name_date on daily_metrics(metric_name, metric_date);
