-- Migration: 008_create_sector_statistics
-- Descriptive statistics per (GICS sector, metric) across the fundamentals
-- universe. Computed daily by sector_stats.py from the latest
-- fundamentals_snapshot rows, grouped by companies.sector (GICS naming).
--
-- History-preserving: every run INSERTs a fresh set of rows stamped with a
-- single run timestamp (calculated_at). Never overwrite — the (sector,
-- metric_name, calculated_at) primary key lets us keep a full time series and
-- always read "the latest" by MAX(calculated_at).
--
-- Run once in the Supabase SQL editor.

CREATE TABLE IF NOT EXISTS sector_statistics (
    sector         TEXT         NOT NULL,   -- GICS sector (from companies.sector)
    metric_name    TEXT         NOT NULL,   -- e.g. 'roic', 'fcf_margin', 'ev_ebitda'
    sample_size    INTEGER      NOT NULL,   -- # companies with a usable (non-null) value

    min_value      NUMERIC,
    max_value      NUMERIC,
    mean           NUMERIC,
    median         NUMERIC,
    std_dev        NUMERIC,
    mad            NUMERIC,                 -- median absolute deviation
    p10            NUMERIC,
    p25            NUMERIC,
    p50            NUMERIC,                 -- == median, kept for clarity
    p75            NUMERIC,
    p90            NUMERIC,

    calculated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT sector_statistics_pkey PRIMARY KEY (sector, metric_name, calculated_at)
);

-- Common read pattern: latest stats for one sector → ORDER BY calculated_at DESC
CREATE INDEX IF NOT EXISTS idx_sector_stats_lookup
    ON sector_statistics (sector, metric_name, calculated_at DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- Dashboard read access. The dashboard queries this table with the anon key, and
-- newly-created tables are NOT exposed to anon by default — so enable RLS with a
-- read-only policy and grant SELECT, mirroring ai_analyses (migration 006).
-- Without this the Sector Benchmarks panel reads back empty.
-- ─────────────────────────────────────────────────────────────────────────────
ALTER TABLE sector_statistics ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Anon and authenticated can read sector_statistics" ON sector_statistics;
CREATE POLICY "Anon and authenticated can read sector_statistics"
    ON sector_statistics FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON sector_statistics TO anon, authenticated;
