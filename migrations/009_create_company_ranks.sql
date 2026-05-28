-- Migration: 009_create_company_ranks
-- Per-company percentile rank for each metric, computed within the company's
-- GICS sector by company_ranks.py from the latest fundamentals_snapshot rows.
--
-- percentile_rank is 0..100 where 100 = best-in-sector. For inverted metrics
-- (pe_ratio, ev_ebitda, ev_revenue, pb_ratio, debt_ebitda, p_ocf) "best" means
-- the LOWEST raw value (cheapest / least levered); the ranking module flips
-- those before storing, so 100 is always "best" regardless of metric.
--
-- History-preserving, same pattern as sector_statistics: one run timestamp per
-- run, read latest via MAX(calculated_at).
--
-- Run once in the Supabase SQL editor.

CREATE TABLE IF NOT EXISTS company_ranks (
    ticker          VARCHAR(10)  NOT NULL,
    metric_name     TEXT         NOT NULL,
    raw_value       NUMERIC,                 -- the company's actual metric value
    percentile_rank NUMERIC,                 -- 0..100, 100 = best in sector
    sector          TEXT,                    -- denormalised for fast filtering
    calculated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT company_ranks_pkey PRIMARY KEY (ticker, metric_name, calculated_at)
);

-- Latest ranks for one ticker (the dashboard's main read)
CREATE INDEX IF NOT EXISTS idx_company_ranks_ticker
    ON company_ranks (ticker, calculated_at DESC);

-- Sector-wide scans (e.g. "top ROIC names in Financials")
CREATE INDEX IF NOT EXISTS idx_company_ranks_sector
    ON company_ranks (sector, metric_name, calculated_at DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- Dashboard read access — see the note in 008_create_sector_statistics.sql.
-- Required so the dashboard's anon key can read this table.
-- ─────────────────────────────────────────────────────────────────────────────
ALTER TABLE company_ranks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Anon and authenticated can read company_ranks" ON company_ranks;
CREATE POLICY "Anon and authenticated can read company_ranks"
    ON company_ranks FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON company_ranks TO anon, authenticated;
