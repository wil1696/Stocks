-- Migration: 010_create_rule_results
-- Stores the output of the Buffett-inspired rule engine (rule_engine.py): a hard
-- pass/fail per rule per company, plus the derived category (shortlist /
-- watchlist / rejected). One row per ticker per run.
--
-- Rule columns are BOOLEAN and NULLABLE: NULL = "not applicable" — either the
-- rule is exempt for the company's GICS sector (e.g. banks skip EV/EBITDA), or
-- the rule could not be evaluated due to missing data. rule_3_3 is a deferred
-- placeholder (Margin of safety vs DCF — not yet implemented) and is always
-- NULL for now; it is excluded from applicable_count / pass_pct.
--
-- applicable_count = (# of the 11 active rules applicable to this company's
-- sector); passed_count = (# of those it passes); pass_pct = passed/applicable.
--   pass_pct == 1.0   → 'shortlist'
--   pass_pct >= 0.80  → 'watchlist'
--   else              → 'rejected'
--
-- History-preserving, same pattern as sector_statistics / company_ranks: every
-- run INSERTs a fresh set of rows stamped with one calculated_at; read "latest"
-- via MAX(calculated_at). The (ticker, calculated_at) primary key keeps the
-- full time series.
--
-- Run once in the Supabase SQL editor.

CREATE TABLE IF NOT EXISTS rule_results (
    ticker           VARCHAR(10)  NOT NULL,
    sector           TEXT,                    -- denormalised GICS sector (companies.sector)
    category         TEXT         NOT NULL,   -- 'shortlist' | 'watchlist' | 'rejected'

    passed_count     INTEGER      NOT NULL,
    applicable_count INTEGER      NOT NULL,
    pass_pct         NUMERIC,                 -- passed_count / applicable_count (0..1)

    -- Category 1 — Quality
    rule_1_1         BOOLEAN,                 -- Operating history (>= 4 annual periods)
    rule_1_2         BOOLEAN,                 -- Earnings consistency (3 of last 4 positive)
    rule_1_3         BOOLEAN,                 -- ROIC: 3yr avg >= sector median AND >= 0.08
    rule_1_4         BOOLEAN,                 -- Gross margin direction (latest >= t-3)

    -- Category 2 — Financial strength
    rule_2_1         BOOLEAN,                 -- Net Debt / EBITDA <= sector median x 1.25
    rule_2_2         BOOLEAN,                 -- Interest coverage >= 5.0
    rule_2_3         BOOLEAN,                 -- FCF consistency (3 of last 4 positive)

    -- Category 3 — Valuation
    rule_3_1         BOOLEAN,                 -- FCF yield >= sector median
    rule_3_2         BOOLEAN,                 -- EV/EBITDA <= sector median x 1.25
    rule_3_3         BOOLEAN,                 -- Margin of safety vs DCF (DEFERRED — always NULL)

    -- Category 4 — Trajectory
    rule_4_1         BOOLEAN,                 -- Revenue growth >= sector median
    rule_4_2         BOOLEAN,                 -- Capital return discipline (dividends or buybacks)

    calculated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT rule_results_pkey PRIMARY KEY (ticker, calculated_at)
);

-- Latest results for one ticker, and category scans (e.g. "show the shortlist")
CREATE INDEX IF NOT EXISTS idx_rule_results_ticker
    ON rule_results (ticker, calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_rule_results_category
    ON rule_results (category, calculated_at DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- Dashboard read access — see the note in 008_create_sector_statistics.sql.
-- Newly-created tables are NOT exposed to anon by default, so the dashboard's
-- anon key reads back empty without this. Mirror ai_analyses / sector_statistics.
-- ─────────────────────────────────────────────────────────────────────────────
ALTER TABLE rule_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Anon and authenticated can read rule_results" ON rule_results;
CREATE POLICY "Anon and authenticated can read rule_results"
    ON rule_results FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON rule_results TO anon, authenticated;
