-- Migration: 004_extend_fundamentals_snapshot
-- Adds new columns required by the Investment Valuation Framework v2 AI engine.
-- Run once in the Supabase SQL editor.

ALTER TABLE fundamentals_snapshot
    -- Identity / metadata
    ADD COLUMN IF NOT EXISTS company_name        TEXT,
    ADD COLUMN IF NOT EXISTS sector              TEXT,
    ADD COLUMN IF NOT EXISTS currency            VARCHAR(10),
    ADD COLUMN IF NOT EXISTS exchange            VARCHAR(20),
    ADD COLUMN IF NOT EXISTS beta                NUMERIC(8, 4),

    -- Income statement (TTM)
    ADD COLUMN IF NOT EXISTS gross_profit_ttm    NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS operating_income_ttm NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS interest_expense_ttm NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS revenue_prior_yr    NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS revenue_2yr_ago     NUMERIC(20, 2),

    -- Cash flow (TTM)
    ADD COLUMN IF NOT EXISTS dividends_paid_ttm  NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS buybacks_ttm        NUMERIC(20, 2),

    -- Balance sheet (latest quarter)
    ADD COLUMN IF NOT EXISTS cash                NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS total_debt          NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS net_debt            NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS total_equity        NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS total_assets        NUMERIC(20, 2),
    ADD COLUMN IF NOT EXISTS tax_rate            NUMERIC(10, 6),

    -- Margin metrics
    ADD COLUMN IF NOT EXISTS gross_margin        NUMERIC(10, 6),
    ADD COLUMN IF NOT EXISTS operating_margin    NUMERIC(10, 6),
    ADD COLUMN IF NOT EXISTS fcf_margin          NUMERIC(10, 6),

    -- Leverage & coverage
    ADD COLUMN IF NOT EXISTS debt_ebitda         NUMERIC(12, 4),
    ADD COLUMN IF NOT EXISTS dividend_coverage   NUMERIC(12, 4),

    -- Dividend
    ADD COLUMN IF NOT EXISTS dividend_yield      NUMERIC(12, 6),
    ADD COLUMN IF NOT EXISTS dividend_per_share  NUMERIC(14, 4),

    -- Growth
    ADD COLUMN IF NOT EXISTS revenue_growth_yoy  NUMERIC(10, 6),
    ADD COLUMN IF NOT EXISTS revenue_cagr_3yr    NUMERIC(10, 6),

    -- Composite metrics
    ADD COLUMN IF NOT EXISTS rule_of_40          NUMERIC(12, 4);
