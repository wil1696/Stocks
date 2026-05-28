-- Migration: 007_extend_companies_universe
-- Extends the existing `companies` table so it can act as the single source of
-- truth for which tickers the data pipeline should process. The S&P 500 seeder
-- (seed_universe.py) upserts into this table; fetch_stocks.py and
-- save_fundamentals.py read `is_active = TRUE` rows from it.
--
-- Why extend instead of create-new: `companies.ticker` is already the FK target
-- for `stock_prices.ticker` (migration 001). Adding sector/sub_industry/etc to
-- the existing table preserves that relationship.
--
-- Run once in the Supabase SQL editor.

ALTER TABLE companies
    -- GICS classification — populated by seed_universe.py from Wikipedia
    ADD COLUMN IF NOT EXISTS company_name  TEXT,
    ADD COLUMN IF NOT EXISTS sector        TEXT,
    ADD COLUMN IF NOT EXISTS sub_industry  TEXT,

    -- Date the ticker entered the S&P 500 (from Wikipedia's "Date added" column)
    ADD COLUMN IF NOT EXISTS date_added    DATE,

    -- Soft-delete flag: tickers removed from the S&P 500 are flipped to FALSE
    -- by the seeder rather than deleted (deletion would CASCADE into
    -- stock_prices and destroy history we may still want).
    ADD COLUMN IF NOT EXISTS is_active     BOOLEAN NOT NULL DEFAULT TRUE;

-- Ensure the 5 pre-existing rows (AAPL, GOOGL, MSFT, AMZN, TSLA) are active.
-- DEFAULT TRUE handles new rows; this UPDATE handles rows that already existed
-- before the ADD COLUMN ran (the default only applies at insert time).
UPDATE companies
SET is_active = TRUE
WHERE is_active IS NULL;

-- Backfill company_name from the existing `name` column so downstream code can
-- read either. seed_universe.py will overwrite both with the Wikipedia name.
UPDATE companies
SET company_name = name
WHERE company_name IS NULL AND name IS NOT NULL;

-- Indexes for the common universe-selection queries
CREATE INDEX IF NOT EXISTS idx_companies_sector    ON companies (sector);
CREATE INDEX IF NOT EXISTS idx_companies_is_active ON companies (is_active);
