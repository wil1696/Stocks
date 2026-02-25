-- Migration: 002_create_fundamentals_tables
-- Stores quarterly/annual income statements, balance sheets, cash flows,
-- and a computed TTM snapshot with valuation and quality metrics.

-- ─────────────────────────────────────────────────────────────────────────────
-- Income Statements
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS income_statements (
    id                   BIGSERIAL PRIMARY KEY,
    ticker               VARCHAR(10)  NOT NULL,
    period_end           DATE         NOT NULL,
    period_type          VARCHAR(10)  NOT NULL CHECK (period_type IN ('annual', 'quarterly')),

    revenue              NUMERIC(20, 2),
    revenue_growth_yoy   NUMERIC(10, 6),   -- decimal  e.g. 0.12 = 12%
    revenue_growth_qoq   NUMERIC(10, 6),

    gross_profit         NUMERIC(20, 2),
    gross_margin         NUMERIC(10, 6),

    operating_income     NUMERIC(20, 2),
    operating_margin     NUMERIC(10, 6),

    ebitda               NUMERIC(20, 2),
    ebitda_margin        NUMERIC(10, 6),

    net_income           NUMERIC(20, 2),
    net_margin           NUMERIC(10, 6),

    eps_basic            NUMERIC(14, 4),
    eps_diluted          NUMERIC(14, 4),
    shares_basic         BIGINT,
    shares_diluted       BIGINT,

    interest_expense     NUMERIC(20, 2),
    tax_rate             NUMERIC(10, 6),

    created_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT income_statements_ukey UNIQUE (ticker, period_end, period_type)
);

CREATE INDEX IF NOT EXISTS idx_income_ticker      ON income_statements (ticker);
CREATE INDEX IF NOT EXISTS idx_income_period_end  ON income_statements (period_end DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- Balance Sheets
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS balance_sheets (
    id                   BIGSERIAL PRIMARY KEY,
    ticker               VARCHAR(10)  NOT NULL,
    period_end           DATE         NOT NULL,
    period_type          VARCHAR(10)  NOT NULL CHECK (period_type IN ('annual', 'quarterly')),

    total_assets         NUMERIC(20, 2),
    current_assets       NUMERIC(20, 2),
    cash_and_equivalents NUMERIC(20, 2),
    total_receivables    NUMERIC(20, 2),
    inventory            NUMERIC(20, 2),

    total_liabilities    NUMERIC(20, 2),
    current_liabilities  NUMERIC(20, 2),
    total_debt           NUMERIC(20, 2),
    long_term_debt       NUMERIC(20, 2),

    total_equity         NUMERIC(20, 2),
    retained_earnings    NUMERIC(20, 2),

    -- Derived
    net_debt             NUMERIC(20, 2),
    current_ratio        NUMERIC(10, 6),
    quick_ratio          NUMERIC(10, 6),
    debt_to_equity       NUMERIC(10, 6),
    debt_to_assets       NUMERIC(10, 6),

    created_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT balance_sheets_ukey UNIQUE (ticker, period_end, period_type)
);

CREATE INDEX IF NOT EXISTS idx_balance_ticker     ON balance_sheets (ticker);
CREATE INDEX IF NOT EXISTS idx_balance_period_end ON balance_sheets (period_end DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- Cash Flows
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cash_flows (
    id                   BIGSERIAL PRIMARY KEY,
    ticker               VARCHAR(10)  NOT NULL,
    period_end           DATE         NOT NULL,
    period_type          VARCHAR(10)  NOT NULL CHECK (period_type IN ('annual', 'quarterly')),

    operating_cash_flow  NUMERIC(20, 2),
    capex                NUMERIC(20, 2),   -- stored as negative (as reported)
    free_cash_flow       NUMERIC(20, 2),   -- OCF + capex  (or direct if available)

    dividends_paid       NUMERIC(20, 2),
    share_buybacks       NUMERIC(20, 2),
    net_change_in_cash   NUMERIC(20, 2),

    created_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT cash_flows_ukey UNIQUE (ticker, period_end, period_type)
);

CREATE INDEX IF NOT EXISTS idx_cashflow_ticker     ON cash_flows (ticker);
CREATE INDEX IF NOT EXISTS idx_cashflow_period_end ON cash_flows (period_end DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- Fundamentals Snapshot  (TTM-based, one row per ticker per run date)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fundamentals_snapshot (
    id                   BIGSERIAL PRIMARY KEY,
    ticker               VARCHAR(10)  NOT NULL,
    snapshot_date        DATE         NOT NULL,

    -- Market data
    price                NUMERIC(14, 4),
    market_cap           NUMERIC(20, 2),
    enterprise_value     NUMERIC(20, 2),
    shares_outstanding   BIGINT,

    -- TTM financials (sum of last 4 quarters)
    revenue_ttm          NUMERIC(20, 2),
    ebitda_ttm           NUMERIC(20, 2),
    net_income_ttm       NUMERIC(20, 2),
    cfo_ttm              NUMERIC(20, 2),
    fcf_ttm              NUMERIC(20, 2),

    -- Valuation multiples
    pe_ratio             NUMERIC(12, 4),
    ps_ratio             NUMERIC(12, 4),
    pb_ratio             NUMERIC(12, 4),
    ev_ebitda            NUMERIC(12, 4),
    ev_revenue           NUMERIC(12, 4),
    peg_ratio            NUMERIC(12, 4),
    fcf_yield            NUMERIC(12, 6),
    price_fcf            NUMERIC(12, 4),

    -- Quality metrics
    roe                  NUMERIC(12, 6),
    roa                  NUMERIC(12, 6),
    roic                 NUMERIC(12, 6),
    fcf_conversion       NUMERIC(12, 6),  -- FCF / Net Income

    created_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT fundamentals_snapshot_ukey UNIQUE (ticker, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_snapshot_ticker ON fundamentals_snapshot (ticker);
CREATE INDEX IF NOT EXISTS idx_snapshot_date   ON fundamentals_snapshot (snapshot_date DESC);
