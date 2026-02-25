-- Migration: 003_create_portfolio
-- Personal portfolio holdings: one row per position.

CREATE TABLE IF NOT EXISTS portfolio_holdings (
    id          BIGSERIAL    PRIMARY KEY,
    ticker      VARCHAR(10)  NOT NULL UNIQUE,
    shares      NUMERIC(14, 4) NOT NULL CHECK (shares > 0),
    avg_cost    NUMERIC(14, 4) NOT NULL CHECK (avg_cost > 0),
    notes       TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
