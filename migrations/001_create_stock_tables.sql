-- Migration: 001_create_stock_tables
-- Creates the initial schema for storing historical stock price data for USA companies

-- Table to store company/ticker metadata
CREATE TABLE IF NOT EXISTS companies (
    id          BIGSERIAL PRIMARY KEY,
    ticker      VARCHAR(10)  NOT NULL UNIQUE,
    name        TEXT,
    exchange    VARCHAR(20),
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Table to store daily historical OHLCV stock price data
CREATE TABLE IF NOT EXISTS stock_prices (
    id          BIGSERIAL PRIMARY KEY,
    ticker      VARCHAR(10)  NOT NULL,
    date        DATE         NOT NULL,
    open        NUMERIC(12, 4),
    close       NUMERIC(12, 4),
    high        NUMERIC(12, 4),
    low         NUMERIC(12, 4),
    volume      BIGINT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT stock_prices_ticker_date_key UNIQUE (ticker, date),
    CONSTRAINT stock_prices_ticker_fkey FOREIGN KEY (ticker)
        REFERENCES companies (ticker)
        ON UPDATE CASCADE
        ON DELETE CASCADE
);

-- Index for fast lookups by ticker
CREATE INDEX IF NOT EXISTS idx_stock_prices_ticker ON stock_prices (ticker);

-- Index for fast lookups by date range
CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON stock_prices (date);

-- Index for the most common query pattern: ticker + date range
CREATE INDEX IF NOT EXISTS idx_stock_prices_ticker_date ON stock_prices (ticker, date DESC);
