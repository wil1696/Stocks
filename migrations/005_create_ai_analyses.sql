-- Migration: 005_create_ai_analyses
-- Stores AI-generated step-by-step valuation analyses produced by Claude
-- in dev sessions (not via the Anthropic API). Dashboard reads from here.

CREATE TABLE IF NOT EXISTS ai_analyses (
    id                BIGSERIAL PRIMARY KEY,
    ticker            VARCHAR(10)  NOT NULL,
    analysis_date     DATE         NOT NULL,
    verdict           VARCHAR(10),         -- 'BUY' | 'WATCH' | 'AVOID' | NULL
    sector            TEXT,
    financial_profile TEXT,
    discount_rate     NUMERIC(6, 4),
    full_text         TEXT         NOT NULL,
    generated_by      TEXT         DEFAULT 'claude-code-session',
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT ai_analyses_ukey UNIQUE (ticker, analysis_date)
);

CREATE INDEX IF NOT EXISTS idx_ai_analyses_ticker
    ON ai_analyses (ticker, analysis_date DESC);
