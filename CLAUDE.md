# Stocks Project — Claude Context

## What this project is

A personal stock analysis dashboard for 5 US tickers (AAPL, GOOGL, MSFT, AMZN, TSLA).
It combines a Streamlit UI, a Supabase PostgreSQL database, yfinance data fetching,
valuation models, and a Claude-powered AI analysis engine.

**Investment goal:** Capital appreciation + dividend income, 1–3 year horizon.

---

## Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (dashboard.py) |
| Database | Supabase (PostgreSQL) |
| Data source | yfinance |
| AI analysis | Anthropic Claude API (`claude-opus-4-7`) |
| Hosting | GitHub Codespaces |
| CI/CD | GitHub Actions (daily data refresh) |

---

## Running the project

```bash
# Start the dashboard
python3 -m streamlit run dashboard.py --server.port 8501 --server.headless true

# Refresh all data in Supabase (prices + fundamentals)
python3 refresh_all.py

# Verify Supabase connection
python3 verify_connection.py
```

Credentials live in `.env` (gitignored). Template: `.env.example`.

Required env vars:
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_KEY`
- `ANTHROPIC_API_KEY` (for AI analysis tab)

---

## File map

### Core dashboard
| File | Purpose |
|---|---|
| `dashboard.py` | Main Streamlit app — 5 tabs: Chart, Compare, Fundamentals, Valuation, Portfolio |
| `indicators.py` | Technical indicators: moving averages, Bollinger Bands, RSI |
| `valuation.py` | Valuation models: DCF, Reverse DCF, Graham Number, historical multiples, signal |

### AI analysis engine
| File | Purpose |
|---|---|
| `analysis_engine.py` | Calls Claude API — streaming analysis, Streamlit display, Supabase data mapper |
| `prompts.py` | System prompt + `build_analysis_prompt()` + example data (MSFT, BCE) |

### Data pipeline
| File | Purpose |
|---|---|
| `refresh_all.py` | Orchestrates full refresh: fetch_stocks → save_fundamentals |
| `fetch_stocks.py` | Pulls 5yr daily OHLCV from yfinance → Supabase `stock_prices` |
| `save_fundamentals.py` | Pulls income, balance, cash flow, snapshot from yfinance → Supabase |
| `fundamentals.py` | Helper functions for fetching/transforming fundamentals data |
| `verify_connection.py` | Quick Supabase connectivity check |

### Utilities
| File | Purpose |
|---|---|
| `fetch_sp500.py` | Fetches S&P 500 company list |
| `export_sp500_excel.py` | Exports S&P 500 data to Excel |

### Config & CI
| File | Purpose |
|---|---|
| `.github/workflows/refresh.yml` | GitHub Actions — runs `refresh_all.py` daily |
| `requirements.txt` | Dashboard dependencies (streamlit, supabase, yfinance, anthropic, etc.) |
| `requirements-refresh.txt` | Minimal deps for the GitHub Actions refresh job |
| `.streamlit/config.toml` | Streamlit theme/server config |

---

## Database schema (Supabase)

Tables created by `migrations/`:

| Table | Description | Rows (approx) |
|---|---|---|
| `companies` | Ticker metadata (name, exchange) | 5 |
| `stock_prices` | Daily OHLCV — 5yr history | ~6,500 |
| `income_statements` | Quarterly + annual P&L | ~60 |
| `balance_sheets` | Quarterly + annual balance sheets | ~65 |
| `cash_flows` | Quarterly + annual cash flows | ~65 |
| `fundamentals_snapshot` | TTM snapshot with valuation multiples + quality metrics | ~225 |
| `portfolio_holdings` | Personal portfolio positions | 0 (user fills in) |

Migrations (run once against Supabase):
- `migrations/001_create_stock_tables.sql`
- `migrations/002_create_fundamentals_tables.sql`
- `migrations/003_create_portfolio.sql`

---

## Dashboard tabs

1. **📊 Chart** — Candlestick + volume + RSI, with MA20/MA50/MA200 and Bollinger Band overlays
2. **⚖️ Compare** — Normalised % return comparison across selected tickers
3. **🏦 Fundamentals** — Valuation multiples, quality metrics (ROE/ROA/ROIC), revenue/margin/cash flow charts
4. **📐 Valuation** — DCF with sliders, P/E + EV/EBITDA + P/S multiples, Reverse DCF, Graham Number, overall BUY/WATCH/AVOID signal
5. **💼 Portfolio** — Add/edit/delete holdings, P&L per position, portfolio signal summary
6. **🤖 AI Analysis** *(to be wired in)* — Full Claude-powered step-by-step analysis using `render_analysis_tab()` from `analysis_engine.py`

---

## AI analysis engine

`analysis_engine.py` wraps the Claude API with:
- **Model:** `claude-opus-4-7`
- **Prompt caching:** System prompt cached with 1h TTL (saves ~90% token cost on repeated analyses)
- **Streaming:** Token-by-token into `st.markdown()` for real-time display
- **Data mapper:** `build_ticker_data_from_supabase()` converts Supabase snapshot rows to the prompt schema

`prompts.py` contains:
- `SYSTEM_PROMPT` — full Investment Valuation Framework v2 (3-step: sector → profile → methods)
- `build_analysis_prompt(data)` — formats company data into a structured analysis request
- `EXAMPLE_DATA_MICROSOFT` / `EXAMPLE_DATA_BCE` — example dicts for testing

To wire the AI tab into `dashboard.py`:
```python
from analysis_engine import render_analysis_tab

with tab6:
    render_analysis_tab(ticker, fundamentals=snap, price_data=snap)
```

---

## Investment framework

Located in `docs/framework/investment_framework_v2.docx`.

**3-step process:**
1. Identify GICS sector (11 sectors)
2. Identify financial profile (8 profiles: Mature FCF Generator, Growth+FCF+, Pre-Profit, Dividend Payer, Asset-Heavy/Regulated, Cyclical, Turnaround, REIT)
3. Apply the right valuation methods per sector × profile combination

**Key rules:**
- Always use 2–3 methods, never just one
- Run bear / base / bull scenarios
- Apply margin of safety: 15–20% (quality) → 25–30% (cyclical) → 40–50% (turnaround)
- Identify a catalyst before buying

Sector deep-dives: `docs/deepdives/` (tech, comms, consumer discretionary).

---

## Tickers tracked

`AAPL`, `GOOGL`, `MSFT`, `AMZN`, `TSLA`

To add more tickers, update the `TICKERS` list in `dashboard.py` and re-run `refresh_all.py`.
