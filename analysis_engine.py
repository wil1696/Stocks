"""
analysis_engine.py
==================
Connects prompts.py to the Claude API and your Supabase fundamentals data.
Drop this into your Stocks project root alongside valuation.py.

Usage in dashboard.py:
    from analysis_engine import render_analysis_tab
    with tab6:
        render_analysis_tab(ticker, snap)
"""

import anthropic
import streamlit as st
from prompts import SYSTEM_PROMPT, build_analysis_prompt


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR BENCHMARKS — peer EV/EBITDA and gross margin by GICS sector
# Used when snapshot doesn't have peer comparison data.
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_BENCHMARKS = {
    "Technology":               {"peer_ev_ebitda": 22.0, "peer_gross_margin": 0.62},
    "Communication Services":   {"peer_ev_ebitda": 14.0, "peer_gross_margin": 0.52},
    "Consumer Discretionary":   {"peer_ev_ebitda": 16.0, "peer_gross_margin": 0.40},
    "Consumer Staples":         {"peer_ev_ebitda": 14.0, "peer_gross_margin": 0.35},
    "Healthcare":               {"peer_ev_ebitda": 16.0, "peer_gross_margin": 0.55},
    "Financials":               {"peer_ev_ebitda": 12.0, "peer_gross_margin": 0.45},
    "Industrials":              {"peer_ev_ebitda": 14.0, "peer_gross_margin": 0.32},
    "Energy":                   {"peer_ev_ebitda": 7.0,  "peer_gross_margin": 0.25},
    "Materials":                {"peer_ev_ebitda": 9.0,  "peer_gross_margin": 0.28},
    "Utilities":                {"peer_ev_ebitda": 11.0, "peer_gross_margin": 0.38},
    "Real Estate":              {"peer_ev_ebitda": 20.0, "peer_gross_margin": 0.60},
}

# Starting classifications for your 5 tickers.
# Claude will validate and can correct these based on the data.
TICKER_CLASSIFICATIONS = {
    "AAPL":  {"sector": "Technology",             "financial_profile": "Mature FCF Generator"},
    "GOOGL": {"sector": "Communication Services", "financial_profile": "Mature FCF Generator"},
    "MSFT":  {"sector": "Technology",             "financial_profile": "Mature FCF Generator"},
    "AMZN":  {"sector": "Consumer Discretionary", "financial_profile": "Growth + FCF Positive"},
    "TSLA":  {"sector": "Consumer Discretionary", "financial_profile": "Growth + FCF Positive"},
}


def build_ticker_data(snap: dict, discount_rate: float = 0.12,
                      sector_override: str = None,
                      profile_override: str = None,
                      analyst_notes: str = None) -> dict:
    """
    Convert a fundamentals_snapshot row into the prompt schema.
    Column names match exactly what get_snapshot() writes to Supabase.
    """
    ticker = snap.get("ticker", "UNKNOWN")
    clf    = TICKER_CLASSIFICATIONS.get(ticker, {})
    sector = sector_override or snap.get("sector") or clf.get("sector")
    profile = profile_override or clf.get("financial_profile")
    benchmarks = SECTOR_BENCHMARKS.get(sector, {})

    return {
        "ticker":           ticker,
        "company_name":     snap.get("company_name", ticker),
        "current_price":    snap.get("price"),
        "currency":         snap.get("currency", "USD"),
        "market":           snap.get("exchange", "NASDAQ"),
        "sector":           sector,
        "financial_profile": profile,
        # Income
        "revenue":              snap.get("revenue_ttm"),
        "revenue_prior_yr":     snap.get("revenue_prior_yr"),
        "revenue_2yr_ago":      snap.get("revenue_2yr_ago"),
        "gross_profit":         snap.get("gross_profit_ttm"),
        "ebitda":               snap.get("ebitda_ttm"),
        "ebit":                 snap.get("operating_income_ttm"),
        "operating_income":     snap.get("operating_income_ttm"),
        "net_income":           snap.get("net_income_ttm"),
        "sbc":                  None,
        "interest_expense":     snap.get("interest_expense_ttm"),
        "tax_rate":             snap.get("tax_rate"),
        # Balance sheet
        "total_debt":           snap.get("total_debt"),
        "cash":                 snap.get("cash"),
        "net_debt":             snap.get("net_debt"),
        "book_value_equity":    snap.get("total_equity"),
        "shares_outstanding":   snap.get("shares_outstanding"),
        "tier1_capital_ratio":  None,
        "npl_ratio":            None,
        # Cash flow
        "operating_cash_flow":  snap.get("cfo_ttm"),
        "capex":                None,
        "free_cash_flow":       snap.get("fcf_ttm"),
        "dividends_paid":       snap.get("dividends_paid_ttm"),
        "buybacks":             snap.get("buybacks_ttm"),
        # Valuation
        "market_cap":           snap.get("market_cap"),
        "enterprise_value":     snap.get("enterprise_value"),
        "pe_ratio":             snap.get("pe_ratio"),
        "ev_ebitda":            snap.get("ev_ebitda"),
        "ev_revenue":           snap.get("ev_revenue"),
        "price_to_book":        snap.get("pb_ratio"),
        "price_to_fcf":         snap.get("price_fcf"),
        "dividend_yield":       snap.get("dividend_yield"),
        "dividend_per_share":   snap.get("dividend_per_share"),
        # Operational
        "revenue_growth_3yr_cagr": snap.get("revenue_cagr_3yr"),
        "gross_margin":         snap.get("gross_margin"),
        "operating_margin":     snap.get("operating_margin"),
        "fcf_margin":           snap.get("fcf_margin"),
        "roic":                 snap.get("roic"),
        "roe":                  snap.get("roe"),
        "roa":                  snap.get("roa"),
        # Composite
        "rule_of_40_score":         snap.get("rule_of_40"),
        "dividend_coverage_ratio":  snap.get("dividend_coverage"),
        # Sector-specific (N/A for current 5 tickers)
        "arr_growth":              None,
        "net_revenue_retention":   None,
        "arpu":                    None,
        "ebitda_after_capex":      None,
        "same_store_sales_growth": None,
        "restaurant_level_margin": None,
        "franchise_pct":           None,
        "ffo_per_share":           None,
        "affo_per_share":          None,
        "affo_payout_ratio":       None,
        "occupancy_rate":          None,
        "nav_per_share":           None,
        "mid_cycle_ebitda":        None,
        "breakeven_price":         None,
        # Context
        "peer_ev_ebitda":    benchmarks.get("peer_ev_ebitda"),
        "peer_gross_margin": benchmarks.get("peer_gross_margin"),
        "discount_rate":     discount_rate,
        "analyst_notes":     analyst_notes or "None provided.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE API — streaming into Streamlit
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(ticker_data: dict) -> str:
    """Stream Claude's response token-by-token into a Streamlit placeholder."""
    client      = anthropic.Anthropic()
    user_prompt = build_analysis_prompt(ticker_data)
    full_text   = ""
    placeholder = st.empty()

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            full_text += chunk
            placeholder.markdown(f"```\n{full_text}\n```")

    return full_text


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT TAB COMPONENT
# ─────────────────────────────────────────────────────────────────────────────

def _load_latest_analysis(ticker: str) -> dict | None:
    """Read the most recent stored analysis for a ticker from Supabase."""
    import os
    from supabase import create_client
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])
    res = (
        client.table("ai_analyses")
        .select("*")
        .eq("ticker", ticker)
        .order("analysis_date", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def render_analysis_tab(ticker: str, snap: dict) -> None:
    """
    🤖 AI Analysis tab. Reads pre-generated analyses from Supabase ai_analyses.

    Analyses are produced in a Claude Code dev session and saved via
    save_analysis.py — no Anthropic API key or credits required.
    """
    st.subheader("🤖 AI Step-by-Step Analysis")
    st.caption(
        "Claude applies the Investment Valuation Framework v2 to each ticker. "
        "Analyses are generated in a dev session and stored in Supabase — "
        "the dashboard reads them instantly, no API key needed."
    )

    if not snap:
        st.warning(
            f"No fundamentals data for **{ticker}**. Run `python3 refresh_all.py`."
        )
        return

    analysis = _load_latest_analysis(ticker)

    if not analysis:
        st.info(
            f"No analysis stored for **{ticker}** yet.\n\n"
            "**To generate one:** open a Claude Code session in this Codespace and ask:\n\n"
            f"> *Generate the AI analysis for {ticker} and save it to Supabase*\n\n"
            "Claude will read the snapshot, apply the framework, and save the result "
            "to the `ai_analyses` table. The dashboard will then display it here."
        )
        return

    # ── Verdict banner ────────────────────────────────────────────────────────
    STYLES = {
        "BUY":   {"bg": "#0a2e1a", "border": "#00d084", "emoji": "✅"},
        "WATCH": {"bg": "#2e2200", "border": "#ffc107", "emoji": "⚠️"},
        "AVOID": {"bg": "#2e0a0a", "border": "#ff4b4b", "emoji": "🚫"},
    }
    verdict = (analysis.get("verdict") or "").upper()
    if verdict in STYLES:
        s = STYLES[verdict]
        st.markdown(
            f'<div style="background:{s["bg"]};border:1px solid {s["border"]}55;'
            f'border-left:5px solid {s["border"]};border-radius:8px;'
            f'padding:18px 22px;margin:16px 0;font-size:20px;'
            f'font-weight:700;color:{s["border"]};">'
            f'{s["emoji"]} &nbsp; {verdict}</div>',
            unsafe_allow_html=True,
        )

    # ── Metadata strip ────────────────────────────────────────────────────────
    meta_bits = [f"📅 {analysis['analysis_date']}"]
    if analysis.get("sector"):
        meta_bits.append(f"🏢 {analysis['sector']}")
    if analysis.get("financial_profile"):
        meta_bits.append(f"📊 {analysis['financial_profile']}")
    if analysis.get("discount_rate"):
        meta_bits.append(f"💰 Discount rate: {float(analysis['discount_rate'])*100:.0f}%")
    st.caption("  ·  ".join(meta_bits))

    # ── Full analysis text ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(analysis["full_text"])

    # ── Download ──────────────────────────────────────────────────────────────
    st.download_button(
        label=f"⬇️  Download {ticker} Analysis (.md)",
        data=analysis["full_text"],
        file_name=f"{ticker}_analysis_{analysis['analysis_date']}.md",
        mime="text/markdown",
        key=f"dl_{ticker}",
    )
