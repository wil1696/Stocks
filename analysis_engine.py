"""
analysis_engine.py
==================
Connects prompts.py to the Claude API and your existing Supabase data.
Drop this into your Stocks project alongside valuation.py.

Usage in Streamlit:
    from analysis_engine import run_analysis, display_analysis
    result = run_analysis(ticker_data)
    display_analysis(result)
"""

import anthropic
import streamlit as st
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT, build_analysis_prompt

load_dotenv()

MODEL = "claude-opus-4-7"

# System prompt cached with 1-hour TTL — it never changes between requests.
# The 1h TTL pays off after 3+ analyses per hour at a 2x write / 0.1x read cost.
_SYSTEM_CACHED = [
    {
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# CORE API CALL
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(ticker_data: dict, stream: bool = True) -> str:
    """
    Send ticker data to Claude and return the full step-by-step analysis.

    Parameters
    ----------
    ticker_data : dict
        Financial data dict — see prompts.py DATA SCHEMA for all fields.
    stream : bool
        If True, streams the response token-by-token into Streamlit.
        If False, waits for complete response (useful for batch processing).

    Returns
    -------
    str
        Full analysis text from Claude.
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    user_prompt = build_analysis_prompt(ticker_data)

    if stream:
        return _run_streaming(client, user_prompt)
    else:
        return _run_blocking(client, user_prompt)


def _run_blocking(client: anthropic.Anthropic, user_prompt: str) -> str:
    """Single API call — waits for full response before returning."""
    message = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=_SYSTEM_CACHED,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


def _run_streaming(client: anthropic.Anthropic, user_prompt: str) -> str:
    """
    Streams Claude's response into a Streamlit container token-by-token.
    Shows the analysis appearing in real-time — much better UX than waiting
    for the full response.

    Must be called from within a Streamlit app context.
    """
    full_text = ""
    placeholder = st.empty()

    with client.messages.stream(
        model=MODEL,
        max_tokens=8192,
        system=_SYSTEM_CACHED,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        for text_chunk in stream.text_stream:
            full_text += text_chunk
            placeholder.markdown(full_text)

    return full_text


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def display_analysis(ticker: str, analysis_text: str) -> None:
    """
    Render the Claude analysis in your Streamlit dashboard.
    Parses the plain-text output into sections with visual separation.

    Parameters
    ----------
    ticker : str
        Ticker symbol for the header.
    analysis_text : str
        Full analysis text returned by run_analysis().
    """
    st.markdown("---")
    st.subheader(f"📊 Full Analysis: {ticker}")

    # ── Verdict extraction ─────────────────────────────────────────────────
    # Parse the structured verdict block for a visual callout at the top
    verdict_color = {"BUY": "green", "WATCH": "orange", "AVOID": "red"}
    verdict_emoji = {"BUY": "✅", "WATCH": "⚠️", "AVOID": "🚫"}

    detected_verdict = None
    for v in ["BUY", "WATCH", "AVOID"]:
        if f"VERDICT: {v}" in analysis_text:
            detected_verdict = v
            break

    if detected_verdict:
        color = verdict_color.get(detected_verdict, "gray")
        emoji = verdict_emoji.get(detected_verdict, "")
        st.markdown(
            f"""
            <div style="
                background-color: {'#d4edda' if color=='green' else '#fff3cd' if color=='orange' else '#f8d7da'};
                border-left: 6px solid {'#28a745' if color=='green' else '#ffc107' if color=='orange' else '#dc3545'};
                padding: 16px 20px;
                border-radius: 6px;
                margin-bottom: 16px;
                font-size: 18px;
                font-weight: bold;
            ">
                {emoji} {detected_verdict}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Full analysis in a scrollable code block ───────────────────────────
    # Using st.text preserves the ASCII formatting and monospace layout
    st.text(analysis_text)

    # ── Download button ────────────────────────────────────────────────────
    st.download_button(
        label=f"⬇️ Download {ticker} Analysis",
        data=analysis_text,
        file_name=f"{ticker}_analysis.txt",
        mime="text/plain",
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA BUILDER — bridges your Supabase data to the prompt schema
# ─────────────────────────────────────────────────────────────────────────────

def build_ticker_data_from_supabase(
    fundamentals: dict,
    price_data: dict,
    sector: str = None,
    financial_profile: str = None,
    discount_rate: float = 0.12,
    analyst_notes: str = None,
) -> dict:
    """
    Convert your existing Supabase fundamentals row into the ticker_data
    dict expected by build_analysis_prompt().

    Parameters
    ----------
    fundamentals : dict
        Row from your fundamentals table in Supabase.
        Adjust the key names below to match your actual column names.
    price_data : dict
        Current price and market cap data (from yfinance or your price table).
    sector : str, optional
        Override sector classification. If None, Claude will classify.
    financial_profile : str, optional
        Override financial profile. If None, Claude will classify.
    discount_rate : float
        Your required rate of return. Default 12%.
    analyst_notes : str, optional
        Any extra context you want Claude to consider.

    Returns
    -------
    dict
        ticker_data dict ready for build_analysis_prompt().

    ── HOW TO ADAPT THIS TO YOUR SCHEMA ─────────────────────────────────────
    Look at the key names on the right side of each assignment below.
    Change them to match your actual Supabase column names.
    For example, if your column is called "total_revenue" instead of "revenue",
    change:   fundamentals.get("revenue")
    to:       fundamentals.get("total_revenue")

    Any field that is None will show as DATA MISSING in the prompt — Claude
    will note this and use industry benchmarks.
    ─────────────────────────────────────────────────────────────────────────
    """
    ticker = fundamentals.get("ticker") or price_data.get("ticker", "UNKNOWN")

    # Derive calculated fields if not stored directly
    revenue      = fundamentals.get("revenue") or fundamentals.get("total_revenue")
    gross_profit = fundamentals.get("gross_profit")
    gross_margin = (
        fundamentals.get("gross_margin")
        or (gross_profit / revenue if revenue and gross_profit else None)
    )
    ocf   = fundamentals.get("operating_cash_flow") or fundamentals.get("cash_from_operations")
    capex = fundamentals.get("capex") or fundamentals.get("capital_expenditure")
    fcf   = (
        fundamentals.get("free_cash_flow")
        or (ocf - capex if ocf and capex else None)
    )
    net_debt = (
        fundamentals.get("net_debt")
        or (
            (fundamentals.get("total_debt", 0) or 0)
            - (fundamentals.get("cash", 0) or 0)
        )
        or None
    )
    ev = (
        price_data.get("enterprise_value")
        or fundamentals.get("enterprise_value")
    )
    market_cap = price_data.get("market_cap") or fundamentals.get("market_cap")

    # Dividend coverage = FCF / annual dividends paid
    dividends = fundamentals.get("dividends_paid")
    div_coverage = (fcf / dividends if fcf and dividends and dividends > 0 else None)

    return {
        # Identity
        "ticker":           ticker,
        "company_name":     fundamentals.get("company_name") or fundamentals.get("name", ticker),
        "current_price":    price_data.get("current_price") or price_data.get("price"),
        "currency":         fundamentals.get("currency", "USD"),
        "market":           fundamentals.get("exchange") or price_data.get("exchange", "Unknown"),

        # Classification
        "sector":           sector or fundamentals.get("sector"),
        "financial_profile": financial_profile or fundamentals.get("financial_profile"),

        # Income statement
        "revenue":              revenue,
        "revenue_prior_yr":     fundamentals.get("revenue_prior_yr") or fundamentals.get("revenue_1yr_ago"),
        "revenue_2yr_ago":      fundamentals.get("revenue_2yr_ago"),
        "gross_profit":         gross_profit,
        "ebitda":               fundamentals.get("ebitda"),
        "ebit":                 fundamentals.get("ebit"),
        "operating_income":     fundamentals.get("operating_income") or fundamentals.get("ebit"),
        "net_income":           fundamentals.get("net_income"),
        "sbc":                  fundamentals.get("sbc") or fundamentals.get("stock_based_compensation"),
        "interest_expense":     fundamentals.get("interest_expense"),
        "tax_rate":             fundamentals.get("effective_tax_rate", 0.21),

        # Balance sheet
        "total_debt":           fundamentals.get("total_debt"),
        "cash":                 fundamentals.get("cash") or fundamentals.get("cash_and_equivalents"),
        "net_debt":             net_debt,
        "book_value_equity":    fundamentals.get("book_value") or fundamentals.get("shareholders_equity"),
        "shares_outstanding":   fundamentals.get("shares_outstanding"),
        "tier1_capital_ratio":  fundamentals.get("tier1_capital_ratio"),
        "npl_ratio":            fundamentals.get("npl_ratio"),

        # Cash flow
        "operating_cash_flow":  ocf,
        "capex":                capex,
        "free_cash_flow":       fcf,
        "dividends_paid":       dividends,
        "buybacks":             fundamentals.get("buybacks") or fundamentals.get("share_repurchases"),

        # Valuation
        "market_cap":           market_cap,
        "enterprise_value":     ev,
        "pe_ratio":             price_data.get("pe_ratio") or fundamentals.get("pe_ratio"),
        "ev_ebitda":            (
            fundamentals.get("ev_ebitda")
            or (ev / fundamentals.get("ebitda") if ev and fundamentals.get("ebitda") else None)
        ),
        "ev_revenue":           (
            fundamentals.get("ev_revenue")
            or (ev / revenue if ev and revenue else None)
        ),
        "price_to_book":        price_data.get("price_to_book") or fundamentals.get("price_to_book"),
        "price_to_fcf":         (
            fundamentals.get("price_to_fcf")
            or (market_cap / fcf if market_cap and fcf and fcf > 0 else None)
        ),
        "dividend_yield":       fundamentals.get("dividend_yield") or price_data.get("dividend_yield"),
        "dividend_per_share":   fundamentals.get("dividend_per_share"),

        # Operational
        "revenue_growth_3yr_cagr": fundamentals.get("revenue_cagr_3yr") or fundamentals.get("revenue_growth_3yr"),
        "gross_margin":         gross_margin,
        "operating_margin":     (
            fundamentals.get("operating_margin")
            or (fundamentals.get("operating_income") / revenue if fundamentals.get("operating_income") and revenue else None)
        ),
        "fcf_margin":           (
            fundamentals.get("fcf_margin")
            or (fcf / revenue if fcf and revenue else None)
        ),
        "roic":                 fundamentals.get("roic") or fundamentals.get("return_on_invested_capital"),
        "roe":                  fundamentals.get("roe") or fundamentals.get("return_on_equity"),
        "roa":                  fundamentals.get("roa") or fundamentals.get("return_on_assets"),

        # Sector-specific
        "arr_growth":           fundamentals.get("arr_growth"),
        "net_revenue_retention":fundamentals.get("net_revenue_retention") or fundamentals.get("nrr"),
        "rule_of_40_score":     fundamentals.get("rule_of_40"),
        "arpu":                 fundamentals.get("arpu"),
        "ebitda_after_capex":   (
            fundamentals.get("ebitda_after_capex")
            or (fundamentals.get("ebitda") - capex if fundamentals.get("ebitda") and capex else None)
        ),
        "dividend_coverage_ratio": div_coverage,
        "same_store_sales_growth": fundamentals.get("same_store_sales_growth") or fundamentals.get("sss_growth"),
        "restaurant_level_margin": fundamentals.get("restaurant_level_margin"),
        "franchise_pct":           fundamentals.get("franchise_pct") or fundamentals.get("franchised_pct"),
        "ffo_per_share":           fundamentals.get("ffo_per_share"),
        "affo_per_share":          fundamentals.get("affo_per_share"),
        "affo_payout_ratio":       fundamentals.get("affo_payout_ratio"),
        "occupancy_rate":          fundamentals.get("occupancy_rate"),
        "nav_per_share":           fundamentals.get("nav_per_share"),
        "mid_cycle_ebitda":        fundamentals.get("mid_cycle_ebitda"),
        "breakeven_price":         fundamentals.get("breakeven_price"),

        # Context
        "peer_ev_ebitda":      fundamentals.get("peer_ev_ebitda") or fundamentals.get("sector_avg_ev_ebitda"),
        "peer_gross_margin":   fundamentals.get("peer_gross_margin") or fundamentals.get("sector_avg_gross_margin"),
        "discount_rate":       discount_rate,
        "analyst_notes":       analyst_notes or fundamentals.get("notes", "None provided."),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT COMPONENT — drop directly into your dashboard.py
# ─────────────────────────────────────────────────────────────────────────────

def render_analysis_tab(
    ticker: str,
    fundamentals: dict,
    price_data: dict,
    sector: str = None,
    financial_profile: str = None,
    discount_rate: float = 0.12,
) -> None:
    """
    Complete Streamlit tab component for the AI analysis feature.
    Call this from your dashboard.py inside a st.tab() block.

    Example usage in dashboard.py:
    ─────────────────────────────────────────────────────────────────
    from analysis_engine import render_analysis_tab

    tab1, tab2, tab3 = st.tabs(["Overview", "Valuation", "AI Analysis"])
    with tab3:
        render_analysis_tab(
            ticker=selected_ticker,
            fundamentals=fundamentals_row,
            price_data=price_row,
            sector=selected_sector,
            financial_profile=selected_profile,
            discount_rate=0.12,
        )
    ─────────────────────────────────────────────────────────────────
    """
    st.subheader("🤖 AI-Powered Step-by-Step Analysis")
    st.caption(
        "Uses Claude to apply the Investment Valuation Framework — "
        "showing every calculation, explaining why each metric matters, "
        "and delivering a Buy / Watch / Avoid verdict."
    )

    # ── Configuration sidebar ──────────────────────────────────────────────
    with st.expander("⚙️ Analysis Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            discount_rate = st.slider(
                "Required Return (Discount Rate)",
                min_value=0.07,
                max_value=0.18,
                value=discount_rate,
                step=0.01,
                format="%.0f%%",
                help="Your minimum required annual return. Higher = more demanding.",
            )
            sector_override = st.selectbox(
                "Sector Override",
                options=[
                    "Auto-detect",
                    "Technology",
                    "Communication Services",
                    "Healthcare",
                    "Financials",
                    "Energy",
                    "Materials",
                    "Consumer Staples",
                    "Consumer Discretionary",
                    "Industrials",
                    "Utilities",
                    "Real Estate",
                ],
                index=0,
            )
        with col2:
            profile_override = st.selectbox(
                "Financial Profile Override",
                options=[
                    "Auto-detect",
                    "Mature FCF Generator",
                    "Growth + FCF Positive",
                    "Pre-Profit / High Growth",
                    "Dividend Payer",
                    "Asset-Heavy / Regulated",
                    "Cyclical Business",
                    "Turnaround / Distressed",
                    "REIT",
                ],
                index=0,
            )
            analyst_notes = st.text_area(
                "Analyst Notes (optional)",
                placeholder="Add any extra context — recent news, one-time items, management guidance...",
                height=100,
            )

    # ── Run analysis button ────────────────────────────────────────────────
    if st.button(
        f"▶ Run Full Analysis for {ticker}",
        type="primary",
        use_container_width=True,
    ):
        ticker_data = build_ticker_data_from_supabase(
            fundamentals=fundamentals,
            price_data=price_data,
            sector=None if sector_override == "Auto-detect" else sector_override,
            financial_profile=None if profile_override == "Auto-detect" else profile_override,
            discount_rate=discount_rate,
            analyst_notes=analyst_notes if analyst_notes else None,
        )

        with st.spinner(f"Analyzing {ticker}... (this takes 15-30 seconds)"):
            try:
                # Streaming renders token-by-token directly into the page
                analysis = run_analysis(ticker_data, stream=True)
                # Store in session state so it persists across reruns
                st.session_state[f"analysis_{ticker}"] = analysis
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

    # ── Display stored analysis ────────────────────────────────────────────
    if f"analysis_{ticker}" in st.session_state:
        display_analysis(ticker, st.session_state[f"analysis_{ticker}"])
