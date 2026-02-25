import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from supabase import create_client

from indicators import add_bollinger_bands, add_moving_averages, add_rsi
from valuation import (
    METHOD_INFO, GRAHAM_INFO, SIGNAL_CONFIG,
    dcf_intrinsic, reverse_dcf, graham_number,
    compute_historical_multiples, multiples_implied_prices, valuation_summary,
    compute_dcf_suggestions,
)

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global constants ──────────────────────────────────────────────────────────
TICKERS      = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
PLOTLY_THEME = "plotly_dark"
BG_COLOR     = "#0e1117"
CARD_COLOR   = "#1a1d23"
GRID_COLOR   = "#2a2d36"
GREEN        = "#00d084"
RED          = "#ff4b4b"
BLUE         = "#00d4ff"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .stat-card {{
    background: {CARD_COLOR};
    border-radius: 10px;
    padding: 18px 16px 14px;
    text-align: center;
    margin-bottom: 4px;
  }}
  .stat-label {{
    font-size: 11px;
    color: #8b8fa8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
  }}
  .stat-value {{
    font-size: 24px;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.1;
  }}
  .stat-sub {{
    font-size: 12px;
    margin-top: 4px;
  }}
  .green {{ color: {GREEN}; }}
  .red   {{ color: {RED};   }}
</style>
""", unsafe_allow_html=True)


# ── Supabase client ───────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])


@st.cache_resource
def get_write_client():
    """Service-role client used for portfolio writes (insert / update / delete)."""
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


# ── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    res = (
        get_client()
        .table("stock_prices")
        .select("date, open, high, low, close, volume")
        .eq("ticker", ticker)
        .gte("date", start)
        .lte("date", end)
        .order("date")
        .execute()
    )
    df = pd.DataFrame(res.data)
    if df.empty:
        return df
    df["date"]   = pd.to_datetime(df["date"])
    df["volume"] = df["volume"].astype(int)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


@st.cache_data(ttl=3600)
def load_annual_income(ticker: str) -> pd.DataFrame:
    res = (
        get_client()
        .table("income_statements")
        .select("period_end, revenue, ebitda, net_income, eps_diluted, shares_diluted, interest_expense, tax_rate")
        .eq("ticker", ticker)
        .eq("period_type", "annual")
        .order("period_end", desc=False)
        .execute()
    )
    df = pd.DataFrame(res.data)
    if not df.empty:
        df["period_end"] = pd.to_datetime(df["period_end"])
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_annual_balance(ticker: str) -> pd.DataFrame:
    res = (
        get_client()
        .table("balance_sheets")
        .select("period_end, total_debt, cash_and_equivalents, total_equity")
        .eq("ticker", ticker)
        .eq("period_type", "annual")
        .order("period_end", desc=False)
        .execute()
    )
    df = pd.DataFrame(res.data)
    if not df.empty:
        df["period_end"] = pd.to_datetime(df["period_end"])
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_latest_balance(ticker: str) -> dict:
    """Returns the most recent quarterly balance sheet row as a dict."""
    res = (
        get_client()
        .table("balance_sheets")
        .select("total_equity, total_debt, cash_and_equivalents")
        .eq("ticker", ticker)
        .eq("period_type", "quarterly")
        .order("period_end", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else {}


@st.cache_data(ttl=3600)
def load_all_prices(ticker: str) -> pd.DataFrame:
    from datetime import date, timedelta
    start = date.today() - timedelta(days=6 * 365)
    return load_data(ticker, str(start), str(date.today()))


@st.cache_data(ttl=3600)
def load_portfolio() -> pd.DataFrame:
    res = get_client().table("portfolio_holdings").select("*").order("ticker").execute()
    return pd.DataFrame(res.data)


def save_holding(ticker: str, shares: float, avg_cost: float, notes: str = "") -> None:
    get_write_client().table("portfolio_holdings").upsert(
        {"ticker": ticker, "shares": shares, "avg_cost": avg_cost,
         "notes": notes, "updated_at": "now()"},
        on_conflict="ticker",
    ).execute()
    st.cache_data.clear()


def delete_holding(ticker: str) -> None:
    get_write_client().table("portfolio_holdings").delete().eq("ticker", ticker).execute()
    st.cache_data.clear()


@st.cache_data(ttl=3600)
def load_snapshot(ticker: str) -> dict:
    res = (
        get_client()
        .table("fundamentals_snapshot")
        .select("*")
        .eq("ticker", ticker)
        .order("snapshot_date", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else {}


@st.cache_data(ttl=3600)
def load_income_stmts(ticker: str) -> pd.DataFrame:
    res = (
        get_client()
        .table("income_statements")
        .select("period_end, revenue, gross_margin, operating_margin, net_margin, net_income, ebitda")
        .eq("ticker", ticker)
        .eq("period_type", "quarterly")
        .order("period_end", desc=False)
        .execute()
    )
    df = pd.DataFrame(res.data)
    if not df.empty:
        df["period_end"] = pd.to_datetime(df["period_end"])
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_cash_flows(ticker: str) -> pd.DataFrame:
    res = (
        get_client()
        .table("cash_flows")
        .select("period_end, operating_cash_flow, free_cash_flow")
        .eq("ticker", ticker)
        .eq("period_type", "quarterly")
        .order("period_end", desc=False)
        .execute()
    )
    df = pd.DataFrame(res.data)
    if not df.empty:
        df["period_end"] = pd.to_datetime(df["period_end"])
        for col in ["operating_cash_flow", "free_cash_flow"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_annual_cashflows(ticker: str) -> pd.DataFrame:
    res = (
        get_client()
        .table("cash_flows")
        .select("period_end, free_cash_flow")
        .eq("ticker", ticker)
        .eq("period_type", "annual")
        .order("period_end", desc=False)
        .execute()
    )
    df = pd.DataFrame(res.data)
    if not df.empty:
        df["period_end"]     = pd.to_datetime(df["period_end"])
        df["free_cash_flow"] = pd.to_numeric(df["free_cash_flow"], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_beta(ticker: str) -> "float | None":
    import yfinance as yf
    v = yf.Ticker(ticker).info.get("beta")
    return float(v) if v is not None else None


# ── Helpers ───────────────────────────────────────────────────────────────────
def stat_card(label: str, value: str, sub: str = "", sub_class: str = "") -> None:
    sub_html = f'<div class="stat-sub {sub_class}">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="stat-card">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def fmt_volume(v: float) -> str:
    if v >= 1e9:
        return f"{v / 1e9:.2f}B"
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    return f"{v / 1e3:.0f}K"


def fmt_large(v) -> str:
    """Format large financial values into B/M strings."""
    if v is None:
        return "N/A"
    v = float(v)
    if abs(v) >= 1e12:
        return f"${v / 1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.1f}M"
    return f"${v:,.0f}"


def fmt_multiple(v, suffix="x") -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.1f}{suffix}"


def fmt_pct(v, decimals=1) -> str:
    if v is None:
        return "N/A"
    return f"{float(v) * 100:.{decimals}f}%"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Stock Dashboard")
    st.divider()

    ticker = st.selectbox("Ticker", TICKERS)

    period = st.radio("Period", ["1M", "3M", "6M", "1Y", "2Y", "5Y"], index=3, horizontal=True)
    period_days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730, "5Y": 1825}
    end_dt   = datetime.today().date()
    start_dt = end_dt - timedelta(days=period_days[period])

    st.divider()
    st.subheader("Overlays")
    show_ma20  = st.checkbox("MA 20",            value=True)
    show_ma50  = st.checkbox("MA 50",            value=True)
    show_ma200 = st.checkbox("MA 200",           value=False)
    show_bb    = st.checkbox("Bollinger Bands",  value=False)


# ── Load & process main data ──────────────────────────────────────────────────
df = load_data(ticker, str(start_dt), str(end_dt))

if df.empty:
    st.warning(f"No data found for **{ticker}** in the selected date range.")
    st.stop()

df = add_moving_averages(df)
df = add_bollinger_bands(df)
df = add_rsi(df)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Chart", "⚖️  Compare", "🏦  Fundamentals",
    "📐  Valuation", "💼  Portfolio",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHART
# ════════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Stat cards ────────────────────────────────────────────────────────────
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else latest

    price_delta     = latest["close"] - prev["close"]
    price_delta_pct = (price_delta / prev["close"]) * 100
    delta_class     = "green" if price_delta >= 0 else "red"
    delta_arrow     = "▲" if price_delta >= 0 else "▼"

    df_52w   = df[df["date"] >= df["date"].max() - pd.Timedelta(days=365)]
    w52_high = df_52w["high"].max()
    w52_low  = df_52w["low"].min()

    avg_vol    = df["volume"].mean()
    volatility = df["close"].pct_change().dropna().std() * (252 ** 0.5) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        stat_card(
            "Current Price",
            f"${latest['close']:.2f}",
            f"{delta_arrow} ${abs(price_delta):.2f} ({price_delta_pct:+.2f}%)",
            delta_class,
        )
    with c2:
        stat_card("52W High", f"${w52_high:.2f}")
    with c3:
        stat_card("52W Low",  f"${w52_low:.2f}")
    with c4:
        stat_card("Avg Volume", fmt_volume(avg_vol))
    with c5:
        stat_card("Volatility (Ann.)", f"{volatility:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main chart ────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.60, 0.18, 0.22],
        vertical_spacing=0.03,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name=ticker,
        increasing_line_color=GREEN,
        decreasing_line_color=RED,
        whiskerwidth=0.5,
    ), row=1, col=1)

    # Moving averages
    ma_styles = {
        "ma20":  ("MA 20",  "#f0b429", 1.5),
        "ma50":  ("MA 50",  BLUE,      1.5),
        "ma200": ("MA 200", "#b794f4", 1.5),
    }
    if show_ma20:
        name, color, w = ma_styles["ma20"]
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"],  name=name,
                                 line=dict(color=color, width=w)), row=1, col=1)
    if show_ma50:
        name, color, w = ma_styles["ma50"]
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma50"],  name=name,
                                 line=dict(color=color, width=w)), row=1, col=1)
    if show_ma200:
        name, color, w = ma_styles["ma200"]
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma200"], name=name,
                                 line=dict(color=color, width=w)), row=1, col=1)

    # Bollinger Bands
    if show_bb:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["bb_upper"], name="BB Upper",
            line=dict(color="rgba(150,150,255,0.6)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["bb_lower"], name="BB Lower",
            line=dict(color="rgba(150,150,255,0.6)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(150,150,255,0.05)",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["bb_mid"], name="BB Mid",
            line=dict(color="rgba(150,150,255,0.4)", width=1, dash="dash"),
        ), row=1, col=1)

    # Volume bars
    bar_colors = [GREEN if c >= o else RED
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"],
        name="Volume", marker_color=bar_colors,
        showlegend=False, opacity=0.7,
    ), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["rsi"], name="RSI",
        line=dict(color="#f0b429", width=1.5),
    ), row=3, col=1)
    fig.add_hline(y=70, line=dict(color="rgba(255,75,75,0.5)",  dash="dash", width=1), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="rgba(0,208,132,0.5)",  dash="dash", width=1), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,75,75,0.04)",  line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,208,132,0.04)",  line_width=0, row=3, col=1)

    # Layout
    fig.update_layout(
        template=PLOTLY_THEME,
        height=750,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=12)),
        xaxis_rangeslider_visible=False,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        hovermode="x unified",
    )
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_xaxes(gridcolor=GRID_COLOR, showgrid=True)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume",      row=2, col=1)
    fig.update_yaxes(title_text="RSI",         row=3, col=1, range=[0, 100])

    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    selected = st.multiselect(
        "Select tickers to compare",
        TICKERS,
        default=["AAPL", "MSFT"],
    )

    if len(selected) < 2:
        st.info("Select at least 2 tickers to compare.")
    else:
        fig2 = go.Figure()

        LINE_COLORS = [BLUE, "#f0b429", GREEN, "#b794f4", RED]

        for i, t in enumerate(selected):
            df_t = load_data(t, str(start_dt), str(end_dt))
            if df_t.empty:
                st.warning(f"No data for {t} in the selected range.")
                continue

            base       = df_t.iloc[0]["close"]
            pct_return = ((df_t["close"] - base) / base) * 100

            fig2.add_trace(go.Scatter(
                x=df_t["date"],
                y=pct_return,
                name=t,
                mode="lines",
                line=dict(color=LINE_COLORS[i % len(LINE_COLORS)], width=2),
                hovertemplate=(
                    f"<b>{t}</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Return: %{y:.2f}%<extra></extra>"
                ),
            ))

        fig2.add_hline(
            y=0,
            line=dict(color="rgba(255,255,255,0.15)", dash="dash", width=1),
        )

        fig2.update_layout(
            template=PLOTLY_THEME,
            height=520,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title="Return (%)",
            xaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="left", x=0, font=dict(size=13)),
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=BG_COLOR,
            hovermode="x unified",
        )
        fig2.update_xaxes(gridcolor=GRID_COLOR)
        fig2.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)

        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — FUNDAMENTALS
# ════════════════════════════════════════════════════════════════════════════════
with tab3:

    snap        = load_snapshot(ticker)
    df_income   = load_income_stmts(ticker)
    df_cf       = load_cash_flows(ticker)

    if not snap:
        st.warning(f"No fundamentals data found for **{ticker}**. Run `save_fundamentals.py` first.")
        st.stop()

    # ── Valuation multiples ───────────────────────────────────────────────────
    st.subheader("Valuation Multiples")
    v1, v2, v3, v4 = st.columns(4)
    v5, v6, v7, v8 = st.columns(4)

    with v1: stat_card("P/E Ratio",    fmt_multiple(snap.get("pe_ratio")))
    with v2: stat_card("P/S Ratio",    fmt_multiple(snap.get("ps_ratio")))
    with v3: stat_card("P/B Ratio",    fmt_multiple(snap.get("pb_ratio")))
    with v4: stat_card("EV / EBITDA",  fmt_multiple(snap.get("ev_ebitda")))
    with v5: stat_card("EV / Revenue", fmt_multiple(snap.get("ev_revenue")))
    with v6: stat_card("PEG Ratio",    fmt_multiple(snap.get("peg_ratio")))
    with v7: stat_card("FCF Yield",    fmt_pct(snap.get("fcf_yield")))
    with v8: stat_card("Price / FCF",  fmt_multiple(snap.get("price_fcf")))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quality metrics ───────────────────────────────────────────────────────
    st.subheader("Quality Metrics")
    q1, q2, q3, q4 = st.columns(4)

    with q1: stat_card("ROE",            fmt_pct(snap.get("roe")))
    with q2: stat_card("ROA",            fmt_pct(snap.get("roa")))
    with q3: stat_card("ROIC",           fmt_pct(snap.get("roic")))
    with q4: stat_card("FCF Conversion", fmt_multiple(snap.get("fcf_conversion")))

    st.markdown("<br>", unsafe_allow_html=True)

    if df_income.empty:
        st.info("No quarterly income statement data available.")
    else:
        # Keep last 8 quarters for readability
        df_i = df_income.tail(8).copy()
        labels = df_i["period_end"].dt.strftime("Q%q %Y") if hasattr(df_i["period_end"].dt, "quarter") else df_i["period_end"].dt.strftime("%b %Y")
        labels = df_i["period_end"].dt.to_period("Q").astype(str)

        # ── Revenue & Net Income ──────────────────────────────────────────────
        st.subheader("Revenue & Net Income (Quarterly)")
        fig_rev = make_subplots(specs=[[{"secondary_y": False}]])

        fig_rev.add_trace(go.Bar(
            x=labels, y=df_i["revenue"] / 1e9,
            name="Revenue", marker_color=BLUE, opacity=0.85,
        ))
        fig_rev.add_trace(go.Bar(
            x=labels, y=df_i["net_income"] / 1e9,
            name="Net Income", marker_color=GREEN, opacity=0.85,
        ))

        fig_rev.update_layout(
            template=PLOTLY_THEME, height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            barmode="group",
            yaxis_title="USD (Billions)",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
            hovermode="x unified",
        )
        fig_rev.update_xaxes(gridcolor=GRID_COLOR)
        fig_rev.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig_rev, use_container_width=True)

        # ── Profit Margins ────────────────────────────────────────────────────
        st.subheader("Profit Margins (Quarterly)")
        fig_margins = go.Figure()

        margin_series = [
            ("gross_margin",     "Gross Margin",     "#f0b429"),
            ("operating_margin", "Operating Margin", BLUE),
            ("net_margin",       "Net Margin",        GREEN),
        ]
        for col, name, color in margin_series:
            if col in df_i.columns:
                fig_margins.add_trace(go.Scatter(
                    x=labels, y=df_i[col] * 100,
                    name=name, mode="lines+markers",
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>{name}</b>: %{{y:.1f}}%<extra></extra>",
                ))

        fig_margins.update_layout(
            template=PLOTLY_THEME, height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="Margin (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
            hovermode="x unified",
        )
        fig_margins.update_xaxes(gridcolor=GRID_COLOR)
        fig_margins.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig_margins, use_container_width=True)

    # ── Cash Flow ─────────────────────────────────────────────────────────────
    if not df_cf.empty:
        st.subheader("Operating Cash Flow vs Free Cash Flow (Quarterly)")
        df_c  = df_cf.tail(8).copy()
        c_labels = df_c["period_end"].dt.to_period("Q").astype(str)

        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(
            x=c_labels, y=df_c["operating_cash_flow"] / 1e9,
            name="Operating CF", marker_color=BLUE, opacity=0.85,
        ))
        fig_cf.add_trace(go.Bar(
            x=c_labels, y=df_c["free_cash_flow"] / 1e9,
            name="Free Cash Flow", marker_color="#b794f4", opacity=0.85,
        ))
        fig_cf.update_layout(
            template=PLOTLY_THEME, height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            barmode="group",
            yaxis_title="USD (Billions)",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
            hovermode="x unified",
        )
        fig_cf.update_xaxes(gridcolor=GRID_COLOR)
        fig_cf.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig_cf, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — VALUATION
# ════════════════════════════════════════════════════════════════════════════════
with tab4:

    snap_v       = load_snapshot(ticker)
    inc_annual   = load_annual_income(ticker)
    bal_annual   = load_annual_balance(ticker)
    bal_latest   = load_latest_balance(ticker)
    all_prices   = load_all_prices(ticker)

    if not snap_v:
        st.warning(f"No fundamentals data for **{ticker}**. Run `save_fundamentals.py` first.")
        st.stop()

    current_price_v = snap_v.get("price") or 0.0

    # ── Session state defaults (only set if not already present) ──────────────
    if "dcf_growth_pct" not in st.session_state:
        st.session_state["dcf_growth_pct"] = 10
    if "dcf_wacc_pct" not in st.session_state:
        st.session_state["dcf_wacc_pct"] = 10
    if "dcf_terminal_pct" not in st.session_state:
        st.session_state["dcf_terminal_pct"] = 2.0
    if "dcf_mos_pct" not in st.session_state:
        st.session_state["dcf_mos_pct"] = 15

    # ── Load additional data for suggestions ──────────────────────────────────
    cf_annual_v = load_annual_cashflows(ticker)
    beta_v      = load_beta(ticker)
    suggestions = compute_dcf_suggestions(snap_v, cf_annual_v, inc_annual, bal_latest, beta_v)

    # ── DCF Assumption sliders ────────────────────────────────────────────────
    st.subheader("DCF Assumptions")
    st.caption(
        "Adjust the parameters below to explore different scenarios. "
        "All charts and the signal update instantly as you move the sliders."
    )

    # Suggestions row
    sg = suggestions
    st.markdown(
        '<div style="background:#111d2b; border-left:3px solid #4a90d9; border-radius:4px; '
        'padding:6px 14px 2px; margin-bottom:4px;">'
        '<span style="font-size:12px; color:#4a90d9; font-weight:600">'
        '&#128161; Suggested values — based on historical data</span></div>',
        unsafe_allow_html=True,
    )
    sg_cols = st.columns([2, 2, 2, 2, 3])
    with sg_cols[0]:
        st.metric("FCF Growth", f"{sg['fcf_growth_pct']}%", help=sg["fcf_growth_note"])
    with sg_cols[1]:
        st.metric("WACC", f"{sg['wacc_pct']}%", help=sg["wacc_note"])
    with sg_cols[2]:
        st.metric("Terminal", f"{sg['terminal_pct']}%", help=sg["terminal_note"])
    with sg_cols[3]:
        st.metric("Margin of Safety", f"{sg['mos_pct']}%", help=sg["mos_note"])
    with sg_cols[4]:
        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
        if st.button("Apply suggested values", type="primary"):
            st.session_state["dcf_growth_pct"]   = int(sg["fcf_growth_pct"])
            st.session_state["dcf_wacc_pct"]      = int(sg["wacc_pct"])
            st.session_state["dcf_terminal_pct"]  = float(sg["terminal_pct"])
            st.session_state["dcf_mos_pct"]       = int(sg["mos_pct"])
            st.rerun()

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        growth_pct = st.slider(
            "FCF Growth Rate (5Y) %", 0, 50, step=1, key="dcf_growth_pct",
            help="Expected annual free cash flow growth over the next 5 years.",
        )
    with sc2:
        wacc_pct = st.slider(
            "Discount Rate / WACC (%)", 5, 20, step=1, key="dcf_wacc_pct",
            help="Your required annual return. Higher = more conservative valuation.",
        )
    with sc3:
        terminal_pct = st.slider(
            "Terminal Growth Rate (%)", 0.0, 5.0, step=0.5, key="dcf_terminal_pct",
            help="Long-run growth rate after year 5. Typically close to GDP growth (~2–3%).",
        )
    with sc4:
        mos_pct = st.slider(
            "Margin of Safety (%)", 5, 50, step=5, key="dcf_mos_pct",
            help="Minimum upside required to classify a stock as Undervalued. "
                 "Conservative investors use 25–35%. More aggressive: 10–15%.",
        )

    growth_rate   = growth_pct   / 100
    wacc          = wacc_pct     / 100
    terminal_rate = terminal_pct / 100
    mos           = mos_pct      / 100

    # ── Method toggles ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Include in Signal")
    st.caption(
        "Uncheck any method you want to exclude from the overall signal calculation. "
        "Graham Number is a conservative floor — excluded by default."
    )
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    use_dcf    = mc1.checkbox("DCF",           value=True)
    use_pe     = mc2.checkbox("P/E Multiple",  value=True)
    use_ev     = mc3.checkbox("EV/EBITDA",     value=True)
    use_ps     = mc4.checkbox("P/S Multiple",  value=True)
    use_graham = mc5.checkbox("Graham Number", value=False,
                              help="Excluded by default — tends to undervalue growth stocks.")

    # ── Compute valuations ────────────────────────────────────────────────────
    fcf_ttm = snap_v.get("fcf_ttm")
    shares  = snap_v.get("shares_outstanding")

    dcf_price  = dcf_intrinsic(fcf_ttm, shares, growth_rate, wacc, terminal_rate)

    hist_avgs  = compute_historical_multiples(inc_annual, bal_annual, all_prices)
    mult_prices = multiples_implied_prices(snap_v, hist_avgs)

    # Graham Number: TTM EPS + book value per share from latest balance sheet
    net_inc_ttm = snap_v.get("net_income_ttm")
    equity      = bal_latest.get("total_equity")
    eps_ttm     = (float(net_inc_ttm) / float(shares)) if (net_inc_ttm and shares) else None
    bvps        = (float(equity) / float(shares))      if (equity and shares) else None
    graham_price = graham_number(eps_ttm, bvps)

    # Reverse DCF
    implied_growth = reverse_dcf(current_price_v, fcf_ttm, shares, wacc, terminal_rate)

    estimates = {
        "DCF":          dcf_price,
        "P/E Multiple": mult_prices.get("pe_implied"),
        "EV/EBITDA":    mult_prices.get("ev_ebitda_implied"),
        "P/S Multiple": mult_prices.get("ps_implied"),
    }

    active = set()
    if use_dcf:    active.add("DCF")
    if use_pe:     active.add("P/E Multiple")
    if use_ev:     active.add("EV/EBITDA")
    if use_ps:     active.add("P/S Multiple")
    if use_graham:
        estimates["Graham Number"] = graham_price
        active.add("Graham Number")

    summary = valuation_summary(current_price_v, estimates, mos, active)

    # ── Method cards ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Method Results")
    st.caption(
        f"Current price: **${current_price_v:.2f}**  ·  "
        "Each card shows what the stock would be worth according to that method."
    )

    def method_card(key: str, info: dict, implied, current: float) -> None:
        if implied is None:
            upside_html = '<span style="color:#8b8fa8">N/A</span>'
            price_html  = '<span style="color:#8b8fa8; font-size:22px">N/A</span>'
        else:
            upside = (implied - current) / current * 100
            color  = GREEN if upside >= 0 else RED
            arrow  = "▲" if upside >= 0 else "▼"
            upside_html = f'<span style="color:{color}; font-size:14px">{arrow} {abs(upside):.1f}%</span>'
            price_html  = f'<span style="font-size:22px; font-weight:700; color:#fff">${implied:.2f}</span>'

        st.markdown(f"""
        <div class="stat-card" style="text-align:left; padding:18px 20px;">
          <div style="font-size:13px; font-weight:700; color:#ffffff; margin-bottom:4px">
            {info["name"]}
          </div>
          <div style="font-size:11px; color:#8b8fa8; margin-bottom:12px; line-height:1.4">
            {info["description"]}
          </div>
          <div style="margin-bottom:6px">{price_html} &nbsp; {upside_html}</div>
          <div style="font-size:10px; color:#6b6f82; font-style:italic; line-height:1.4">
            {info["limitation"]}
          </div>
        </div>""", unsafe_allow_html=True)

    col_m = st.columns(len([k for k in ["DCF","P/E Multiple","EV/EBITDA","P/S Multiple"] if k in estimates]))
    for i, key in enumerate([k for k in ["DCF","P/E Multiple","EV/EBITDA","P/S Multiple"] if k in estimates]):
        with col_m[i]:
            method_card(key, METHOD_INFO[key], estimates.get(key), current_price_v)

    # ── Graham Number reference card ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📖  Graham Number — Conservative Reference Floor", expanded=False):
        st.markdown(f"""
        **{GRAHAM_INFO['name']}**

        {GRAHAM_INFO['description']}

        **Implied price:** {"$" + f"{graham_price:.2f}" if graham_price else "N/A"}
        {"  ·  Upside: " + (f"{(graham_price - current_price_v)/current_price_v*100:+.1f}%" if graham_price else "") }

        *{GRAHAM_INFO['limitation']}*
        """)

    # ── Reverse DCF insight ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Reverse DCF — What Growth Is Priced In?")
    if implied_growth is not None:
        color_g = GREEN if implied_growth < growth_rate else RED
        st.markdown(f"""
        <div class="stat-card" style="text-align:left; padding:20px 24px;">
          <div style="font-size:13px; color:#8b8fa8; margin-bottom:8px">
            At the current price of <strong>${current_price_v:.2f}</strong>, the market is implying:
          </div>
          <div style="font-size:28px; font-weight:700; color:{color_g}; margin-bottom:8px">
            {implied_growth * 100:.1f}% annual FCF growth
          </div>
          <div style="font-size:12px; color:#8b8fa8; line-height:1.6">
            Your DCF assumption is <strong>{growth_pct}%</strong>. &nbsp;
            {"The market is pricing in <strong>less</strong> growth than your assumption — suggesting potential upside."
              if implied_growth < growth_rate else
             "The market is pricing in <strong>more</strong> growth than your assumption — suggesting the stock may be expensive relative to your expectations."}
            <br>Use this as a sanity check: is {implied_growth*100:.1f}% annual FCF growth realistic for {ticker}?
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Reverse DCF not available — requires positive TTM free cash flow.")

    # ── Historical multiples comparison chart ─────────────────────────────────
    if hist_avgs:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Current vs Historical Average Multiples")
        st.caption(
            "Are today's multiples higher or lower than the stock's own historical average? "
            "A multiple above its historical average may indicate the market has become more "
            "optimistic — or that the stock is more expensive relative to its own history."
        )

        mult_labels, curr_vals, hist_vals = [], [], []
        snap_mult = {"P/E": snap_v.get("pe_ratio"), "EV/EBITDA": snap_v.get("ev_ebitda"),
                     "P/S": snap_v.get("ps_ratio")}
        hist_map  = {"P/E": hist_avgs.get("pe_avg"), "EV/EBITDA": hist_avgs.get("ev_ebitda_avg"),
                     "P/S": hist_avgs.get("ps_avg")}
        for label in ["P/E", "EV/EBITDA", "P/S"]:
            c, h = snap_mult.get(label), hist_map.get(label)
            if c and h:
                mult_labels.append(label)
                curr_vals.append(round(float(c), 1))
                hist_vals.append(round(h, 1))

        if mult_labels:
            fig_mult = go.Figure()
            fig_mult.add_trace(go.Bar(name="Current", x=mult_labels, y=curr_vals,
                                      marker_color=BLUE, opacity=0.85))
            fig_mult.add_trace(go.Bar(name="5Y Historical Avg", x=mult_labels, y=hist_vals,
                                      marker_color="#8b8fa8", opacity=0.7))
            fig_mult.update_layout(
                template=PLOTLY_THEME, height=320, barmode="group",
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Multiple (x)",
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
                paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
            )
            fig_mult.update_xaxes(gridcolor=GRID_COLOR)
            fig_mult.update_yaxes(gridcolor=GRID_COLOR)
            st.plotly_chart(fig_mult, use_container_width=True)

    # ── Signal banner ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Overall Signal")

    signal     = summary.get("signal", "Insufficient Data")
    confidence = summary.get("confidence")
    avg_upside = summary.get("avg_upside")
    agreeing   = summary.get("methods_agreeing", 0)
    total_m    = summary.get("total_active_methods", 0)
    cfg        = SIGNAL_CONFIG.get(signal, SIGNAL_CONFIG["Insufficient Data"])

    conf_text = {
        "High":     "Most methods point in the same direction — signal is reliable.",
        "Moderate": "Some methods disagree — treat the signal with moderate caution.",
        "Low":      "Methods disagree significantly or few methods have data — treat with caution.",
        None:       "Not enough data to compute a signal.",
    }.get(confidence, "")

    avg_upside_str = f"{avg_upside * 100:+.1f}%" if avg_upside is not None else "N/A"
    agree_str      = f"{agreeing}/{total_m} selected methods agree" if total_m else ""

    st.markdown(f"""
    <div style="background:{cfg['bg']}; border:1px solid {cfg['color']}33;
                border-radius:12px; padding:24px 28px; margin-top:8px">
      <div style="font-size:32px; font-weight:800; color:{cfg['color']}; margin-bottom:6px">
        {cfg['icon']}  {signal}
      </div>
      <div style="font-size:15px; color:#cccccc; margin-bottom:10px">
        Average upside across selected methods: <strong style="color:{cfg['color']}">{avg_upside_str}</strong>
        {"  ·  <strong>" + agree_str + "</strong>" if agree_str else ""}
        {"  ·  Confidence: <strong>" + (confidence or "—") + "</strong>" if confidence else ""}
      </div>
      <div style="font-size:12px; color:#8b8fa8; line-height:1.6; border-top:1px solid #2a2d36; padding-top:10px">
        <strong>How is this calculated?</strong>
        The signal is based on the average upside/downside across the selected methods above.
        A stock is considered <em>Undervalued</em> if the average implied upside exceeds your
        Margin of Safety ({mos_pct}%), <em>Overvalued</em> if it is below -{mos_pct}%,
        and <em>Fairly Valued</em> otherwise.
        Confidence reflects how many selected methods agree with the overall direction.
        <br><br>{conf_text}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — PORTFOLIO
# ════════════════════════════════════════════════════════════════════════════════
with tab5:

    # ── Add / Edit position form ──────────────────────────────────────────────
    with st.expander("➕  Add or Edit a Position", expanded=True):
        with st.form("holding_form", clear_on_submit=True):
            st.caption("To edit an existing position, enter the same ticker with updated values.")
            pf1, pf2, pf3 = st.columns(3)
            with pf1:
                h_ticker = st.selectbox("Ticker", TICKERS)
            with pf2:
                h_shares = st.number_input("Number of Shares", min_value=0.001,
                                           step=1.0, format="%.3f")
            with pf3:
                h_cost   = st.number_input("Average Cost per Share ($)", min_value=0.01,
                                           step=0.01, format="%.2f")
            h_notes    = st.text_input("Notes (optional)", placeholder="e.g. Long-term hold")
            submitted  = st.form_submit_button("💾  Save Position")
            if submitted:
                if h_shares > 0 and h_cost > 0:
                    save_holding(h_ticker, h_shares, h_cost, h_notes)
                    st.success(f"Position saved for **{h_ticker}**.")
                    st.rerun()
                else:
                    st.error("Shares and cost must be greater than zero.")

    # ── Load holdings ─────────────────────────────────────────────────────────
    holdings = load_portfolio()

    if holdings.empty:
        st.info("No holdings yet. Add your first position using the form above.")
        st.stop()

    # ── Enrich with current prices and signals ────────────────────────────────
    rows = []
    for _, h in holdings.iterrows():
        t         = h["ticker"]
        h_shares  = float(h["shares"])
        h_cost    = float(h["avg_cost"])
        cost_basis = h_shares * h_cost

        # Current price — latest available in stock_prices
        price_res = (
            get_client().table("stock_prices")
            .select("close, date")
            .eq("ticker", t)
            .order("date", desc=True)
            .limit(1)
            .execute()
        )
        curr_price = float(price_res.data[0]["close"]) if price_res.data else None

        mkt_value  = (curr_price * h_shares)           if curr_price else None
        gain_abs   = (mkt_value - cost_basis)           if mkt_value else None
        gain_pct   = (gain_abs / cost_basis * 100)      if gain_abs is not None else None

        # Quick signal: compare PE to historical average PE
        snap_h    = load_snapshot(t)
        signal_h  = "N/A"
        if snap_h:
            inc_a   = load_annual_income(t)
            bal_a   = load_annual_balance(t)
            pr_all  = load_all_prices(t)
            h_avgs  = compute_historical_multiples(inc_a, bal_a, pr_all)
            m_imp   = multiples_implied_prices(snap_h, h_avgs)
            est_h   = {k: m_imp.get(v) for k, v in
                       [("P/E Multiple","pe_implied"),("EV/EBITDA","ev_ebitda_implied"),
                        ("P/S Multiple","ps_implied")]}
            sum_h   = valuation_summary(curr_price or snap_h.get("price") or 0,
                                        est_h, mos=0.15,
                                        active_methods={"P/E Multiple","EV/EBITDA","P/S Multiple"})
            signal_h = sum_h.get("signal", "N/A")

        rows.append({
            "Ticker":        t,
            "Shares":        h_shares,
            "Avg Cost":      h_cost,
            "Current Price": curr_price,
            "Market Value":  mkt_value,
            "Gain / Loss $": gain_abs,
            "Gain / Loss %": gain_pct,
            "Signal":        signal_h,
            "Notes":         h.get("notes", ""),
        })

    df_port = pd.DataFrame(rows)

    # ── Holdings table ────────────────────────────────────────────────────────
    st.subheader("Holdings")
    st.caption(
        "Signal uses P/E, EV/EBITDA, and P/S multiples vs 5-year historical averages "
        "with a 15% margin of safety. For a full analysis, open the 📐 Valuation tab."
    )

    SIGNAL_ICONS = {
        "Undervalued":       "✅ Undervalued",
        "Fairly Valued":     "⚖️ Fairly Valued",
        "Overvalued":        "🔴 Overvalued",
        "Insufficient Data": "❓ Insufficient Data",
        "N/A":               "—",
    }

    def fmt_cell(v, fmt="$"):
        if v is None:
            return "N/A"
        if fmt == "$":
            return f"${v:,.2f}"
        if fmt == "%":
            return f"{v:+.2f}%"
        return str(v)

    for _, r in df_port.iterrows():
        gc = GREEN if (r["Gain / Loss $"] or 0) >= 0 else RED
        sig_label = SIGNAL_ICONS.get(r["Signal"], r["Signal"])

        col_a, col_b, col_c, col_d, col_e, col_f, col_g = st.columns([1,1,1,1,1.2,1.2,1.5])
        col_a.markdown(f"**{r['Ticker']}**")
        col_b.markdown(fmt_cell(r["Shares"],        fmt=""))
        col_c.markdown(fmt_cell(r["Avg Cost"],      fmt="$"))
        col_d.markdown(fmt_cell(r["Current Price"], fmt="$"))
        col_e.markdown(fmt_cell(r["Market Value"],  fmt="$"))
        col_f.markdown(f'<span style="color:{gc}">{fmt_cell(r["Gain / Loss $"],"$")} '
                       f'({fmt_cell(r["Gain / Loss %"],"%")})</span>',
                       unsafe_allow_html=True)
        col_g.markdown(sig_label)

        # Delete button
        if st.button(f"🗑 Remove {r['Ticker']}", key=f"del_{r['Ticker']}"):
            delete_holding(r["Ticker"])
            st.rerun()
        st.divider()

    # ── Portfolio summary ─────────────────────────────────────────────────────
    st.subheader("Portfolio Summary")
    st.caption("Based on latest available prices in the database.")

    valid        = df_port.dropna(subset=["Market Value"])
    total_cost   = (df_port["Shares"] * df_port["Avg Cost"]).sum()
    total_value  = valid["Market Value"].sum()
    total_gain   = total_value - (valid["Shares"] * valid["Avg Cost"]).sum()
    total_gain_p = (total_gain / (valid["Shares"] * valid["Avg Cost"]).sum() * 100) if total_cost else 0

    ps1, ps2, ps3, ps4 = st.columns(4)
    with ps1: stat_card("Total Cost Basis", fmt_large(total_cost))
    with ps2: stat_card("Current Value",    fmt_large(total_value))
    gc_port = "green" if total_gain >= 0 else "red"
    with ps3: stat_card("Total Gain / Loss",
                        f'<span style="color:{"#00d084" if total_gain >= 0 else "#ff4b4b"}">'
                        f'{fmt_large(abs(total_gain))}</span>',
                        f'{"+" if total_gain >= 0 else "-"}{abs(total_gain_p):.2f}%', gc_port)
    with ps4:
        signal_counts = df_port["Signal"].value_counts().to_dict()
        dominant = max(signal_counts, key=signal_counts.get) if signal_counts else "N/A"
        cfg_p    = SIGNAL_CONFIG.get(dominant, SIGNAL_CONFIG["Insufficient Data"])
        stat_card("Portfolio Signal",
                  f'{cfg_p["icon"]} {dominant}',
                  f"({signal_counts.get(dominant,0)} of {len(df_port)} positions)")
