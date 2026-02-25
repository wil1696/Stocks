"""
valuation.py
Four valuation models: DCF, Comparable Multiples, Reverse DCF, Graham Number.
All functions return None rather than raising on missing or invalid data.
"""
from __future__ import annotations

import math
import pandas as pd
from typing import Optional


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


# ── Method metadata (used by dashboard for explanations) ─────────────────────

METHOD_INFO = {
    "DCF": {
        "name":        "Discounted Cash Flow (DCF)",
        "description": "Projects free cash flows for 5 years and discounts them back "
                       "to today using your required rate of return (WACC).",
        "limitation":  "⚠ Highly sensitive to the growth rate assumption. Small changes "
                       "can produce very different results — treat as a range, not a fact.",
    },
    "P/E Multiple": {
        "name":        "P/E Multiple (Historical Avg)",
        "description": "Asks: if the market valued this stock at the same earnings multiple "
                       "it paid on average over the past 5 years, what price would that imply?",
        "limitation":  "⚠ Less meaningful when earnings are negative or highly volatile.",
    },
    "EV/EBITDA": {
        "name":        "EV/EBITDA Multiple (Historical Avg)",
        "description": "Same concept as P/E but uses enterprise value and operating profit "
                       "(EBITDA), making it more comparable across capital structures.",
        "limitation":  "⚠ May undervalue asset-light companies; does not account for capex.",
    },
    "P/S Multiple": {
        "name":        "P/S Multiple (Historical Avg)",
        "description": "Compares the stock's price-to-sales ratio to its own 5-year average. "
                       "Useful when earnings are temporarily depressed.",
        "limitation":  "⚠ Ignores profitability — a high-revenue but unprofitable company "
                       "can look cheap on P/S while burning cash.",
    },
}

GRAHAM_INFO = {
    "name":        "Graham Number (Reference Floor)",
    "description": "Benjamin Graham's formula: √(22.5 × EPS × Book Value per Share). "
                   "Represents a conservative lower bound on intrinsic value.",
    "limitation":  "⚠ Designed for mature, asset-heavy companies. Almost always undervalues "
                   "high-growth tech stocks. Shown as a reference floor only — excluded "
                   "from the overall signal by default.",
}

SIGNAL_CONFIG = {
    "Undervalued":       {"color": "#00d084", "icon": "✅", "bg": "rgba(0,208,132,0.08)"},
    "Fairly Valued":     {"color": "#f0b429", "icon": "⚖️", "bg": "rgba(240,180,41,0.08)"},
    "Overvalued":        {"color": "#ff4b4b", "icon": "🔴", "bg": "rgba(255,75,75,0.08)"},
    "Insufficient Data": {"color": "#8b8fa8", "icon": "❓", "bg": "rgba(139,143,168,0.08)"},
}


# ── 1. Discounted Cash Flow ───────────────────────────────────────────────────

def dcf_intrinsic(
    fcf_ttm:       float,
    shares:        float,
    growth_rate:   float,
    discount_rate: float,
    terminal_rate: float,
    years:         int = 5,
) -> Optional[float]:
    """
    Intrinsic value per share via 5-year DCF + terminal value (Gordon Growth).

    Returns None when:
      - Any input is missing or invalid
      - FCF is negative (DCF is not meaningful with negative free cash flow)
      - discount_rate <= terminal_rate (terminal value formula would break)
    """
    fcf_ttm, shares      = _safe(fcf_ttm), _safe(shares)
    growth_rate          = _safe(growth_rate)
    discount_rate        = _safe(discount_rate)
    terminal_rate        = _safe(terminal_rate)

    if any(v is None for v in [fcf_ttm, shares, growth_rate, discount_rate, terminal_rate]):
        return None
    if shares <= 0 or discount_rate <= 0 or discount_rate <= terminal_rate:
        return None
    if fcf_ttm <= 0:
        return None  # DCF is not meaningful with negative free cash flow

    pv, fcf = 0.0, fcf_ttm
    for n in range(1, years + 1):
        fcf *= (1 + growth_rate)
        pv  += fcf / (1 + discount_rate) ** n

    # Terminal value — Gordon Growth Model
    tv  = (fcf * (1 + terminal_rate)) / (discount_rate - terminal_rate)
    pv += tv / (1 + discount_rate) ** years

    result = pv / shares
    return result if result > 0 else None


# ── 2. Reverse DCF ────────────────────────────────────────────────────────────

def reverse_dcf(
    current_price: float,
    fcf_ttm:       float,
    shares:        float,
    discount_rate: float,
    terminal_rate: float,
    years:         int = 5,
) -> Optional[float]:
    """
    Finds the implied FCF growth rate baked into the current stock price.
    Uses binary search over the range [-30%, +100%].
    Returns the implied growth rate as a decimal (e.g. 0.12 = 12%), or None.
    """
    price = _safe(current_price)
    if price is None or price <= 0:
        return None

    lo, hi = -0.30, 1.00

    lo_val = dcf_intrinsic(fcf_ttm, shares, lo, discount_rate, terminal_rate, years)
    hi_val = dcf_intrinsic(fcf_ttm, shares, hi, discount_rate, terminal_rate, years)
    if lo_val is None or hi_val is None:
        return None

    for _ in range(120):
        mid     = (lo + hi) / 2.0
        mid_val = dcf_intrinsic(fcf_ttm, shares, mid, discount_rate, terminal_rate, years)
        if mid_val is None:
            return None
        if abs(mid_val - price) < 0.001:
            break
        if mid_val < price:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0


# ── 3. Graham Number ──────────────────────────────────────────────────────────

def graham_number(eps_ttm: float, book_value_per_share: float) -> Optional[float]:
    """
    Graham's conservative intrinsic value floor = √(22.5 × EPS × BVPS).
    Returns None if either input is zero or negative.
    """
    eps  = _safe(eps_ttm)
    bvps = _safe(book_value_per_share)
    if eps is None or bvps is None or eps <= 0 or bvps <= 0:
        return None
    return math.sqrt(22.5 * eps * bvps)


# ── 4. Historical Comparable Multiples ───────────────────────────────────────

def compute_historical_multiples(
    income_annual:  pd.DataFrame,
    balance_annual: pd.DataFrame,
    prices_df:      pd.DataFrame,
) -> dict:
    """
    Computes 5-year historical average PE, PS, and EV/EBITDA.

    For each annual period:
      1. Finds the year-end stock price (last trading day on or before period_end)
      2. Computes the multiple using that price + the annual financial data
      3. Averages across all available years

    Returns a dict with keys: pe_avg, pe_history, ps_avg, ps_history,
    ev_ebitda_avg, ev_ebitda_history  (only keys with at least one valid
    data point are included).
    """
    if income_annual.empty or prices_df.empty:
        return {}

    prices = prices_df.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date").reset_index(drop=True)

    pe_list, ps_list, ev_list = [], [], []

    for _, row in income_annual.iterrows():
        period_end = pd.to_datetime(row["period_end"])

        past = prices[prices["date"] <= period_end]
        if past.empty:
            continue
        price = float(past.iloc[-1]["close"])

        # ── P/E ──────────────────────────────────────────────────────────────
        eps = _safe(row.get("eps_diluted"))
        if eps and eps > 0:
            pe_list.append({"period": str(period_end.year), "value": round(price / eps, 2)})

        # ── P/S ──────────────────────────────────────────────────────────────
        revenue = _safe(row.get("revenue"))
        shares  = _safe(row.get("shares_diluted"))
        if revenue and revenue > 0 and shares and shares > 0:
            ps_list.append({"period": str(period_end.year),
                            "value": round((price * shares) / revenue, 2)})

        # ── EV/EBITDA ─────────────────────────────────────────────────────────
        ebitda = _safe(row.get("ebitda"))
        if ebitda and ebitda > 0 and not balance_annual.empty:
            bal = balance_annual.copy()
            bal["period_end"] = pd.to_datetime(bal["period_end"])
            close_bal = bal[abs((bal["period_end"] - period_end).dt.days) <= 95]
            if not close_bal.empty and shares and shares > 0:
                b    = close_bal.iloc[0]
                debt = _safe(b.get("total_debt"))            or 0.0
                cash = _safe(b.get("cash_and_equivalents")) or 0.0
                ev   = price * shares + debt - cash
                if ev > 0:
                    ev_list.append({"period": str(period_end.year),
                                    "value": round(ev / ebitda, 2)})

    result: dict = {}
    if pe_list:
        result["pe_avg"]          = sum(d["value"] for d in pe_list) / len(pe_list)
        result["pe_history"]      = pe_list
    if ps_list:
        result["ps_avg"]          = sum(d["value"] for d in ps_list) / len(ps_list)
        result["ps_history"]      = ps_list
    if ev_list:
        result["ev_ebitda_avg"]   = sum(d["value"] for d in ev_list) / len(ev_list)
        result["ev_ebitda_history"] = ev_list

    return result


def multiples_implied_prices(snapshot: dict, hist_avgs: dict) -> dict:
    """
    Derives implied stock prices by applying historical average multiples
    to the current TTM financials stored in the snapshot.
    """
    results: dict = {}

    shares      = _safe(snapshot.get("shares_outstanding"))
    net_inc_ttm = _safe(snapshot.get("net_income_ttm"))
    revenue_ttm = _safe(snapshot.get("revenue_ttm"))
    ebitda_ttm  = _safe(snapshot.get("ebitda_ttm"))
    ev          = _safe(snapshot.get("enterprise_value"))
    mc          = _safe(snapshot.get("market_cap"))

    if not shares or shares <= 0:
        return results

    if net_inc_ttm and net_inc_ttm > 0 and "pe_avg" in hist_avgs:
        results["pe_implied"] = hist_avgs["pe_avg"] * (net_inc_ttm / shares)

    if revenue_ttm and revenue_ttm > 0 and "ps_avg" in hist_avgs:
        results["ps_implied"] = hist_avgs["ps_avg"] * (revenue_ttm / shares)

    if ebitda_ttm and ebitda_ttm > 0 and ev and mc and "ev_ebitda_avg" in hist_avgs:
        net_debt   = ev - mc                               # total_debt - cash
        implied_mc = hist_avgs["ev_ebitda_avg"] * ebitda_ttm - net_debt
        if implied_mc > 0:
            results["ev_ebitda_implied"] = implied_mc / shares

    return results


# ── 5. Valuation Summary Signal ───────────────────────────────────────────────

def valuation_summary(
    current_price:    float,
    estimates:        dict,
    margin_of_safety: float       = 0.15,
    active_methods:   set | None  = None,
) -> dict:
    """
    Aggregates method estimates into one overall signal.

    Parameters
    ----------
    current_price    : current stock price
    estimates        : {method_key: implied_price_or_None}
    margin_of_safety : threshold separating Undervalued / Fairly Valued / Overvalued
    active_methods   : which method keys to include in the averaged signal
                       (Graham excluded by default via the caller)

    Returns
    -------
    dict with:
      signal, confidence, avg_upside, method_results,
      methods_agreeing, total_active_methods
    """
    price = _safe(current_price)
    if price is None or price <= 0:
        return {}

    if active_methods is None:
        active_methods = set(estimates.keys())

    method_results: list  = []
    active_upsides: list  = []

    for method, implied in estimates.items():
        f = _safe(implied)
        if f is None or f <= 0:
            method_results.append({"method": method, "implied_price": None,
                                   "upside": None, "available": False})
            continue

        upside = (f - price) / price
        method_results.append({"method": method, "implied_price": f,
                               "upside": upside, "available": True})
        if method in active_methods:
            active_upsides.append(upside)

    if not active_upsides:
        return {
            "signal":               "Insufficient Data",
            "confidence":           None,
            "avg_upside":           None,
            "method_results":       method_results,
            "methods_agreeing":     0,
            "total_active_methods": 0,
        }

    avg_upside = sum(active_upsides) / len(active_upsides)

    # ── Overall signal ────────────────────────────────────────────────────────
    if avg_upside > margin_of_safety:
        signal = "Undervalued"
    elif avg_upside < -margin_of_safety:
        signal = "Overvalued"
    else:
        signal = "Fairly Valued"

    # ── Agreement & confidence ────────────────────────────────────────────────
    def _agrees(u: float) -> bool:
        if signal == "Undervalued":  return u >  margin_of_safety
        if signal == "Overvalued":   return u < -margin_of_safety
        return abs(u) <= margin_of_safety

    agreeing = sum(1 for u in active_upsides if _agrees(u))
    n        = len(active_upsides)
    ratio    = agreeing / n if n else 0

    if n >= 3:
        confidence = "High" if ratio == 1.0 else ("Moderate" if ratio >= 0.6 else "Low")
    elif n == 2:
        confidence = "Moderate" if ratio == 1.0 else "Low"
    else:
        confidence = "Low"

    # Wide spread between methods → downgrade confidence
    if n >= 2 and (max(active_upsides) - min(active_upsides)) > 1.0:
        confidence = "Low"

    return {
        "signal":               signal,
        "confidence":           confidence,
        "avg_upside":           avg_upside,
        "method_results":       method_results,
        "methods_agreeing":     agreeing,
        "total_active_methods": n,
    }


# ── 6. DCF Parameter Suggestions ──────────────────────────────────────────────

def compute_dcf_suggestions(
    snap: dict,
    cf_annual: pd.DataFrame,
    inc_annual: pd.DataFrame,
    bal_latest: dict,
    beta: Optional[float] = None,
) -> dict:
    """
    Computes data-driven suggestions for DCF parameters based on historical data.

    Parameters
    ----------
    snap       : fundamentals_snapshot dict (needs market_cap)
    cf_annual  : annual cash_flows rows with period_end + free_cash_flow, sorted asc
    inc_annual : annual income rows with period_end + revenue + interest_expense +
                 tax_rate, sorted asc
    bal_latest : most recent balance sheet dict (total_debt, cash_and_equivalents)
    beta       : trailing 5-yr monthly beta from yfinance, or None

    Returns
    -------
    dict with keys:
        fcf_growth_pct, wacc_pct, terminal_pct, mos_pct  (numeric, for sliders)
        fcf_growth_note, wacc_note, terminal_note, mos_note  (explanation strings)
    """
    RISK_FREE = 0.045  # ~10-yr US Treasury
    ERP       = 0.055  # equity risk premium

    # ── FCF Growth Rate ───────────────────────────────────────────────────────
    fcf_growth_pct  = 10
    fcf_growth_note = "Insufficient annual data (<2 years available)"

    def _rev_cagr_fallback(reason: str) -> None:
        nonlocal fcf_growth_pct, fcf_growth_note
        if inc_annual.empty or "revenue" not in inc_annual.columns:
            return
        rev_rows = inc_annual.copy()
        rev_rows["revenue"] = pd.to_numeric(rev_rows["revenue"], errors="coerce")
        rev_rows = rev_rows[rev_rows["revenue"] > 0].dropna(subset=["revenue"]).iloc[-4:]
        if len(rev_rows) >= 2:
            r_old = float(rev_rows.iloc[0]["revenue"])
            r_new = float(rev_rows.iloc[-1]["revenue"])
            n     = len(rev_rows) - 1
            raw   = (r_new / r_old) ** (1.0 / n) - 1
            yr0   = pd.to_datetime(rev_rows.iloc[0]["period_end"]).year
            yr1   = pd.to_datetime(rev_rows.iloc[-1]["period_end"]).year
            fcf_growth_pct  = round(max(0.0, min(0.40, raw)) * 100)
            fcf_growth_note = (
                f"Using {n}-yr revenue CAGR: {raw * 100:.1f}% "
                f"(FY{yr0}–FY{yr1}) — {reason}"
            )

    if not cf_annual.empty and "free_cash_flow" in cf_annual.columns:
        fcf_rows = cf_annual.copy()
        fcf_rows["free_cash_flow"] = pd.to_numeric(fcf_rows["free_cash_flow"], errors="coerce")
        fcf_pos = fcf_rows[fcf_rows["free_cash_flow"] > 0].dropna(subset=["free_cash_flow"]).iloc[-4:]
        if len(fcf_pos) >= 2:
            f_old = float(fcf_pos.iloc[0]["free_cash_flow"])
            f_new = float(fcf_pos.iloc[-1]["free_cash_flow"])
            n     = len(fcf_pos) - 1
            raw   = (f_new / f_old) ** (1.0 / n) - 1
            yr0   = pd.to_datetime(fcf_pos.iloc[0]["period_end"]).year
            yr1   = pd.to_datetime(fcf_pos.iloc[-1]["period_end"]).year
            fcf_growth_pct  = round(max(0.0, min(0.40, raw)) * 100)
            fcf_growth_note = (
                f"{n}-yr FCF CAGR: {raw * 100:.1f}% "
                f"(FY{yr0}–FY{yr1}, annual data)"
            )
        else:
            reason = "FCF negative in base year" if len(cf_annual) > 0 else "no FCF data"
            _rev_cagr_fallback(reason)
    else:
        _rev_cagr_fallback("no FCF data available")

    # ── WACC ─────────────────────────────────────────────────────────────────
    beta_val = beta if beta is not None else 1.0
    ke       = RISK_FREE + beta_val * ERP

    kd       = 0.04
    kd_note  = "default 4%"
    if not inc_annual.empty:
        latest_inc    = inc_annual.iloc[-1]
        interest_exp  = _safe(latest_inc.get("interest_expense"))
        tax_rate_raw  = _safe(latest_inc.get("tax_rate"))
        total_debt_kd = _safe(bal_latest.get("total_debt"))
        if interest_exp is not None and total_debt_kd and total_debt_kd > 0:
            pre_tax_kd = abs(interest_exp) / total_debt_kd
            tax_used   = tax_rate_raw if (tax_rate_raw is not None and 0 < tax_rate_raw <= 0.6) else 0.21
            kd         = pre_tax_kd * (1 - tax_used)
            yr_inc     = pd.to_datetime(latest_inc["period_end"]).year
            kd_note    = (
                f"{pre_tax_kd * 100:.1f}% pre-tax × (1−{tax_used * 100:.0f}%) "
                f"= {kd * 100:.1f}% after-tax (FY{yr_inc})"
            )

    market_cap_val = _safe(snap.get("market_cap"))
    total_debt_w   = _safe(bal_latest.get("total_debt")) or 0.0
    we = (market_cap_val / (market_cap_val + total_debt_w)
          if market_cap_val and market_cap_val > 0 else 0.85)

    wacc_raw = ke * we + kd * (1 - we)
    wacc_pct = round(max(0.06, min(0.18, wacc_raw)) * 100)

    beta_src  = f"{beta:.2f} (5yr monthly)" if beta is not None else "unavailable → using 1.00"
    wacc_note = (
        f"Ke = {RISK_FREE * 100:.1f}% + {beta_val:.2f}β × {ERP * 100:.1f}% "
        f"= {ke * 100:.1f}% (beta {beta_src}) | "
        f"Kd = {kd_note} | "
        f"Weights: {we * 100:.0f}% equity / {(1 - we) * 100:.0f}% debt"
    )

    # ── Terminal Growth Rate ──────────────────────────────────────────────────
    terminal_pct  = 2.0
    terminal_note = "GDP growth proxy | insufficient revenue data → default 2.0%"

    if not inc_annual.empty and "revenue" in inc_annual.columns:
        rev_all = inc_annual.copy()
        rev_all["revenue"] = pd.to_numeric(rev_all["revenue"], errors="coerce")
        rev_all = rev_all[rev_all["revenue"] > 0].dropna(subset=["revenue"]).iloc[-6:]
        if len(rev_all) >= 2:
            n       = len(rev_all) - 1
            r_old   = float(rev_all.iloc[0]["revenue"])
            r_new   = float(rev_all.iloc[-1]["revenue"])
            rev_cagr = (r_new / r_old) ** (1.0 / n) - 1
            yr0     = pd.to_datetime(rev_all.iloc[0]["period_end"]).year
            yr1     = pd.to_datetime(rev_all.iloc[-1]["period_end"]).year

            terminal_pct = 2.0 if rev_cagr < 0.05 else (2.5 if rev_cagr <= 0.15 else 3.0)
            n_label      = f"{n}-yr" if n >= 5 else f"{n}-yr (only {n} yrs available)"
            terminal_note = (
                f"GDP growth proxy | {n_label} revenue CAGR = {rev_cagr * 100:.1f}% "
                f"(FY{yr0}–FY{yr1}) → {terminal_pct:.1f}%"
            )

    # ── Margin of Safety ─────────────────────────────────────────────────────
    if beta is None:
        mos_pct  = 15
        mos_note = "Beta unavailable → default 15%"
    elif beta < 0.8:
        mos_pct  = 10
        mos_note = f"Beta = {beta:.2f} (5yr monthly) → below-market volatility → 10%"
    elif beta <= 1.2:
        mos_pct  = 15
        mos_note = f"Beta = {beta:.2f} (5yr monthly) → market-like volatility → 15%"
    elif beta <= 1.6:
        mos_pct  = 20
        mos_note = f"Beta = {beta:.2f} (5yr monthly) → above-market volatility → 20%"
    else:
        mos_pct  = 25
        mos_note = f"Beta = {beta:.2f} (5yr monthly) → high volatility → 25%"

    return {
        "fcf_growth_pct":  fcf_growth_pct,
        "wacc_pct":        wacc_pct,
        "terminal_pct":    terminal_pct,
        "mos_pct":         mos_pct,
        "fcf_growth_note": fcf_growth_note,
        "wacc_note":       wacc_note,
        "terminal_note":   terminal_note,
        "mos_note":        mos_note,
    }
