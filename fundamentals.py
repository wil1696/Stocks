"""
fundamentals.py
Extracts and computes financial metrics from yfinance for a single ticker.
All values return None rather than raising exceptions on missing data.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf
from datetime import date
from typing import Optional


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _to_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _to_int(v) -> Optional[int]:
    f = _to_float(v)
    return int(f) if f is not None else None


def _safe_div(a, b) -> Optional[float]:
    """Return a/b, or None if either is None/NaN or b is zero."""
    a, b = _to_float(a), _to_float(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def _growth(new_val, old_val) -> Optional[float]:
    """Return (new - old) / abs(old), or None."""
    n, o = _to_float(new_val), _to_float(old_val)
    if n is None or o is None or o == 0:
        return None
    return (n - o) / abs(o)


# ── Main extractor ────────────────────────────────────────────────────────────

class FundamentalsExtractor:
    """
    Lazily loads yfinance financial statements and exposes clean row lists
    ready for Supabase upsert.
    """

    # yfinance row name variants (tries each in order until one matches)
    _ROWS: dict[str, list[str]] = {
        # Income statement
        "revenue":          ["Total Revenue", "Revenue"],
        "gross_profit":     ["Gross Profit"],
        "operating_income": ["Operating Income", "EBIT"],
        "ebitda":           ["EBITDA", "Normalized EBITDA"],
        "net_income":       ["Net Income", "Net Income Common Stockholders"],
        "eps_basic":        ["Basic EPS"],
        "eps_diluted":      ["Diluted EPS"],
        "shares_basic":     ["Basic Average Shares"],
        "shares_diluted":   ["Diluted Average Shares"],
        "interest_expense": ["Interest Expense", "Net Interest Income"],
        "tax_rate":         ["Tax Rate For Calcs"],
        # Balance sheet
        "total_assets":         ["Total Assets"],
        "current_assets":       ["Current Assets"],
        "cash":                 [
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Short Term Investments",
        ],
        "receivables":          ["Net Receivables", "Accounts Receivable", "Receivables"],
        "inventory":            ["Inventory"],
        "total_liabilities":    ["Total Liabilities Net Minority Interest", "Total Liabilities"],
        "current_liabilities":  ["Current Liabilities"],
        "total_debt":           ["Total Debt"],
        "long_term_debt":       ["Long Term Debt"],
        "total_equity":         [
            "Common Stock Equity",
            "Stockholders Equity",
            "Total Equity Gross Minority Interest",
        ],
        "retained_earnings":    ["Retained Earnings"],
        # Cash flow
        "operating_cf":  [
            "Operating Cash Flow",
            "Total Cash From Operating Activities",
            "Cash Flow From Continuing Operating Activities",
        ],
        "capex":         ["Capital Expenditure", "Capital Expenditures", "Purchase Of PPE"],
        "free_cf":       ["Free Cash Flow"],
        "dividends":     ["Common Stock Dividend Paid", "Payment Of Dividends", "Cash Dividends Paid"],
        "buybacks":      ["Repurchase Of Capital Stock", "Common Stock Repurchase",
                          "Repurchase Of Common Stock"],
        "cash_change":   ["Changes In Cash", "Change In Cash And Cash Equivalents"],
    }

    def __init__(self, ticker: str) -> None:
        self.ticker   = ticker
        self._yf      = yf.Ticker(ticker)
        self._cache: dict[str, pd.DataFrame] = {}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _stmt(self, kind: str, period: str) -> pd.DataFrame:
        """Load and cache a financial statement DataFrame."""
        key = f"{kind}_{period}"
        if key not in self._cache:
            try:
                mapping = {
                    ("income",   "quarterly"): lambda: self._yf.quarterly_income_stmt,
                    ("income",   "annual"):    lambda: self._yf.income_stmt,
                    ("balance",  "quarterly"): lambda: self._yf.quarterly_balance_sheet,
                    ("balance",  "annual"):    lambda: self._yf.balance_sheet,
                    ("cashflow", "quarterly"): lambda: self._yf.quarterly_cashflow,
                    ("cashflow", "annual"):    lambda: self._yf.cashflow,
                }
                df = mapping[(kind, period)]()
                self._cache[key] = df if (df is not None and not df.empty) else pd.DataFrame()
            except Exception:
                self._cache[key] = pd.DataFrame()
        return self._cache[key]

    def _get_row(self, df: pd.DataFrame, key: str) -> Optional[pd.Series]:
        for name in self._ROWS.get(key, [key]):
            if name in df.index:
                return df.loc[name]
        return None

    def _val(self, df: pd.DataFrame, key: str, col: int) -> Optional[float]:
        row = self._get_row(df, key)
        if row is None or col >= len(row):
            return None
        return _to_float(row.iloc[col])

    def _period_end(self, df: pd.DataFrame, col: int) -> Optional[str]:
        try:
            return pd.Timestamp(df.columns[col]).strftime("%Y-%m-%d")
        except Exception:
            return None

    def _fast_info_val(self, *attrs) -> Optional[float]:
        """Try multiple attribute names on fast_info, return first non-None."""
        try:
            fi = self._yf.fast_info
            for attr in attrs:
                try:
                    v = getattr(fi, attr, None)
                    if v is None:
                        v = fi[attr]
                    result = _to_float(v)
                    if result is not None:
                        return result
                except (KeyError, TypeError):
                    continue
        except Exception:
            pass
        return None

    # ── Public methods ────────────────────────────────────────────────────────

    def get_income_rows(self, period_type: str = "quarterly") -> list[dict]:
        df = self._stmt("income", period_type)
        if df.empty:
            return []

        rows = []
        n = len(df.columns)

        for i in range(n):
            period_end = self._period_end(df, i)
            if not period_end:
                continue

            revenue  = self._val(df, "revenue", i)
            g_profit = self._val(df, "gross_profit", i)
            op_inc   = self._val(df, "operating_income", i)
            ebitda   = self._val(df, "ebitda", i)
            net_inc  = self._val(df, "net_income", i)

            # Revenue growth — requires prior periods in the same DataFrame
            rev_yoy = rev_qoq = None
            if revenue is not None:
                if period_type == "quarterly":
                    rev_yoy = _growth(revenue, self._val(df, "revenue", i + 4)) if i + 4 < n else None
                    rev_qoq = _growth(revenue, self._val(df, "revenue", i + 1)) if i + 1 < n else None
                else:  # annual
                    rev_yoy = _growth(revenue, self._val(df, "revenue", i + 1)) if i + 1 < n else None

            rows.append({
                "ticker":             self.ticker,
                "period_end":         period_end,
                "period_type":        period_type,
                "revenue":            revenue,
                "revenue_growth_yoy": rev_yoy,
                "revenue_growth_qoq": rev_qoq,
                "gross_profit":       g_profit,
                "gross_margin":       _safe_div(g_profit, revenue),
                "operating_income":   op_inc,
                "operating_margin":   _safe_div(op_inc, revenue),
                "ebitda":             ebitda,
                "ebitda_margin":      _safe_div(ebitda, revenue),
                "net_income":         net_inc,
                "net_margin":         _safe_div(net_inc, revenue),
                "eps_basic":          self._val(df, "eps_basic", i),
                "eps_diluted":        self._val(df, "eps_diluted", i),
                "shares_basic":       _to_int(self._val(df, "shares_basic", i)),
                "shares_diluted":     _to_int(self._val(df, "shares_diluted", i)),
                "interest_expense":   self._val(df, "interest_expense", i),
                "tax_rate":           self._val(df, "tax_rate", i),
            })

        return rows

    def get_balance_rows(self, period_type: str = "quarterly") -> list[dict]:
        df = self._stmt("balance", period_type)
        if df.empty:
            return []

        rows = []
        for i in range(len(df.columns)):
            period_end = self._period_end(df, i)
            if not period_end:
                continue

            current_assets      = self._val(df, "current_assets", i)
            current_liabilities = self._val(df, "current_liabilities", i)
            inventory           = self._val(df, "inventory", i)
            total_debt          = self._val(df, "total_debt", i)
            cash                = self._val(df, "cash", i)
            total_equity        = self._val(df, "total_equity", i)
            total_assets        = self._val(df, "total_assets", i)

            net_debt = (
                (total_debt or 0) - (cash or 0)
                if total_debt is not None or cash is not None
                else None
            )

            # Quick ratio: (current assets - inventory) / current liabilities
            quick_numerator = None
            if current_assets is not None:
                quick_numerator = current_assets - (inventory or 0)

            rows.append({
                "ticker":               self.ticker,
                "period_end":           period_end,
                "period_type":          period_type,
                "total_assets":         total_assets,
                "current_assets":       current_assets,
                "cash_and_equivalents": cash,
                "total_receivables":    self._val(df, "receivables", i),
                "inventory":            inventory,
                "total_liabilities":    self._val(df, "total_liabilities", i),
                "current_liabilities":  current_liabilities,
                "total_debt":           total_debt,
                "long_term_debt":       self._val(df, "long_term_debt", i),
                "total_equity":         total_equity,
                "retained_earnings":    self._val(df, "retained_earnings", i),
                "net_debt":             net_debt,
                "current_ratio":        _safe_div(current_assets, current_liabilities),
                "quick_ratio":          _safe_div(quick_numerator, current_liabilities),
                "debt_to_equity":       _safe_div(total_debt, total_equity),
                "debt_to_assets":       _safe_div(total_debt, total_assets),
            })

        return rows

    def get_cashflow_rows(self, period_type: str = "quarterly") -> list[dict]:
        df = self._stmt("cashflow", period_type)
        if df.empty:
            return []

        rows = []
        for i in range(len(df.columns)):
            period_end = self._period_end(df, i)
            if not period_end:
                continue

            ocf   = self._val(df, "operating_cf", i)
            capex = self._val(df, "capex", i)

            # Use direct FCF if available, otherwise derive from OCF + CapEx
            fcf = self._val(df, "free_cf", i)
            if fcf is None and ocf is not None and capex is not None:
                fcf = ocf + capex  # capex is negative in yfinance

            rows.append({
                "ticker":             self.ticker,
                "period_end":         period_end,
                "period_type":        period_type,
                "operating_cash_flow": ocf,
                "capex":              capex,
                "free_cash_flow":     fcf,
                "dividends_paid":     self._val(df, "dividends", i),
                "share_buybacks":     self._val(df, "buybacks", i),
                "net_change_in_cash": self._val(df, "cash_change", i),
            })

        return rows

    def get_snapshot(self) -> dict:
        """
        Build a TTM-based fundamentals snapshot for today.

        Extended to include all fields required by the Investment Valuation
        Framework v2 AI analysis engine (analysis_engine.py / prompts.py).
        New fields are clearly marked with # NEW below.
        """

        # ── Market data ───────────────────────────────────────────────────────
        price        = self._fast_info_val("last_price", "regularMarketPrice")
        market_cap   = self._fast_info_val("market_cap", "marketCap")
        shares       = _to_int(self._fast_info_val("shares", "sharesOutstanding"))

        # ── yfinance .info for fields not in financial statements ─────────────
        # NEW: dividend yield, dividend per share, sector — from yf.info
        info = {}
        try:
            info = self._yf.info or {}
        except Exception:
            info = {}

        dividend_yield     = _to_float(info.get("dividendYield"))           # NEW
        dividend_per_share = _to_float(info.get("dividendRate"))            # NEW
        sector             = info.get("sector") or info.get("sectorDisp")   # NEW
        currency           = info.get("currency", "USD")                    # NEW
        exchange           = info.get("exchange") or info.get("exchangeName") # NEW
        company_name       = info.get("shortName") or info.get("longName")  # NEW
        beta               = _to_float(info.get("beta"))                    # NEW

        # ── TTM helpers ───────────────────────────────────────────────────────
        income_rows  = self.get_income_rows("quarterly")
        cf_rows      = self.get_cashflow_rows("quarterly")
        balance_rows = self.get_balance_rows("quarterly")

        # Annual income for revenue CAGR — need prior years
        annual_income = self.get_income_rows("annual")  # NEW

        def ttm_sum(rows: list[dict], key: str, n: int = 4) -> Optional[float]:
            vals = [r[key] for r in rows[:n] if r.get(key) is not None]
            return sum(vals) if vals else None

        def latest(rows: list[dict], key: str):
            for r in rows:
                if r.get(key) is not None:
                    return r[key]
            return None

        revenue_ttm    = ttm_sum(income_rows,  "revenue")
        gross_profit_ttm = ttm_sum(income_rows, "gross_profit")             # NEW
        ebitda_ttm     = ttm_sum(income_rows,  "ebitda")
        net_income_ttm = ttm_sum(income_rows,  "net_income")
        op_income_ttm  = ttm_sum(income_rows,  "operating_income")
        interest_exp_ttm = ttm_sum(income_rows, "interest_expense")         # NEW
        cfo_ttm        = ttm_sum(cf_rows,      "operating_cash_flow")
        fcf_ttm        = ttm_sum(cf_rows,      "free_cash_flow")
        dividends_ttm  = ttm_sum(cf_rows,      "dividends_paid")            # NEW
        buybacks_ttm   = ttm_sum(cf_rows,      "share_buybacks")            # NEW

        cash_latest    = latest(balance_rows, "cash_and_equivalents")
        debt_latest    = latest(balance_rows, "total_debt")
        equity_latest  = latest(balance_rows, "total_equity")
        assets_latest  = latest(balance_rows, "total_assets")
        tax_rate       = latest(income_rows,  "tax_rate")

        # ── Net debt ──────────────────────────────────────────────────────────
        # NEW: net_debt in snapshot for the AI engine
        net_debt = None
        if debt_latest is not None or cash_latest is not None:
            net_debt = (debt_latest or 0) - (cash_latest or 0)

        # ── Enterprise value ──────────────────────────────────────────────────
        ev = None
        if market_cap is not None and debt_latest is not None and cash_latest is not None:
            ev = market_cap + debt_latest - cash_latest

        # ── Valuation multiples ───────────────────────────────────────────────
        pe        = _safe_div(market_cap, net_income_ttm)
        ps        = _safe_div(market_cap, revenue_ttm)
        pb        = _safe_div(market_cap, equity_latest)
        ev_ebitda = _safe_div(ev, ebitda_ttm)
        ev_rev    = _safe_div(ev, revenue_ttm)
        price_fcf = _safe_div(market_cap, fcf_ttm)
        fcf_yield = _safe_div(fcf_ttm, market_cap)

        # PEG = PE / (revenue_growth_yoy * 100)
        rev_growth = latest(income_rows, "revenue_growth_yoy")
        peg = None
        if pe is not None and rev_growth is not None and rev_growth > 0:
            peg = pe / (rev_growth * 100)

        # ── Revenue CAGR (3yr) — from annual income ───────────────────────────
        # NEW: use annual rows to compute 3yr revenue CAGR for the AI prompt
        rev_cagr_3yr = None
        if len(annual_income) >= 4:
            # annual_income[0] = most recent, [3] = 3 years ago
            rev_now = annual_income[0].get("revenue")
            rev_3yr = annual_income[3].get("revenue")
            if rev_now and rev_3yr and rev_3yr > 0:
                rev_cagr_3yr = (rev_now / rev_3yr) ** (1 / 3) - 1

        # Revenue prior years for AI prompt (from annual)                   # NEW
        revenue_prior_yr = annual_income[1].get("revenue") if len(annual_income) > 1 else None
        revenue_2yr_ago  = annual_income[2].get("revenue") if len(annual_income) > 2 else None

        # ── Margin metrics ────────────────────────────────────────────────────
        gross_margin   = _safe_div(gross_profit_ttm, revenue_ttm)           # NEW
        operating_margin = _safe_div(op_income_ttm, revenue_ttm)            # NEW
        fcf_margin     = _safe_div(fcf_ttm, revenue_ttm)                    # NEW

        # ── Quality metrics ───────────────────────────────────────────────────
        roe = _safe_div(net_income_ttm, equity_latest)
        roa = _safe_div(net_income_ttm, assets_latest)

        # ROIC = NOPAT / Invested Capital
        roic = None
        if op_income_ttm is not None and tax_rate is not None:
            invested_cap = None
            if equity_latest is not None and debt_latest is not None and cash_latest is not None:
                invested_cap = equity_latest + debt_latest - cash_latest
            if invested_cap:
                nopat = op_income_ttm * (1 - tax_rate)
                roic  = _safe_div(nopat, invested_cap)

        fcf_conversion = _safe_div(fcf_ttm, net_income_ttm)

        # ── Dividend coverage ratio ───────────────────────────────────────────
        # NEW: FCF / dividends paid — key metric for dividend sustainability
        div_coverage = _safe_div(fcf_ttm, abs(dividends_ttm)) if dividends_ttm else None

        # ── Debt / EBITDA ─────────────────────────────────────────────────────
        # NEW: primary leverage metric for the AI framework
        debt_ebitda = _safe_div(debt_latest, ebitda_ttm)

        # ── Rule of 40 (for tech companies) ──────────────────────────────────
        # NEW: revenue_growth_yoy * 100 + fcf_margin * 100
        rule_of_40 = None
        if rev_growth is not None and fcf_margin is not None:
            rule_of_40 = (rev_growth * 100) + (fcf_margin * 100)

        return {
            # ── Identity ──────────────────────────────────────────────────────
            "ticker":              self.ticker,
            "snapshot_date":       date.today().isoformat(),
            "company_name":        company_name,        # NEW
            "sector":              sector,              # NEW
            "currency":            currency,            # NEW
            "exchange":            exchange,            # NEW

            # ── Market data ───────────────────────────────────────────────────
            "price":               price,
            "market_cap":          market_cap,
            "enterprise_value":    ev,
            "shares_outstanding":  shares,
            "beta":                beta,                # NEW

            # ── Income statement (TTM) ────────────────────────────────────────
            "revenue_ttm":         revenue_ttm,
            "revenue_prior_yr":    revenue_prior_yr,   # NEW
            "revenue_2yr_ago":     revenue_2yr_ago,    # NEW
            "gross_profit_ttm":    gross_profit_ttm,   # NEW
            "ebitda_ttm":          ebitda_ttm,
            "net_income_ttm":      net_income_ttm,
            "operating_income_ttm": op_income_ttm,     # NEW
            "interest_expense_ttm": interest_exp_ttm,  # NEW

            # ── Cash flow (TTM) ───────────────────────────────────────────────
            "cfo_ttm":             cfo_ttm,
            "fcf_ttm":             fcf_ttm,
            "dividends_paid_ttm":  dividends_ttm,      # NEW
            "buybacks_ttm":        buybacks_ttm,       # NEW

            # ── Balance sheet (latest quarter) ────────────────────────────────
            "cash":                cash_latest,        # NEW
            "total_debt":          debt_latest,        # NEW
            "net_debt":            net_debt,           # NEW
            "total_equity":        equity_latest,      # NEW
            "total_assets":        assets_latest,      # NEW
            "tax_rate":            tax_rate,           # NEW

            # ── Valuation multiples ───────────────────────────────────────────
            "pe_ratio":            pe,
            "ps_ratio":            ps,
            "pb_ratio":            pb,
            "ev_ebitda":           ev_ebitda,
            "ev_revenue":          ev_rev,
            "peg_ratio":           peg,
            "fcf_yield":           fcf_yield,
            "price_fcf":           price_fcf,
            "dividend_yield":      dividend_yield,     # NEW
            "dividend_per_share":  dividend_per_share, # NEW

            # ── Margin metrics ────────────────────────────────────────────────
            "gross_margin":        gross_margin,       # NEW
            "operating_margin":    operating_margin,   # NEW
            "fcf_margin":          fcf_margin,         # NEW

            # ── Quality metrics ───────────────────────────────────────────────
            "roe":                 roe,
            "roa":                 roa,
            "roic":                roic,
            "fcf_conversion":      fcf_conversion,

            # ── Leverage & coverage ───────────────────────────────────────────
            "debt_ebitda":         debt_ebitda,        # NEW
            "dividend_coverage":   div_coverage,       # NEW

            # ── Growth ────────────────────────────────────────────────────────
            "revenue_growth_yoy":  rev_growth,         # already existed in income rows, now in snapshot
            "revenue_cagr_3yr":    rev_cagr_3yr,       # NEW

            # ── Composite metrics ─────────────────────────────────────────────
            "rule_of_40":          rule_of_40,         # NEW
        }
