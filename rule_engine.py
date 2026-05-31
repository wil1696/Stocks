"""
rule_engine.py
Stage 2b — Buffett-inspired rule engine.

Applies 11 hard pass/fail rules (plus one deferred placeholder, 3.3) to every
company in the universe and classifies each into shortlist / watchlist /
rejected. No scoring, no weights — pure rules, with sector-aware exemptions.

Design (see CLAUDE.md "Stage 2b"):
  - Inputs: latest fundamentals_snapshot (joined to companies.sector / GICS),
    the 4-year annual income/balance/cashflow history, and the latest
    sector_statistics medians (for the five sector-relative thresholds).
  - Each rule returns True / False / None. None means "not applicable" — either
    the rule is exempt for the company's GICS sector, OR it could not be
    evaluated because of missing data. Both are stored as SQL NULL.
  - applicable_count = the active rules (11; rule 3.3 is deferred) that yielded a
    concrete True/False for this company; passed_count = how many were True.
    pass_pct = passed / applicable.
        pass_pct == 1.0   → shortlist
        pass_pct >= 0.80  → watchlist
        else              → rejected

Data conventions verified against Supabase (2026-05-30), some of which differ
from a naive reading of the spec — see inline notes:
  - interest_expense is stored POSITIVE.
  - capex is stored NEGATIVE  → FCF = operating_cash_flow + capex.
  - dividends_paid is stored NEGATIVE (an outflow) → "paid" means value < 0.
  - balance_sheets has NO shares_outstanding column → the buyback leg of rule
    4.2 uses income_statements.shares_diluted (annual) instead.
  - Rule 1.1 "4 years of quarterly statements" is read as ">= 4 annual periods"
    (yfinance only ever gives us 5–8 quarters, never 16).

Usage:
    python3 rule_engine.py                       # full universe, write
    python3 rule_engine.py --dry-run             # compute + print, no writes
    python3 rule_engine.py --tickers MSFT,KO     # spot-check subset (prints grid)
    python3 rule_engine.py --limit 10            # first 10 of the universe
    python3 rule_engine.py --sector "Financials" # one GICS sector only
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from sector_stats import client, fetch_all


# ── Rule registry ──────────────────────────────────────────────────────────────
# (rule_id, db_column, category, short label). Order is the display order.
RULES: list[tuple[str, str, str, str]] = [
    ("1.1", "rule_1_1", "Quality",    "Operating history (>=4y)"),
    ("1.2", "rule_1_2", "Quality",    "Earnings consistency"),
    ("1.3", "rule_1_3", "Quality",    "ROIC vs sector & >=8%"),
    ("1.4", "rule_1_4", "Quality",    "Gross-margin trend"),
    ("2.1", "rule_2_1", "Financial",  "Net debt / EBITDA"),
    ("2.2", "rule_2_2", "Financial",  "Interest coverage >=5x"),
    ("2.3", "rule_2_3", "Financial",  "FCF consistency"),
    ("3.1", "rule_3_1", "Valuation",  "FCF yield vs sector"),
    ("3.2", "rule_3_2", "Valuation",  "EV/EBITDA vs sector"),
    ("3.3", "rule_3_3", "Valuation",  "Margin of safety (DEFERRED)"),
    ("4.1", "rule_4_1", "Trajectory", "Revenue growth vs sector"),
    ("4.2", "rule_4_2", "Trajectory", "Capital-return discipline"),
]
RULE_COL = {rid: col for rid, col, _, _ in RULES}
RULE_IDS = [rid for rid, *_ in RULES]

# 3.3 is not yet implemented; it is always NULL and excluded from applicable_count.
DEFERRED = {"3.3"}
ACTIVE_RULE_IDS = [rid for rid in RULE_IDS if rid not in DEFERRED]   # the 11 active rules

# Sector exemptions. NULL is stored (not False) for a company whose sector is
# listed here for a rule. Group on companies.sector (GICS naming).
RULE_NOT_APPLICABLE_FOR_SECTOR: dict[str, list[str]] = {
    "1.3": ["Financials"],   # banks use ROE, not ROIC
    "1.4": ["Financials"],   # gross margin not meaningful for banks
    "2.1": ["Financials"],   # debt is the product, not leverage
    "2.2": ["Financials"],   # interest expense is part of the business
    "3.1": ["Financials"],   # FCF doesn't apply cleanly to banks
    "3.2": ["Financials"],   # banks use P/B, not EV/EBITDA
    # TODO: REITs (Real Estate) need their own rules (P/FFO, P/AFFO, NAV, occupancy,
    # AFFO payout, LTV) — for now they run through the standard rules and will tend
    # to fail the FCF/EBITDA-based ones. Revisit when REIT-specific metrics exist.
}

# Categorisation thresholds.
WATCHLIST_THRESHOLD = 0.80

# Snapshot columns the rules read (besides ticker).
SNAPSHOT_COLS = [
    "roic", "operating_income_ttm", "interest_expense_ttm", "ebitda_ttm",
    "net_debt", "fcf_yield", "ev_ebitda", "revenue_cagr_3yr",
]


# ── Numeric coercion ───────────────────────────────────────────────────────────
def _f(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


# ── Data loading ───────────────────────────────────────────────────────────────
def load_snapshots() -> dict[str, dict]:
    """Latest snapshot per ticker, joined to GICS sector from companies (active
    universe only). Returns {ticker: {sector, company_name, <snapshot cols>}}."""
    snaps = fetch_all(
        "fundamentals_snapshot",
        "ticker,snapshot_date," + ",".join(SNAPSHOT_COLS),
    )
    # Most recent snapshot per ticker.
    latest: dict[str, dict] = {}
    for r in snaps:
        t = r["ticker"]
        if t not in latest or r["snapshot_date"] > latest[t]["snapshot_date"]:
            latest[t] = r

    comps = {c["ticker"]: c for c in fetch_all("companies", "ticker,company_name,sector,is_active")
             if c.get("is_active") and c.get("sector")}

    out: dict[str, dict] = {}
    for t, snap in latest.items():
        comp = comps.get(t)
        if not comp:
            continue   # no active companies row or null sector → can't sector-rank → skip
        row = {"ticker": t, "sector": comp["sector"], "company_name": comp.get("company_name")}
        for c in SNAPSHOT_COLS:
            row[c] = _f(snap.get(c))
        out[t] = row
    return out


def _group_annual(table: str, cols: list[str]) -> dict[str, list[dict]]:
    """Fetch a statement table (annual only) and group by ticker, most-recent
    period first, with the selected columns coerced to float."""
    rows = fetch_all(table, "ticker,period_end,period_type," + ",".join(cols))
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("period_type") != "annual":
            continue
        rec = {"period_end": r["period_end"]}
        for c in cols:
            rec[c] = _f(r.get(c))
        by_ticker[r["ticker"]].append(rec)
    for t in by_ticker:
        by_ticker[t].sort(key=lambda x: x["period_end"], reverse=True)
    return by_ticker


def _group_quarterly(table: str, cols: list[str]) -> dict[str, list[dict]]:
    """Same as _group_annual but for quarterly rows (the fallback series)."""
    rows = fetch_all(table, "ticker,period_end,period_type," + ",".join(cols))
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("period_type") != "quarterly":
            continue
        rec = {"period_end": r["period_end"]}
        for c in cols:
            rec[c] = _f(r.get(c))
        by_ticker[r["ticker"]].append(rec)
    for t in by_ticker:
        by_ticker[t].sort(key=lambda x: x["period_end"], reverse=True)
    return by_ticker


def _fcf(r: dict) -> Optional[float]:
    """Free cash flow for a cash-flow row: stored free_cash_flow, else OCF+capex
    (capex is stored negative)."""
    fcf = r.get("free_cash_flow")
    if fcf is not None:
        return fcf
    ocf, capex = r.get("operating_cash_flow"), r.get("capex")
    if ocf is not None and capex is not None:
        return ocf + capex
    return None


def load_sector_medians() -> dict[str, dict[str, float]]:
    """Latest sector_statistics medians as {sector: {metric_name: median}}."""
    rows = fetch_all("sector_statistics", "sector,metric_name,median,calculated_at")
    if not rows:
        return {}
    latest = max(r["calculated_at"] for r in rows)
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["calculated_at"] != latest:
            continue
        m = _f(r.get("median"))
        if m is not None:
            out[r["sector"]][r["metric_name"]] = m
    return out


# ── A company's full data context ──────────────────────────────────────────────
class Ctx:
    """All data needed to evaluate one company's rules."""

    def __init__(self, snap, inc_a, inc_q, bal_a, cf_a, cf_q, medians):
        self.snap = snap
        self.sector = snap["sector"]
        self.inc_a = inc_a          # annual income rows, newest first
        self.inc_q = inc_q          # quarterly income rows, newest first
        self.bal_a = bal_a          # annual balance rows, newest first
        self.cf_a = cf_a            # annual cashflow rows, newest first
        self.cf_q = cf_q            # quarterly cashflow rows, newest first
        self.bal_by_period = {b["period_end"]: b for b in bal_a}
        self._medians = medians

    def median(self, metric: str) -> Optional[float]:
        return self._medians.get(self.sector, {}).get(metric)


# ── Small numeric helpers shared by rules ───────────────────────────────────────
def _annual_or_ttm(annual_vals: list, quarterly_vals: list, n: int = 4) -> list[float]:
    """Up to n yearly figures, newest first. Prefer >=n annual values; otherwise
    build rolling 4-quarter TTM sums from quarterly data."""
    a = [v for v in annual_vals if v is not None]
    if len(a) >= n:
        return a[:n]
    q = [v for v in quarterly_vals if v is not None]
    if len(q) >= 4:
        ttm = [sum(q[i:i + 4]) for i in range(0, len(q) - 3)]   # newest-first windows
        return ttm[:n]
    return a


def _consistency(series: list[float], need: int = 3) -> Optional[bool]:
    """>= `need` positive values among (up to 4) periods. None if too little data."""
    vals = series[:4]
    if len(vals) < need:
        return None
    return sum(1 for v in vals if v > 0) >= need


def _or3(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    """Three-valued (Kleene) OR: True wins; unknown propagates over False."""
    if a is True or b is True:
        return True
    if a is None or b is None:
        return None
    return False


# ── The rules ───────────────────────────────────────────────────────────────────
def rule_1_1(c: Ctx) -> Optional[bool]:
    """Operating history: >= 4 annual income periods (~4 years)."""
    return len(c.inc_a) >= 4


def rule_1_2(c: Ctx) -> Optional[bool]:
    """Earnings consistency: positive net_income in >= 3 of the last 4 periods."""
    series = _annual_or_ttm([r.get("net_income") for r in c.inc_a],
                            [r.get("net_income") for r in c.inc_q])
    return _consistency(series)


def rule_1_3(c: Ctx) -> Optional[bool]:
    """ROIC (hybrid): 3yr-avg annual ROIC >= sector median AND >= 0.08.
    Falls back to TTM snapshot ROIC if annual ROIC can't be computed."""
    avg = _annual_roic_3yr(c)
    if avg is None:
        avg = c.snap.get("roic")          # TTM fallback
    if avg is None:
        return None
    med = c.median("roic")
    if med is None:
        return None
    return (avg >= med) and (avg >= 0.08)


def _annual_roic_3yr(c: Ctx) -> Optional[float]:
    """Mean of up to the last 3 annual ROIC values.
    ROIC = operating_income*(1-tax_rate) / (equity + debt - cash)."""
    vals: list[float] = []
    for r in c.inc_a[:3]:
        op, tax = r.get("operating_income"), r.get("tax_rate")
        if op is None or tax is None:
            continue
        b = c.bal_by_period.get(r["period_end"])
        if not b:
            continue
        eq, debt, cash = b.get("total_equity"), b.get("total_debt"), b.get("cash_and_equivalents")
        if eq is None or debt is None or cash is None:
            continue
        invested = eq + debt - cash
        if invested <= 0:
            continue
        vals.append(op * (1 - tax) / invested)
    return sum(vals) / len(vals) if vals else None


def rule_1_4(c: Ctx) -> Optional[bool]:
    """Gross-margin direction: latest annual gross margin >= that of 3 years ago."""
    if len(c.inc_a) < 4:
        return None

    def gm(r):
        gp, rev = r.get("gross_profit"), r.get("revenue")
        if gp is None or not rev:
            return None
        return gp / rev

    g0, g3 = gm(c.inc_a[0]), gm(c.inc_a[3])
    if g0 is None or g3 is None:
        return None
    return g0 >= g3


def rule_2_1(c: Ctx) -> Optional[bool]:
    """Net debt / EBITDA <= sector median(debt_ebitda) x 1.25."""
    nd, eb = c.snap.get("net_debt"), c.snap.get("ebitda_ttm")
    if nd is None or eb is None or eb <= 0:
        return None
    med = c.median("debt_ebitda")
    if med is None:
        return None
    return (nd / eb) <= med * 1.25   # net cash (nd < 0) → ratio negative → passes


def rule_2_2(c: Ctx) -> Optional[bool]:
    """Interest coverage: operating_income_ttm / interest_expense_ttm >= 5.0.
    No interest expense (None/0) → solvency not at risk → PASS."""
    ie = c.snap.get("interest_expense_ttm")
    if ie is None or abs(ie) == 0:
        return True
    oi = c.snap.get("operating_income_ttm")
    if oi is None:
        return None
    return (oi / abs(ie)) >= 5.0


def rule_2_3(c: Ctx) -> Optional[bool]:
    """FCF consistency: positive FCF in >= 3 of the last 4 periods."""
    series = _annual_or_ttm([_fcf(r) for r in c.cf_a],
                            [_fcf(r) for r in c.cf_q])
    return _consistency(series)


def rule_3_1(c: Ctx) -> Optional[bool]:
    """FCF yield >= sector median(fcf_yield)."""
    fy = c.snap.get("fcf_yield")
    if fy is None:
        return None
    med = c.median("fcf_yield")
    if med is None:
        return None
    return fy >= med


def rule_3_2(c: Ctx) -> Optional[bool]:
    """EV/EBITDA <= sector median(ev_ebitda) x 1.25. Non-positive EV/EBITDA
    (negative EBITDA) is not meaningful → not applicable."""
    ee = c.snap.get("ev_ebitda")
    if ee is None or ee <= 0:
        return None
    med = c.median("ev_ebitda")
    if med is None:
        return None
    return ee <= med * 1.25


def rule_3_3(c: Ctx) -> Optional[bool]:
    """Margin of safety vs DCF — DEFERRED, not yet implemented."""
    return None


def rule_4_1(c: Ctx) -> Optional[bool]:
    """Revenue growth: revenue_cagr_3yr >= sector median."""
    g = c.snap.get("revenue_cagr_3yr")
    if g is None:
        return None
    med = c.median("revenue_cagr_3yr")
    if med is None:
        return None
    return g >= med


def rule_4_2(c: Ctx) -> Optional[bool]:
    """Capital-return discipline: dividends paid in each of the last 4 years OR
    diluted share count declined over the last 3 years."""
    # Dividend leg — dividends_paid is stored NEGATIVE (an outflow).
    divs = [r.get("dividends_paid") for r in c.cf_a[:4]]
    paid = [(d is not None and d < 0) for d in divs]
    if len(c.cf_a) >= 4:
        div_leg: Optional[bool] = all(paid)
    elif all(paid):
        div_leg = None            # paid in every year we have, but <4 years to confirm
    else:
        div_leg = False           # a year with no dividend → definitively not "every year"

    # Buyback leg — shares_diluted latest vs 3-years-ago (no shares col on balance).
    def shares(r):
        return r.get("shares_diluted") or r.get("shares_basic")

    buy_leg: Optional[bool] = None
    if len(c.inc_a) >= 4:
        s0, s3 = shares(c.inc_a[0]), shares(c.inc_a[3])
        if s0 and s3:
            buy_leg = s0 < s3

    return _or3(div_leg, buy_leg)


RULE_FN = {
    "1.1": rule_1_1, "1.2": rule_1_2, "1.3": rule_1_3, "1.4": rule_1_4,
    "2.1": rule_2_1, "2.2": rule_2_2, "2.3": rule_2_3,
    "3.1": rule_3_1, "3.2": rule_3_2, "3.3": rule_3_3,
    "4.1": rule_4_1, "4.2": rule_4_2,
}


# ── Evaluation ──────────────────────────────────────────────────────────────────
def evaluate(c: Ctx, exempt_counts: dict, missing_counts: dict) -> dict[str, Optional[bool]]:
    """Evaluate every rule for one company. Handles deferral and sector
    exemption centrally; tallies why each rule came back N/A."""
    results: dict[str, Optional[bool]] = {}
    for rid in RULE_IDS:
        if rid in DEFERRED:
            results[rid] = None
            continue
        if c.sector in RULE_NOT_APPLICABLE_FOR_SECTOR.get(rid, []):
            results[rid] = None
            exempt_counts[rid] += 1
            continue
        v = RULE_FN[rid](c)
        results[rid] = v
        if v is None:
            missing_counts[rid] += 1
    return results


def categorise(results: dict[str, Optional[bool]]) -> tuple[int, int, Optional[float], str]:
    """(passed_count, applicable_count, pass_pct, category) over the active rules."""
    applicable = [results[rid] for rid in ACTIVE_RULE_IDS if results[rid] is not None]
    applicable_count = len(applicable)
    passed_count = sum(1 for v in applicable if v)
    if applicable_count == 0:
        return 0, 0, None, "rejected"
    pass_pct = passed_count / applicable_count
    if pass_pct >= 1.0:
        category = "shortlist"
    elif pass_pct >= WATCHLIST_THRESHOLD:
        category = "watchlist"
    else:
        category = "rejected"
    return passed_count, applicable_count, round(pass_pct, 4), category


def build_results(tickers: Optional[list[str]] = None) -> tuple[pd.DataFrame, dict, dict]:
    """Evaluate the universe (or a subset). Returns (df, exempt_counts, missing_counts)."""
    print("  Loading snapshots, statements, and sector medians...")
    snaps = load_snapshots()
    inc_a = _group_annual("income_statements",
                          ["net_income", "operating_income", "tax_rate",
                           "gross_profit", "revenue", "shares_diluted", "shares_basic"])
    inc_q = _group_quarterly("income_statements", ["net_income"])
    bal_a = _group_annual("balance_sheets",
                          ["total_equity", "total_debt", "cash_and_equivalents"])
    cf_a = _group_annual("cash_flows",
                         ["free_cash_flow", "operating_cash_flow", "capex", "dividends_paid"])
    cf_q = _group_quarterly("cash_flows",
                            ["free_cash_flow", "operating_cash_flow", "capex"])
    medians = load_sector_medians()

    universe = sorted(snaps)
    if tickers is not None:
        want = {t.upper() for t in tickers}
        universe = [t for t in universe if t in want]

    exempt_counts: dict[str, int] = defaultdict(int)
    missing_counts: dict[str, int] = defaultdict(int)

    rows: list[dict] = []
    for t in universe:
        c = Ctx(snaps[t], inc_a.get(t, []), inc_q.get(t, []),
                bal_a.get(t, []), cf_a.get(t, []), cf_q.get(t, []), medians)
        results = evaluate(c, exempt_counts, missing_counts)
        passed, applicable, pass_pct, category = categorise(results)
        row = {
            "ticker": t,
            "company_name": c.snap.get("company_name"),
            "sector": c.sector,
            "category": category,
            "passed_count": passed,
            "applicable_count": applicable,
            "pass_pct": pass_pct,
        }
        for rid in RULE_IDS:
            row[RULE_COL[rid]] = results[rid]
        rows.append(row)

    df = pd.DataFrame(rows)
    # Boolean (nullable) dtype for the rule columns so True/False/<NA> survive.
    for rid in RULE_IDS:
        df[RULE_COL[rid]] = df[RULE_COL[rid]].astype("boolean")
    return df, exempt_counts, missing_counts


# ── Persistence ─────────────────────────────────────────────────────────────────
def write_rows(df: pd.DataFrame, calculated_at: str) -> None:
    rows = []
    for _, r in df.iterrows():
        row = {
            "ticker": r["ticker"],
            "sector": r["sector"],
            "category": r["category"],
            "passed_count": int(r["passed_count"]),
            "applicable_count": int(r["applicable_count"]),
            "pass_pct": None if pd.isna(r["pass_pct"]) else float(r["pass_pct"]),
            "calculated_at": calculated_at,
        }
        for rid in RULE_IDS:
            v = r[RULE_COL[rid]]
            row[RULE_COL[rid]] = None if pd.isna(v) else bool(v)
        rows.append(row)

    BATCH = 500
    for i in range(0, len(rows), BATCH):
        client.table("rule_results").insert(rows[i:i + BATCH]).execute()
        print(f"  inserted {min(i + BATCH, len(rows))}/{len(rows)}")


# ── Console reporting ───────────────────────────────────────────────────────────
def _cell(v) -> str:
    if v is True:
        return "✓"
    if v is False:
        return "✗"
    return "–"


def print_grid(df: pd.DataFrame) -> None:
    """Compact per-ticker pass/fail grid — used for the spot-check subset."""
    header = (f"{'TICKER':<7}{'SECTOR':<26}"
              + "".join(f"{rid:>5}" for rid in RULE_IDS)
              + f"{'PASS':>8}{'CAT':>11}")
    print("\n" + header)
    print("─" * len(header))
    for _, r in df.sort_values("pass_pct", ascending=False, na_position="last").iterrows():
        cells = "".join(f"{_cell(r[RULE_COL[rid]]):>5}" for rid in RULE_IDS)
        pp = "—" if pd.isna(r["pass_pct"]) else f"{r['pass_pct']*100:.0f}%"
        passfrac = f"{r['passed_count']}/{r['applicable_count']}"
        print(f"{r['ticker']:<7}{(r['sector'] or '')[:25]:<26}{cells}"
              f"{passfrac:>8}{r['category']:>11}")
    print("\n  (✓ pass · ✗ fail · – not applicable)   PASS = passed/applicable")


def print_summary(df: pd.DataFrame, exempt_counts: dict, missing_counts: dict) -> None:
    n = len(df)
    cats = df["category"].value_counts().to_dict()
    print(f"\n{'=' * 64}")
    print(f"  {n} companies   "
          f"shortlist={cats.get('shortlist', 0)}  "
          f"watchlist={cats.get('watchlist', 0)}  "
          f"rejected={cats.get('rejected', 0)}")
    print(f"{'=' * 64}")

    print("\n  Per-rule pass rate (of companies where the rule applied):")
    for rid, col, cat, label in RULES:
        applicable = df[col].notna().sum()
        passed = (df[col] == True).sum()                     # noqa: E712
        rate = f"{passed / applicable * 100:5.1f}%" if applicable else "   n/a"
        flags = []
        if rid in DEFERRED:
            flags.append("DEFERRED")
        if exempt_counts.get(rid):
            flags.append(f"{exempt_counts[rid]} sector-exempt")
        if missing_counts.get(rid):
            flags.append(f"{missing_counts[rid]} missing-data")
        tail = ("   [" + ", ".join(flags) + "]") if flags else ""
        print(f"    {rid:>3} {label:<28} {rate}  ({passed}/{applicable} pass){tail}")

    # Flag any shortlisted company that cleared an unusually small rule set.
    thin = df[(df["category"] == "shortlist") & (df["applicable_count"] < 5)]
    if not thin.empty:
        print(f"\n  ⚠  {len(thin)} shortlisted on <5 applicable rules (data-thin — verify): "
              f"{', '.join(thin['ticker'].tolist()[:10])}")


# ── Entry point ─────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Apply Buffett-inspired pass/fail rules to the universe.")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print, but do not write.")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated subset (prints a per-ticker grid).")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N tickers.")
    parser.add_argument("--sector", type=str, default=None, help='Restrict to one GICS sector.')
    args = parser.parse_args(argv)

    print("=" * 64)
    print("  Rule engine — Buffett-inspired filter (Stage 2b)")
    print("=" * 64)

    explicit = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    df, exempt_counts, missing_counts = build_results(tickers=explicit)

    if df.empty:
        print("  No companies to evaluate — is the universe seeded / snapshotted?")
        return

    if args.sector:
        df = df[df["sector"] == args.sector]
        if df.empty:
            print(f"  No companies in sector '{args.sector}'.")
            return
    if args.limit:
        df = df.head(args.limit)

    # Print the per-ticker grid for small selections; always print the summary.
    if explicit or (args.limit and args.limit <= 25) or args.sector:
        print_grid(df)
    print_summary(df, exempt_counts, missing_counts)

    if args.dry_run:
        print("\n  --dry-run: no writes made.")
        return

    calculated_at = datetime.now(timezone.utc).isoformat()
    print(f"\nWriting {len(df)} rows to rule_results @ {calculated_at} ...")
    write_rows(df, calculated_at)
    print("\nDone.")


if __name__ == "__main__":
    main()
