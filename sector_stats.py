"""
sector_stats.py
Computes descriptive statistics for each (GICS sector, metric) pair across the
fundamentals universe and stores them in the sector_statistics table.

This is the foundation for cross-sector percentile ranking (company_ranks.py):
sector stats give the peer-relative cut-offs (p10..p90) that let us judge a
company against its own sector instead of against textbook numbers.

Source of truth:
  - metric values  → latest fundamentals_snapshot row per ticker
  - sector grouping → companies.sector (GICS naming, e.g. "Information
    Technology"), NOT fundamentals_snapshot.sector (yfinance naming). The two
    use different vocabularies; the framework and the --sector flag both speak
    GICS, so we group on companies.sector.

Population-level by design: always computed over the full universe (a sector
statistic over a --limit subset would be meaningless). Standalone flags are for
inspection only.

Usage:
    python3 sector_stats.py                 # compute + write for all sectors
    python3 sector_stats.py --dry-run       # compute + log, no writes
    python3 sector_stats.py --sector "Financials"   # one sector only
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)

# ── Metric definitions ─────────────────────────────────────────────────────────
# The 19 metrics we profile. 17 map directly to fundamentals_snapshot columns;
# the last two are derived on the fly.
QUALITY_METRICS = [
    "roic", "roe", "roa", "fcf_margin", "operating_margin", "gross_margin",
    "revenue_cagr_3yr", "debt_ebitda", "fcf_conversion", "rule_of_40",
]
VALUE_METRICS = [
    "fcf_yield", "pe_ratio", "ev_ebitda", "ev_revenue", "pb_ratio",
    "dividend_yield", "dividend_coverage",
]
DERIVED_METRICS = ["earnings_yield", "p_ocf"]

METRICS = QUALITY_METRICS + VALUE_METRICS + DERIVED_METRICS

# Direct snapshot columns to pull (all metrics except the two derived ones),
# plus the inputs needed to derive earnings_yield and p_ocf.
DIRECT_METRICS = [m for m in METRICS if m not in DERIVED_METRICS]
SNAPSHOT_COLS = DIRECT_METRICS + ["market_cap", "cfo_ttm"]

# Inverted ("lower is better") multiples. For these, non-positive values are not
# meaningful (a negative P/E is lossmaking, not "cheap"), so we drop value <= 0
# before computing stats — mirroring how earnings_yield/p_ocf require > 0.
POSITIVE_ONLY = {"pe_ratio", "ev_ebitda", "ev_revenue", "pb_ratio", "debt_ebitda", "p_ocf"}

MIN_RELIABLE_SAMPLE = 10   # below this, stats are noisy — logged as a warning


# ── Supabase helpers ───────────────────────────────────────────────────────────
def fetch_all(table: str, cols: str) -> list[dict]:
    """Fetch every row from a table, paginating past PostgREST's 1000-row cap."""
    rows: list[dict] = []
    start, step = 0, 1000
    while True:
        res = client.table(table).select(cols).range(start, start + step - 1).execute()
        batch = res.data or []
        rows += batch
        if len(batch) < step:
            return rows
        start += step


def load_latest_snapshots() -> pd.DataFrame:
    """One row per active ticker: its most recent snapshot, with GICS sector
    attached from companies. Metric columns coerced to numeric."""
    snaps = fetch_all("fundamentals_snapshot",
                      "ticker,snapshot_date," + ",".join(SNAPSHOT_COLS))
    df = pd.DataFrame(snaps)
    if df.empty:
        return df

    # Latest snapshot per ticker
    df = (df.sort_values("snapshot_date", ascending=False)
            .drop_duplicates(subset="ticker", keep="first"))

    # Attach GICS sector from companies (active universe only)
    comps = pd.DataFrame(fetch_all("companies", "ticker,sector,is_active"))
    comps = comps[comps["is_active"] == True]                    # noqa: E712
    comps = comps[comps["sector"].notna()][["ticker", "sector"]]
    df = df.merge(comps, on="ticker", how="inner")

    # Coerce metric inputs to numeric (PostgREST may hand back strings)
    for col in SNAPSHOT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """earnings_yield = 1/pe_ratio (pe>0 only); p_ocf = market_cap/cfo_ttm (cfo>0)."""
    df = df.copy()
    pe = df["pe_ratio"]
    df["earnings_yield"] = np.where(pe > 0, 1.0 / pe, np.nan)

    cfo = df["cfo_ttm"]
    df["p_ocf"] = np.where(cfo > 0, df["market_cap"] / cfo, np.nan)
    return df


# ── Statistics ─────────────────────────────────────────────────────────────────
def _f(v) -> Optional[float]:
    """numpy scalar → JSON-safe Python float (NaN → None)."""
    if v is None:
        return None
    f = float(v)
    return None if np.isnan(f) else f


def clean_values(series: pd.Series, metric: str) -> np.ndarray:
    """Drop nulls, and for inverted multiples drop non-positive values."""
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if metric in POSITIVE_ONLY:
        vals = vals[vals > 0]
    return vals.to_numpy(dtype=float)


def compute_stats(values: np.ndarray) -> dict:
    """Descriptive stats for one cleaned metric array (len >= 1)."""
    med = np.median(values)
    p10, p25, p50, p75, p90 = np.percentile(values, [10, 25, 50, 75, 90])
    return {
        "sample_size": int(values.size),
        "min_value":   _f(values.min()),
        "max_value":   _f(values.max()),
        "mean":        _f(values.mean()),
        "median":      _f(med),
        # Population std/MAD: we have the whole sector, not a sample (ddof=0).
        "std_dev":     _f(values.std()),
        "mad":         _f(np.median(np.abs(values - med))),
        "p10":         _f(p10),
        "p25":         _f(p25),
        "p50":         _f(p50),
        "p75":         _f(p75),
        "p90":         _f(p90),
    }


def build_rows(df: pd.DataFrame, calculated_at: str,
               only_sector: Optional[str] = None) -> tuple[list[dict], list[tuple[str, str, int]]]:
    """Return (rows_to_insert, thin_pairs). thin_pairs are (sector, metric,
    sample_size) with sample_size < MIN_RELIABLE_SAMPLE."""
    rows: list[dict] = []
    thin: list[tuple[str, str, int]] = []

    sectors = [only_sector] if only_sector else sorted(df["sector"].unique())
    for sector in sectors:
        sub = df[df["sector"] == sector]
        for metric in METRICS:
            values = clean_values(sub[metric], metric)
            if values.size == 0:
                continue   # no usable data → no row
            stats = compute_stats(values)
            if stats["sample_size"] < MIN_RELIABLE_SAMPLE:
                thin.append((sector, metric, stats["sample_size"]))
            rows.append({
                "sector": sector,
                "metric_name": metric,
                "calculated_at": calculated_at,
                **stats,
            })
    return rows, thin


def write_rows(rows: list[dict]) -> None:
    """Insert in chunks. PK includes calculated_at, so each run appends history."""
    BATCH = 500
    for i in range(0, len(rows), BATCH):
        client.table("sector_statistics").insert(rows[i:i + BATCH]).execute()
        print(f"  inserted {min(i + BATCH, len(rows))}/{len(rows)}")


# ── Entry point ──────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute per-sector metric statistics.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and log, but do not write to Supabase.")
    parser.add_argument("--sector", type=str, default=None,
                        help='Restrict to one GICS sector (e.g. "Financials").')
    args = parser.parse_args(argv)

    print("=" * 60)
    print("  Sector statistics")
    print("=" * 60)

    df = load_latest_snapshots()
    if df.empty:
        print("  No snapshot data found — run save_fundamentals.py first.")
        return
    df = add_derived_metrics(df)

    if args.sector and args.sector not in set(df["sector"]):
        print(f"  Unknown sector '{args.sector}'. Available: {sorted(df['sector'].unique())}")
        return

    n_companies = df["sector"].notna().sum()
    n_sectors = df["sector"].nunique() if not args.sector else 1
    print(f"  Companies: {n_companies}   Sectors: {n_sectors}   Metrics: {len(METRICS)}")

    calculated_at = datetime.now(timezone.utc).isoformat()
    rows, thin = build_rows(df, calculated_at, only_sector=args.sector)
    print(f"  Computed {len(rows)} (sector × metric) statistic rows "
          f"@ {calculated_at}")

    if thin:
        print(f"\n  ⚠  {len(thin)} pairs below sample_size {MIN_RELIABLE_SAMPLE} "
              f"(noisy — interpret with care):")
        for sector, metric, n in thin:
            print(f"      {sector:<26} {metric:<18} n={n}")
    else:
        print(f"  ✓ every (sector × metric) pair has sample_size ≥ {MIN_RELIABLE_SAMPLE}")

    if args.dry_run:
        print("\n  --dry-run: no writes made.")
        return

    print("\nWriting to sector_statistics...")
    write_rows(rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
