"""
company_ranks.py
Computes each company's percentile rank for every metric, within its GICS
sector, and stores them in the company_ranks table.

percentile_rank is 0..100 where 100 = best-in-sector, ALWAYS — for inverted
("lower is better") multiples the rank is flipped so that cheap/low-leverage
names score high. This lets the dashboard show one consistent "higher = better"
column regardless of metric.

Method (per the approved design): empirical rank against the full set of sector
peers' actual values, not interpolation from the stored 5 percentile cut-offs.
A company's rank uses the midrank convention:
    base = (#peers below + 0.5 * #peers equal) / n * 100
    rank = 100 - base   for inverted metrics, else base

It deliberately reuses sector_stats.load_latest_snapshots / add_derived_metrics /
POSITIVE_ONLY so the peer universe and the ≤0 exclusions match sector_statistics
exactly. Run AFTER sector_stats in the daily pipeline.

Usage:
    python3 company_ranks.py                # rank all companies, write
    python3 company_ranks.py --dry-run      # compute + log, no writes
    python3 company_ranks.py --sector "Financials"
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from sector_stats import (
    METRICS, POSITIVE_ONLY, client,
    load_latest_snapshots, add_derived_metrics,
)


def rank_metric(sub: pd.DataFrame, metric: str) -> list[tuple[str, float, float]]:
    """Rank every company in one sector for one metric.

    Returns (ticker, raw_value, percentile_rank) for each company that has a
    usable value. Companies with null (or, for inverted multiples, non-positive)
    values are omitted — they can't be meaningfully ranked for this metric.
    """
    s = pd.to_numeric(sub[metric], errors="coerce")
    inverted = metric in POSITIVE_ONLY
    if inverted:
        s = s.where(s > 0)              # non-positive → not meaningful → NaN

    valid = s.dropna()
    if valid.empty:
        return []

    vals = valid.to_numpy(dtype=float)
    n = vals.size

    out: list[tuple[str, float, float]] = []
    for ticker, v in valid.items():
        below = int((vals < v).sum())
        equal = int((vals == v).sum())
        base = (below + 0.5 * equal) / n * 100.0
        rank = (100.0 - base) if inverted else base
        out.append((ticker, float(v), round(rank, 2)))
    return out


def build_rows(df: pd.DataFrame, calculated_at: str,
               only_sector: Optional[str] = None) -> list[dict]:
    rows: list[dict] = []
    sectors = [only_sector] if only_sector else sorted(df["sector"].unique())
    for sector in sectors:
        sub = df[df["sector"] == sector].set_index("ticker")
        for metric in METRICS:
            for ticker, raw_value, pct in rank_metric(sub, metric):
                rows.append({
                    "ticker":          ticker,
                    "metric_name":     metric,
                    "raw_value":       raw_value,
                    "percentile_rank": pct,
                    "sector":          sector,
                    "calculated_at":   calculated_at,
                })
    return rows


def write_rows(rows: list[dict]) -> None:
    """Insert in chunks. PK includes calculated_at, so each run appends history."""
    BATCH = 500
    for i in range(0, len(rows), BATCH):
        client.table("company_ranks").insert(rows[i:i + BATCH]).execute()
        print(f"  inserted {min(i + BATCH, len(rows))}/{len(rows)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute per-company percentile ranks.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and log, but do not write to Supabase.")
    parser.add_argument("--sector", type=str, default=None,
                        help='Restrict to one GICS sector (e.g. "Financials").')
    args = parser.parse_args(argv)

    print("=" * 60)
    print("  Company percentile ranks")
    print("=" * 60)

    df = load_latest_snapshots()
    if df.empty:
        print("  No snapshot data found — run save_fundamentals.py first.")
        return
    df = add_derived_metrics(df)

    if args.sector and args.sector not in set(df["sector"]):
        print(f"  Unknown sector '{args.sector}'. Available: {sorted(df['sector'].unique())}")
        return

    scope = df if not args.sector else df[df["sector"] == args.sector]
    rankable = set(scope["ticker"])

    calculated_at = datetime.now(timezone.utc).isoformat()
    rows = build_rows(df, calculated_at, only_sector=args.sector)

    ranked = {r["ticker"] for r in rows}
    print(f"  Ranked {len(ranked)}/{len(rankable)} companies across {len(METRICS)} metrics "
          f"→ {len(rows)} rank rows @ {calculated_at}")

    # Unrankable = had a snapshot + sector but no usable value for ANY metric.
    unrankable = sorted(rankable - ranked)
    if unrankable:
        print(f"  ⚠  {len(unrankable)} companies got no ranks at all: "
              f"{', '.join(unrankable[:10])}{'...' if len(unrankable) > 10 else ''}")
    print("  (companies with no active companies row or null sector are excluded upstream)")

    if args.dry_run:
        print("\n  --dry-run: no writes made.")
        return

    print("\nWriting to company_ranks...")
    write_rows(rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
