"""
seed_universe.py
Pulls the current S&P 500 constituent list from Wikipedia and upserts it into
the `companies` table in Supabase. Idempotent — safe to re-run weekly as the
index changes.

Behaviour:
  - Tickers in the current S&P 500 list are upserted with is_active=TRUE.
  - Tickers present in `companies` but NOT in the current list are flipped to
    is_active=FALSE (soft-delete, preserves historical stock_prices via FK).
  - All fields (company_name, sector, sub_industry, date_added) are refreshed
    on every run so Wikipedia changes flow through.

Usage:
    python3 seed_universe.py
    python3 seed_universe.py --dry-run         # print what would change, no writes
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

from fetch_sp500 import fetch_sp500

load_dotenv()

client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)


def _parse_date_added(value) -> str | None:
    """Wikipedia's 'Date added' column is mostly YYYY-MM-DD strings but has
    edge cases (NaN, empty, occasionally a parenthetical note). Return an ISO
    date string or None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    # Strip any trailing parenthetical (e.g. "1976-12-31 (founded)")
    s = s.split("(")[0].strip()
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def build_rows(df: pd.DataFrame) -> list[dict]:
    """Map the fetch_sp500 DataFrame to companies-table row dicts."""
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "ticker":       r["ticker"],
            "name":         r["name"],            # keep legacy column populated
            "company_name": r["name"],
            "sector":       r["sector"],
            "sub_industry": r["sub_industry"],
            "date_added":   _parse_date_added(r.get("date_added")),
            "is_active":    True,
        })
    return rows


def fetch_existing_tickers() -> set[str]:
    """Return the set of tickers currently in the companies table (any status)."""
    res = client.table("companies").select("ticker").execute()
    return {row["ticker"] for row in (res.data or [])}


def upsert_active(rows: list[dict]) -> None:
    """Upsert the active universe. Done in chunks to stay under Supabase
    payload limits."""
    BATCH = 200
    for i in range(0, len(rows), BATCH):
        client.table("companies").upsert(
            rows[i:i + BATCH],
            on_conflict="ticker",
        ).execute()
        print(f"  upserted {min(i + BATCH, len(rows))}/{len(rows)}")


def deactivate_dropped(current_tickers: set[str], existing: set[str]) -> list[str]:
    """Flip is_active=FALSE for tickers in `existing` but not in `current_tickers`."""
    dropped = sorted(existing - current_tickers)
    if not dropped:
        return []
    # Update in chunks to keep the .in_() filter reasonable
    BATCH = 100
    for i in range(0, len(dropped), BATCH):
        chunk = dropped[i:i + BATCH]
        client.table("companies").update({"is_active": False}).in_("ticker", chunk).execute()
    return dropped


def summarize_by_sector(df: pd.DataFrame) -> None:
    print("\nBy sector:")
    counts = df.groupby("sector").size().sort_values(ascending=False)
    for sector, n in counts.items():
        print(f"  {sector:<30} {n:>3}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the companies universe from S&P 500.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned changes without writing to Supabase.")
    args = parser.parse_args()

    print("=" * 60)
    print("  Seeding companies universe from S&P 500 (Wikipedia)")
    print("=" * 60)

    df = fetch_sp500()
    current_tickers = set(df["ticker"].tolist())
    print(f"  Fetched {len(df)} constituents across {df['sector'].nunique()} GICS sectors")

    existing = fetch_existing_tickers()
    to_add      = sorted(current_tickers - existing)
    to_refresh  = sorted(current_tickers & existing)
    to_drop     = sorted(existing - current_tickers)

    print(f"\n  Existing companies rows : {len(existing)}")
    print(f"  New tickers to add      : {len(to_add)}"
          + (f"  (e.g. {', '.join(to_add[:5])}{'...' if len(to_add) > 5 else ''})" if to_add else ""))
    print(f"  Existing to refresh     : {len(to_refresh)}")
    print(f"  Dropped from S&P 500    : {len(to_drop)}"
          + (f"  ({', '.join(to_drop)})" if to_drop and len(to_drop) <= 20 else ""))

    if args.dry_run:
        print("\n  --dry-run: no writes made.")
        summarize_by_sector(df)
        return

    print("\nUpserting active universe...")
    upsert_active(build_rows(df))

    if to_drop:
        print(f"\nDeactivating {len(to_drop)} dropped tickers...")
        deactivate_dropped(current_tickers, existing)

    summarize_by_sector(df)

    print("\n" + "=" * 60)
    print(f"  Done — {len(current_tickers)} active, {len(to_drop)} deactivated")
    print("=" * 60)


if __name__ == "__main__":
    main()
