"""
save_fundamentals.py
Fetches fundamental data for the active companies universe via yfinance and
saves to Supabase. Universe-driven (reads `companies` where is_active=TRUE),
resilient (per-ticker try/except + retry with backoff), and resumable (skips
tickers already snapshotted today).

Usage:
    python3 save_fundamentals.py                       # full universe
    python3 save_fundamentals.py --limit 10            # first 10 only
    python3 save_fundamentals.py --tickers AAPL,MSFT   # explicit subset
    python3 save_fundamentals.py --sector "Information Technology"
    python3 save_fundamentals.py --no-resume           # re-process even if fresh today
"""
from __future__ import annotations

import argparse
import os
import time
import traceback
from datetime import date

from dotenv import load_dotenv
from supabase import create_client

from fundamentals import FundamentalsExtractor
from universe import add_universe_args, resolve_tickers

load_dotenv()

client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)

# ── Tuning ───────────────────────────────────────────────────────────────────
SLEEP_BETWEEN_TICKERS = 0.5     # seconds — gentle on yfinance
MAX_ATTEMPTS          = 3       # 1 initial + 2 retries
BACKOFF_BASE_SEC      = 2       # 2s, 4s, 8s


def upsert(table: str, rows: list[dict], conflict: str) -> int:
    if not rows:
        return 0
    client.table(table).upsert(rows, on_conflict=conflict).execute()
    return len(rows)


def already_snapshotted_today() -> set[str]:
    """Return the set of tickers that already have a snapshot for today.
    Used by --resume to skip work that's already fresh."""
    today = date.today().isoformat()
    res = (
        client.table("fundamentals_snapshot")
        .select("ticker")
        .eq("snapshot_date", today)
        .execute()
    )
    return {row["ticker"] for row in (res.data or [])}


def save_ticker_once(ticker: str) -> None:
    """Single attempt at saving all fundamentals for one ticker. Raises on
    failure; caller decides whether to retry."""
    ext = FundamentalsExtractor(ticker)

    for period in ("quarterly", "annual"):
        rows = ext.get_income_rows(period)
        n    = upsert("income_statements", rows, "ticker,period_end,period_type")
        print(f"  ✓ income_statements   ({period:9s}): {n:2d} rows")

    for period in ("quarterly", "annual"):
        rows = ext.get_balance_rows(period)
        n    = upsert("balance_sheets", rows, "ticker,period_end,period_type")
        print(f"  ✓ balance_sheets      ({period:9s}): {n:2d} rows")

    for period in ("quarterly", "annual"):
        rows = ext.get_cashflow_rows(period)
        n    = upsert("cash_flows", rows, "ticker,period_end,period_type")
        print(f"  ✓ cash_flows          ({period:9s}): {n:2d} rows")

    snapshot    = ext.get_snapshot()
    null_fields = [k for k, v in snapshot.items()
                   if v is None and k not in ("ticker", "snapshot_date")]
    upsert("fundamentals_snapshot", [snapshot], "ticker,snapshot_date")
    print(f"  ✓ fundamentals_snapshot: saved")

    if null_fields:
        print(f"  ⚠  null fields ({len(null_fields)}): {', '.join(null_fields[:8])}"
              + ("..." if len(null_fields) > 8 else ""))


def save_ticker(ticker: str, index: int, total: int) -> tuple[bool, str | None]:
    """Returns (success, error_message). Retries with exponential backoff on
    failure; one bad ticker never aborts the run."""
    prefix = f"[{index}/{total}] {ticker}"
    print(f"\n{'─' * 50}\n  {prefix}\n{'─' * 50}")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            save_ticker_once(ticker)
            return True, None
        except Exception as exc:
            if attempt < MAX_ATTEMPTS:
                wait = BACKOFF_BASE_SEC ** attempt    # 2, 4, 8
                print(f"  ⚠  attempt {attempt}/{MAX_ATTEMPTS} failed: {exc}"
                      f"\n     retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ✗ all {MAX_ATTEMPTS} attempts failed: {exc}")
                traceback.print_exc()
                return False, str(exc)


def print_summary(succeeded: list[str], failed: list[tuple[str, str]],
                  skipped: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Run summary")
    print(f"{'=' * 60}")
    print(f"  Succeeded : {len(succeeded)}")
    print(f"  Failed    : {len(failed)}")
    print(f"  Skipped   : {len(skipped)}  (resume — already fresh today)")

    if failed:
        print("\n  Failed tickers:")
        for ticker, err in failed:
            err_short = (err[:80] + "...") if len(err) > 80 else err
            print(f"    - {ticker:<8}  {err_short}")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch fundamentals for the active companies universe.",
    )
    add_universe_args(parser)
    args = parser.parse_args(argv)

    tickers = resolve_tickers(args)
    if not tickers:
        print("No tickers to process — is the companies table seeded?")
        return

    print("=" * 60)
    print(f"  Fundamentals fetch starting — {len(tickers)} tickers")
    print("=" * 60)

    # Resume: drop tickers we already processed today
    skipped: list[str] = []
    if not args.no_resume:
        fresh = already_snapshotted_today()
        if fresh:
            skipped = [t for t in tickers if t in fresh]
            tickers = [t for t in tickers if t not in fresh]
            print(f"  Resume: skipping {len(skipped)} already-fresh tickers "
                  f"({len(tickers)} remaining)")

    succeeded: list[str] = []
    failed:    list[tuple[str, str]] = []

    for i, ticker in enumerate(tickers, start=1):
        ok, err = save_ticker(ticker, i, len(tickers))
        if ok:
            succeeded.append(ticker)
        else:
            failed.append((ticker, err or "unknown"))
        time.sleep(SLEEP_BETWEEN_TICKERS)

    print_summary(succeeded, failed, skipped)


if __name__ == "__main__":
    main()
