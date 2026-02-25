"""
save_fundamentals.py
Fetches fundamental data for all tickers via yfinance and saves to Supabase.
Run this script once (and periodically) to keep fundamentals up to date.
"""
import os
from dotenv import load_dotenv
from supabase import create_client
from fundamentals import FundamentalsExtractor

load_dotenv()

TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)


def upsert(table: str, rows: list[dict], conflict: str) -> int:
    if not rows:
        return 0
    client.table(table).upsert(rows, on_conflict=conflict).execute()
    return len(rows)


def save_ticker(ticker: str, index: int, total: int) -> None:
    prefix = f"[{index}/{total}] {ticker}"
    print(f"\n{'─' * 50}")
    print(f"  {prefix}")
    print(f"{'─' * 50}")

    ext = FundamentalsExtractor(ticker)

    # ── Income statements ─────────────────────────────────────────────────────
    for period in ("quarterly", "annual"):
        rows = ext.get_income_rows(period)
        n    = upsert("income_statements", rows, "ticker,period_end,period_type")
        print(f"  ✓ income_statements   ({period:9s}): {n:2d} rows")

    # ── Balance sheets ────────────────────────────────────────────────────────
    for period in ("quarterly", "annual"):
        rows = ext.get_balance_rows(period)
        n    = upsert("balance_sheets", rows, "ticker,period_end,period_type")
        print(f"  ✓ balance_sheets      ({period:9s}): {n:2d} rows")

    # ── Cash flows ────────────────────────────────────────────────────────────
    for period in ("quarterly", "annual"):
        rows = ext.get_cashflow_rows(period)
        n    = upsert("cash_flows", rows, "ticker,period_end,period_type")
        print(f"  ✓ cash_flows          ({period:9s}): {n:2d} rows")

    # ── Snapshot ──────────────────────────────────────────────────────────────
    snapshot    = ext.get_snapshot()
    null_fields = [k for k, v in snapshot.items()
                   if v is None and k not in ("ticker", "snapshot_date")]
    upsert("fundamentals_snapshot", [snapshot], "ticker,snapshot_date")
    print(f"  ✓ fundamentals_snapshot: saved")

    if null_fields:
        print(f"  ⚠  null fields ({len(null_fields)}): {', '.join(null_fields)}")
    else:
        print(f"  ✓ all snapshot fields populated")


def main() -> None:
    print("=" * 50)
    print("  Fundamentals fetch starting")
    print("=" * 50)

    for i, ticker in enumerate(TICKERS, start=1):
        save_ticker(ticker, i, len(TICKERS))

    print(f"\n{'=' * 50}")
    print(f"  Done — {len(TICKERS)} tickers processed")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
