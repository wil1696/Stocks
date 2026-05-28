"""
fetch_stocks.py
Pulls 5-year daily OHLCV from yfinance into Supabase `stock_prices` for the
active companies universe. Universe-driven, resilient, resumable.

Usage:
    python3 fetch_stocks.py                       # full universe
    python3 fetch_stocks.py --limit 10            # first 10 only
    python3 fetch_stocks.py --tickers AAPL,MSFT   # explicit subset
    python3 fetch_stocks.py --sector "Information Technology"
    python3 fetch_stocks.py --no-resume           # re-fetch even if today's data exists
"""
from __future__ import annotations

import argparse
import os
import time
import traceback
from datetime import date

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from supabase import create_client

from universe import add_universe_args, resolve_tickers

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

PRICE_COLS = ["open", "high", "low", "close"]
BATCH_SIZE = 500       # rows per upsert chunk

# ── Resilience tuning ────────────────────────────────────────────────────────
SLEEP_BETWEEN_TICKERS = 0.5
MAX_ATTEMPTS          = 3
BACKOFF_BASE_SEC      = 2

client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ── Resume helpers ───────────────────────────────────────────────────────────

def already_fetched_today() -> set[str]:
    """Return the set of tickers whose latest stock_prices row is today's
    date. A ticker in this set is treated as already-fresh by --resume."""
    today = date.today().isoformat()
    res = (
        client.table("stock_prices")
        .select("ticker")
        .eq("date", today)
        .execute()
    )
    return {row["ticker"] for row in (res.data or [])}


# ── Price helpers (unchanged behaviour from previous version) ────────────────

def analyze_skipped_rows(df: pd.DataFrame) -> dict:
    """Categorize rows with NaN price columns."""
    nan_mask = df[PRICE_COLS].isna()
    skipped = df[nan_mask.any(axis=1)]

    fully_empty = skipped[nan_mask.loc[skipped.index].all(axis=1)]
    partial     = skipped[~nan_mask.loc[skipped.index].all(axis=1)]

    partial_detail = {}
    for _, row in partial.iterrows():
        missing = [col for col in PRICE_COLS if pd.isna(row[col])]
        key = ", ".join(missing)
        partial_detail[key] = partial_detail.get(key, 0) + 1

    return {
        "fully_empty":    len(fully_empty),
        "partial":        len(partial),
        "partial_detail": partial_detail,
    }


def upload_prices(rows: list) -> None:
    total_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(rows), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        print(f"      → Uploading batch {batch_num}/{total_batches}...", end=" ")
        client.table("stock_prices").upsert(
            rows[i:i + BATCH_SIZE],
            on_conflict="ticker,date",
        ).execute()
        print("done")


# ── Per-ticker fetch ─────────────────────────────────────────────────────────

def fetch_and_save_once(ticker: str, prefix: str) -> None:
    """One attempt at fetching + uploading prices for a ticker. Raises on
    failure; caller handles retry. NOTE: company metadata is no longer
    written here — the `companies` table is owned by seed_universe.py."""
    print(f"{prefix} — Fetching 5 years of price data...", end=" ")
    raw = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)

    if raw.empty:
        print("no data returned")
        return

    raw = raw.reset_index()
    raw.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in raw.columns]
    raw["date"]   = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    raw["ticker"] = ticker
    raw["volume"] = raw["volume"].fillna(0).astype(int)
    total_rows = len(raw)
    print(f"{total_rows} rows fetched")

    skip_info = analyze_skipped_rows(raw)
    clean     = raw[~raw[PRICE_COLS].isna().any(axis=1)].copy()
    uploaded  = len(clean)
    skipped   = total_rows - uploaded

    rows = clean[["ticker", "date", "open", "high", "low", "close", "volume"]].to_dict(orient="records")
    upload_prices(rows)

    skip_rate = (skipped / total_rows * 100) if total_rows > 0 else 0
    print(f"      → {uploaded} rows uploaded, {skipped} skipped ({skip_rate:.2f}%)")
    if skip_info["fully_empty"] > 0:
        print(f"         - {skip_info['fully_empty']} fully empty (all OHLCV columns NaN)")
    if skip_info["partial"] > 0:
        print(f"         - {skip_info['partial']} partial:")
        for missing_cols, count in skip_info["partial_detail"].items():
            print(f"           · {count} row(s) missing: {missing_cols}")


def fetch_and_save(ticker: str, index: int, total: int) -> tuple[bool, str | None]:
    """Returns (success, error_message). Retries with exponential backoff."""
    prefix = f"[{index}/{total}] {ticker}"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            fetch_and_save_once(ticker, prefix)
            return True, None
        except Exception as exc:
            if attempt < MAX_ATTEMPTS:
                wait = BACKOFF_BASE_SEC ** attempt
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
    print(f"  Skipped   : {len(skipped)}  (resume — today's data already present)")

    if failed:
        print("\n  Failed tickers:")
        for ticker, err in failed:
            err_short = (err[:80] + "...") if len(err) > 80 else err
            print(f"    - {ticker:<8}  {err_short}")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch 5yr daily OHLCV for the active companies universe.",
    )
    add_universe_args(parser)
    args = parser.parse_args(argv)

    tickers = resolve_tickers(args)
    if not tickers:
        print("No tickers to process — is the companies table seeded?")
        return

    print("=" * 60)
    print(f"  Stock-price fetch starting — {len(tickers)} tickers")
    print("=" * 60)

    skipped: list[str] = []
    if not args.no_resume:
        fresh = already_fetched_today()
        if fresh:
            skipped = [t for t in tickers if t in fresh]
            tickers = [t for t in tickers if t not in fresh]
            print(f"  Resume: skipping {len(skipped)} already-fresh tickers "
                  f"({len(tickers)} remaining)")

    succeeded: list[str] = []
    failed:    list[tuple[str, str]] = []

    for i, ticker in enumerate(tickers, start=1):
        ok, err = fetch_and_save(ticker, i, len(tickers))
        if ok:
            succeeded.append(ticker)
        else:
            failed.append((ticker, err or "unknown"))
        print()
        time.sleep(SLEEP_BETWEEN_TICKERS)

    print_summary(succeeded, failed, skipped)


if __name__ == "__main__":
    main()
