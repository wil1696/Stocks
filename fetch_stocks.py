import os
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
PRICE_COLS = ["open", "high", "low", "close"]
BATCH_SIZE = 500

client = create_client(SUPABASE_URL, SUPABASE_KEY)


def upsert_company(ticker: str) -> None:
    info = yf.Ticker(ticker).info
    client.table("companies").upsert({
        "ticker":   ticker,
        "name":     info.get("longName"),
        "exchange": info.get("exchange"),
    }, on_conflict="ticker").execute()


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
        "fully_empty": len(fully_empty),
        "partial":     len(partial),
        "partial_detail": partial_detail,
    }


def upload_prices(ticker: str, rows: list) -> None:
    total_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(rows), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        print(f"      → Uploading batch {batch_num}/{total_batches}...", end=" ")
        client.table("stock_prices").upsert(
            rows[i:i + BATCH_SIZE],
            on_conflict="ticker,date"
        ).execute()
        print("done")


def fetch_and_save(ticker: str, index: int, total: int) -> None:
    prefix = f"[{index}/{total}] {ticker}"

    # Company info
    print(f"{prefix} — Saving company info...", end=" ")
    upsert_company(ticker)
    print("done")

    # Historical prices
    print(f"{prefix} — Fetching 5 years of price data...", end=" ")
    raw = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)
    raw = raw.reset_index()
    raw.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    raw["ticker"] = ticker
    raw["volume"] = raw["volume"].fillna(0).astype(int)
    total_rows = len(raw)
    print(f"{total_rows} rows fetched")

    # Analyze and skip rows with NaN price columns
    skip_info = analyze_skipped_rows(raw)
    clean = raw[~raw[PRICE_COLS].isna().any(axis=1)].copy()
    uploaded = len(clean)
    skipped = total_rows - uploaded

    # Upload
    rows = clean[["ticker", "date", "open", "high", "low", "close", "volume"]].to_dict(orient="records")
    upload_prices(ticker, rows)

    # Per-ticker summary
    skip_rate = (skipped / total_rows * 100) if total_rows > 0 else 0
    print(f"      → {uploaded} rows uploaded, {skipped} skipped ({skip_rate:.2f}%)")
    if skip_info["fully_empty"] > 0:
        print(f"         - {skip_info['fully_empty']} fully empty (all OHLCV columns NaN)")
    if skip_info["partial"] > 0:
        print(f"         - {skip_info['partial']} partial:")
        for missing_cols, count in skip_info["partial_detail"].items():
            print(f"           · {count} row(s) missing: {missing_cols}")


def main() -> None:
    print("=" * 50)
    print("Stock data fetch starting")
    print("=" * 50)

    total_uploaded = 0
    total_skipped = 0

    for i, ticker in enumerate(TICKERS, start=1):
        fetch_and_save(ticker, i, len(TICKERS))
        print()

    print("=" * 50)
    print(f"All done! {len(TICKERS)} companies processed.")
    print("=" * 50)


if __name__ == "__main__":
    main()
