"""
fetch_sp500.py
Fetches S&P 500 constituents from Wikipedia and groups them by GICS sector,
aligned with the 11 sectors in the investment framework.
"""

import io
import requests
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS   = {"User-Agent": "Mozilla/5.0 (compatible; research-script/1.0)"}

SECTOR_ORDER = [
    "Information Technology",
    "Communication Services",
    "Health Care",
    "Financials",
    "Energy",
    "Materials",
    "Consumer Staples",
    "Consumer Discretionary",
    "Industrials",
    "Utilities",
    "Real Estate",
]


def fetch_sp500() -> pd.DataFrame:
    html = requests.get(SP500_URL, headers=HEADERS, timeout=15).text
    tables = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})
    df = tables[0]
    df = df.rename(columns={
        "Symbol":               "ticker",
        "Security":             "name",
        "GICS Sector":          "sector",
        "GICS Sub-Industry":    "sub_industry",
        "Headquarters Location":"headquarters",
        "Date added":           "date_added",
        "CIK":                  "cik",
        "Founded":              "founded",
    })
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df[["ticker", "name", "sector", "sub_industry", "headquarters", "date_added"]]


def _get_market_cap(ticker: str) -> tuple[str, float | None]:
    try:
        v = yf.Ticker(ticker).fast_info.market_cap
        return ticker, float(v) if v else None
    except Exception:
        return ticker, None


def fetch_market_caps(tickers: list[str], workers: int = 20) -> dict[str, float | None]:
    """Fetches market cap for a list of tickers in parallel."""
    result = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_get_market_cap, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures), 1):
            ticker, cap = future.result()
            result[ticker] = cap
            if i % 50 == 0 or i == len(tickers):
                print(f"  Market cap progress: {i}/{len(tickers)}", end="\r")
    print()
    return result


def fmt_mcap(v: float | None) -> str:
    if v is None:
        return "N/A"
    if v >= 1e12:
        return f"${v / 1e12:.2f}T"
    if v >= 1e9:
        return f"${v / 1e9:.1f}B"
    if v >= 1e6:
        return f"${v / 1e6:.0f}M"
    return f"${v:,.0f}"


def add_market_caps(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a market_cap column to the DataFrame (fetched from yfinance)."""
    print("Fetching market caps from yfinance...")
    caps = fetch_market_caps(df["ticker"].tolist())
    df = df.copy()
    df["market_cap"] = df["ticker"].map(caps)
    return df


def summarize(df: pd.DataFrame) -> None:
    has_mcap = "market_cap" in df.columns
    total_mcap = df["market_cap"].sum() if has_mcap else None

    header = f"S&P 500 — {len(df)} companies across {df['sector'].nunique()} GICS sectors"
    if total_mcap:
        header += f"  |  Total market cap: {fmt_mcap(total_mcap)}"
    print(f"\n{header}\n")

    col_mcap = f"{'Mkt Cap':>10}" if has_mcap else ""
    print(f"{'Sector':<30} {'Cos':>4}  {col_mcap}  Top 5 tickers")
    print("─" * 90)

    for sector in SECTOR_ORDER:
        group = df[df["sector"] == sector]
        if group.empty:
            continue
        if has_mcap:
            group = group.sort_values("market_cap", ascending=False, na_position="last")
        tickers = "  ".join(group["ticker"].head(5).tolist())
        mcap_str = fmt_mcap(group["market_cap"].sum()) if has_mcap else ""
        print(f"{sector:<30} {len(group):>4}  {mcap_str:>10}  {tickers}")
    print()


def by_sector(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Returns a dict of {sector_name: DataFrame} for all 11 sectors."""
    return {sector: df[df["sector"] == sector].reset_index(drop=True)
            for sector in SECTOR_ORDER if sector in df["sector"].values}


if __name__ == "__main__":
    print("Fetching S&P 500 constituents from Wikipedia...")
    df = fetch_sp500()
    df = add_market_caps(df)

    summarize(df)

    print("Full list by sector (sorted by market cap):\n")
    sectors = by_sector(df)
    for sector, group in sectors.items():
        if "market_cap" in group.columns:
            group = group.sort_values("market_cap", ascending=False, na_position="last")

        print(f"\n{'─' * 70}")
        sector_mcap = fmt_mcap(group["market_cap"].sum()) if "market_cap" in group.columns else ""
        print(f"  {sector}  ({len(group)} companies)  {sector_mcap}")
        print(f"{'─' * 70}")

        sub_industries = group["sub_industry"].unique()
        for sub in sorted(sub_industries):
            sub_group = group[group["sub_industry"] == sub]
            if "market_cap" in sub_group.columns:
                sub_group = sub_group.sort_values("market_cap", ascending=False, na_position="last")
            print(f"\n  [{sub}]")
            for _, row in sub_group.iterrows():
                mcap_str = fmt_mcap(row.get("market_cap")) if "market_cap" in row else ""
                print(f"    {row['ticker']:<8}  {row['name']:<45}  {mcap_str:>10}")
