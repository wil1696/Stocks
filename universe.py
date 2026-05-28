"""
universe.py
Shared helper for selecting which tickers the data pipeline should process.

The single source of truth is the `companies` table in Supabase (seeded by
seed_universe.py). Both fetch_stocks.py and save_fundamentals.py call into here
so they stay in sync — no more duplicated TICKERS lists.

Public API:
  - load_active_tickers(...)     fetch the working set, with optional filters
  - add_universe_args(parser)    register --tickers / --limit / --sector flags
  - resolve_tickers(args)        turn parsed argparse Namespace into a ticker list
"""
from __future__ import annotations

import argparse
import os
from typing import Iterable

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

_client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)


def load_active_tickers(
    only: Iterable[str] | None = None,
    limit: int | None = None,
    sector: str | None = None,
) -> list[str]:
    """
    Return the list of tickers the pipeline should process.

    Parameters
    ----------
    only : explicit list — bypasses the DB query and just returns these (still
           uppercased and de-duplicated). Useful for testing a known subset.
    limit : cap the result to the first N tickers (after ordering by ticker).
    sector : restrict to one GICS sector ("Information Technology" etc).

    The DB query always filters is_active=TRUE so deactivated tickers are
    excluded.
    """
    if only:
        return _normalize(only)

    query = _client.table("companies").select("ticker").eq("is_active", True)
    if sector:
        query = query.eq("sector", sector)
    query = query.order("ticker")
    if limit:
        query = query.limit(limit)

    res = query.execute()
    return [row["ticker"] for row in (res.data or [])]


def _normalize(raw: Iterable[str]) -> list[str]:
    """Uppercase, strip, drop empties, preserve order, de-duplicate."""
    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        t = (t or "").strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ── argparse integration ─────────────────────────────────────────────────────

def add_universe_args(parser: argparse.ArgumentParser) -> None:
    """Register the standard universe-selection flags on a parser.

    All three pipeline scripts (fetch_stocks, save_fundamentals, refresh_all)
    use the same flags so muscle memory is consistent."""
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated override (e.g. AAPL,MSFT). Skips the DB query.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N tickers from the universe.",
    )
    parser.add_argument(
        "--sector",
        type=str,
        default=None,
        help='Restrict to one GICS sector (e.g. "Information Technology").',
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume — re-process tickers even if already up to date today.",
    )


def resolve_tickers(args: argparse.Namespace) -> list[str]:
    """Translate parsed args into the actual ticker list."""
    explicit = None
    if args.tickers:
        explicit = [t for t in args.tickers.split(",") if t.strip()]
    return load_active_tickers(only=explicit, limit=args.limit, sector=args.sector)
