"""
refresh_all.py
Orchestrates the daily data refresh, in dependency order:
    1. fetch_stocks      (prices)
    2. save_fundamentals (snapshot)
    3. sector_stats      (per-sector descriptive statistics)
    4. company_ranks     (per-company percentile ranks within sector)

Universe-selection flags (--tickers / --limit / --sector / --no-resume) are
forwarded to steps 1-2 so a single command can drive an end-to-end refresh over
any subset of the universe. Steps 3-4 deliberately IGNORE those flags and always
run over the full universe: a sector statistic or percentile rank computed over a
partial subset would be meaningless.

Usage:
    python3 refresh_all.py                                # full universe
    python3 refresh_all.py --limit 10                     # smoke test
    python3 refresh_all.py --tickers AAPL,MSFT --no-resume

Universe seeding (`seed_universe.py`) is NOT run from here — it's a separate
weekly/manual job because the S&P 500 list rarely changes.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone

import fetch_stocks
import save_fundamentals
import sector_stats
import company_ranks
from universe import add_universe_args


def run(argv: list[str] | None = None) -> bool:
    parser = argparse.ArgumentParser(
        description="Refresh prices + fundamentals for the active companies universe.",
    )
    add_universe_args(parser)
    args = parser.parse_args(argv)

    # Reconstruct the argv to pass through to the two step scripts so they
    # see the same flags. This keeps the two steps in lock-step (same subset,
    # same resume policy) without inventing a shared-config object.
    step_argv = _rebuild_argv(args)

    started = datetime.now(timezone.utc)
    print(f"[refresh_all] Starting at {started.isoformat()}")
    if step_argv:
        print(f"[refresh_all] Forwarding flags: {' '.join(step_argv)}")

    prices_ok       = _run_step("fetch_stocks",      fetch_stocks.main,      step_argv)
    fundamentals_ok = _run_step("save_fundamentals", save_fundamentals.main, step_argv)

    # Steps 3-4 always run over the full universe (no step_argv): partial-subset
    # statistics/ranks are meaningless. Both depend on fresh snapshots from step 2.
    stats_ok = _run_step("sector_stats",  sector_stats.main,  [])
    ranks_ok = _run_step("company_ranks", company_ranks.main, [])

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    all_ok  = prices_ok and fundamentals_ok and stats_ok and ranks_ok
    status  = "SUCCESS" if all_ok else "PARTIAL FAILURE"
    print(f"\n[refresh_all] Done in {elapsed:.1f}s — {status}")
    print(f"  fetch_stocks:      {'OK' if prices_ok else 'FAILED'}")
    print(f"  save_fundamentals: {'OK' if fundamentals_ok else 'FAILED'}")
    print(f"  sector_stats:      {'OK' if stats_ok else 'FAILED'}")
    print(f"  company_ranks:     {'OK' if ranks_ok else 'FAILED'}")
    return all_ok


def _rebuild_argv(args: argparse.Namespace) -> list[str]:
    """Turn the parsed Namespace back into a flag list for the step scripts."""
    out: list[str] = []
    if args.tickers:
        out += ["--tickers", args.tickers]
    if args.limit:
        out += ["--limit", str(args.limit)]
    if args.sector:
        out += ["--sector", args.sector]
    if args.no_resume:
        out += ["--no-resume"]
    return out


def _run_step(name: str, fn, argv: list[str]) -> bool:
    print(f"\n{'=' * 60}\n[refresh_all] Running {name}...\n{'=' * 60}")
    try:
        fn(argv)
        return True
    except Exception:
        print(f"\n[refresh_all] ERROR in {name}:")
        traceback.print_exc()
        return False


def lambda_handler(event: dict, context) -> dict:
    """AWS Lambda entry point. Event may contain a `flags` key with a list
    of argv-style flags to forward (e.g. {"flags": ["--limit", "50"]})."""
    flags = event.get("flags") if isinstance(event, dict) else None
    success = run(flags)
    return {"statusCode": 200 if success else 500, "body": "done"}


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
