from __future__ import annotations
import sys
import traceback
from datetime import datetime, timezone

import fetch_stocks
import save_fundamentals


def run() -> bool:
    started = datetime.now(timezone.utc)
    print(f"[refresh_all] Starting at {started.isoformat()}")

    prices_ok = _run_step("fetch_stocks", fetch_stocks.main)
    fundamentals_ok = _run_step("save_fundamentals", save_fundamentals.main)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    status = "SUCCESS" if (prices_ok and fundamentals_ok) else "PARTIAL FAILURE"
    print(f"\n[refresh_all] Done in {elapsed:.1f}s — {status}")
    print(f"  fetch_stocks:      {'OK' if prices_ok else 'FAILED'}")
    print(f"  save_fundamentals: {'OK' if fundamentals_ok else 'FAILED'}")
    return prices_ok and fundamentals_ok


def _run_step(name: str, fn) -> bool:
    print(f"\n{'=' * 60}\n[refresh_all] Running {name}...\n{'=' * 60}")
    try:
        fn()
        return True
    except Exception:
        print(f"\n[refresh_all] ERROR in {name}:")
        traceback.print_exc()
        return False


def lambda_handler(event: dict, context) -> dict:
    """AWS Lambda entry point."""
    success = run()
    return {"statusCode": 200 if success else 500, "body": "done"}


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
