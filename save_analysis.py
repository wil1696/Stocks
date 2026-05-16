"""
save_analysis.py
================
Save an AI-generated analysis (produced by Claude in a dev session)
into the Supabase ai_analyses table. No Anthropic API key needed.

Usage:
    python3 save_analysis.py <ticker> <analysis_file.md>

The script auto-detects the verdict (BUY / WATCH / AVOID) from the text
and reads sector + financial profile from the latest fundamentals snapshot.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


def detect_verdict(text: str) -> str | None:
    """Find BUY / WATCH / AVOID in the text (case-insensitive, near 'verdict')."""
    m = re.search(r"VERDICT[:\s]+\**\s*(BUY|WATCH|AVOID)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    for v in ["BUY", "WATCH", "AVOID"]:
        if re.search(rf"\b{v}\b", text):
            return v
    return None


def main(ticker: str, file_path: str) -> None:
    text = Path(file_path).read_text(encoding="utf-8")
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    snap_res = (
        client.table("fundamentals_snapshot")
        .select("sector")
        .eq("ticker", ticker)
        .order("snapshot_date", desc=True)
        .limit(1)
        .execute()
    )
    sector = snap_res.data[0].get("sector") if snap_res.data else None

    row = {
        "ticker":            ticker.upper(),
        "analysis_date":     date.today().isoformat(),
        "verdict":           detect_verdict(text),
        "sector":            sector,
        "financial_profile": None,
        "discount_rate":     0.12,
        "full_text":         text,
        "generated_by":      "claude-code-session",
    }

    client.table("ai_analyses").upsert(row, on_conflict="ticker,analysis_date").execute()
    print(f"✓ Saved {ticker} analysis ({len(text)} chars, verdict={row['verdict']})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 save_analysis.py <ticker> <analysis_file.md>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
