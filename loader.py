"""Load CPCB/KSPCB 'wide' hourly air quality CSVs (as published on OpenCity).

File layout (one block per month):
    Year,2018
    July-2018,00:00:00,01:00:00,...,23:00:00      <- month header + 24 hour columns
    1,,,,,,,,,,,42.0,,...                         <- day number, then 24 hourly values
    2,...
Blank cells are missing readings. We turn this into a tidy hourly table.
"""
import csv
from datetime import datetime

import pandas as pd

MONTHS = {
    name: i
    for i, name in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"],
        start=1,
    )
}


def _to_float(cell):
    try:
        return float(cell)
    except (TypeError, ValueError):
        return float("nan")


def load_wide_hourly(path, station=None):
    """Return a DataFrame with columns: timestamp, value, station (hourly, sorted).

    Missing readings are kept as NaN so later quality checks can measure gaps.
    """
    rows = []
    year = month = None
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row:
                continue
            first = row[0].strip()
            if not first or first.lower() == "year":
                continue
            # Month header, e.g. "July-2018"
            if "-" in first and not first.isdigit():
                name, _, yr = first.partition("-")
                if name in MONTHS and yr.strip().isdigit():
                    month, year = MONTHS[name], int(yr)
                continue
            # Day row, e.g. "17,12.0,,,..."
            if first.isdigit() and year and month:
                day = int(first)
                for hour, cell in enumerate(row[1:25]):
                    try:
                        ts = datetime(year, month, day, hour)
                    except ValueError:  # e.g. day 31 in a 30-day month
                        break
                    rows.append((ts, _to_float(cell)))
    df = pd.DataFrame(rows, columns=["timestamp", "value"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df["station"] = station or str(path).rsplit("/", 1)[-1]
    return df


def coverage_report(df):
    """Quick data-quality summary: how much is missing and the longest gap (hours)."""
    missing = df["value"].isna()
    longest = run = 0
    for m in missing:
        run = run + 1 if m else 0
        longest = max(longest, run)
    return {
        "rows": len(df),
        "first": df["timestamp"].min(),
        "last": df["timestamp"].max(),
        "missing_pct": round(100 * missing.mean(), 1),
        "longest_gap_hours": longest,
        "max_value": df["value"].max(),
    }
