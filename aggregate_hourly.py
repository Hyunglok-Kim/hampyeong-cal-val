#!/usr/bin/env python3
"""
Aggregate raw 5-min station timeseries to 1-hour for the web UI.

The web page is slow when it has to download ~76 MB of 5-min CSVs and
compute daily means in JS. This script keeps the raw files in
`data/timeseries/` (source of truth) and writes 1-hour-mean copies to
`data/timeseries_hourly/`. The HTML reads from the hourly folder.

Aggregation rules per column:
  - precipitation       → sum  (rate × time = total accumulated)
  - everything else     → mean (soil_moisture_*, soil_temperature_*,
                                tree_SM, tree_temperature, ...)

Time bucket: floor(time, 'H'). UTC, same as the raw files.

Usage:
  python3 aggregate_hourly.py
  python3 aggregate_hourly.py --src data/timeseries --dst data/timeseries_hourly
"""
import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SRC = ROOT / "data" / "timeseries"
DEFAULT_DST = ROOT / "data" / "timeseries_hourly"

SUM_COLS = {"precipitation"}        # everything else is averaged

# Output decimal places per variable (None = column-name fallback below).
# SM is kept at 3 decimals to preserve SMAP-level accuracy (~0.04 m³/m³);
# everything else rounds to 1 decimal to shrink the CSVs.
DECIMALS_DEFAULT = 1
DECIMALS_BY_NAME = {
    "soil_moisture_10cm": 3, "soil_moisture_20cm": 3,
    "tree_SM":             3,
}


def decimals_for(col: str) -> int:
    if col in DECIMALS_BY_NAME:
        return DECIMALS_BY_NAME[col]
    cl = col.lower()
    if "soil_moist" in cl or cl.endswith("_sm"):    # any future *_sm column
        return 3
    return DECIMALS_DEFAULT


def parse_time(s: str) -> datetime:
    # raw uses "YYYY-MM-DD HH:MM:SS"
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def aggregate_file(src: Path, dst: Path) -> tuple[int, int]:
    with src.open(newline="") as fp:
        reader = csv.DictReader(fp)
        cols = [c for c in reader.fieldnames if c != "time"]
        # bucket → {col → (sum, n)}
        buckets: dict[datetime, dict[str, list[float]]] = defaultdict(
            lambda: {c: [0.0, 0] for c in cols}
        )
        rows_in = 0
        for row in reader:
            try:
                t = parse_time(row["time"])
            except (ValueError, KeyError):
                continue
            rows_in += 1
            hour = t.replace(minute=0, second=0, microsecond=0)
            b = buckets[hour]
            for c in cols:
                v = row.get(c, "")
                if v == "" or v is None:
                    continue
                try:
                    fv = float(v)
                except ValueError:
                    continue
                b[c][0] += fv
                b[c][1] += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["time", *cols])
        for hour in sorted(buckets):
            b = buckets[hour]
            row = [hour.strftime("%Y-%m-%d %H:%M:%S")]
            for c in cols:
                s, n = b[c]
                if n == 0:
                    row.append("")
                else:
                    val = s if c in SUM_COLS else s / n
                    # round to N decimals, then drop trailing zeros to save bytes
                    row.append(f"{round(val, decimals_for(c)):g}")
            writer.writerow(row)
    return rows_in, len(buckets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = ap.parse_args()

    src_files = sorted(args.src.glob("HP_*.csv"))
    if not src_files:
        print(f"no CSVs found in {args.src}")
        return
    print(f"aggregating {len(src_files)} files: {args.src} → {args.dst}")
    total_in = total_out = 0
    for src in src_files:
        dst = args.dst / src.name
        n_in, n_out = aggregate_file(src, dst)
        total_in += n_in
        total_out += n_out
        print(f"  {src.name:20s}  {n_in:7d} → {n_out:6d} rows")
    print(f"done: {total_in:,} raw rows → {total_out:,} hourly rows")


if __name__ == "__main__":
    main()
