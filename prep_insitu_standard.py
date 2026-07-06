#!/usr/bin/env python3
"""
prep_insitu_standard.py — Convert the ISMN-format GIST_ISMN_user dataset
into the wide-format CSVs the portal currently consumes.

Source (read-only):
    /Users/hyunglokkim/data_1/IN_SITU/Sensor_Data/GIST_ISMN_user/
        Soil_Moisture/      GIST_GIST_<STATION>_<LC>_<ST>_sm_<DF>_<DT>_<SDATE>.csv
        Soil_Temperature/   GIST_GIST_<STATION>_<LC>_<ST>_temp_<DF>_<DT>_<SDATE>.csv
        Metadata/           GIST_GIST_<STATION>_<LC>_<ST>.csv
        Readme.txt

47 unique stations (HP0102 … HP2503), 82 sensor entries — forest stations
have both *_root (0.10 m) and *_tree (-1.00 m); rice-paddy stations have
*_irr (0.10 m) and *_bank (0.20 m); upland / grassland have a single sensor.

This script writes:
    data/in_situ/stations.csv                       — one row per station
    data/in_situ/timeseries/<STATION>.csv           — 5-min wide format
    data/in_situ/timeseries_hourly/<STATION>.csv    — hourly means
    data/in_situ/metadata/<STATION>_<LC>_<ST>.csv   — copies of source meta

Wide-format column names per sensor type:
    sm_irr_10cm  / temp_irr_10cm        (rice paddy irrigation zone)
    sm_bank_20cm / temp_bank_20cm       (rice paddy bank)
    sm_field_10cm / temp_field_10cm     (upland field)
    sm_root_10cm / temp_root_10cm       (forest tree root)
    sm_soil_10cm / temp_soil_10cm       (other grassland)
    sm_tree_above / temp_tree_above     (forest tree trunk, sensor 1 m above ground)

Run:
    python3 prep_insitu_standard.py
"""
import csv
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SRC = Path("/Users/hyunglokkim/data_1/IN_SITU/Sensor_Data/GIST_ISMN_user")
ROOT = Path(__file__).resolve().parent
OUT  = ROOT / "data" / "in_situ"
OUT_TS_5MIN  = OUT / "timeseries"
OUT_TS_HOUR  = OUT / "timeseries_hourly"
OUT_META     = OUT / "metadata"

# Depth → token used in the column name.
def depth_token(depth_m: float, sensortype: str) -> str:
    if depth_m < 0:
        # only the tree-trunk case (-1.00 m); collapse to a stable label
        return "above"
    cm = int(round(depth_m * 100))
    return f"{cm}cm"


def col_name(var: str, sensortype: str, depth_m: float) -> str:
    """var is 'sm' or 'temp'."""
    return f"{var}_{sensortype}_{depth_token(depth_m, sensortype)}"


FILE_RE = re.compile(
    # Trailing _YYYYMMDD start-date is optional: the upstream dataset moved
    # to the canonical ISMN naming (no date in the filename) in mid-2026.
    r"^GIST_GIST_(HP\d{4})_([A-Z]{2})_([a-z]+)_(sm|temp)_(-?\d+\.\d+)_(-?\d+\.\d+)(?:_\d{8})?\.csv$"
)
META_RE = re.compile(r"^GIST_GIST_(HP\d{4})_([A-Z]{2})_([a-z]+)\.csv$")


def parse_data_file(p: Path):
    m = FILE_RE.match(p.name)
    if not m:
        return None
    station, lc, st, var, df, dt = m.groups()
    return {
        "station": station, "lc": lc, "sensortype": st,
        "var": var, "depth_m": float(df),
        "path": p,
    }


def parse_meta_file(p: Path):
    m = META_RE.match(p.name)
    if not m:
        return None
    station, lc, st = m.groups()
    rows = {}
    with p.open() as fp:
        for row in csv.reader(fp):
            if not row or not row[0]:
                continue
            key = row[0].strip()
            val = row[1].strip() if len(row) > 1 else ""
            # multiple climate rows — keep the 2017 one if present
            if key == "Climate_classification" and key in rows:
                continue
            rows[key] = val
    return {"station": station, "lc": lc, "sensortype": st, "meta": rows}


def parse_dt(date_str: str, time_str: str) -> datetime:
    # "2025/10/28", "06:35"
    return datetime.strptime(f"{date_str} {time_str}", "%Y/%m/%d %H:%M")


def main():
    # ───────────────────────────────────────────── meta
    # Key by (station, sensortype) only — the landcover token has typos in
    # the source dataset (RF vs RP, GR vs UL, CF vs DF for a handful of
    # stations) that would otherwise cause look-up misses.
    meta_by_key = {}     # (station, sensortype) → meta dict (lc preserved inside)
    meta_by_station = {} # station → first meta dict (fallback when sensortype mismatches)
    lc_disagreements = []
    sensortype_disagreements = []
    for p in sorted((SRC / "Metadata").glob("GIST_GIST_*.csv")):
        info = parse_meta_file(p)
        if not info:
            continue
        meta_by_key[(info["station"], info["sensortype"])] = info
        meta_by_station.setdefault(info["station"], info)

    # ───────────────────────────────────────────── data files (root level only)
    data_files = []
    for sub in ("Soil_Moisture", "Soil_Temperature"):
        for p in sorted((SRC / sub).glob("GIST_GIST_*.csv")):
            info = parse_data_file(p)
            if info:
                data_files.append(info)
    print(f"[scan] {len(data_files)} data files, {len(meta_by_key)} metadata files")

    # group by station, then by (sensortype, depth, var)
    by_station = defaultdict(list)
    for f in data_files:
        by_station[f["station"]].append(f)

    print(f"[scan] {len(by_station)} unique stations")

    # ───────────────────────────────────────────── per-station 5-min CSV
    OUT_TS_5MIN.mkdir(parents=True, exist_ok=True)
    OUT_TS_HOUR.mkdir(parents=True, exist_ok=True)
    OUT_META.mkdir(parents=True, exist_ok=True)

    stations_rows = []   # for stations.csv
    # ── overlay + pulse indices (pre-computed here so index.html doesn't
    # ── have to fetch + parse 47 hourly CSVs on every page load).
    daily_by_date = {}         # "YYYY-MM-DD" → { station_id → mean surface SM }
    last_ts_by_station = {}    # station_id → "YYYY-MM-DD" of newest row

    def is_surface_sm(col):
        """Which columns feed the surface-SM map dot / pulse recency.
        `sm_tree_above` is above-ground bole moisture, not soil."""
        return col.startswith("sm_") and col != "sm_tree_above"

    for station in sorted(by_station):
        files = by_station[station]
        # collect column set for this station
        col_set = {}  # column_name → (var, sensortype, depth_m)
        for f in files:
            c = col_name(f["var"], f["sensortype"], f["depth_m"])
            col_set[c] = (f["var"], f["sensortype"], f["depth_m"])
        cols = sorted(col_set.keys())

        # accumulate by timestamp
        rows = defaultdict(dict)   # datetime → {col: value}
        for f in files:
            c = col_name(f["var"], f["sensortype"], f["depth_m"])
            with f["path"].open(newline="") as fp:
                reader = csv.reader(fp)
                header = next(reader, None)
                if not header:
                    continue
                # expected: Date,Time,<varname>,Quality_flag,Location_flag
                for r in reader:
                    if len(r) < 3:
                        continue
                    try:
                        t = parse_dt(r[0], r[1])
                        v = float(r[2])
                    except (ValueError, IndexError):
                        continue
                    # use quality flag if available — drop rows flagged > 0
                    if len(r) >= 4 and r[3] not in ("", "0"):
                        continue
                    rows[t][c] = v

        if not rows:
            print(f"  {station}: no usable rows, skipped")
            continue

        # write 5-min CSV
        out_5min = OUT_TS_5MIN / f"{station}.csv"
        with out_5min.open("w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["time", *cols])
            for t in sorted(rows):
                rec = rows[t]
                w.writerow([t.strftime("%Y-%m-%d %H:%M:%S"),
                            *[rec.get(c, "") for c in cols]])

        # daily surface-SM mean for this station (feeds the map dot color
        # overlay + the "recent" pulse-ring window).
        daily_sums = defaultdict(lambda: [0.0, 0])
        for t, rec in rows.items():
            day = t.strftime("%Y-%m-%d")
            for c, v in rec.items():
                if is_surface_sm(c):
                    daily_sums[day][0] += v
                    daily_sums[day][1] += 1
        for day, (s, n) in daily_sums.items():
            if n:
                daily_by_date.setdefault(day, {})[station] = round(s / n, 3)
        last_ts_by_station[station] = max(t.strftime("%Y-%m-%d") for t in rows)

        # hourly aggregation
        hourly = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
        for t, rec in rows.items():
            h = t.replace(minute=0, second=0, microsecond=0)
            for c, v in rec.items():
                hourly[h][c][0] += v
                hourly[h][c][1] += 1

        out_hour = OUT_TS_HOUR / f"{station}.csv"
        with out_hour.open("w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["time", *cols])
            for h in sorted(hourly):
                rec = hourly[h]
                vals = []
                for c in cols:
                    s, n = rec.get(c, [0.0, 0])
                    if n == 0:
                        vals.append("")
                    else:
                        val = s / n
                        # 3 decimals for SM, 1 for temp
                        if c.startswith("sm_"):
                            vals.append(f"{round(val, 3):g}")
                        else:
                            vals.append(f"{round(val, 1):g}")
                w.writerow([h.strftime("%Y-%m-%d %H:%M:%S"), *vals])

        # ───────────────────────── station record from metadata
        # Use the first sensortype's metadata for lat/lon/alt (they're the
        # same for both sensors at one physical station).
        chosen_meta = None
        for f in files:
            key = (station, f["sensortype"])
            if key in meta_by_key:
                chosen_meta = meta_by_key[key]
                meta_lc = chosen_meta["lc"]
                if meta_lc != f["lc"]:
                    lc_disagreements.append(
                        f"{station}: data files use '{f['lc']}', metadata uses '{meta_lc}'"
                    )
                break
        if chosen_meta is None and station in meta_by_station:
            # Station-level fallback for cases where both landcover AND
            # sensortype tokens disagree between data and metadata.
            chosen_meta = meta_by_station[station]
            data_sensortypes = sorted({f["sensortype"] for f in files})
            sensortype_disagreements.append(
                f"{station}: data sensortype(s) {data_sensortypes}, "
                f"metadata uses '{chosen_meta['sensortype']}' "
                f"(lc {chosen_meta['lc']})"
            )
        if chosen_meta is None:
            print(f"  {station}: no metadata, using defaults")
            continue
        m = chosen_meta["meta"]

        # land cover description: pretty version
        lc_pretty = m.get("Landcover_classification", "")
        # depths summary: e.g. "10, 20" or "10, -100"
        depth_cms = sorted({int(round(c[2] * 100)) for c in col_set.values()})
        depths_str = ", ".join(str(d) for d in depth_cms)
        # sensor flavors at this station, comma-joined: e.g. "irr, bank"
        flavors = sorted({c[1] for c in col_set.values()})

        # parse install date "MM/DD/YY" → "YYYY-MM-DD"
        install = m.get("Installed", "")
        try:
            install_iso = datetime.strptime(install, "%m/%d/%y").strftime("%Y-%m-%d")
        except ValueError:
            install_iso = install

        stations_rows.append({
            "station_id": station,
            "lat": m.get("Latitude", ""),
            "lon": m.get("Longitude", ""),
            "elevation_m": m.get("Altitude", ""),
            "land_cover": lc_pretty,
            "install_date": install_iso,
            "sensor_model": "METER TEROS-12",
            "sensor_flavors": ", ".join(flavors),
            "depths_cm": depths_str,
            "clay_pct": m.get("Clay_fraction", ""),
            "sand_pct": m.get("Sand_fraction", ""),
            "silt_pct": m.get("Silt_fraction", ""),
            "organic_carbon_pct": m.get("Organic_carbon", ""),
            "climate": m.get("Climate_classification", ""),
            "notes": "",
        })

        print(f"  {station}: {len(rows):>6} rows, {len(cols)} cols [{', '.join(cols)}]")

    # ───────────────────────────────────────────── stations.csv
    stations_csv = OUT / "stations.csv"
    fields = ["station_id", "lat", "lon", "elevation_m", "land_cover",
              "install_date", "sensor_model", "sensor_flavors", "depths_cm",
              "clay_pct", "sand_pct", "silt_pct", "organic_carbon_pct",
              "climate", "notes"]
    with stations_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in sorted(stations_rows, key=lambda x: x["station_id"]):
            w.writerow(r)
    print(f"[done] wrote {stations_csv} ({len(stations_rows)} stations)")

    # ───────────────────────────────────────────── copy metadata files
    for meta_p in sorted((SRC / "Metadata").glob("GIST_GIST_*.csv")):
        shutil.copy2(meta_p, OUT_META / meta_p.name.replace("GIST_GIST_", ""))
    print(f"[done] copied {len(list(OUT_META.glob('*.csv')))} metadata files")

    # ───────────────────────────────────────────── overlay / pulse indices
    # Single small JSON avoids the browser having to fetch + parse every
    # station's hourly CSV on page load just to color the map dots.
    daily_json = OUT / "daily_sm.json"
    with daily_json.open("w") as fp:
        json.dump({
            "dates": sorted(daily_by_date.keys()),
            "byDate": daily_by_date,
        }, fp, separators=(",", ":"))
    last_ts_json = OUT / "last_ts.json"
    with last_ts_json.open("w") as fp:
        json.dump(last_ts_by_station, fp, separators=(",", ":"))
    print(f"[done] wrote {daily_json.name} ({len(daily_by_date)} days) "
          f"+ {last_ts_json.name} ({len(last_ts_by_station)} stations)")

    if lc_disagreements:
        print()
        print("[warn] landcover-token mismatches between data filenames and metadata")
        print("       (matched by sensortype only; data filename takes precedence):")
        for msg in lc_disagreements:
            print(f"  - {msg}")
    if sensortype_disagreements:
        print()
        print("[warn] sensortype mismatches — station-level fallback used:")
        for msg in sensortype_disagreements:
            print(f"  - {msg}")


if __name__ == "__main__":
    main()
