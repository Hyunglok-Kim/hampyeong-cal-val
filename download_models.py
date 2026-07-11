"""
download_models.py — Extract land-surface-model time series at the Hampyeong
site center from Google Earth Engine.

Models are 9–25 km resolution, so the whole 1-km cal/val cell sits inside a
single model pixel — a map overlay would be one flat rectangle. Instead we
pull the pixel value covering the site center as a daily time series, and the
Models tab plots those lines against the in-situ network mean.

Products (EE collection → data/models/<file>.csv):
    SMAP L4  NASA/SMAP/SPL4SMGP/008          9 km  3-hourly  ~2-4 day lag
             model-assimilated surface + root-zone SM — the very product
             this site exists to validate.
    ERA5-Land ECMWF/ERA5_LAND/DAILY_AGGR     9 km  daily     ~1 week lag
    GLDAS-2.2 NASA/GLDAS/V022/CLSM/G025/DA1D 25 km daily     months lag
             (kept for reference; refresh picks up whatever is new)

CSV schema (daily):
    date, sm_surface, sm_rootzone, soil_temp_c, precip_mm
Empty fields where a model doesn't provide the variable.

Run:
    /usr/bin/python3 download_models.py --project nodal-skein-411619
    /usr/bin/python3 download_models.py --project ... --start 2026-06-01
Existing CSVs are merged by date (new rows win), so incremental refreshes
are safe — same convention as the satellite downloaders.
"""

import argparse
import csv
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path

try:
    import ee
except ImportError:
    sys.exit("earthengine-api not installed.  pip install --user earthengine-api")

SITE_LAT = 35.015074737786415
SITE_LON = 126.55082987551783

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "data" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = ["date", "sm_surface", "sm_rootzone", "soil_temp_c", "precip_mm"]


def init_ee(project=None):
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception as ex:
        sys.exit(f"Could not initialize Earth Engine: {ex}")


def get_region_rows(coll_id, bands, start, end, scale_m):
    """getRegion → list of dicts {time_ms, <band>: value}. One EE call."""
    pt = ee.Geometry.Point([SITE_LON, SITE_LAT])
    coll = (ee.ImageCollection(coll_id)
            .filterDate(start, end)
            .filterBounds(pt)
            .select(bands))
    raw = coll.getRegion(pt, scale_m).getInfo()
    header, rows = raw[0], raw[1:]
    idx = {name: header.index(name) for name in ["time", *bands]}
    out = []
    for r in rows:
        rec = {"time_ms": r[idx["time"]]}
        for b in bands:
            rec[b] = r[idx[b]]
        out.append(rec)
    return out


def daily_mean(rows, band, transform=lambda v: v):
    """Group sub-daily rows into date → mean(transform(value))."""
    acc = defaultdict(lambda: [0.0, 0])
    for r in rows:
        v = r.get(band)
        if v is None:
            continue
        d = dt.datetime.utcfromtimestamp(r["time_ms"] / 1000).strftime("%Y-%m-%d")
        acc[d][0] += transform(v)
        acc[d][1] += 1
    return {d: s / n for d, (s, n) in acc.items() if n}


def merge_write(path, new_by_date):
    """Merge {date: {field: value}} into an existing CSV (new rows win)."""
    merged = {}
    if path.exists():
        with path.open() as f:
            for row in csv.DictReader(f):
                merged[row["date"]] = row
    for d, rec in new_by_date.items():
        merged[d] = {"date": d, **{k: rec.get(k, "") for k in FIELDS if k != "date"}}
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        w.writeheader()
        for d in sorted(merged):
            w.writerow(merged[d])
    return len(new_by_date), len(merged)


def fmt(v, nd=3):
    return "" if v is None else f"{round(v, nd):g}"


def pull_smap_l4(start, end):
    print("=== SMAP L4 (NASA/SMAP/SPL4SMGP/008, 9 km, 3-hourly) ===")
    bands = ["sm_surface", "sm_rootzone", "soil_temp_layer1",
             "precipitation_total_surface_flux"]
    # 3-hourly × a year exceeds EE's getRegion memory limit in one call —
    # pull in 60-day chunks and concatenate.
    rows = []
    d0 = dt.date.fromisoformat(start)
    d_end = dt.date.fromisoformat(end)
    while d0 < d_end:
        d1 = min(d0 + dt.timedelta(days=60), d_end)
        rows.extend(get_region_rows("NASA/SMAP/SPL4SMGP/008", bands,
                                    d0.isoformat(), d1.isoformat(), 11000))
        d0 = d1
    sm  = daily_mean(rows, "sm_surface")
    rz  = daily_mean(rows, "sm_rootzone")
    st  = daily_mean(rows, "soil_temp_layer1", lambda k: k - 273.15)
    # mean flux (kg m-2 s-1 == mm/s) × 86400 s → mm/day
    pr  = daily_mean(rows, "precipitation_total_surface_flux", lambda f: f * 86400)
    out = {d: {"sm_surface": fmt(sm.get(d)), "sm_rootzone": fmt(rz.get(d)),
               "soil_temp_c": fmt(st.get(d), 2), "precip_mm": fmt(pr.get(d), 2)}
           for d in sm}
    n_new, n_tot = merge_write(OUT_DIR / "SMAP_L4.csv", out)
    print(f"  {n_new} days pulled, {n_tot} total in CSV")


def pull_era5_land(start, end):
    print("=== ERA5-Land (ECMWF/ERA5_LAND/DAILY_AGGR, 9 km, daily) ===")
    bands = ["volumetric_soil_water_layer_1", "soil_temperature_level_1",
             "total_precipitation_sum"]
    rows = get_region_rows("ECMWF/ERA5_LAND/DAILY_AGGR", bands, start, end, 11132)
    sm = daily_mean(rows, "volumetric_soil_water_layer_1")        # m³/m³, 0-7 cm
    st = daily_mean(rows, "soil_temperature_level_1", lambda k: k - 273.15)
    pr = daily_mean(rows, "total_precipitation_sum", lambda m: m * 1000)  # m → mm
    out = {d: {"sm_surface": fmt(sm.get(d)), "sm_rootzone": "",
               "soil_temp_c": fmt(st.get(d), 2), "precip_mm": fmt(pr.get(d), 2)}
           for d in sm}
    n_new, n_tot = merge_write(OUT_DIR / "ERA5_LAND.csv", out)
    print(f"  {n_new} days pulled, {n_tot} total in CSV")


def pull_gldas22(start, end):
    print("=== GLDAS-2.2 (NASA/GLDAS/V022/CLSM/G025/DA1D, 25 km, daily) ===")
    bands = ["SoilMoist_S_tavg", "SoilMoist_RZ_tavg", "AvgSurfT_tavg"]
    rows = get_region_rows("NASA/GLDAS/V022/CLSM/G025/DA1D", bands, start, end, 27830)
    # CLSM surface-excess SM is kg/m² over the top 2 cm → ÷(1000·0.02)=÷20
    # for m³/m³; root zone is kg/m² over the top 1 m → ÷1000.
    sm = daily_mean(rows, "SoilMoist_S_tavg", lambda kg: kg / 20.0)
    rz = daily_mean(rows, "SoilMoist_RZ_tavg", lambda kg: kg / 1000.0)
    st = daily_mean(rows, "AvgSurfT_tavg", lambda k: k - 273.15)
    out = {d: {"sm_surface": fmt(sm.get(d)), "sm_rootzone": fmt(rz.get(d)),
               "soil_temp_c": fmt(st.get(d), 2), "precip_mm": ""}
           for d in sm}
    n_new, n_tot = merge_write(OUT_DIR / "GLDAS22.csv", out)
    print(f"  {n_new} days pulled, {n_tot} total in CSV")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-06-01", help="ISO date (inclusive)")
    ap.add_argument("--end", default=None, help="ISO date (exclusive); default = tomorrow")
    ap.add_argument("--project", default=None)
    args = ap.parse_args()
    end = args.end or (dt.date.today() + dt.timedelta(days=1)).isoformat()

    init_ee(args.project)
    print(f"site point: {SITE_LAT:.5f} N, {SITE_LON:.5f} E   window: {args.start} .. {end}")
    for fn in (pull_smap_l4, pull_era5_land, pull_gldas22):
        try:
            fn(args.start, end)
        except Exception as ex:
            print(f"  FAILED: {ex}")
    print("\nDone. CSVs in data/models/ — commit + push to publish.")


if __name__ == "__main__":
    main()
