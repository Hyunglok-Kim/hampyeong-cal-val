"""
prep_polra.py — convert PolRA L-band radiometer NCs into the overlay schema
                used by nc_to_overlay.py.

Inputs (drop into the website root or this folder):
    HP_polra_SM_Retrieved_<YYYYMMDD>.nc      # 64×64, ~25 m, SM_Retrieved/TBv/TBh/Tir
    HP_polra_SM_LGB_DS_<YYYYMMDD or YYMMDD>.nc  # ~1 m, SM_LGB_downscaled

Both files have 2-D curvilinear lat(y,x), lon(y,x). They're close enough to
regular within the 1-km cal/val cell that we collapse them by averaging
lat across columns (per row) and lon across rows (per column). The maximum
sub-pixel error from this approximation is <~10 m at 1 m resolution, well
under what users notice on the In-situ leaflet map.

Outputs into data/uav/raw/<YYYYMMDD>_polra/:
    HP_UAV_SM_RETR_<YYYYMMDD>.nc   product=sm_retr   25 m
    HP_UAV_TBV_<YYYYMMDD>.nc       product=tb_v      25 m
    HP_UAV_TBH_<YYYYMMDD>.nc       product=tb_h      25 m
    HP_UAV_SM_LGB_<YYYYMMDD>.nc    product=sm_lgb     1 m

PolRA Tir is skipped — UAV LWIR already occupies product='tir' for the same
date and the two would clash on PNG filename / catalog.

Run:
    python3 prep_polra.py                  # process all polra NCs found
    python3 prep_polra.py /path/to/file.nc # one specific file
"""

import re
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).parent
WEBSITE_ROOT = ROOT.parent              # website/ — where lab drops files
RAW_ROOT = ROOT / "data" / "uav" / "raw"

# Filename → flight date.
RETR_RE = re.compile(r"HP_polra_SM_Retrieved_(\d{8})\.nc$", re.I)
LGB_RE  = re.compile(r"HP_polra_SM_LGB[ _]?DS[ _](\d{6,8})\.nc$", re.I)

DEFAULT_TIME = "11:00:00"


def collapse_2d_to_1d(lat2d, lon2d):
    """Reduce curvilinear (y,x) lat/lon arrays to 1-D regular lat(y), lon(x).
    Uses per-row mean for lat (varies primarily with y) and per-col mean for
    lon (varies primarily with x). Sub-pixel error is acceptable for our
    1-km cell."""
    lat1d = np.nanmean(lat2d, axis=1).astype(np.float64)
    lon1d = np.nanmean(lon2d, axis=0).astype(np.float64)
    return lat1d, lon1d


def parse_date_token(tok):
    """Accept either YYYYMMDD or YYMMDD. Returns 'YYYYMMDD'."""
    if len(tok) == 8: return tok
    if len(tok) == 6: return "20" + tok
    raise ValueError(f"unrecognised date token: {tok}")


def write_nc(out_path, *, data, lats, lons, product, date_str, resolution_m,
             time_str=DEFAULT_TIME, source="lband", description="",
             units=None, valid_min=None, valid_max=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    with Dataset(out_path, "w") as nc:
        date_iso = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        nc.date = date_iso
        nc.time = time_str
        nc.product = product
        nc.resolution_m = float(resolution_m)
        nc.source = source
        if description:
            nc.description = description
        nc.crs = "EPSG:4326"
        nc.createDimension("lat", len(lats))
        nc.createDimension("lon", len(lons))
        v_lat = nc.createVariable("lat", "f8", ("lat",)); v_lat.units = "degrees_north"
        v_lon = nc.createVariable("lon", "f8", ("lon",)); v_lon.units = "degrees_east"
        v_lat[:] = lats
        v_lon[:] = lons
        v = nc.createVariable(product, "f4", ("lat", "lon"),
                              fill_value=np.float32(np.nan))
        v[:] = data.astype(np.float32)
        if units:     v.units = units
        if valid_min is not None: v.valid_min = float(valid_min)
        if valid_max is not None: v.valid_max = float(valid_max)


def find_polra_files():
    """Search common locations for PolRA NC files."""
    seen = set()
    out = []
    for base in (WEBSITE_ROOT, ROOT):
        for p in sorted(base.glob("HP_polra*.nc")):
            if p.resolve() in seen: continue
            seen.add(p.resolve())
            out.append(p)
    return out


def process_retrieved(src_path, date_str):
    """64×64 file with SM_Retrieved + TBv + TBh + Tir. Write 3 NCs (skip Tir)."""
    out_dir = RAW_ROOT / f"{date_str}_polra"
    with Dataset(src_path, "r") as nc:
        lat2d = np.array(nc.variables["lat"][:])
        lon2d = np.array(nc.variables["lon"][:])
        sm  = np.array(nc.variables["SM_Retrieved"][:], dtype=np.float32)
        tbv = np.array(nc.variables["TBv"][:], dtype=np.float32)
        tbh = np.array(nc.variables["TBh"][:], dtype=np.float32)
    lats, lons = collapse_2d_to_1d(lat2d, lon2d)
    # ensure north→south for our convention (nc_to_overlay flips if needed)
    write_nc(out_dir / f"HP_UAV_SM_RETR_{date_str}.nc",
             data=sm, lats=lats, lons=lons,
             product="sm_retr", date_str=date_str, resolution_m=25,
             description="L-band radiometer retrieved soil moisture (PolRA)",
             units="m^3 m^-3", valid_min=0.0, valid_max=0.6)
    write_nc(out_dir / f"HP_UAV_TBV_{date_str}.nc",
             data=tbv, lats=lats, lons=lons,
             product="tb_v", date_str=date_str, resolution_m=25,
             description="L-band brightness temperature, V-pol (PolRA)",
             units="K", valid_min=200.0, valid_max=290.0)
    write_nc(out_dir / f"HP_UAV_TBH_{date_str}.nc",
             data=tbh, lats=lats, lons=lons,
             product="tb_h", date_str=date_str, resolution_m=25,
             description="L-band brightness temperature, H-pol (PolRA)",
             units="K", valid_min=200.0, valid_max=290.0)
    print(f"  OK Retrieved → 3 NCs in {out_dir.relative_to(ROOT)}")


def process_lgb(src_path, date_str):
    out_dir = RAW_ROOT / f"{date_str}_polra"
    with Dataset(src_path, "r") as nc:
        lat2d = np.array(nc.variables["lat"][:])
        lon2d = np.array(nc.variables["lon"][:])
        sm = np.array(nc.variables["SM_LGB_downscaled"][:], dtype=np.float32)
    lats, lons = collapse_2d_to_1d(lat2d, lon2d)
    write_nc(out_dir / f"HP_UAV_SM_LGB_{date_str}.nc",
             data=sm, lats=lats, lons=lons,
             product="sm_lgb", date_str=date_str, resolution_m=1,
             description="LGB-downscaled soil moisture from L-band (PolRA)",
             units="m^3 m^-3", valid_min=0.0, valid_max=0.6)
    print(f"  OK LGB        → HP_UAV_SM_LGB_{date_str}.nc in {out_dir.relative_to(ROOT)}")


def main():
    targets = sys.argv[1:] or [str(p) for p in find_polra_files()]
    if not targets:
        sys.exit("no HP_polra*.nc files found.")
    for t in targets:
        p = Path(t).resolve()
        if not p.exists():
            print(f"  MISS {p}")
            continue
        m = RETR_RE.search(p.name)
        if m:
            date_str = parse_date_token(m.group(1))
            print(f"=== {p.name}  (Retrieved, {date_str}) ===")
            process_retrieved(p, date_str)
            continue
        m = LGB_RE.search(p.name)
        if m:
            date_str = parse_date_token(m.group(1))
            print(f"=== {p.name}  (LGB DS, {date_str}) ===")
            process_lgb(p, date_str)
            continue
        print(f"  SKIP {p.name} (filename does not match polra patterns)")


if __name__ == "__main__":
    main()
