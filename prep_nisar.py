"""
prep_nisar.py — convert lab-format NISAR SME2 NCs into the overlay schema.

The lab supplies one .nc per NISAR overpass with this schema:
    NISAR_beta_HP_YYYYMMDD_HHMMSS.nc
    - 2-D dims (y, x)
    - lat(y, x), lon(y, x)   float32, 2-D coords (regular grid in lat/lon)
    - HH, HV, LIA           float32 backscatter / incidence
    - "NISAR (Beta)"        float32  soil moisture in m³/m³  ← we want this
    - "GIST (Beta)"         float32  alternative SM estimate
    - LC                    int      land cover class
    - time                  scalar   (info only)
    - global attrs include  zero_doppler_start_time_utc, sensor, etc.

This script:
    1. Scans for lab NCs (`NISAR_beta_HP_*.nc`) in two locations:
         a) data/satellites/raw/NISAR_SM/_lab/   (preferred, in-project)
         b) the parent website/ folder           (fallback for newly dropped)
    2. For each lab NC, produces an overlay-ready NC at:
         data/satellites/raw/NISAR_SM/NISAR_SM_YYYYMMDD_HHMM.nc
       with 1-D lat/lon, single 'sm' variable, and our standard global attrs.
    3. Original lab files are also moved into _lab/ as a backup.

Run:
    python3 prep_nisar.py
Then:
    python3 nc_to_overlay.py data/satellites/
"""

import datetime as dt
import re
import shutil
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).parent
NISAR_DIR = ROOT / "data" / "satellites" / "raw" / "NISAR_SM"
LAB_DIR   = NISAR_DIR / "_lab"          # where we keep the original lab files
NISAR_DIR.mkdir(parents=True, exist_ok=True)
LAB_DIR.mkdir(parents=True, exist_ok=True)

# fall back to scanning the project parent for newly dropped lab files
WEBSITE_ROOT = ROOT.parent

NAME_RE = re.compile(r"NISAR_beta_HP_(\d{8})_(\d{6})\.nc$")


def collect_lab_files():
    """Return all lab NISAR NCs across known locations."""
    files = list(LAB_DIR.glob("NISAR_beta_HP_*.nc"))
    files += list(WEBSITE_ROOT.glob("NISAR_beta_HP_*.nc"))
    # de-dupe by basename
    seen = {}
    for f in files:
        seen[f.name] = f
    return sorted(seen.values(), key=lambda p: p.name)


# Variables to extract from each lab NC.
# (lab_var_name, product_key, var_name_in_nc, units, valid_min, valid_max, description)
EXTRACTS = [
    ("NISAR (Beta)", "sm",      "soil_moisture", "m^3/m^3",  0.05,  0.55,
        "NISAR LSAR SME2 soil moisture beta (m³/m³)"),
    ("GIST (Beta)",  "gist_sm", "soil_moisture", "m^3/m^3",  0.05,  0.55,
        "GIST soil moisture beta product (m³/m³)"),
    ("HH",           "hh",      "sigma0_hh",     "linear",   0.00,  0.50,
        "NISAR sigma0 backscatter HH polarization (linear)"),
    ("HV",           "hv",      "sigma0_hv",     "linear",   0.00,  0.30,
        "NISAR sigma0 backscatter HV polarization (linear)"),
    ("LIA",          "lia",     "incidence_angle", "degrees", 20.0, 60.0,
        "Local Incidence Angle (degrees)"),
]


def lab_to_overlay(lab_path):
    """Read one lab NC and write one overlay NC per variable in EXTRACTS.

    Returns list of (overlay_filename, mean_value, valid_pixel_count) per
    variable found.  Lab files lacking a variable just skip that variable.
    """
    m = NAME_RE.search(lab_path.name)
    if not m:
        print(f"  SKIP {lab_path.name} (filename pattern not matched)")
        return []
    yyyymmdd, hhmmss = m.group(1), m.group(2)
    date = f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
    time_ = f"{hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"

    written = []
    with Dataset(lab_path, "r") as src:
        if "lat" not in src.variables or "lon" not in src.variables:
            print(f"  SKIP {lab_path.name} (no lat/lon)"); return []
        lat2d = np.array(src.variables["lat"][:], dtype="float64")
        lon2d = np.array(src.variables["lon"][:], dtype="float64")
        # collapse 2-D regular grid to 1-D coords
        lats = lat2d[:, 0]
        lons = lon2d[0, :]
        flip_rows = lats[0] < lats[-1]
        if flip_rows:
            lats = lats[::-1]

        for lab_var, prod_key, nc_var, units, vmin, vmax, descr in EXTRACTS:
            if lab_var not in src.variables:
                continue
            arr = np.array(src.variables[lab_var][:], dtype="float32")
            if flip_rows:
                arr = arr[::-1, :]
            ny, nx = arr.shape

            out_name = f"NISAR_{prod_key.upper()}_{yyyymmdd}_{hhmmss[:4]}.nc"
            out_path = NISAR_DIR / out_name
            if out_path.exists():
                out_path.unlink()
            with Dataset(out_path, "w", format="NETCDF4") as dst:
                dst.createDimension("lat", ny)
                dst.createDimension("lon", nx)
                v_lat = dst.createVariable("lat", "f8", ("lat",))
                v_lat.units = "degrees_north"; v_lat[:] = lats
                v_lon = dst.createVariable("lon", "f8", ("lon",))
                v_lon.units = "degrees_east"; v_lon[:] = lons
                v = dst.createVariable(nc_var, "f4", ("lat", "lon"),
                                       zlib=True, complevel=4)
                v[:] = arr
                v.valid_min = np.float32(vmin)
                v.valid_max = np.float32(vmax)
                v.units = units
                dst.title = f"NISAR LSAR {prod_key.upper()} {date} {time_}"
                dst.date = date
                dst.time = time_
                dst.source = "nisar"
                dst.product = prod_key
                dst.resolution_m = 200.0
                dst.crs = "EPSG:4326"
                dst.description = descr
                dst.created_utc = dt.datetime.utcnow().isoformat() + "Z"
            mean_v = float(np.nanmean(arr))
            valid = int((~np.isnan(arr)).sum())
            written.append((out_name, mean_v, valid))

    # archive lab file in _lab/ so prep is idempotent
    if lab_path.parent != LAB_DIR:
        target = LAB_DIR / lab_path.name
        if not target.exists():
            shutil.move(str(lab_path), str(target))
        else:
            lab_path.unlink()
    return written


def main():
    files = collect_lab_files()
    if not files:
        print(f"no NISAR lab files found in {LAB_DIR} or {WEBSITE_ROOT}")
        return
    print(f"found {len(files)} lab NCs")
    total = 0
    for f in files:
        results = lab_to_overlay(f)
        if not results:
            continue
        for out_name, mean_v, valid in results:
            total += 1
            print(f"  ✓ {f.name}  →  {out_name}   "
                  f"mean={mean_v:.3f}  valid_pixels={valid}")
    print(f"saved {total} overlay NCs in {NISAR_DIR}")


if __name__ == "__main__":
    main()
