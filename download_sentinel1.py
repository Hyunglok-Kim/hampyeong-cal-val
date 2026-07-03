"""
download_sentinel1.py — Sentinel-1 GRD backscatter (VV + VH, dB) for the
                       Hampyeong cal/val cell.

Sentinel-1 is C-band SAR, ~10 m native resolution, ~6-day revisit per
platform (S1A after the S1B loss + S1C launch: roughly 3–6 days combined
over Korea, mixing ascending & descending passes).

Products downloaded (each pass → one NC per polarization):
    - VV backscatter (co-pol)     ~ soil moisture, floods, urban
    - VH backscatter (cross-pol)  ~ vegetation biomass, volume scattering

EE `COPERNICUS/S1_GRD` serves both bands already in dB (10·log10 σ⁰).

Output (per-polarization subfolders, standard overlay NC schema):
    data/satellites/raw/S1_VV/S1_VV_YYYYMMDD_HHMM.nc
    data/satellites/raw/S1_VH/S1_VH_YYYYMMDD_HHMM.nc

Then:
    python3 nc_to_overlay.py data/satellites/

Setup (once):
    pip install --user earthengine-api requests netCDF4 numpy
    earthengine authenticate

Run:
    python3 download_sentinel1.py --project YOUR_PROJECT_ID
    python3 download_sentinel1.py --project YOUR_PROJECT_ID --start 2026-01-01 --end 2026-07-04
    python3 download_sentinel1.py --project YOUR_PROJECT_ID --vv-only
    python3 download_sentinel1.py --project YOUR_PROJECT_ID --vh-only
"""

import argparse
import datetime as dt
import io
import math
import os
import sys
from pathlib import Path

import numpy as np
import requests
from netCDF4 import Dataset

try:
    import ee
except ImportError:
    sys.exit("earthengine-api not installed.  Run:\n"
             "    pip install --user earthengine-api requests netCDF4 numpy")

# ----- Cell geometry (same as download_modis.py / download_hls.py) ----------
SITE_LAT = 35.015074737786415
SITE_LON = 126.55082987551783
BUFFER_KM = 2.0

half_lat = (BUFFER_KM / 2) / 110.574
half_lon = (BUFFER_KM / 2) / (111.320 * math.cos(math.radians(SITE_LAT)))
BBOX = [SITE_LON - half_lon, SITE_LAT - half_lat,
        SITE_LON + half_lon, SITE_LAT + half_lat]

# ----- Output dirs -----------------------------------------------------------
ROOT = Path(__file__).parent
SAT_RAW = ROOT / "data" / "satellites" / "raw"
VV_DIR  = SAT_RAW / "S1_VV"
VH_DIR  = SAT_RAW / "S1_VH"
VV_DIR.mkdir(parents=True, exist_ok=True)
VH_DIR.mkdir(parents=True, exist_ok=True)

# Physical range for post-download validity check + colormap defaults
VV_MIN, VV_MAX = -25.0, -5.0    # dB, typical C-band VV over land
VH_MIN, VH_MAX = -32.0, -12.0   # dB, typical C-band VH over land


def init_ee(project=None):
    project = project or os.environ.get("EARTHENGINE_PROJECT")
    def _try():
        ee.Initialize(project=project) if project else ee.Initialize()
    try:
        _try()
    except Exception:
        print("Earth Engine not initialized — running ee.Authenticate()…")
        ee.Authenticate()
        try:
            _try()
        except Exception as ex:
            sys.exit(
                f"Could not initialize Earth Engine: {ex}\n"
                f"Pass --project YOUR_PROJECT_ID or  export EARTHENGINE_PROJECT=…"
            )


def fetch_array(image, region, scale_m, band_name, label):
    """Download EE image as NPY → 2-D float32 array."""
    url = image.getDownloadURL({
        "region": region,
        "scale":  scale_m,
        "crs":    "EPSG:4326",
        "format": "NPY",
    })
    print(f"     download  {label}")
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    raw = np.load(io.BytesIO(r.content))
    if raw.dtype.names:
        return np.array(raw[band_name], dtype="float32")
    return np.array(raw, dtype="float32")


def write_nc(path, *, date, time_, source, product, resolution_m,
             var_name, values, vmin=None, vmax=None, description=""):
    ny, nx = values.shape
    n = BBOX[3]; s = BBOX[1]; e = BBOX[2]; w = BBOX[0]
    lats = np.linspace(n, s, ny, dtype="float64")
    lons = np.linspace(w, e, nx, dtype="float64")
    if path.exists():
        path.unlink()
    with Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("lat", ny)
        nc.createDimension("lon", nx)
        nc.createVariable("lat", "f8", ("lat",))[:] = lats
        nc.createVariable("lon", "f8", ("lon",))[:] = lons
        nc.variables["lat"].units = "degrees_north"
        nc.variables["lon"].units = "degrees_east"
        v = nc.createVariable(var_name, "f4", ("lat", "lon"), zlib=True, complevel=4)
        v[:] = values
        if vmin is not None: v.valid_min = np.float32(vmin)
        if vmax is not None: v.valid_max = np.float32(vmax)
        nc.title = f"{source.upper()} {product.upper()} {date} {time_}"
        nc.date = date
        nc.time = time_
        nc.source = source
        nc.product = product
        nc.resolution_m = float(resolution_m)
        nc.crs = "EPSG:4326"
        nc.description = description
        nc.created_utc = dt.datetime.utcnow().isoformat() + "Z"


# -----------------------------------------------------------------------
# Sentinel-1 GRD  —  VV + VH backscatter (dB)
# -----------------------------------------------------------------------
def s1_collection(start, end):
    """IW GRD scenes over the cell, both ascending & descending, VV + VH.

    We keep both orbit directions since S1's revisit is already sparse and
    the map UI already surfaces the scene datetime — the user can compare
    passes if they need to. Filter to IW (10 m) mode only.
    """
    region = ee.Geometry.Rectangle(BBOX)
    coll = (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("resolution_meters", 10))
            .sort("system:time_start"))
    return coll, region


def _download_pol(start, end, pol, out_dir, vmin, vmax, description):
    coll, region = s1_collection(start, end)
    n = coll.size().getInfo()
    print(f"=== Sentinel-1 {pol} (IW GRD, 10 m) {start} .. {end} ===")
    print(f"   {n} candidate scenes")
    images = coll.toList(n)

    saved, skipped = 0, 0
    for i in range(n):
        img = ee.Image(images.get(i))
        ts_ms = img.date().millis().getInfo()
        when = dt.datetime.utcfromtimestamp(ts_ms / 1000)
        date = when.strftime("%Y-%m-%d")
        time_ = when.strftime("%H:%M:%S")

        # EE serves S1_GRD in dB already. Select just the requested band,
        # cast to float, and rename to a stable name for fetch_array().
        band_img = img.select(pol).toFloat().rename(pol)

        try:
            arr = fetch_array(band_img, region, 10, pol,
                              f"S1 {pol} {date} {time_[:5]}")
        except Exception as ex:
            print(f"     ✗ {date} {time_}: {ex}"); continue

        # Drop obviously-bad pixels (fill values, extreme outliers). S1_GRD
        # can hit the EE fill sentinel outside the scene footprint; also
        # clamp to a wide physical envelope.
        arr[np.isnan(arr) | (np.abs(arr) > 1e30) | (arr < -50) | (arr > 5)] = np.nan
        valid_frac = float(np.mean(~np.isnan(arr)))
        if valid_frac < 0.1:
            skipped += 1
            print(f"     · {date} {time_} skipped ({valid_frac*100:.0f}% valid)")
            continue

        out = out_dir / f"S1_{pol}_{date.replace('-','')}_{time_[:5].replace(':','')}.nc"
        write_nc(out, date=date, time_=time_,
                 source="sentinel1",
                 product=f"s1_{pol.lower()}",
                 resolution_m=10,
                 var_name=pol.lower(),
                 values=arr,
                 vmin=vmin, vmax=vmax,
                 description=description)
        saved += 1
        print(f"     ✓ {out.name}  shape={arr.shape}  valid={valid_frac*100:.0f}%")
    print(f"   saved {saved}, skipped {skipped}")


def download_vv(start, end):
    _download_pol(start, end, "VV", VV_DIR,
                  VV_MIN, VV_MAX,
                  "Sentinel-1 GRD VV backscatter (dB)")


def download_vh(start, end):
    _download_pol(start, end, "VH", VH_DIR,
                  VH_MIN, VH_MAX,
                  "Sentinel-1 GRD VH backscatter (dB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01", help="ISO date (inclusive)")
    ap.add_argument("--end",   default="2027-01-01", help="ISO date (exclusive)")
    ap.add_argument("--project", default=None)
    ap.add_argument("--vv-only", action="store_true")
    ap.add_argument("--vh-only", action="store_true")
    args = ap.parse_args()

    init_ee(args.project)
    print(f"BBOX  west,south,east,north = {BBOX}")
    print(f"OUT   {SAT_RAW}  (per-polarization subfolders: S1_VV/, S1_VH/)")

    if not args.vh_only:
        download_vv(args.start, args.end)
    if not args.vv_only:
        download_vh(args.start, args.end)
    print("\nDone.  Convert to PNG + catalog with:\n    python3 nc_to_overlay.py data/satellites/")


if __name__ == "__main__":
    main()
