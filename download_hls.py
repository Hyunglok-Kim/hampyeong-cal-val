"""
download_hls.py — Download Harmonized Landsat-Sentinel (HLS) NDVI and
                  Landsat 8/9 LST for the Hampyeong cal/val cell.

Both products are 30 m, much sharper than MODIS (500 m / 1 km) for our 1-km cell.

Products downloaded:
    - HLS NDVI : NASA/HLS/HLSL30 + NASA/HLS/HLSS30  combined  (~2-3 day revisit)
    - Landsat LST: LC08 + LC09  ST_B10 thermal  (~8-day combined revisit)

Output (per-source subfolders, same NC schema as download_modis.py):
    data/satellites/raw/HLS_NDVI/HLS_NDVI_YYYYMMDD_HHMM.nc
    data/satellites/raw/LANDSAT_LST/LANDSAT_LST_YYYYMMDD_HHMM.nc

Then:
    python3 nc_to_overlay.py data/satellites/

Setup (once, same as download_modis.py):
    pip install --user earthengine-api requests netCDF4 numpy
    earthengine authenticate
    earthengine set_project YOUR_GCP_PROJECT_ID

Run:
    python3 download_hls.py --project YOUR_PROJECT_ID
    python3 download_hls.py --project YOUR_PROJECT_ID --start 2025-06-01 --end 2025-07-01
    python3 download_hls.py --project YOUR_PROJECT_ID --ndvi-only
    python3 download_hls.py --project YOUR_PROJECT_ID --lst-only
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

# ----- Cell geometry (same as download_modis.py) -----------------------------
SITE_LAT = 35.015074737786415
SITE_LON = 126.55082987551783
BUFFER_KM = 2.0

half_lat = (BUFFER_KM / 2) / 110.574
half_lon = (BUFFER_KM / 2) / (111.320 * math.cos(math.radians(SITE_LAT)))
BBOX = [SITE_LON - half_lon, SITE_LAT - half_lat,
        SITE_LON + half_lon, SITE_LAT + half_lat]

# ----- Output dirs -----------------------------------------------------------
ROOT = Path(__file__).parent
SAT_RAW   = ROOT / "data" / "satellites" / "raw"
HLS_DIR   = SAT_RAW / "HLS_NDVI"
LST_DIR   = SAT_RAW / "LANDSAT_LST"
HLS_DIR.mkdir(parents=True, exist_ok=True)
LST_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------
# EE helpers (mirror download_modis.py)
# -----------------------------------------------------------------------
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
    """Write NC with the standard overlay schema + `source` global attr."""
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
# HLS NDVI  (Landsat L30 + Sentinel-2 S30, both at 30 m, harmonized)
# -----------------------------------------------------------------------
def _hls_ndvi(image, nir_band, red_band="B04"):
    """Compute NDVI on a single HLS image, masking cloud/cloud-shadow via Fmask.

    HLS Fmask bits:
      0 cirrus | 1 cloud | 2 adj cloud | 3 cloud shadow | 4 snow/ice | 5 water
    We mask out bits 1 + 2 + 3.  Reflectance bands are int16 * 0.0001.
    """
    fmask = image.select("Fmask")
    bad = fmask.bitwiseAnd((1 << 1) | (1 << 2) | (1 << 3)).neq(0)
    nir = image.select(nir_band).multiply(0.0001)
    red = image.select(red_band).multiply(0.0001)
    ndvi = (nir.subtract(red).divide(nir.add(red))
              .rename("NDVI")
              .updateMask(bad.Not()))
    return ndvi.copyProperties(image, ["system:time_start"])


def download_hls_ndvi(start, end):
    print(f"=== HLS NDVI (Landsat L30 + Sentinel-2 S30, 30 m) {start} .. {end} ===")
    region = ee.Geometry.Rectangle(BBOX)

    # HLS bands: both products use single-digit names (B1..B11/B12) plus B8A.
    # Red = B4 in both. NIR = B5 in HLS-L30 (Landsat), B8A in HLS-S30 (S2-narrow).
    l30 = (ee.ImageCollection("NASA/HLS/HLSL30/v002")
           .filterDate(start, end).filterBounds(region)
           .select(["B4", "B5", "Fmask"])
           .map(lambda i: _hls_ndvi(i, nir_band="B5", red_band="B4")))
    s30 = (ee.ImageCollection("NASA/HLS/HLSS30/v002")
           .filterDate(start, end).filterBounds(region)
           .select(["B4", "B8A", "Fmask"])
           .map(lambda i: _hls_ndvi(i, nir_band="B8A", red_band="B4")))
    coll = l30.merge(s30).sort("system:time_start")
    n = coll.size().getInfo()
    print(f"   {n} candidate scenes (after merge, before per-scene cloud filter)")
    images = coll.toList(n)

    saved, skipped = 0, 0
    for i in range(n):
        img = ee.Image(images.get(i))
        ts_ms = img.date().millis().getInfo()
        when = dt.datetime.utcfromtimestamp(ts_ms / 1000)
        date = when.strftime("%Y-%m-%d")
        time_ = when.strftime("%H:%M:%S")
        try:
            arr = fetch_array(img.toFloat().rename("NDVI"), region, 30,
                              "NDVI", f"HLS NDVI {date} {time_[:5]}")
        except Exception as ex:
            print(f"     ✗ {date} {time_}: {ex}"); continue
        # mask out-of-physical-range FIRST (EE may use very-large fill for
        # masked cloud pixels), then compute valid fraction — this prevents
        # cloudy scenes from passing the threshold on garbage values.
        arr[(arr < -0.2) | (arr > 1.0) | (np.abs(arr) > 1e30)] = np.nan
        valid_frac = float(np.mean(~np.isnan(arr)))
        if valid_frac < 0.1:
            skipped += 1
            print(f"     · {date} {time_} skipped ({valid_frac*100:.0f}% valid)")
            continue
        out = HLS_DIR / f"HLS_NDVI_{date.replace('-','')}_{time_[:5].replace(':','')}.nc"
        write_nc(out, date=date, time_=time_,
                 source="hls", product="ndvi", resolution_m=30,
                 var_name="ndvi", values=arr,
                 vmin=0.0, vmax=0.9,
                 description="HLS L30+S30 30 m NDVI (Fmask-masked)")
        saved += 1
        print(f"     ✓ {out.name}  shape={arr.shape}  valid={valid_frac*100:.0f}%")
    print(f"   saved {saved}, skipped {skipped}")


# -----------------------------------------------------------------------
# Landsat 8/9 LST  (ST_B10 thermal, 30 m resampled from 100 m TIRS)
# -----------------------------------------------------------------------
def _landsat_lst(image):
    """Convert Landsat C2 L2 ST_B10 to °C with cloud mask from QA_PIXEL.

    QA_PIXEL bits we mask:
      1 dilated cloud | 3 cloud | 4 cloud shadow
    ST_B10 scale: 0.00341802, offset: +149.0  → Kelvin
    """
    qa = image.select("QA_PIXEL")
    bad = (qa.bitwiseAnd(1 << 1).neq(0)
        .Or(qa.bitwiseAnd(1 << 3).neq(0))
        .Or(qa.bitwiseAnd(1 << 4).neq(0)))
    lst = (image.select("ST_B10")
                 .multiply(0.00341802).add(149.0)
                 .subtract(273.15)
                 .rename("LST")
                 .updateMask(bad.Not()))
    return lst.copyProperties(image, ["system:time_start"])


def download_landsat_lst(start, end):
    print(f"=== Landsat 8+9 LST (ST_B10, 30 m) {start} .. {end} ===")
    region = ee.Geometry.Rectangle(BBOX)
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterDate(start, end).filterBounds(region)
          .select(["ST_B10", "QA_PIXEL"])
          .map(_landsat_lst))
    l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
          .filterDate(start, end).filterBounds(region)
          .select(["ST_B10", "QA_PIXEL"])
          .map(_landsat_lst))
    coll = l8.merge(l9).sort("system:time_start")
    n = coll.size().getInfo()
    print(f"   {n} candidate scenes")
    images = coll.toList(n)

    saved, skipped = 0, 0
    for i in range(n):
        img = ee.Image(images.get(i))
        ts_ms = img.date().millis().getInfo()
        when = dt.datetime.utcfromtimestamp(ts_ms / 1000)
        date = when.strftime("%Y-%m-%d")
        time_ = when.strftime("%H:%M:%S")
        try:
            arr = fetch_array(img.toFloat().rename("LST"), region, 30,
                              "LST", f"Landsat LST {date} {time_[:5]}")
        except Exception as ex:
            print(f"     ✗ {date} {time_}: {ex}"); continue
        arr[(arr < -50) | (arr > 80) | (np.abs(arr) > 1e30)] = np.nan
        valid_frac = float(np.mean(~np.isnan(arr)))
        if valid_frac < 0.1:
            skipped += 1
            print(f"     · {date} {time_} skipped ({valid_frac*100:.0f}% valid)")
            continue
        out = LST_DIR / f"LANDSAT_LST_{date.replace('-','')}_{time_[:5].replace(':','')}.nc"
        write_nc(out, date=date, time_=time_,
                 source="landsat", product="lst", resolution_m=30,
                 var_name="lst", values=arr,
                 vmin=-10, vmax=45,
                 description="Landsat 8/9 C2 L2 ST_B10 LST (°C, QA-masked)")
        saved += 1
        print(f"     ✓ {out.name}  shape={arr.shape}  valid={valid_frac*100:.0f}%")
    print(f"   saved {saved}, skipped {skipped}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end",   default="2026-01-01")
    ap.add_argument("--project", default=None,
                    help="Registered GCP project ID for Earth Engine")
    ap.add_argument("--ndvi-only", action="store_true")
    ap.add_argument("--lst-only",  action="store_true")
    args = ap.parse_args()

    init_ee(project=args.project)
    print(f"BBOX  west,south,east,north = {BBOX}")
    print(f"OUT   HLS NDVI    → {HLS_DIR}")
    print(f"OUT   Landsat LST → {LST_DIR}")

    if not args.lst_only:
        download_hls_ndvi(args.start, args.end)
    if not args.ndvi_only:
        download_landsat_lst(args.start, args.end)

    print("\nDone.  Convert to PNG + catalog with:")
    print("    python3 nc_to_overlay.py data/satellites/")


if __name__ == "__main__":
    main()
