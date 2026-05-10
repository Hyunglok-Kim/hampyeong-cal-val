"""
download_modis.py — Download MODIS NDVI + LST for the Hampyeong cal/val cell
                    and save as NCs the web map can render.

Products downloaded:
    - NDVI : MOD13A1.061  (Terra, 500 m, 16-day composite)  ~ 23 images / year
    - LST  : MOD11A2.061  (Terra, 1 km,  8-day composite)   ~ 46 images / year

Output:
    data/satellites/raw/MODIS_NDVI_YYYYMMDD.nc
    data/satellites/raw/MODIS_LST_YYYYMMDD.nc

Then run:
    python3 nc_to_overlay.py data/satellites/

============================================================
SETUP (one-time)
============================================================
    pip install --user earthengine-api requests netCDF4 numpy pillow
    earthengine authenticate           # opens browser, signs you in to Earth Engine
    earthengine set_project YOUR_GCP_PROJECT_ID    # if prompted (any GCP project works)

If you don't have an Earth Engine account yet:
    1. https://earthengine.google.com → Sign up (free for non-commercial use)
    2. Create or pick a Google Cloud project
    3. Run the authenticate step above

============================================================
RUN
============================================================
    python3 download_modis.py                 # full year 2025, both products
    python3 download_modis.py --ndvi-only     # NDVI only
    python3 download_modis.py --lst-only      # LST  only
    python3 download_modis.py --start 2025-06-01 --end 2025-09-01

If you only want a quick demo:
    python3 download_modis.py --start 2025-06-01 --end 2025-07-01
"""

import argparse
import datetime as dt
import io
import math
import sys
from pathlib import Path

import numpy as np
import requests
from netCDF4 import Dataset

try:
    import ee
except ImportError:
    sys.exit("earthengine-api not installed.  Run:\n"
             "    pip install --user earthengine-api requests netCDF4 numpy pillow")

# ----- Cell geometry (matches generate_sample_data.py / make_sample_nc.py) ----
SITE_LAT = 35.015074737786415
SITE_LON = 126.55082987551783
BUFFER_KM = 2.0   # download a slightly larger area so MODIS pixels have margin

half_lat = (BUFFER_KM / 2) / 110.574
half_lon = (BUFFER_KM / 2) / (111.320 * math.cos(math.radians(SITE_LAT)))
BBOX = [
    SITE_LON - half_lon, SITE_LAT - half_lat,    # west, south
    SITE_LON + half_lon, SITE_LAT + half_lat,    # east, north
]

ROOT = Path(__file__).parent
SAT_RAW = ROOT / "data" / "satellites" / "raw"
NDVI_DIR = SAT_RAW / "MODIS_NDVI"
LST_DIR  = SAT_RAW / "MODIS_LST"
NDVI_DIR.mkdir(parents=True, exist_ok=True)
LST_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------
# Earth Engine helpers
# -----------------------------------------------------------------------
def init_ee(project=None):
    """Initialize Earth Engine — prompt for auth if needed.

    `project` is your registered GCP project ID (required by EE since 2024).
    Falls back to the EARTHENGINE_PROJECT env var if --project is not given.
    """
    import os
    project = project or os.environ.get("EARTHENGINE_PROJECT")

    def _try_init():
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()

    try:
        _try_init()
    except Exception:
        print("Earth Engine not initialized — running ee.Authenticate()…")
        ee.Authenticate()
        try:
            _try_init()
        except Exception as ex:
            sys.exit(
                f"Could not initialize Earth Engine: {ex}\n\n"
                f"Pass your GCP project ID:\n"
                f"    python3 download_modis.py --project YOUR_PROJECT_ID\n"
                f"or set it once:\n"
                f"    export EARTHENGINE_PROJECT=YOUR_PROJECT_ID\n\n"
                f"If you don't have one yet:\n"
                f"  1. https://code.earthengine.google.com  (sign in, accept terms)\n"
                f"  2. Pick or create a Cloud project, register it for Earth Engine\n"
                f"     (free for noncommercial / academic use)\n"
                f"  3. The project ID you registered → use that here\n"
            )


def fetch_array(image, region, scale_m, band_name, label):
    """Ask Earth Engine for a small NPY (numpy) chunk and return a 2-D float32 array.

    NPY format is preferred over GEO_TIFF because it preserves dtype exactly —
    PIL's GeoTIFF reader silently mishandles float32 multi-band tiles, producing
    garbage values like 4e-35.  np.load on an EE NPY blob just works.
    """
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
    # EE NPY returns a structured array with one field per band
    if raw.dtype.names:
        arr = np.array(raw[band_name], dtype="float32")
    else:
        arr = np.array(raw, dtype="float32")
    return arr


def write_nc(path, *, date, time_, product, resolution_m, var_name, values,
             vmin=None, vmax=None, description=""):
    """Write to the standard overlay NC schema (matches make_sample_nc.py)."""
    ny, nx = values.shape
    n = BBOX[3]; s = BBOX[1]; e = BBOX[2]; w = BBOX[0]
    lats = np.linspace(n, s, ny, dtype="float64")     # north-to-south
    lons = np.linspace(w, e, nx, dtype="float64")
    if path.exists():
        path.unlink()
    with Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("lat", ny)
        nc.createDimension("lon", nx)
        v_lat = nc.createVariable("lat", "f8", ("lat",)); v_lat[:] = lats; v_lat.units = "degrees_north"
        v_lon = nc.createVariable("lon", "f8", ("lon",)); v_lon[:] = lons; v_lon.units = "degrees_east"
        v = nc.createVariable(var_name, "f4", ("lat", "lon"), zlib=True, complevel=4)
        v[:] = values
        if vmin is not None: v.valid_min = np.float32(vmin)
        if vmax is not None: v.valid_max = np.float32(vmax)
        nc.title = f"MODIS {product.upper()} {date} {time_}"
        nc.date = date
        nc.time = time_
        nc.product = product
        nc.resolution_m = float(resolution_m)
        nc.crs = "EPSG:4326"
        nc.description = description
        nc.source = "NASA MODIS via Google Earth Engine"
        nc.created_utc = dt.datetime.utcnow().isoformat() + "Z"


# -----------------------------------------------------------------------
# Downloaders
# -----------------------------------------------------------------------
def iter_collection(coll):
    """Yield (date_str, ee.Image) for each image in a collection."""
    n = coll.size().getInfo()
    images = coll.toList(n)
    for i in range(n):
        img = ee.Image(images.get(i))
        date_ms = img.date().millis().getInfo()
        date = dt.datetime.utcfromtimestamp(date_ms / 1000).strftime("%Y-%m-%d")
        yield date, img


def download_ndvi(start, end):
    print(f"=== MODIS NDVI (MOD13A1.061, 500 m, 16-day) {start} .. {end} ===")
    region = ee.Geometry.Rectangle(BBOX)
    coll = (ee.ImageCollection("MODIS/061/MOD13A1")
            .filterDate(start, end)
            .filterBounds(region)
            .select(["NDVI"]))
    for date, img in iter_collection(coll):
        # NDVI is stored as int16 * 10000 → scale to native [-1, 1]
        # NOTE: do NOT .clip(region) here — MOD13A1 sinusoidal can fail to
        # transform certain edges to EPSG:4326. Letting getDownloadURL handle
        # the region subset (with crs='EPSG:4326') reprojects safely.
        scaled = img.multiply(0.0001).toFloat().rename("NDVI")
        try:
            arr = fetch_array(scaled, region, 500, "NDVI", f"NDVI {date}")
        except Exception as ex:
            print(f"     ✗ {date}: {ex}"); continue
        # mask invalid (clouds / fill)
        arr[(arr < -0.2) | (arr > 1.0)] = np.nan
        out = NDVI_DIR / f"MODIS_NDVI_{date.replace('-','')}.nc"
        write_nc(out,
                 date=date, time_="00:00:00",
                 product="ndvi", resolution_m=500,
                 var_name="ndvi", values=arr,
                 vmin=0.0, vmax=0.9,
                 description="MODIS Terra MOD13A1 16-day NDVI composite")
        print(f"     ✓ {out.name}  shape={arr.shape}")


def download_lst(start, end):
    print(f"=== MODIS LST (MOD11A2.061, 1 km, 8-day day-time) {start} .. {end} ===")
    region = ee.Geometry.Rectangle(BBOX)
    coll = (ee.ImageCollection("MODIS/061/MOD11A2")
            .filterDate(start, end)
            .filterBounds(region)
            .select(["LST_Day_1km"]))
    for date, img in iter_collection(coll):
        # LST is Kelvin*0.02 → convert to Celsius
        # (drop .clip(region) for the same sinusoidal-edge reason as NDVI)
        scaled = img.multiply(0.02).subtract(273.15).toFloat().rename("LST")
        try:
            arr = fetch_array(scaled, region, 1000, "LST", f"LST {date}")
        except Exception as ex:
            print(f"     ✗ {date}: {ex}"); continue
        arr[(arr < -50) | (arr > 80)] = np.nan
        out = LST_DIR / f"MODIS_LST_{date.replace('-','')}.nc"
        write_nc(out,
                 date=date, time_="13:00:00",     # ~Terra overpass time over Korea
                 product="lst", resolution_m=1000,
                 var_name="lst", values=arr,
                 vmin=-10, vmax=45,
                 description="MODIS Terra MOD11A2 8-day Day LST (°C)")
        print(f"     ✓ {out.name}  shape={arr.shape}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-01", help="ISO date (inclusive)")
    ap.add_argument("--end",   default="2026-01-01", help="ISO date (exclusive)")
    ap.add_argument("--project", default=None,
                    help="Your registered GCP project ID for Earth Engine "
                         "(or set EARTHENGINE_PROJECT env var)")
    ap.add_argument("--ndvi-only", action="store_true")
    ap.add_argument("--lst-only",  action="store_true")
    args = ap.parse_args()

    init_ee(project=args.project)
    print(f"BBOX  west,south,east,north = {BBOX}")
    print(f"OUT   {SAT_RAW}  (per-product subfolders: MODIS_NDVI/, MODIS_LST/)")

    if not args.lst_only:
        download_ndvi(args.start, args.end)
    if not args.ndvi_only:
        download_lst(args.start, args.end)

    print("\nDone.  Convert to PNG + catalog with:")
    print("    python3 nc_to_overlay.py data/satellites/")


if __name__ == "__main__":
    main()
