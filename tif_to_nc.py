"""
tif_to_nc.py - convert UAV GeoTIFFs into the NC overlay schema used by
                nc_to_overlay.py.

Input layout (one folder per flight date, YYYYMMDD):
    data/uav/raw/<YYYYMMDD>/HP_<YYYYMMDD>_<EQUIP>_<VAR>_<RES>m.tif

  EQUIP: LiDAR | MS
  VAR:   DSM (LiDAR)
         NDVI | NDWI | blue | green | red | nir | lwir   (MS)

Output (drop-in for nc_to_overlay.py):
    data/uav/raw/<YYYYMMDD>/HP_UAV_<PRODUCT>_<YYYYMMDD>.nc

Products written per date:
    lidar  ← LiDAR_DSM
    ndvi   ← MS_NDVI
    ndwi   ← MS_NDWI                 (new product key)
    tir    ← MS_lwir   (DN→°C: DN/100 − 273.15, i.e. centi-Kelvin)
    rgb    ← stack(MS_red, MS_green, MS_blue) with per-band 2–98 % stretch

Reprojects each TIF from its source CRS (these were UTM 52N / EPSG:32652) to
WGS84 / EPSG:4326 with bilinear resampling.  Lat/lon written as CELL CENTERS
per CF convention so nc_to_overlay's half-pixel padding lines up.

Run:
    python3 tif_to_nc.py                       # all dates under data/uav/raw/
    python3 tif_to_nc.py 20260430              # one date
"""

import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)
from netCDF4 import Dataset

ROOT = Path(__file__).parent
RAW_ROOT = ROOT / "data" / "uav" / "raw"

DST_CRS = "EPSG:4326"
DEFAULT_TIME = "11:00:00"  # UAV flights have no per-file time tag; use noon-ish

# LWIR raw → °C.  These FLIR-style files store centi-Kelvin (DN/100 = K).
LWIR_DN_PER_K = 100.0
KELVIN_OFFSET = 273.15

FILE_RE = re.compile(
    r"^HP_(\d{8})_(LiDAR|MS)_([A-Za-z]+)_(\d+)m\.tif$"
)


def warp_to_wgs84(src_path):
    """Reproject a single-band GeoTIFF to EPSG:4326. Returns (data, lats, lons,
    nodata) where lat/lon are CELL CENTERS, north→south for lat (so row 0 is
    the top of the image)."""
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds
        )
        dst = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=DST_CRS,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
        nodata = src.nodata

    # cell-center lat/lon arrays
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    # affine: x = a*col + b*row + c (so column → lon when b==0); y = d*col + e*row + f
    cols = np.arange(width)
    rows = np.arange(height)
    lons = c + a * (cols + 0.5) + b * 0.5
    lats = f + e * (rows + 0.5) + d * 0.5  # e is negative → lats decrease north→south
    return dst, lats, lons, nodata


def write_nc(out_path, *, data, lats, lons, product, date_str, resolution_m,
             time_str=DEFAULT_TIME, source="uav", description="", valid_min=None,
             valid_max=None, units=None, rgb=False):
    """Write a single-product overlay NC matching nc_to_overlay.py's schema."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    with Dataset(out_path, "w") as nc:
        # Required globals
        date_iso = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        nc.date = date_iso
        nc.time = time_str
        nc.product = product
        nc.resolution_m = float(resolution_m)
        # Optional but useful
        nc.source = source
        if description:
            nc.description = description
        nc.crs = "EPSG:4326"

        nc.createDimension("lat", len(lats))
        nc.createDimension("lon", len(lons))
        v_lat = nc.createVariable("lat", "f8", ("lat",))
        v_lon = nc.createVariable("lon", "f8", ("lon",))
        v_lat.units = "degrees_north"
        v_lon.units = "degrees_east"
        v_lat[:] = lats
        v_lon[:] = lons

        if rgb:
            # data shape (3, ny, nx), values in [0, 1]
            nc.createDimension("band", 3)
            v = nc.createVariable("rgb", "f4", ("band", "lat", "lon"),
                                  fill_value=np.float32(np.nan))
            v[:] = data.astype(np.float32)
        else:
            v = nc.createVariable(product, "f4", ("lat", "lon"),
                                  fill_value=np.float32(np.nan))
            v[:] = data.astype(np.float32)
            if valid_min is not None:
                v.valid_min = float(valid_min)
            if valid_max is not None:
                v.valid_max = float(valid_max)
            if units:
                v.units = units


def stretch_band(arr, lo_pct=2.0, hi_pct=98.0):
    """Per-band percentile stretch to [0, 1]; NaN-safe."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = np.percentile(finite, lo_pct)
    hi = np.percentile(finite, hi_pct)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def find_tif(folder, equip, var):
    """Find HP_*_<equip>_<var>_*m.tif (case-insensitive on equip+var)."""
    e_l, v_l = equip.lower(), var.lower()
    for p in folder.iterdir():
        if not p.suffix.lower() == ".tif":
            continue
        m = FILE_RE.match(p.name)
        if not m:
            continue
        _, e, v, _ = m.groups()
        if e.lower() == e_l and v.lower() == v_l:
            return p
    return None


def parse_resolution(path):
    m = FILE_RE.match(path.name)
    return float(m.group(4)) if m else 1.0


def process_date(date_dir):
    """Convert all TIFs under data/uav/raw/<DATE>/ into the 5 product NCs."""
    if not date_dir.is_dir():
        return
    date_str = date_dir.name
    if not re.fullmatch(r"\d{8}", date_str):
        print(f"  SKIP {date_dir} (folder name not YYYYMMDD)")
        return

    print(f"=== {date_str} ===")
    out_dir = date_dir  # put NCs alongside the TIFs

    # ---- LiDAR DSM → product 'lidar' ----
    p = find_tif(date_dir, "LiDAR", "DSM")
    if p:
        data, lats, lons, _ = warp_to_wgs84(p)
        res = parse_resolution(p)
        write_nc(out_dir / f"HP_UAV_LIDAR_{date_str}.nc",
                 data=data, lats=lats, lons=lons,
                 product="lidar", date_str=date_str, resolution_m=res,
                 description="UAV LiDAR DSM (m)", units="m")
        print(f"  OK lidar  ← {p.name}")

    # ---- MS NDVI ----
    p = find_tif(date_dir, "MS", "NDVI")
    if p:
        data, lats, lons, _ = warp_to_wgs84(p)
        res = parse_resolution(p)
        write_nc(out_dir / f"HP_UAV_NDVI_{date_str}.nc",
                 data=data, lats=lats, lons=lons,
                 product="ndvi", date_str=date_str, resolution_m=res,
                 description="UAV multispectral NDVI", units="")
        print(f"  OK ndvi   ← {p.name}")

    # ---- MS NDWI ----
    p = find_tif(date_dir, "MS", "NDWI")
    if p:
        data, lats, lons, _ = warp_to_wgs84(p)
        res = parse_resolution(p)
        write_nc(out_dir / f"HP_UAV_NDWI_{date_str}.nc",
                 data=data, lats=lats, lons=lons,
                 product="ndwi", date_str=date_str, resolution_m=res,
                 description="UAV multispectral NDWI", units="")
        print(f"  OK ndwi   ← {p.name}")

    # ---- MS LWIR → product 'tir' (°C) ----
    p = find_tif(date_dir, "MS", "lwir")
    if p:
        data, lats, lons, _ = warp_to_wgs84(p)
        res = parse_resolution(p)
        # centi-Kelvin → °C
        data_c = data / LWIR_DN_PER_K - KELVIN_OFFSET
        write_nc(out_dir / f"HP_UAV_TIR_{date_str}.nc",
                 data=data_c, lats=lats, lons=lons,
                 product="tir", date_str=date_str, resolution_m=res,
                 description="UAV LWIR (DN/100 − 273.15 → °C)", units="degC")
        print(f"  OK tir    ← {p.name}  (range {np.nanmin(data_c):.1f}–{np.nanmax(data_c):.1f} °C)")

    # ---- RGB composite from MS red/green/blue ----
    pr = find_tif(date_dir, "MS", "red")
    pg = find_tif(date_dir, "MS", "green")
    pb = find_tif(date_dir, "MS", "blue")
    if pr and pg and pb:
        # warp each on the same target grid by reprojecting to a common one.
        # Easiest: warp red first to get the grid, then reproject the others
        # onto that exact grid using its transform.
        with rasterio.open(pr) as src_r:
            transform, width, height = calculate_default_transform(
                src_r.crs, DST_CRS, src_r.width, src_r.height, *src_r.bounds
            )
            ref_crs = src_r.crs

        def warp_onto(path):
            with rasterio.open(path) as src:
                out = np.full((height, width), np.nan, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=out,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=DST_CRS,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=np.nan,
                )
            return out

        r = warp_onto(pr)
        g = warp_onto(pg)
        b = warp_onto(pb)

        # build cell-center lat/lon from the shared transform
        a, b_, c0, d, e_, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
        cols = np.arange(width); rows = np.arange(height)
        lons = c0 + a * (cols + 0.5) + b_ * 0.5
        lats = f + e_ * (rows + 0.5) + d * 0.5

        rgb = np.stack([
            stretch_band(r),
            stretch_band(g),
            stretch_band(b),
        ], axis=0)
        res = parse_resolution(pr)
        write_nc(out_dir / f"HP_UAV_RGB_{date_str}.nc",
                 data=rgb, lats=lats, lons=lons,
                 product="rgb", date_str=date_str, resolution_m=res,
                 description="UAV multispectral RGB (per-band 2–98 % stretch)",
                 rgb=True)
        print(f"  OK rgb    ← red/green/blue (composite, 2–98 % stretch)")


def main():
    if len(sys.argv) > 1:
        target = sys.argv[1]
        date_dir = RAW_ROOT / target
        if not date_dir.is_dir():
            sys.exit(f"not a folder: {date_dir}")
        process_date(date_dir)
    else:
        if not RAW_ROOT.is_dir():
            sys.exit(f"raw root not found: {RAW_ROOT}")
        for d in sorted(RAW_ROOT.iterdir()):
            if d.is_dir() and re.fullmatch(r"\d{8}", d.name):
                process_date(d)


if __name__ == "__main__":
    main()
