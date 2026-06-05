"""
prep_uav_raw.py — one-off helper to crop very large UAV TIFs (cm-resolution,
multi-GB) down to the 1-km Hampyeong cal/val cell at 1-m resolution. Output
filenames follow the schema tif_to_nc.py expects, so the existing pipeline
can take over from there.

Usage:
    python3 prep_uav_raw.py <SRC_DIR> <YYYYMMDD>
        SRC_DIR contains the lab's raw TIFs (LiDAR/, MS_indices/, etc.).
        YYYYMMDD is the flight date — output files land in
        data/uav/raw/<YYYYMMDD>/.

Lab filename pattern (observed for 20260530):
    LiDAR/<prefix>_dem.tif                 → HP_<DATE>_LiDAR_DSM_1m.tif
    MS_indices/<prefix>_blue.tif           → HP_<DATE>_MS_blue_1m.tif
    MS_indices/<prefix>_green.tif          → HP_<DATE>_MS_green_1m.tif
    MS_indices/<prefix>_red.tif            → HP_<DATE>_MS_red_1m.tif
    MS_indices/<prefix>_nir.tif            → HP_<DATE>_MS_nir_1m.tif
    MS_indices/<prefix>_lwir.tif           → HP_<DATE>_MS_lwir_1m.tif
    MS_indices/<prefix>_ndvi.tif           → HP_<DATE>_MS_NDVI_1m.tif
    MS_indices/<prefix>_ndwi.tif (if any)  → HP_<DATE>_MS_NDWI_1m.tif

After this finishes, run:
    python3 tif_to_nc.py <YYYYMMDD>
    python3 nc_to_overlay.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from rasterio.transform import from_origin
from rasterio.enums import Resampling

ROOT = Path(__file__).parent
RAW_ROOT = ROOT / "data" / "uav" / "raw"

# Hampyeong 1-km cal/val cell, in WGS84 (from CLAUDE.md)
SITE_LON_W, SITE_LAT_S = 126.54565, 35.01031
SITE_LON_E, SITE_LAT_N = 126.55602, 35.01984

# Output grid: 1 m resolution, snapped to integer UTM metres
TARGET_RES_M = 1.0
TARGET_CRS   = "EPSG:32652"          # UTM 52N (matches existing TIFs)

# Lab filename suffix → (equip, var)
NAME_MAP = {
    "_dem":   ("LiDAR", "DSM"),
    "_blue":  ("MS",    "blue"),
    "_green": ("MS",    "green"),
    "_red":   ("MS",    "red"),
    "_nir":   ("MS",    "nir"),
    "_lwir":  ("MS",    "lwir"),
    "_ndvi":  ("MS",    "NDVI"),
    "_ndwi":  ("MS",    "NDWI"),
}


def site_bbox_utm():
    left, bot, right, top = transform_bounds(
        "EPSG:4326", TARGET_CRS,
        SITE_LON_W, SITE_LAT_S, SITE_LON_E, SITE_LAT_N
    )
    # Snap to integer metres so the grid aligns predictably
    left  = float(np.floor(left));  bot = float(np.floor(bot))
    right = float(np.ceil(right));  top = float(np.ceil(top))
    return left, bot, right, top


def find_source(src_dir, suffix):
    """Find a TIF anywhere under src_dir whose name ends in <suffix>.tif"""
    for p in src_dir.rglob(f"*{suffix}.tif"):
        return p
    return None


def crop_resample(src_path, dst_path, bbox):
    left, bot, right, top = bbox
    width  = int(round((right - left) / TARGET_RES_M))
    height = int(round((top - bot)   / TARGET_RES_M))
    new_transform = from_origin(left, top, TARGET_RES_M, TARGET_RES_M)

    with rasterio.open(src_path) as src:
        if str(src.crs) != TARGET_CRS:
            raise RuntimeError(f"{src_path.name}: CRS {src.crs} != {TARGET_CRS}")
        # Pixel-window for the bbox in the source raster
        window = from_bounds(left, bot, right, top, transform=src.transform)
        data = src.read(
            1,
            window=window,
            out_shape=(height, width),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=src.nodata if src.nodata is not None else np.nan,
        )
        dtype = data.dtype if np.issubdtype(data.dtype, np.floating) else "float32"
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype("float32")
        nodata = src.nodata if src.nodata is not None else -9999.0
        # Replace original nodata sentinel with -9999.0 for consistency with
        # the existing 20260430-era files that tif_to_nc.py is tuned for.
        if src.nodata is not None:
            data = np.where(data == src.nodata, -9999.0, data)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        dst_path, "w",
        driver="GTiff",
        height=height, width=width, count=1,
        dtype="float32",
        crs=TARGET_CRS,
        transform=new_transform,
        nodata=-9999.0,
        compress="deflate",
        predictor=2,
        tiled=True,
    ) as dst:
        dst.write(data.astype("float32"), 1)


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: prep_uav_raw.py <SRC_DIR> <YYYYMMDD>")
    src_dir   = Path(sys.argv[1]).resolve()
    date_str  = sys.argv[2]
    if not src_dir.is_dir():
        sys.exit(f"source dir not found: {src_dir}")

    out_dir = RAW_ROOT / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox = site_bbox_utm()
    print(f"site bbox (UTM 52N): {bbox}")
    print(f"output → {out_dir}")
    t0_all = time.time()

    for suffix, (equip, var) in NAME_MAP.items():
        src = find_source(src_dir, suffix)
        if not src:
            print(f"  -- skip {suffix}: not found in {src_dir}")
            continue
        dst = out_dir / f"HP_{date_str}_{equip}_{var}_1m.tif"
        t0 = time.time()
        try:
            crop_resample(src, dst, bbox)
            elapsed = time.time() - t0
            size_mb = dst.stat().st_size / 1024 / 1024
            print(f"  OK {src.name:50s}  →  {dst.name}   ({elapsed:5.1f} s, {size_mb:.1f} MB)")
        except Exception as ex:
            print(f"  FAIL {src.name}: {ex}")

    print(f"\ntotal {time.time() - t0_all:.1f} s")


if __name__ == "__main__":
    main()
