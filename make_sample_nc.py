"""
Hampyeong Cal/Val Site - Sample NetCDF generator (overlay source format)

This script writes example .nc files in the schema the web map's converter
(`nc_to_overlay.py`) understands.  The real workflow is:

    1) You produce one .nc per (date, product) using the schema below
       and drop them into  data/uav/raw/  (or data/satellites/raw/, etc.)
    2) `python3 nc_to_overlay.py`  reads every .nc, applies the
       product-specific colormap, writes data/uav/{file}.png, and rebuilds
       data/uav/catalog.csv
    3) The web page (index.html) loads the PNGs as Leaflet imageOverlays.

============================================================
NETCDF SCHEMA  (this is the contract between you and the converter)
============================================================

Per-file structure:

    Dimensions:
        lat : N       (number of latitude rows; data is row-major north→south
                       OR south→north - both supported, lat[] is checked)
        lon : M       (number of longitude columns west→east)
        band: 3       (only for the "rgb" product — R, G, B)

    Variables:
        lat(lat)             float64   units = "degrees_north"
        lon(lon)             float64   units = "degrees_east"
        <data variable>      float32   2-D for scalar products,
                                       3-D (band, lat, lon) for "rgb"
                                       missing values via _FillValue or NaN

        Variable name should match the product, but the converter also
        falls back to the first 2-D (or 3-D for rgb) variable it finds:
            ndvi             → product "ndvi"
            tir / tir_celsius / lst   → product "tir"
            dsm / lidar_dsm / elevation → product "lidar"
            soil_moisture / sm / vwc → product "sm"
            rgb (3-D, band x lat x lon)  → product "rgb"

    Global attributes (REQUIRED):
        date         "YYYY-MM-DD"        e.g. "2024-06-15"
        time         "HH:MM:SS"          e.g. "10:40:00"
        product      one of: rgb|tir|ndvi|lidar|sm
        resolution_m float                e.g. 0.3 (UAV) or 1.0 (UAV-SM)

    Global attributes (OPTIONAL but useful):
        title         free-text title for the dataset
        description   short description shown in the map info bar
        crs           default "EPSG:4326" (WGS84). The converter assumes WGS84;
                      reproject before writing the NC if your data is in
                      another CRS (e.g. EASE-Grid 2.0).
        valid_min, valid_max   override the default colorbar range for this product

============================================================
This script writes 5 sample NCs (one per UAV product) for date 2024-06-15
so you can verify the converter works end-to-end before producing real data.
"""

import math
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

# ----- site geometry (must match generate_sample_data.py) ----------------
SITE_CENTER_LAT = 35.015074737786415
SITE_CENTER_LON = 126.55082987551783
CELL_SIZE_KM = 1.0

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "uav" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def cell_bounds_wgs84():
    """Lat/lon bounds of the 1 km cell in WGS84."""
    half_lat = (CELL_SIZE_KM / 2) / 110.574
    half_lon = (CELL_SIZE_KM / 2) / (111.320 * math.cos(math.radians(SITE_CENTER_LAT)))
    return (
        SITE_CENTER_LAT + half_lat,   # n
        SITE_CENTER_LAT - half_lat,   # s
        SITE_CENTER_LON + half_lon,   # e
        SITE_CENTER_LON - half_lon,   # w
    )


def synth_field(seed, ny, nx):
    """Same smooth-field generator used elsewhere - deterministic per seed."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    X, Y = np.meshgrid(x, y)
    f = np.zeros_like(X)
    for _ in range(6):
        amp = rng.uniform(-1, 1)
        fx, fy = rng.uniform(0.5, 4), rng.uniform(0.5, 4)
        px, py = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
        f += amp * np.sin(fx * X + px) * np.sin(fy * Y + py)
    f = (f - f.min()) / (f.max() - f.min())
    return f.astype("float32")


def write_nc(path, *, date, time_, product, resolution_m, var_name,
             values, valid_min=None, valid_max=None, description=""):
    """Write a single overlay NC matching the schema above."""
    n, s, e, w = cell_bounds_wgs84()
    if values.ndim == 2:
        ny, nx = values.shape
    else:
        nb, ny, nx = values.shape
    lats = np.linspace(n, s, ny, dtype="float64")     # north-to-south
    lons = np.linspace(w, e, nx, dtype="float64")

    if path.exists():
        path.unlink()
    with Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("lat", ny)
        nc.createDimension("lon", nx)
        v_lat = nc.createVariable("lat", "f8", ("lat",))
        v_lon = nc.createVariable("lon", "f8", ("lon",))
        v_lat.units = "degrees_north"
        v_lon.units = "degrees_east"
        v_lat[:] = lats
        v_lon[:] = lons

        if values.ndim == 2:
            v = nc.createVariable(var_name, "f4", ("lat", "lon"),
                                  zlib=True, complevel=4)
            v[:, :] = values
        else:
            nc.createDimension("band", values.shape[0])
            v = nc.createVariable(var_name, "f4", ("band", "lat", "lon"),
                                  zlib=True, complevel=4)
            v[:, :, :] = values

        if valid_min is not None: v.valid_min = np.float32(valid_min)
        if valid_max is not None: v.valid_max = np.float32(valid_max)

        nc.title = f"Hampyeong UAV {product.upper()} {date} {time_}"
        nc.date = date
        nc.time = time_
        nc.product = product
        nc.resolution_m = float(resolution_m)
        nc.crs = "EPSG:4326"
        nc.description = description
        nc.created_utc = datetime.utcnow().isoformat() + "Z"
    print(f"wrote {path.relative_to(ROOT)}  ({values.shape}, {var_name})")


# ----- one date worth of all 5 products ---------------------------------
DATE = "2025-06-15"
TIMES = {
    "rgb":   "10:30:00", "tir": "10:35:00", "ndvi": "10:40:00",
    "lidar": "10:45:00", "sm":  "11:00:00",
}
NX = NY = 200          # demo grid (use your real shape in production)
SM_NX = SM_NY = 100    # SM at 1 m → coarser grid

def make_one(product):
    if product == "rgb":
        # R, G, B channels in [0, 1]; map to typical aerial scene
        veg = synth_field(110, NY, NX)
        soil = synth_field(111, NY, NX)
        is_veg = veg > 0.45
        r = np.where(is_veg,  80 +  60 * (1 - veg), 130 + 80 * soil) / 255.0
        g = np.where(is_veg, 130 +  80 * veg,       100 + 50 * soil) / 255.0
        b = np.where(is_veg,  50 +  40 * (1 - veg),  70 + 30 * soil) / 255.0
        rgb = np.stack([r, g, b], axis=0).astype("float32")
        write_nc(RAW_DIR / f"HP_UAV_RGB_{DATE.replace('-','')}.nc",
                 date=DATE, time_=TIMES["rgb"], product="rgb",
                 resolution_m=0.3, var_name="rgb", values=rgb,
                 description="Multispectral true-color")
    elif product == "tir":
        f = synth_field(120, NY, NX)
        tir = (20 + f * 20).astype("float32")  # 20–40 °C
        write_nc(RAW_DIR / f"HP_UAV_TIR_{DATE.replace('-','')}.nc",
                 date=DATE, time_=TIMES["tir"], product="tir",
                 resolution_m=0.3, var_name="tir", values=tir,
                 valid_min=20, valid_max=40,
                 description="Thermal IR brightness temperature")
    elif product == "ndvi":
        f = synth_field(130, NY, NX)
        ndvi = (0.0 + f * 0.9).astype("float32")  # 0–0.9
        write_nc(RAW_DIR / f"HP_UAV_NDVI_{DATE.replace('-','')}.nc",
                 date=DATE, time_=TIMES["ndvi"], product="ndvi",
                 resolution_m=0.3, var_name="ndvi", values=ndvi,
                 valid_min=0, valid_max=0.9,
                 description="Normalised Difference Vegetation Index")
    elif product == "lidar":
        f = synth_field(140, NY, NX)
        dsm = (30 + f * 50).astype("float32")  # 30–80 m elevation
        write_nc(RAW_DIR / f"HP_UAV_LIDAR_{DATE.replace('-','')}.nc",
                 date=DATE, time_=TIMES["lidar"], product="lidar",
                 resolution_m=0.3, var_name="dsm", values=dsm,
                 valid_min=30, valid_max=80,
                 description="LiDAR digital surface model")
    elif product == "sm":
        f = synth_field(150, SM_NY, SM_NX)
        sm = (0.05 + f * 0.5).astype("float32")  # 0.05–0.55 m³/m³
        write_nc(RAW_DIR / f"HP_UAV_SM_{DATE.replace('-','')}.nc",
                 date=DATE, time_=TIMES["sm"], product="sm",
                 resolution_m=1.0, var_name="soil_moisture", values=sm,
                 valid_min=0.05, valid_max=0.55,
                 description="UAV-derived volumetric soil moisture")


if __name__ == "__main__":
    for p in ["rgb", "tir", "ndvi", "lidar", "sm"]:
        make_one(p)
    print("done — sample NCs written to data/uav/raw/")
