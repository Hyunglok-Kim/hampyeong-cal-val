#!/usr/bin/env python3
"""
Generate FAKE L-band radiometer UAV data for the Hampyeong cal/val portal.

Real flights cover only specific cells of the EASE-Grid 2.0 3-km grid (M03)
on any given day. We mimic that by:

  - picking 8 flight dates in the past ~2 months
  - each flight covers 2-3 small (200 m × 200 m) "patches" within the
    central ~5-km radius
  - per patch, 3 products are produced:
      * raw    : ~120 lat/lon point measurements along a serpentine path
                 → CSV with TB_V, TB_H, SM
      * 10m    : 20×20 regular grid (one NetCDF per variable)
      * 30cm   : 333×333 regular grid covering the central 100 m × 100 m
                 of the patch (DL super-resolution proxy)
                 → one NetCDF per variable

Output tree:
  data/uav/lband/
  ├── catalog.json
  ├── raw/HP_LBAND_<DATE>_<patchid>.csv
  ├── 10m/HP_LBAND_10M_<DATE>_<patchid>_<VAR>.nc
  └── 30cm/HP_LBAND_30CM_<DATE>_<patchid>_<VAR>.nc

Variables per file: TB_V (K), TB_H (K), SM (m³/m³). TB_H is ~5-15 K below
TB_V (typical L-band V/H polarisation gap), and SM is anti-correlated with
brightness temperature (wetter soil → lower TB).
"""
from __future__ import annotations

import csv
import json
import math
import random
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

# -----------------------------------------------------------------------------
# Site configuration
# -----------------------------------------------------------------------------
SITE_LAT = 35.015074737786415
SITE_LON = 126.55082987551783
DEG_PER_M_LAT = 1 / 111_320.0
DEG_PER_M_LON = 1 / (111_320.0 * math.cos(math.radians(SITE_LAT)))

ROOT = Path(__file__).resolve().parent
OUT  = ROOT / "data" / "uav" / "lband"

# -----------------------------------------------------------------------------
# Generation parameters (kept modest so the on-disk size stays sane)
# -----------------------------------------------------------------------------
FLIGHT_CADENCE_D  = 7             # flights every N days
N_FLIGHTS         = 10            # past 10 weeks
PATCHES_PER_FLT   = (2, 3)        # min, max
PATCH_M           = 200           # patch extent on each side (m) — both 10m and 30cm grids cover this
N_RAW_POINTS      = 120           # raw GPS points per patch (location only — no values)
GRID10_N          = 20            # 200m / 10m  → 20×20  (low-res aggregate)
GRID30_N          = 333           # 200m / ~0.6m  ≈ 333×333  (DL-model output)

# Patches are placed inside / very close to the M01 1-km cal/val cell so
# they're visible at the page's default zoom. Real flights would cover
# specific M03 (3-km) cells; this is a small visual approximation.
# Patch half-size is PATCH_M/2 = 100 m, so a cluster center within ±300 m
# of the site keeps every patch edge inside the ±475 m M01 cell.
CLUSTER_OFFSET_M = 300            # cluster center ± from site (m)
PATCH_OFFSET_M   = 150            # per-patch ± from cluster center (m)

VARS = ["TB_V", "TB_H", "SM"]
UNITS = {"TB_V": "K", "TB_H": "K", "SM": "m^3/m^3"}

RNG = random.Random(20260505)
NPRNG = np.random.default_rng(20260505)


# -----------------------------------------------------------------------------
# Fake-physics helpers
# -----------------------------------------------------------------------------
def fake_sm_field(ny: int, nx: int) -> np.ndarray:
    """Smoothly varying SM 0.10..0.50 with a few wetter blobs (large-scale only).

    Used to seed the 10-m grid. The 30-cm grid uses `fake_sm_field_hires`
    which adds finer-scale features so the two resolutions look distinct.
    """
    base = NPRNG.uniform(0.18, 0.32)
    yy, xx = np.mgrid[0:ny, 0:nx] / max(ny, nx)
    field = np.full((ny, nx), base, dtype=np.float32)
    for _ in range(NPRNG.integers(2, 4)):
        cy, cx = NPRNG.uniform(0.1, 0.9, 2)
        amp = NPRNG.uniform(0.10, 0.20)
        sig = NPRNG.uniform(0.10, 0.25)
        field += amp * np.exp(-((yy - cy)**2 + (xx - cx)**2) / (2 * sig**2))
    field += NPRNG.normal(0, 0.01, field.shape)
    return np.clip(field, 0.05, 0.55).astype(np.float32)


def upsample_with_detail(coarse: np.ndarray, n_hi: int) -> np.ndarray:
    """Bilinear-upsample `coarse` to (n_hi, n_hi) and add multi-scale fine
    detail so the result looks visibly higher-resolution than the input."""
    nc = coarse.shape[0]
    # bilinear upsample
    yy = np.linspace(0, nc - 1, n_hi)
    xx = np.linspace(0, nc - 1, n_hi)
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    y0 = np.clip(np.floor(Y).astype(int), 0, nc - 1)
    y1 = np.clip(y0 + 1, 0, nc - 1)
    x0 = np.clip(np.floor(X).astype(int), 0, nc - 1)
    x1 = np.clip(x0 + 1, 0, nc - 1)
    fy = (Y - y0).astype(np.float32)
    fx = (X - x0).astype(np.float32)
    up = ((1 - fy) * (1 - fx) * coarse[y0, x0] +
          (1 - fy) *      fx  * coarse[y0, x1] +
                fy  * (1 - fx) * coarse[y1, x0] +
                fy  *      fx  * coarse[y1, x1]).astype(np.float32)

    # fine-scale features that only show at high resolution: a few small
    # blobs (~5 m scale) and per-pixel speckle (sub-meter)
    yy_n, xx_n = np.mgrid[0:n_hi, 0:n_hi] / float(n_hi)
    detail = np.zeros((n_hi, n_hi), dtype=np.float32)
    for _ in range(NPRNG.integers(8, 14)):
        cy, cx = NPRNG.uniform(0.05, 0.95, 2)
        amp = NPRNG.uniform(-0.04, 0.04)
        sig = NPRNG.uniform(0.015, 0.04)         # ~5 m blobs in 200-m extent
        detail += amp * np.exp(-((yy_n - cy)**2 + (xx_n - cx)**2) / (2 * sig**2))
    detail += NPRNG.normal(0, 0.012, (n_hi, n_hi)).astype(np.float32)   # speckle
    return np.clip(up + detail, 0.05, 0.55).astype(np.float32)


def tb_from_sm(sm: np.ndarray, pol: str) -> np.ndarray:
    """Anti-correlate TB with SM. Wet → low TB; V slightly higher than H."""
    # rough mapping: SM=0.05 → 280 K, SM=0.55 → 200 K (linear)
    base = 280.0 - (sm - 0.05) / 0.50 * 80.0
    # V vs H gap grows with SM (wet/smooth surfaces → bigger V/H gap)
    if pol == "V":
        offset = +5.0 + 8.0 * (sm - 0.10)
    else:  # H
        offset = -5.0 - 8.0 * (sm - 0.10)
    base = base + offset
    base = base + NPRNG.normal(0, 0.8, base.shape)        # instrument noise
    return base.astype(np.float32)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def patch_bounds(center_lat: float, center_lon: float, side_m: float):
    """Return (n, s, e, w) for a square patch of `side_m` centred on lat/lon."""
    half_lat = (side_m / 2.0) * DEG_PER_M_LAT
    half_lon = (side_m / 2.0) * DEG_PER_M_LON
    return (center_lat + half_lat, center_lat - half_lat,
            center_lon + half_lon, center_lon - half_lon)


def random_patch_center() -> tuple[float, float]:
    """Pick a flight cluster center near the M01 1-km cell so patches stay
    visible at the page's default zoom level."""
    dx = NPRNG.uniform(-CLUSTER_OFFSET_M, CLUSTER_OFFSET_M)
    dy = NPRNG.uniform(-CLUSTER_OFFSET_M, CLUSTER_OFFSET_M)
    return (SITE_LAT + dy * DEG_PER_M_LAT,
            SITE_LON + dx * DEG_PER_M_LON)


def serpentine_track(n: int, n_lat: int, s_lat: int, e_lon: int, w_lon: int,
                     n_lines: int = 6) -> list[tuple[float, float]]:
    """Serpentine raster: `n_lines` parallel passes, snake direction."""
    lats = np.linspace(s_lat, n_lat, n_lines)
    pts: list[tuple[float, float]] = []
    per_line = max(1, n // n_lines)
    for i, lat in enumerate(lats):
        lons = np.linspace(w_lon, e_lon, per_line)
        if i % 2 == 1:
            lons = lons[::-1]
        for lon in lons:
            jit_lat = lat + NPRNG.normal(0, 0.5) * DEG_PER_M_LAT     # ±0.5 m
            jit_lon = lon + NPRNG.normal(0, 0.5) * DEG_PER_M_LON
            pts.append((jit_lat, jit_lon))
    return pts[:n]


# -----------------------------------------------------------------------------
# NetCDF writer
# -----------------------------------------------------------------------------
def write_grid_nc(path: Path, lat_centers: np.ndarray, lon_centers: np.ndarray,
                  field: np.ndarray, var: str, date_str: str, time_str: str,
                  resolution_m: float, patch_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("lat", lat_centers.size)
        nc.createDimension("lon", lon_centers.size)
        v_lat = nc.createVariable("lat", "f8", ("lat",))
        v_lat.units = "degrees_north"
        v_lat[:] = lat_centers
        v_lon = nc.createVariable("lon", "f8", ("lon",))
        v_lon.units = "degrees_east"
        v_lon[:] = lon_centers
        v = nc.createVariable(var.lower(), "f4", ("lat", "lon"),
                              zlib=True, complevel=4)
        v.units = UNITS[var]
        v.valid_min = float(np.nanmin(field))
        v.valid_max = float(np.nanmax(field))
        v[:] = field
        # global attrs (match the contract in CLAUDE.md)
        nc.date         = date_str
        nc.time         = time_str
        nc.source       = "lband"
        nc.product      = var.lower()       # tb_v, tb_h, sm
        nc.resolution_m = float(resolution_m)
        nc.title        = f"Fake L-band {var} {date_str}"
        nc.crs          = "EPSG:4326"
        nc.patch_id     = patch_id


# -----------------------------------------------------------------------------
# Per-patch generation
# -----------------------------------------------------------------------------
def build_patch(date_str: str, time_str: str, patch_id: str,
                center_lat: float, center_lon: float) -> dict:
    n, s, e, w = patch_bounds(center_lat, center_lon, PATCH_M)
    bounds = {"n": n, "s": s, "e": e, "w": w}

    # ---- raw flight track (LOCATIONS ONLY — no measurements per point) -----
    # Real raw output is just the UAV's GPS trail; values are derived later.
    track = serpentine_track(N_RAW_POINTS, n, s, e, w)
    t0 = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    raw_rows = [((t0 + timedelta(seconds=k * 2)).strftime("%H:%M:%S"),
                 round(lat, 7), round(lon, 7))
                for k, (lat, lon) in enumerate(track)]
    raw_path = OUT / "raw" / f"HP_LBAND_{date_str.replace('-','')}_{patch_id}.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", newline="") as fp:
        wr = csv.writer(fp)
        wr.writerow(["time", "lat", "lon"])
        wr.writerows(raw_rows)

    # ---- 30-cm "truth" field (multi-scale) used to derive 10 m and 200 m ---
    # Build SM at 30-cm resolution over the SAME 200×200 m patch as 10 m, with
    # explicit fine-scale features so it visibly outresolves the 10 m grid.
    sm10_seed = fake_sm_field(GRID10_N, GRID10_N)
    sm30 = upsample_with_detail(sm10_seed, GRID30_N)              # adds 5-m blobs + speckle
    # 10 m = block-mean of the 30-cm field (so they share the same large-scale
    # pattern but 10 m is smooth, 30 cm has fine texture)
    factor = GRID30_N // GRID10_N                                 # ~16
    n_use  = factor * GRID10_N
    sm10   = sm30[:n_use, :n_use].reshape(GRID10_N, factor, GRID10_N, factor).mean(axis=(1, 3))
    sm10   = sm10.astype(np.float32)
    # 200 m "upscaled" = single patch-mean value (1×1 grid)
    sm200  = np.array([[float(sm10.mean())]], dtype=np.float32)

    def derive_tb(field, pol):
        return tb_from_sm(field, pol)

    fields_by_res = {
        "200m": {"SM": sm200,
                 "TB_V": derive_tb(sm200, "V"),
                 "TB_H": derive_tb(sm200, "H")},
        "10m":  {"SM": sm10,
                 "TB_V": derive_tb(sm10, "V"),
                 "TB_H": derive_tb(sm10, "H")},
        "30cm": {"SM": sm30,
                 "TB_V": derive_tb(sm30, "V"),
                 "TB_H": derive_tb(sm30, "H")},
    }
    grid_n_by_res = {"200m": 1, "10m": GRID10_N, "30cm": GRID30_N}
    res_m_by_res  = {"200m": float(PATCH_M),
                     "10m":  PATCH_M / GRID10_N,
                     "30cm": PATCH_M / GRID30_N}
    folder_by_res = {"200m": "200m", "10m": "10m", "30cm": "30cm"}
    fname_tag     = {"200m": "200M", "10m": "10M",  "30cm": "30CM"}

    out_files: dict[str, dict[str, str]] = {}
    for res, vars_dict in fields_by_res.items():
        gn = grid_n_by_res[res]
        # cell centers (true centers, not edges — half-pixel padding handled
        # downstream by nc_to_overlay's read_nc)
        lat_c = np.linspace(s, n, gn + 1)[:-1] + (n - s) / gn / 2
        lon_c = np.linspace(w, e, gn + 1)[:-1] + (e - w) / gn / 2
        out_files[res] = {}
        for var, field in vars_dict.items():
            p = OUT / folder_by_res[res] / (
                f"HP_LBAND_{fname_tag[res]}_{date_str.replace('-','')}_{patch_id}_{var}.nc")
            write_grid_nc(p, lat_c, lon_c, field, var, date_str, time_str,
                          resolution_m=res_m_by_res[res], patch_id=patch_id)
            out_files[res][var] = str(p.relative_to(ROOT / "data"))

    return {
        "id": patch_id,
        "bounds": bounds,
        "raw_file": str(raw_path.relative_to(ROOT / "data")),
        "raw_n_points": len(raw_rows),
        "200m": out_files["200m"],
        "10m":  out_files["10m"],
        "30cm": out_files["30cm"],
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Flights every FLIGHT_CADENCE_D days, going back N_FLIGHTS weeks
    today = date.today()
    flight_dates = sorted({today - timedelta(days=FLIGHT_CADENCE_D * (i + 1))
                           for i in range(N_FLIGHTS)})

    flights = []
    for i, fdate in enumerate(flight_dates):
        date_str = fdate.strftime("%Y-%m-%d")
        # take-off around mid-morning local (UTC+9)  → 01:30..03:30 UTC
        hh = RNG.randint(1, 3)
        mm = RNG.randint(0, 59)
        time_str = f"{hh:02d}:{mm:02d}:00"

        n_patches = RNG.randint(*PATCHES_PER_FLT)
        # cluster patches loosely so they look like a real "today's coverage"
        cluster_lat, cluster_lon = random_patch_center()
        patches = []
        for k in range(n_patches):
            dx = NPRNG.uniform(-PATCH_OFFSET_M, PATCH_OFFSET_M)
            dy = NPRNG.uniform(-PATCH_OFFSET_M, PATCH_OFFSET_M)
            plat = cluster_lat + dy * DEG_PER_M_LAT
            plon = cluster_lon + dx * DEG_PER_M_LON
            pid  = f"P{i+1:02d}_{k+1}"
            patches.append(build_patch(date_str, time_str, pid, plat, plon))
            print(f"  {date_str} {pid}: ({plat:.5f}, {plon:.5f})  raw={N_RAW_POINTS} pts")

        flights.append({
            "date":  date_str,
            "time":  time_str,
            "patches": patches,
        })

    catalog = {
        "site": {"lat": SITE_LAT, "lon": SITE_LON},
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_flights": len(flights),
        "flights": flights,
    }
    cat_path = OUT / "catalog.json"
    with cat_path.open("w") as fp:
        json.dump(catalog, fp, indent=2)
    print(f"\nwrote {cat_path.relative_to(ROOT)}  ({len(flights)} flights)")


if __name__ == "__main__":
    main()
