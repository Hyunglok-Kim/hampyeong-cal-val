"""
nc_to_overlay.py - turn raw .nc rasters into web-ready PNG + catalog.csv

Default: processes data/uav/raw/.  Pass another folder to process satellites
(or any other overlay source) in the same way:

    python3 nc_to_overlay.py                          # data/uav/
    python3 nc_to_overlay.py data/satellites/         # MODIS/SMAP/NISAR
    python3 nc_to_overlay.py data/models/             # model outputs
    python3 nc_to_overlay.py /abs/path/to/anything/   # any folder with raw/

For each input dir, looks at every .nc inside <dir>/raw/, applies a
product-specific colormap, writes <dir>/<file>.png, and rebuilds
<dir>/catalog.csv that the web map (index.html) reads.

Per-NC schema is documented in `make_sample_nc.py`.  The required global
attributes are:  date, time, product, resolution_m   (others are optional).
"""

import csv
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from PIL import Image

# -----------------------------------------------------------------------
# Paths (overridden per-run by CLI arg)
# -----------------------------------------------------------------------
ROOT = Path(__file__).parent


def resolve_dirs(arg=None):
    """Return (RAW_DIR, OUT_DIR, CATALOG) given an optional CLI argument.

    The argument is a folder.  If the folder ends in 'raw', its parent is
    treated as the output dir.  Otherwise we look for <folder>/raw/.
    """
    if arg is None:
        out = ROOT / "data" / "uav"
    else:
        p = Path(arg)
        out = (p if p.is_absolute() else (ROOT / p)).resolve()
        if out.name == "raw":
            out = out.parent
    raw = out / "raw"
    return raw, out, out / "catalog.csv"


# -----------------------------------------------------------------------
# Colormaps  (must match COLORBARS in index.html)
# -----------------------------------------------------------------------
def _seg(t, x0, x1, y0, y1):
    return y0 + (y1 - y0) * (t - x0) / (x1 - x0)


def cmap_sm(field, vmin=0.05, vmax=0.55, alpha=200):
    """Dry → wet: dark brown → orange → light blue → deep blue.

    Stays saturated through the middle (no gray) and matches the JS
    smToColor() + the CSS legend gradient in index.html.
    """
    invalid = ~np.isfinite(field)
    t = np.clip((field - vmin) / (vmax - vmin), 0, 1)
    stops = [
        (0.00, (120,  60,  20)),
        (0.33, (220, 160,  60)),
        (0.66, ( 80, 170, 220)),
        (1.00, ( 20,  60, 160)),
    ]
    r = np.zeros_like(t); g = np.zeros_like(t); b = np.zeros_like(t)
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        m = (t >= p0) & (t <= p1)
        u = (t[m] - p0) / max(p1 - p0, 1e-9)
        r[m] = c0[0] + (c1[0] - c0[0]) * u
        g[m] = c0[1] + (c1[1] - c0[1]) * u
        b[m] = c0[2] + (c1[2] - c0[2]) * u
    return _stack(r, g, b, alpha, mask_invalid=invalid)


def cmap_tir(field, vmin=20, vmax=40, alpha=200):
    """Cold blue → cyan → yellow-green → yellow → orange → red (hot).

    Weather-style thermal palette: low temperatures are blue, high are red,
    so the visual matches user intuition (hot = red).  Used for both UAV TIR
    and MODIS LST.
    """
    invalid = ~np.isfinite(field)
    t = np.clip((field - vmin) / (vmax - vmin), 0, 1)
    stops = [
        (0.00, (15,  50, 130)),     # deep blue   (very cold)
        (0.20, (40, 130, 230)),     # blue
        (0.40, (120, 200, 200)),    # cyan
        (0.55, (210, 230, 110)),    # yellow-green
        (0.70, (255, 220,  60)),    # yellow
        (0.85, (240, 130,  40)),    # orange
        (1.00, (200,  30,  30)),    # red         (very hot)
    ]
    r = np.zeros_like(t); g = np.zeros_like(t); b = np.zeros_like(t)
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i+1]
        m = (t >= p0) & (t <= p1)
        u = (t[m] - p0) / max(p1 - p0, 1e-9)
        r[m] = c0[0] + (c1[0] - c0[0]) * u
        g[m] = c0[1] + (c1[1] - c0[1]) * u
        b[m] = c0[2] + (c1[2] - c0[2]) * u
    return _stack(r, g, b, alpha, mask_invalid=invalid)


def cmap_ndvi(field, vmin=0.0, vmax=0.9, alpha=200):
    """Brown → tan → light green → dark green."""
    t = np.clip((field - vmin) / (vmax - vmin), 0, 1)
    r = np.where(t < 0.3, _seg(t, 0,   0.3, 120, 220),
        np.where(t < 0.6, _seg(t, 0.3, 0.6, 220, 200),
                          _seg(t, 0.6, 1.0, 200,  20)))
    g = np.where(t < 0.3, _seg(t, 0,   0.3,  80, 200),
        np.where(t < 0.6, _seg(t, 0.3, 0.6, 200, 230),
                          _seg(t, 0.6, 1.0, 230, 100)))
    b = np.where(t < 0.3, _seg(t, 0,   0.3,  40, 150),
        np.where(t < 0.6, _seg(t, 0.3, 0.6, 150, 140),
                          _seg(t, 0.6, 1.0, 140,  30)))
    return _stack(r, g, b, alpha)


def cmap_ndwi(field, vmin=-0.5, vmax=0.5, alpha=200):
    """NDWI: deep brown (dry) → near-white (transition) → deep blue (water).

    Distinct from the SM ramp: a near-white midpoint plus heavier
    saturation at both ends so users don't confuse the two on the map.
    """
    invalid = ~np.isfinite(field)
    t = np.clip((field - vmin) / (vmax - vmin), 0, 1)
    stops = [
        (0.00, ( 80,  35,  10)),     # very dark brown
        (0.35, (180, 110,  50)),     # rich saturated brown
        (0.55, (245, 245, 245)),     # near-white transition (key contrast vs SM)
        (0.70, ( 30, 120, 200)),     # rich blue
        (1.00, (  5,  25, 100)),     # very deep navy
    ]
    r = np.zeros_like(t); g = np.zeros_like(t); b = np.zeros_like(t)
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        m = (t >= p0) & (t <= p1)
        u = (t[m] - p0) / max(p1 - p0, 1e-9)
        r[m] = c0[0] + (c1[0] - c0[0]) * u
        g[m] = c0[1] + (c1[1] - c0[1]) * u
        b[m] = c0[2] + (c1[2] - c0[2]) * u
    return _stack(r, g, b, alpha, mask_invalid=invalid)


def cmap_lidar(field, vmin=30, vmax=80, alpha=200):
    """Dark green → light green → yellow → orange → red."""
    t = np.clip((field - vmin) / (vmax - vmin), 0, 1)
    r = np.where(t < 0.25, _seg(t, 0, 0.25, 30, 80),
        np.where(t < 0.50, _seg(t, 0.25, 0.5, 80, 220),
        np.where(t < 0.75, _seg(t, 0.5, 0.75, 220, 220),
                           _seg(t, 0.75, 1.0, 220, 200))))
    g = np.where(t < 0.25, _seg(t, 0, 0.25, 80, 160),
        np.where(t < 0.50, _seg(t, 0.25, 0.5, 160, 200),
        np.where(t < 0.75, _seg(t, 0.5, 0.75, 200, 130),
                           _seg(t, 0.75, 1.0, 130, 50))))
    b = np.where(t < 0.25, _seg(t, 0, 0.25, 30, 60),
        np.where(t < 0.50, _seg(t, 0.25, 0.5, 60, 80),
        np.where(t < 0.75, _seg(t, 0.5, 0.75, 80, 50),
                           _seg(t, 0.75, 1.0, 50, 40))))
    return _stack(r, g, b, alpha)


def _stack(r, g, b, alpha, mask_invalid=None):
    """Pack r/g/b channels into RGBA. Pixels in `mask_invalid` (or any NaN in
    r/g/b) get alpha=0 so reproject borders don't render as opaque black."""
    rgba = np.zeros((*r.shape, 4), dtype=np.uint8)
    if mask_invalid is None:
        mask_invalid = ~np.isfinite(r) | ~np.isfinite(g) | ~np.isfinite(b)
    r_safe = np.where(mask_invalid, 0, r)
    g_safe = np.where(mask_invalid, 0, g)
    b_safe = np.where(mask_invalid, 0, b)
    rgba[..., 0] = np.clip(r_safe, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(g_safe, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(b_safe, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(mask_invalid, 0, alpha).astype(np.uint8)
    return rgba


# product → colormap function and default range
PRODUCT_CMAP = {
    "sm":      (cmap_sm,    0.05, 0.55, "m³/m³"),
    "sm_retr": (cmap_sm,    0.05, 0.55, "m³/m³"),  # PolRA L-band retrieved SM
    "sm_lgb":  (cmap_sm,    0.05, 0.55, "m³/m³"),  # LGB downscaled SM
    "gist_sm": (cmap_sm,    0.05, 0.55, "m³/m³"),  # GIST SM beta — same colormap as SM
    "tir":     (cmap_tir,  -15.0, 50.0, "°C"),
    "lst":     (cmap_tir,  -10.0, 45.0, "°C"),     # MODIS / Landsat LST — same as TIR
    "ndvi":    (cmap_ndvi,  0.0,  0.9,  ""),
    "ndwi":    (cmap_ndwi, -0.5,  0.5,  ""),
    "lidar":   (cmap_lidar, 30.0, 80.0, "m"),
    "hh":      (cmap_lidar, 0.00, 0.50, "linear"), # SAR backscatter — green→red ramp
    "hv":      (cmap_lidar, 0.00, 0.30, "linear"),
    "lia":     (cmap_lidar, 20.0, 60.0, "°"),      # Local Incidence Angle
    # L-band radiometer brightness temperatures — reuse weather thermal cmap.
    # Range 200..280 K (typical L-band over land/veg).
    "tb_v":    (cmap_tir,   200.0, 280.0, "K"),
    "tb_h":    (cmap_tir,   200.0, 280.0, "K"),
}


# -----------------------------------------------------------------------
# NC reader
# -----------------------------------------------------------------------
VAR_ALIASES = {
    "sm":      ["soil_moisture", "sm", "vwc", "theta", "sm_5cm"],
    "sm_retr": ["sm_retr", "sm_retrieved", "SM_Retrieved"],
    "sm_lgb":  ["sm_lgb", "sm_downscaled", "SM_LGB_downscaled"],
    "gist_sm": ["soil_moisture", "gist", "sm"],
    "tir":     ["tir", "tir_celsius", "thermal", "bt"],
    "lst":     ["lst", "lst_day", "lst_night", "land_surface_temperature"],
    "ndvi":    ["ndvi"],
    "ndwi":    ["ndwi"],
    "lidar":   ["dsm", "lidar_dsm", "elevation", "height"],
    "rgb":     ["rgb"],
    "hh":      ["sigma0_hh", "hh", "backscatter_hh"],
    "hv":      ["sigma0_hv", "hv", "backscatter_hv"],
    "lia":     ["incidence_angle", "lia", "local_incidence_angle"],
    "tb_v":    ["tb_v", "brightness_temperature_v", "tbv"],
    "tb_h":    ["tb_h", "brightness_temperature_h", "tbh"],
}


def read_nc(path):
    """Return dict with keys: date, time, source, product, resolution_m,
    n, s, e, w, values (2-D for scalar; 3-D R,G,B for rgb), vmin, vmax, description."""
    with Dataset(path, "r") as nc:
        try:
            date = nc.getncattr("date")
            time_ = nc.getncattr("time")
            product = nc.getncattr("product").lower()
            res_m = float(nc.getncattr("resolution_m"))
        except AttributeError as ex:
            raise ValueError(f"{path.name}: missing required global attribute ({ex})")
        # source is optional (older NCs may not have it).  We normalize:
        #   pull the first matching token from a controlled vocabulary out of
        #   the NC's source attr OR the filename itself.
        SOURCE_TOKENS = ("modis", "hls", "landsat", "sentinel", "smap", "nisar", "ecostress", "lband")
        raw_src = (nc.getncattr("source") if "source" in nc.ncattrs() else "")
        haystack = (str(raw_src) + " " + path.stem).lower()
        source = ""
        for tok in SOURCE_TOKENS:
            if tok in haystack:
                source = tok
                break

        if "lat" not in nc.variables or "lon" not in nc.variables:
            raise ValueError(f"{path.name}: missing lat/lon variables")
        lats = nc.variables["lat"][:]
        lons = nc.variables["lon"][:]
        # CF convention: lat/lon values are pixel CENTERS, not edges.
        # Expand the geographic bounds by half a pixel so the imageOverlay
        # covers the full grid extent (matters for coarse rasters like NISAR
        # 200 m where half a pixel = 100 m of misalignment).
        lat_max, lat_min = float(np.max(lats)), float(np.min(lats))
        lon_max, lon_min = float(np.max(lons)), float(np.min(lons))
        nlat, nlon = max(len(lats), 1), max(len(lons), 1)
        dlat_half = (lat_max - lat_min) / max(nlat - 1, 1) / 2.0 if nlat > 1 else 0.0
        dlon_half = (lon_max - lon_min) / max(nlon - 1, 1) / 2.0 if nlon > 1 else 0.0
        n = lat_max + dlat_half
        s = lat_min - dlat_half
        e = lon_max + dlon_half
        w = lon_min - dlon_half

        # find the data variable by product alias, then by the first
        # variable that isn't lat/lon and has the right dimensionality
        target = None
        for name in VAR_ALIASES.get(product, []) + [product]:
            if name in nc.variables:
                target = name
                break
        if target is None:
            for name, var in nc.variables.items():
                if name in ("lat", "lon"):
                    continue
                if (product == "rgb" and var.ndim == 3) or (product != "rgb" and var.ndim == 2):
                    target = name
                    break
        if target is None:
            raise ValueError(f"{path.name}: no suitable data variable for product '{product}'")

        var = nc.variables[target]
        values = np.array(var[:])
        # honor masked/fill
        if hasattr(var, "_FillValue"):
            values = np.where(values == var._FillValue, np.nan, values)

        vmin = float(getattr(var, "valid_min", np.nan))
        vmax = float(getattr(var, "valid_max", np.nan))
        if np.isnan(vmin) or np.isnan(vmax):
            d = PRODUCT_CMAP.get(product)
            if d:
                vmin, vmax = d[1], d[2]

        description = getattr(nc, "description", "") or getattr(nc, "title", "")

    # ensure lat is north→south so PNG row 0 is the top of the image
    if lats[0] < lats[-1]:
        if values.ndim == 2:
            values = values[::-1, :]
        else:
            values = values[:, ::-1, :]

    return dict(
        date=date, time=time_, source=source,
        product=product, resolution_m=res_m,
        n=n, s=s, e=e, w=w,
        values=values, vmin=vmin, vmax=vmax, description=description,
    )


# -----------------------------------------------------------------------
# NC → PNG
# -----------------------------------------------------------------------
def to_png(meta, png_path):
    p = meta["product"]
    v = meta["values"]
    if p == "rgb":
        # values shape (3, ny, nx) in [0, 1]
        if v.ndim != 3 or v.shape[0] != 3:
            raise ValueError(f"rgb expects (3, ny, nx); got {v.shape}")
        invalid = ~np.isfinite(v[0]) | ~np.isfinite(v[1]) | ~np.isfinite(v[2])
        r_ = np.where(invalid, 0, v[0]) * 255
        g_ = np.where(invalid, 0, v[1]) * 255
        b_ = np.where(invalid, 0, v[2]) * 255
        rgba = np.zeros((v.shape[1], v.shape[2], 4), dtype=np.uint8)
        rgba[..., 0] = np.clip(r_, 0, 255).astype(np.uint8)
        rgba[..., 1] = np.clip(g_, 0, 255).astype(np.uint8)
        rgba[..., 2] = np.clip(b_, 0, 255).astype(np.uint8)
        rgba[..., 3] = np.where(invalid, 0, 220).astype(np.uint8)
    else:
        if p not in PRODUCT_CMAP:
            raise ValueError(f"unknown product '{p}' (no colormap registered)")
        cmap_fn, *_ = PRODUCT_CMAP[p]
        rgba = cmap_fn(v, vmin=meta["vmin"], vmax=meta["vmax"])
    Image.fromarray(rgba).save(png_path)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
CATALOG_FIELDS = [
    "date", "time", "source", "product", "file",
    "n", "s", "e", "w",
    "resolution_m", "description",
]


def main(arg=None):
    raw_dir, out_dir, catalog = resolve_dirs(arg)
    if not raw_dir.exists():
        sys.exit(f"raw NC dir not found: {raw_dir}")
    # Skip files in any directory whose name starts with "_" — those are
    # archive / scratch folders (e.g., _lab/ for original lab-format inputs).
    nc_files = [p for p in sorted(raw_dir.rglob("*.nc"))
                if not any(part.startswith("_") for part in p.relative_to(raw_dir).parts)]
    if not nc_files:
        sys.exit(f"no .nc files in {raw_dir}")

    # filename prefix derived from folder name (data/uav → HP_UAV, data/satellites → HP_SAT)
    prefix_map = {"uav": "HP_UAV", "satellites": "HP_SAT", "models": "HP_MODEL"}
    prefix = prefix_map.get(out_dir.name, f"HP_{out_dir.name.upper()}")
    print(f"=== {out_dir.name} ===  raw={raw_dir}  prefix={prefix}")

    rows = []
    for ncp in nc_files:
        try:
            meta = read_nc(ncp)
        except Exception as ex:
            print(f"  SKIP {ncp.name}: {ex}")
            continue
        t_compact = str(meta["time"])[:5].replace(":", "")
        d_compact = str(meta["date"]).replace("-", "")
        # UAV: same product is flown ≤1× per day, but resolution varies (1 m,
        # 30 cm, …) and is what the user actually distinguishes on. Other
        # sources (satellites with multiple daily passes) keep the timestamp.
        if out_dir.name == "uav":
            res = meta["resolution_m"]
            res_tag = f"{int(round(res))}m" if res >= 1 else f"{int(round(res * 100))}cm"
            suffix = res_tag
        else:
            suffix = t_compact
        png_name = f"{prefix}_{meta['product'].upper()}_{d_compact}_{suffix}.png"
        to_png(meta, out_dir / png_name)

        rows.append({
            "date": meta["date"], "time": meta["time"],
            "source": meta.get("source", ""),
            "product": meta["product"], "file": png_name,
            "n": round(meta["n"], 6), "s": round(meta["s"], 6),
            "e": round(meta["e"], 6), "w": round(meta["w"], 6),
            "resolution_m": meta["resolution_m"],
            "description": meta["description"],
        })
        print(f"  OK  {ncp.name}  →  {png_name}")

    rows.sort(key=lambda r: (r["date"], r["time"], r["product"]))
    with catalog.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CATALOG_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {catalog}  ({len(rows)} entries)")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
