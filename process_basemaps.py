"""
process_basemaps.py — turn raw GeoTIFF basemaps into web-ready PNG + catalog.

Scans data/basemaps/raw/ for *.tif files, extracts WGS84 bounds from TIFF
georeferencing tags, applies a colormap (categorical for class rasters,
continuous for numeric properties), and writes:

    data/basemaps/<name>.png       # color-mapped, transparent on nodata
    data/basemaps/catalog.json     # all entries with bounds + legend

Run after dropping new TIFs in:
    python3 process_basemaps.py

============================================================
INPUT FORMAT  (per-TIF)
============================================================

Required:
  data/basemaps/raw/<name>.tif
      - WGS84 (EPSG:4326).  If your raster is in EASE-Grid 2.0 or another CRS,
        reproject first:
            gdalwarp -t_srs EPSG:4326 in.tif out_wgs84.tif
      - Single band.
      - GeoTIFF tags ModelTiepointTag (33922) + ModelPixelScaleTag (33550) must
        be present (gdal_translate writes these by default).

Optional sidecars (same folder, same basename):
  <name>.classes.csv       — categorical-only legend
        class_id, label, color   (color = "#RRGGBB" or named CSS color)

  <name>.json              — overrides for label, type, colormap, range, etc.
        { "label": "Sand fraction (%)",
          "type":  "continuous",
          "colormap": "viridis",
          "vmin": 0, "vmax": 100,
          "unit":  "%" }

If a .classes.csv exists  → categorical raster (PNG colored per class id)
Else if values are integer-looking with ≤32 unique values → categorical (auto colors)
Else → continuous (default colormap = viridis)
"""

import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, TiffTags

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "basemaps" / "raw"
OUT_DIR = ROOT / "data" / "basemaps"
CATALOG = OUT_DIR / "catalog.json"

# ---------- TIFF tag constants ----------
TAG_TIEPOINT     = 33922
TAG_PIXEL_SCALE  = 33550
TAG_NODATA       = 42113

# ---------- defaults for auto categorical colors (color-blind friendly) ----------
AUTO_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def hex_to_rgb(c):
    """Accept '#rrggbb' or named 'red' (very small named map)."""
    NAMED = {"red": "#ff0000", "green": "#00ff00", "blue": "#0000ff",
             "white": "#ffffff", "black": "#000000"}
    s = NAMED.get(c.lower(), c).lstrip("#")
    return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))


# ---------- TIFF reading ----------
def read_geotiff(path):
    """Return (array, west, north, dx, dy, nodata)."""
    img = Image.open(path)
    tags = img.tag_v2
    try:
        tp = tags[TAG_TIEPOINT]                 # (i, j, k, x, y, z)
        sc = tags[TAG_PIXEL_SCALE]              # (sx, sy, sz)
    except KeyError as ex:
        raise ValueError(f"{path.name}: missing GeoTIFF georef tag {ex}")
    west, north = float(tp[3]), float(tp[4])
    dx, dy = float(sc[0]), float(sc[1])
    nodata = float(tags.get(TAG_NODATA, np.nan))
    arr = np.array(img)
    return arr, west, north, dx, dy, nodata


def bounds_wgs84(arr, west, north, dx, dy):
    h, w = arr.shape
    east = west + w * dx
    south = north - h * dy
    return north, south, east, west


# ---------- colormaps ----------
def cmap_viridis(t):
    """Tiny inline viridis (5-stop) for continuous rasters."""
    stops = [(0.00, (68, 1, 84)), (0.25, (59, 82, 139)),
             (0.50, (33, 145, 140)), (0.75, (94, 201, 98)), (1.00, (253, 231, 37))]
    t = np.clip(t, 0, 1)
    r = np.zeros_like(t); g = np.zeros_like(t); b = np.zeros_like(t)
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]; t1, c1 = stops[i+1]
        m = (t >= t0) & (t <= t1)
        u = (t[m] - t0) / (t1 - t0)
        r[m] = c0[0] + (c1[0] - c0[0]) * u
        g[m] = c0[1] + (c1[1] - c0[1]) * u
        b[m] = c0[2] + (c1[2] - c0[2]) * u
    return r, g, b


COLORMAPS = {"viridis": cmap_viridis}


# ---------- QML (QGIS style) parsing ----------
def parse_qml(qml_path):
    """Parse a QGIS .qml style file.

    Returns one of:
        {"type": "categorical", "classes": [{id, label, color}, ...]}
        {"type": "continuous",  "stops":   [(value, "#hex"), ...],
                                "vmin": float, "vmax": float}
        None  (couldn't recognise)
    """
    text = qml_path.read_text(encoding="utf-8", errors="ignore")

    # categorical: <paletteEntry value="111" color="#fee6c2" label="Forest" alpha="255"/>
    pal = re.findall(
        r'<paletteEntry\b[^/]*?'
        r'(?=[^/]*value="(\d+)")'
        r'(?=[^/]*color="(#[0-9a-fA-F]{6})")'
        r'(?=[^/]*label="([^"]*)")',
        text,
    )
    if pal:
        seen = set()
        classes = []
        for v, c, l in pal:
            v = int(v)
            if v in seen: continue
            seen.add(v)
            classes.append({"id": v, "label": l or f"Class {v}", "color": c})
        return {"type": "categorical", "classes": classes}

    # continuous: <colorrampshader minimumValue="..." maximumValue="...">
    #   <item alpha="255" color="#30123b" label="..." value="-1.94"/>
    head = re.search(
        r'<colorrampshader[^>]*?minimumValue="([^"]+)"[^>]*?maximumValue="([^"]+)"',
        text,
    )
    items = re.findall(
        r'<item\b[^/]*?'
        r'(?=[^/]*color="(#[0-9a-fA-F]{6})")'
        r'(?=[^/]*value="([^"]+)")',
        text,
    )
    if head and items:
        stops = sorted(((float(val), color) for color, val in items), key=lambda s: s[0])
        return {
            "type": "continuous",
            "stops": [[v, c] for v, c in stops],
            "vmin": float(head.group(1)),
            "vmax": float(head.group(2)),
        }
    return None


def cmap_from_stops(field, stops, alpha=200):
    """Linearly interpolate RGB across (value, hex) stops."""
    vals = np.array([s[0] for s in stops], dtype="float64")
    rgbs = np.array([hex_to_rgb(s[1]) for s in stops], dtype="float64")
    flat = field.ravel()
    r = np.interp(flat, vals, rgbs[:, 0])
    g = np.interp(flat, vals, rgbs[:, 1])
    b = np.interp(flat, vals, rgbs[:, 2])
    h, w = field.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = r.reshape(h, w).astype(np.uint8)
    rgba[..., 1] = g.reshape(h, w).astype(np.uint8)
    rgba[..., 2] = b.reshape(h, w).astype(np.uint8)
    rgba[..., 3] = alpha
    return rgba


# ---------- per-raster processing ----------
def categorical_to_png(arr, classes, nodata, out_path, max_dim=2500):
    """Render class map to PNG.

    For categorical data we always use NEAREST resampling so class boundaries
    stay crisp.  Output dimensions are bounded:
        - small inputs (max dim < 800) → upscale to ~800 for visual clarity
        - very large inputs (max dim > max_dim) → downsample by integer factor
        - in between → keep native size
    """
    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for c in classes:
        col = hex_to_rgb(c["color"])
        m = arr == c["id"]
        rgba[m, 0], rgba[m, 1], rgba[m, 2] = col
        rgba[m, 3] = 220
    # nodata transparent
    nodata_mask = (arr < -1e30) | (arr > 1e30)
    if not np.isnan(nodata):
        nodata_mask |= (arr == nodata)
    rgba[nodata_mask, 3] = 0

    img = Image.fromarray(rgba)
    longest = max(h, w)
    if longest < 800:
        scale = max(1, 800 // longest)
        img = img.resize((w * scale, h * scale), Image.NEAREST)
    elif longest > max_dim:
        scale = longest / max_dim
        img = img.resize((int(w / scale), int(h / scale)), Image.NEAREST)
    img.save(out_path, optimize=True)


def continuous_to_png(arr, nodata, out_path, *,
                      stops=None, vmin=None, vmax=None, cmap_fn=cmap_viridis,
                      max_dim=2500):
    """Render a continuous raster to PNG.

    If `stops` (list of [value, "#hex"]) is given, interpolate directly in
    data units (matches a QGIS colorRampShader exactly).  Otherwise normalise
    [vmin, vmax] → [0, 1] and feed `cmap_fn`.
    """
    h, w = arr.shape
    valid = ~((arr == nodata) | (arr < -1e30) | (arr > 1e30) | np.isnan(arr))

    if stops:
        rgba = cmap_from_stops(np.where(valid, arr, stops[0][0]), stops, alpha=200)
    else:
        t = np.zeros_like(arr, dtype="float32")
        if vmax > vmin:
            t[valid] = (arr[valid] - vmin) / (vmax - vmin)
        r, g, b = cmap_fn(t)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = np.clip(r, 0, 255).astype(np.uint8)
        rgba[..., 1] = np.clip(g, 0, 255).astype(np.uint8)
        rgba[..., 2] = np.clip(b, 0, 255).astype(np.uint8)
        rgba[..., 3] = 200

    rgba[~valid, 3] = 0      # transparent on nodata
    img = Image.fromarray(rgba)

    longest = max(h, w)
    if longest < 800:
        scale = max(1, 800 // longest)
        img = img.resize((w * scale, h * scale), Image.BILINEAR)
    elif longest > max_dim:
        scale = longest / max_dim
        img = img.resize((int(w / scale), int(h / scale)), Image.BILINEAR)
    img.save(out_path, optimize=True)


def auto_classes(unique_vals):
    return [{"id": int(v), "label": f"Class {int(v)}", "color": AUTO_COLORS[i % len(AUTO_COLORS)]}
            for i, v in enumerate(sorted(unique_vals))]


def load_classes_csv(path):
    out = []
    with path.open() as f:
        for row in csv.DictReader(f):
            out.append({
                "id": int(float(row["class_id"])),
                "label": row["label"].strip(),
                "color": row["color"].strip(),
            })
    return out


# ---------- main ----------
def process_one(tif_path):
    name = tif_path.stem            # e.g. "LULC_5m" or "DEM_30m"
    arr, west, north, dx, dy, nodata = read_geotiff(tif_path)
    n, s, e, w = bounds_wgs84(arr, west, north, dx, dy)

    # sidecars (matched by basename)
    classes_path = tif_path.with_suffix(".classes.csv")
    qml_path = tif_path.with_suffix(".qml")
    json_path = tif_path.with_suffix(".json")
    overrides = json.loads(json_path.read_text()) if json_path.exists() else {}

    qml_info = parse_qml(qml_path) if qml_path.exists() else None

    # decide categorical vs continuous
    valid = ~((arr == nodata) | (arr < -1e30) | (arr > 1e30) | np.isnan(arr))
    uniq = np.unique(arr[valid])
    is_categorical = (
        (qml_info and qml_info["type"] == "categorical")
        or overrides.get("type") == "categorical"
        or classes_path.exists()
        or (overrides.get("type") != "continuous"
            and not (qml_info and qml_info["type"] == "continuous")
            and arr.dtype.kind in "iuf"
            and len(uniq) <= 32
            and np.allclose(uniq, np.round(uniq)))
    )

    out_png = OUT_DIR / f"{name}.png"
    entry = {
        "id": overrides.get("id", name.lower().replace(" ", "_")),
        "label": overrides.get("label", name.replace("_", " ")),
        "file": out_png.name,
        "n": round(n, 6), "s": round(s, 6), "e": round(e, 6), "w": round(w, 6),
    }

    if is_categorical:
        if qml_info and qml_info["type"] == "categorical":
            classes = qml_info["classes"]               # QML wins
        elif classes_path.exists():
            classes = load_classes_csv(classes_path)
        else:
            classes = auto_classes(uniq)
        # keep only classes actually present in this raster
        present = set(int(v) for v in uniq.tolist())
        classes_present = [c for c in classes if int(c["id"]) in present]
        if classes_present:
            classes = classes_present
        categorical_to_png(arr, classes, nodata, out_png)
        entry["type"] = "categorical"
        entry["classes"] = classes
    else:
        # Continuous: prefer QML colorRampShader stops if present
        if qml_info and qml_info["type"] == "continuous":
            stops = qml_info["stops"]
            vmin = float(overrides.get("vmin", qml_info["vmin"]))
            vmax = float(overrides.get("vmax", qml_info["vmax"]))
            continuous_to_png(arr, nodata, out_png, stops=stops)
            entry["type"] = "continuous"
            entry["vmin"], entry["vmax"] = vmin, vmax
            entry["unit"] = overrides.get("unit", "")
            entry["stops"] = stops                      # for legend gradient
        else:
            cmap_name = overrides.get("colormap", "viridis")
            cmap_fn = COLORMAPS.get(cmap_name, cmap_viridis)
            vmin = float(overrides.get("vmin", float(np.nanmin(arr[valid]))))
            vmax = float(overrides.get("vmax", float(np.nanmax(arr[valid]))))
            continuous_to_png(arr, nodata, out_png,
                              vmin=vmin, vmax=vmax, cmap_fn=cmap_fn)
            entry["type"] = "continuous"
            entry["colormap"] = cmap_name
            entry["vmin"], entry["vmax"] = vmin, vmax
            entry["unit"] = overrides.get("unit", "")
    print(f"  OK  {tif_path.name}  →  {out_png.name}  ({entry['type']}, "
          f"bbox {s:.5f}..{n:.5f}, {w:.5f}..{e:.5f})")
    return entry


def main():
    if not RAW_DIR.exists():
        raise SystemExit(f"raw basemap dir not found: {RAW_DIR}")
    # Each basemap lives in its own subfolder, e.g.
    #   data/basemaps/raw/LULC_5m/LULC_5m.tif (+ sidecars)
    #   data/basemaps/raw/DEM_30m/DEM_30m.tif (+ sidecars)
    # Sidecars (.qml, .json, .classes.csv) sit next to the .tif and use the
    # same basename — recursive search picks them all up.
    tifs = sorted(RAW_DIR.rglob("*.tif"))
    if not tifs:
        raise SystemExit(f"no .tif files in {RAW_DIR} (or its sub-folders)")
    entries = [process_one(t) for t in tifs]
    catalog = {"basemaps": entries}
    CATALOG.write_text(json.dumps(catalog, indent=2, ensure_ascii=False))
    print(f"wrote {CATALOG.relative_to(ROOT)}  ({len(entries)} basemap(s))")


if __name__ == "__main__":
    main()
