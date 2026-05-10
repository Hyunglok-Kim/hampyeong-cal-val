# Hampyeong Cal/Val Data Portal

Interactive web portal for the Hampyeong (함평) 1-km cal/val site, run by
HydroAI lab. Designed to be embedded in **hydroai.net** (Webflow) via iframe.
Stack is pure static: Leaflet + Plotly + PapaParse, no backend.

## Quick start

```bash
# in hampyeong_data_portal/
python3 -m http.server 8765
# open http://localhost:8765/index.html
```

That's it — every script writes into the static `data/` tree the page already
reads.

## Site geometry (single source of truth: KMLs)

Real EASE-Grid 2.0 1-km cell, derived from the "L2SMSP 1 km" folder of
`data/grid/nested_grids_200m_1km_3km_9km_36km.kml`:

- **Center**: `35.015074737786415, 126.55082987551783`
- **bbox**: lat `35.01031..35.01984`, lon `126.54565..126.55602`
- **Half-extent**: ~470 m N-S, ~470 m E-W

Three Python scripts hard-code these (`generate_sample_data.py`,
`make_sample_nc.py`, `download_modis.py`/`download_hls.py`'s SITE_LAT/LON).
**If the cell ever moves, change all three** — the KMLs are visual only, the
Python derivations of `BBOX` are independent.

## Data layout

```
data/
├── stations.csv                 # 50 in-situ stations
├── timeseries/HP_1-N.csv        # one per station, hourly 2025-01..2026-01
├── daily_sm_5cm.csv             # wide-format daily SM means → map dot coloring
│
├── grid/                        # EASE-Grid 2.0 KML (lab-supplied)
│   └── nested_grids_200m_1km_3km_9km_36km.kml
│                                # single nested file with 5 folders:
│                                #   "NISAR 200 m"   interior 200-m gridlines
│                                #   "L2SMSP 1 km"   outer 1-km validation cell (fitBounds target)
│                                #   "L2SMSP 3 km"   3-km cells (SMAP enhanced)
│                                #   "L2SMSP 9 km"   9-km cells (SMAP native)
│                                #   "L2SMSP 36 km"  36-km composite cell
│
├── basemaps/                    # static rasters (LULC, DEM, ...)
│   ├── raw/<NAME>/<NAME>.tif    # input GeoTIFF (WGS84)
│   ├── raw/<NAME>/<NAME>.qml    # QGIS style (palette OR colorRampShader)
│   ├── raw/<NAME>/<NAME>.classes.csv  # alt to QML for categorical
│   ├── raw/<NAME>/<NAME>.json   # overrides: label, unit, colormap, vmin/vmax
│   ├── <NAME>.png               # ← generated
│   └── catalog.json             # ← generated (read by index.html)
│
├── uav/                         # 30-cm UAV products (lab-supplied or synthetic)
│   ├── raw/HP_UAV_*.nc          # one .nc per (date, product); see schema below
│   ├── HP_UAV_*.png             # ← generated
│   └── catalog.csv              # ← generated
│
└── satellites/                  # MODIS, HLS, Landsat, NISAR (and future SMAP)
    ├── raw/MODIS_NDVI/*.nc      # download_modis.py output
    ├── raw/MODIS_LST/*.nc       # download_modis.py output
    ├── raw/HLS_NDVI/*.nc        # download_hls.py output
    ├── raw/LANDSAT_LST/*.nc     # download_hls.py output
    ├── raw/NISAR_SM/*.nc        # prep_nisar.py output (one .nc per variable)
    ├── raw/NISAR_SM/_lab/       # original lab-format NISAR files (preserved,
    │                            #   skipped by nc_to_overlay because of `_`)
    ├── raw/SMAP_SM/             # empty placeholder
    ├── HP_SAT_*.png             # ← generated
    └── catalog.csv              # ← generated
```

## Pipelines

### 1. In-situ data
```bash
python3 generate_sample_data.py
```
Writes 50 station files + daily aggregate. Replace these with real measurements
when ready — same column structure (see file's docstring).

### 2. Basemaps (LULC, DEM, soil properties, ...)
```bash
# Drop TIF + .qml (or .classes.csv) + optional .json into raw/<NAME>/
python3 process_basemaps.py
```
QML auto-detected: `<paletteEntry>` → categorical, `<colorrampshader>` →
continuous (256 stops preserved verbatim into legend gradient).
Resolution-aware: tiny rasters get NEAREST-upscaled to ~800 px, huge ones get
NEAREST-downsampled to ≤2500 px on the longest side.

### 3. UAV products
```bash
# Drop NCs into data/uav/raw/ matching make_sample_nc.py schema
python3 nc_to_overlay.py data/uav/
```
Required NC global attrs: `date, time, product, resolution_m`. Optional:
`source, description, valid_min/max`. See `make_sample_nc.py` for the canonical
schema and a synthetic example for all 5 products (rgb, tir, ndvi, lidar, sm).

### 4. Satellites — MODIS
```bash
# one-time setup
pip install --user earthengine-api requests netCDF4 numpy
earthengine authenticate

# every run
/usr/bin/python3 download_modis.py --project nodal-skein-411619
/usr/bin/python3 nc_to_overlay.py data/satellites/
```
- Project ID `nodal-skein-411619` is the user's registered GCP project.
- Use `/usr/bin/python3` NOT conda — Earth Engine deps are installed there.
- `--start YYYY-MM-DD --end YYYY-MM-DD` for partial date range.
- Use `format='NPY'` not `'GEO_TIFF'` (PIL silently mishandles float32 GeoTIFFs
  from EE — produces ~1e-30 garbage values that look like all-zero data).

### 5. Satellites — HLS NDVI + Landsat LST
```bash
/usr/bin/python3 download_hls.py --project nodal-skein-411619
/usr/bin/python3 nc_to_overlay.py data/satellites/
```
HLS-L30 bands: `B4` (red), `B5` (NIR). HLS-S30 bands: `B4` (red), `B8A` (NIR).
Landsat ST_B10 → Celsius via `*0.00341802 + 149 - 273.15`.
Cloud-masked via Fmask / QA_PIXEL bitmasks. Scenes <10% valid are auto-skipped.

### 6. NISAR (lab-supplied L-band SAR)
```bash
# Drop NISAR_beta_HP_YYYYMMDD_HHMMSS.nc into data/satellites/raw/NISAR_SM/_lab/
# (or in the website root — prep_nisar.py finds them either place and archives)
python3 prep_nisar.py
/usr/bin/python3 nc_to_overlay.py data/satellites/
```
Lab format has 5 variables per scene: NISAR(Beta) SM, GIST(Beta) SM, HH, HV,
LIA. `prep_nisar.py` writes one overlay NC per variable (5 × N dates).
Files in `_lab/` are kept as backup, `nc_to_overlay.py` skips them.

### 7. Visitor History (Pictures tab)
```bash
# Drop folders into data/pictures/YYYYMMDD[_<title>]/.  Folder name rules:
#   YYYYMMDD                          → date-only, auto-titled "YYYY-MM-DD"
#   YYYYMMDD_NASA_JPL_Visit           → underscores in title become spaces in UI
#   (no spaces / parentheses — rename "(Dr. Yueh)" → "Dr_Yueh", etc.)
python3 build_pictures.py             # builds catalog.json + _web/ thumbnails
python3 build_pictures.py --force     # recompress everything (e.g. after changing JPEG_Q)
```
- Inputs: any of `.jpg .jpeg .png .webp .gif .heic .heif`. HEIC needs
  `pillow-heif` (`pip install --user pillow-heif`); registered automatically
  if present, silently skipped if not.
- Web-optimized copies land in each folder's `_web/` subfolder (long-edge
  1600 px, JPEG q82, EXIF-rotated, EXIF metadata stripped). Originals stay
  put for archival.
- The site reads `data/pictures/catalog.json` (newest visit first) and
  serves photos from the `_web/` subfolder. Folders without images print
  EMPTY and are skipped. Folder names that don't start with 8 digits are
  skipped; folders starting with `_` are treated as archive (skipped).
- The Visitor History tab in `index.html` swaps the leaflet column for a
  gallery + side-panel folder list; nothing on disk needs to change when
  the user clicks around the UI.

### 8. Per-station photos (in-situ side panel)

Three sibling folders under `photos/`:

```
photos/
├── installation/   HP_<X>_<Y>_Installation_<N>.png   (N = 1..6)
│                   students installing the sensor at the in-situ point
├── sensors/        HP_<X>_<Y>_SM_10cm.png
│                   HP_<X>_<Y>_SM_10cm_Depth.png
│                   HP_<X>_<Y>_SM_20cm.png
│                   HP_<X>_<Y>_SM_20cm_Depth.png
│                   HP_<X>_<Y>_Tree.png
│                   HP_<X>_<Y>_Logger.png
│                   close-ups of the actual installed sensor at the site
└── uav_payloads/   UAV_polra_L_band.png, UAV_LiDAR.png, UAV_Multispectral.png
                    photos of the UAV sensor payloads (referenced by
                    PAYLOAD_SENSOR in index.html)
```

Filename convention uses underscores even though `station_id` uses dashes
(e.g. station_id `HP_1-2` → photo prefix `HP_1_2_…`).

When a station is selected, `selectStation()` probes a fixed candidate list
(installation_1 → SM_10cm → SM_20cm → Tree → installation_2/3 → depth →
logger) and renders the **first 4 photos that load**, in candidate order.
Missing files are skipped silently. To change the priority or the cap,
edit the `candidates` array and `MAX_PHOTOS` in `index.html`'s
`selectStation`.

## NC overlay schema (the contract `nc_to_overlay.py` enforces)

```
Dimensions: lat (N), lon (M)        # 1-D regular grid (curvilinear → flatten in prep)
Variables:
  lat(lat)              float64    units = "degrees_north"   # CELL CENTERS
  lon(lon)              float64    units = "degrees_east"    # CELL CENTERS
  <var>(lat, lon)       float32    units = ...
  rgb(band, lat, lon)   float32    # only when product == "rgb"
Required global attrs:
  date         "YYYY-MM-DD"
  time         "HH:MM:SS"
  product      sm | gist_sm | tir | lst | ndvi | lidar | rgb | hh | hv | lia
  resolution_m float
Optional global attrs:
  source       modis | hls | landsat | sentinel | smap | nisar | ecostress
  title, description, crs ("EPSG:4326")
  valid_min / valid_max  on the data variable
```

**CF convention**: lat/lon are CELL CENTERS. `nc_to_overlay.py` expands the
geographic bounds by HALF a pixel when it computes the imageOverlay extent.
**Don't write edge coordinates** (e.g. `np.linspace(n_corner, s_corner, ny)`)
unless you also subtract the half-pixel — older `download_modis.py` and
`download_hls.py` write edges, so they're off by half a pixel; NISAR (200 m)
is the only product where the user noticed this and prep_nisar.py writes
proper centers.

## Front-end UI conventions (index.html)

### Layer toggles (left of map)

```
☐ Stations  ☐ UAV [product▼]  ☐ Satellite [src:product▼]  ☐ NISAR [var▼]  ☐ SMAP (future)
```

Each top-level layer is independent (can stack). NISAR is its own toggle
(L-band radar, distinct from optical satellites) and stays on top of Satellite
when both are visible.

### Variable colormaps (must match between Python writer and HTML legend)

| product key | colormap | range | unit |
|---|---|---|---|
| `sm`, `gist_sm` | brown → tan → blue | 0.05 — 0.55 | m³/m³ |
| `tir` | weather thermal (blue → red) | 20 — 40 | °C |
| `lst` | weather thermal (blue → red) | -10 — 45 | °C |
| `ndvi` | brown → tan → green | 0 — 0.9 | — |
| `lidar` | terrain (green → red) | 30 — 80 | m |
| `hh`, `hv` | terrain (green → red) | 0 — 0.5 / 0.3 | linear |
| `lia` | terrain (green → red) | 20 — 60 | ° |

Python sources of truth: `nc_to_overlay.py`'s `PRODUCT_CMAP` and the per-product
`cmap_*` functions. Legend gradients live in `index.html`'s `COLORBARS` object.
**Keep both in sync** when changing a colormap — the page won't error if they
differ, just look misleading.

### z-order (Leaflet panes, see index.html)

Bottom → top. Set explicitly so order is independent of layer add/remove timing:
```
basemapPane    z 350   (LULC, DEM)
satellitePane  z 360   (MODIS, HLS, Landsat, NISAR — NISAR brought to front)
uavPane        z 370   (UAV products)
gridPane       z 380   (KML cell + gridlines; M09 brought to front of M01/M03)
markerPane     z 600   (stations — always on top, default Leaflet)
```

`satellitePane` carries the `pane-pixelated` CSS class by default → MODIS pixels
render as a crisp grid. The "interpolate" checkbox toggles this class off.

## Tabs

5 tabs: **In-situ** (fully implemented), UAVs / Satellites / Models /
AI prediction (currently placeholders that say "coming soon"). The In-situ tab
already houses ALL overlays (UAV, satellite, NISAR, basemaps), so the other
tabs are for richer per-source views to be designed later.

## Footer logos (in order)

GIST, HydroAI, JPL, GSFC, SMAP, FMI, MIT — `logos/<NAME>.png`. To add or
re-order, edit the `<footer class="logos">` block in `index.html`.

## Webflow embed

Once finalized, host this folder on GitHub Pages / Netlify / Webflow assets,
then in Webflow:
```html
<iframe src="https://your-host/hampyeong_data_portal/index.html"
        width="100%" height="900" style="border:0;"></iframe>
```

## Known gotchas

- **Insync / Google Drive sync** can clobber files written rapidly by Python
  (this hit us during MODIS download — half the NDVI files vanished mid-run).
  Per-product subfolders helped. If you see catalog row count ≠ files on disk
  after a download, suspect Insync.
- **PIL + EE GeoTIFF**: PIL reads EE-served GeoTIFFs as wrong dtype, producing
  garbage like `4e-35`. Always use `format='NPY'` from `getDownloadURL`.
- **System Python only**: scripts use `/usr/bin/python3` (3.9). Earth Engine
  and other deps were installed there with `pip install --user`. Conda Python
  doesn't have them.
- **Catalog source field normalization**: `nc_to_overlay.py` matches a small
  vocabulary (`modis | hls | landsat | sentinel | smap | nisar | ecostress`)
  out of either the NC's `source` global attr OR the filename. So lab files
  with `source = "NASA MODIS via Google Earth Engine"` correctly land as
  `source = "modis"` in the catalog.
- **Half-pixel padding**: only correct when lat/lon arrays really are CELL
  CENTERS. Our older MODIS/HLS NCs store edges, so they're off by half a
  pixel. Not user-noticeable for coarse rasters but worth fixing on the next
  re-download if it matters.

## Pending / next things to consider

- Tabs: UAVs / Satellites / Models / AI prediction need real per-source views
- SMAP overlay (placeholder checkbox exists, no data pipeline yet)
- Data request form (currently uses `mailto:hyunglokkim@gmail.com`)
- LULC overlay tweaks (legend wraps OK at 21 classes; revisit if more)
- The 5 in-situ time-series variables expose opacity sliders (per-trace) —
  consider also a global "show only checked, fade unchecked" mode
