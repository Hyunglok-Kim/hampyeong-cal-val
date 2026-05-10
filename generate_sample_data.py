"""
Hampyeong Cal/Val Site - Sample Data Generator

Generates example CSV files for the in-situ soil moisture network so that the
interactive map (index.html) has something to display.  Replace these CSVs with
your real measurements when ready - the column structure is what matters.

============================================================
DATA FORMAT (this is the contract the map expects)
============================================================

1) data/stations.csv  --  one row per station (metadata)
   columns:
     station_id      : str   e.g. "HP_1-1"  (used as filename + map label)
     lat             : float WGS84 latitude  (decimal degrees)
     lon             : float WGS84 longitude (decimal degrees)
     elevation_m     : float meters above sea level
     land_cover      : str   e.g. "Rice paddy", "Upland crop", "Forest"
     install_date    : str   YYYY-MM-DD
     sensor_model    : str   e.g. "METER TEROS-12"
     depths_cm       : str   comma-separated, e.g. "5,10,20,50"
     notes           : str   free text (kept short for tooltip)

2) data/timeseries/{station_id}.csv  --  one file per station
   columns:
     datetime                  : str  ISO 8601, hourly  e.g. 2024-03-15T00:00:00
     soil_moisture_5cm         : float volumetric, m3/m3 (0-1)
     soil_moisture_10cm        : float volumetric, m3/m3
     soil_moisture_20cm        : float volumetric, m3/m3
     soil_temperature_5cm      : float degC
     soil_temperature_10cm     : float degC
     soil_temperature_20cm     : float degC
     precipitation_mm          : float mm in the past hour

   Missing values: leave blank (the map skips NaN gracefully).

3) photos/{station_id}_{season}.jpg  --  optional (spring/summer/fall/winter)
   The map will show whichever exist; missing seasons fall back to a placeholder.

============================================================
"""

import csv
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

# ----- site geometry -----------------------------------------------------
# Real EASE-Grid 2.0 1-km cell center for the Hampyeong site, derived from
# data/grid/Gridlines_M200M.kml extents:
#     lat 35.01030606..35.01984341  →  span 1058.6 m
#     lon 126.54564315..126.55601660 →  span  943.7 m  (slight aspect
#         distortion is normal when an EASE-Grid 2.0 cell is expressed in WGS84)
SITE_CENTER_LAT = 35.015074737786415
SITE_CENTER_LON = 126.55082987551783
CELL_SIZE_KM = 1.0
N_STATIONS = 50

# ----- time window --------------------------------------------------------
START = datetime(2025, 1, 1, 0, 0, 0)
END = datetime(2026, 1, 1, 0, 0, 0)
STEP = timedelta(hours=1)

# ----- output paths -------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
TS_DIR = DATA_DIR / "timeseries"
UAV_DIR = DATA_DIR / "uav"
DATA_DIR.mkdir(exist_ok=True)
TS_DIR.mkdir(exist_ok=True)
UAV_DIR.mkdir(exist_ok=True)

LAND_COVER_TYPES = [
    "Rice paddy", "Upland crop", "Forest", "Grassland",
    "Bare soil", "Mixed crop", "Orchard",
]

random.seed(42)


def km_offset_to_latlon(dlat_km: float, dlon_km: float, lat0: float):
    """Convert a small km offset to degrees at latitude lat0."""
    dlat = dlat_km / 110.574
    dlon = dlon_km / (111.320 * math.cos(math.radians(lat0)))
    return dlat, dlon


def make_stations():
    rows = []
    half = CELL_SIZE_KM / 2.0
    for i in range(N_STATIONS):
        # uniform random within the 1 km x 1 km cell
        dy_km = random.uniform(-half, half)
        dx_km = random.uniform(-half, half)
        dlat, dlon = km_offset_to_latlon(dy_km, dx_km, SITE_CENTER_LAT)

        # station_id like HP_1-1, HP_1-2, ..., HP_1-50
        sid = f"HP_1-{i+1}"
        rows.append({
            "station_id": sid,
            "lat": round(SITE_CENTER_LAT + dlat, 6),
            "lon": round(SITE_CENTER_LON + dlon, 6),
            "elevation_m": round(random.uniform(15, 60), 1),
            "land_cover": random.choice(LAND_COVER_TYPES),
            "install_date": "2024-12-15",
            "sensor_model": "METER TEROS-12",
            "depths_cm": "5,10,20",
            "notes": f"Hampyeong cal/val station {i+1}",
        })
    return rows


def write_stations_csv(rows):
    path = DATA_DIR / "stations.csv"
    fields = [
        "station_id", "lat", "lon", "elevation_m", "land_cover",
        "install_date", "sensor_model", "depths_cm", "notes",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} stations)")


def synth_timeseries(station, t_index):
    """Generate a plausible-looking hourly series for one station."""
    # baseline values vary by land cover so the map looks realistic
    lc = station["land_cover"]
    base_sm = {
        "Rice paddy": 0.45, "Upland crop": 0.25, "Forest": 0.30,
        "Grassland": 0.22, "Bare soil": 0.15, "Mixed crop": 0.28,
        "Orchard": 0.27,
    }.get(lc, 0.25)

    # tiny per-station bias so the 50 series aren't identical
    bias = random.uniform(-0.04, 0.04)

    rows = []
    sm5 = base_sm + bias
    rng = random.Random(hash(station["station_id"]) & 0xFFFFFFFF)

    for t in t_index:
        # diurnal soil temperature wave + seasonal warming
        hour = t.hour
        doy = t.timetuple().tm_yday
        seasonal = 22 + 5 * math.sin((doy - 172) / 365 * 2 * math.pi)
        diurnal = 4 * math.sin((hour - 6) / 24 * 2 * math.pi)
        st5 = seasonal + diurnal + rng.gauss(0, 0.4)
        st10 = st5 - 1.0 + rng.gauss(0, 0.3)
        st20 = st5 - 2.0 + rng.gauss(0, 0.2)

        # rainfall: rare, bursty
        precip = 0.0
        if rng.random() < 0.02:
            precip = round(rng.expovariate(0.25), 2)

        # soil moisture: slow drying + jumps after precip
        sm5 += precip * 0.01 - 0.0008 + rng.gauss(0, 0.002)
        sm5 = max(0.05, min(0.55, sm5))
        sm10 = sm5 - 0.02 + rng.gauss(0, 0.005)
        sm20 = sm5 - 0.04 + rng.gauss(0, 0.005)

        rows.append({
            "datetime": t.strftime("%Y-%m-%dT%H:%M:%S"),
            "soil_moisture_5cm": round(sm5, 4),
            "soil_moisture_10cm": round(max(0.05, sm10), 4),
            "soil_moisture_20cm": round(max(0.05, sm20), 4),
            "soil_temperature_5cm": round(st5, 2),
            "soil_temperature_10cm": round(st10, 2),
            "soil_temperature_20cm": round(st20, 2),
            "precipitation_mm": precip,
        })
    return rows


def write_timeseries(stations):
    t_index = []
    t = START
    while t < END:
        t_index.append(t)
        t += STEP

    fields = [
        "datetime",
        "soil_moisture_5cm", "soil_moisture_10cm", "soil_moisture_20cm",
        "soil_temperature_5cm", "soil_temperature_10cm", "soil_temperature_20cm",
        "precipitation_mm",
    ]
    for s in stations:
        rows = synth_timeseries(s, t_index)
        path = TS_DIR / f"{s['station_id']}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {len(stations)} timeseries files in {TS_DIR}")


# =============================================================
# Map-overlay helpers
# =============================================================

def write_daily_sm_5cm(stations):
    """Wide-format daily mean SM at 5 cm for fast map-overlay lookups.

    File: data/daily_sm_5cm.csv
        date, HP_1-1, HP_1-2, ..., HP_1-50
        2024-06-01, 0.245, 0.155, ...
    The map's date picker reads this one file (small, ~250 KB) instead of
    fetching all 50 timeseries CSVs.
    """
    by_station_day = defaultdict(lambda: defaultdict(list))   # sid -> day -> [vals]
    for s in stations:
        with (TS_DIR / f"{s['station_id']}.csv").open() as f:
            for row in csv.DictReader(f):
                day = row["datetime"][:10]
                v = row.get("soil_moisture_5cm", "")
                if v:
                    by_station_day[s["station_id"]][day].append(float(v))

    days = sorted({d for d_map in by_station_day.values() for d in d_map})
    sids = [s["station_id"] for s in stations]

    path = DATA_DIR / "daily_sm_5cm.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + sids)
        for day in days:
            row = [day]
            for sid in sids:
                vals = by_station_day[sid].get(day, [])
                row.append(round(sum(vals) / len(vals), 4) if vals else "")
            w.writerow(row)
    print(f"wrote {path}  ({len(days)} days x {len(sids)} stations)")


def smooth_field(seed, size=64):
    """Smooth synthetic 2-D field in [0, 1] using sums of sinusoids."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, size)
    y = np.linspace(0, 2 * np.pi, size)
    X, Y = np.meshgrid(x, y)
    f = np.zeros_like(X)
    for _ in range(6):
        amp = rng.uniform(-1, 1)
        fx, fy = rng.uniform(0.5, 4), rng.uniform(0.5, 4)
        px, py = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
        f += amp * np.sin(fx * X + px) * np.sin(fy * Y + py)
    f = (f - f.min()) / (f.max() - f.min())
    return f


def sm_colormap_array(field, vmin=0.05, vmax=0.55, alpha=180):
    """Map a [0,1] field to RGBA brown→tan→blue array (uint8)."""
    # rescale [0,1] interp into vmin..vmax SM range visualization
    sm = vmin + field * (vmax - vmin)
    t = np.clip((sm - vmin) / (vmax - vmin), 0, 1)

    # piecewise: brown(102,51,25) → tan(230,204,128) → blue(25,77,178)
    def lerp(a, b, u):
        return a + (b - a) * u

    u1 = np.clip(t / 0.5, 0, 1)
    u2 = np.clip((t - 0.5) / 0.5, 0, 1)

    r = np.where(t < 0.5, lerp(102, 230, u1), lerp(230, 25, u2))
    g = np.where(t < 0.5, lerp(51, 204, u1), lerp(204, 77, u2))
    b = np.where(t < 0.5, lerp(25, 128, u1), lerp(128, 178, u2))

    rgba = np.zeros((*field.shape, 4), dtype=np.uint8)
    rgba[..., 0] = r.astype(np.uint8)
    rgba[..., 1] = g.astype(np.uint8)
    rgba[..., 2] = b.astype(np.uint8)
    rgba[..., 3] = alpha
    return rgba


def make_rgb_array(seed, size=200, alpha=220):
    """Synthetic multispectral true-color: vegetation greens + bare soil browns."""
    veg  = smooth_field(seed=seed,     size=size)   # vegetation index proxy
    soil = smooth_field(seed=seed + 1, size=size)   # bare-soil texture proxy
    is_veg = veg > 0.45

    r = np.where(is_veg,  80 +  60 * (1 - veg), 130 + 80 * soil)
    g = np.where(is_veg, 130 +  80 * veg,       100 + 50 * soil)
    b = np.where(is_veg,  50 +  40 * (1 - veg),  70 + 30 * soil)

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(r, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(g, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(b, 0, 255).astype(np.uint8)
    rgba[..., 3] = alpha
    return rgba


def tir_colormap_array(field, alpha=200):
    """Thermal IR colormap (black → red → yellow → white)."""
    t = np.clip(field, 0, 1)
    r = np.where(t < 0.33, 250 * t / 0.33, 250)
    g = np.where(t < 0.33, 0,
        np.where(t < 0.66, 240 * (t - 0.33) / 0.33, 240))
    b = np.where(t < 0.66, 0, 255 * (t - 0.66) / 0.34)
    rgba = np.zeros((*field.shape, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(r, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(g, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(b, 0, 255).astype(np.uint8)
    rgba[..., 3] = alpha
    return rgba


def ndvi_colormap_array(field, alpha=200):
    """NDVI colormap (brown → tan → light green → dark green)."""
    t = np.clip(field, 0, 1)

    def seg(x, x0, x1, y0, y1):
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    r = np.where(t < 0.3, seg(t, 0,   0.3, 120, 220),
        np.where(t < 0.6, seg(t, 0.3, 0.6, 220, 200),
                          seg(t, 0.6, 1.0, 200,  20)))
    g = np.where(t < 0.3, seg(t, 0,   0.3,  80, 200),
        np.where(t < 0.6, seg(t, 0.3, 0.6, 200, 230),
                          seg(t, 0.6, 1.0, 230, 100)))
    b = np.where(t < 0.3, seg(t, 0,   0.3,  40, 150),
        np.where(t < 0.6, seg(t, 0.3, 0.6, 150, 140),
                          seg(t, 0.6, 1.0, 140,  30)))

    rgba = np.zeros((*field.shape, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(r, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(g, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(b, 0, 255).astype(np.uint8)
    rgba[..., 3] = alpha
    return rgba


def lidar_colormap_array(field, alpha=200):
    """LiDAR DSM (digital surface model) colormap
    (dark green → light green → yellow → orange → red)."""
    t = np.clip(field, 0, 1)

    def seg(x, x0, x1, y0, y1):
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    r = np.where(t < 0.25, seg(t, 0, 0.25, 30, 80),
        np.where(t < 0.50, seg(t, 0.25, 0.5, 80, 220),
        np.where(t < 0.75, seg(t, 0.5, 0.75, 220, 220),
                           seg(t, 0.75, 1.0, 220, 200))))
    g = np.where(t < 0.25, seg(t, 0, 0.25, 80, 160),
        np.where(t < 0.50, seg(t, 0.25, 0.5, 160, 200),
        np.where(t < 0.75, seg(t, 0.5, 0.75, 200, 130),
                           seg(t, 0.75, 1.0, 130, 50))))
    b = np.where(t < 0.25, seg(t, 0, 0.25, 30, 60),
        np.where(t < 0.50, seg(t, 0.25, 0.5, 60, 80),
        np.where(t < 0.75, seg(t, 0.5, 0.75, 80, 50),
                           seg(t, 0.75, 1.0, 50, 40))))

    rgba = np.zeros((*field.shape, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(r, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(g, 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(b, 0, 255).astype(np.uint8)
    rgba[..., 3] = alpha
    return rgba


def write_uav_data():
    """Generate synthetic UAV products (RGB / TIR / LiDAR) at 30 cm for a few
    flight dates, and a catalog CSV the map uses to find the nearest date.

    File layout:
        data/uav/
          catalog.csv                  one row per (date, product)
          HP_UAV_RGB_YYYYMMDD_HHMM.png True-color multispectral
          HP_UAV_TIR_YYYYMMDD_HHMM.png Thermal IR (°C)
          HP_UAV_LIDAR_YYYYMMDD_HHMM.png LiDAR canopy height (m)
    """
    # clear old PNGs (schema changed - filenames now include product + time)
    for old in UAV_DIR.glob("*.png"):
        old.unlink()

    # geographic bounds of the 1 km cell in WGS84 (used by Leaflet imageOverlay)
    half_lat = (CELL_SIZE_KM / 2) / 110.574
    half_lon = (CELL_SIZE_KM / 2) / (111.320 * math.cos(math.radians(SITE_CENTER_LAT)))
    n = SITE_CENTER_LAT + half_lat
    s = SITE_CENTER_LAT - half_lat
    e = SITE_CENTER_LON + half_lon
    w = SITE_CENTER_LON - half_lon

    # 4 flight dates; each date carries multiple products with per-product times
    flights = [
        {"date": "2024-06-15", "times": {
            "rgb": "10:30:00", "tir": "10:35:00", "ndvi": "10:40:00",
            "lidar": "10:45:00", "sm": "11:00:00",
        }},
        {"date": "2024-07-05", "times": {
            "rgb": "11:00:00", "tir": "11:05:00", "ndvi": "11:10:00",
            "lidar": "11:15:00", "sm": "11:30:00",
        }},
        {"date": "2024-07-25", "times": {
            "rgb": "09:45:00", "tir": "09:50:00", "ndvi": "09:55:00",
            "lidar": "10:00:00", "sm": "10:15:00",
        }},
        {"date": "2024-08-15", "times": {
            "rgb": "10:15:00", "tir": "10:20:00", "ndvi": "10:25:00",
            "lidar": "10:30:00", "sm": "10:45:00",
        }},
    ]
    # (key, label, unit, resolution_m)
    products = [
        ("rgb",   "Multispectral RGB",   "",      0.3),
        ("tir",   "Thermal IR",          "°C",    0.3),
        ("ndvi",  "NDVI",                "",      0.3),
        ("lidar", "LiDAR DSM",           "m",     0.3),
        ("sm",    "Soil moisture (UAV)", "m³/m³", 1.0),
    ]
    PNG_SIZE = 200       # demo PNG grid (true product ≈ 3300 px @ 30 cm or 1000 @ 1 m)

    rows = []
    for d_idx, flight in enumerate(flights):
        date = flight["date"]
        for p_idx, (pkey, plabel, unit, res_m) in enumerate(products):
            if pkey not in flight["times"]:
                continue
            time = flight["times"][pkey]
            seed = 1000 + d_idx * 10 + p_idx
            f = smooth_field(seed, PNG_SIZE)
            if pkey == "rgb":
                rgba = make_rgb_array(seed, size=PNG_SIZE)
            elif pkey == "tir":
                rgba = tir_colormap_array(f)
            elif pkey == "ndvi":
                rgba = ndvi_colormap_array(f)
            elif pkey == "lidar":
                rgba = lidar_colormap_array(f)
            elif pkey == "sm":
                # Reuse the same SM colormap as the station dots so the legend matches
                rgba = sm_colormap_array(f, vmin=0.05, vmax=0.55, alpha=200)
            img = Image.fromarray(rgba)
            t_compact = time[:5].replace(":", "")
            fname = f"HP_UAV_{pkey.upper()}_{date.replace('-', '')}_{t_compact}.png"
            img.save(UAV_DIR / fname)
            rows.append({
                "date": date,
                "time": time,
                "product": pkey,
                "file": fname,
                "n": round(n, 6), "s": round(s, 6),
                "e": round(e, 6), "w": round(w, 6),
                "resolution_m": res_m,
                "description": plabel + (f" ({unit})" if unit else ""),
            })

    cat_path = UAV_DIR / "catalog.csv"
    with cat_path.open("w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        w_.writerows(rows)
    print(f"wrote {cat_path}  ({len(rows)} UAV files in {UAV_DIR})")


if __name__ == "__main__":
    stations = make_stations()
    write_stations_csv(stations)
    write_timeseries(stations)
    write_daily_sm_5cm(stations)
    # NOTE: UAV / satellite overlays now come from NetCDF (the data scientist's
    # native format).  Run these separately:
    #     python3 make_sample_nc.py    # only the first time, to seed examples
    #     python3 nc_to_overlay.py     # NC → PNG + catalog (re-run on updates)
    print("done.  For overlays: python3 nc_to_overlay.py")
