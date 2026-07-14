"""
build_soil_sampling.py — Ingest the lab-measured soil-sampling metadata.

Source (NAS, ISMN admin tree):
    IN_SITU/Sensor_Data/GIST_ISMN_admin/Metadata_Sampling/
        GIST_GIST_<STATION>_<LC>_<SPOT>.csv     e.g. GIST_GIST_HP0307_RP_irr.csv

Each file is an ISMN "static variables" sheet from an actual soil sample that
was dug at the station and analysed in the lab:

    ,Depth_from,Depth_to,Value
    Clay_fraction,0.1,0.1,23.77
    Sand_fraction,0.1,0.1,12.66
    Silt_fraction,0.1,0.1,63.57
    Soil_texture,0.1,0.1,SiL
    Organic_carbon,0.1,0.1,1.69
    Total_carbon,0.1,0.1,2.16
    Organic_matter,0.1,0.1,2.92
    Landcover_classification / Climate_classification / Saturation …

This is NOT the same as the soil numbers already in stations.csv — those come
from the Korean Soil Information System (a national database estimate, one
generic value for the whole site). The sampling files are ground truth, so the
portal shows them with a "lab-measured" badge and pulses those stations blue.

Only a handful of stations are sampled so far; the rest keep the DB estimate.

Outputs:
    data/in_situ/metadata_sampling/<STATION>_<LC>_<SPOT>.csv   verbatim copies
    data/in_situ/soil_sampling.json                            site index

Run:  /usr/bin/python3 build_soil_sampling.py
"""

import csv
import json
import shutil
import sys
from pathlib import Path

SRC = Path("/Users/hyunglokkim/data_1/IN_SITU/Sensor_Data/GIST_ISMN_admin/Metadata_Sampling")
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "in_situ"
OUT_META = OUT_DIR / "metadata_sampling"
OUT_JSON = OUT_DIR / "soil_sampling.json"

# ISMN row name -> short key used by the site
NUMERIC = {
    "Sand_fraction":   "sand",
    "Silt_fraction":   "silt",
    "Clay_fraction":   "clay",
    "Organic_carbon":  "oc",
    "Total_carbon":    "tc",
    "Organic_matter":  "om",
}
TEXT = {
    "Soil_texture":             "texture",
    "Landcover_classification": "landcover",
    "Climate_classification":   "climate",
}


def parse_sampling(path):
    """ISMN static-variable sheet -> {depth_m, sand, silt, clay, texture, ...}."""
    rec, depth = {}, None
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            name, d_from, _d_to, value = row[0], row[1], row[2], row[3]
            if not name or value == "":
                continue                      # e.g. the empty Saturation row
            if depth is None and d_from:
                try:
                    depth = float(d_from)
                except ValueError:
                    pass
            if name in NUMERIC:
                try:
                    rec[NUMERIC[name]] = float(value)
                except ValueError:
                    pass
            elif name in TEXT:
                rec.setdefault(TEXT[name], value)   # Climate is listed twice
    if depth is not None:
        rec["depth_m"] = depth
    return rec


def main():
    if not SRC.is_dir():
        sys.exit(f"Source not found: {SRC}\n"
                 "Is the NAS mounted?  ls /Users/hyunglokkim/data_1")

    files = sorted(SRC.glob("GIST_GIST_*.csv"))
    if not files:
        sys.exit(f"No GIST_GIST_*.csv under {SRC}")

    OUT_META.mkdir(parents=True, exist_ok=True)
    index = {}

    for p in files:
        stem = p.stem[len("GIST_GIST_"):]          # HP0307_RP_irr
        parts = stem.split("_")
        if len(parts) != 3:
            print(f"  [skip] unexpected name: {p.name}")
            continue
        station, lc, spot = parts

        shutil.copy2(p, OUT_META / f"{stem}.csv")

        rec = parse_sampling(p)
        st = index.setdefault(station, {"landcover": "", "climate": "", "spots": {}})
        st["landcover"] = st["landcover"] or rec.pop("landcover", "")
        st["climate"]   = st["climate"]   or rec.pop("climate", "")
        rec.pop("landcover", None)
        rec.pop("climate", None)
        st["spots"][spot] = rec

        tex = rec.get("texture", "?")
        print(f"  {station} {spot:4} @ {rec.get('depth_m','?')} m  "
              f"sand {rec.get('sand','?')} silt {rec.get('silt','?')} "
              f"clay {rec.get('clay','?')}  {tex}")

        # sanity: the three fractions should close to 100 %
        try:
            tot = rec["sand"] + rec["silt"] + rec["clay"]
            if abs(tot - 100.0) > 0.5:
                print(f"    [warn] sand+silt+clay = {tot:.2f}, not ~100")
        except KeyError:
            print("    [warn] missing one of sand/silt/clay")

    OUT_JSON.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    print(f"\n[done] {len(files)} sampling files -> {OUT_META.relative_to(ROOT)}")
    print(f"[done] {len(index)} stations -> {OUT_JSON.relative_to(ROOT)}")
    print("       stations with lab-measured soil:", ", ".join(sorted(index)))


if __name__ == "__main__":
    main()
