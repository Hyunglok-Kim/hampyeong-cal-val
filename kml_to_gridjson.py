#!/usr/bin/env python3
"""
kml_to_gridjson.py — one-time converter: nested EASE-Grid KML → compact JSON.

The browser used to download the full 3.9 MB (411 KB gz) lab KML and parse
it with DOMParser just to extract, per folder, the set of row-line latitudes
and column-line longitudes (from which it brackets the one cell around the
site center). This script does that extraction offline and writes
data/grid/nested_grids.json (~8 KB), which index.html now fetches instead.

The KML stays in the repo as the lab-supplied source of truth; re-run this
script if it ever changes:
    python3 kml_to_gridjson.py
"""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "data" / "grid" / "nested_grids_200m_1km_3km_9km_36km.kml"
DST  = ROOT / "data" / "grid" / "nested_grids.json"

NS = {"k": "http://www.opengis.net/kml/2.2"}


def local(tag):
    return tag.rsplit("}", 1)[-1]


def main():
    tree = ET.parse(SRC)
    root = tree.getroot()

    folders = {}
    for folder in root.iter():
        if local(folder.tag) != "Folder":
            continue
        name_el = next((c for c in folder if local(c.tag) == "name"), None)
        folder_name = (name_el.text or "").strip() if name_el is not None else ""
        if not folder_name:
            continue

        row_lats, col_lons = set(), set()
        for pm in folder.iter():
            if local(pm.tag) != "Placemark":
                continue
            pm_name = ""
            coords_text = None
            for el in pm.iter():
                t = local(el.tag)
                if t == "name" and not pm_name:
                    pm_name = (el.text or "").strip()
                elif t == "coordinates":
                    coords_text = (el.text or "").strip()
            if not coords_text:
                continue
            first = coords_text.split()[0].split(",")
            try:
                lon, lat = float(first[0]), float(first[1])
            except (ValueError, IndexError):
                continue
            if re.search(r"\brow\b", pm_name, re.I):
                row_lats.add(lat)
            elif re.search(r"\bcol\b", pm_name, re.I):
                col_lons.add(lon)

        if row_lats or col_lons:
            folders[folder_name] = {
                "rowLats": sorted(row_lats),
                "colLons": sorted(col_lons),
            }

    DST.write_text(json.dumps(
        {"source": SRC.name, "folders": folders}, separators=(",", ":")
    ))
    for name, f in folders.items():
        print(f"  {name:15s}  rows={len(f['rowLats']):>4}  cols={len(f['colLons']):>4}")
    print(f"wrote {DST}  ({DST.stat().st_size:,} bytes; source KML {SRC.stat().st_size:,})")


if __name__ == "__main__":
    main()
