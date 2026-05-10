#!/usr/bin/env python3
"""
Render L-band 10m / 30cm NetCDFs to PNGs and update catalog.json.

L-band has a different on-disk layout than the satellite/UAV trees that
nc_to_overlay.py was built for, so it gets its own tiny driver:

  data/uav/lband/
  ├── catalog.json
  ├── raw/   (CSV per patch — not rendered as overlay; shown on the map
  │           as L.circleMarker points by the front-end)
  ├── 10m/   <-- NCs here, PNGs written next to them
  └── 30cm/  <-- same

For each NC we call nc_to_overlay.read_nc() + to_png(), write a sibling
PNG named exactly like the NC but with .png, then patch catalog.json so
each patch entry gets parallel "10m_png" and "30cm_png" maps next to the
existing NC paths.

Idempotent. Skips PNGs that are already up-to-date.
"""
from __future__ import annotations

import json
from pathlib import Path

# reuse the existing infrastructure
from nc_to_overlay import read_nc, to_png  # type: ignore

ROOT = Path(__file__).resolve().parent
LBAND = ROOT / "data" / "uav" / "lband"


def render_dir(d: Path) -> dict[str, dict[str, str]]:
    """Render every NC under d → sibling PNG. Return {nc_rel: png_rel}."""
    mapping: dict[str, dict[str, str]] = {}
    for ncp in sorted(d.glob("*.nc")):
        png = ncp.with_suffix(".png")
        if png.exists() and png.stat().st_mtime >= ncp.stat().st_mtime:
            print(f"  skip  {ncp.name}  (png up-to-date)")
        else:
            try:
                meta = read_nc(ncp)
                to_png(meta, png)
                print(f"  OK    {ncp.name}  →  {png.name}")
            except Exception as ex:
                print(f"  FAIL  {ncp.name}: {ex}")
                continue
        mapping[ncp.name] = {
            "nc":  str(ncp.relative_to(ROOT / "data")),
            "png": str(png.relative_to(ROOT / "data")),
        }
    return mapping


def patch_catalog(maps: dict[str, dict]) -> None:
    """Add <res>_png references to each patch in catalog.json."""
    cat_path = LBAND / "catalog.json"
    cat = json.loads(cat_path.read_text())
    for fl in cat["flights"]:
        for pt in fl["patches"]:
            for res, m in maps.items():
                for var, ncrel in pt.get(res, {}).items():
                    ncname = Path(ncrel).name
                    if ncname in m:
                        pt.setdefault(f"{res}_png", {})[var] = m[ncname]["png"]
    cat_path.write_text(json.dumps(cat, indent=2))
    print(f"\nupdated {cat_path.relative_to(ROOT)}")


def main() -> None:
    maps = {}
    for res in ("200m", "10m", "30cm"):
        d = LBAND / res
        if not d.exists():
            print(f"(skip {res}: no such folder)")
            continue
        print(f"=== L-band {res} ===")
        maps[res] = render_dir(d)
    patch_catalog(maps)


if __name__ == "__main__":
    main()
