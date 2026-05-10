"""
build_pictures.py — scan data/pictures/<YYYYMMDD>[_<title>]/ folders, compress
                    images for the web, and emit data/pictures/catalog.json
                    (read by the Pictures tab in index.html).

Folder name conventions:
    20260430                          → date only, auto-titled "2026-04-30"
    20260430_NASA_JPL_Visits          → date + title (underscores → spaces)
    20260223_Polra_Field_Campaign     → another example

For each image (.jpg / .jpeg / .png / .webp / .heic), we generate a web-
optimized copy in <folder>/_web/ — long edge resized to MAX_EDGE px and
re-saved as JPEG quality 82, EXIF stripped. The catalog points the website
at the _web/ copies so the originals (often 4–10 MB each) never travel
over HTTP. Originals stay in the folder for archival.

Run any time you add or remove pictures:
    python3 build_pictures.py
    python3 build_pictures.py --force        # recompress everything
"""

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageOps

# Register HEIC/HEIF support so iPhone-style photos can be read alongside
# JPG/PNG. Optional — falls back gracefully if pillow-heif isn't installed.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

ROOT = Path(__file__).parent
PIC_ROOT = ROOT / "data" / "pictures"
CATALOG  = PIC_ROOT / "catalog.json"

DATE_RE = re.compile(r"^(\d{8})(?:_(.*))?$")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif"}

# Compression knobs — tuned for fast page loads while keeping site activity
# photos sharp enough to be presentable.
MAX_EDGE = 1600     # px on the long edge
JPEG_Q   = 82       # JPEG quality (1–95)


def title_from(date_token, title_part):
    if title_part:
        return title_part.replace("_", " ").strip()
    y, m, d = date_token[0:4], date_token[4:6], date_token[6:8]
    return f"{y}-{m}-{d}"


def web_path_for(src):
    """Web-optimized companion path: <parent>/_web/<stem>.jpg"""
    return src.parent / "_web" / (src.stem + ".jpg")


def compress_one(src, force=False):
    """Generate a web-optimized JPEG. Returns the relative web filename
    (always under _web/)."""
    dst = web_path_for(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst.name
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)            # honor camera rotation
            if im.mode in ("RGBA", "P", "LA"):
                im = im.convert("RGB")
            im.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
            im.save(dst, "JPEG", quality=JPEG_Q, optimize=True, progressive=True)
        old_kb = src.stat().st_size / 1024
        new_kb = dst.stat().st_size / 1024
        print(f"      {src.name:40s} {old_kb:7.0f} KB → {new_kb:6.0f} KB  ({new_kb / old_kb * 100:.0f}%)")
        return dst.name
    except Exception as ex:
        print(f"      ERROR {src.name}: {ex}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Recompress every image even if a fresh _web/ copy exists.")
    args = ap.parse_args()

    if not PIC_ROOT.exists():
        PIC_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"created {PIC_ROOT}")

    folders = []
    for p in sorted(PIC_ROOT.iterdir()):
        if not p.is_dir(): continue
        if p.name.startswith("_"): continue            # archive / scratch
        m = DATE_RE.match(p.name)
        if not m:
            print(f"  SKIP {p.name} (folder name not YYYYMMDD[_title])")
            continue
        date_token, title_part = m.group(1), m.group(2)
        date_iso = f"{date_token[0:4]}-{date_token[4:6]}-{date_token[6:8]}"
        title    = title_from(date_token, title_part)

        # Collect originals (anything in the folder that's a known image type)
        originals = sorted([f for f in p.iterdir()
                            if f.is_file() and f.suffix.lower() in EXTS])
        if not originals:
            print(f"  EMPTY {p.name} (no images)")
            continue

        print(f"  {p.name} → {title}  ({len(originals)} image(s))")
        web_names = []
        for src in originals:
            wn = compress_one(src, force=args.force)
            if wn: web_names.append(wn)

        # The catalog points at the _web/ subfolder, not the originals.
        folders.append({
            "folder": p.name,
            "date":   date_iso,
            "title":  title,
            "subdir": "_web",
            "images": web_names,
        })

    # newest first in the side-panel
    folders.sort(key=lambda f: f["date"], reverse=True)

    CATALOG.write_text(json.dumps({"folders": folders}, indent=2))
    print(f"\nwrote {CATALOG.relative_to(ROOT)}  ({len(folders)} folders)")


if __name__ == "__main__":
    main()
