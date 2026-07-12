"""
make_why_no_map_gif.py — Build the "why the Models tab has no map" explainer GIF.

Story (dark theme, matches the site):
  1. Reality: soil moisture varies over meters. A fine field with the 1-km
     cal/val cell and its stations sampling different spots.
  2. Coarsen the field step by step: 1 km -> 3 km -> 9 km -> 25 km. As the
     model pixel grows past the cell, the whole 1-km cal/val cell (and its
     stations) collapses into ONE flat pixel value.
  3. Punchline: one pixel = one number, so a "map" would be a flat rectangle.
     That's why we plot the pixel's value over time instead — shown by a real
     SMAP L4 surface-SM series drawing itself in.

Output: assets/why-no-map.gif  (+ assets/why-no-map-frames.png contact sheet)

Run:  /usr/bin/python3 make_why_no_map_gif.py
Deps: numpy, matplotlib, pillow (no scipy — blur is done by hand).
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# ---- palette (site dark theme) -------------------------------------------
BG     = "#0f172a"   # slate-900
INK    = "#e2e8f0"   # light text
MUTED  = "#94a3b8"   # slate-400
ACCENT = "#f43f5e"   # rose — cal/val cell + stations (matches pulse ring)
GRIDC  = "#e2e8f0"

# dry (tan) -> wet (deep blue) soil-moisture ramp
SM_CMAP = LinearSegmentedColormap.from_list(
    "sm", ["#e9d8a6", "#94d2bd", "#0a9396", "#005f73", "#03045e"])

# ---- domain --------------------------------------------------------------
DOM = 27.0          # km across the scene
N   = 270           # fine grid (0.1 km / cell)
DX  = DOM / N
CELL = 1.0          # cal/val cell size (km)
CX = CY = DOM / 2   # cell centre


def smooth_field(n, low=16, seed=7):
    """Bilinearly-upsampled low-res noise -> a smooth continuous field."""
    rng = np.random.default_rng(seed)
    lr = rng.standard_normal((low, low))
    base = np.arange(low)
    xi = np.linspace(0, low - 1, n)
    tmp = np.empty((low, n))
    for i in range(low):
        tmp[i] = np.interp(xi, base, lr[i])
    out = np.empty((n, n))
    for j in range(n):
        out[:, j] = np.interp(xi, base, tmp[:, j])
    return out


def coarsen(field, px_km):
    """Block-average `field` onto square pixels of size px_km (km)."""
    block = max(1, int(round(px_km / DX)))
    n = field.shape[0]
    out = field.copy()
    for bi in range(0, n, block):
        for bj in range(0, n, block):
            si = slice(bi, min(bi + block, n))
            sj = slice(bj, min(bj + block, n))
            out[si, sj] = field[si, sj].mean()
    return out


# smooth VSM field, scaled to a believable 0.10–0.45 m3/m3 range
raw = smooth_field(N)
raw = (raw - raw.min()) / (raw.max() - raw.min())
BASE = 0.10 + 0.35 * raw
VMIN, VMAX = 0.10, 0.45

PIX = [1, 3, 9, 27]                    # coarsening steps (27 ~ "25 km")
FIELDS = {p: coarsen(BASE, p) for p in PIX}

# stations: a handful of spots inside the 1-km cell
rng = np.random.default_rng(3)
STN = np.column_stack([
    rng.uniform(CX - 0.42, CX + 0.42, 6),
    rng.uniform(CY - 0.42, CY + 0.42, 6),
])

# real SMAP surface-SM series for the ending time plot
def load_smap():
    f = ROOT / "data" / "models" / "SMAP_L4.csv"
    ys = []
    if f.exists():
        with f.open() as fh:
            for row in csv.DictReader(fh):
                v = row.get("sm_surface", "")
                if v not in ("", None):
                    try:
                        ys.append(float(v))
                    except ValueError:
                        pass
    if len(ys) < 20:                    # fallback synthetic
        t = np.linspace(0, 6 * np.pi, 200)
        ys = (0.25 + 0.06 * np.sin(t) + 0.03 * np.sin(3.1 * t)).tolist()
    return np.asarray(ys)

SERIES = load_smap()

# --------------------------------------------------------------------------
# Build a flat frame list. Each entry: (kind, a, b, t, res, cap)
#   map phase: display = lerp(FIELDS[a], FIELDS[b], t)
#   concl phase: ('concl', progress)
FR = []
def hold(px, res, cap, n):
    for _ in range(n):
        FR.append(("map", px, px, 0.0, res, cap))
def trans(a, b, res, cap, n):
    for k in range(n):
        FR.append(("map", a, b, (k + 1) / n, res, cap))

hold(1, "~1 km", "Reality: soil moisture varies over meters — and stations sample it", 20)
trans(1, 3,  "3 km",  "Now view it at a model's resolution…", 12)
hold(3,  "3 km",  "…the picture starts to blur into blocks", 8)
trans(3, 9,  "9 km",  "SMAP L4 · ERA5-Land = 9 km pixels", 12)
hold(9,  "9 km",  "SMAP L4 · ERA5-Land — the whole cell fits inside one pixel", 18)
trans(9, 27, "25 km", "GLDAS-2.2 = 25 km pixels", 12)
hold(27, "25 km", "GLDAS-2.2 — the entire scene is a single flat pixel", 20)
N_CONCL = 46
for k in range(N_CONCL):
    FR.append(("concl", 27, 27, (k + 1) / N_CONCL, "25 km",
               "One pixel = one number → we plot it over time"))
for _ in range(12):                     # end hold
    FR.append(("concl", 27, 27, 1.0, "25 km",
               "One pixel = one number → we plot it over time"))

# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=100)
fig.patch.set_facecolor(BG)


def draw_map(a, b, t, res, cap, frame_i):
    ax.clear()
    ax.set_facecolor(BG)
    disp = FIELDS[a] if a == b else (1 - t) * FIELDS[a] + t * FIELDS[b]
    ax.imshow(disp, origin="lower", extent=[0, DOM, 0, DOM],
              cmap=SM_CMAP, vmin=VMIN, vmax=VMAX, interpolation="nearest")

    # pixel grid of the coarser (target) resolution, fading in
    tgt = b
    if tgt > 1:
        galpha = 0.35 * (t if a != b else 1.0)
        if galpha > 0.02:
            for g in np.arange(0, DOM + 0.01, tgt):
                ax.axvline(g, color=GRIDC, lw=0.6, alpha=galpha)
                ax.axhline(g, color=GRIDC, lw=0.6, alpha=galpha)

    # 1-km cal/val cell
    ax.add_patch(Rectangle((CX - CELL / 2, CY - CELL / 2), CELL, CELL,
                           fill=False, ec=ACCENT, lw=2.2, zorder=5))
    # stations — vivid while fine, dissolve as the cell becomes one pixel
    coarse = max(a, b)
    s_alpha = 1.0 if coarse <= 3 else (0.9 if coarse < 9 else 0.25)
    ax.scatter(STN[:, 0], STN[:, 1], s=26, c=ACCENT, edgecolors="white",
               linewidths=0.7, alpha=s_alpha, zorder=6)

    # callout: cell fits inside one pixel
    if coarse >= 9:
        ax.annotate("your 1-km cell",
                    xy=(CX + CELL / 2, CY + CELL / 2),
                    xytext=(CX + 5.2, CY + 6.4),
                    color=INK, fontsize=9, ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4))

    ax.set_xlim(0, DOM); ax.set_ylim(0, DOM)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    _titles(res, cap)


def draw_concl(t, res, cap):
    ax.clear()
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # left: the single flat pixel
    flat = float(FIELDS[27].mean())
    col = SM_CMAP((flat - VMIN) / (VMAX - VMIN))
    ax.add_patch(Rectangle((0.4, 3.0), 2.6, 2.6, fc=col, ec=ACCENT, lw=2.2))
    ax.text(1.7, 2.5, "1 model pixel\n= 1 value", color=INK, fontsize=9,
            ha="center", va="top")
    ax.annotate("", xy=(4.2, 4.3), xytext=(3.2, 4.3),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2))

    # right: SMAP series drawing itself in
    x0, x1, y0, y1 = 4.6, 9.7, 1.6, 7.2
    ax.plot([x0, x0, x1], [y1, y0, y0], color=MUTED, lw=1.0)  # axes
    n = max(2, int(len(SERIES) * t))
    ys = SERIES[:n]
    xs = np.linspace(x0, x1, len(SERIES))[:n]
    yy = y0 + (ys - VMIN) / (VMAX - VMIN) * (y1 - y0)
    ax.plot(xs, yy, color="#38bdf8", lw=1.8)
    if n:
        ax.scatter([xs[-1]], [yy[-1]], s=22, color="#38bdf8", zorder=5)
    ax.text((x0 + x1) / 2, y1 + 0.55, "SMAP L4 surface soil moisture",
            color=MUTED, fontsize=8.5, ha="center")
    ax.text(x0 - 0.15, y1, "wet", color=MUTED, fontsize=7.5, ha="right", va="center")
    ax.text(x0 - 0.15, y0, "dry", color=MUTED, fontsize=7.5, ha="right", va="center")
    ax.text((x0 + x1) / 2, y0 - 0.55, "time →", color=MUTED, fontsize=8, ha="center")

    _titles(res, cap)


def _titles(res, cap):
    # title sits ABOVE the axes; resolution badge sits INSIDE the top-right
    # corner (chip on a dark box) so the two never collide.
    ax.text(0.0, 1.09, "Why the Models tab has no map",
            transform=ax.transAxes, color=INK, fontsize=13, fontweight="bold",
            ha="left", va="top")
    ax.text(0.965, 0.94, res, transform=ax.transAxes, color=ACCENT,
            fontsize=15, fontweight="bold", ha="right", va="top", zorder=8,
            bbox=dict(boxstyle="round,pad=0.32", fc=BG, ec=ACCENT, lw=1.3))
    ax.text(0.0, -0.06, cap, transform=ax.transAxes, color=MUTED,
            fontsize=9.5, ha="left", va="top")


def update(i):
    kind, a, b, t, res, cap = FR[i]
    if kind == "map":
        draw_map(a, b, t, res, cap, i)
    else:
        draw_concl(t, res, cap)
    return []


anim = FuncAnimation(fig, update, frames=len(FR), interval=1000 / 12, blit=False)
gif = ASSETS / "why-no-map.gif"
anim.save(gif, writer=PillowWriter(fps=12), savefig_kwargs={"facecolor": BG})
print(f"wrote {gif}  ({gif.stat().st_size // 1024} KB, {len(FR)} frames)")

# contact sheet for quick visual QA
picks = [10, 34, 52, 74, len(FR) - 25, len(FR) - 1]
csheet, axes = plt.subplots(2, 3, figsize=(13, 8.8), dpi=90)
for axc, fi in zip(axes.ravel(), picks):
    plt.sca(ax := axc)  # noqa
    globals()["ax"] = axc
    update(fi)
    axc.set_title(f"frame {fi}", color="#333", fontsize=9)
csheet.patch.set_facecolor("white")
csheet.tight_layout()
sheet = ASSETS / "why-no-map-frames.png"
csheet.savefig(sheet, facecolor="white")
print(f"wrote {sheet}")
