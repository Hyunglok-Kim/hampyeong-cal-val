"""
make_why_no_map_gif.py — Build the "why the Models tab has no map" explainer GIF.

Portrait / stacked banner (~470x620) sized to drop in as the 4th card of the
Models side panel, next to Models / Stations-vs-network / Skill-vs-in-situ:

  TOP    — a fine soil-moisture field coarsens 1 km -> 3 km -> 9 km -> 25 km
           until the whole 1-km cal/val cell (and its stations) collapses into
           ONE flat model pixel.
  BOTTOM — an arrow leads down to the real SMAP L4 surface-SM series drawing
           itself in: one pixel = one number per day.

Punchline: a model "map" would be a single flat rectangle, so we plot the
pixel's value over time instead.

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
BG     = "#0f172a"   # slate-900 — matches the side-panel card background
INK    = "#e2e8f0"
MUTED  = "#94a3b8"
ACCENT = "#f43f5e"   # rose — cal/val cell + stations
LINEC  = "#38bdf8"   # sky — the model series
GRIDC  = "#e2e8f0"

SM_CMAP = LinearSegmentedColormap.from_list(
    "sm", ["#e9d8a6", "#94d2bd", "#0a9396", "#005f73", "#03045e"])

# ---- domain --------------------------------------------------------------
DOM = 27.0
N   = 270
DX  = DOM / N
CELL = 1.0
CX = CY = DOM / 2


def smooth_field(n, low=16, seed=7):
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
    block = max(1, int(round(px_km / DX)))
    n = field.shape[0]
    out = field.copy()
    for bi in range(0, n, block):
        for bj in range(0, n, block):
            si = slice(bi, min(bi + block, n))
            sj = slice(bj, min(bj + block, n))
            out[si, sj] = field[si, sj].mean()
    return out


raw = smooth_field(N)
raw = (raw - raw.min()) / (raw.max() - raw.min())
BASE = 0.10 + 0.35 * raw
VMIN, VMAX = 0.10, 0.45

PIX = [1, 3, 9, 27]
FIELDS = {p: coarsen(BASE, p) for p in PIX}

rng = np.random.default_rng(3)
STN = np.column_stack([
    rng.uniform(CX - 0.42, CX + 0.42, 6),
    rng.uniform(CY - 0.42, CY + 0.42, 6),
])


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
    if len(ys) < 20:
        t = np.linspace(0, 6 * np.pi, 200)
        ys = (0.25 + 0.06 * np.sin(t) + 0.03 * np.sin(3.1 * t)).tolist()
    return np.asarray(ys)

SERIES = load_smap()

# --------------------------------------------------------------------------
# entry: (a, b, t, res, cap, sprog)  — sprog grows the series in the conclusion
FR = []
def hold(px, res, cap, n, sprog=0.0):
    for _ in range(n):
        FR.append((px, px, 0.0, res, cap, sprog))
def trans(a, b, res, cap, n):
    for k in range(n):
        FR.append((a, b, (k + 1) / n, res, cap, 0.0))

hold(1, "~1 km", "Reality: soil moisture varies over meters", 16)
trans(1, 3,  "3 km",  "View it at a model's resolution…", 10)
hold(3,  "3 km",  "…the picture blurs into blocks", 6)
trans(3, 9,  "9 km",  "SMAP L4 · ERA5-Land = 9 km", 10)
hold(9,  "9 km",  "The whole 1-km cell fits in one pixel", 14)
trans(9, 27, "25 km", "GLDAS-2.2 = 25 km", 10)
hold(27, "25 km", "The entire site is one flat pixel", 14)
N_CONCL = 40
for k in range(N_CONCL):
    FR.append((27, 27, 0.0, "25 km",
               "One pixel = one number → plot it over time", (k + 1) / N_CONCL))
for _ in range(12):
    FR.append((27, 27, 0.0, "25 km",
               "One pixel = one number → plot it over time", 1.0))

# --------------------------------------------------------------------------
# portrait: square map on top, wide-short series below.
fig = plt.figure(figsize=(4.7, 6.5), dpi=100)
fig.patch.set_facecolor(BG)
axMap = fig.add_axes([0.15, 0.545, 0.70, 0.385])   # square-ish, upper
axSer = fig.add_axes([0.16, 0.115, 0.72, 0.195])   # wide-short, lower


def draw(a, b, t, res, cap, sprog):
    axMap.clear(); axSer.clear()
    axMap.set_facecolor(BG); axSer.set_facecolor(BG)

    # ---- TOP: coarsening map ---------------------------------------------
    disp = FIELDS[a] if a == b else (1 - t) * FIELDS[a] + t * FIELDS[b]
    axMap.imshow(disp, origin="lower", extent=[0, DOM, 0, DOM],
                 cmap=SM_CMAP, vmin=VMIN, vmax=VMAX, interpolation="nearest")
    if b > 1:
        ga = 0.35 * (t if a != b else 1.0)
        if ga > 0.02:
            for g in np.arange(0, DOM + 0.01, b):
                axMap.axvline(g, color=GRIDC, lw=0.6, alpha=ga)
                axMap.axhline(g, color=GRIDC, lw=0.6, alpha=ga)
    axMap.add_patch(Rectangle((CX - CELL / 2, CY - CELL / 2), CELL, CELL,
                              fill=False, ec=ACCENT, lw=2.2, zorder=5))
    coarse = max(a, b)
    s_alpha = 1.0 if coarse <= 3 else (0.9 if coarse < 9 else 0.25)
    axMap.scatter(STN[:, 0], STN[:, 1], s=22, c=ACCENT, edgecolors="white",
                  linewidths=0.6, alpha=s_alpha, zorder=6)
    axMap.text(0.95, 0.94, res, transform=axMap.transAxes, color=ACCENT,
               fontsize=15, fontweight="bold", ha="right", va="top", zorder=8,
               bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=ACCENT, lw=1.4))
    axMap.set_xlim(0, DOM); axMap.set_ylim(0, DOM); axMap.set_aspect("equal")
    axMap.set_xticks([]); axMap.set_yticks([])
    for s in axMap.spines.values():
        s.set_edgecolor(MUTED); s.set_linewidth(0.8)

    # ---- arrow down + label ----------------------------------------------
    down_on = sprog > 0
    axMap.annotate("", xy=(0.5, 0.475), xytext=(0.5, 0.525),
                   xycoords="figure fraction", textcoords="figure fraction",
                   arrowprops=dict(arrowstyle="-|>",
                                   color=INK if down_on else MUTED, lw=2.4),
                   annotation_clip=False)
    axMap.text(0.5, 0.44, "1 pixel = 1 value / day", transform=fig.transFigure,
               color=INK if down_on else MUTED, fontsize=10.5, ha="center",
               va="center", clip_on=False)

    # ---- BOTTOM: the model series ----------------------------------------
    axSer.set_xlim(0, len(SERIES)); axSer.set_ylim(VMIN, VMAX)
    axSer.set_xticks([]); axSer.set_yticks([])
    for side, sp in axSer.spines.items():
        sp.set_visible(side in ("left", "bottom"))
        sp.set_edgecolor(MUTED); sp.set_linewidth(1.0)
    axSer.text(0.5, 1.16, "SMAP L4 surface soil moisture", transform=axSer.transAxes,
               color=MUTED, fontsize=10.5, ha="center")
    axSer.text(-0.03, 1.0, "wet", transform=axSer.transAxes, color=MUTED,
               fontsize=9.5, ha="right", va="center")
    axSer.text(-0.03, 0.0, "dry", transform=axSer.transAxes, color=MUTED,
               fontsize=9.5, ha="right", va="center")
    axSer.text(0.5, -0.16, "time →", transform=axSer.transAxes, color=MUTED,
               fontsize=10, ha="center")
    if sprog <= 0:
        axSer.text(0.5, 0.5, "extract the pixel value every day",
                   transform=axSer.transAxes, color=MUTED, fontsize=10,
                   ha="center", va="center", alpha=0.5)
    else:
        n = max(2, int(len(SERIES) * sprog))
        axSer.plot(np.arange(n), SERIES[:n], color=LINEC, lw=1.8)
        axSer.scatter([n - 1], [SERIES[n - 1]], s=24, color=LINEC, zorder=5)

    # ---- title + caption -------------------------------------------------
    axMap.text(0.5, 0.955, "Why a chart, not a map?", transform=fig.transFigure,
               color=INK, fontsize=16, fontweight="bold", ha="center",
               va="center", clip_on=False)
    axMap.text(0.5, 0.03, cap, transform=fig.transFigure, color=MUTED,
               fontsize=10.5, ha="center", va="center", clip_on=False)


def update(i):
    draw(*FR[i])
    return []


anim = FuncAnimation(fig, update, frames=len(FR), interval=1000 / 12, blit=False)
gif = ASSETS / "why-no-map.gif"
anim.save(gif, writer=PillowWriter(fps=12), savefig_kwargs={"facecolor": BG})
print(f"wrote {gif}  ({gif.stat().st_size // 1024} KB, {len(FR)} frames)")

# contact sheet — snapshot the real banner at key frames
picks = [8, 30, 46, 66, len(FR) - 20, len(FR) - 1]
shots = []
for fi in picks:
    update(fi)
    fig.canvas.draw()
    shots.append((fi, np.asarray(fig.canvas.buffer_rgba()).copy()))
cs, axes = plt.subplots(1, 6, figsize=(16, 4.6), dpi=80)
for axc, (fi, buf) in zip(axes.ravel(), shots):
    axc.imshow(buf); axc.set_title(f"frame {fi}", fontsize=9); axc.axis("off")
cs.patch.set_facecolor("white"); cs.tight_layout()
sheet = ASSETS / "why-no-map-frames.png"
cs.savefig(sheet, facecolor="white")
print(f"wrote {sheet}")
