"""Data charts of Figure 1 rendered with matplotlib at exact print size.

Text inside these charts (7 pt labels, 6 pt tick numbers and notes, Arial)
is baked into the image so that row labels, markers and tick marks align
identically in PowerPoint, LibreOffice and the PDF. The placing script
(make_fig1.js) puts each PNG at the same print size it was drawn at.

Outputs (assets/):
  chart_kstrip.png       corpus k_eff distribution against the target      1.42 x 0.56 in
  chart_fields.png       OpenMC thermal flux and pin fission rate squares   2.05 x 1.42 in
  chart_attribution.png  inventory-axis attribution plot (panel c, left)    4.60 x 1.80 in
  chart_right.png        best objective, chemistry probability and error   2.15 x 1.80 in
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon, Rectangle
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
A = HERE / "assets"
D = json.load(open(HERE / "data/fig1_assets.json"))
SUPP = HERE / "data/supervised_generation_summary.json"
SUP = json.load(open(SUPP)) if SUPP.exists() else None

for p in ["/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
          "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Italic.ttf"]:
    if Path(p).exists():
        font_manager.fontManager.addfont(p)
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "savefig.dpi": 600, "savefig.transparent": True})
INK, LINE, BAR, DARK, GD = "#1F2328", "#3F4650", "#C9CED3", "#4A5057", "#0072B2"
SMALL = 6


def save(fig, name):
    fig.savefig(A / name, dpi=600, transparent=True)
    plt.close(fig)


def clean(ax, keep=("bottom",)):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=INK, labelcolor=INK)


bar = tqdm(total=4, desc="charts", ncols=70)

# ================================================================ 1. corpus k_eff strip
ck = D["corpus_k"]
fig = plt.figure(figsize=(1.42, 0.70))
ax = fig.add_axes([0.06, 0.24, 0.80, 0.50])
edges, counts = np.array(ck["edges"]), np.array(ck["counts"])
ax.bar(edges[:-1], counts / counts.max(), width=np.diff(edges), align="edge", color=BAR, linewidth=0)
ax.axvline(1.05, 0, 1.0, color=INK, lw=0.9)
ax.annotate("", xy=(ck["min"] - 0.001, 0.42), xytext=(1.052, 0.42),
            arrowprops=dict(arrowstyle="<|-|>", color=LINE, lw=0.5, mutation_scale=4, shrinkA=0, shrinkB=0))
ax.text((1.05 + ck["min"]) / 2, 0.52, f"{ck['gap_pcm']:,} pcm", ha="center", va="bottom", fontsize=7, color=INK)
ax.text(1.046, 1.10, "target", ha="left", va="bottom", fontsize=7, color=INK, clip_on=False)
ax.text(1.222, 1.10, "corpus, 10,000", ha="right", va="bottom", fontsize=7, color=INK, clip_on=False)
ax.set_xlim(1.04, 1.22); ax.set_ylim(0, 1.0)
ax.set_xticks([1.05, 1.10, 1.15, 1.20]); ax.set_yticks([])
clean(ax)
ax.text(1.226, 0.0, r"$k$", ha="left", va="bottom", fontsize=SMALL, color=INK)
save(fig, "chart_kstrip.png"); bar.update(1)

# ================================================================ 2. OpenMC fields, two squares
F = np.load(ROOT / "claude_figure/revised_nature_3/fields_dpo_best_hi.npz")
thermal = gaussian_filter(F["thermal"], sigma=0.8)
thermal_rel = thermal / thermal.max()
power = F["power"]
pin_rel = power / power.mean()                      # mean over all 289 cells, as in F_q
grid = str(F["grid"])
fig = plt.figure(figsize=(2.05, 1.56))
sq, y0 = 0.95 / 2.05, 0.42 / 1.56
for k, (title, cmap) in enumerate([("thermal flux", "magma"), ("pin fission rate", "viridis")]):
    ax = fig.add_axes([0.02 + k * 0.52, y0, sq, 0.95 / 1.56])
    if k == 0:
        ax.imshow(thermal_rel, cmap=cmap, vmin=0, vmax=1, origin="lower", interpolation="bilinear",
                  extent=(-8.5, 8.5, -8.5, 8.5))
        vmax, hi = 1.0, "1"
    else:
        vmax, hi = float(np.ceil(pin_rel.max() * 100) / 100), f"{pin_rel.max():.2f}"
        ax.set_facecolor("none")
        for i in range(17):
            for j in range(17):
                cx, cy = j - 8.5 + 0.5, 8.5 - i - 0.5
                if grid[i * 17 + j] == "c":
                    ax.add_patch(plt.Circle((cx, cy), 0.44, fc="white", ec=LINE, lw=0.5))
                else:
                    ax.add_patch(plt.Circle((cx, cy), 0.42, fc=matplotlib.colormaps[cmap](pin_rel[16 - i, j] / vmax), ec="none"))
        ax.set_xlim(-8.5, 8.5); ax.set_ylim(-8.5, 8.5); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(LINE); s.set_linewidth(0.5)
    ax.set_title(title, fontsize=7, color=INK, pad=2)
    cax = fig.add_axes([0.02 + k * 0.52 + 0.10 * sq, y0 - 0.12, sq * (0.74 if k == 0 else 0.66), 0.04])
    cax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap, aspect="auto")
    cax.set_xticks([]); cax.set_yticks([])
    for s in cax.spines.values():
        s.set_color(LINE); s.set_linewidth(0.4)
    fig.text(0.02 + k * 0.52 + 0.08 * sq, y0 - 0.10, "0", ha="right", va="center", fontsize=SMALL, color=INK)
    fig.text(0.02 + k * 0.52 + (0.86 if k == 0 else 0.78) * sq, y0 - 0.10, hi, ha="left", va="center", fontsize=SMALL, color=INK)
    fig.text(0.02 + k * 0.52 + 0.5 * sq, y0 - 0.165, "relative to maximum" if k == 0 else "relative to mean", ha="center", va="top", fontsize=SMALL, color=INK)
save(fig, "chart_fields.png"); bar.update(1)
json.dump({"pin_rel_max": float(pin_rel.max()), "k": float(F["k"]), "kstd": float(F["kstd"])}, open(HERE / "data/fields_chart.json", "w"), indent=1)

# ================================================================ 3. attribution on the inventory axis
hist = {int(k): v for k, v in D["policy_inventory"]["hist"].items()}
n = D["policy_inventory"]["n"]
maxc = max(v for k, v in hist.items() if k <= 64)
dens = lambda c, nn: (c / nn) / (maxc / n)          # 1.0 = tallest aligned bar
fig = plt.figure(figsize=(4.60, 2.05))
W, H = 4.60, 2.05
# main axis occupies x 0.30-4.10 in for inventory 0-64, break, then 264 at 4.50
axL = fig.add_axes([0.30 / W, 0.58 / H, 3.80 / W, 1.05 / H])
axR = fig.add_axes([4.20 / W, 0.58 / H, 0.30 / W, 1.05 / H])
ymax = 1.45
for ax in (axL, axR):
    ax.set_ylim(0, ymax); ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=INK, labelcolor=INK)
axL.set_xlim(-1, 65); axR.set_xlim(240, 266)
axL.set_xticks([0, 16, 32, 48, 64]); axR.set_xticks([264])
# aligned policies
xs = np.arange(0, 65)
axL.bar(xs, [dens(hist.get(v, 0), n) for v in xs], width=1.0, color=BAR, linewidth=0, zorder=2)
# uniform density
hU = dens(n / 265, n)
for ax in (axL, axR):
    ax.axhline(hU, color=LINE, lw=0.7, ls=(0, (3, 2)), zorder=3)
# axis break marks
for ax, x in ((axL, 65), (axR, 240)):
    ax.plot([x - 0.6, x + 0.6], [-0.05, 0.05], color=LINE, lw=0.6, clip_on=False, transform=ax.transData)
# supervised policies (outlined, clipped)
if SUP:
    clipH = 1.22
    sh = {int(k): v for k, v in SUP["hist"].items()}
    for v, c in sh.items():
        if v > 64:
            continue
        full = dens(c, SUP["n"]); h = min(full, clipH)
        axL.add_patch(Rectangle((v - 0.5, 0), 1.0, h, fc="white", ec=INK, lw=0.6, zorder=4))
        if full > clipH:
            axL.plot([v - 0.9, v + 0.9], [clipH - 0.17, clipH - 0.10], color="white", lw=1.6, zorder=5)
            axL.plot([v - 0.9, v + 0.9], [clipH - 0.14, clipH - 0.07], color="white", lw=1.6, zorder=5)
            axL.plot([v - 0.9, v + 0.9], [clipH - 0.155, clipH - 0.085], color=INK, lw=0.4, zorder=6)
    axL.add_patch(Polygon([[16 - 0.8, 0.0], [16 + 0.8, 0.0], [16, 0.09]], closed=True, fc=INK, ec=INK, zorder=7))
    axL.text(13.3, 0.70, "supervised policies", ha="right", va="center", fontsize=7, color=INK)
    axL.text(13.3, 0.57, f"{100 * SUP['exactly16']:.0f} % at 16", ha="right", va="center", fontsize=7, color=INK)
    axL.text(13.3, 0.44, "corpus, all at 16", ha="right", va="center", fontsize=7, color=INK)
else:
    axL.plot([16, 16], [0, clipH := 1.22], color=INK, lw=2.2, zorder=4)
    axL.text(13.3, 0.57, "corpus, all at 16", ha="right", va="center", fontsize=7, color=INK)
# objective removal
axL.annotate("", xy=(0.3, 1.02), xytext=(14.4, 1.02), arrowprops=dict(arrowstyle="-|>", color=LINE, lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0))
axL.text(0.0, 1.17, "target term removed", ha="left", va="bottom", fontsize=7, color=INK, transform=axL.transAxes, clip_on=False)
axL.text(0.0, 1.05, "4 of 5 runs → 0", ha="left", va="bottom", fontsize=7, color=INK, transform=axL.transAxes, clip_on=False)
# labels
axL.text(41.2, 0.86, "Redistribution", ha="left", va="center", fontsize=7, fontweight="bold", color=INK)
axL.text(41.2, 0.73, "aligned policies", ha="left", va="center", fontsize=7, color=INK)
axR.text(265.5, hU + 0.36, "Availability", ha="right", va="center", fontsize=7, fontweight="bold", color=INK)
axR.text(265.5, hU + 0.22, "uniform, 0–264", ha="right", va="center", fontsize=7, color=INK)
# selected designs bracket
axL.plot([27.5, 27.5, 35.5, 35.5], [1.18, 1.24, 1.24, 1.18], color=LINE, lw=0.6, zorder=6)
axL.text(31.5, 1.28, "selected designs, 28–35", ha="center", va="bottom", fontsize=7, color=INK)
# informed bracket below the axis
axL.plot([19.5, 19.5, 40.5, 40.5], [-0.30, -0.36, -0.36, -0.30], color=LINE, lw=0.6, clip_on=False, zorder=6)
axL.text(30, -0.42, "Information value", ha="center", va="top", fontsize=7, fontweight="bold", color=INK, clip_on=False)
axL.text(30, -0.57, "informed random, 20–40", ha="center", va="top", fontsize=7, color=INK, clip_on=False)
axR.text(266, -0.42, "absorber inventory", ha="right", va="top", fontsize=7, color=INK, clip_on=False)
save(fig, "chart_attribution.png"); bar.update(1)

# ================================================================ 4. right column: best objective, chemistry
fig = plt.figure(figsize=(2.15, 2.05))
W, H = 2.15, 2.05
# best objective dot plot
ax = fig.add_axes([1.00 / W, 1.28 / H, 1.05 / W, 0.52 / H])
rows = [("uniform, 0–264", "uniform", "white"), ("informed, 20–40", "informed", "white"), ("DPO", "dpo", INK), ("GRPO", "grpo", INK)]
for r, (lab, key, fc) in enumerate(rows):
    b = D["best_objective"][key]; y = 3 - r
    ax.plot([b["mean"] - b["sd"], b["mean"] + b["sd"]], [y, y], color=LINE, lw=0.7, zorder=2)
    ax.plot(b["mean"], y, marker="o", ms=4.2, mfc=fc, mec=INK, mew=0.5, zorder=3)
    ax.text(1.535, y, lab, ha="right", va="center", fontsize=7, color=INK, clip_on=False)
ax.set_xlim(1.55, 1.85); ax.set_ylim(-0.6, 3.6)
ax.set_xticks([1.6, 1.7, 1.8]); ax.set_yticks([])
clean(ax)
fig.text(0.0, 1.95 / H, "best objective, 2,000 evaluations", ha="left", va="top", fontsize=7, color=INK)
fig.text(2.05 / W, 1.10 / H, "lower is better", ha="right", va="top", fontsize=SMALL, color=INK)
# chemistry: probability and error
chem = D["chem"]
rows2 = [("supervised", "sft_base", BAR), ("DPO", "dpo", DARK), ("GRPO", "grpo", DARK)]
for k, (title, key, scale, ticks, xoff) in enumerate([("off-corpus\nprobability", "off_corpus", 1.0, [0, 0.5, 1], 0.62),
                                                       ("expected\nyield error, %", "error", 60.0, [0, 30, 60], 1.45)]):
    axc = fig.add_axes([xoff / W, 0.34 / H, 0.60 / W, 0.32 / H])
    for r, (lab, kk, col) in enumerate(rows2):
        y = 2 - r
        axc.barh(y, chem[key][kk], height=0.72, color=col, linewidth=0, zorder=2)
        if k == 0:
            axc.text(-0.06, y, lab, ha="right", va="center", fontsize=7, color=INK, clip_on=False)
    axc.axvline(chem[key]["random_full"], color=LINE, lw=0.5, ls=(0, (3, 2)), zorder=3)
    axc.set_xlim(0, scale); axc.set_ylim(-0.55, 2.55)
    axc.set_xticks(ticks); axc.set_xticklabels([f"{t:g}" for t in ticks]); axc.set_yticks([])
    clean(axc, keep=("bottom", "left"))
    axc.set_title(title, fontsize=7, color=INK, pad=2, linespacing=1.1)
fig.text(0.0, 0.04 / H, "dashed, full-catalog uniform", ha="left", va="bottom", fontsize=SMALL, color=INK)
save(fig, "chart_right.png"); bar.update(1)
bar.close()
print("charts written")
