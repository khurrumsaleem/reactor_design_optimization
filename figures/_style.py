"""Shared Nature-style plotting conventions for the ReactorGen final figures.

Conventions (Nature / Nature Machine Intelligence figure guidelines)
-------------------------------------------------------------------
- Double-column width 183 mm (7.2 in); single column 89 mm (3.5 in).
- Sans-serif (Arial/Helvetica) throughout, 7-8 pt text, 5-7 pt minimum.
- Lower-case bold panel letters outside the top-left corner of each panel.
- No background grid, no top/right spines, thin (0.6-0.7 pt) axes.
- One method = one colour across every figure.  Solid line = CPT+SFT base,
  dashed line = SFT-only base.  Dotted grey = reference value.
- Uncertainty: shaded band = mean +/- two-sided 95 % Student-t interval over
  n = 5 independent seeds (pointwise, after interpolation onto a common grid).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from scipy import stats

WIDTH_SINGLE = 3.50
WIDTH_DOUBLE = 7.20

# --------------------------------------------------------------------------- #
# Colours (Wong colour-blind-safe palette; stable across all figures)         #
# --------------------------------------------------------------------------- #
C = {
    "DPO":        "#0072B2",   # blue
    "DPO_multi":  "#56B4E9",   # sky blue
    "GRPO":       "#D55E00",   # vermillion
    "GRPO_multi": "#E69F00",   # orange
    "GA":         "#009E73",   # bluish green   (unconstrained GA)
    "GA16":       "#CC79A7",   # reddish purple (GA fixed at 16 Gd)
    "RS":         "#4D4D4D",   # dark grey      (informed random search)
    "URS":        "#E69F00",   # orange         (uninformed full-space random search)
    "NU16":       "#8C564B",   # brown          (NuScale-type 16-Gd reference)
    "NU24":       "#9467BD",   # purple         (NuScale-type 24-Gd reference)
    "NOPEN":      "#000000",   # black          (no-penalty control)
    "SFT_ONLY":   "#9E9E9E",   # grey
    "ref":        "#6E6E6E",   # dotted reference lines
    "seed":       "#B5B5B5",   # thin individual-seed traces
}
C["REF16"], C["REF24"] = C["NU16"], C["NU24"]   # depletion-figure aliases

LABEL = {
    "DPO":   "DPO (CPT + SFT)",
    "GRPO":  "GRPO (CPT + SFT)",
    "GA":    "GA (unconstrained)",
    "GA16":  "GA (Gd = 16)",
    "RS":    "Informed random search",
    "URS":   "Uninformed random search",
    "NU16":  "16-Gd symmetric reference",
    "NU24":  "24-Gd symmetric reference",
}

# Lattice (core-map) colours
LAT_FUEL, LAT_FUEL_EDGE = "#E4E4E4", "#C4C4C4"
LAT_GD,   LAT_GD_EDGE   = "#2F5F9E", "#234A7C"
LAT_TUBE_EDGE           = "#8A8A8A"


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.labelpad": 2.5,
        "axes.titlepad": 4.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.1,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #
def mean_ci(values, axis: int = 0, confidence: float = 0.95):
    """Mean and two-sided Student-t interval across seeds (rows)."""
    v = np.asarray(values, dtype=float)
    n = np.sum(~np.isnan(v), axis=axis)
    mean = np.nanmean(v, axis=axis)
    sem = stats.sem(v, axis=axis, nan_policy="omit")
    if hasattr(sem, "filled"):
        sem = sem.filled(np.nan)
    safe_n = np.where(n > 1, n, 2)
    half = stats.t.ppf((1 + confidence) / 2.0, safe_n - 1) * sem
    return mean, mean - half, mean + half


def rolling_mean(arr, window: int):
    s = np.asarray(arr, dtype=float)
    if window <= 1:
        return s
    out = np.full_like(s, np.nan)
    for i in range(len(s)):
        out[i] = np.nanmean(s[max(0, i - window + 1): i + 1])
    return out


# --------------------------------------------------------------------------- #
# Drawing helpers                                                             #
# --------------------------------------------------------------------------- #
def panel_label(ax, label: str, x: float = -0.16, y: float = 1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


def band(ax, x, mat, *, color, ls="-", lw=1.2, alpha=0.16, label=None, zorder=2):
    """Mean line + 95 % CI band from an (n_seeds, n_x) matrix."""
    if mat.size == 0:
        return
    mean, lo, hi = mean_ci(mat, axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=zorder - 1)
    ax.plot(x, mean, color=color, linestyle=ls, linewidth=lw, label=label, zorder=zorder)


def refline(ax, y, text=None, *, color=C["ref"], x_text=0.99, va="bottom", fontsize=6.3, ls=":", ha="right"):
    ax.axhline(y, color=color, linestyle=ls, linewidth=0.8, zorder=0)
    if text:
        ax.annotate(text, xy=(x_text, y), xycoords=ax.get_yaxis_transform(),
                    xytext=(0, 1.5 if va == "bottom" else -1.5), textcoords="offset points",
                    ha=ha, va=va, fontsize=fontsize, color=color, zorder=6)


def dots_with_mean(ax, x, values, *, color, jitter=0.10, ms=4.2, seed=0,
                   mean_width=0.30, show_sd=True, hollow=False, zorder=3):
    """Per-seed points with a horizontal mean bar and +/- s.d. whisker."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    xs = x + rng.uniform(-jitter, jitter, size=len(values)) if len(values) > 1 else np.array([x])
    ax.scatter(xs, values, s=ms ** 2, facecolor="white" if hollow else color,
               edgecolor=color, linewidth=0.8, zorder=zorder)
    if len(values) > 1:
        m, sd = values.mean(), values.std(ddof=1)
        ax.hlines(m, x - mean_width, x + mean_width, color=color, linewidth=1.4, zorder=zorder + 1)
        if show_sd:
            ax.vlines(x + mean_width * 1.15, m - sd, m + sd, color=color, linewidth=0.8, zorder=zorder + 1)


def draw_lattice(ax, grid: np.ndarray, frame_color: str, *, lw_frame=1.3):
    """17 x 17 assembly drawn as pins: grey fuel, blue Gd-bearing fuel,
    hollow rings for water-filled guide tubes."""
    n = grid.shape[0]
    ax.set_xlim(-0.65, n - 0.35)
    ax.set_ylim(n - 0.35, -0.65)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-0.62, -0.62), n + 0.24, n + 0.24, facecolor="white",
                           edgecolor=frame_color, linewidth=lw_frame, zorder=0))
    for i in range(n):
        for j in range(n):
            v = grid[i, j]
            if v == 0:
                ax.add_patch(Circle((j, i), 0.40, facecolor=LAT_FUEL, edgecolor=LAT_FUEL_EDGE, linewidth=0.25))
            elif v == 1:
                ax.add_patch(Circle((j, i), 0.40, facecolor=LAT_GD, edgecolor=LAT_GD_EDGE, linewidth=0.25))
            else:
                ax.add_patch(Circle((j, i), 0.44, facecolor="white", edgecolor=LAT_TUBE_EDGE, linewidth=0.7))


def lattice_legend_handles():
    mk = dict(marker="o", linestyle="none", markersize=6, markeredgewidth=0.6)
    return [
        Line2D([], [], markerfacecolor=LAT_FUEL, markeredgecolor=LAT_FUEL_EDGE, label="UO$_2$ fuel pin", **mk),
        Line2D([], [], markerfacecolor=LAT_GD, markeredgecolor=LAT_GD_EDGE, label="Gd-bearing fuel pin", **mk),
        Line2D([], [], markerfacecolor="white", markeredgecolor=LAT_TUBE_EDGE, label="Guide tube (water)", **mk),
    ]


def ci_proxy(label=r"mean $\pm$ 95% CI ($n$ = 5 seeds)"):
    return Patch(facecolor="#888888", alpha=0.3, edgecolor="none", label=label)


def bottom_legend(fig, handles, *, ncol=None, y=0.0, fontsize=6.8):
    labels = [h.get_label() for h in handles]
    return fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y),
                      ncol=ncol or min(len(handles), 6), frameon=False,
                      fontsize=fontsize, handlelength=1.8, columnspacing=1.3)


def save(fig, stem: str | Path, formats: Sequence[str] = ("pdf", "png")) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(stem.with_suffix(f".{ext}"))
        print(f"  saved -> {stem.with_suffix('.' + ext)}")
    plt.close(fig)


def panel_title(ax, letter: str, title: str, *, x: float = -0.16, fontsize: float = 8.0) -> None:
    """Bold panel letter plus a short left-aligned title (controls / depletion / ED3 figures)."""
    ax.set_title(title, loc="left", pad=9, fontweight="normal", fontsize=fontsize)
    ax.text(x, 1.07, letter, transform=ax.transAxes, weight="bold", fontsize=11, va="bottom")


def table_panel(ax, letter: str, title: str, rows, cols, *, widths=None, fontsize: float = 7.0):
    """Numeric table drawn as a figure panel (replaces a separate manuscript table)."""
    ax.axis("off")
    ax.text(-0.018, 1.06, letter, transform=ax.transAxes, weight="bold", fontsize=11, va="bottom")
    ax.text(0.025, 1.08, title, transform=ax.transAxes, fontsize=8, va="bottom")
    t = ax.table(cellText=rows, colLabels=cols, cellLoc="center", colWidths=widths, bbox=[0, 0, 1, 0.98])
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(0.7)
        cell.set_facecolor("#E8EDF1" if r == 0 else ("#F3F5F6" if r % 2 else "white"))
        if r == 0:
            cell.set_text_props(weight="bold", fontsize=fontsize - 0.1)
            if any("\n" in col for col in cols):
                cell.set_height(cell.get_height() * 1.65)
    return t


def mean_sd(values, dec: int = 2) -> str:
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.{dec}f} ± {v.std(ddof=1):.{dec}f}"
