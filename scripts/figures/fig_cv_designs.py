"""One figure for the five cross-validation designs, and what separates them.

The designs differ in one thing only: the spatial structure of what is withheld. A
map per design shows that directly, and the panel that makes them comparable is the
last one, the distance from a held-out gauge to the nearest gauge the model was
allowed to train on. That distance is the axis the whole section turns on, and it
spans nearly three orders of magnitude across the designs.

The climate-zone panel is included precisely because it breaks the ladder. Its
separation sits next to the blocked design's, yet it withholds something no spatial
design does: every gauge of one Koeppen zone, on every continent at once. A reader
who sees only the four spatial designs would take separation in kilometres for the
only axis of difficulty, which is the misreading this panel exists to prevent.

    python -m scripts.figures.fig_cv_designs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.figures.plot_style import FONT_SCALE, draw_borders, fs as _fs

# Six panels on one sheet leaves every label small at the shared print scale, and this
# figure is read on a slide. Lifted here rather than by raising global_map.FONT_SCALE,
# which every other figure also reads.
FONT_BOOST = 1.15


def fs(size: float) -> float:
    return round(_fs(size) * FONT_BOOST, 2)

OUT = Path("reports/figures/cv_designs.png")

# Okabe-Ito, which stays distinguishable in the common colour-vision deficiencies and
# in greyscale. Five folds need five; the continental design uses the first four.
FOLD_COLOURS = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9"]

# The designs are ordered by how far they separate, so the last panel gets a sequential
# ramp rather than the qualitative fold palette. Two different kinds of quantity, two
# different kinds of colour: a reader should never wonder whether the blue line in the
# last panel has anything to do with the blue gauges in the first.
DESIGN_COLOURS = ["#cfd8e0", "#9fb0c0", "#6b8299", "#3d5a75", "#16324f"]

DESIGNS = (
    # file, distance column, short name, what a fold withholds
    ("folds/folds_random.csv", "nearest_other_fold_km_random", "Random",
     "a stratified fifth of the gauges"),
    ("folds/folds_blocked.csv", "nearest_other_fold_km_blocked", "Blocked",
     "whole $k$-means blocks, so a gauge loses its neighbourhood"),
    ("folds/folds_longitudinal.csv", "nearest_other_fold_km_longitudinal", "Longitudinal",
     "a meridional band, so a gauge loses every longitude near it"),
    ("folds/folds_continental.csv", "nearest_other_fold_km_continental", "Continental",
     "a landmass, so a gauge loses every neighbour it has"),
    ("folds/folds_climate.csv", "nearest_other_fold_km_climate", "Climate zone",
     "a Koeppen zone worldwide, so a gauge keeps its neighbours"),
)


def load(path: str, distance_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "lat" not in frame.columns:
        # The random table predates the spatial designs and carries no coordinates;
        # the blocked table covers exactly the same stations and has them.
        coords = pd.read_csv("folds/folds_blocked.csv")[
            ["station_id", "lat", "long", distance_column]]
        frame = frame.merge(coords, on="station_id", how="left")
    return frame


# The densest 8x6 degree window in the network (1,179 gauges, LamaH-CE and CAMELS-DE).
# Every panel carries the same inset over it, because at world scale the random and the
# blocked design are indistinguishable: the difference between them is a property of the
# neighbourhood, and a reader cannot see a neighbourhood in a 280-degree map.
ZOOM = (10.0, 18.0, 45.0, 51.0)


def panel_label(ax, letter):
    """The (a)..(f) a caption refers to. Set above the axes at its left edge, with the
    title pushed right, so neither the letter nor the title lands on the other."""
    ax.text(0.0, 1.030, f"({letter})", transform=ax.transAxes, fontsize=fs(9.5),
            fontweight="bold", va="bottom", ha="left")


def map_panel(ax, frame, title, subtitle, distance_column, letter):
    draw_borders(ax)
    # Draw in a shuffled order so no fold sits systematically on top of another; with
    # 8,990 points at this size, plotting fold by fold would hide whichever came first.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(frame))
    shuffled = frame.iloc[order]
    colours = [FOLD_COLOURS[f % len(FOLD_COLOURS)] for f in shuffled["fold"]]
    ax.scatter(shuffled["long"], shuffled["lat"], s=1.6, c=colours,
               linewidths=0, alpha=0.85, zorder=2)
    # Cropped to where gauges actually are (lon -124..153, lat -39..66). The full
    # 360x140 frame spent two thirds of the panel on empty ocean.
    ax.set_xlim(-132, 162)
    ax.set_ylim(-46, 72)
    ax.set_aspect("equal")
    lo, hi, la, lb = ZOOM
    ax.add_patch(plt.Rectangle((lo, la), hi - lo, lb - la, fill=False,
                               edgecolor="#222222", linewidth=0.7, zorder=5))

    inset = ax.inset_axes([0.015, 0.03, 0.235, 0.46])
    window = frame[(frame["long"].between(lo, hi)) & (frame["lat"].between(la, lb))]
    wshuf = window.iloc[np.random.default_rng(1).permutation(len(window))]
    inset.scatter(wshuf["long"], wshuf["lat"], s=5.0,
                  c=[FOLD_COLOURS[f % len(FOLD_COLOURS)] for f in wshuf["fold"]],
                  linewidths=0, alpha=0.95)
    inset.set_xlim(lo, hi)
    inset.set_ylim(la, lb)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("#fbfbfb")
    for side in inset.spines.values():
        side.set_linewidth(0.7)
        side.set_color("#222222")
    ax.set_xticks([-120, -60, 0, 60, 120])
    ax.set_yticks([-30, 0, 30, 60])
    ax.set_xticklabels([f"{abs(v)}\u00b0{'W' if v < 0 else ('E' if v > 0 else '')}"
                        for v in (-120, -60, 0, 60, 120)])
    ax.set_yticklabels([f"{abs(v)}\u00b0{'S' if v < 0 else ('N' if v > 0 else '')}"
                        for v in (-30, 0, 30, 60)])
    ax.tick_params(labelsize=fs(7), length=2.5, width=0.5, pad=1.5, colors="#666666")
    for side in ax.spines.values():
        side.set_linewidth(0.6)
        side.set_color("#bbbbbb")
    median = frame[distance_column].median()
    n_folds = frame["fold"].nunique()
    # Everything goes in the title: a corner label sat on top of Australia, which is
    # one of the four places this network actually has gauges.
    # "median separation" spelled out five times did not fit once the font was lifted,
    # and panel (f) is where that distance is defined anyway.
    ax.set_title(f"{title}   ·   {n_folds} folds   ·   {median:,.0f} km",
                 fontsize=fs(9), pad=4, loc="left", x=0.115)
    panel_label(ax, letter)


def ladder_panel(ax, frames, letter):
    order = np.argsort([frames[i][DESIGNS[i][1]].median() for i in range(len(DESIGNS))])
    for rank, i in enumerate(order):
        column, title = DESIGNS[i][1], DESIGNS[i][2]
        d = np.sort(frames[i][column].dropna().to_numpy())
        d = np.clip(d, 0.5, None)          # a few coincident gauges sit at 0 km
        y = np.arange(1, d.size + 1) / d.size
        ax.plot(d, y, linewidth=2.0, label=title, color=DESIGN_COLOURS[rank],
                solid_capstyle="round")
    ax.set_xscale("log")
    ax.set_xlim(0.5, 2e4)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Distance to the nearest gauge the model may train on (km)",
                  fontsize=fs(8.5))
    ax.set_ylabel("Fraction of held-out gauges", fontsize=fs(8.5), labelpad=2)
    ax.tick_params(labelsize=fs(7.5))
    ax.grid(True, which="both", linewidth=0.3, color="#dddddd")
    ax.set_axisbelow(True)
    ax.legend(fontsize=fs(7.5), frameon=False, loc="upper left")
    ax.set_title("Separation achieved", fontsize=fs(9), pad=4, loc="left", x=0.115)
    panel_label(ax, letter)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT, type=Path)
    args = parser.parse_args()

    frames = [load(path, column) for path, column, _, _ in DESIGNS]

    fig = plt.figure(figsize=(13.2, 7.9))
    # No suite title and no per-panel gloss: both go in the caption. What is left on the
    # canvas is the legend strip, the five maps and the panel that compares them.
    grid = fig.add_gridspec(3, 2, hspace=0.30, wspace=0.10,
                            left=0.042, right=0.988, top=0.905, bottom=0.075)
    slots = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
    letters = "abcdef"
    for i, ((row, col), (path, column, title, subtitle), frame) in enumerate(
            zip(slots, DESIGNS, frames)):
        map_panel(fig.add_subplot(grid[row, col]), frame, title, subtitle, column,
                  letters[i])
    ladder_panel(fig.add_subplot(grid[2, 1]), frames, letters[len(DESIGNS)])

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=5,
                          color=FOLD_COLOURS[i], label=f"Fold {i}") for i in range(5)]
    # Its own strip above the panels, so it cannot land on the first map's title.
    fig.legend(handles=handles, loc="upper left", ncol=5, frameon=False,
               fontsize=fs(8.5), bbox_to_anchor=(0.042, 0.985), handletextpad=0.25,
               columnspacing=1.6)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"wrote {args.out}")
    for (path, column, title, _), frame in zip(DESIGNS, frames):
        d = frame[column]
        print(f"  {title:14s} folds {frame['fold'].nunique()}  "
              f"median {d.median():8,.0f} km  "
              f"10th {d.quantile(0.10):7,.0f}  90th {d.quantile(0.90):8,.0f}")


if __name__ == "__main__":
    main()
