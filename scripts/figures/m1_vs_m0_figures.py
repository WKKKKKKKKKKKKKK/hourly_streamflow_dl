"""Figures and summary numbers of daily fine-tuning, M1 against M0.

Writes into reports/m1_vs_m0/:

  distance_and_dissimilarity  (a) median hourly KGE of M0 and M1 against the
                          median distance to the nearest trainable gauge, five CV designs,
                          capped bars spanning the three seeds; (b, c) the same against
                          quintiles of the dissimilarity index on climate and on all
                          continuous static attributes
  gain_robustness_cells   per 4 degree grid cell, whether the gain M1 - M0 is
                          robust over the three seeds
  amount_not_timing       shares of the KGE gain carried by r, alpha and beta, and the
                          reduction of three flood-peak errors
  hydrographs             one gauge per design (scripts/figures/fig_hydrograph_examples.py)
  numbers.json            summary numbers of these analyses (see REPRODUCE.md)

Hourly KGE in distance_and_dissimilarity and gain_robustness_cells is the transfer-stage score of each target gauge
(per_station_hourly_*_target_hourly.csv); amount_not_timing and hydrographs use the continuous hourly series of
every test day (scripts/analysis/full_series_eval.py). Every statistic is computed per seed
and then the median over the three seeds is taken.

    python -m scripts.figures.m1_vs_m0_figures
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# One font for every figure of the section, embedded as TrueType (Type 42);
# the same settings the map scripts use, applied here so that figures drawn before those
# scripts are imported do not fall back to DejaVu Sans and Type 3 fonts.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Liberation Sans", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic", "mathtext.bf": "Nimbus Sans:bold",
    "mathtext.cal": "Nimbus Sans:italic",
})
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

import scripts.figures.plot_style as _gm
from scripts.figures.plot_style import fs

# Print layout: width 177 mm (two columns of 85 mm). Figures are
# drawn at that width so they print at 100 %, with fonts sized for it (fs(x) = x pt).
_gm.FONT_SCALE = 1.0
W = 17.7 / 2.54   # full width, inches

OUT = Path("reports/m1_vs_m0")
GREY, BLUE = "#8d99a6", "#2a78d6"
DESIGNS = [  # key, label, output root, median separation (km) from cv_designs.json
    ("random", "Random", "al_v2_runB"),
    ("blocked", "Blocked", "al_v2_blocked"),
    ("climate", "Climate zone", "al_v2_climate"),
    ("longitudinal", "Longitudinal", "al_v2_longitudinal"),
    ("continental", "Continental", "al_v2_continental"),
]
SEEDS = {"continental": ["al_v2_continental", "var_al_continental_s43", "var_al_continental_s44"]}


def hourly(root: str, model: str) -> pd.Series:
    f = pd.concat([pd.read_csv(p) for p in glob.glob(
        f"outputs/{root}/fold*/transfer/per_station_hourly_fold*_{model}_target_hourly.csv")])
    f = f[(f.score_status == "ok") & (f.obs_std >= 1e-3)]
    return f.set_index("station_id")["kge"]


def paired(root: str) -> pd.DataFrame:
    return pd.concat([hourly(root, "M0"), hourly(root, "M1")], axis=1, keys=["M0", "M1"]).dropna()


def style(ax) -> None:
    ax.grid(alpha=0.18, linewidth=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ---------------------------------------------------------------- KGE against distance: CV designs
def ladder_data(numbers: dict | None = None):
    """Per design: separation (km) and M0/M1/gain of each of the three seeds."""
    sep = {k: v["separation_km"]["median"] for k, v in
           json.load(open("outputs/v2_cv_designs/cv_designs.json"))["designs"].items()}
    xs, lv = [], {}
    for key, label, _ in DESIGNS:
        runs = [paired(r) for r in SEED_RUNS[key]]
        m0 = [float(t.M0.median()) for t in runs]
        m1 = [float(t.M1.median()) for t in runs]
        g = [float((t.M1 - t.M0).median()) for t in runs]
        fr = [float((t.M1 > t.M0).mean()) for t in runs]
        xs.append(sep[key])
        lv[key] = {"M0": m0, "M1": m1, "gain": g}
        if numbers is not None:
            numbers["ladder"][key] = {"n": len(runs[0]), "separation_km": sep[key], "seeds": SEED_RUNS[key],
                                      "M0_per_seed": m0, "M1_per_seed": m1, "paired_gain_per_seed": g,
                                      "frac_improved_per_seed": fr,
                                      "M0": float(np.median(m0)), "M1": float(np.median(m1)),
                                      "paired_gain": float(np.median(g))}
    return xs, lv


# ---------------------------------------------------------------- whole-hydrograph tables
def full_tables() -> dict:
    """design -> full-series tables, one per seed. Uses all three seeds only when every seed's
    full-series evaluation exists for every design; otherwise seed 42 alone."""
    def complete(design, root):   # every fold evaluated, not just the combined file present
        n = 4 if design == "continental" else 5
        return all(Path(f"outputs/{root}/full_series/full_series_fold{k}.csv").exists() for k in range(n))
    have = all(complete(d, r) for d, rs in SEED_RUNS.items() for r in rs)
    out = {}
    for key, label, root in DESIGNS:
        roots = SEED_RUNS[key] if have else [root]
        ok = scored_gauges(root)
        out[key] = [pd.read_csv(f"outputs/{r}/full_series/full_series.csv").pipe(lambda t: t[t.station_id.isin(ok)])
                    for r in roots]
    return out


def scored_gauges(root: str) -> set[str]:
    """The 8,709 gauges every hourly statistic uses: score_status ok and obs_std >= 1e-3."""
    f = pd.concat([pd.read_csv(p) for p in glob.glob(
        f"outputs/{root}/fold*/transfer/per_station_hourly_fold*_M1_target_hourly.csv")])
    return set(f.loc[(f.score_status == "ok") & (f.obs_std >= 1e-3), "station_id"])


LEVEL_METRICS = {  # numbers.json key -> (full-series column, transform applied per gauge)
    "kge_decomposition": [("kge_full", None), ("r", None), ("alpha", None), ("beta", None)],
    "flood_peaks": [("peak_mape", None), ("fhv_pct", np.abs), ("peak_timing_h", None), ("missed_peaks", None)],
}


def full_series_levels(tables, numbers) -> None:
    """Median M0 and M1 of each whole-hydrograph metric and the paired median change, per design.

    Per seed: the median over gauges of M0, of M1 and of the per-gauge change; then the median
    over seeds (seed 42 alone if the other seeds' full-series evaluation is missing).
    """
    for name, cols in LEVEL_METRICS.items():
        numbers[name] = {}
        for col, tf in cols:
            tf = tf or (lambda v: v)
            numbers[name][col] = {}
            for key, _, _ in DESIGNS:
                per_seed = []
                for t in tables[key]:
                    a, b = tf(t[f"M0_{col}"].to_numpy(float)), tf(t[f"M1_{col}"].to_numpy(float))
                    d = (b - a)[np.isfinite(b - a)]
                    per_seed.append((float(np.nanmedian(a)), float(np.nanmedian(b)), float(np.median(d)), a, b))
                ms = []
                for j in (0, 1):
                    if len(per_seed) > 1:
                        ms.append(float(np.median([p[j] for p in per_seed])))
                    else:
                        x = per_seed[0][3 + j]
                        ms.append(float(np.median(x[np.isfinite(x)])))
                numbers[name][col][key] = {"M0": ms[0], "M1": ms[1],
                                           "paired_change": float(np.median([p[2] for p in per_seed])),
                                           "paired_change_per_seed": [p[2] for p in per_seed],
                                           "M0_per_seed": [p[0] for p in per_seed],
                                           "M1_per_seed": [p[1] for p in per_seed]}


# ---------------------------------------------------------------- hydrographs
def plot_hydrographs(numbers) -> None:
    """Hydrograph examples, and their numbers: which gauge, the share of gauges eligible
    (positive M1 KGE among those with >= 3 peaks), and each model's whole-period metrics."""
    import scripts.figures.fig_hydrograph_examples as m
    picks = m.collect_picks("main")
    m.plot(picks, str(OUT / "hydrographs.png"))
    numbers["hydrograph_examples"] = {
        label: {"station_id": p.station_id, "fold": int(p.fold), "gain_quantile": float(p.q),
                "share_of_gauges_eligible": float(p.kept),
                **{f"{tag}_{c}": float(p[f"{tag}_{c}"]) for tag in ("M0", "M1")
                   for c in ("kge_full", "fhv_pct", "peak_timing_h")}}
        for label, _, p in picks}


SEED_RUNS = {  # design -> output roots of seeds 42, 43, 44 (aligned)
    "random": ["al_v2_runB", "var_al_random_s43", "var_al_random_s44"],
    "blocked": ["al_v2_blocked", "var_al_blocked_s43", "var_al_blocked_s44"],
    "climate": ["al_v2_climate", "var_al_climate_s43", "var_al_climate_s44"],
    "longitudinal": ["al_v2_longitudinal", "var_al_longitudinal_s43", "var_al_longitudinal_s44"],
    "continental": ["al_v2_continental", "var_al_continental_s43", "var_al_continental_s44"],
}
DELTA = 0.05   # the practical-equivalence margin used throughout
CLASSES = [  # name, colour -- after the IPCC AR6 Atlas idea of "robust" vs "conflicting" signals
    ("robust gain", "#b2182b"),           # every seed >= +DELTA
    ("gain, size varies", "#f4a582"),     # every seed > 0, not all >= +DELTA
    ("no meaningful change", "#cfd4da"),  # every seed within +/-DELTA
    ("loss in every seed", "#2166ac"),    # every seed < 0 (and not all within +/-DELTA)
    ("seeds conflict", "#35978f"),        # signs disagree between seeds; teal, not orange: orange and
                                          # salmon merge for tritanopes (CAM02-UCS dE 2.7 -> 13.8)
]


def seed_gains(design: str):
    """Per-gauge M1 - M0 for each seed, or None until every fold of every seed is done.

    Each seed repeats pretraining and fine-tuning, so the gain is paired within a seed.
    """
    roots = SEED_RUNS[design]
    n_folds = 4 if design == "continental" else 5
    cols = []
    for r in roots:
        if len(glob.glob(f"outputs/{r}/fold*/transfer/per_station_hourly_fold*_M1_target_hourly.csv")) < n_folds:
            return None
        t = paired(r)
        cols.append((t.M1 - t.M0).rename(r))
    return pd.concat(cols, axis=1).dropna()


def classify(g: pd.DataFrame) -> pd.Series:
    lo, hi = g.min(axis=1), g.max(axis=1)
    cls = pd.Series("seeds conflict", index=g.index)
    cls[lo > 0] = "gain, size varies"
    cls[hi < 0] = "loss in every seed"
    cls[lo >= DELTA] = "robust gain"
    cls[(lo > -DELTA) & (hi < DELTA)] = "no meaningful change"
    return cls


def draw_cells(ax, agg, cls, title, note):
    from matplotlib.patches import Rectangle
    from scripts.figures.plot_style import draw_borders
    draw_borders(ax)
    colour = dict(CLASSES)
    for (lat, lon), c in cls.items():
        ax.add_patch(Rectangle((lon, lat), CELL, CELL, facecolor=colour[c], edgecolor="white", linewidth=0.4,
                               zorder=2))
    ax.set_xlim(-132, 162)
    ax.set_ylim(-46, 72)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_linewidth(0.5)
        side.set_color("#bbbbbb")
    ax.set_title(title, fontsize=fs(8.5), weight="bold", loc="left", pad=3)
    ax.text(0.015, 0.03, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=fs(6.8), color="#555555")


CELL = 4.0       # degrees; grid cells for the gain-robustness map
MIN_CELL = 5     # gauges per cell


def cell_gains(g: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Per grid cell, the median gauge gain of each seed. Gauge-level noise (SD of the gain over
    seeds ~0.10 even under the random design) averages out; regional disagreement does not."""
    f = g.join(coords, how="inner")
    f["clat"], f["clon"] = np.floor(f.lat / CELL) * CELL, np.floor(f.long / CELL) * CELL
    n = f.groupby(["clat", "clon"]).size()
    agg = f.groupby(["clat", "clon"])[list(g.columns)].median()
    agg = agg[n >= MIN_CELL]
    agg["n"] = n[n >= MIN_CELL]
    return agg


def gain_robustness(numbers) -> None:
    """Gain robustness: per grid cell, the class of the gain over the three seeds (CLASSES)."""
    coords = pd.read_csv("folds/folds_blocked.csv").set_index("station_id")[["lat", "long"]]
    ready = [(k, lab) for k, lab, _ in DESIGNS if seed_gains(k) is not None]
    if not ready:
        print("gain_robustness_cells: no design has all three seeds yet")
        return
    numbers["gain_robustness_cells"] = {}
    n = len(ready)
    nrows = int(np.ceil((n + 1) / 2)) if n > 1 else 1
    fig = plt.figure(figsize=(W, W * (0.25 * nrows + 0.06)))
    gs = fig.add_gridspec(nrows, 2, width_ratios=[1, 1] if n > 1 else [2.6, 1],
                          hspace=0.16, wspace=0.06 if n > 1 else 0.22, left=0.02, right=0.98, top=0.97, bottom=0.08)
    shares = {}
    for idx, (key, label) in enumerate(ready):
        g = seed_gains(key)
        med = g.median(axis=0)
        ax = fig.add_subplot(gs[idx // 2, idx % 2] if n > 1 else gs[0, 0])
        title = f"({'abcdefgh'[idx]})  {label}"
        agg = cell_gains(g, coords)
        cls = classify(agg[list(g.columns)])
        draw_cells(ax, agg, cls, title,
                   f"robust gain in {np.mean(cls == 'robust gain'):.0%} of {len(cls)} cells  |  "
                   f"seeds conflict in {np.mean(cls == 'seeds conflict'):.0%}")
        shares[label] = {c: float(np.mean(cls == c)) for c, _ in CLASSES}
        n_located = int(g.index.isin(coords.index).sum())
        numbers["gain_robustness_cells"][key] = {
            "n": int(len(cls)), "shares": shares[label],
            "paired_median_gain_per_seed": [float(v) for v in med],
            "n_gauges_in_cells": int(agg["n"].sum()), "n_gauges": n_located,
            "share_of_gauges_in_cells": float(agg["n"].sum() / n_located)}
    # shares per design
    ax = fig.add_subplot(gs[n // 2, n % 2] if n > 1 else gs[0, 1])
    labels = list(shares)
    left = np.zeros(len(labels))
    for c, col in CLASSES:
        v = np.array([shares[l][c] for l in labels])
        ax.barh(range(len(labels)), v, left=left, color=col, edgecolor="white", linewidth=1.5, height=0.62)
        for i, (x0, w) in enumerate(zip(left, v)):
            if w >= 0.06:
                ax.text(x0 + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=fs(7.4),
                        color="white" if c in ("robust gain", "robust loss") else "#222222")
        left += v
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=fs(8))
    ax.invert_yaxis()
    pos = ax.get_position()     # room for the design names on the left of the bars
    ax.set_position([pos.x0 + 0.09, pos.y0 + 0.02, pos.width - 0.10, pos.height - 0.05])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", labelsize=fs(7.6))
    ax.set_title(f"({'abcdefgh'[n]})  Proportion of grid cells per class", loc="left",
                 fontsize=fs(9), weight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, c in CLASSES], [c for c, _ in CLASSES],
               fontsize=fs(7.8), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.055), ncol=len(CLASSES))
    save(fig, "gain_robustness_cells")




# Plain print style: full frame, inward ticks, no grid or shading, open M0 and filled M1
# markers, and the seed range as capped error bars.
M0_INK = "#5f6b77"


def frame(ax) -> None:
    for side in ax.spines.values():
        side.set_visible(True)
        side.set_linewidth(0.8)
        side.set_color("#222222")
    ax.tick_params(direction="in", top=True, right=True, length=3, width=0.7)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=1.6, width=0.5)
    ax.grid(False)


def plot_pair(ax, x, m0, m1) -> None:
    ax.plot(x, m0, "o-", color=M0_INK, linewidth=1.1, markersize=5, markerfacecolor="white",
            markeredgewidth=1.1, zorder=3, label="$M_0$  before fine-tuning")
    ax.plot(x, m1, "o-", color=BLUE, linewidth=1.3, markersize=5, zorder=3,
            label="$M_1$  after daily-only fine-tuning")


def draw_ladder(ax, xs, lv) -> None:
    keys = [k for k, _, _ in DESIGNS]
    med = {t: np.array([np.median(lv[k][t]) for k in keys]) for t in ("M0", "M1")}
    for tag, c in (("M0", M0_INK), ("M1", BLUE)):
        lo = med[tag] - np.array([min(lv[k][tag]) for k in keys])
        hi = np.array([max(lv[k][tag]) for k in keys]) - med[tag]
        ax.errorbar(xs, med[tag], yerr=[lo, hi], fmt="none", ecolor=c, elinewidth=0.9, capsize=3,
                    capthick=0.9, zorder=2)
    plot_pair(ax, xs, med["M0"], med["M1"])
    # offsets in points; Blocked and Climate zone are 31 km apart on the log axis
    name_dx = {"random": 0, "blocked": -14, "climate": 14, "longitudinal": 0, "continental": 0}
    gain_dx = {"random": 0, "blocked": -14, "climate": 6, "longitudinal": 0, "continental": 0}
    for x, (key, label, _) in zip(xs, DESIGNS):
        ax.annotate(label, (x, max(lv[key]["M1"])), textcoords="offset points", xytext=(name_dx[key], 6),
                    ha="center", va="bottom", fontsize=fs(8), color="#222222")
        g = lv[key]["gain"]
        txt = f"{np.median(g):+.2f}" + (f"\n({min(g):+.2f} to {max(g):+.2f})" if max(g) - min(g) >= 0.02 else "")
        ax.annotate(txt, (x, min(lv[key]["M0"])), textcoords="offset points", xytext=(gain_dx[key], -6),
                    ha="center", va="top", fontsize=fs(7.5), color="#222222")
    ax.axhline(0, color="#999999", linewidth=0.6, linestyle=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(5, 1.4e4)
    ax.set_ylim(-0.62, 0.80)
    ax.set_xlabel("Median distance to the nearest trainable gauge (km)", fontsize=fs(9))
    ax.tick_params(labelsize=fs(8.5))
    ax.legend(fontsize=fs(8.5), frameon=False, loc="lower left")
    frame(ax)


def draw_dissim(ax, t: pd.DataFrame, col: str, title: str, numbers: dict | None = None) -> None:
    """Quintiles of the dissimilarity index pooled over the five designs (8,709 gauges x 5): per
    fifth of the gauges, the median index (x tick label) and the median M0, M1 and gain."""
    t = t.assign(bin=pd.qcut(t[col], 5, labels=False))
    g = t.groupby("bin").agg(x=(col, "median"), M0=("M0", "median"), M1=("M1", "median"),
                             gain=("gain", "median"))
    if numbers is not None:
        edges = pd.qcut(t[col], 5, retbins=True)[1]
        numbers.setdefault("dissimilarity_quintiles", {})[col] = [
            {"lower": float(edges[b]), "upper": float(edges[b + 1]), "n": int((t.bin == b).sum()),
             "median_index": float(r.x), "M0": float(r.M0), "M1": float(r.M1), "gain": float(r.gain),
             "share_M1_below_0": float((t.loc[t.bin == b, "M1"] < 0).mean())}
            for b, r in g.iterrows()]
    pos = np.arange(len(g))
    plot_pair(ax, pos, g.M0, g.M1)
    for x, a_, gn in zip(pos, g.M0, g.gain):
        ax.annotate(f"{gn:+.2f}", (x, a_), textcoords="offset points", xytext=(0, -7),
                    ha="center", va="top", fontsize=fs(7.2), color="#222222")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{x:.3f}" if x < 0.1 else f"{x:.2f}" for x in g.x])
    ax.set_xlim(-0.4, len(g) - 0.6)
    ax.axhline(0, color="#999999", linewidth=0.6, linestyle=":", zorder=1)
    ax.set_title(title, loc="left", fontsize=fs(8.5), weight="bold")
    ax.set_xlabel("Dissimilarity index", fontsize=fs(7.4))
    ax.tick_params(labelsize=fs(7))
    frame(ax)
    ax.tick_params(axis="x", which="both", top=False)   # categorical axis: no ticks along the top


def distance_and_dissimilarity(numbers) -> None:
    """(a) designs against geographic distance, (b, c) quintiles of the dissimilarity
    index (scripts/analysis/dissimilarity_index.py) on climate and on all continuous attributes."""
    numbers["ladder"] = {}
    xs, lv = ladder_data(numbers)
    t = pd.read_csv(OUT / "dissimilarity_per_gauge.csv")
    fig = plt.figure(figsize=(W, W * 0.92))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.38, wspace=0.10)
    ax = fig.add_subplot(gs[0, :])
    draw_ladder(ax, xs, lv)
    ax.set_ylabel("Median hourly KGE", fontsize=fs(8.5))
    ax.set_title("(a)  Cross-validation designs", loc="left", fontsize=fs(8.5), weight="bold")
    b = fig.add_subplot(gs[1, 0])
    c = fig.add_subplot(gs[1, 1], sharey=b)
    draw_dissim(b, t, "DI_clim", "(b)  Dissimilarity, climate attributes", numbers)
    draw_dissim(c, t, "DI_all", "(c)  Dissimilarity, all continuous attributes", numbers)
    b.set_ylabel("Median hourly KGE", fontsize=fs(8))
    b.set_ylim(-0.25, 0.8)
    plt.setp(c.get_yticklabels(), visible=False)
    save(fig, "distance_and_dissimilarity")


def kge_gain_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Exact additive split of a gauge's KGE gain into its r, alpha and beta parts.

    KGE = 1 - ED with ED^2 = (r-1)^2 + (alpha-1)^2 + (beta-1)^2, so
    KGE1 - KGE0 = ED0 - ED1 = (ED0^2 - ED1^2) / (ED0 + ED1) = sum over components of
    [(c0-1)^2 - (c1-1)^2] / (ED0 + ED1). Each part is positive where that component moved
    towards 1; the three parts sum to the gain exactly.
    """
    e = {tag: {c: (df[f"{tag}_{c}"] - 1) ** 2 for c in ("r", "alpha", "beta")} for tag in ("M0", "M1")}
    ed0 = np.sqrt(sum(e["M0"].values()))
    ed1 = np.sqrt(sum(e["M1"].values()))
    out = pd.DataFrame({c: (e["M0"][c] - e["M1"][c]) / (ed0 + ed1) for c in ("r", "alpha", "beta")})
    out["total"] = df["M1_kge_full"] - df["M0_kge_full"]
    return out.replace([np.inf, -np.inf], np.nan).dropna()


PURPLE, ORANGE_, GREEN = "#7f3c8d", "#e68310", "#11a579"   # timing, variability/magnitude, volume


def gain_shares(tables, key) -> tuple[dict, list[dict]]:
    """Median over seeds of the r/alpha/beta shares of the KGE gain.

    Per gauge the gain splits exactly into r, alpha and beta parts (kge_gain_parts). Means are
    additive where medians are not, so the parts are averaged over the central 90 % of gauges by
    gain (KGE is unbounded below), and each is expressed as a share of that mean gain; the
    median shares over seeds are renormalised to sum to one."""
    shares = []
    for df in tables[key]:
        p = kge_gain_parts(df)
        lo, hi = p.total.quantile([0.05, 0.95])
        mu = p[(p.total >= lo) & (p.total <= hi)].mean()
        shares.append({c: float(mu[c] / mu["total"]) for c in ("r", "alpha", "beta")})
    med = {c: float(np.median([sh[c] for sh in shares])) for c in ("r", "alpha", "beta")}
    norm = sum(med.values())
    return {c: v / norm for c, v in med.items()}, shares


def error_reduction(tables, key, col, absolute=False) -> list[float]:
    """Per seed, 100 x (1 - median M1 error / median M0 error) over the stations."""
    out = []
    for df in tables[key]:
        a, b = df[f"M0_{col}"].astype(float), df[f"M1_{col}"].astype(float)
        if absolute:
            a, b = a.abs(), b.abs()
        out.append(100 * (1 - np.nanmedian(b) / np.nanmedian(a)))
    return out


def amount_not_timing(tables, numbers) -> None:
    """What daily fine-tuning corrects: how much water and how strongly it varies, not when it peaks.

    (a) Shares of the KGE gain carried by r, alpha and beta (gain_shares). (b) Reduction of three
    error measures on the continuous hourly series, each as 100 x (1 - median M1 / median M0)
    over the stations; bars are medians over three seeds, lines the seed range. Timing is purple
    in both panels.
    """
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, W * 0.40), sharey=True,
                                 gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.16})
    numbers["amount_not_timing"] = {}
    parts = [("r", "correlation $r$ (timing)", PURPLE), ("alpha", "variability $\\alpha$", ORANGE_),
             ("beta", "bias $\\beta$ (volume)", GREEN)]
    metrics = [("peak_timing_h", "peak timing error", PURPLE, False),
               ("peak_mape", "peak magnitude error", ORANGE_, False),
               ("fhv_pct", "high-flow volume bias |FHV|", GREEN, True)]
    h = 0.25
    for i, (key, label, _) in enumerate(DESIGNS):
        share, _ = gain_shares(tables, key)
        left = 0.0
        for c, name, col in parts:
            ax.barh(i, share[c], left=left, color=col, height=0.62, edgecolor="white", linewidth=1.2)
            if share[c] >= 0.06:
                ax.text(left + share[c] / 2, i, f"{100 * share[c]:.0f}%", ha="center", va="center",
                        fontsize=fs(6.8), color="white")
            left += share[c]
        rec = {"share": share}
        for j, (col, name, colr, ab) in enumerate(metrics):
            v = error_reduction(tables, key, col, ab)
            y = i + (j - 1) * h
            bx.barh(y, np.median(v), height=h * 0.88, color=colr)
            bx.plot([min(v), max(v)], [y, y], color="#333333", linewidth=0.8)
            bx.text(max(max(v), np.median(v)) + 1.2, y, f"{np.median(v):.0f}%", va="center",
                    fontsize=fs(6.4), color="#333333")
            rec[col] = v
        numbers["amount_not_timing"][key] = rec
    ax.set_yticks(range(len(DESIGNS)))
    ax.set_yticklabels([lab for _, lab, _ in DESIGNS], fontsize=fs(7.6))
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Contribution to KGE improvement", fontsize=fs(7.6))
    ax.set_title("(a)  Sources of KGE improvement", loc="left", fontsize=fs(8.5), weight="bold")
    bx.set_xlim(0, 60)
    bx.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(100))
    bx.set_xlabel("Error reduction by fine-tuning", fontsize=fs(7.6))
    bx.set_title("(b)  Flood-peak error reduction", loc="left", fontsize=fs(8.5), weight="bold")
    for a_ in (ax, bx):
        a_.tick_params(axis="x", labelsize=fs(7))
        for side in ("top", "right"):
            a_.spines[side].set_visible(False)
    bx.tick_params(axis="y", length=0)
    # One legend for both panels: the colours mean the same thing on either side.
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in (PURPLE, ORANGE_, GREEN)],
               ["timing: $r$ (a), peak timing error (b)",
                "variability: $\\alpha$ (a), peak magnitude error (b)",
                "volume: $\\beta$ (a), high-flow volume bias |FHV| (b)"],
               fontsize=fs(6.8), frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3,
               columnspacing=1.2, handlelength=1.2)
    save(fig, "amount_not_timing")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    numbers: dict = {}
    distance_and_dissimilarity(numbers)
    gain_robustness(numbers)
    tables = full_tables()
    full_series_levels(tables, numbers)
    amount_not_timing(tables, numbers)
    plot_hydrographs(numbers)
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print("wrote", OUT / "numbers.json")


if __name__ == "__main__":
    main()
