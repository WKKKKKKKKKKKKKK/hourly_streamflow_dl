"""Generate the report's figures from the result files.

Phase I produced exactly one figure -- the global map -- against 36 in the 100-gauge
experiment, so every quantitative finding here was carried by a table. These are the plots
those findings should have had. Each reads the same CSV the corresponding report table reads,
so a figure cannot disagree with the number beside it.

Design decisions, stated because they are choices:

* Light surface only. These are embedded in a Word document and printed, where there is no
  dark mode to respond to.
* Palette is the validated three-slot categorical set (blue, orange, aqua). Validated
  all-pairs on a white surface: worst CVD separation 9.2 (deutan), worst normal-vision 24.0.
  Aqua sits at 2.82:1 contrast, below the 3:1 bar, so every series carries a direct label --
  identity never rests on colour alone.
* Diverging blue-to-red with a grey midpoint wherever the quantity has a meaningful zero.
* No dual axes anywhere. Where two quantities have different scales they get two panels.

    python -m scripts.make_figures
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED, GREY = "#d03b3b", "#8a8a85"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dedcd6"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlesize": 9.5,
    "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def tidy(ax, grid_axis="both"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def dumbbell(ax, y, x0, x1, color, label=None, size=34):
    """A start->end pair. The line carries the change; the filled dot is the end state.

    The label sits on the far side of the end dot, away from the start, so a rightward and
    a leftward movement both leave it clear -- placing it always to the right put it on top
    of the start marker whenever the value decreased.
    """
    ax.plot([x0, x1], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.scatter([x0], [y], s=size, facecolor="white", edgecolor=color, lw=1.6, zorder=3)
    ax.scatter([x1], [y], s=size, facecolor=color, edgecolor="white", lw=1.0, zorder=3)
    if label:
        right = x1 >= x0
        ax.annotate(label, (x1, y), textcoords="offset points",
                    xytext=(7 if right else -7, 0), ha="left" if right else "right",
                    va="center", fontsize=8, color=INK)


def components(path: Path) -> dict | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    get = lambda c, f: float(frame.loc[frame["component"].eq(c), f].iloc[0])  # noqa: E731
    return {k: {f: get(k, f) for f in ("M0_median", "M1_median")}
            for k in ("kge", "kge_r", "kge_alpha", "kge_beta")}


# ---------------------------------------------------------------- figure 1
def fig_components(out: Path) -> str | None:
    """Where the gain comes from: r barely moves, alpha does."""
    runs = {"Random split": "outputs/v2_runB/diagnostics_allhours",
            "Blocked split": "outputs/v2_blocked/diagnostics_allhours"}
    data = {k: components(Path(v) / "kge_components_summary_target.csv") for k, v in runs.items()}
    data = {k: v for k, v in data.items() if v}
    if not data:
        return None
    labels = [("kge_r", "r  (timing)"), ("kge_alpha", "alpha  (variance)"),
              ("kge_beta", "beta  (water balance)")]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.5), sharex=True, sharey=True)
    for ax, (name, comp) in zip(axes, data.items()):
        for i, (key, lab) in enumerate(labels):
            y = len(labels) - 1 - i
            m0, m1 = comp[key]["M0_median"], comp[key]["M1_median"]
            colour = ORANGE if key == "kge_alpha" else BLUE
            # Annotate the END value, not the change: beta's ideal is 1.0, so a signed
            # delta there reads as "worse" while the value is moving toward correct.
            dumbbell(ax, y, m0, m1, colour, f"{m1:.3f}")
        ax.axvline(1.0, color=GREY, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.set_title(name)
        ax.set_xlim(0.72, 1.18)
        ax.set_xlabel("median across target gauges")
        tidy(ax, "x")
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels([l for _, l in labels][::-1])
    handles = [Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, mew=1.5,
                      ms=6, label="M0  zero-shot"),
               Line2D([], [], marker="o", ls="none", color=INK, ms=6,
                      label="M1  after daily-only fine-tuning"),
               Line2D([], [], color=GREY, lw=0.8, ls=(0, (3, 3)), label="ideal value 1.0")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16),
               fontsize=8, labelcolor=INK)
    fig.suptitle("Daily-aggregate fine-tuning re-calibrates variance, not timing",
                 fontsize=10.5, fontweight="semibold", color=INK, y=1.04)
    fig.text(0.5, -0.30, "Each end is a median across gauges, so the bar length is a "
                         "difference of medians. The stricter paired median difference is "
                         "in the report's component table.",
             ha="center", fontsize=7.2, color=MUTED)
    path = out / "fig01_kge_components.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


# ---------------------------------------------------------------- figure 2
def fig_gain_drivers(out: Path) -> str | None:
    """The gain rises with isolation and falls with catchment area."""
    src = Path("outputs/v2_stratify/stratified_gain_target.csv")
    if not src.exists():
        return None
    strat = pd.read_csv(src)
    dist = strat[strat.variable.str.startswith("nearest_other_fold")].copy()
    area = strat[strat.variable.eq("area_km2")].copy()
    if dist.empty or area.empty:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0))

    # Two series, so they are labelled at their own ends -- placed above the first point and
    # below the last so neither sits on its own line.
    for key, colour, name, anchor in (("random", BLUE, "Random split", "first"),
                                      ("blocked", ORANGE, "Blocked split", "last")):
        sub = dist[dist.variable.str.endswith(key)]
        ax1.plot(sub.covariate_median, sub.gain, marker="o", ms=5, lw=2,
                 color=colour, zorder=3)
        i = 0 if anchor == "first" else -1
        ax1.annotate(name, (sub.covariate_median.iloc[i], sub.gain.iloc[i]),
                     textcoords="offset points",
                     xytext=(0, 11) if anchor == "first" else (0, -15),
                     fontsize=8, color=colour, ha="center", fontweight="semibold")
    ax1.set_xscale("log")
    ax1.set_xlabel("km to nearest trainable neighbour\n(quintile median)")
    ax1.set_ylabel("gain  M1 - M0")
    ax1.set_title("Isolation does not reduce it", fontsize=9)
    ax1.set_ylim(0.030, 0.100)
    tidy(ax1)

    # One series: the title names it, so no legend and no label on the line.
    ax2.plot(area.covariate_median, area.gain, marker="o", ms=5, lw=2, color=AQUA, zorder=3)
    ax2.set_xscale("log")
    ax2.set_xlabel("catchment area, km2\n(quintile median)")
    ax2.set_title("Small catchments gain most (rho -0.17)", fontsize=9)
    tidy(ax2)
    fig.suptitle("What predicts the gain, and what does not",
                 fontsize=10.5, fontweight="semibold", color=INK, y=1.02)
    fig.tight_layout(w_pad=2.4)
    path = out / "fig02_gain_drivers.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


# ---------------------------------------------------------------- figure 3
def fig_configurations(out: Path) -> str | None:
    """v1 -> v2 -> v3, per configuration, as M0 -> M1 movements."""
    def numbers(run: str):
        files = sorted(glob.glob(f"outputs/{run}/fold*/transfer/summary.json"))
        if not files:
            return None
        m0, m1 = [], []
        for f in files:
            j = json.loads(Path(f).read_text())
            for target, key in ((m0, "step1_M0_target_hourly"), (m1, "step2_M1_target_hourly")):
                block = j.get(key)
                if isinstance(block, dict) and block.get("median_kge") is not None:
                    target.append(block["median_kge"])
        return (float(np.mean(m0)), float(np.mean(m1))) if m0 else None

    rows = [("run A  sampled daily", "runA_regwin24", "v2_runA", None),
            ("run B  true daily", "runB_truedaily", "v2_runB", "v3_runB"),
            ("run B  blocked", "runB_blocked", "v2_blocked", "v3_blocked"),
            ("run B  replay 0.25", "runB_replay", "v2_replay025", "v3_replay025")]
    entries = []
    for label, v1, v2, v3 in rows:
        for variant, run, colour in (("v1", v1, BLUE), ("v2", v2, ORANGE), ("v3", v3, AQUA)):
            if run is None:
                continue
            got = numbers(run)
            if got:
                entries.append((f"{label}   {variant}", got[0], got[1], colour))
    if not entries:
        return None

    # Identity lives entirely in the tick label, so nothing has to be annotated inside the
    # plot area and nothing can collide with a mark.
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for i, (label, m0, m1, colour) in enumerate(entries):
        y = len(entries) - 1 - i
        dumbbell(ax, y, m0, m1, colour, f"{m1:.3f}")
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([e[0] for e in entries][::-1], fontsize=8)
    ax.set_xlim(0.36, 0.70)
    ax.set_xlabel("target-domain hourly KGE, median across gauges")
    tidy(ax, "x")
    handles = [Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, mew=1.5, ms=6,
                      label="M0  zero-shot"),
               Line2D([], [], marker="o", ls="none", color=INK, ms=6, label="M1  fine-tuned"),
               Line2D([], [], color=BLUE, lw=2, label="v1  H=72, no forget gate"),
               Line2D([], [], color=ORANGE, lw=2, label="v2  H=336, forget gate 3"),
               Line2D([], [], color=AQUA, lw=2, label="v3  v2 + 50 epochs")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.10),
               fontsize=8, labelcolor=INK)
    ax.set_title("run A moves backwards under both configurations; every run B variant gains",
                 fontsize=10.5, pad=10)
    path = out / "fig03_configurations.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


# ---------------------------------------------------------------- figure 4
def fig_agency_recovery(out: Path) -> str | None:
    """The blocking cost per agency, and how much fine-tuning recovers."""
    src = Path("outputs/v2_split_effect/recovery_by_agency.csv")
    if not src.exists():
        return None
    frame = pd.read_csv(src).sort_values("recovered")
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    for i, (_, r) in enumerate(frame.iterrows()):
        worst = r["recovered"] < 0.5
        colour = RED if worst else BLUE
        dumbbell(ax, i, r["M0"], r["M1"], colour, f'{r["recovered"]:.0%} recovered')
    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels([f'{s}  (n={int(n):,})' for s, n in
                        zip(frame["source"], frame["n_stations"])])
    ax.axvline(0, color=GREY, lw=0.8, zorder=1)
    ax.set_xlabel("paired median KGE drop, blocked minus random")
    ax.set_xlim(frame["M0"].min() - 0.02, 0.055)
    tidy(ax, "x")
    handles = [Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, mew=1.5, ms=6,
                      label="drop at M0  zero-shot"),
               Line2D([], [], marker="o", ls="none", color=INK, ms=6,
                      label="drop at M1  after fine-tuning"),
               Line2D([], [], color=RED, lw=2, label="recovery below 50%")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13),
               fontsize=8, labelcolor=INK)
    ax.set_title("Fine-tuning recovers the blocking cost everywhere except the sparsest network",
                 fontsize=10.5, pad=10)
    path = out / "fig04_agency_recovery.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


# ---------------------------------------------------------------- figure 5
def fig_metric_disagreement(out: Path) -> str | None:
    """KGE improves on more gauges than point-wise error does. This is why."""
    src = Path("outputs/v2_runB/significance/per_station_tests.csv")
    if not src.exists():
        return None
    frame = pd.read_csv(src)
    dk = frame["kge_M1"] - frame["kge_M0"]
    de = frame["median_error_reduction"]
    keep = np.isfinite(dk) & np.isfinite(de)
    dk, de = dk[keep].to_numpy(), de[keep].to_numpy()
    n = len(dk)

    # Quadrant counts use every gauge; the hexbin shows only the window. Points outside are
    # DROPPED rather than clipped -- clipping piled thousands into the corner bins and the
    # colour scale then read the artefact instead of the data.
    xlim, ylim = 0.30, 0.010
    inside = (np.abs(dk) <= xlim) & (np.abs(de) <= ylim)
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    hb = ax.hexbin(dk[inside], de[inside], gridsize=40, cmap="Blues", mincnt=1,
                   bins="log", linewidths=0, zorder=2, extent=(-xlim, xlim, -ylim, ylim))
    cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("gauges per cell (log scale)", fontsize=8, color=MUTED)
    cb.outline.set_visible(False)
    ax.axhline(0, color=INK, lw=1.0, zorder=3)
    ax.axvline(0, color=INK, lw=1.0, zorder=3)
    quads = (("both improve", (dk > 0) & (de > 0), xlim * 0.96, ylim * 0.94, "right", "top"),
             ("KGE up, error worse", (dk > 0) & (de <= 0), xlim * 0.96, -ylim * 0.94, "right", "bottom"),
             ("both worse", (dk <= 0) & (de <= 0), -xlim * 0.96, -ylim * 0.94, "left", "bottom"),
             ("KGE down, error better", (dk <= 0) & (de > 0), -xlim * 0.96, ylim * 0.94, "left", "top"))
    for label, mask, x, y, ha, va in quads:
        count = int(mask.sum())
        ax.annotate(f"{label}\n{count:,}  ({count / n:.0%})", (x, y), ha=ha, va=va,
                    fontsize=7.6, color=INK, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.7, alpha=0.94))
    ax.set_xlabel("change in KGE   (M1 - M0)")
    ax.set_ylabel("reduction in point-wise absolute error  (mm/h)")
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    tidy(ax)
    ax.set_title("The two metrics disagree on a third of gauges", fontsize=10.5, pad=10)
    fig.text(0.5, -0.06,
             f"Quadrant counts use all {n:,} gauges. The density window omits "
             f"{(~inside).sum():,} outside it. Spearman 0.46; the metrics agree on 67.9%.",
             ha="center", fontsize=7.3, color=MUTED)
    path = out / "fig05_metric_disagreement.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


# ---------------------------------------------------------------- figure 6
def fig_convergence(out: Path) -> str | None:
    """v2 stopped on patience, not on the epoch cap -- and it truncated the splits unequally."""
    panels = [("Random split", "v2_runB", "v3_runB"), ("Blocked split", "v2_blocked", "v3_blocked")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    drew = False
    for ax, (name, v2run, v3run) in zip(axes, panels):
        for fold in range(5):
            for run, colour, width, alpha in ((v2run, BLUE, 1.6, 1.0), (v3run, ORANGE, 1.6, 1.0)):
                path = Path(f"outputs/{run}/fold{fold}/pretrain/training_history.csv")
                if not path.exists():
                    continue
                hist = pd.read_csv(path)
                if "val/median_kge" not in hist.columns:
                    continue
                ax.plot(hist["epoch"], hist["val/median_kge"], color=colour, lw=width,
                        alpha=alpha if run == v2run else 0.75, zorder=3 if run == v2run else 2)
                drew = True
                if run == v2run:
                    last = hist.iloc[-1]
                    ax.scatter([last["epoch"]], [last["val/median_kge"]], s=22, color=BLUE,
                               edgecolor="white", lw=0.8, zorder=4)
        ax.axvline(30, color=GREY, lw=0.9, ls=(0, (3, 3)), zorder=1)
        # Zoomed to the plateau: the question is whether the curve was still rising when it
        # stopped, and at the full range (which starts near 0.42) that is invisible.
        ax.set_xlim(12, 51)
        ax.set_ylim(0.595, 0.648)
        ax.annotate("v2 epoch cap", (30, 0.5975), textcoords="offset points",
                    xytext=(4, 0), fontsize=7.4, color=MUTED)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        tidy(ax)
    if not drew:
        plt.close(fig)
        return None
    axes[0].set_ylabel("source validation median KGE")
    handles = [Line2D([], [], color=BLUE, lw=2, label="v2  (30 epochs, patience 6)"),
               Line2D([], [], color=ORANGE, lw=2, label="v3  (50 epochs, patience 10)"),
               Line2D([], [], marker="o", ls="none", color=BLUE, ms=5, label="where v2 stopped")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.14),
               fontsize=8, labelcolor=INK)
    fig.suptitle("Early stopping, not the epoch cap, is what ended v2 -- and it cut the "
                 "blocked split earliest", fontsize=10.5, fontweight="semibold", color=INK,
                 y=1.04)
    fig.text(0.5, -0.22, "Zoomed to the plateau; epochs 1-11 rise from about 0.42 and are "
                         "omitted. Five folds per configuration.",
             ha="center", fontsize=7.3, color=MUTED)
    path = out / "fig06_convergence.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


# ---------------------------------------------------------------- figure 7
def fig_africa_hydrographs(out: Path) -> str | None:
    """What the gain looks like as a hydrograph, on a continent absent from training."""
    base = Path("outputs/v2_africa_insitu_summary")
    m0p, m1p = base / "ensemble_series_M0.csv.gz", base / "ensemble_series_M1.csv.gz"
    per = base / "ensemble_per_basin_M1.csv"
    if not (m0p.exists() and m1p.exists() and per.exists()):
        return None
    m0 = pd.read_csv(m0p, parse_dates=["date"])
    m1 = pd.read_csv(m1p, parse_dates=["date"])
    scored = pd.read_csv(per)

    # Three catchments spanning the outcome, not three good ones: the median gauge by M1
    # KGE and the quartiles either side. Cherry-picking the best would misrepresent it.
    scored = scored.loc[np.isfinite(scored["kge"])].sort_values("kge").reset_index(drop=True)
    picks = [scored.iloc[int(len(scored) * q)] for q in (0.25, 0.5, 0.75)]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.2), sharex=False)
    for ax, row in zip(axes, picks):
        sid = row["station_id"]
        a = m0.loc[m0.station_id.eq(sid)].sort_values("date")
        b = m1.loc[m1.station_id.eq(sid)].sort_values("date")
        if a.empty or b.empty:
            continue
        # One representative window, so the daily shape is visible rather than a smear.
        end = min(a["date"].max(), b["date"].max())
        start = end - pd.Timedelta(days=730)
        a, b = a[a.date.between(start, end)], b[b.date.between(start, end)]
        ax.fill_between(a["date"], 0, a["obs"], color=GREY, alpha=0.28, lw=0, zorder=1)
        ax.plot(a["date"], a["obs"], color=INK, lw=1.1, zorder=4, label="observed")
        ax.plot(a["date"], a["ensemble"], color=BLUE, lw=1.3, zorder=2, label="M0 zero-shot")
        ax.plot(b["date"], b["ensemble"], color=ORANGE, lw=1.3, zorder=3,
                label="M1 after African daily fine-tuning")
        m0kge = float(np.nan_to_num(row.get("kge_M0", np.nan), nan=np.nan)) \
            if "kge_M0" in row else np.nan
        ax.set_title(f'{sid}   M1 KGE {row["kge"]:+.3f}'
                     + (f'   (M0 {m0kge:+.3f})' if np.isfinite(m0kge) else ""),
                     fontsize=8.5, loc="left")
        ax.set_ylabel("mm/d")
        tidy(ax, "y")
    handles = [Line2D([], [], color=INK, lw=1.6, label="observed"),
               Line2D([], [], color=BLUE, lw=1.8, label="M0  zero-shot"),
               Line2D([], [], color=ORANGE, lw=1.8, label="M1  after African daily fine-tuning")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05),
               fontsize=8, labelcolor=INK)
    fig.suptitle("Africa in situ: lower-quartile, median and upper-quartile catchments by M1 KGE",
                 fontsize=10.5, fontweight="semibold", color=INK, y=0.995)
    fig.tight_layout(h_pad=1.6)
    path = out / "fig07_africa_hydrographs.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


# ---------------------------------------------------------------- figure 8
def fig_intraday(out: Path) -> str | None:
    """No degenerate flat-line solution -- and under v2 no over-dispersion either."""
    rows = []
    for label, run in (("v1", "runB_truedaily"), ("v2", "v2_runB")):
        path = Path(f"outputs/{run}/degenerate/degenerate_summary.json")
        if not path.exists():
            continue
        med = json.loads(path.read_text())["medians"]
        for key, name in (("flashiness", "flashiness"),
                          ("intraday_std", "within-day std"),
                          ("intraday_range", "within-day range"),
                          ("q95_events_per_year", "Q95 events / yr"),
                          ("mean", "mean flow")):
            if key not in med or not med[key].get("observed"):
                continue
            obs = med[key]["observed"]
            rows.append({"config": label, "metric": name,
                         "M0": med[key]["M0"] / obs, "M1": med[key]["M1"] / obs})
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    metrics = list(dict.fromkeys(frame["metric"]))
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for i, metric in enumerate(metrics):
        for cfg, colour, offset in (("v1", BLUE, 0.18), ("v2", ORANGE, -0.18)):
            sub = frame[(frame.metric == metric) & (frame.config == cfg)]
            if sub.empty:
                continue
            y = len(metrics) - 1 - i + offset
            dumbbell(ax, y, sub["M0"].iloc[0], sub["M1"].iloc[0], colour,
                     f'{cfg}  {sub["M1"].iloc[0]:.2f}x')
    ax.axvline(1.0, color=GREY, lw=1.0, zorder=1)
    ax.annotate("observed", (1.0, -0.62), textcoords="offset points", xytext=(0, 0),
                fontsize=7.6, color=MUTED, ha="center", annotation_clip=False)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics[::-1])
    ax.set_xscale("log")
    ax.set_xlabel("ratio to the observed median  (log scale)")
    ax.set_xlim(0.35, 12)
    ax.set_xticks([0.5, 1, 2, 4, 8])
    ax.set_xticklabels(["0.5x", "1x", "2x", "4x", "8x"])
    tidy(ax, "x")
    handles = [Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, mew=1.5, ms=6,
                      label="M0  zero-shot"),
               Line2D([], [], marker="o", ls="none", color=INK, ms=6, label="M1  fine-tuned"),
               Line2D([], [], color=BLUE, lw=2, label="v1"),
               Line2D([], [], color=ORANGE, lw=2, label="v2")]
    ax.legend(handles=handles, loc="lower right", fontsize=7.8, labelcolor=INK, ncol=2)
    ax.set_title("v1 was over-dispersed, not flattened; v2 is calibrated before fine-tuning",
                 fontsize=10.5, pad=10)
    path = out / "fig08_intraday_shape.png"
    fig.savefig(path)
    plt.close(fig)
    return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the report's figures.")
    # reports/, not outputs/: these are deliverables, and outputs/ is gitignored,
    # so figures written there would exist only inside the .docx and never on the
    # published branch.
    parser.add_argument("--out-dir", default="reports/figures")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    makers = [fig_components, fig_gain_drivers, fig_configurations, fig_agency_recovery,
              fig_metric_disagreement, fig_convergence, fig_africa_hydrographs, fig_intraday]
    for maker in makers:
        try:
            name = maker(out)
        except Exception as exc:  # noqa: BLE001
            print(f"  {maker.__name__}: FAILED -- {exc}")
            continue
        print(f"  {maker.__name__}: {name or 'skipped, inputs missing'}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
