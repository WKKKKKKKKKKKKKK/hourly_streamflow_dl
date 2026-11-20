"""Distributions over the target gauges of the hourly KGE, |FHV| and the peak-timing error, before
(M0) and after (M1) daily-only fine-tuning, one column per cross-validation design.

Per-gauge whole-test-period metrics of scripts/analysis/full_series_eval.py (full_series.csv):
kge_full (KGE of the continuous hourly series; values below -1 are piled at the left edge),
peak_timing_h (mean absolute timing error of the observed peaks, +/-12 h window) and fhv_pct (bias
of the top 2 % of the flow duration curve). Gauges: the scored target gauges
(m1_vs_m0_figures.scored_gauges) with all three metrics finite for M0 and M1 in all three seeds, the
same gauges under every curve of a column. Line: pointwise median of the three seeds' ECDFs;
band: their envelope; the numbers are the median over gauges, median over seeds.

Writes into reports/m1_vs_m0/:
  magnitude_timing_cdf.png/.pdf
  magnitude_timing_cdf.json   per design and metric: n, the per-seed medians of M0 and M1, the
                              per-seed median paired change, the per-seed shares of gauges with
                              KGE below 0 and with a higher KGE after fine-tuning, and the per-seed
                              shares of gauges whose timing error changes by at most 1 h, improves
                              or worsens by more

    python -m scripts.figures.magnitude_timing_cdf
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scripts.figures.m1_vs_m0_figures import BLUE, DESIGNS, OUT, SEED_RUNS, W, plt, scored_gauges

M0_COLOUR = "#3a3a3a"
INK, MUTED, GRID = "#16202b", "#5a6672", "#e3e7ea"
BASE = 10.0
ROWS = {  # key -> (column of full_series.csv without the model prefix, transform, grid, label, unit)
    "kge": ("kge_full", np.asarray, np.linspace(-1, 1, 801), "Hourly KGE", ""),
    "fhv": ("fhv_pct", np.abs, np.linspace(0, 150, 601), "|FHV| (%)", "%"),
    "timing": ("peak_timing_h", np.asarray, np.linspace(0, 12, 601), "Peak-timing error (h)", "h"),
}


def table(run: str) -> pd.DataFrame:
    t = pd.read_csv(f"outputs/{run}/full_series/full_series.csv")
    return t[t.station_id.isin(scored_gauges(run))].set_index("station_id")


def ecdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    v = np.sort(values)
    return np.searchsorted(v, grid, side="right") / len(v)


NON_CONTINENTAL = ["random", "blocked", "climate", "longitudinal"]


def text_numbers(numbers: dict) -> dict:
    """Summary numbers from the per-seed values: per design the median over the three
    seeds; ranges are over the four non-continental designs. "timing_median_shift" is the
    difference of the plotted medians (M1 - M0, each the median over seeds of the median over
    gauges). Shares are fractions; "quoted" holds them rounded and formatted."""
    med = lambda x: float(np.median(x))
    per = {}
    for design in SEED_RUNS:
        k, f, t = numbers[f"kge/{design}"], numbers[f"fhv/{design}"], numbers[f"timing/{design}"]
        per[design] = {
            "kge_M0": med(k["M0"]), "kge_M1": med(k["M1"]),
            "kge_below_0_M0": med(k["share_below_0"]["M0"]), "kge_below_0_M1": med(k["share_below_0"]["M1"]),
            "kge_improved": med(k["share_improved"]),
            "abs_fhv_M0": med(f["M0"]), "abs_fhv_M1": med(f["M1"]),
            "timing_M0": med(t["M0"]), "timing_M1": med(t["M1"]),
            "timing_median_shift": med(t["M1"]) - med(t["M0"]),
            "timing_within_1h": med(t["share_within_1h"]),
            "timing_better_1h": med(t["share_better_by_more_than_1h"]),
            "timing_worse_1h": med(t["share_worse_by_more_than_1h"])}
    rng = lambda key: (min(per[d][key] for d in NON_CONTINENTAL), max(per[d][key] for d in NON_CONTINENTAL))
    every = lambda key: (min(per[d][key] for d in SEED_RUNS), max(per[d][key] for d in SEED_RUNS))
    f2 = lambda lo, hi: f"{lo:.2f}-{hi:.2f}"
    pc = lambda lo, hi: f"{100 * lo:.0f}-{100 * hi:.0f} %"
    c = per["continental"]
    quoted = {
        "kge_median_M0_noncontinental": f2(*rng("kge_M0")),
        "kge_median_M1_noncontinental": f2(*rng("kge_M1")),
        "kge_improved_noncontinental": pc(*rng("kge_improved")),
        "kge_below_0_M0_noncontinental": pc(*rng("kge_below_0_M0")),
        "kge_below_0_M1_noncontinental": pc(*rng("kge_below_0_M1")),
        "kge_median_continental": f"{c['kge_M0']:.2f} -> {c['kge_M1']:.2f}",
        "kge_below_0_M1_continental": f"{100 * c['kge_below_0_M1']:.0f} %",
        "abs_fhv_median_M0_noncontinental": f"{rng('abs_fhv_M0')[0]:.0f}-{rng('abs_fhv_M0')[1]:.0f} %",
        "abs_fhv_median_M1_noncontinental": f"{rng('abs_fhv_M1')[0]:.0f}-{rng('abs_fhv_M1')[1]:.0f} %",
        "abs_fhv_median_continental": f"{c['abs_fhv_M0']:.0f} -> {c['abs_fhv_M1']:.0f} %",
        "timing_largest_median_shift_h": f"{max(abs(per[d]['timing_median_shift']) for d in SEED_RUNS):.2f}",
        "timing_median_M1_all_designs_h": f"{every('timing_M1')[0]:.1f}-{every('timing_M1')[1]:.1f}",
        "timing_within_1h_all_designs": pc(*every("timing_within_1h")),
        "timing_better_exceeds_worse_in_every_design": all(per[d]["timing_better_1h"] > per[d]["timing_worse_1h"]
                                                           for d in SEED_RUNS),
    }
    return {"per_design": per, "quoted": quoted}


def main() -> None:
    labels = {k: lab for k, lab, _ in DESIGNS}
    sep = {k: v["separation_km"]["median"] for k, v in
           json.load(open("outputs/v2_cv_designs/cv_designs.json"))["designs"].items()}
    fig = plt.figure(figsize=(W, W * 0.92))
    gap, left0 = 0.013, 0.098
    width = (0.988 - left0 - 4 * gap) / 5
    ypos, height = {"kge": 0.705, "fhv": 0.410, "timing": 0.115}, 0.225
    numbers = {}
    for i, (design, runs) in enumerate(SEED_RUNS.items()):
        tabs = [table(r) for r in runs]
        cols = [f"{m}_{ROWS[k][0]}" for k in ROWS for m in ("M0", "M1")]
        keep = tabs[0].index
        for t in tabs:
            keep = keep.intersection(t.dropna(subset=cols).index)
        for row, (col, f, grid, label, unit) in ROWS.items():
            ax = fig.add_axes([left0 + i * (width + gap), ypos[row], width, height])
            ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)
            rec = {"n": len(keep)}
            for m, colour, lw, dash in (("M0", M0_COLOUR, 1.0, (0, (3.2, 2.0))), ("M1", BLUE, 1.3, "-")):
                vals = [f(t.loc[keep, f"{m}_{col}"].to_numpy(float)) for t in tabs]
                curves = np.array([ecdf(v, grid) for v in vals])
                ax.fill_between(grid, curves.min(0), curves.max(0), color=colour, alpha=0.18, linewidth=0, zorder=2)
                ax.plot(grid, np.median(curves, 0), color=colour, linewidth=lw, linestyle=dash, zorder=3)
                rec[m] = [float(np.median(v)) for v in vals]
            d = [f(t.loc[keep, f"M1_{col}"].to_numpy(float)) - f(t.loc[keep, f"M0_{col}"].to_numpy(float)) for t in tabs]
            rec["paired_change"] = [float(np.median(x)) for x in d]
            if row == "kge":
                rec["share_below_0"] = {m: [float((t.loc[keep, f"{m}_{col}"] < 0).mean()) for t in tabs] for m in ("M0", "M1")}
                rec["share_improved"] = [float((x > 0).mean()) for x in d]
            if row == "timing":
                rec["share_within_1h"] = [float((np.abs(x) <= 1).mean()) for x in d]
                rec["share_better_by_more_than_1h"] = [float((x < -1).mean()) for x in d]
                rec["share_worse_by_more_than_1h"] = [float((x > 1).mean()) for x in d]
            numbers[f"{row}/{design}"] = rec
            fmt = {"kge": lambda x: f"{x:.2f}".replace("-", "\u2212"), "timing": lambda x: f"{x:.1f} {unit}",
                   "fhv": lambda x: f"{x:.0f}{unit}"}[row]
            ax.text(0.97, 0.30, fmt(np.median(rec["M0"])), transform=ax.transAxes, ha="right", va="top",
                    fontsize=BASE - 2.4, color=M0_COLOUR)
            ax.text(0.97, 0.19, fmt(np.median(rec["M1"])), transform=ax.transAxes, ha="right", va="top",
                    fontsize=BASE - 2.4, color=BLUE)
            ax.set_xlim(grid[0], grid[-1])
            ax.set_ylim(0, 1)
            ticks = {"kge": [-1, -0.5, 0, 0.5, 1], "timing": [0, 3, 6, 9, 12], "fhv": [0, 25, 50, 75, 100, 125, 150]}[row]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}".replace("-", "\u2212") if j % 2 == 0 else "" for j, t in enumerate(ticks)],
                               fontsize=BASE - 2.0)
            xt = ax.get_xticklabels()
            xt[0].set_horizontalalignment("left")
            xt[-1].set_horizontalalignment("right")
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            if i == 0:
                ax.set_yticklabels(["0", "", "0.5", "", "1"], fontsize=BASE - 2.0)
                ax.set_ylabel("Fraction of gauges", fontsize=BASE - 1.8, color=INK)
            else:
                ax.set_yticklabels([])
            ax.tick_params(length=2, pad=1.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
                ax.spines[side].set_linewidth(0.6)
            if row == "kge":
                ax.set_title(f"{labels[design]}\n{sep[design]:,.0f} km", fontsize=BASE - 1.6,
                             fontweight="bold", color=INK, loc="left", pad=3, linespacing=1.3)
            letter = "abcdefghijklmno"[i + 5 * list(ROWS).index(row)]
            ax.text(0.04, 0.96, f"({letter})", transform=ax.transAxes, ha="left", va="top",
                    fontsize=BASE - 1.6, fontweight="bold", color=INK)
            if i == 2:
                ax.text(0.5, -0.19, label, transform=ax.transAxes, ha="center", va="top",
                        fontsize=BASE - 1.6, color=INK)
    handles = [Line2D([], [], color=M0_COLOUR, linewidth=1.0, linestyle=(0, (3.2, 2.0)), label="$M_0$  pretrained"),
               Line2D([], [], color=BLUE, linewidth=1.3, label="$M_1$  after daily-only fine-tuning")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=2, fontsize=BASE - 2.0,
               frameon=False, handlelength=1.8, columnspacing=2.2, borderpad=0.0)
    fig.savefig(OUT / "magnitude_timing_cdf.png", dpi=300)
    fig.savefig(OUT / "magnitude_timing_cdf.pdf")
    plt.close(fig)
    numbers["text"] = text_numbers(numbers)
    (OUT / "magnitude_timing_cdf.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers["text"]["quoted"], indent=1))
    print("wrote", OUT / "magnitude_timing_cdf.png")
    for k, v in numbers.items():
        if "share_below_0" in v:
            print(f"{k:22s} KGE<0 M0 {np.median(v['share_below_0']['M0']):.1%} -> M1 {np.median(v['share_below_0']['M1']):.1%} | M1>M0 {np.median(v['share_improved']):.0%}")
        extra = (f" | within 1 h {np.median(v['share_within_1h']):.0%}, better >1 h "
                 f"{np.median(v['share_better_by_more_than_1h']):.0%}, worse >1 h "
                 f"{np.median(v['share_worse_by_more_than_1h']):.0%}") if "share_within_1h" in v else ""
        print(f"{k:22s} n={v['n']} M0 {np.median(v['M0']):.2f} -> M1 {np.median(v['M1']):.2f} "
              f"(paired {np.median(v['paired_change']):+.2f}){extra}")


if __name__ == "__main__":
    main()
