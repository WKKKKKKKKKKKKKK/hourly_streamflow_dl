"""Figure and numbers of Appendix C (source-domain forgetting and replay).

Writes into reports/sec42_forgetting_replay/:

  fig2_cdf_replay         Fig. C1: distributions of hourly KGE over gauges, source row and target
                          row, one column per CV design, for M0, M1 fine-tuned without replay and
                          M1 fine-tuned with source replay (ratio 0.25). The line is the pointwise
                          median of the three seeds' ECDFs (it crosses 0.5 at the median of the
                          seed medians), the band their envelope. A source gauge carries one
                          value, the median over the folds in which it is a source gauge.
  fig2_text_numbers.csv   every number of the appendix text on the basis of the figure: levels,
                          share of KGE < 0, |ln alpha|, |ln beta|, |alpha - 1|, |beta - 1|, r, the
                          paired per-gauge changes and the share of gauges that improve; per seed
                          (fig2_text_numbers_per_seed.csv), then median and range over seeds
  numbers.json            the medians annotated on the figure

Source scores are the source gauges' own test period, re-evaluated hourly with M0 and with the
fine-tuned M1 of the same fold. Every replay run starts from the pretrained weights of the plain
run of the same seed (M0 is identical), so replay against no replay is paired within a seed. A
seed counts only when all its folds are done. Per-continent numbers of the continental design:
scripts/numbers/sec42_continental_folds.py.

    python -m scripts.figures.sec42_figures
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.figures.sec41_figures import BLUE, DESIGNS, ORANGE_, SEED_RUNS, W, fs, plt, style

OUT = Path("reports/sec42_forgetting_replay")
REPLAY_RUNS = {  # design -> output roots of the replay runs (ratio 0.25), seeds 42, 43, 44
    "random": ["al_v2_replay025", "var_al_random_replay025_s43", "var_al_random_replay025_s44"],
    "blocked": ["al_v2_blocked_replay025", "var_al_blocked_replay025_s43", "var_al_blocked_replay025_s44"],
    "climate": ["al_v2_climate_replay025", "var_al_climate_replay025_s43", "var_al_climate_replay025_s44"],
    "longitudinal": ["al_v2_longitudinal_replay025", "var_al_longitudinal_replay025_s43",
                     "var_al_longitudinal_replay025_s44"],
    "continental": ["al_v2_continental_replay025", "var_al_continental_replay025_s43",
                    "var_al_continental_replay025_s44"],
}
N_FOLDS = {"continental": 4}


def complete(root: str, design: str) -> bool:
    return len(glob.glob(f"outputs/{root}/fold*/transfer/DONE")) == N_FOLDS.get(design, 5)


def scores(root: str, model: str, domain: str) -> pd.Series:
    """Hourly KGE per (fold, gauge) for one run, model (M0/M1) and domain (source/target)."""
    out = []
    for p in sorted(glob.glob(f"outputs/{root}/fold*/transfer/per_station_hourly_fold*_{model}_{domain}_hourly.csv")):
        f = pd.read_csv(p)
        f = f[(f.score_status == "ok") & (f.obs_std >= 1e-3)]
        f["fold"] = int(p.split("/fold")[1].split("/")[0])
        out.append(f.set_index(["fold", "station_id"]).kge)
    return pd.concat(out)


def signed(x: float) -> str:
    return f"{x:+.2f}".replace("-", "\u2212")


# ---------------------------------------------------------------- Fig. C1: CDFs
LABEL = {k: lab for k, lab, _ in DESIGNS}
# colours as Sect. 3 (M1 blue, replay orange)
REFERENCE, PLAIN, REPLAY = "#3a3a3a", BLUE, ORANGE_
INK, MUTED, GRID = "#16202b", "#5a6672", "#e3e7ea"
BASE = 10.0
XGRID = np.linspace(-1.0, 1.0, 801)


def per_gauge(root: str, model: str, domain: str) -> pd.Series:
    """One value per gauge: a target gauge has one fold; a source gauge, the median over its folds."""
    return scores(root, model, domain).groupby(level="station_id").median()


def seed_ecdfs(runs: list[str], model: str, domain: str, keep: pd.Index) -> np.ndarray:
    rows = []
    for r in runs:
        v = np.sort(per_gauge(r, model, domain).reindex(keep).dropna().to_numpy())
        rows.append(np.searchsorted(v, XGRID, side="right") / len(v))
    return np.array(rows)


def draw_ecdf(ax, curves: np.ndarray, colour: str, dashes=None, lw: float = 1.3, z: int = 3) -> None:
    if len(curves) > 1:
        ax.fill_between(XGRID, curves.min(0), curves.max(0), color=colour, alpha=0.18, linewidth=0, zorder=z - 1)
    ax.plot(XGRID, np.median(curves, axis=0), color=colour, linewidth=lw, zorder=z,
            linestyle=(0, dashes) if dashes else "-", solid_capstyle="round")


def seed_median_text(runs: list[str], model: str, domain: str, keep: pd.Index) -> tuple[str, list[float]]:
    m = [float(per_gauge(r, model, domain).reindex(keep).median()) for r in runs]
    return signed(np.median(m)), m   # bare number: "median" ran into the curves of (e) and (j); the caption says it


def fig2_cdf_replay(numbers: dict) -> None:
    from matplotlib.lines import Line2D

    keys = [k for k, _, _ in DESIGNS]
    fig = plt.figure(figsize=(W, W * 0.60))
    gap, left0 = 0.013, 0.098
    width = (0.988 - left0 - 4 * gap) / 5
    rows, height = {"source": 0.560, "target": 0.200}, 0.330
    numbers["cdf"] = {}
    n_rep = []
    sep = {k: v["separation_km"]["median"] for k, v in   # as the x axis of Sect. 3.1
           json.load(open("outputs/v2_cv_designs/cv_designs.json"))["designs"].items()}
    for row, y0 in rows.items():
        for i, design in enumerate(keys):
            ax = fig.add_axes([left0 + i * (width + gap), y0, width, height])
            plain = SEED_RUNS[design]
            replay = [r for r in REPLAY_RUNS[design] if complete(r, design)]
            n_rep.append(len(replay))
            # the same gauges under every curve of a panel
            keep = per_gauge(plain[0], "M0", row).index
            for r in plain + replay:
                keep = keep.intersection(per_gauge(r, "M1", row).index)
            ax.axvline(0.0, color="#b9c1c7", linewidth=0.8, zorder=1)
            ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)
            draw_ecdf(ax, seed_ecdfs(plain, "M0", row, keep), REFERENCE, dashes=(3.2, 2.0), lw=1.0, z=4)
            draw_ecdf(ax, seed_ecdfs(plain, "M1", row, keep), PLAIN)
            draw_ecdf(ax, seed_ecdfs(replay, "M1", row, keep), REPLAY)
            tp, mp = seed_median_text(plain, "M1", row, keep)
            tr, mr = seed_median_text(replay, "M1", row, keep)
            _, m0 = seed_median_text(plain, "M0", row, keep)
            numbers["cdf"][f"{row}/{design}"] = {"n": len(keep), "M0": m0, "M1": mp, "M1_replay": mr,
                                                 "replay_seeds": replay}
            # upper left is empty in every panel; at the lower right the numbers sat on the curves
            ax.text(0.05, 0.84, tp, transform=ax.transAxes, ha="left", va="top", fontsize=BASE - 2.4, color=PLAIN)
            ax.text(0.05, 0.73, tr, transform=ax.transAxes, ha="left", va="top", fontsize=BASE - 2.4, color=REPLAY)
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["\u22121", "", "0", "", "1"] if row == "target" else [], fontsize=BASE - 2.0)
            if row == "target":   # pull the end labels inward so "1" and "-1" of adjacent panels do not touch
                ticks = ax.get_xticklabels()
                ticks[0].set_horizontalalignment("left")
                ticks[-1].set_horizontalalignment("right")
            if i == 0:
                ax.set_yticklabels(["0", "", "0.5", "", "1"], fontsize=BASE - 2.0)
                ax.set_ylabel(f"{row.capitalize()} basins", fontsize=BASE - 1.8, color=INK,
                              linespacing=1.5)
            else:
                ax.set_yticklabels([])
            ax.tick_params(length=2, pad=1.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
                ax.spines[side].set_linewidth(0.6)
            if row == "source":
                ax.set_title(f"{LABEL[design]}\n{sep[design]:,.0f} km", fontsize=BASE - 1.6,
                             fontweight="bold", color=INK, loc="left", pad=3, linespacing=1.3)
            # every panel numbered, (a)-(e) source, (f)-(j) target, so the text can cite each one
            letter = "abcdefghij"[i + (5 if row == "target" else 0)]
            ax.text(0.04, 0.96, f"({letter})", transform=ax.transAxes, ha="left", va="top",
                    fontsize=BASE - 1.6, fontweight="bold", color=INK)
    handles = [Line2D([], [], color=REFERENCE, linewidth=1.0, linestyle=(0, (3.2, 2.0)),
                      label="$M_0$  pretrained"),
               Line2D([], [], color=PLAIN, linewidth=1.3, label="$M_1$  fine-tuned, no replay"),
               Line2D([], [], color=REPLAY, linewidth=1.3, label="$M_1$  fine-tuned with source replay")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.004), ncol=3, fontsize=BASE - 2.0,
               frameon=False, handlelength=1.8, columnspacing=2.2, borderpad=0.0)
    fig.text(0.5, 0.082, "Hourly KGE in the test period", ha="center", fontsize=BASE - 1.6, color=INK)
    if min(n_rep) < 3:
        fig.text(0.99, 0.082, f"DRAFT: replay with {min(n_rep)} seed(s), no replay with 3", ha="right",
                 fontsize=fs(7), color="#b2182b")
    fig.savefig(OUT / "fig2_cdf_replay.png", dpi=300)   # fixed axes positions: no tight bbox, as fig36
    fig.savefig(OUT / "fig2_cdf_replay.pdf")
    plt.close(fig)
    print("wrote", OUT / "fig2_cdf_replay.png")


# ---------------------------------------------------------------- what is forgotten
def components(root: str, model: str, domain: str) -> pd.DataFrame:
    """As scores(), with the KGE components: |ln beta| (volume), |ln alpha| (variability), r."""
    out = []
    for p in sorted(glob.glob(f"outputs/{root}/fold*/transfer/per_station_hourly_fold*_{model}_{domain}_hourly.csv")):
        f = pd.read_csv(p)
        f = f[(f.score_status == "ok") & (f.obs_std >= 1e-3)]
        f["fold"] = int(p.split("/fold")[1].split("/")[0])
        out.append(f.set_index(["fold", "station_id"]))
    f = pd.concat(out)
    # a non-positive ratio has no log: those gauges drop out of that column's median only
    return pd.DataFrame({"kge": f.kge, "r": f.kge_r,
                         "abs_ln_alpha": np.abs(np.log(f.kge_alpha.where(f.kge_alpha > 0))),
                         "abs_ln_beta": np.abs(np.log(f.kge_beta.where(f.kge_beta > 0))),
                         # the terms as they enter KGE, |alpha - 1| and |beta - 1|, defined for every gauge
                         "abs_alpha_m1": (f.kge_alpha - 1).abs(), "abs_beta_m1": (f.kge_beta - 1).abs()})


def fig2_text_numbers() -> None:
    """Numbers for the text of Sect. 3.2 on the basis of fig2: one value per gauge (a source gauge,
    the median over the folds it is a source gauge in), the same gauges for every model of a panel;
    per seed, then the median and range over seeds. Levels and KGE components of M0, M1 and M1 with
    replay, and the paired per-gauge changes."""
    rows = []
    for domain in ("source", "target"):
        for k, _, _ in DESIGNS:
            for seed, plain, rep in zip((42, 43, 44), SEED_RUNS[k], REPLAY_RUNS[k]):
                parts = {m: components(r, mm, domain).groupby(level="station_id").median()
                         for m, r, mm in (("M0", plain, "M0"), ("M1", plain, "M1"), ("M1_replay", rep, "M1"))}
                keep = parts["M0"].index.intersection(parts["M1"].index).intersection(parts["M1_replay"].index)
                rec = {"domain": domain, "design": k, "seed": seed, "n": len(keep)}
                for m, t in parts.items():
                    t = t.loc[keep]
                    rec.update({f"{m}_{c}": v for c, v in t.median().items()})
                    rec[f"{m}_kge_below0"] = float((t.kge < 0).mean())
                kge = {m: parts[m].loc[keep, "kge"] for m in parts}
                for x, y in (("M1", "M0"), ("M1_replay", "M0"), ("M1_replay", "M1")):
                    rec[f"d_{x}_vs_{y}"] = float((kge[x] - kge[y]).median())
                    rec[f"frac_{x}_gt_{y}"] = float((kge[x] > kge[y]).mean())
                rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "fig2_text_numbers_per_seed.csv", index=False)
    g = t.drop(columns="seed").groupby(["domain", "design"], sort=False)
    pd.concat({"median": g.median(), "min": g.min(), "max": g.max()}, axis=1).to_csv(OUT / "fig2_text_numbers.csv")
    print("wrote", OUT / "fig2_text_numbers.csv")


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    numbers: dict = {}
    fig2_cdf_replay(numbers)
    fig2_text_numbers()
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print("wrote", OUT / "numbers.json")


if __name__ == "__main__":
    main()
