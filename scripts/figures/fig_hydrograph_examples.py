"""Hourly hydrographs before and after fine-tuning, one gauge per CV design (Fig. 8 of the
manuscript, fig7a_hydrographs_main.png; drawn through scripts.figures.sec41_figures).

The gauges are chosen by rule, not by eye: in each design, among the 8,709 scored gauges with
at least three observed peaks and a positive M1 KGE, the gauge whose whole-hydrograph KGE gain
(M1 - M0) is nearest the 75th percentile of that design's gain. The event shown is the
observed test-period peak with the largest topographic prominence (the property the
peak-timing metric filters on) and no missing hour within +/-2 days. Left: 60 days around it;
right: +/-2 days at hourly resolution, with each model's peak inside the +/-12 h search window
of the peak-timing metric.

Predicting five gauges is cheap, so the series are rebuilt here with the same stitching as
scripts/analysis/full_series_eval.py and cached under <output_root>/full_series/examples/.

    python -m scripts.figures.fig_hydrograph_examples [--predict-only]
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from scripts.figures.plot_style import fs

DESIGNS = [  # label, config, output root
    ("Random", "configs/aligned/phase1_runB_v2.yaml", "al_v2_runB"),
    ("Blocked", "configs/aligned/phase1_runB_blocked_v2.yaml", "al_v2_blocked"),
    ("Climate zone", "configs/aligned/phase1_runB_climate.yaml", "al_v2_climate"),
    ("Longitudinal", "configs/aligned/phase1_runB_longitudinal.yaml", "al_v2_longitudinal"),
    ("Continental", "configs/aligned/phase1_runB_continental.yaml", "al_v2_continental"),
]
OUT = "reports/sec41_m1_vs_m0/fig7a_hydrographs_main.png"
LONG = 30 * 24         # hours either side of the event, left panel
SHORT = 2 * 24         # hours either side of the event, right panel
WINDOW = 12            # peak-timing search window (Gauch et al., 2021)
GREY, BLUE = "#8d99a6", "#2a78d6"


def pick(root: str, q: float = 0.5, skilful: bool = False) -> pd.Series:
    """The gauge whose whole-hydrograph KGE gain is nearest the design's q-quantile gain.

    KGE is unbounded below, so the upper quantiles of the raw gain are gauges where M0
    collapsed and M1 is still poor (-1.06 -> -0.70): a large number, a useless picture.
    ``skilful`` ranks only the gauges where M1 has positive skill, which is what an
    example of a working model should show; the fraction kept is reported.
    """
    t = pd.read_csv(f"outputs/{root}/full_series/full_series.csv")
    f = pd.concat([pd.read_csv(p) for p in glob.glob(
        f"outputs/{root}/fold*/transfer/per_station_hourly_fold*_M1_target_hourly.csv")])
    ok = set(f.loc[(f.score_status == "ok") & (f.obs_std >= 1e-3), "station_id"])
    t = t[t.station_id.isin(ok) & (t.M1_n_peaks >= 3)].dropna(subset=["M0_kge_full", "M1_kge_full"])
    n_all = len(t)
    if skilful:
        t = t[t.M1_kge_full > 0]
    gain = t.M1_kge_full - t.M0_kge_full
    row = t.loc[(gain - gain.quantile(q)).abs().idxmin()].copy()
    row["q"], row["q_gain"], row["kept"] = q, float(gain.quantile(q)), len(t) / n_all
    return row


# Every design at the same stated percentile of the gain (75th, among gauges where M1 has
# positive skill), so the rows compare designs.
LAYOUTS = {"main": [(label, 0.75, True) for label, _, _ in DESIGNS]}


def predict(config: str, root: str, station: str, fold: int, cache_dir: Path) -> None:
    """Write series_<station>.csv (hour index, obs, M0, M1) unless it is cached."""
    target = cache_dir / f"series_{station}.csv"
    if target.exists():
        return
    import torch

    from common.config import load_config, resolve
    from common.utils import get_device, setup_logging
    from data.dataset import load_dataset_config, load_scalers, make_loader, resolve_static_spec
    from data.hourly_windows import load_cache
    from data.sources import build_eval_set
    from models.mtslstm import build_model
    from scripts.analysis.full_series_eval import DAY, collect

    cfg = load_config(config, ["eval.max_batches_per_station=0"])
    logger = setup_logging(cache_dir / "examples.log")
    device = get_device()
    scalers = load_scalers(cfg.data.root)
    y_mean, y_std = float(scalers["y_mean"]), float(scalers["y_std"])
    dyn_size = len(load_dataset_config(cfg.data.root)["dyn_features"])
    _, _, static_names = resolve_static_spec(cfg.data.root, cfg.data.get("static_exclude"),
                                             cfg.data.get("onehot_static"))
    cache = load_cache(cfg.data.cache_dir, stride=int(cfg.data.get("stride", 24)))
    run_dir = resolve(cfg.output_root)
    ds = build_eval_set(cfg, {station}, "validation", logger=logger)
    loader = make_loader(ds, num_workers=2, pin_memory=device.type == "cuda")
    got = {}
    for tag, ck in (("M0", run_dir / f"fold{fold}" / "pretrain" / "best_model.pth"),
                    ("M1", run_dir / f"fold{fold}" / "transfer" / "best_transfer_model.pth")):
        model = build_model(cfg, dyn_input_size=dyn_size, static_input_size=len(static_names)).to(device)
        model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        got[tag] = collect(model, loader, device, y_mean, y_std)[station]
    days = sorted(set(got["M0"]) & set(got["M1"]))
    t0 = days[0] - DAY + 1
    n = days[-1] + 1 - t0
    out = pd.DataFrame({"hour": np.arange(t0, t0 + n)})
    cover = np.zeros(n, bool)
    for tag in ("M0", "M1"):
        v = np.full(n, np.nan)
        for t in days:
            v[t - DAY + 1 - t0: t + 1 - t0] = got[tag][t]
            cover[t - DAY + 1 - t0: t + 1 - t0] = True
        out[tag] = v
    k = list(cache["stations"]).index(station)
    out["obs"] = np.where(cover, np.asarray(cache["forcing"][k, t0:t0 + n, 3], float), np.nan)
    out.to_csv(target, index=False)
    logger.info("cached %s (%d hours) from %s fold %d", station, n, root, fold)


def event_index(s: pd.DataFrame) -> int:
    obs = s.obs.to_numpy()
    ok = np.isfinite(obs) & np.isfinite(s.M1.to_numpy())
    pk, prop = signal.find_peaks(np.where(ok, obs, 0.0), prominence=0)
    # An event is drawn only where the record around it is whole: no missing hour within
    # the +/-2-day zoom and at most 10 % missing over the 60-day panel.
    csum = np.concatenate([[0], np.cumsum(ok)])
    def frac(i, h):
        lo, hi = i - h, i + h + 1
        return (csum[hi] - csum[lo]) / (hi - lo)
    keep = np.array([LONG <= i < len(s) - LONG and frac(i, SHORT) == 1.0 and frac(i, LONG) >= 0.9
                     for i in pk], dtype=bool)
    pk, prom = pk[keep], prop["prominences"][keep]
    return int(pk[np.argmax(prom)])


def draw(ax, s: pd.DataFrame, i: int, half: int, marks: bool) -> None:
    lo, hi = i - half, i + half + 1
    h = (np.arange(lo, hi) - i) / 24
    obs = s.obs.to_numpy()
    ax.fill_between(h, 0, obs[lo:hi], color="#e3e6ea", linewidth=0, zorder=1)
    ax.plot(h, obs[lo:hi], color="#111111", linewidth=1.5, zorder=4)
    ax.plot(h, s.M0.to_numpy()[lo:hi], color=GREY, linewidth=1.5, zorder=2)
    ax.plot(h, s.M1.to_numpy()[lo:hi], color=BLUE, linewidth=1.8, zorder=3)
    if marks:
        ax.axvspan(-WINDOW / 24, WINDOW / 24, color="#f3f4f6", zorder=0)
        for tag, c in (("M0", GREY), ("M1", BLUE)):
            w = s[tag].to_numpy()[i - WINDOW: i + WINDOW + 1]
            j = int(np.nanargmax(w))
            ax.plot((j - WINDOW) / 24, w[j], "v", color=c, markersize=7.5,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=5)
        ax.plot(0, obs[i], "v", color="#111111", markersize=7.5, markeredgecolor="white",
                markeredgewidth=1.0, zorder=6)
    ax.set_xlim(h[0], h[-1])
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=fs(7.5))
    ax.grid(axis="y", alpha=0.18, linewidth=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def ordinal(q: float) -> str:
    return f"{int(round(q * 100))}th"


def signed(v: float, fmt: str = "{:+.2f}") -> str:
    """Signed number with a true minus sign."""
    return fmt.format(v).replace("-", "\u2212")


def metrics_legend(ax, p) -> None:
    """Whole-period KGE, FHV and peak timing of M0 and M1, as a small legend in the panel.

    The line swatch carries which model a row belongs to; the text stays dark.
    """
    def row(kge, fhv, t):
        return f"KGE {signed(kge)}   FHV {signed(fhv, '{:+.0f}%')}   timing {t:.1f} h"
    handles = [plt.Line2D([], [], color=GREY, linewidth=1.5), plt.Line2D([], [], color=BLUE, linewidth=1.8)]
    leg = ax.legend(handles, [row(p.M0_kge_full, p.M0_fhv_pct, p.M0_peak_timing_h),
                              row(p.M1_kge_full, p.M1_fhv_pct, p.M1_peak_timing_h)],
                    loc="best", fontsize=fs(6.4), handlelength=1.2, handletextpad=0.5,
                    borderpad=0.3, labelspacing=0.25, frameon=True, framealpha=0.85, edgecolor="none")
    leg.set_zorder(10)


def mark_zoom(big, zoom) -> None:
    """Outline the enlarged window in the 60-day panel and join it to the hourly panel.

    Called after the layout is final, so the connectors end on the panels' real corners.
    """
    from matplotlib.patches import ConnectionPatch, Rectangle
    lo, hi = big.get_ylim()
    x0, x1 = -SHORT / 24, SHORT / 24
    big.add_patch(Rectangle((x0, lo), x1 - x0, hi - lo, fill=False, edgecolor="#555555",
                            linewidth=0.9, linestyle=(0, (3, 2)), zorder=6, clip_on=False))
    for ya, yb in ((hi, 1.0), (lo, 0.0)):
        con = ConnectionPatch(xyA=(x1, ya), coordsA=big.transData, xyB=(0.0, yb), coordsB=zoom.transAxes,
                              color="#888888", linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
        big.figure.add_artist(con)
    # The enlarged panel framed in the same dashed style as the box it enlarges.
    zoom.add_patch(Rectangle((0, 0), 1, 1, transform=zoom.transAxes, fill=False, edgecolor="#555555",
                             linewidth=0.9, linestyle=(0, (3, 2)), zorder=6, clip_on=False))


def plot(picks, out: str) -> None:
    fig, axes = plt.subplots(len(picks), 2, figsize=(17.7 / 2.54, 1.42 * len(picks)),
                             gridspec_kw={"width_ratios": [2.3, 1]}, squeeze=False)
    for r, (label, root, p) in enumerate(picks):
        s = pd.read_csv(f"outputs/{root}/full_series/examples/series_{p.station_id}.csv")
        i = event_index(s)
        draw(axes[r, 0], s, i, LONG, marks=False)
        draw(axes[r, 1], s, i, SHORT, marks=True)
        # Enlarged panel's axis on its right, so the zoom connectors meet its left edge
        # without crossing tick labels.
        axes[r, 1].yaxis.tick_right()
        axes[r, 1].spines["left"].set_visible(False)
        axes[r, 1].spines["right"].set_visible(True)
        axes[r, 0].set_ylabel("mm/h", fontsize=fs(8))
        src = p.station_id.split("__")[0]
        # every panel numbered: (a) the 60 days and (b) the enlarged event of the first row, and so on
        axes[r, 0].set_title(f"({'abcdefghij'[2 * r]})  {label}, {ordinal(p.q)}-percentile gain  ({src})",
                             loc="left", fontsize=fs(8.2), weight="bold")
        axes[r, 1].set_title(f"({'abcdefghij'[2 * r + 1]})", loc="left", fontsize=fs(8.2), weight="bold")
        metrics_legend(axes[r, 0], p)
    axes[-1, 0].set_xlabel("Days from the most prominent observed peak of the test period", fontsize=fs(8))
    axes[-1, 1].set_xlabel("Days (hourly resolution)", fontsize=fs(8))
    handles = [plt.Line2D([], [], color="#111111", linewidth=1.5),
               plt.Line2D([], [], color=GREY, linewidth=1.5),
               plt.Line2D([], [], color=BLUE, linewidth=1.8),
               plt.Line2D([], [], marker="v", linestyle="", color="#555555", markersize=7.5)]
    fig.legend(handles, ["Observed", "$M_0$  before fine-tuning", "$M_1$  after daily-only fine-tuning",
                         "peak (model peak searched within the shaded \u00b112 h)"],
               loc="lower center", ncol=4, frameon=False, fontsize=fs(7.8), bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.045 if len(picks) < 5 else 0.035, 1, 1))
    for r in range(len(picks)):
        mark_zoom(axes[r, 0], axes[r, 1])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    for label, _, p in picks:
        print(f"  {label:13s} q{p.q:.2f} {p.station_id:28s} fold {int(p.fold)}  "
              f"KGE {p.M0_kge_full:+.2f}->{p.M1_kge_full:+.2f}")


def collect_picks(layout: str):
    design = {label: (config, root) for label, config, root in DESIGNS}
    picks = []
    for label, q, skilful in LAYOUTS[layout]:
        config, root = design[label]
        p = pick(root, q, skilful)
        cache_dir = Path(f"outputs/{root}/full_series/examples")
        cache_dir.mkdir(parents=True, exist_ok=True)
        predict(config, root, p.station_id, int(p.fold), cache_dir)
        picks.append((label, root, p))
    return picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predict-only", action="store_true", help="cache the five series, draw nothing")
    args = ap.parse_args()
    picks = collect_picks("main")
    if not args.predict_only:
        plot(picks, OUT)


if __name__ == "__main__":
    main()
