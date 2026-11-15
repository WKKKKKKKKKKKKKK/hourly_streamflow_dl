"""Figures and numbers of Appendix D (daily-scale evaluation: daily LSTM and ERA5-Land).

Writes into reports/sec43_daily_scale/:

  fig4_maps              Fig. D1: daily KGE of M1 (24-h mean of its hourly output) minus that of
                         a daily LSTM trained from scratch on the same target gauges, period and
                         daily observations; one map per CV design and one for the 282 African
                         catchments, each with the histogram of the differences and bars of the
                         two models' median daily KGE
  fig2_africa_era5       Fig. D2: the 282 African catchments, daily LSTM, M1 and ERA5-Land:
                         (a) distribution of daily KGE, (b-d) of r, alpha and beta
  fig3_africa_examples   Fig. D3: (a-f) three African catchments at the 25th/50th/75th percentile
                         of M1's daily KGE, the season around the largest observed flood of the
                         test period and a 15-day zoom with M1's (unscored) hourly output;
                         (g-i) daily FHV, peak-MAPE and peak timing over every catchment
  numbers.json           every number quoted from these figures

Design scores come from scripts/analysis/score_daily_target.py (outputs/v2_design_daily/<run>);
each gauge is a target once per design, so the folds pool (8,574 gauges with all arms scored).
Statistics are taken per M1 seed (each against the one daily LSTM) and then the median over the
seeds; numbers.json keeps every seed's values. Africa is outputs/al_v2_africa_five_way (per basin,
the median over the five members; scripts/analysis/africa_four_way.py --prefix al_v2).

    python -m scripts.figures.sec43_figures
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.figures.sec41_figures import BLUE, DESIGNS, SEED_RUNS, W, fs, plt, style
from scripts.figures.sec42_figures import N_FOLDS, signed

OUT = Path("reports/sec43_daily_scale")
DAILY = Path("outputs/v2_design_daily")
GREEN, ERA5 = "#11a579", "#b2182b"   # daily LSTM, ERA5-Land
TEXT_UP = 1.2   # fig2-4 go in the main text: their labels one step larger than the other sec43 figures


def fb(size: float) -> float:
    return fs(size * TEXT_UP)


def pooled(root: str, tag: str) -> pd.Series | None:
    files = sorted(glob.glob(str(DAILY / root / "fold*" / f"per_station_daily_fold*_{tag}_target.csv")))
    if not files:
        return None
    f = pd.concat([pd.read_csv(p) for p in files])
    f = f[(f.score_status == "ok") & (f.obs_std >= 1e-3)]
    return f.set_index("station_id").kge


def scored(root: str, tag: str, design: str) -> bool:
    return len(glob.glob(str(DAILY / root / "fold*" / f"per_station_daily_fold*_{tag}_target.csv"))) \
        == N_FOLDS.get(design, 5)


def groups() -> dict:
    """Per group (five designs, then Africa): one paired table of daily KGE per seed, M1 vs daily LSTM.

    Designs: one table per M1 seed whose folds are all scored, each against the single-seed daily
    LSTM -- statistics are taken within a seed and then the median over seeds, so every
    comparison is one model against one model (a per-gauge median over seeds would narrow M1's
    spread by averaging out seed noise the daily LSTM keeps). Africa: one table, each arm the
    per-basin median over its five members (outputs/al_v2_africa_five_way).
    """
    out = {}
    for key, label, _ in DESIGNS:
        base = pooled(f"al_v2_dailylstm_{key}", "daily_lstm")
        seeds = [r for r in SEED_RUNS[key] if scored(r, "M1", key)]   # seeds 43/44 may still be scoring
        tabs = [pd.concat([pooled(r, "M1"), base], axis=1, keys=["M1", "D"]).dropna() for r in seeds]
        out[key] = {"label": label, "tables": tabs, "seeds": seeds}
    af = pd.read_csv("outputs/al_v2_africa_five_way/per_basin_four_way.csv").set_index("station_id")
    out["africa"] = {"label": "Africa", "seeds": [],
                     "tables": [af[["M1", "daily_lstm"]].rename(columns={"daily_lstm": "D"}).dropna()]}
    return out


def paired_stats(t: pd.DataFrame) -> dict:
    from scipy.stats import wilcoxon

    d = t.M1 - t.D
    out = {"n": len(t), "paired_median": float(d.median()), "frac_M1_better": float((d > 0).mean()),
           "wilcoxon_p": float(wilcoxon(t.M1, t.D).pvalue)}
    for col, name in (("M1", "M1"), ("D", "daily_lstm")):
        p5, q1, med, q3, p95 = np.percentile(t[col], [10, 25, 50, 75, 90])   # whiskers 10th-90th
        out.update({name: float(med), f"{name}_p5": float(p5), f"{name}_q1": float(q1),
                    f"{name}_q3": float(q3), f"{name}_p95": float(p95)})
    return out


def m1_vs_daily_lstm(numbers: dict) -> None:
    """Paired daily KGE, M1 against the daily LSTM, per design and for Africa (printed on
    fig4_maps). Every number is taken within a seed (one M1 seed against the one daily LSTM),
    then the median over seeds."""
    g = groups()
    numbers["m1_vs_daily_lstm"] = {}
    for k in g:
        per_seed = [paired_stats(t) for t in g[k]["tables"]]
        st = {c: float(np.median([p[c] for p in per_seed])) for c in per_seed[0]}
        st.update(seeds=g[k]["seeds"], per_seed=per_seed, n=per_seed[0]["n"])
        # 5th, 25th, 50th, 75th and 95th percentile of the per-gauge difference M1 - daily LSTM,
        # per seed, then the median over seeds
        per = [np.percentile(t.M1 - t.D, [5, 25, 50, 75, 95]) for t in g[k]["tables"]]
        st["diff_p5_q1_med_q3_p95"] = [float(v) for v in np.median(per, axis=0)]
        numbers["m1_vs_daily_lstm"][k] = st


# ---------------------------------------------------------------- Fig. D1: maps
# The global gauges sit in North America and Europe (lon -125..19, lat 25..66) and in Japan and
# Australia (lon 121..154, lat -39..44); nothing between. Each global map is therefore two pieces at
# one scale: the Americas-Europe band (a short strip) and the western Pacific (tall). The space under
# the strip holds the histogram and the numbers. Africa is drawn at the same scale.
BAND = ((-128, 26), (20, 70))      # lon, lat
PACIFIC = ((108, 156), (-45, 46))
AFRICA = ((-20, 55), (-36, 38))


def fig4_maps(numbers: dict) -> None:
    """M1 - daily LSTM per gauge, one map per design and Africa (3 x 2). Under each map's
    Americas-Europe strip: the histogram of the differences, and two bars with the median daily KGE
    of M1 and of the daily LSTM, so the absolute level is read off the same panel. A gauge's value
    is the median over seeds of M1 - daily LSTM; the printed numbers are m1_vs_daily_lstm's (per
    seed, then the median over seeds)."""
    import geopandas as gpd
    from matplotlib.colors import BoundaryNorm, ListedColormap

    from scripts.figures.plot_style import GAIN_BOUNDS as DB, GAIN_COLOURS as DC, RDBU as _RDBU
    from scripts.figures.plot_style import draw_borders

    dcm = ListedColormap(DC)
    dcm.set_under(_RDBU(0.0))
    dcm.set_over(_RDBU(1.0))
    dnorm = BoundaryNorm(DB, dcm.N)
    coords = pd.read_csv("folds/folds_blocked.csv").set_index("station_id")[["lat", "long"]]
    af = gpd.read_file("africa/africa_basins.gpkg").astype({"station_id": str}).set_index("station_id")[["lat", "long"]]
    g = groups()
    st = numbers["m1_vs_daily_lstm"]
    keys = [k for k, _, _ in DESIGNS] + ["africa"]

    left, right, colgap, rowgap, head, foot = 0.04, 0.04, 0.14, 0.30, 0.30, 0.62
    span = lambda r: r[1] - r[0]
    gap_deg = 3
    cell_w = (W - left - right - colgap) / 2
    scale = cell_w / (span(BAND[0]) + gap_deg + span(PACIFIC[0]))   # inches per degree
    row_h = span(PACIFIC[1]) * scale
    H = head + 3 * row_h + 2 * rowgap + foot
    fig = plt.figure(figsize=(W, H))

    def add(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    def piece(x, y, frame, box, size, edge=0.0):
        (lo, hi), (la, lb) = box
        ax = add(x, y, span(box[0]) * scale, span(box[1]) * scale)
        draw_borders(ax)
        order = np.random.default_rng(0).permutation(len(frame))
        ax.scatter(frame["long"].to_numpy()[order], frame["lat"].to_numpy()[order], c=frame.d.to_numpy()[order],
                   s=size, cmap=dcm, norm=dnorm, linewidths=edge, edgecolors="#333333", zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(la, lb)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_linewidth(0.4)
            side.set_color("#bbbbbb")
        return ax

    for i, k in enumerate(keys):
        r, c = divmod(i, 2)
        x0 = left + c * (cell_w + colgap)
        y0 = H - head - (r + 1) * row_h - r * rowgap
        tabs = g[k]["tables"]
        f = pd.DataFrame({"d": pd.concat([tb.M1 - tb.D for tb in tabs], axis=1).median(axis=1)})
        f = f.join(af if k == "africa" else coords, how="inner")
        if k == "africa":
            aw = span(AFRICA[0]) * scale
            ah = span(AFRICA[1]) * scale
            # sparse, and daily gauges only (no hourly observation): larger, outlined markers
            top = piece(x0 + cell_w - aw, y0 + row_h - ah, f, AFRICA, 9.0, edge=0.25)
            fx, fy, fw, fh = x0, y0 + row_h - ah, cell_w - aw - 0.1, ah
        else:
            band_h = span(BAND[1]) * scale
            top = piece(x0, y0 + row_h - band_h, f, BAND, 0.7)
            piece(x0 + (span(BAND[0]) + gap_deg) * scale, y0, f, PACIFIC, 0.7)
            fx, fy, fw, fh = x0, y0, span(BAND[0]) * scale, row_h - band_h - 0.06
        fig.text(x0 / W, (y0 + row_h + 0.05) / H, f"({'abcdef'[i]})  {g[k]['label']}", fontsize=fb(8.5),
                 weight="bold", va="bottom")
        if k == "africa":
            fig.text((x0 + cell_w) / W, (y0 + row_h + 0.05) / H, "daily gauges, no hourly observation",
                     fontsize=fb(6.5), color="#555555", ha="right", va="bottom")
        # histogram of the differences; Africa's free space is tall, so there the bars go under it
        if k == "africa":
            hist_box = (fx + 0.05, fy + fh * 0.54, fw - 0.15, fh * 0.36)
            bar_box = (fx + 0.62, fy + 0.14, fw - 1.05, fh * 0.22)
        else:
            hw = fw * 0.42   # narrower than before the larger fonts, so "Daily LSTM" clears the histogram
            hist_box = (fx + 0.05, fy + 0.16, hw, fh - 0.22)
            bx = fx + 0.05 + hw + 0.62
            bar_box = (bx, fy + 0.20, fw - (bx - fx) - 0.30, fh - 0.34)
        ins = add(*hist_box)
        ins.hist(np.clip(f.d, -0.5, 0.5), bins=np.linspace(-0.5, 0.5, 21), color="#9aa6b2", edgecolor="white",
                 linewidth=0.3)
        ins.axvline(0, color="#333333", linewidth=0.6)
        ins.axvline(st[k]["paired_median"], color="#222222", linewidth=1.2, linestyle="--")
        ins.set_yticks([])
        ins.set_xticks([-0.5, 0, 0.5], ["\u22120.5", "0", "0.5"], fontsize=fb(5.8))
        ins.tick_params(length=1.5, pad=1, width=0.4)
        for side in ("top", "right", "left"):
            ins.spines[side].set_visible(False)
        ins.spines["bottom"].set_linewidth(0.4)
        ins.patch.set_alpha(0)
        ins.set_title(f"median {signed(st[k]['paired_median'])};  $M_1$ better {100 * st[k]['frac_M1_better']:.0f}%",
                      fontsize=fb(6.3), loc="left", pad=1.5, color="#222222")
        # absolute level: median daily KGE of the two, as bars
        bars = add(*bar_box)
        vals = (st[k]["M1"], st[k]["daily_lstm"])
        bars.barh([1, 0], vals, color=[BLUE, GREEN], height=0.62)
        for yb, v in zip((1, 0), vals):
            bars.text(v + 0.02, yb, f"{v:.2f}", va="center", ha="left", fontsize=fb(6.3), color="#222222")
        bars.set_yticks([1, 0], ["$M_1$", "Daily LSTM"], fontsize=fb(6.3))
        bars.set_xlim(0, 1)
        bars.set_xticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=fb(5.8))
        bars.tick_params(length=1.5, pad=1, width=0.4)
        bars.set_title("median KGE", fontsize=fb(6.3), loc="left", pad=1.5, color="#222222")
        for side in ("top", "right"):
            bars.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            bars.spines[side].set_linewidth(0.4)
    cax = add(W * 0.2, 0.36, W * 0.6, 0.09)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=dnorm, cmap=dcm), cax=cax, orientation="horizontal",
                      extend="both", ticks=DB)
    cb.ax.set_xticklabels([f"{b:g}" for b in DB])
    cb.ax.tick_params(labelsize=fb(6), length=1.5)
    cb.set_label("$M_1$ \u2212 daily LSTM, daily KGE per gauge   (red: $M_1$ better)", fontsize=fb(7.5))
    n_seed = min(len(g[k]["seeds"]) for k in keys[:5])
    if n_seed < 3:
        fig.text((W - 0.05) / W, (H - 0.03) / H, f"DRAFT: designs' M1 with {n_seed} seed(s) at least", ha="right",
                 va="top", fontsize=fb(6.5), color="#b2182b")
    fig.savefig(OUT / "fig4_maps.png", dpi=300)   # placed by hand: no tight bbox
    fig.savefig(OUT / "fig4_maps.pdf")
    plt.close(fig)
    print("wrote", OUT / "fig4_maps.png")


# ---------------------------------------------------------------- Fig. D2: Africa and ERA5-Land
def africa_arms() -> dict[str, pd.DataFrame]:
    """Per basin KGE, r, alpha, beta for the daily LSTM, M1 and ERA5-Land on identical basins."""
    basins = pd.read_csv("outputs/al_v2_africa_five_way/per_basin_four_way.csv").set_index("station_id").dropna().index
    cols = ["kge", "kge_r", "kge_alpha", "kge_beta"]

    def members(pattern: str) -> pd.DataFrame:
        t = [pd.read_csv(p).set_index("station_id")[cols] for p in sorted(glob.glob(f"{pattern}/per_basin_M1.csv"))]
        return pd.concat(t).groupby(level=0).median().reindex(basins)

    tw = pd.read_csv("outputs/al_v2_africa_three_way/daily_three_way_per_basin.csv").set_index("station_id")
    era5 = tw[[f"{c}_era5_land" for c in cols]].set_axis(cols, axis=1).reindex(basins)
    return {"daily_lstm": members("outputs/al_v2_africa_daily_lstm_fold*"),
            "M1": members("outputs/al_v2_africa_m1b_fold*"), "era5_land": era5}


def fig2_africa_era5(numbers: dict) -> None:
    from matplotlib.patches import Rectangle

    arms = africa_arms()
    style_of = {"era5_land": (ERA5, "-.", "ERA5-Land"), "daily_lstm": (GREEN, "-", "Daily LSTM"),
                "M1": (BLUE, "-", "$M_1$")}
    fig = plt.figure(figsize=(W, W * 0.40))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1, 1, 1], wspace=0.42)
    a = fig.add_subplot(gs[0])
    numbers["africa_era5"] = {}
    for k, (c, ls, lab) in style_of.items():
        v = np.sort(arms[k].kge.dropna().to_numpy())
        a.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, linestyle=ls, linewidth=1.5 if k == "M1" else 1.2,
               label=f"{lab}  ({signed(np.median(v))})")
        numbers["africa_era5"][k] = {"n": len(v), "median_kge": float(np.median(v)),
                                     "share_kge_below_0": float((v < 0).mean()),
                                     **{f"median_{c}": float(arms[k][c].median()) for c in ("kge_r", "kge_alpha", "kge_beta")},
                                     "median_abs_beta_minus_1": float((arms[k].kge_beta - 1).abs().median()),
                                     "median_abs_alpha_minus_1": float((arms[k].kge_alpha - 1).abs().median())}
    d = (arms["M1"].kge - arms["era5_land"].kge).dropna()
    numbers["africa_era5"]["M1_minus_era5"] = {"paired_median": float(d.median()), "frac_M1_better": float((d > 0).mean())}
    a.axvline(0, color="#b9c1c7", linewidth=0.8, zorder=0)
    a.set_xlim(-1, 1)
    a.set_ylim(0, 1)
    a.set_xlabel("Daily KGE", fontsize=fb(8))
    a.set_ylabel("Cumulative share of catchments", fontsize=fb(8))
    a.set_title("(a)  Daily KGE", loc="left", fontsize=fb(8), weight="bold")
    a.text(0.97, 0.03, f"{len(arms['M1'])} catchments", transform=a.transAxes, ha="right", va="bottom",
           fontsize=fb(6.5), color="#333333")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=fb(7.5),
               title="median daily KGE in brackets", title_fontsize=fb(7))
    a.tick_params(labelsize=fb(7.5))
    style(a)
    comps = (("kge_r", "(b)  $r$  timing", False, (-0.2, 1.0)),
             ("kge_alpha", "(c)  $\\alpha$  variability", True, (0.1, 10)),
             ("kge_beta", "(d)  $\\beta$  volume", True, (0.1, 10)))
    order = ["daily_lstm", "M1", "era5_land"]
    for j, (col, title, logy, ylim) in enumerate(comps):
        ax = fig.add_subplot(gs[j + 1])
        for i, k in enumerate(order):
            v = arms[k][col].dropna().to_numpy()
            if logy:
                v = v[v > 0]
            lo, q1, med, q3, hi = np.percentile(v, [5, 25, 50, 75, 95])
            c = style_of[k][0]
            ax.plot([i, i], [lo, hi], color=c, linewidth=0.9)
            ax.add_patch(Rectangle(
                (i - 0.3, q1), 0.6, q3 - q1, facecolor=c, alpha=0.25, edgecolor=c, linewidth=0.8))
            ax.plot([i - 0.3, i + 0.3], [med, med], color=c, linewidth=2)
        ax.axhline(1, color="#333333", linewidth=0.8, linestyle="--")
        if logy:
            ax.set_yscale("log")
            ax.set_yticks([0.1, 0.3, 1, 3, 10], ["0.1", "0.3", "1", "3", "10"])
        ax.set_ylim(*ylim)
        ax.set_xticks(range(3), ["Daily\nLSTM", "$M_1$", "ERA5-\nLand"], fontsize=fb(6))   # wider ran together
        ax.set_xlim(-0.6, 2.6)
        ax.tick_params(axis="y", labelsize=fb(7))
        ax.set_title(title, loc="left", fontsize=fb(8), weight="bold")
        style(ax)
        ax.grid(axis="x", visible=False)
    save(fig, "fig2_africa_era5")


# ---------------------------------------------------------------- fig3 African examples
EX = Path("outputs/al_v2_africa_examples")
OBS = "#222222"


def kge(o: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(o) & np.isfinite(s)
    o, s = o[ok], s[ok]
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)


PEAK_WINDOW = 3   # days; NeuralHydrology mean_peak_timing default for '1D' (max(12h in days, 3))


def daily_peak_metrics(dates: pd.Series, obs: np.ndarray, sim: np.ndarray) -> dict:
    """FHV, peak-MAPE and peak timing of one daily series, as scripts/analysis/full_series_eval.py does
    hourly: observed peaks with prominence >= std(obs), >= 100 steps apart; simulated peak the
    largest value within +/-PEAK_WINDOW days, peaks whose window crosses a gap skipped."""
    from scipy import signal

    from common.metrics import peak_flow_bias

    keep = np.isfinite(obs) & np.isfinite(sim)
    o, s = obs[keep], sim[keep]
    t = ((pd.to_datetime(dates[keep]) - pd.Timestamp("1970-01-01")).dt.days).to_numpy()
    if o.size < 365 or np.std(o) == 0:
        return {}
    out = {"fhv_pct": float(peak_flow_bias(o, s))}
    peaks, _ = signal.find_peaks(o, distance=100, prominence=np.std(o))
    w, lags = PEAK_WINDOW, []
    for i in peaks:
        if i - w < 0 or i + w >= o.size or t[i + w] - t[i - w] != 2 * w:
            continue
        j = i if (s[i] > s[i - 1] and s[i] > s[i + 1]) else i - w + int(np.argmax(s[i - w: i + w + 1]))
        lags.append(j - i)
    out["peak_timing_d"] = float(np.mean(np.abs(lags))) if lags else float("nan")
    out["peak_timing_signed_d"] = float(np.mean(lags)) if lags else float("nan")
    out["peak_mape"] = (float(np.mean(np.abs(s[peaks] - o[peaks]) / o[peaks]) * 100)
                        if peaks.size and np.all(o[peaks] != 0) else float("nan"))
    out["n_peaks"] = len(lags)
    return out


def africa_peaks() -> dict[str, pd.DataFrame]:
    """Per basin daily peak metrics: models per member, then the median over members; ERA5-Land once."""
    cache = EX / "daily_peak_metrics.csv"
    if cache.exists() and cache.stat().st_mtime > (EX / "daily_series_members.csv.gz").stat().st_mtime:
        t = pd.read_csv(cache)
    else:
        rows = []
        mem = pd.read_csv(EX / "daily_series_members.csv.gz", parse_dates=["date"])
        for (sid, k), g in mem.groupby(["station_id", "member"]):
            g = g.sort_values("date")
            for arm in ("M1", "daily_lstm"):
                rows.append({"station_id": sid, "arm": arm, "member": k,
                             **daily_peak_metrics(g.date, g.obs.to_numpy(float), g[arm].to_numpy(float))})
        d = pd.read_csv(EX / "daily_series.csv", parse_dates=["date"])
        for sid, g in d.groupby("station_id"):
            g = g.sort_values("date")
            rows.append({"station_id": sid, "arm": "era5_land", "member": 0,
                         **daily_peak_metrics(g.date, g.obs.to_numpy(float), g.era5_land.to_numpy(float))})
        t = pd.DataFrame(rows)
        t.to_csv(cache, index=False)
    basins = pd.read_csv("outputs/al_v2_africa_five_way/per_basin_four_way.csv").set_index("station_id").dropna().index
    return {arm: g.drop(columns=["arm", "member"]).groupby("station_id").median().reindex(basins)
            for arm, g in t.groupby("arm")}


def fig3_africa_examples(numbers: dict) -> None:
    import geopandas as gpd
    import matplotlib.dates as mdates
    from matplotlib.patches import ConnectionPatch, Rectangle

    d = pd.read_csv(EX / "daily_series.csv", parse_dates=["date"])
    h = pd.read_csv(EX / "hourly_series_M1.csv.gz", parse_dates=["time"])
    h["M1"] *= 24   # mm/h -> mm/d, so the hourly curve shares the daily axis
    area = gpd.read_file("africa/africa_basins.gpkg").astype({"station_id": str}).set_index("station_id")
    m1 = pd.read_csv("outputs/al_v2_africa_five_way/per_basin_four_way.csv").set_index("station_id").M1.dropna()
    ranked = m1.sort_values()
    order = [ranked.index[int(len(ranked) * q)] for q in (0.25, 0.50, 0.75)]   # spanning the outcome
    arms = (("M1", BLUE, "-", 1.3, "$M_1$"), ("daily_lstm", GREEN, "-", 1.1, "Daily LSTM"),
            ("era5_land", ERA5, "-.", 1.0, "ERA5-Land"))
    fig = plt.figure(figsize=(W, W * 1.02))
    outer = fig.add_gridspec(2, 1, height_ratios=[3, 1.05], hspace=0.30)
    top = outer[0].subgridspec(3, 2, width_ratios=[2.3, 1], wspace=0.08, hspace=0.55)
    axes = np.array([[fig.add_subplot(top[r, c]) for c in range(2)] for r in range(3)])
    numbers["africa_examples"] = {}
    for row, sid in enumerate(order):
        g = d[d.station_id == sid].sort_values("date").set_index("date")
        peak = g.obs.idxmax()
        big, zoom = axes[row]
        sc = {a: kge(g.obs.to_numpy(), g[a].to_numpy()) for a, *_ in arms}
        numbers["africa_examples"][sid] = {"peak": str(peak.date()), "kge_full_period_member_mean": sc,
                                           "M1_percentile": float((m1 < m1[sid]).mean())}
        for ax, (lo, hi) in ((big, (peak - pd.Timedelta(days=120), peak + pd.Timedelta(days=120))),
                             (zoom, (peak - pd.Timedelta(days=7), peak + pd.Timedelta(days=8)))):
            w = g.loc[lo:hi]
            ax.plot(w.index + (pd.Timedelta(hours=12) if ax is zoom else pd.Timedelta(0)), w.obs,
                    "o" if ax is zoom else "-", color=OBS, markersize=2.5, linewidth=1.0,
                    zorder=5)   # zoom: a day's value at its centre
            for a, c, ls, lw, lab in arms:
                ax.step(w.index, w[a], where="post" if ax is zoom else "mid", color=c, linestyle=ls,
                        linewidth=lw, zorder=4 if a == "M1" else 3)
            if ax is zoom:
                hh = h[(h.station_id == sid) & (h.time >= lo) & (h.time < hi)]
                ax.plot(hh.time, hh.M1, color=BLUE, linewidth=0.6, alpha=0.75, zorder=2)
            ax.set_xlim(lo, hi)
            ax.tick_params(labelsize=fb(7))
            style(ax)
        ymax = g.loc[peak - pd.Timedelta(days=7):peak + pd.Timedelta(days=8),
                     ["obs", "M1", "daily_lstm", "era5_land"]].max().max()
        hz = h[(h.station_id == sid) & (h.time >= peak - pd.Timedelta(days=7)) & (h.time < peak + pd.Timedelta(days=8))]
        ymax = max(ymax, hz.M1.max() if len(hz) else 0)
        big.add_patch(Rectangle((peak - pd.Timedelta(days=7), 0), pd.Timedelta(days=15), ymax * 1.05,
                                fill=False, linestyle="--", linewidth=0.8, edgecolor="#555555", zorder=6))
        for yb, yz in ((ymax * 1.05, 1), (0, 0)):
            fig.add_artist(ConnectionPatch((peak + pd.Timedelta(days=8), yb), (0, yz), coordsA="data",
                                           coordsB="axes fraction", axesA=big, axesB=zoom, color="#888888",
                                           linewidth=0.6, linestyle="--"))
        big.set_ylim(bottom=0)
        zoom.set_ylim(0, ymax * 1.05)
        zoom.yaxis.tick_right()
        # two-monthly ticks, none in the last 8 % of the axis, where a label runs into the zoom panel's first
        lo_n, hi_n = big.get_xlim()
        ticks = mdates.MonthLocator(interval=2).tick_values(mdates.num2date(lo_n), mdates.num2date(hi_n))
        big.set_xticks([t for t in ticks if lo_n <= t <= hi_n - 0.08 * (hi_n - lo_n)])
        big.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        zoom.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        zoom.xaxis.set_major_locator(mdates.DayLocator(interval=4))
        big.set_ylabel("Discharge (mm d$^{-1}$)", fontsize=fb(7.5))
        pct = round(100 * numbers["africa_examples"][sid]["M1_percentile"])
        big.set_title(f"({'ace'[row]})  {sid.split('__')[-1]}, {area.loc[sid, 'area']:,.0f} km$^2$ "
                      f"({area.loc[sid, 'lat']:.1f}\u00b0, {area.loc[sid, 'long']:.1f}\u00b0); "
                      f"{pct}th percentile", loc="left", fontsize=fb(8), weight="bold")
        zoom.text(0.03, 0.97, f"({'bdf'[row]})", transform=zoom.transAxes, ha="left", va="top",
                  fontsize=fb(8), weight="bold")
        # daily KGE of this basin, upper right of the main panel (empty in all three rows)
        big.text(0.99, 0.97, "KGE", transform=big.transAxes, ha="right", va="top", fontsize=fb(6.5), color=OBS)
        for k, (a, c, _, _, lab) in enumerate(arms):
            big.text(0.99, 0.82 - 0.15 * k, f"{lab}  {signed(sc[a])}", transform=big.transAxes, ha="right",
                     va="top", fontsize=fb(6.5), color=c)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=OBS, linewidth=1.0, marker="o", markersize=2.5,
                      label="Observed daily (dots in zoom)"),
               Line2D([], [], color=BLUE, linewidth=1.3, label="$M_1$ daily"),
               Line2D([], [], color=BLUE, linewidth=0.6, alpha=0.75, label="$M_1$ hourly"),
               Line2D([], [], color=GREEN, linewidth=1.1, label="Daily LSTM"),
               Line2D([], [], color=ERA5, linewidth=1.0, linestyle="-.", label="ERA5-Land")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.905), ncol=5, frameon=False,
               fontsize=fb(6.5), handlelength=1.8, columnspacing=1.4)

    # bottom row: daily flood-peak metrics over every catchment
    pk = africa_peaks()
    bottom = outer[1].subgridspec(1, 3, wspace=0.35)
    panels = (("fhv_pct", "(g)  High-flow bias FHV (%)", 0.0),
              ("peak_mape", "(h)  Peak magnitude error (%)", None),
              ("peak_timing_d", "(i)  Peak timing error (days)", None))
    order_arms = ["daily_lstm", "M1", "era5_land"]
    colour = {a: c for a, c, *_ in arms}
    numbers["africa_peaks"] = {}
    for j, (col, title, ref) in enumerate(panels):
        ax = fig.add_subplot(bottom[j])
        for i, arm in enumerate(order_arms):
            v = pk[arm][col].dropna().to_numpy()
            lo, q1, med, q3, hi = np.percentile(v, [5, 25, 50, 75, 95])
            c = colour[arm]
            ax.plot([i, i], [lo, hi], color=c, linewidth=0.9)
            ax.add_patch(Rectangle((i - 0.3, q1), 0.6, q3 - q1, facecolor=c, alpha=0.25, edgecolor=c, linewidth=0.8))
            ax.plot([i - 0.3, i + 0.3], [med, med], color=c, linewidth=2)
            numbers["africa_peaks"].setdefault(arm, {})[col] = {"median": float(med), "n": int(len(v)),
                                                                "p5_p25_p75_p95": [lo, q1, q3, hi]}
        if ref is not None:
            ax.axhline(ref, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(3), ["Daily\nLSTM", "$M_1$", "ERA5-\nLand"], fontsize=fb(7))
        ax.set_xlim(-0.6, 2.6)
        ax.tick_params(axis="y", labelsize=fb(7))
        ax.set_title(title, loc="left", fontsize=fb(8), weight="bold")
        style(ax)
        ax.grid(axis="x", visible=False)
    save(fig, "fig3_africa_examples")


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    numbers: dict = {}
    m1_vs_daily_lstm(numbers)
    fig4_maps(numbers)
    fig2_africa_era5(numbers)
    if (EX / "daily_series.csv").exists():
        fig3_africa_examples(numbers)
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print("wrote", OUT / "numbers.json")


if __name__ == "__main__":
    main()
