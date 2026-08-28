"""Where on Earth does this work? (PLAN.md 5, item 5)

The 5-fold design exists so every station gets exactly one turn as a target, which
means the target-domain hourly KGE covers all ~8,900 stations rather than a fifth of
them. That is what makes a map worth drawing: no sampling luck, a real value at every
gauge.

Four panels, because the interesting content is not one field:

  M0            zero-shot skill -- what the source domain alone transfers
  M1            after daily-only fine-tuning
  M1 - M0       the gain, on a diverging scale centred at zero, so sign reads at a glance
  alpha at M0   variance ratio, the deficit that dominates the KGE gap everywhere

Written as PNG via a non-interactive backend, so it runs on a compute node with no
display. Coastlines are drawn from the station cloud itself rather than a basemap
package, which keeps this dependency-free -- the point is the spatial pattern of the
metric, not cartographic polish.

    python -m scripts.global_map --run outputs/runB_truedaily/diagnostics_allhours
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

STATIC_CSV = (
    "/ibex/project/c2266/abbaa0a/data/gscad_database/processed/20250630/"
    "hourly/dataframes/static.csv"
)
MIN_OBS_STD = 1e-3


def panel(ax, lon, lat, values, title, cmap, norm, label, extend="neither", seed=0,
          ticks=None, ticklabels=None):
    # RANDOM plot order. Sorting by value puts the extremes on top, and at this point
    # density that silently repaints whole regions in the tail colour: an earlier
    # version showed the gain panel as mostly deep red when the median gain is +0.026,
    # and alpha as mostly deep purple when its median is 0.854.
    order = np.random.default_rng(seed).permutation(len(values))
    handle = ax.scatter(
        lon[order], lat[order], c=values[order], s=4, cmap=cmap, norm=norm,
        linewidths=0, rasterized=True,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_aspect("equal")
    ax.grid(alpha=0.15, linewidth=0.4)
    ax.tick_params(labelsize=7)
    bar = plt.colorbar(handle, ax=ax, fraction=0.03, pad=0.02, extend=extend, ticks=ticks)
    if ticklabels is not None:
        bar.ax.set_yticklabels(ticklabels)
    bar.set_label(label, fontsize=8)
    bar.ax.tick_params(labelsize=7)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map the target-domain hourly metrics.")
    parser.add_argument("--run", default="outputs/runB_truedaily/diagnostics_allhours")
    parser.add_argument("--domain", default="target")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dpi", type=int, default=170)
    args = parser.parse_args()

    run = Path(args.run)
    out_dir = Path(args.out_dir) if args.out_dir else run / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(run / f"kge_components_{args.domain}.csv")
    before = len(table)
    table = table.loc[table["obs_std"] >= MIN_OBS_STD].copy()
    table["gain"] = table["M1_kge"] - table["M0_kge"]

    static = pd.read_csv(STATIC_CSV, comment="#", index_col=0)[["lat", "long"]]
    table = table.merge(static, left_on="station_id", right_index=True, how="left")
    missing = int(table[["lat", "long"]].isna().any(axis=1).sum())
    table = table.dropna(subset=["lat", "long"])
    print(f"{len(table)} stations mapped ({before - len(table)} dropped: "
          f"{before - len(table) - missing} degenerate, {missing} without coordinates)")

    # Every station appears exactly once across the folds, so no aggregation is needed;
    # verify that rather than assume it, since a duplicated station would double-plot.
    duplicated = int(table["station_id"].duplicated().sum())
    if duplicated:
        raise SystemExit(f"{duplicated} stations appear in more than one fold -- the fold "
                         "table is not a partition, so the map would double-plot them")

    lon = table["long"].to_numpy()
    lat = table["lat"].to_numpy()

    # 2x3, not 2x2. The top row is before / after / change for KGE; the bottom row was
    # only ever "before" for alpha, which showed the defect and left the reader to assume
    # from the KGE panels that it had been repaired. It is repaired only partly, and that
    # cannot be said with one panel: the median alpha moves 0.813 -> 0.851 while the share
    # of under-dispersed gauges barely shifts (71.4% -> 72.4%). What actually happens is a
    # tightening toward 1.0 from both sides, which needs before, after and change to show.
    fig, axes = plt.subplots(2, 3, figsize=(20.5, 8))
    kge_norm = Normalize(vmin=-0.4, vmax=0.9)
    panel(axes[0, 0], lon, lat, table["M0_kge"].to_numpy(),
          f"M0 zero-shot hourly KGE (median {table['M0_kge'].median():.3f})",
          "viridis", kge_norm, "KGE")
    panel(axes[0, 1], lon, lat, table["M1_kge"].to_numpy(),
          f"M1 after daily-only fine-tuning (median {table['M1_kge'].median():.3f})",
          "viridis", kge_norm, "KGE")
    # Span from the 10-90% range, not the extremes, so the bulk of the distribution is
    # resolvable; the colorbar arrows say values run past the ends.
    gain = table["gain"].to_numpy()
    span = float(max(abs(np.nanpercentile(gain, 10)), abs(np.nanpercentile(gain, 90)))) or 0.1
    clipped = float(np.mean(np.abs(gain) > span))
    panel(axes[0, 2], lon, lat, gain,
          f"gain M1 - M0 (median {table['gain'].median():+.3f}, "
          f"{(gain > 0).mean():.0%} improved; {clipped:.0%} beyond ±{span:.2f})",
          "RdBu_r", TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span), "ΔKGE", extend="both")
    # Alpha on a LOG axis, and this is not cosmetic. Alpha is a ratio, so 0.5 (half the
    # observed variability) and 2.0 (double it) are equally wrong, but a linear scale
    # centred at 1.0 put 0.5 halfway down the bar and pushed 2.0 off the top. With the
    # 10-90 percentile range it used before, 19.5% of stations fell outside the bar and
    # rendered at the end colour -- alpha 0.08 looked identical to 0.40 -- while 35% of
    # them crowded into 0.6-0.9, one narrow slice of orange. The panel came out a nearly
    # uniform orange wash that could not be read. log2 with a symmetric range of +/-2
    # (a factor of four each way) puts 1.0 at the centre by construction, spaces halving
    # and doubling equally, and covers 93% of stations.
    marks = [0.25, 0.5, 0.71, 1.0, 1.41, 2.0, 4.0]
    log_ticks = [np.log2(m) for m in marks]
    tick_text = [f"{m:g}" for m in marks]
    alphas = {}
    for col, tag in ((0, "M0"), (1, "M1")):
        alpha = table[f"{tag}_kge_alpha"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            log_alpha = np.log2(np.where(alpha > 0, alpha, np.nan))
        alphas[tag] = (alpha, log_alpha)
        inside = float(np.mean((alpha >= 0.25) & (alpha <= 4.0)))
        when = "zero-shot" if tag == "M0" else "after daily-only fine-tuning"
        panel(axes[1, col], lon, lat, log_alpha,
              f"alpha at {tag}, {when}: std(sim)/std(obs), log scale\n"
              f"median {np.nanmedian(alpha):.3f}; orange = swings too little "
              f"({np.mean(alpha < 1):.0%}), purple = too much",
              "PuOr", Normalize(vmin=-2.0, vmax=2.0), "alpha",
              extend="both", ticks=log_ticks, ticklabels=tick_text)

    # The repair, measured as distance to 1.0 in log space -- the thing KGE actually pays
    # for. Negative means alpha moved closer to 1.0, from either side. Plotting the raw
    # difference alpha_M1 - alpha_M0 would call an over-dispersed gauge getting worse an
    # "improvement" purely because the number went up.
    d0 = np.abs(alphas["M0"][1])
    d1 = np.abs(alphas["M1"][1])
    # Signed so that POSITIVE means improvement, and drawn with RdBu_r so red means
    # improvement -- the same convention as the gain panel above it. The first version
    # plotted (after - before) with RdBu, which made improvement negative and therefore
    # red, while its own caption said blue meant improvement. The caption and the colours
    # contradicted each other, and the two change panels would have used opposite
    # semantics for "better" in the same figure.
    repair = d0 - d1
    rspan = float(np.nanpercentile(np.abs(repair), 90)) or 0.5
    closer = float(np.nanmean(repair > 0))
    panel(axes[1, 2], lon, lat, repair,
          f"alpha repair: how much closer to 1.0 "
          f"({closer:.0%} improved; |log2 alpha| {np.nanmedian(d0):.3f} -> "
          f"{np.nanmedian(d1):.3f})\n"
          f"red = alpha moved toward 1.0, blue = away from it",
          "RdBu_r", TwoSlopeNorm(vmin=-rspan, vcenter=0.0, vmax=rspan),
          "reduction in |log2 alpha|", extend="both")

    # Say plainly what the station cloud covers: "global" describes the model, not the
    # gauge network. Africa, South America and mainland Asia contribute no stations.
    agencies = table["source"].value_counts()
    fig.suptitle(
        f"Target-domain hourly metrics, {len(table)} stations, one turn as target each "
        f"({run.parent.name})\n"
        + " | ".join(f"{name} {count}" for name, count in agencies.items()),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out_dir / f"global_map_{args.domain}.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path} ({path.stat().st_size / 1024**2:.1f} MiB)")

    # Regional medians, so the map has numbers behind it rather than only colour.
    bands = pd.cut(table["lat"], [-90, -30, 0, 30, 45, 60, 90],
                   labels=["<-30", "-30..0", "0..30", "30..45", "45..60", ">60"])
    regional = (
        table.assign(band=bands)
        .groupby("band", observed=True)
        .agg(n=("station_id", "size"), M0=("M0_kge", "median"), M1=("M1_kge", "median"),
             gain=("gain", "median"), alpha_M0=("M0_kge_alpha", "median"))
    )
    print("\nby latitude band:")
    print(regional.to_string(float_format=lambda v: f"{v: .4f}"))
    regional.to_csv(out_dir / f"by_latitude_{args.domain}.csv")


if __name__ == "__main__":
    main()
