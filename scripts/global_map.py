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
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

from common.config import resolve

STATIC_CSV = (
    "/ibex/project/c2266/abbaa0a/data/gscad_database/processed/20250630/"
    "hourly/dataframes/static.csv"
)
MIN_OBS_STD = 1e-3


def panel(ax, lon, lat, values, title, cmap, norm, label, extend="neither", seed=0,
          ticks=None, ticklabels=None, overlay=None):
    # RANDOM plot order. Sorting by value puts the extremes on top, and at this point
    # density that silently repaints whole regions in the tail colour: an earlier
    # version showed the gain panel as mostly deep red when the median gain is +0.026,
    # and alpha as mostly deep purple when its median is 0.854.
    order = np.random.default_rng(seed).permutation(len(values))
    handle = ax.scatter(
        lon[order], lat[order], c=values[order], s=4, cmap=cmap, norm=norm,
        linewidths=0, rasterized=True,
    )
    if overlay is not None:
        # African basins, drawn LARGER and with a dark edge. 282 of them against 8,843
        # gauges would otherwise vanish, and the edge is what tells a reader that these
        # markers are a different kind of measurement: the gauges are scored on hourly
        # observations, the basins on daily ones, because no African catchment has hourly
        # discharge. Same metric, same colour scale, different observation resolution --
        # a distinction that has to be visible in the marks, not only in the caption.
        o_lon, o_lat, o_values = overlay
        ax.scatter(o_lon, o_lat, c=o_values, s=30, cmap=cmap, norm=norm,
                   linewidths=0.6, edgecolors="#0b0b0b", marker="o", zorder=4,
                   rasterized=True)
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
    parser.add_argument("--africa", default=None,
                        help="Directory with ensemble_per_basin_M{0,1}.csv, to overlay the "
                             "African basins. They are scored on DAILY observations because "
                             "no African catchment has hourly discharge, so they are drawn "
                             "as larger dark-edged markers rather than blended in.")
    parser.add_argument("--basins", default="africa/africa_basins.gpkg")
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

    # The African basins, on the same axes. Phase I's premise is "pretend these gauges have
    # no hourly observations and supervise with daily aggregates only". On the target domain
    # that premise is SIMULATED; in Africa it is genuine -- the continent has daily discharge
    # and no hourly discharge at all. Same pretrained models, same daily-only fine-tuning,
    # so M0 and M1 mean structurally the same thing in both places. What differs is the
    # observation the score is computed against: hourly for the gauges, daily for the basins.
    africa = None
    if args.africa:
        summary = Path(args.africa)
        m0 = pd.read_csv(summary / "ensemble_per_basin_M0.csv").set_index("station_id")
        m1 = pd.read_csv(summary / "ensemble_per_basin_M1.csv").set_index("station_id")
        joined = m0.join(m1, lsuffix="_M0", rsuffix="_M1", how="inner")
        basins = gpd.read_file(resolve(args.basins))
        basins["station_id"] = basins["station_id"].astype(str)
        # The centroid is a point on a map of the whole world; the polygon is irrelevant at
        # this scale. Taken in an equal-area projection and converted back, because a
        # centroid of raw lat/lon is not a centroid of the shape -- shapely warns about
        # exactly this, and the warning is right even if the error is small here.
        projected = basins.set_index("station_id").to_crs(6933)
        centroids = projected.geometry.centroid.to_crs(4326)
        joined = joined.join(pd.DataFrame({"long": centroids.x, "lat": centroids.y}),
                             how="inner")
        africa = joined.rename(columns={
            "kge_M0": "M0_kge", "kge_M1": "M1_kge",
            "kge_r_M0": "M0_kge_r", "kge_r_M1": "M1_kge_r",
            "kge_alpha_M0": "M0_kge_alpha", "kge_beta_M0": "M0_kge_beta",
            "kge_alpha_M1": "M1_kge_alpha", "kge_beta_M1": "M1_kge_beta"})
        # Every column the panels ask for must exist, or a panel silently loses its overlay.
        need = [f"{m}_kge_{c}" for m in ("M0", "M1") for c in ("r", "alpha", "beta")]
        missing = [c for c in need if c not in africa]
        if missing:
            raise SystemExit(f"African table is missing {missing}; the per-basin files "
                             "changed shape and the rename map above is stale")
        africa["gain"] = africa["M1_kge"] - africa["M0_kge"]
        print(f"{len(africa)} African basins overlaid (scored on DAILY observations; "
              f"the gauges above are scored on hourly)")

    def over(column):
        """The overlay triple for one column, or None when Africa was not requested."""
        if africa is None or column not in africa:
            return None
        return (africa["long"].to_numpy(), africa["lat"].to_numpy(),
                africa[column].to_numpy())

    def afr(column, fmt):
        """Africa's own median for a column, as a title suffix. Empty when not overlaid.

        ``fmt`` carries the whole suffix, separator included, so a panel that is not
        overlaid gets nothing rather than a dangling separator.

        Stated separately and never pooled with the gauges: the gauge medians are computed
        against hourly observations and Africa's against daily ones, so a single combined
        median would average two different measurements.
        """
        if africa is None or column not in africa:
            return ""
        return fmt.format(float(np.nanmedian(africa[column])))

    def over_log(column):
        """Same, on the log2 axis the two ratio panels use."""
        raw = over(column)
        if raw is None:
            return None
        o_lon, o_lat, values = raw
        with np.errstate(divide="ignore", invalid="ignore"):
            return o_lon, o_lat, np.log2(np.where(values > 0, values, np.nan))

    # 2x3, not 2x2. The top row is before / after / change for KGE; the bottom row was
    # only ever "before" for alpha, which showed the defect and left the reader to assume
    # from the KGE panels that it had been repaired. It is repaired only partly, and that
    # cannot be said with one panel: the median alpha moves 0.813 -> 0.851 while the share
    # of under-dispersed gauges barely shifts (71.4% -> 72.4%). What actually happens is a
    # tightening toward 1.0 from both sides, which needs before, after and change to show.
    fig, axes = plt.subplots(3, 3, figsize=(20.5, 12))
    kge_norm = Normalize(vmin=-0.4, vmax=0.9)
    panel(axes[0, 0], lon, lat, table["M0_kge"].to_numpy(),
          f"M0 zero-shot KGE (gauges, hourly: {table['M0_kge'].median():.3f}"
          f"{afr('M0_kge', ' | basins, daily {:.3f}')})",
          "viridis", kge_norm, "KGE", overlay=over("M0_kge"))
    panel(axes[0, 1], lon, lat, table["M1_kge"].to_numpy(),
          f"M1 after daily-only fine-tuning (gauges {table['M1_kge'].median():.3f}"
          f"{afr('M1_kge', ' | basins {:.3f}')})",
          "viridis", kge_norm, "KGE", overlay=over("M1_kge"))
    # Span from the 10-90% range, not the extremes, so the bulk of the distribution is
    # resolvable; the colorbar arrows say values run past the ends.
    gain = table["gain"].to_numpy()
    span = float(max(abs(np.nanpercentile(gain, 10)), abs(np.nanpercentile(gain, 90)))) or 0.1
    clipped = float(np.mean(np.abs(gain) > span))
    panel(axes[0, 2], lon, lat, gain,
          f"gain M1 - M0 (gauges {table['gain'].median():+.3f}, "
          f"{(gain > 0).mean():.0%} improved"
          f"{afr('gain', ' | basins {:+.3f}')})",
          "RdBu_r", TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span), "ΔKGE",
          extend="both", overlay=over("gain"))
    # Alpha on a LOG axis, and this is not cosmetic. Alpha is a ratio, so 0.5 (half the
    # observed variability) and 2.0 (double it) are equally wrong, but a linear scale
    # centred at 1.0 put 0.5 halfway down the bar and pushed 2.0 off the top. With the
    # 10-90 percentile range it used before, 19.5% of stations fell outside the bar and
    # rendered at the end colour -- alpha 0.08 looked identical to 0.40 -- while 35% of
    # them crowded into 0.6-0.9, one narrow slice of orange. The panel came out a nearly
    # uniform orange wash that could not be read. log2 with a symmetric range of +/-2
    # (a factor of four each way) puts 1.0 at the centre by construction, spaces halving
    # and doubling equally, and covers 93% of stations.
    # Both alpha and beta are RATIOS, so both go on a log axis: halving and doubling are
    # equally wrong and belong equidistant from the white centre. A linear scale centred at
    # 1.0 put 0.5 halfway down the bar and pushed 2.0 off the top, which rendered the alpha
    # panel as a near-uniform orange wash with 19.5% of gauges clipped to the end colour.
    # Rows: result, then where the zero-shot model loses, then what the daily signal fixes.
    # Every KGE component appears in both of the lower rows. An earlier version mapped only
    # alpha at M0 and alpha's repair, on the reasoning that r "barely moves" -- 0.797 to
    # 0.812 -- and would render as a uniform panel. That conflated a small median CHANGE
    # with a uniform spatial distribution: r's level spans 0.091 to 0.519 across gauges at
    # the 10-90 percentiles, so it maps perfectly well. And once Africa is overlaid the
    # reasoning fails outright, because Africa's r repair is 32%, not 7%.
    marks = [0.25, 0.5, 0.71, 1.0, 1.41, 2.0, 4.0]
    log_ticks = [np.log2(m) for m in marks]
    tick_text = [f"{m:g}" for m in marks]

    def deficit(values, component):
        """Distance from the ideal: 1 - r for the correlation, |log2 x| for the two ratios.

        A ratio's 0.5 and 2.0 are equally wrong and their arithmetic mean is not 1, so the
        two kinds of component cannot share one definition of "how far off".
        """
        if component == "r":
            return 1.0 - values
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.abs(np.log2(np.where(values > 0, values, np.nan)))

    SPEC = (
        ("r", "r at M0 = correlation", "orange = timing too poor", "r"),
        ("alpha", "alpha at M0 = std(sim)/std(obs)", "orange = swings too little", "alpha"),
        ("beta", "beta at M0 = mean(sim)/mean(obs)", "orange = too little water", "beta"),
    )
    for col, (component, heading, sense, label) in enumerate(SPEC):
        column = f"M0_kge_{component}"
        values = table[column].to_numpy()
        lo, hi = np.nanpercentile(values, [10, 90])
        if component == "r":
            # r is not a ratio: it lives on (-inf, 1] with 1 the ideal, so it gets a
            # sequential scale running to 1 rather than a log axis centred on it.
            panel(axes[1, col], lon, lat, values,
                  f"{heading} (gauges {np.nanmedian(values):.3f}"
                  f"{afr(column, ' | basins {:.3f}')})\n"
                  f"dark = timing wrong; 10-90%: {lo:.2f}-{hi:.2f}",
                  "viridis", Normalize(vmin=0.0, vmax=1.0), label,
                  extend="min", overlay=over(column))
        else:
            inside = float(np.mean((values >= 0.25) & (values <= 4.0)))
            with np.errstate(divide="ignore", invalid="ignore"):
                log_values = np.log2(np.where(values > 0, values, np.nan))
            panel(axes[1, col], lon, lat, log_values,
                  f"{heading}, log scale (gauges {np.nanmedian(values):.3f}"
                  f"{afr(column, ' | basins {:.3f}')})\n"
                  f"{sense} ({np.mean(values < 1):.0%}), purple = too much; "
                  f"10-90%: {lo:.2f}-{hi:.2f}",
                  "PuOr", Normalize(vmin=-2.0, vmax=2.0), label,
                  extend="both", ticks=log_ticks, ticklabels=tick_text,
                  overlay=over_log(column))

    # Bottom row: how much of each deficit the daily signal removes. Signed so POSITIVE is
    # improvement and drawn with RdBu_r, so red is improvement here exactly as in the gain
    # panel at the top -- the two change rows must not use opposite conventions.
    for col, (component, _, _, _) in enumerate(SPEC):
        d0 = deficit(table[f"M0_kge_{component}"].to_numpy(), component)
        d1 = deficit(table[f"M1_kge_{component}"].to_numpy(), component)
        repair = d0 - d1
        rspan = float(np.nanpercentile(np.abs(repair), 90)) or 0.5
        closer = float(np.nanmean(repair > 0))
        removed = 100 * (np.nanmedian(d0) - np.nanmedian(d1)) / np.nanmedian(d0)
        a_over = None
        a_note = ""
        if africa is not None:
            ad0 = deficit(africa[f"M0_kge_{component}"].to_numpy(), component)
            ad1 = deficit(africa[f"M1_kge_{component}"].to_numpy(), component)
            a_over = (africa["long"].to_numpy(), africa["lat"].to_numpy(), ad0 - ad1)
            a_removed = 100 * (np.nanmedian(ad0) - np.nanmedian(ad1)) / np.nanmedian(ad0)
            a_note = (f"basins {np.nanmean(ad0 - ad1 > 0):.0%} improved, "
                      f"{a_removed:.0f}% of deficit.  ")
        panel(axes[2, col], lon, lat, repair,
              f"{component} repair: gauges {closer:.0%} improved, "
              f"{removed:.0f}% of deficit\n"
              f"{a_note}red = moved toward the ideal, blue = away",
              "RdBu_r", TwoSlopeNorm(vmin=-rspan, vcenter=0.0, vmax=rspan),
              f"reduction in {component} deficit", extend="both", overlay=a_over)

    # Say plainly what the station cloud covers: "global" describes the model, not the
    # gauge network. Africa, South America and mainland Asia contribute no stations.
    agencies = table["source"].value_counts()
    overlay_note = ""
    if africa is not None:
        # The distinction the marker size carries, said in words as well: same models, same
        # daily-only fine-tuning, but the gauges are scored against hourly observations and
        # the basins against daily ones, because no African catchment has hourly discharge.
        # Phase I's premise is simulated on the gauges and genuine in Africa.
        overlay_note = (f"\nSmall dots: {len(table)} target gauges, scored on HOURLY "
                        f"observations, Phase I's premise simulated.  "
                        f"Large outlined dots: {len(africa)} African basins, scored on "
                        f"DAILY observations because no hourly discharge exists there, "
                        f"Phase I's premise genuine.")
    fig.suptitle(
        f"Target-domain hourly metrics, {len(table)} stations, one turn as target each "
        f"({run.parent.name})\n"
        + " | ".join(f"{name} {count}" for name, count in agencies.items())
        + overlay_note,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955 if africa is None else 0.925))
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
