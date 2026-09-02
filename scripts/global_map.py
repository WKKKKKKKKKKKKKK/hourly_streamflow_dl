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
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

from common.config import resolve


def truncated(name: str, lo: float = 0.10, hi: float = 0.92):
    """A sequential colormap with its palest extreme removed.

    magma_r runs to near-white at its low end. Alpha sits below 1 at \SI{73}{\percent} of
    gauges, so that end carries most of the data, and pale markers on a white ground are
    close to invisible. Trimming the extremes keeps the ordering and keeps every point
    legible.
    """
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        f"{name}_trim", base(np.linspace(lo, hi, 256)))

STATIC_CSV = (
    "/ibex/project/c2266/abbaa0a/data/gscad_database/processed/20250630/"
    "hourly/dataframes/static.csv"
)
MIN_OBS_STD = 1e-3


# Axis limits, set once from the data by main(). A fixed (-180, 180) x (-60, 80) window
# spent 21% of every panel's width on empty ocean -- there is not one gauge west of -124 or
# east of 153 -- which shrank the clusters that do carry data.
XLIM = (-180.0, 180.0)
YLIM = (-60.0, 80.0)


def panel(ax, lon, lat, values, title, cmap, norm, extend="neither", seed=0,
          overlay=None):
    """Draw one map and return its mappable, WITHOUT a colorbar.

    Twelve panels once meant twelve colorbars, and the figure only has seven distinct
    scales: the M0/M1 pair of each row shares one by construction, and alpha and beta share
    the same log axis as each other. Colorbars are therefore attached per scale by the
    caller, spanning the axes they serve.
    """
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
    # A tag, not a title. Everything a title used to carry -- which model, which metric,
    # what the colours mean, the medians -- is enumerated in the caption instead, so twelve
    # panels of running text do not compete with twelve maps for the reader's attention.
    ax.annotate(title, (0.010, 0.960), xycoords="axes fraction", ha="left", va="top",
                fontsize=14, fontweight="semibold", color="#0b0b0b")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.grid(alpha=0.15, linewidth=0.4)
    ax.tick_params(labelsize=11)
    return handle


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
    # Height is tied to width on purpose. set_aspect("equal") holds the data's 2.6:1 shape, so
    # an axes rectangle taller than that letterboxes the map with white above and below -- which
    # is what narrowing the columns without shrinking the figure produced. 10.8 keeps each
    # rectangle at roughly the data aspect, so the maps fill them.
    fig, axes = plt.subplots(4, 3, figsize=(18.6, 9.6))
    # A 4x3 matrix: columns are M0 / M1 / difference, rows are KGE and its three
    # components. Every quantity therefore appears before, after, and as a change, and the
    # first two columns of a row share one scale so the pair can be compared by eye.
    marks = [0.25, 0.5, 0.71, 1.0, 1.41, 2.0, 4.0]
    log_ticks = [np.log2(m) for m in marks]
    tick_text = [f"{m:g}" for m in marks]

    def log2_of(values):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log2(np.where(values > 0, values, np.nan))

    # key, colourbar label, kind, colormap, shared norm for the M0/M1 pair. What each
    # panel shows is enumerated in the caption, not printed on the panel.
    # A diverging map is reserved for the difference column, where zero is a real centre
    # and the sign is the reading. The four value panels use sequential maps. For alpha and
    # beta that costs something worth naming: their ideal is 1.0, deviation either way is an
    # error, and a diverging map centred at 1.0 shows that directly. A sequential map does
    # not, so a tick line is drawn on their colorbars at 1.0 to put the ideal back.
    #
    # KGE and r keep viridis. The two ratios take magma so that a reader cannot mistake one
    # family of quantities for the other at a glance.
    # One ramp per quantity, four in all, because the four quantities do not share a scale
    # and a shared ramp made the same colour mean different things by row. r is linear on
    # [0.3, 0.95] with its ideal at the top; alpha and beta are logarithmic on [0.25, 4] with
    # their ideal at 1 in the middle. Reading a colour across those rows was never valid.
    #
    # Alpha and beta keep identical ranges, ticks and ideal marker, so the fact that those
    # two ARE directly comparable stays readable from the axis even though the hues differ.
    #
    # The four ramps were checked for separability rather than chosen by eye: the minimum
    # pairwise CIELAB distance across six samples of each is 16.6, above a floor of 15. Two earlier
    # picks were rejected by that check: GnBu for beta at 14.6 against viridis, and a
    # darkened Purples for r at 11.9 against the same.
    ROWS = (
        ("kge", "KGE", "score", "viridis", Normalize(vmin=-0.4, vmax=0.9)),
        # 0.3 to 1.0, not 0 to 1. r's 5th to 95th percentile is 0.35 to 0.92, so a full
        # 0-1 ramp spent most of its range on values that barely occur and rendered the two
        # panels as almost uniformly dark. The top is 0.95 rather than 1.0 so that its
        # end can be pointed like every other end in the figure: 1.0 is a correlation's
        # ceiling, nothing can exceed it, and that end had to be flat. Above 0.95 sit 1.2% of
        # values, comparable to KGE's 1.4%, so the point is earned rather than assumed.
        # BuPu over its upper three quarters. r's mass sits near the top of its range, so this
        # ramp needs a genuinely dark high end; Purples was too pale there, and darkening
        # Purples instead collided with viridis at a CIELAB distance of 11.9.
        ("kge_r", "$r$", "score", truncated("BuPu", 0.25, 1.0),
         Normalize(vmin=0.3, vmax=0.95)),
        ("kge_alpha", r"$\alpha$", "ratio", truncated("YlOrBr"), Normalize(vmin=-2.0, vmax=2.0)),
        ("kge_beta", r"$\beta$", "ratio", truncated("copper_r"), Normalize(vmin=-2.0, vmax=2.0)),
    )

    # Fit the window to everything that will be drawn, gauges and basins alike, with a
    # small margin. Computed rather than hard-coded so a different station set re-fits.
    all_lon = np.concatenate([lon] + ([africa["long"].to_numpy()] if africa is not None else []))
    all_lat = np.concatenate([lat] + ([africa["lat"].to_numpy()] if africa is not None else []))
    pad_x = 0.03 * (np.nanmax(all_lon) - np.nanmin(all_lon))
    pad_y = 0.05 * (np.nanmax(all_lat) - np.nanmin(all_lat))
    global XLIM, YLIM
    XLIM = (float(np.nanmin(all_lon) - pad_x), float(np.nanmax(all_lon) + pad_x))
    YLIM = (float(np.nanmin(all_lat) - pad_y), float(np.nanmax(all_lat) + pad_y))
    print(f"map window: lon {XLIM[0]:.0f} to {XLIM[1]:.0f}, lat {YLIM[0]:.0f} to {YLIM[1]:.0f}")

    pair_bars: list = []
    diff_bars: list = []
    for row, (key, bar_label, kind, cmap, norm) in enumerate(ROWS):
        ratio = kind == "ratio"
        for col, tag in enumerate(("M0", "M1")):
            column = f"{tag}_{key}" if key != "kge" else f"{tag}_kge"
            values = table[column].to_numpy()
            drawn = log2_of(values) if ratio else values
            handle = panel(axes[row, col], lon, lat, drawn,
                           f"({chr(ord('a') + row * 3 + col)})", cmap, norm,
                           extend="both" if ratio else "min",
                           overlay=(over_log(column) if ratio else over(column)))
        # One bar for the M0/M1 pair, deferred until the layout is final so it can be
        # placed from the axes' settled positions.
        # Derived from the values on both sides of the row, so a pointed end always means
        # data lies beyond it. r's top comes out flat, because 1 is a correlation's ceiling.
        lo, hi = norm.vmin, norm.vmax
        raw = np.concatenate([table[f"M0_{key}" if key != "kge" else "M0_kge"].to_numpy(),
                              table[f"M1_{key}" if key != "kge" else "M1_kge"].to_numpy()])
        if africa is not None:
            raw = np.concatenate([raw, africa[f"M0_{key}" if key != "kge" else "M0_kge"].to_numpy(),
                                  africa[f"M1_{key}" if key != "kge" else "M1_kge"].to_numpy()])
        seen = log2_of(raw) if ratio else raw
        below, above = bool(np.nanmin(seen) < lo), bool(np.nanmax(seen) > hi)
        extend = ("both" if below and above else
                  "min" if below else "max" if above else "neither")
        print(f"colourbar {bar_label}: range [{lo:g}, {hi:g}] extend={extend}")
        pair_bars.append((handle, [axes[row, 0], axes[row, 1]], ratio, bar_label, extend))

        # Third column: the plain difference M1 - M0, in the quantity's own units. For KGE
        # and r that is unambiguous -- higher is better, so red is better. For alpha and
        # beta it is NOT: their ideal is 1, so an increase helps a gauge below 1 and hurts
        # one above it. The title says so, and carries the fraction of the DEFICIT removed
        # beside it, which is the number that does mean better or worse.
        m0 = table[f"M0_{key}" if key != "kge" else "M0_kge"].to_numpy()
        m1 = table[f"M1_{key}" if key != "kge" else "M1_kge"].to_numpy()
        diff = m1 - m0
        span = float(np.nanpercentile(np.abs(diff), 90)) or 0.1
        a_over = None
        if africa is not None:
            am0 = africa[f"M0_{key}" if key != "kge" else "M0_kge"].to_numpy()
            am1 = africa[f"M1_{key}" if key != "kge" else "M1_kge"].to_numpy()
            a_over = (africa["long"].to_numpy(), africa["lat"].to_numpy(), am1 - am0)
        diff_handle = panel(axes[row, 2], lon, lat, diff,
                            f"({chr(ord('a') + row * 3 + 2)})", "RdBu_r",
                            TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span),
                            extend="both", overlay=a_over)
        diff_bars.append((diff_handle, axes[row, 2], f"change in {bar_label}"))

    # No suptitle. What it carried -- the agency composition, the run, and what the two
    # marker sizes mean -- belongs in the caption, which is generated from the same files
    # and can be read at reading size. Printed here so a run still reports it.
    agencies = table["source"].value_counts()
    print("composition: " + " | ".join(f"{n} {c}" for n, c in agencies.items()))
    top_frac = 0.985

    # One colorbar per panel, each hugging the map it belongs to, and the gaps between
    # panels pulled in. Scales are still shared wherever the colours are: M0 and M1 of a row
    # use one norm, so the pair is comparable at a glance, and alpha and beta use the same
    # log norm as each other.
    #
    # The four difference panels are the one place a shared scale is NOT applied. They are
    # all RdBu_r, but their magnitudes differ eightfold -- change in KGE spans +/-0.65 and
    # change in r +/-0.085 -- so one scale would render the r panel uniformly white. With a
    # bar beside every panel the differing spans are stated where they are used, which is
    # what makes per-panel bars safe here.
    # COL_GAP only has to clear the bar's tick labels and the next map's y labels; 0.042
    # was nearly an inch of white between columns. The figure width comes down with it, so
    # the maps keep filling their rectangles instead of being letterboxed inside wider ones
    # -- set_aspect("equal") holds the 2.6:1 data shape whatever the rectangle is.
    LEFT, RIGHT_PAD, COL_GAP = 0.022, 0.005, 0.016
    BAR_W, BAR_GAP = 0.006, 0.006
    width = (1.0 - LEFT - RIGHT_PAD - 2 * COL_GAP - 3 * (BAR_W + BAR_GAP)) / 3.0
    stride = width + BAR_GAP + BAR_W + COL_GAP
    xs = tuple(LEFT + i * stride for i in range(3))

    # Rows 1-3 carry no tick labels now, so the gap only has to separate the frames.
    fig.subplots_adjust(top=top_frac, bottom=0.055, hspace=0.07)
    last_row = axes.shape[0] - 1
    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            ax = axes[row, col]
            box = ax.get_position()
            ax.set_position([xs[col], box.y0, width, box.height])
            # Every panel shares one window, so repeating the numbers twelve times only
            # spends ink and forces the columns apart. Longitude on the bottom row,
            # latitude on the left column, nothing else.
            if row != last_row:
                ax.set_xticklabels([])
            if col != 0:
                ax.set_yticklabels([])

    def place_bar(handle, ax, label, ratio, extend):
        box = ax.get_position()
        cax = fig.add_axes([box.x1 + BAR_GAP, box.y0, BAR_W, box.height])
        bar = fig.colorbar(handle, cax=cax, extend=extend,
                           ticks=log_ticks if ratio else None)
        if ratio:
            bar.ax.set_yticklabels(tick_text)
            # The ideal value, marked because a sequential map gives it no colour of its own.
            bar.ax.axhline(0.0, color="#ffffff", lw=2.4, zorder=4)
            bar.ax.axhline(0.0, color="#0b0b0b", lw=1.1, zorder=5)
        bar.set_label(label, fontsize=11)
        bar.ax.tick_params(labelsize=10)

    # Every panel gets its own bar, including both halves of a shared-scale pair.
    for handle, axs, ratio, label, extend in pair_bars:
        for ax in axs:
            place_bar(handle, ax, label, ratio, extend)
    for handle, ax, label in diff_bars:
        place_bar(handle, ax, label, False, "both")

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
