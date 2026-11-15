"""How much of the world has daily streamflow but no hourly record?

The approach assumes that hourly labels are scarce; this script counts it. It is also the one claim a physics-embedded CONUS
model (Yang et al. 2026, arXiv:2609.06794) cannot answer, because its route needs a
river-network dataset that does not exist globally. So the number is worth getting right.

Counted from the GSCAD archive, which stores daily and hourly gauges side by side and
ships an explicit hourly-to-daily correspondence. The naive set difference on directory
names is WRONG: the US hourly gauges are filed under CAMELSH and their daily records
under USGS, which inflates "hourly with no daily" to 6,489. hourly_daily_map.json is the
authority and gives 505.

Two filters are applied, both taken from the training pipeline rather than invented here,
so the count is on the same footing as the experiments: at least 1,095 valid daily values
since 1980, and a catchment area of at most 10,000 km2.

What this measures is PUBLICLY ASSEMBLED availability, not what agencies hold. Many
record sub-daily and publish only daily. That is the right denominator for a data-driven
modelling study -- a model can only be trained on what can be obtained.

    python -m scripts.figures.label_availability
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from common.utils import setup_logging
from scripts.figures.plot_style import XLIM, YLIM, draw_borders


# Figures are read at column width in print, where 8.5 pt body text becomes
# marginal. One scale factor in one place, applied to every explicit size, so the
# relative hierarchy the figures already have is preserved.
FONT_SCALE = 1.3


def fs(size: float) -> float:
    """An explicit font size, scaled for legibility at print size."""
    return round(size * FONT_SCALE, 2)


ARCHIVE = Path("/ibex/project/c2266/abbaa0a/data/gscad_database/processed/20250630")
MIN_DAYS = 1095          # data.dataset min_data, in days
MAX_AREA_KM2 = 10000     # data.dataset max_area
START = "1980-01-01"
END = "2025-12-31"

# Categorical, two identities, in fixed order. Validated: worst adjacent pair is
# dE 24.7 under protanopia and 33.6 for normal vision, both above their floors.
C_HOURLY = "#2a78d6"
C_DAILY = "#eb6834"
# A darker step of the same blue ramp for the second network, so the bar reads as one
# family rather than as two unrelated categories.
C_HOURLY2 = "#184f95"
# The African evaluation basins. A third hue whose worst adjacent CVD separation against
# the orange is dE 9.2, above the floor but not the target, so it also differs in SHAPE and
# carries a dark edge: identity never rests on hue alone here.
C_AFRICA = "#1baf7a"
GREY = "#c3c2b7"
NETWORK_LABEL = {"CAMELSH": "United States", "BOMAustralia": "Australia",
                 "LamaHCE": "Austria", "Japan": "Japan", "Germany": "Germany",
                 "LamaHIce": "Iceland"}
INK, MUTED = "#0b0b0b", "#898781"


def valid_day_counts(cache: Path) -> pd.Series:
    """Valid daily values per gauge over the study window. ~13 GB, so cached."""
    if cache.exists():
        got = pd.read_csv(cache, index_col=0)["n_days"]
        return got
    ds = xr.open_dataset(ARCHIVE / "daily/dataframes/daily_q.nc").sel(
        time=slice(START, END))
    names = ds.stations.values.astype(str)
    counts = np.zeros(len(names), dtype=np.int32)
    for i in range(0, len(names), 4000):
        block = ds["q"].isel(stations=slice(i, i + 4000)).values
        counts[i:i + 4000] = np.isfinite(block).sum(axis=0)
    out = pd.Series(counts, index=names, name="n_days")
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_frame().to_csv(cache)
    return out


def hourly_gauges(logger) -> pd.DataFrame:
    """Where hourly records exist, read from the hourly side's own station table.

    The correspondence table is the right tool for COUNTING what share of daily gauges
    carry an hourly record, and the wrong tool for drawing a map. In the June snapshot it
    omits one network entirely (Germany, 494 gauges), which would put on the map the claim
    that German rivers have no hourly data while 458 German hourly gauges are among
    the training and target gauges here. Plotting the hourly stations at their own coordinates
    needs no correspondence at all.
    """
    from aqua_fetch import rr
    alias = {"BOMAustralia": "CAMELS_AUS", "Germany": "CAMELS_DE"}
    registry = set(rr.DATASETS)
    frame = pd.read_csv(ARCHIVE / "hourly/dataframes/static.csv", comment="#",
                        low_memory=False)
    frame = frame.rename(columns={frame.columns[0]: "stations"}).set_index("stations")
    frame["network"] = [s.split("__")[0] for s in frame.index]
    frame["public"] = [alias.get(n, n) in registry for n in frame["network"]]
    keep = (frame[["long", "lat", "area"]].notna().all(axis=1)
            & (frame["area"] <= MAX_AREA_KM2))
    out = frame.loc[keep, ["long", "lat", "area", "network"]]
    logger.info("hourly gauges plotted: %s across %d networks",
                f"{len(out):,}", out["network"].nunique())
    return out


AFRICA_SOURCES = "GRDC-Caravan / GRDC / ADHI"


def african_basins():
    """Centroids of the African catchments this study evaluates, or None if absent."""
    path = Path("africa/africa_basins.csv")
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    columns = {c.lower(): c for c in frame.columns}
    lon = columns.get("long") or columns.get("lon") or columns.get("longitude")
    lat = columns.get("lat") or columns.get("latitude")
    if not (lon and lat):
        return None
    return frame[[lon, lat]].rename(columns={lon: "long", lat: "lat"}).dropna()


def build_table(logger) -> pd.DataFrame:
    # Named for the snapshot. A bare filename silently served December counts against a
    # June station list when ARCHIVE changed, and the join hid it by dropping to NaN.
    days = valid_day_counts(
        Path(f"outputs/v2_label_availability/valid_day_counts_{ARCHIVE.name}.csv"))
    static = pd.read_csv(
        ARCHIVE / "daily/dataframes/static.csv", comment="#",
        usecols=["stations", "long", "lat", "area"]).set_index("stations")
    pairing = json.loads((ARCHIVE / "hourly/hourly_daily_map.json").read_text())
    hourly_backed = set(pairing.values())
    # The June 2025 map names 9,856 gauges across six networks; the December one names
    # 12,845 across more. The floor guards against reading an empty or truncated file, not
    # against a particular snapshot.
    if len(hourly_backed) < 9000:
        raise SystemExit(f"only {len(hourly_backed)} daily ids are named by the hourly "
                         "map; the archive layout has changed")

    table = days.to_frame().join(static, how="left")
    table["hourly"] = [s in hourly_backed for s in table.index]
    table["network"] = [s.split("__")[0] for s in table.index]
    from aqua_fetch import rr
    # A handful of networks are filed under a country name here and a dataset class name
    # in the package. Everything else matches on the name itself.
    alias = {"BOMAustralia": "CAMELS_AUS", "Germany": "CAMELS_DE", "Sweden": "CAMELS_SE",
             "Denmark": "CAMELS_DK", "France": "CAMELS_FR"}
    registry = set(rr.DATASETS)
    table["public"] = [alias.get(n, n) in registry for n in table["network"]]
    logger.info("archive holds %s daily gauges, %s of them hourly-backed",
                f"{len(table):,}", f"{table['hourly'].sum():,}")
    return table


def summarise(table: pd.DataFrame, logger) -> dict:
    keep = (table["n_days"] >= MIN_DAYS) & (table["area"] <= MAX_AREA_KM2)
    steps = {
        "archive": pd.Series(True, index=table.index),
        "record": table["n_days"] >= MIN_DAYS,
        "record_and_area": keep,
        "public_release": keep & table["public"],
    }
    payload = {"filters": {"min_days": MIN_DAYS, "max_area_km2": MAX_AREA_KM2,
                           "window": [START, END]}, "steps": {}}
    for name, mask in steps.items():
        sub = table[mask]
        n_h = int(sub["hourly"].sum())
        payload["steps"][name] = {
            "n_daily": int(len(sub)), "n_hourly_backed": n_h,
            "n_daily_only": int(len(sub) - n_h),
            "ratio_daily_over_hourly": float(len(sub) / n_h),
        }
        logger.info("%-16s daily %6s  hourly-backed %6s  DAILY-ONLY %6s  ratio %.2fx",
                    name, f"{len(sub):,}", f"{n_h:,}", f"{len(sub) - n_h:,}",
                    len(sub) / n_h)

    # Gauges dropped only because the area attribute is absent, so the headline count is
    # reported as the conservative bound it is rather than as an exact figure.
    rec = table[table["n_days"] >= MIN_DAYS]
    payload["daily_only_missing_area"] = int(rec.loc[~rec["hourly"], "area"].isna().sum())

    # Whole networks with no hourly record anywhere. This is the number that survives
    # being read aloud: not a ratio, a list of countries.
    per_net = table.groupby("network")["hourly"].agg(["size", "sum"])
    barren = per_net[per_net["sum"] == 0].sort_values("size", ascending=False)
    payload["networks_total"] = int(len(per_net))
    payload["networks_without_any_hourly"] = int(len(barren))
    payload["gauges_in_those_networks"] = int(barren["size"].sum())
    payload["barren_networks"] = {k: int(v) for k, v in barren["size"].items()}
    logger.info("%d of %d daily networks contain no hourly gauge at all (%s gauges)",
                len(barren), len(per_net), f"{barren['size'].sum():,}")
    return payload


def figure(table: pd.DataFrame, hourly: pd.DataFrame, payload: dict, out: Path) -> Path:
    """Two layers, not one classification: daily gauges, then hourly gauges over them."""
    keep = table[(table["n_days"] >= MIN_DAYS) & (table["area"] <= MAX_AREA_KM2)
                 & table["long"].notna() & table["lat"].notna()]
    fig = plt.figure(figsize=(13.0, 7.0))
    grid = fig.add_gridspec(2, 1, height_ratios=(10, 1.9), hspace=0.20)
    ax, bx = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])

    draw_borders(ax)
    ax.scatter(keep["long"], keep["lat"], c=C_DAILY, s=3.2, linewidths=0,
               alpha=0.7, rasterized=True, zorder=2)
    # Hourly on top and slightly larger. Overlap is the point: a blue dot sitting on an
    # orange one is a gauge recorded at both resolutions, and the orange showing through
    # anywhere else is a gauge that is not.
    ax.scatter(hourly["long"], hourly["lat"], c=C_HOURLY, s=4.0, linewidths=0,
               alpha=0.85, rasterized=True, zorder=3)
    # The African evaluation basins, on top of both layers. Every one of them sits in the
    # orange, and the experiment asks what can
    # be done there. Drawn larger with a dark edge because 294 basins against 25,075 gauges
    # would otherwise be invisible, and because they are a different kind of site -- scored
    # on daily observations, since the hourly ones the other markers stand for do not exist.
    africa = african_basins()
    if africa is not None:
        ax.scatter(africa["long"], africa["lat"], marker="D", s=15,
                   facecolor=C_AFRICA, edgecolors="#0b0b0b", linewidths=0.4,
                   zorder=4, rasterized=True)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.grid(alpha=0.15, linewidth=0.4)
    # Degrees with a hemisphere letter. A bare "-150" on a world map is ambiguous to
    # anyone who does not already know the axis is longitude.
    ax.set_xticks([-150, -100, -50, 0, 50, 100, 150])
    ax.set_yticks([-60, -40, -20, 0, 20, 40, 60, 80])
    ax.set_xticklabels([f"{abs(v)}\u00b0{'W' if v < 0 else ('E' if v > 0 else '')}"
                        for v in (-150, -100, -50, 0, 50, 100, 150)])
    ax.set_yticklabels([f"{abs(v)}\u00b0{'S' if v < 0 else ('N' if v > 0 else '')}"
                        for v in (-60, -40, -20, 0, 20, 40, 60, 80)])
    ax.tick_params(labelsize=fs(10), colors=MUTED)
    for handle, label in ((C_DAILY, f"daily record  ({len(keep):,})"),
                          (C_HOURLY, f"hourly record  ({len(hourly):,})")):
        ax.scatter([], [], c=handle, s=34, linewidths=0, label=label)
    if africa is not None:
        ax.scatter([], [], marker="D", s=34, facecolor=C_AFRICA, edgecolors="#0b0b0b",
                   linewidths=0.4,
                   label=f"{AFRICA_SOURCES}, daily  ({len(africa):,})")
    ax.legend(loc="lower left", frameon=False, fontsize=fs(11), labelcolor=INK,
              handletextpad=0.4, borderpad=0.2)

    # One stacked row: which networks the hourly gauges belong to. Two networks carry most
    # of them, and that concentration is the claim the figure exists to support, so it is
    # drawn rather than left to the caption. The remaining networks fold into one segment
    # instead of becoming four more colours nobody can tell apart at this size.
    counts = hourly["network"].value_counts()
    top = counts.head(2)
    rest = int(counts.iloc[2:].sum())
    total = int(counts.sum())
    segments = [(int(top.iloc[0]), NETWORK_LABEL.get(top.index[0], top.index[0]), C_HOURLY),
                (int(top.iloc[1]), NETWORK_LABEL.get(top.index[1], top.index[1]), C_HOURLY2),
                (rest, f"{len(counts) - 2} other networks", GREY)]
    left = 0.0
    for value, label, colour in segments:
        bx.barh(0, value, left=left, color=colour, height=0.55)
        # Name above the bar in ink, count inside it. A two-line label inside a 0.55-high
        # bar was clipped at both ends, and the clipping was silent.
        bx.annotate(label, (left + value / 2, 0.52), color=INK, ha="center", va="bottom",
                    fontsize=fs(9.5), fontweight="semibold")
        bx.annotate(f"{value:,}  ({value / total:.0%})", (left + value / 2, 0.02),
                    color="#ffffff" if colour != GREY else INK,
                    ha="center", va="center", fontsize=fs(9.5), fontweight="semibold")
        left += value + total * 0.004
    bx.set_xlim(0, left)
    bx.set_ylim(-0.95, 1.45)
    bx.axis("off")
    bx.annotate(f"Where the {total:,} hourly gauges are",
                (0, -0.68), ha="left", va="center", fontsize=fs(9.5), color=MUTED)

    fig.subplots_adjust(left=0.045, right=0.995, top=0.985, bottom=0.045)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="outputs/v2_label_availability", type=Path)
    parser.add_argument("--figure", default="reports/figures/label_availability.png",
                        type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.out_dir / "label_availability.log")

    table = build_table(logger)
    payload = summarise(table, logger)
    keep = (table["n_days"] >= MIN_DAYS) & (table["area"] <= MAX_AREA_KM2)
    table.loc[keep & ~table["hourly"]].to_csv(args.out_dir / "daily_only_gauges.csv")
    (args.out_dir / "label_availability.json").write_text(json.dumps(payload, indent=2))
    path = figure(table, hourly_gauges(logger), payload, args.figure)
    logger.info("wrote %s and %s", args.out_dir / "label_availability.json", path)


if __name__ == "__main__":
    main()
