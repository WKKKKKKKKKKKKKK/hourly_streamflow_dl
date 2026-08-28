"""Score M0, M1 and ERA5-Land on the SAME African basin-days, so the three are comparable.

The in-situ experiment reports M0 and M1 per basin over its own validation window
(``v2_africa_insitu_summary/ensemble_per_basin_M{0,1}.csv``). ERA5-Land's per-basin scores
that already exist come from a different experiment path -- the temperate-transfer Africa
run of section 3.1 -- over a different period. Printing one beside the other would compare
scores computed on different days and call it a three-way comparison.

So this re-scores all three on exactly the basin-days the in-situ ensemble was scored on,
with the same ``score()`` the in-situ ensemble used, and writes one table. The pooled
ERA5-Land median it reports is therefore not the -0.3336 quoted elsewhere in the report:
that number is correct for its own run and window, and this one is correct for this one.

    python -m scripts.africa_daily_three_way
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from common.utils import setup_logging
from scripts.africa_insitu_ensemble import score, summarise

DEFAULT_ERA5 = "/ibex/user/kongw0a/era5_land_africa/era5_land_africa_daily_runoff.nc"


def load_era5_daily(path: Path, logger=None) -> pd.DataFrame:
    """Tidy (station_id, date, era5_land) in mm/d from the basin-averaged daily product."""
    with xr.open_dataset(path) as ds:
        stations = [str(x) for x in ds["station"].values]
        dates = pd.DatetimeIndex(ds["date"].values)
        values = np.asarray(ds["runoff"].values, dtype=np.float64)
    frame = pd.DataFrame(values, index=stations, columns=dates)
    tidy = (frame.stack(future_stack=True).rename("era5_land").reset_index()
            .rename(columns={"level_0": "station_id", "level_1": "date"}))
    if logger:
        logger.info("ERA5-Land daily: %d stations x %d dates, %.1f%% finite",
                    len(stations), len(dates), 100 * np.isfinite(values).mean())
    return tidy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score M0, M1 and ERA5-Land on identical African basin-days.")
    parser.add_argument("--insitu-summary", default="outputs/v2_africa_insitu_summary", type=Path)
    parser.add_argument("--era5-daily", default=DEFAULT_ERA5, type=Path)
    parser.add_argument("--out-dir", default="outputs/v2_africa_hourly", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.out_dir / "daily_three_way.log")

    merged = None
    for tag in ("M0", "M1"):
        path = args.insitu_summary / f"ensemble_series_{tag}.csv.gz"
        frame = pd.read_csv(path, parse_dates=["date"])[["station_id", "date", "obs", "ensemble"]]
        frame = frame.rename(columns={"ensemble": tag})
        merged = frame if merged is None else merged.merge(
            frame.drop(columns=["obs"]), on=["station_id", "date"], how="inner")
    merged["station_id"] = merged.station_id.astype(str)
    logger.info("in-situ validation days: %d rows over %d basins, %s .. %s",
                len(merged), merged.station_id.nunique(),
                merged.date.min().date(), merged.date.max().date())

    era5 = load_era5_daily(args.era5_daily, logger=logger)
    era5["station_id"] = era5.station_id.astype(str)
    # Inner join, so every score below rests on the identical set of basin-days. An outer
    # join would let ERA5 be scored on days the models were not, which is the whole thing
    # this script exists to avoid.
    both = merged.merge(era5, on=["station_id", "date"], how="inner")
    logger.info("after intersecting with ERA5-Land: %d rows over %d basins (%.1f%% of the "
                "model rows kept)", len(both), both.station_id.nunique(),
                100 * len(both) / len(merged))

    tables = {}
    for column in ("M0", "M1", "era5_land"):
        scored = score(both, column)
        tables[column] = scored.set_index("station_id")
        logger.info("%s: %d basins scored | median KGE %+.4f | median NSE %+.4f",
                    column, len(scored), scored.kge.median(), scored.nse.median())

    wide = None
    for column, table in tables.items():
        slim = table[["kge", "nse", "kge_r", "kge_alpha", "kge_beta", "n_days"]]
        slim = slim.add_suffix(f"_{column}")
        wide = slim if wide is None else wide.join(slim, how="inner")
    wide = wide.reset_index()
    out_csv = args.out_dir / "daily_three_way_per_basin.csv"
    wide.to_csv(out_csv, index=False)

    summary = {"n_basins": int(len(wide)), "n_basin_days": int(len(both)),
               "window": [str(both.date.min().date()), str(both.date.max().date())]}
    for column in tables:
        summary[column] = summarise(tables[column].reset_index())
    # The paired statement, which the pooled medians cannot make: on how many basins does
    # each model beat the reanalysis on that basin's own days?
    for column in ("M0", "M1"):
        beats = (wide[f"kge_{column}"] > wide["kge_era5_land"]).mean()
        summary[f"share_of_basins_{column}_beats_era5_land"] = float(beats)
        logger.info("%s beats ERA5-Land on %.1f%% of the %d basins", column,
                    100 * beats, len(wide))
    with open(args.out_dir / "daily_three_way_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("wrote %s and daily_three_way_summary.json", out_csv)


if __name__ == "__main__":
    main()
