"""Turn the hourly ERA5-Land grids into per-basin runoff in mm/h.

The daily sibling (``scripts.basin_average_era5_land``) reads one step per day, the
00:00 UTC stamp that already holds the whole day's total. Hourly needs two extra
conventions handled explicitly, and getting either wrong would invent a timing error
in exactly the figure that is meant to show timing:

* **De-accumulation.** ERA5-Land runoff accumulates from 00 UTC and resets each day,
  and each step is stamped at the END of its accumulation period. So the step at
  01:00 on day D holds the first hour of day D, and the step at 00:00 on day D+1
  closes day D. Grouping by "accumulation day" and differencing inside the group,
  with 0 as the pre-01:00 baseline, recovers the hourly increment.

* **Stamp convention.** After differencing, the value stamped t covers the hour
  [t-1h, t). The model's hourly output is hour-BEGINNING: its slot h of date d covers
  [d h:00, d h+1:00). The ERA5 axis is therefore shifted back one hour so the two
  series mean the same thing. Skipping this shift would show ERA5 peaking one hour
  after the model for purely bookkeeping reasons.

**Units.** ERA5-Land accumulations are metres of water equivalent; x1000 gives mm.
Since each increment is one hour, that is mm/h -- the model's own target unit.

**What this is and is not.** ERA5-Land runoff is the land-surface scheme's runoff
GENERATION per grid cell. ERA5-Land contains no river routing, so the basin average
is water leaving the soil column, not water passing a gauge. At a daily step the
difference is modest; within a day it is the dominant one -- routing is precisely what
delays and smooths an hourly hydrograph. Treat this series as a physical baseline
whose sub-daily shape is expected to be too fast, not as an hourly reference.

    python -m scripts.basin_average_era5_land_hourly \
        --era5-dir /ibex/user/kongw0a/era5_land_africa_hourly3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from common.config import add_common_args, load_config, resolve
from common.utils import setup_logging
from scripts.basin_average_era5_land import cell_weights, normalise_names
from scripts.era5_tiles import open_tile

# ERA5-Land short names; the CDS netCDF uses the short form.
RUNOFF_ALIASES = ("runoff", "ro")


def deaccumulate(cumulative: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
    """Hourly increments from ERA5-Land's within-day cumulative curve.

    ``cumulative`` is (time,) for one basin. The accumulation day of a step is the day
    it accumulates INTO, so the 00:00 stamp belongs to the previous day.
    """
    accum_day = (times - pd.Timedelta(hours=1)).normalize()
    out = np.full(cumulative.shape, np.nan, dtype=np.float64)
    frame = pd.DataFrame({"c": cumulative, "day": accum_day})
    for _, idx in frame.groupby("day").groups.items():
        pos = frame.index.get_indexer(idx)
        block = cumulative[pos]
        # Baseline 0 at the 00:00 reset, so the first step of a day is its own increment.
        out[pos] = np.diff(np.concatenate([[0.0], block]))
    return out


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(
        description="Basin-average hourly ERA5-Land runoff."))
    parser.add_argument("--era5-dir", default="/ibex/user/kongw0a/era5_land_africa_hourly3")
    parser.add_argument("--pattern", default="era5land_runoff_*_??????.nc")
    parser.add_argument("--basins", default="africa/africa_basins.gpkg")
    parser.add_argument("--stations", required=True,
                        help="Comma-separated station ids to average (one tile each).")
    parser.add_argument("--out", default=None,
                        help="Output CSV; default <era5-dir>/basin_hourly_runoff.csv.gz")
    args = parser.parse_args()

    era5_dir = Path(args.era5_dir)
    out_path = Path(args.out) if args.out else era5_dir / "basin_hourly_runoff.csv.gz"
    logger = setup_logging(era5_dir / "basin_average_hourly.log")

    basins = gpd.read_file(resolve(args.basins))
    basins["station_id"] = basins["station_id"].astype(str)
    wanted = [s.strip() for s in args.stations.split(",") if s.strip()]

    files = sorted(str(p) for p in era5_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files matching {args.pattern} under {era5_dir}")
    # Group by tile id, the token between the variable name and the YYYYMM stamp.
    by_tile: dict[str, list[str]] = {}
    for path in files:
        stem = Path(path).stem                       # era5land_runoff_<tile>_<YYYYMM>
        tile = "_".join(stem.split("_")[2:-1])
        by_tile.setdefault(tile, []).append(path)
    logger.info("%d files over %d tiles: %s", len(files), len(by_tile),
                ", ".join(f"{k} ({len(v)})" for k, v in sorted(by_tile.items())))

    rows = []
    for tile, tile_files in sorted(by_tile.items()):
        ds = normalise_names(open_tile(sorted(tile_files)))
        name = next((v for v in RUNOFF_ALIASES if v in ds.variables), None)
        if name is None:
            raise SystemExit(f"{tile}: no runoff variable among {list(ds.data_vars)}")
        lons = np.asarray(ds["longitude"].values, dtype=np.float64)
        lats = np.asarray(ds["latitude"].values, dtype=np.float64)
        times = pd.DatetimeIndex(ds["time"].values)
        field = ds[name]

        # A tile holds exactly the basins whose polygon it was cut for.
        for sid in wanted:
            geom = basins.loc[basins.station_id.eq(sid)]
            if geom.empty:
                raise SystemExit(f"{sid} not in {args.basins}")
            polygon = geom.geometry.iloc[0]
            x0, y0, x1, y1 = polygon.bounds
            if not (lons.min() - 0.1 <= x0 and x1 <= lons.max() + 0.1
                    and lats.min() - 0.1 <= y0 and y1 <= lats.max() + 0.1):
                continue                              # this basin belongs to another tile
            sel = cell_weights(polygon, lons, lats)
            if sel is None:
                raise SystemExit(f"{sid}: no intersecting cell in tile {tile}")
            r, c, w = sel
            values = field.values[:, r, c]             # (time, cell), metres, cumulative
            cumulative = np.einsum("tc,c->t", np.nan_to_num(values, nan=0.0), w)
            hourly_mm = deaccumulate(cumulative, times) * 1000.0
            # Hour-ending -> hour-beginning, matching the model's slot convention.
            rows.append(pd.DataFrame({
                "station_id": sid,
                "time": times - pd.Timedelta(hours=1),
                "era5_land_hourly": hourly_mm,
                "n_cells": len(w),
            }))
            logger.info("%s in tile %s: %d cells, %d hours, mean %.4f mm/h, "
                        "daily total %.3f mm/d", sid, tile, len(w), len(times),
                        float(np.nanmean(hourly_mm)), float(np.nanmean(hourly_mm)) * 24)
        ds.close()

    if not rows:
        raise SystemExit("no basin matched any tile")
    out = pd.concat(rows, ignore_index=True).sort_values(["station_id", "time"])
    out.to_csv(out_path, index=False, compression="gzip")
    logger.info("wrote %s: %d rows, %d basins", out_path, len(out), out.station_id.nunique())

    # De-accumulation is easy to get subtly wrong, so check it closes: the 24 hourly
    # increments of a day must sum to that day's 00:00 total, which is what the daily
    # product reports. A mismatch here means the grouping or the baseline is wrong.
    daily = (out.assign(day=out.time.dt.normalize())
             .groupby(["station_id", "day"])["era5_land_hourly"]
             .agg(["sum", "count"]).reset_index())
    full = daily.loc[daily["count"].eq(24)]
    logger.info("closure check: %d complete days, hourly sums mean %.3f mm/d "
                "(min %.3f, max %.3f); negative increments: %d of %d hours",
                len(full), float(full["sum"].mean()), float(full["sum"].min()),
                float(full["sum"].max()),
                int((out.era5_land_hourly < -1e-9).sum()), len(out))


if __name__ == "__main__":
    main()
