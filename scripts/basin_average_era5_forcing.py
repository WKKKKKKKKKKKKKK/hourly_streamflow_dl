"""Turn the ERA5-Land hourly grids into per-basin forcing for the African basins.

Produces the three dynamic features the model was trained on, in the same units
as ``hourly_q_dl``:

    era5_land_africa_hourly_forcing.nc
        dims (station, time)
        pcp   [mm h-1]   from total_precipitation
        pet   [mm h-1]   from potential_evaporation
        temp  [degC]     from 2m_temperature

Two ERA5-Land conventions are handled here.

**Accumulations are cumulative from 00 UTC, not hourly.** ``total_precipitation``
and ``potential_evaporation`` at stamp H hold the total since 00:00 of that day,
so the hourly amount is a difference of consecutive stamps -- and the stamp at
00:00 belongs to the *previous* day (it is that day's 24-hour total). Treating
the raw values as hourly rates would inflate precipitation by roughly a factor
of twelve, so:

    hour == 01:00 -> increment = value(01:00)                  (accumulator reset)
    otherwise     -> increment = value(t) - value(t - 1h)

An increment stamped ``t`` covers the interval ``(t-1h, t]``.

**Sign.** ERA5 fluxes use a downward-positive convention, so evaporative fluxes
come out negative. The sign of ``potential_evaporation`` is detected from the
data and flipped if needed, so ``pet`` always ends up positive.

    python -m scripts.basin_average_era5_forcing --era5-dir /ibex/user/kongw0a/era5_land_africa_forcing
"""

from __future__ import annotations

import argparse
import glob
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from common.config import add_common_args, load_config, resolve
from common.utils import setup_logging
from scripts.basin_average_era5_land import cell_weights
from scripts.era5_tiles import assign_basins_to_tiles, group_by_tile, open_tile, union_time_axis

warnings.filterwarnings("ignore", category=RuntimeWarning)

MM_PER_M = 1000.0
KELVIN = 273.15

# ERA5-Land short name -> (output name, kind)
SOURCES = {
    "tp": ("pcp", "accumulated"),
    "total_precipitation": ("pcp", "accumulated"),
    "pev": ("pet", "accumulated"),
    "potential_evaporation": ("pet", "accumulated"),
    "t2m": ("temp", "instant"),
    "2m_temperature": ("temp", "instant"),
}


def deaccumulate(values: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """Cumulative-since-00-UTC series -> per-hour increments.

    ``values`` is (time, ...) and ``hours`` the hour-of-day of each step.
    """
    out = np.empty_like(values)
    out[0] = values[0]
    out[1:] = values[1:] - values[:-1]
    # 01:00 is the first step after the daily reset, so it is already the increment.
    reset = hours == 1
    out[reset] = values[reset]
    # A gap in the series (missing file) makes the difference meaningless.
    return out


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Basin-average ERA5-Land hourly forcing."))
    parser.add_argument("--era5-dir", default="/ibex/user/kongw0a/era5_land_africa_forcing")
    parser.add_argument("--basins", default="africa/africa_basins.gpkg")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    era5_dir = Path(args.era5_dir)
    out_path = Path(args.out) if args.out else era5_dir / "era5_land_africa_hourly_forcing.nc"
    logger = setup_logging(era5_dir / "basin_average_forcing.log")

    basins = gpd.read_file(resolve(args.basins)).to_crs("EPSG:4326")
    station_ids = basins["station_id"].astype(str).tolist()
    logger.info("basins: %d", len(station_ids))

    grouped = group_by_tile(era5_dir, "era5land_forcing_*.nc")
    logger.info("%d files across %d spatial tiles", sum(map(len, grouped.values())), len(grouped))
    basin_index = assign_basins_to_tiles(basins, grouped, logger)
    times = union_time_axis(grouped, logger)
    logger.info("time: %s .. %s (%d steps)", times[0], times[-1], len(times))

    gaps = np.unique(np.diff(times.to_numpy()).astype("timedelta64[h]").astype(int))
    if not np.array_equal(gaps, np.array([1])):
        logger.warning(
            "time axis is not strictly hourly (gaps seen: %s h) -- de-accumulation "
            "differences across a gap are invalid and those steps will be wrong.", gaps,
        )

    out: dict[str, np.ndarray] = {}
    n_cells = np.zeros(len(station_ids), dtype=np.int32)
    mapping: dict[str, tuple[str, str]] = {}
    done = 0

    for tile, files in grouped.items():
        rows_wanted = basin_index[tile]
        if rows_wanted.size == 0:
            logger.info("tile %s holds no basins -- skipped", tile)
            continue
        ds = open_tile(files)
        tile_mapping = {name: SOURCES[name] for name in ds.data_vars if name in SOURCES}
        if not tile_mapping:
            raise SystemExit(f"no recognised forcing variables in {list(ds.data_vars)}")
        mapping.update(tile_mapping)
        for _, (target, _) in tile_mapping.items():
            out.setdefault(target, np.full((len(station_ids), len(times)), np.nan, dtype=np.float32))

        lons = np.asarray(ds["longitude"].values, dtype=np.float64)
        lats = np.asarray(ds["latitude"].values, dtype=np.float64)
        tile_times = pd.to_datetime(ds["time"].values)
        hour_of_day = tile_times.hour.to_numpy()
        # Where this tile's steps sit on the shared axis, so tiles with different
        # spans still land in the right columns instead of silently shifting.
        slot = times.get_indexer(tile_times)
        if (slot < 0).any():
            raise SystemExit(f"tile {tile} has timestamps absent from the shared axis")

        for k in rows_wanted:
            station_id, geom = station_ids[k], basins.geometry.iloc[k]
            sel = cell_weights(geom, lons, lats)
            if sel is None:
                logger.warning("%s: no overlapping cell in tile %s", station_id, tile)
                continue
            rows, cols, weights = sel
            n_cells[k] = len(weights)

            picker = dict(
                latitude=xr.DataArray(rows, dims="cell"),
                longitude=xr.DataArray(cols, dims="cell"),
            )
            block = ds[list(tile_mapping)].isel(**picker).load()
            w = xr.DataArray(weights, dims="cell")

            for name, (target, kind) in tile_mapping.items():
                series = np.asarray((block[name] * w).sum("cell").values, dtype=np.float64)
                if kind == "accumulated":
                    series = deaccumulate(series, hour_of_day) * MM_PER_M
                elif target == "temp":
                    series = series - KELVIN
                out[target][k, slot] = series.astype(np.float32)

            done += 1
            if done % 20 == 0:
                logger.info("  %d/%d basins", done, len(station_ids))
        ds.close()

    if not out:
        raise SystemExit("no basin produced any output")
    logger.info("mapping: %s", {k: v[0] for k, v in mapping.items()})

    # ERA5 fluxes are downward-positive, so evaporation arrives negative.
    if "pet" in out:
        finite = np.isfinite(out["pet"])
        if finite.any() and np.nanmean(out["pet"][finite]) < 0:
            logger.info("potential_evaporation is negative on average -- flipping sign so pet > 0")
            out["pet"] = -out["pet"]

    result = xr.Dataset(
        {name: (("station", "time"), values) for name, values in out.items()},
        coords={"station": station_ids, "time": times},
    )
    result["n_cells"] = ("station", n_cells)
    units = {"pcp": "mm h-1", "pet": "mm h-1", "temp": "degC"}
    for name in out:
        result[name].attrs.update(units=units[name], source="ERA5-Land")
    result.attrs.update(
        note=(
            "Basin-averaged ERA5-Land hourly forcing. Accumulated fields de-accumulated "
            "from the 00-UTC cumulative convention; a value stamped t covers (t-1h, t]. "
            "NOTE: precipitation is ERA5-Land, whereas the model was trained on MSWEP V3.16."
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(out_path)
    logger.info("wrote %s (%.1f MiB)", out_path, out_path.stat().st_size / 1024**2)

    # Compare against the scalers the model was trained with -- a large gap here
    # is the forcing-product mismatch showing up, and it matters for Step 4.
    from data.dataset import load_scalers

    try:
        scalers = load_scalers(cfg.data.root)
        for name in ("pet", "pcp", "temp"):
            if name in out:
                logger.info(
                    "%-5s ERA5-Land mean %8.4f std %8.4f  |  training mean %8.4f std %8.4f",
                    name, float(np.nanmean(out[name])), float(np.nanstd(out[name])),
                    scalers["x_dyn_mean"][name], scalers["x_dyn_std"][name],
                )
    except Exception as exc:
        logger.info("could not load training scalers for comparison: %s", exc)

    logger.info("cells per basin: median %d min %d max %d",
                int(np.median(n_cells)), int(n_cells.min()), int(n_cells.max()))


if __name__ == "__main__":
    main()
