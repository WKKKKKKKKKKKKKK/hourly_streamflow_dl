"""Turn the ERA5-Land grids into per-basin daily runoff in mm/d.

For each of the 294 African basins, area-weights the ERA5-Land cells that
intersect its catchment polygon and writes one tidy netCDF:

    era5_land_africa_daily_runoff.nc
        dims  (station, date)
        vars  runoff, surface_runoff, sub_surface_runoff   [mm d-1]
              n_cells, weight_sum                          per station

Two conventions applied here rather than at download time:

* **Day shift.** ERA5-Land accumulations reset at 00 UTC, so the field stamped
  00:00 on day D+1 holds the total for day D. The date axis is shifted back one
  day accordingly, which is also why the download covers one year past the end
  of the period you care about.
* **Units.** ERA5-Land accumulations are metres of water equivalent; x1000 gives
  mm/d, the same units as the observed ``q_mm``.

Cells are weighted by their fractional overlap with the polygon, so a basin
smaller than one 0.1 degree cell still gets a sensible value (it falls back to
the single covering cell). Basins whose catchment is small relative to the grid
are flagged in ``n_cells`` -- treat single-cell basins with care, ERA5-Land
cannot resolve them.

    python -m scripts.basin_average_era5_land --era5-dir /ibex/user/kongw0a/era5_land_africa
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
from shapely.geometry import box

from common.config import add_common_args, load_config, resolve
from common.utils import setup_logging
from scripts.era5_tiles import assign_basins_to_tiles, group_by_tile, open_tile, union_time_axis

warnings.filterwarnings("ignore", category=RuntimeWarning)

MM_PER_M = 1000.0
VARIABLES = ["runoff", "surface_runoff", "sub_surface_runoff"]
# ERA5-Land short names as they may appear in the returned netCDF.
ALIASES = {
    "runoff": ["ro", "runoff"],
    "surface_runoff": ["sro", "surface_runoff"],
    "sub_surface_runoff": ["ssro", "sub_surface_runoff"],
}


def normalise_names(ds: xr.Dataset) -> xr.Dataset:
    """Canonical variable and axis names, whatever the CDS request produced."""
    rename = {}
    for canonical, options in ALIASES.items():
        for name in options:
            if name in ds.variables and name != canonical:
                rename[name] = canonical
    return ds.rename(rename)


def cell_weights(polygon, lons: np.ndarray, lats: np.ndarray, res: float = 0.1):
    """Fractional overlap of each intersecting grid cell with the polygon."""
    minx, miny, maxx, maxy = polygon.bounds
    lon_sel = np.flatnonzero((lons >= minx - res) & (lons <= maxx + res))
    lat_sel = np.flatnonzero((lats >= miny - res) & (lats <= maxy + res))
    if lon_sel.size == 0 or lat_sel.size == 0:
        return None

    half = res / 2
    rows, cols, weights = [], [], []
    for j in lat_sel:
        for i in lon_sel:
            cell = box(lons[i] - half, lats[j] - half, lons[i] + half, lats[j] + half)
            inter = polygon.intersection(cell)
            if not inter.is_empty and inter.area > 0:
                rows.append(j)
                cols.append(i)
                weights.append(inter.area)

    if not weights:
        # Basin smaller than a cell and fully inside it: use the covering cell.
        cx, cy = polygon.centroid.x, polygon.centroid.y
        return (
            np.array([int(np.abs(lats - cy).argmin())]),
            np.array([int(np.abs(lons - cx).argmin())]),
            np.array([1.0]),
        )
    weights = np.asarray(weights, dtype=np.float64)
    return np.asarray(rows), np.asarray(cols), weights / weights.sum()


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Basin-average ERA5-Land runoff."))
    parser.add_argument("--era5-dir", default="/ibex/user/kongw0a/era5_land_africa")
    parser.add_argument("--basins", default="africa/africa_basins.gpkg")
    parser.add_argument("--out", default=None, help="Defaults to <era5-dir>/era5_land_africa_daily_runoff.nc")
    parser.add_argument("--day-shift", type=int, default=-1,
                        help="Days to shift the time axis; -1 maps the 00:00 stamp back onto the day it accumulates.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    era5_dir = Path(args.era5_dir)
    out_path = Path(args.out) if args.out else era5_dir / "era5_land_africa_daily_runoff.nc"
    logger = setup_logging(era5_dir / "basin_average.log")

    basins = gpd.read_file(resolve(args.basins)).to_crs("EPSG:4326")
    logger.info("basins: %d", len(basins))

    grouped = group_by_tile(era5_dir, "era5land_*.nc")
    logger.info("%d files across %d spatial tiles", sum(map(len, grouped.values())), len(grouped))
    basin_index = assign_basins_to_tiles(basins, grouped, logger)
    raw_times = union_time_axis(grouped, logger)
    times = raw_times + pd.Timedelta(days=args.day_shift)
    dates = times.normalize()
    logger.info("time axis after %+d day shift: %s .. %s (%d steps)",
                args.day_shift, dates[0].date(), dates[-1].date(), len(dates))

    station_ids = basins["station_id"].astype(str).tolist()
    out: dict[str, np.ndarray] = {}
    n_cells = np.zeros(len(station_ids), dtype=np.int32)
    present: list[str] = []
    done = 0

    for tile, files in grouped.items():
        rows_wanted = basin_index[tile]
        if rows_wanted.size == 0:
            logger.info("tile %s holds no basins -- skipped", tile)
            continue
        ds = normalise_names(open_tile(files, time_chunk=512))
        tile_present = [v for v in VARIABLES if v in ds]
        if not tile_present:
            raise SystemExit(f"none of {VARIABLES} in tile {tile}; present: {list(ds.data_vars)}")
        for v in tile_present:
            if v not in present:
                present.append(v)
            out.setdefault(v, np.full((len(station_ids), len(dates)), np.nan, dtype=np.float32))

        lons = np.asarray(ds["longitude"].values, dtype=np.float64)
        lats = np.asarray(ds["latitude"].values, dtype=np.float64)
        tile_dates = (pd.to_datetime(ds["time"].values)
                      + pd.Timedelta(days=args.day_shift)).normalize()
        slot = dates.get_indexer(tile_dates)
        if (slot < 0).any():
            raise SystemExit(f"tile {tile} has dates absent from the shared axis")

        for k in rows_wanted:
            station_id, geom = station_ids[k], basins.geometry.iloc[k]
            sel = cell_weights(geom, lons, lats)
            if sel is None:
                logger.warning("%s: no overlapping ERA5-Land cell in tile %s", station_id, tile)
                continue
            rows, cols, weights = sel
            n_cells[k] = len(weights)

            picker = dict(
                latitude=xr.DataArray(rows, dims="cell"),
                longitude=xr.DataArray(cols, dims="cell"),
            )
            block = ds[tile_present].isel(**picker).load()
            w = xr.DataArray(weights, dims="cell")
            for v in tile_present:
                series = (block[v] * w).sum("cell") * MM_PER_M
                out[v][k, slot] = np.asarray(series.values, dtype=np.float32)

            done += 1
            if done % 25 == 0:
                logger.info("  %d/%d basins", done, len(station_ids))
        ds.close()

    if not present:
        raise SystemExit("no basin produced any output")

    result = xr.Dataset(
        {v: (("station", "date"), out[v]) for v in present},
        coords={"station": station_ids, "date": dates},
    )
    result["n_cells"] = ("station", n_cells)
    for v in present:
        result[v].attrs.update(units="mm d-1", source="ERA5-Land", long_name=v)
    result.attrs.update(
        note=(
            "ERA5-Land accumulated runoff, area-weighted over each catchment polygon. "
            f"time axis shifted {args.day_shift} day(s) so a value dated D is the total for day D."
        ),
        n_basins=len(station_ids),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(out_path)
    logger.info("wrote %s (%.1f MiB)", out_path, out_path.stat().st_size / 1024**2)

    single = int((n_cells <= 1).sum())
    logger.info("cells per basin: median %d, min %d, max %d | %d basins covered by <=1 cell",
                int(np.median(n_cells)), int(n_cells.min()), int(n_cells.max()), single)
    for v in present:
        finite = np.isfinite(out[v])
        logger.info("%s: mean %.3f mm/d over %d finite values", v, float(np.nanmean(out[v])), int(finite.sum()))


if __name__ == "__main__":
    main()
