"""Opening the downloaded ERA5-Land files, one spatial tile at a time.

The download is cut into spatial tiles crossed with year-chunks and month-groups.
Within a tile the files differ only in time, so they concatenate cleanly. Across
tiles they do NOT form a rectangular hypercube -- for the African request the three
tiles are

    t+00+00   lat  36.70 ..   3.90   lon  -0.10 .. 39.20
    t+00-01   lat  -6.60 .. -34.60   lon  18.40 .. 37.80
    t-01+00   lat  35.70 ..   6.00   lon -14.40 ..  0.60

which overlap in latitude while spanning different longitudes. Handing all of them
to ``xr.open_mfdataset(combine="by_coords")`` therefore fails outright with
"Resulting object does not have monotonic global indexes along dimension
longitude" -- it is being asked to assemble a shape that does not exist.

Per-tile processing is not a workaround here, it is exact: every one of the 294
African basin polygons lies wholly inside a single tile (verified, not assumed --
``assign_basins_to_tiles`` re-checks it and refuses to guess), so no basin is
split across tiles, nothing is double-counted, and no basin is partly missing.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

TILE_RE = re.compile(r"_(t[-+]\d+[-+]\d+)_")


def tile_of(path: str | Path) -> str:
    """Tile token embedded in a downloaded filename, e.g. 't+00-01'."""
    match = TILE_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"no tile token in {Path(path).name}")
    return match.group(1)


def group_by_tile(era5_dir: str | Path, pattern: str) -> dict[str, list[str]]:
    files = sorted(glob.glob(str(Path(era5_dir) / pattern)))
    if not files:
        raise SystemExit(f"no {pattern} under {era5_dir}")
    grouped: dict[str, list[str]] = {}
    for path in files:
        grouped.setdefault(tile_of(path), []).append(path)
    return dict(sorted(grouped.items()))


def open_tile(files: list[str], time_chunk: int = 744) -> xr.Dataset:
    """One tile's files concatenated along time, axes renamed to latitude/longitude/time."""
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"valid_time": time_chunk},
                           parallel=False)
    for axis, options in (("longitude", ["lon", "x"]), ("latitude", ["lat", "y"])):
        for name in options:
            if name in ds.coords and axis not in ds.coords:
                ds = ds.rename({name: axis})
                break
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds


def tile_bounds(files: list[str]) -> tuple[float, float, float, float]:
    """(south, north, west, east) of a tile, read from its first file."""
    with xr.open_dataset(files[0]) as ds:
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        lat = np.asarray(ds[lat_name].values, dtype=np.float64)
        lon = np.asarray(ds[lon_name].values, dtype=np.float64)
    return float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())


def assign_basins_to_tiles(basins, grouped: dict[str, list[str]], logger=None) -> dict[str, np.ndarray]:
    """Tile -> positional indices of the basins wholly inside it.

    Refuses to proceed if any basin is not wholly inside exactly one tile: a basin
    straddling a boundary would be averaged over only part of its area, which is a
    silent bias rather than an error, so it has to stop here.
    """
    bounds = {tile: tile_bounds(files) for tile, files in grouped.items()}
    box = basins.geometry.bounds
    assigned = np.full(len(basins), "", dtype=object)
    for tile, (south, north, west, east) in bounds.items():
        inside = (
            (box["miny"] >= south) & (box["maxy"] <= north)
            & (box["minx"] >= west) & (box["maxx"] <= east)
        ).to_numpy()
        take = inside & (assigned == "")
        assigned[take] = tile

    if logger:
        for tile, (south, north, west, east) in bounds.items():
            logger.info("tile %s: lat %7.2f..%7.2f lon %7.2f..%7.2f | %d files | %d basins",
                        tile, north, south, west, east, len(grouped[tile]),
                        int((assigned == tile).sum()))

    orphans = np.flatnonzero(assigned == "")
    if orphans.size:
        ids = basins.iloc[orphans]["station_id"].astype(str).tolist()
        raise SystemExit(
            f"{orphans.size} basins are not wholly inside any single tile (e.g. {ids[:5]}). "
            "Averaging them over one tile would silently cover only part of the basin. "
            "Extend the download area or merge the overlapping tiles first."
        )
    return {tile: np.flatnonzero(assigned == tile) for tile in grouped}


def union_time_axis(grouped: dict[str, list[str]], logger=None) -> pd.DatetimeIndex:
    """The time axis shared by the tiles, verified rather than assumed equal."""
    axes = {}
    for tile, files in grouped.items():
        with open_tile(files) as ds:
            axes[tile] = pd.to_datetime(ds["time"].values)
    reference = next(iter(axes.values()))
    mismatched = [t for t, a in axes.items() if not a.equals(reference)]
    if mismatched and logger:
        logger.warning("tiles %s have a different time axis from the others; taking the union "
                       "and leaving missing steps as NaN", mismatched)
    if not mismatched:
        return reference
    combined = reference
    for axis in axes.values():
        combined = combined.union(axis)
    return combined
