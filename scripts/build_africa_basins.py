"""Build the African basin table for Plan.docx Phase I Step 4/5.

Resolves, for every African basin in the daily database:

  * the station id, source, gauge lat/long and catchment area,
  * the catchment polygon (needed to basin-average ERA5-Land runoff),
  * the polygon's bounding box, snapped to the ERA5-Land 0.1 degree grid.

Polygon source: ``processed/20250630/daily/stations/<station_id>/boundary.shp`` --
the per-station catchment the gscad pipeline itself uses. All 294 African basins
have one, and their areas agree with ``static.csv`` to a few percent. (Caravan's
``grdc_basin_shapes.shp`` and GSHA's ``boundaries.shp`` are used only as
fallbacks; between them they miss the 49 ADHI basins entirely.)

Outputs (default under the repo's africa/ folder):
  africa_basins.csv         one row per basin + bbox + polygon_source
  africa_basins.gpkg        the polygons themselves, EPSG:4326
  africa_era5_tiles.csv     merged download tiles for the CDS request

    python -m scripts.build_africa_basins
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from common.config import add_common_args, load_config, resolve
from common.utils import setup_logging

warnings.filterwarnings("ignore", category=UserWarning)

GSCAD = Path("/ibex/project/c2266/abbaa0a/data/gscad_database")
DAILY_STATIC = GSCAD / "processed/20250630/daily/dataframes/static.csv"
DAILY_STATIONS = GSCAD / "processed/20250630/daily/stations"
CARAVAN_SHP = GSCAD / "raw/GRDCCaravan/GRDC-Caravan-extension-nc/shapefiles/grdc/grdc_basin_shapes.shp"
GSHA_SHP = GSCAD / "raw/GSHA/boundaries.shp"

# The continent-holdout PUB run that defines the African test set (294 basins).
AFRICA_BASELINE = Path(
    "/ibex/project/c2266/abbaa0a/results/regionalization/20250630/"
    "gscad_continent_lstm/47515076_africa/2000385_test_performance.csv"
)

# Generous Africa box; only used as a cross-check / fallback.
AFRICA_BBOX = dict(lon_min=-20.0, lon_max=55.0, lat_min=-36.0, lat_max=38.0)
ERA5_LAND_RES = 0.1


def africa_station_ids(logger) -> tuple[list[str], str]:
    """The authoritative African basin list, preferring the baseline run's columns."""
    if AFRICA_BASELINE.exists():
        header = pd.read_csv(AFRICA_BASELINE, nrows=0)
        ids = [c for c in header.columns if c not in ("metric", "mean", "median")]
        logger.info("using the %d basins scored by the continent-PUB baseline", len(ids))
        return ids, "continent_pub_baseline"
    logger.warning("baseline result not found at %s -- falling back to a lat/long box", AFRICA_BASELINE)
    return [], "bbox_fallback"


def load_daily_static(logger) -> pd.DataFrame:
    static = pd.read_csv(DAILY_STATIC, comment="#", low_memory=False)
    static = static.rename(columns={static.columns[0]: "station_id"})
    static["source"] = static["station_id"].astype(str).str.split("__").str[0]
    logger.info("daily static: %d stations x %d columns", *static.shape)
    return static


def _lazy_fallback_polygons(logger):
    """Caravan + GSHA lookups, loaded only if a per-station boundary is missing."""
    caravan = gpd.read_file(CARAVAN_SHP)
    gsha = gpd.read_file(GSHA_SHP, columns=["agency", "ID"])
    if gsha.crs is None:
        gsha = gsha.set_crs("EPSG:4326")
    gsha["bare_id"] = gsha["ID"].astype(str).str.rsplit("_", n=1).str[0]
    logger.info("fallback polygon sets loaded: Caravan %d, GSHA %d", len(caravan), len(gsha))
    return (
        caravan.set_index(caravan["gauge_id"].astype(str))["geometry"],
        gsha.drop_duplicates("bare_id").set_index("bare_id")["geometry"],
    )


def resolve_polygons(basins: pd.DataFrame, logger) -> gpd.GeoDataFrame:
    """Attach each basin's catchment polygon, preferring its own boundary.shp."""
    caravan_by_key = gsha_by_bare = None
    geometries, origins, shp_areas = [], [], []

    for station_id in basins["station_id"]:
        source, _, raw_id = str(station_id).partition("__")
        geom, origin = None, ""

        boundary = DAILY_STATIONS / str(station_id) / "boundary.shp"
        if boundary.exists():
            frame = gpd.read_file(boundary)
            if frame.crs is None:
                frame = frame.set_crs("EPSG:4326")
            frame = frame.to_crs("EPSG:4326")
            geom = frame.union_all() if hasattr(frame, "union_all") else frame.unary_union
            origin = "station_boundary"

        if geom is None:
            if caravan_by_key is None:
                caravan_by_key, gsha_by_bare = _lazy_fallback_polygons(logger)
            for candidate in (f"GRDC_{raw_id}", str(station_id), raw_id):
                if candidate in caravan_by_key.index:
                    geom, origin = caravan_by_key.loc[candidate], "caravan_grdc"
                    break
            if geom is None and raw_id in gsha_by_bare.index:
                geom, origin = gsha_by_bare.loc[raw_id], "gsha"

        geometries.append(geom)
        origins.append(origin)
        shp_areas.append(np.nan)

    out = basins.copy()
    out["polygon_source"] = origins
    result = gpd.GeoDataFrame(out, geometry=geometries, crs="EPSG:4326")

    # Cross-check the polygon area against static.csv -- a mismatch means the
    # polygon belongs to a different gauge and would corrupt the basin average.
    ok = result.geometry.notna()
    if ok.any() and "area" in result.columns:
        equal_area = result.loc[ok].to_crs("EPSG:6933")
        result.loc[ok, "polygon_area_km2"] = equal_area.area.to_numpy() / 1e6
        ratio = result.loc[ok, "polygon_area_km2"] / result.loc[ok, "area"]
        result.loc[ok, "area_ratio"] = ratio
        bad = ok & (~result["area_ratio"].between(0.5, 2.0))
        logger.info("polygon area vs static area: median ratio %.3f | %d outside 0.5-2x",
                    float(ratio.median()), int(bad.sum()))
        if bad.any():
            logger.warning("suspect polygons (area ratio far from 1): %s",
                           list(result.loc[bad, "station_id"].head(8)))
    return result


def snap_bbox(bounds: tuple[float, float, float, float], pad: float = ERA5_LAND_RES) -> tuple[float, ...]:
    """Expand a bbox to the ERA5-Land 0.1 degree grid, with one cell of padding."""
    minx, miny, maxx, maxy = bounds
    return (
        float(np.floor((minx - pad) / ERA5_LAND_RES) * ERA5_LAND_RES),
        float(np.floor((miny - pad) / ERA5_LAND_RES) * ERA5_LAND_RES),
        float(np.ceil((maxx + pad) / ERA5_LAND_RES) * ERA5_LAND_RES),
        float(np.ceil((maxy + pad) / ERA5_LAND_RES) * ERA5_LAND_RES),
    )


def merge_tiles(boxes: list[tuple[float, ...]], tile_deg: float = 10.0) -> pd.DataFrame:
    """Group basin boxes into coarse tiles, then shrink each tile to its contents.

    One CDS request per basin would mean hundreds of queued jobs; one request for
    all of Africa would be tens of terabytes. Coarse tiles trimmed to the basins
    they actually contain sit between the two.
    """
    assign: dict[tuple[int, int], list[tuple[float, ...]]] = {}
    for box in boxes:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        assign.setdefault((int(np.floor(cx / tile_deg)), int(np.floor(cy / tile_deg))), []).append(box)

    rows = []
    for (tx, ty), members in sorted(assign.items()):
        arr = np.asarray(members, dtype=float)
        west, south = arr[:, 0].min(), arr[:, 1].min()
        east, north = arr[:, 2].max(), arr[:, 3].max()
        n_lon = int(round((east - west) / ERA5_LAND_RES)) + 1
        n_lat = int(round((north - south) / ERA5_LAND_RES)) + 1
        rows.append(
            {
                "tile": f"t{tx:+03d}{ty:+03d}",
                "n_basins": len(members),
                "north": north,
                "west": west,
                "south": south,
                "east": east,
                "n_lon": n_lon,
                "n_lat": n_lat,
                "n_cells": n_lon * n_lat,
            }
        )
    return pd.DataFrame(rows).sort_values("n_cells", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Build the African basin table."))
    parser.add_argument("--out-dir", default="africa")
    parser.add_argument("--tile-deg", type=float, default=10.0)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "build_africa_basins.log")

    ids, provenance = africa_station_ids(logger)
    static = load_daily_static(logger)

    if ids:
        basins = static.loc[static["station_id"].isin(ids)].copy()
        missing = sorted(set(ids) - set(basins["station_id"]))
        if missing:
            logger.warning("%d baseline basins absent from the daily static table: %s",
                           len(missing), missing[:5])
    else:
        inside = static["long"].between(AFRICA_BBOX["lon_min"], AFRICA_BBOX["lon_max"]) & static[
            "lat"
        ].between(AFRICA_BBOX["lat_min"], AFRICA_BBOX["lat_max"])
        basins = static.loc[inside].copy()

    keep = ["station_id", "source", "lat", "long", "area"]
    keep = [c for c in keep if c in basins.columns]
    basins = basins[keep].reset_index(drop=True)
    basins["provenance"] = provenance
    logger.info("African basins: %d | by source: %s", len(basins), basins["source"].value_counts().to_dict())
    if "area" in basins.columns:
        logger.info("area km2: min %.0f  median %.0f  max %.0f",
                    basins["area"].min(), basins["area"].median(), basins["area"].max())

    geo = resolve_polygons(basins, logger)
    have = geo.geometry.notna()
    logger.info("polygons resolved for %d/%d basins: %s",
                int(have.sum()), len(geo), geo.loc[have, "polygon_source"].value_counts().to_dict())
    if (~have).any():
        logger.warning("no polygon for %d basins (%s ...) -- they will fall back to the "
                       "nearest ERA5-Land cell at the gauge",
                       int((~have).sum()), list(geo.loc[~have, "station_id"].head(5)))

    boxes = []
    for _, row in geo.iterrows():
        if row.geometry is not None and not row.geometry.is_empty:
            boxes.append(snap_bbox(row.geometry.bounds))
        else:
            boxes.append(snap_bbox((row["long"], row["lat"], row["long"], row["lat"])))
    box_arr = np.asarray(boxes, dtype=float)
    geo["west"], geo["south"], geo["east"], geo["north"] = box_arr.T

    geo.drop(columns="geometry").to_csv(out_dir / "africa_basins.csv", index=False)
    geo.loc[have].to_file(out_dir / "africa_basins.gpkg", driver="GPKG")

    tiles = merge_tiles(boxes, tile_deg=args.tile_deg)
    tiles.to_csv(out_dir / "africa_era5_tiles.csv", index=False)

    total_cells = int(tiles["n_cells"].sum())
    logger.info("download tiles: %d covering %d ERA5-Land cells", len(tiles), total_cells)
    logger.info("\n%s", tiles.to_string(index=False))

    # Daily runoff, float32, 1980-2024 inclusive.
    n_days = 45 * 365 + 11
    gib = total_cells * n_days * 4 / 1024**3
    logger.info(
        "estimated volume for ONE daily variable 1980-2024: %.1f GiB "
        "(%d cells x %d days x 4 B). Basin-averaged output is ~%.0f MiB.",
        gib, total_cells, n_days, len(geo) * n_days * 4 / 1024**2,
    )
    logger.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
