"""Download ERA5-Land runoff over the African basins (Plan.docx Phase I Step 5).

Downloads, per tile from ``africa/africa_era5_tiles.csv``, the daily runoff
totals of ERA5-Land, then leaves ``scripts.basin_average_era5_land`` to turn the
grids into per-basin mm/d series comparable with the observed ``q_mm``.

Why only 00:00 UTC
------------------
ERA5-Land accumulated fields (``runoff``, ``surface_runoff``,
``sub_surface_runoff``) accumulate from 00 UTC and reset each day, so the value
stamped 00:00 on day D+1 IS the total for day D. Requesting that single hour
gives daily totals directly and is 24x smaller than pulling every hour. The
one-day shift is applied in the averaging step, not here -- what lands on disk
is exactly what CDS returned. Pass ``--hourly`` to fetch all 24 hours instead if
you want to verify the convention yourself.

Requirements
------------
1. A CDS account, and the ERA5-Land licence accepted once in the web UI at
   https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land -- the API
   returns 403 until you have clicked through it.
2. ``~/.cdsapirc``:
       url: https://cds.climate.copernicus.eu/api
       key: <your personal access token>

    python -m scripts.download_era5_land_africa --out-dir /ibex/user/kongw0a/era5_land_africa
    python -m scripts.download_era5_land_africa --dry-run          # print the requests only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from common.config import add_common_args, load_config, resolve
from common.utils import setup_logging

DATASET = "reanalysis-era5-land"
ALL_VARIABLES = ["runoff", "surface_runoff", "sub_surface_runoff"]
DEFAULT_OUT = "/ibex/user/kongw0a/era5_land_africa"

MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]


def build_request(tile: pd.Series, years: list[int], hourly: bool, variables: list[str]) -> dict:
    return {
        "variable": variables,
        "year": [str(y) for y in years],
        "month": MONTHS,
        "day": DAYS,
        # Accumulations reset at 00 UTC, so 00:00 of day D+1 is the day-D total.
        "time": [f"{h:02d}:00" for h in range(24)] if hourly else ["00:00"],
        # CDS order is [north, west, south, east].
        "area": [float(tile["north"]), float(tile["west"]), float(tile["south"]), float(tile["east"])],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def year_chunks(start: int, end: int, per_request: int) -> list[list[int]]:
    years = list(range(start, end + 1))
    return [years[i : i + per_request] for i in range(0, len(years), per_request)]


def check_credentials(logger) -> bool:
    rc = Path.home() / ".cdsapirc"
    if not rc.exists():
        logger.error(
            "~/.cdsapirc not found. Create it with:\n"
            "    url: https://cds.climate.copernicus.eu/api\n"
            "    key: <your CDS personal access token>\n"
            "Get the token from https://cds.climate.copernicus.eu/profile after logging in, "
            "and accept the ERA5-Land licence at "
            "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land (Download tab)."
        )
        return False
    text = rc.read_text()
    if "url:" not in text or "key:" not in text:
        logger.error("~/.cdsapirc exists but has no url:/key: lines")
        return False
    return True


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Download ERA5-Land runoff for African basins."))
    parser.add_argument("--tiles", default="africa/africa_era5_tiles.csv")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--years-per-request", type=int, default=5)
    parser.add_argument("--hourly", action="store_true", help="Fetch all 24 hours instead of daily totals (24x bigger).")
    parser.add_argument("--dry-run", action="store_true", help="Print the requests and the plan, download nothing.")
    parser.add_argument("--only-tile", default=None, help="Restrict to one tile id (useful for a first test).")
    parser.add_argument(
        "--variables", nargs="+", default=ALL_VARIABLES, choices=ALL_VARIABLES,
        help="Trim to just 'runoff' to cut the download to a third; the surface/sub-surface "
             "split is only needed if you want to separate fast and slow response.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "download.log")

    tiles = pd.read_csv(resolve(args.tiles))
    if args.only_tile:
        tiles = tiles.loc[tiles["tile"] == args.only_tile]
        if tiles.empty:
            raise SystemExit(f"tile {args.only_tile!r} not in {args.tiles}")

    chunks = year_chunks(args.start_year, args.end_year, args.years_per_request)
    jobs = [(tile, years) for _, tile in tiles.iterrows() for years in chunks]
    logger.info(
        "%d tiles x %d year-chunks = %d CDS requests | %s | %d-%d",
        len(tiles), len(chunks), len(jobs), "hourly" if args.hourly else "daily totals (00:00 UTC)",
        args.start_year, args.end_year,
    )
    total_cells = int(tiles["n_cells"].sum())
    n_days = (args.end_year - args.start_year + 1) * 365
    per_var = total_cells * n_days * 4 * (24 if args.hourly else 1) / 1024**3
    logger.info("%d grid cells | rough volume %.1f GiB for %d variable(s): %s",
                total_cells, per_var * len(args.variables), len(args.variables), ", ".join(args.variables))

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    if args.dry_run:
        example = build_request(tiles.iloc[0], chunks[0], args.hourly, args.variables)
        logger.info("example request for tile %s:\n%s", tiles.iloc[0]["tile"], json.dumps(example, indent=2))
        logger.info("targets would be written to %s", out_dir)
        for tile, years in jobs[:5]:
            logger.info("  %s", out_dir / f"era5land_{tile['tile']}_{years[0]}-{years[-1]}.nc")
        logger.info("... %d more", max(0, len(jobs) - 5))
        return

    if not check_credentials(logger):
        raise SystemExit(2)

    import cdsapi

    client = cdsapi.Client()
    done = skipped = failed = 0

    for index, (tile, years) in enumerate(jobs, start=1):
        name = f"era5land_{tile['tile']}_{years[0]}-{years[-1]}.nc"
        target = out_dir / name
        if target.exists() and target.stat().st_size > 0 and manifest.get(name, {}).get("complete"):
            skipped += 1
            continue

        request = build_request(tile, years, args.hourly, args.variables)
        logger.info("[%d/%d] %s  area=%s  years=%s-%s",
                    index, len(jobs), name, request["area"], years[0], years[-1])
        started = time.time()
        try:
            client.retrieve(DATASET, request, str(target))
        except Exception as exc:
            failed += 1
            logger.error("  FAILED: %s: %s", type(exc).__name__, str(exc)[:400])
            manifest[name] = {"complete": False, "error": str(exc)[:400]}
            manifest_path.write_text(json.dumps(manifest, indent=2))
            continue

        size_mib = target.stat().st_size / 1024**2
        logger.info("  ok: %.1f MiB in %.0f s", size_mib, time.time() - started)
        manifest[name] = {
            "complete": True,
            "tile": tile["tile"],
            "years": [years[0], years[-1]],
            "area": request["area"],
            "variables": args.variables,
            "hourly": bool(args.hourly),
            "size_bytes": target.stat().st_size,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        done += 1

    logger.info("finished: %d downloaded, %d already present, %d failed", done, skipped, failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
