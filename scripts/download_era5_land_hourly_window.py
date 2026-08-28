"""Fetch hourly ERA5-Land total runoff for a few basins over a few months only.

``scripts.download_era5_land_africa`` splits requests by year, and a year of hourly
ERA5-Land is refused by CDS outright ("cost limits exceeded ... request is too large"):
2 years x 12 months x 31 days x 24 hours is far past the per-request field cap. Asking
for whole years and keeping the four months that get plotted would also fetch three
times the data for nothing.

So this asks for exactly the (basin, year, month) combinations the hourly figure draws.
One request per basin-month keeps every request inside the cap.

ERA5-Land runoff is an ACCUMULATED field: it accumulates from 00 UTC and resets each
day, and each step is stamped at the END of its accumulation period. The hourly
increment is therefore a within-day difference, which ``basin_average_era5_land_hourly``
takes -- not this script's job, but the reason all 24 steps are needed rather than one.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from common.config import add_common_args
from common.utils import setup_logging

DATASET = "reanalysis-era5-land"
HOURS = [f"{h:02d}:00" for h in range(24)]
DAYS = [f"{d:02d}" for d in range(1, 32)]


def build_request(tile: pd.Series, year: int, month: int, variable: str) -> dict:
    return {
        "variable": [variable],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": DAYS,
        "time": HOURS,
        "area": [float(tile["north"]), float(tile["west"]),
                 float(tile["south"]), float(tile["east"])],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(
        description="Hourly ERA5-Land runoff for specific basin-months."))
    parser.add_argument("--tiles", default="africa/africa_era5_tiles_hourly3.csv")
    parser.add_argument("--windows", required=True,
                        help="JSON: {tile_id: [\"YYYY-MM\", ...], ...}")
    parser.add_argument("--variable", default="runoff")
    parser.add_argument("--out-dir", default="/ibex/user/kongw0a/era5_land_africa_hourly3")
    parser.add_argument("--workers", type=int, default=3,
                        help="Concurrent CDS requests. Queue time dominates; CDS caps "
                             "concurrency per user and rejected requests are retried next run.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "download_window.log")

    tiles = pd.read_csv(args.tiles).set_index("tile")
    windows = json.loads(Path(args.windows).read_text()) if Path(args.windows).exists() \
        else json.loads(args.windows)

    jobs = []
    for tile_id, months in windows.items():
        if tile_id not in tiles.index:
            raise SystemExit(f"tile {tile_id!r} not in {args.tiles}")
        for stamp in months:
            year, month = (int(x) for x in stamp.split("-"))
            target = out_dir / f"era5land_{args.variable}_{tile_id}_{year}{month:02d}.nc"
            jobs.append((tile_id, year, month, target))

    logger.info("%d basin-month requests | variable %s | hourly, all 24 steps",
                len(jobs), args.variable)
    if args.dry_run:
        tile_id, year, month, target = jobs[0]
        logger.info("example request:\n%s",
                    json.dumps(build_request(tiles.loc[tile_id], year, month, args.variable),
                               indent=2))
        for _, _, _, target in jobs:
            logger.info("  would write %s", target)
        return

    import cdsapi
    client = cdsapi.Client(quiet=True, progress=False)

    def fetch(job) -> tuple[Path, str]:
        tile_id, year, month, target = job
        if target.exists() and target.stat().st_size > 0:
            return target, "present"
        request = build_request(tiles.loc[tile_id], year, month, args.variable)
        client.retrieve(DATASET, request, str(target))
        return target, "downloaded"

    counts = {"downloaded": 0, "present": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, job): job for job in jobs}
        for future in as_completed(futures):
            tile_id, year, month, target = futures[future]
            try:
                _, state = future.result()
                counts[state] += 1
                logger.info("%s %s (%.1f MiB)", state, target.name,
                            target.stat().st_size / 2**20)
            except Exception as exc:  # a rejected request is retried on the next run
                counts["failed"] += 1
                logger.error("FAILED %s: %s: %s", target.name, type(exc).__name__, exc)

    logger.info("finished: %d downloaded, %d already present, %d failed",
                counts["downloaded"], counts["present"], counts["failed"])
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
