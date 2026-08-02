"""Recover the 42 static attributes per station, in physical units.

The prepared batches store static features already standardized inside every
``*_x.pt``, and identically across the 512 rows of a batch. This script reads
ONE batch per station, takes row 0, and inverts the ``scalers.json``
standardization, giving a plain table with lat/long/area/KGZ/slope/... .

That table is what the later Phase I analysis needs -- global maps, climate-zone
stratification, the Africa subset in Steps 4-5, and stratified fold splits.

It reads ~6 MB per station (~55 GB for 9,006 stations), so run it once on a
compute node:

    python -m scripts.build_station_table --workers 8
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.config import add_common_args, load_config, resolve
from common.utils import setup_logging
from data.dataset import load_dataset_config, load_scalers
from data.index import load_index


def _read_static(job: tuple[str, str]) -> tuple[str, np.ndarray] | None:
    station, path = job
    try:
        _, x_static = torch.load(path, map_location="cpu", weights_only=True)
        return station, x_static[0].numpy().astype(np.float64)
    except Exception:
        return None


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Extract per-station static attributes."))
    parser.add_argument("--out", default="index/station_static.csv")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Only the first N stations (for testing).")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    out_path = resolve(args.out)
    logger = setup_logging(out_path.parent / "build_station_table.log")

    root = Path(cfg.data.root)
    scalers = load_scalers(root)
    names = list(load_dataset_config(root)["static_features"])
    mean = np.array([scalers["x_st_mean"][name] for name in names], dtype=np.float64)
    std = np.array([scalers["x_st_std"][name] for name in names], dtype=np.float64)

    frame = load_index(resolve(cfg.data.index_dir), "training")
    regular = frame.loc[frame["kind"] == "regular"]
    first_batch = regular.groupby("station", sort=True)["prefix"].first()
    if args.limit:
        first_batch = first_batch.iloc[: args.limit]
    logger.info("reading one batch for each of %d stations", len(first_batch))

    jobs = [(station, str(root / "training" / f"{prefix}_x.pt")) for station, prefix in first_batch.items()]
    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(_read_static, jobs, chunksize=8), start=1):
            if result is None:
                failures.append(i)
                continue
            station, values = result
            rows.append((station, values))
            if i % 500 == 0:
                logger.info("  %d/%d", i, len(jobs))

    if failures:
        logger.warning("%d batches could not be read", len(failures))

    stations = [station for station, _ in rows]
    matrix = np.vstack([values for _, values in rows]) * std + mean
    table = pd.DataFrame(matrix, columns=names, index=pd.Index(stations, name="station_id")).reset_index()
    table.insert(1, "source", table["station_id"].str.split("__").str[0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    logger.info("wrote %s (%d stations x %d attributes)", out_path, len(table), len(names))

    for column in ("lat", "long", "area"):
        if column in table.columns:
            logger.info("  %s: min %.3f  median %.3f  max %.3f",
                        column, table[column].min(), table[column].median(), table[column].max())
    with open(out_path.with_suffix(".meta.json"), "w", encoding="utf-8") as handle:
        json.dump({"static_features": names, "n_stations": len(table)}, handle, indent=2)


if __name__ == "__main__":
    main()
