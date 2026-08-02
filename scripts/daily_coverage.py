"""How complete are the 24-hour windows? -- picks `transfer.min_daily_hours`.

Samples random training batches and reports, per source, how many of the 24
hours ending at each row are actually present. That number decides how much
target-domain supervision Step 2 gets:

  * high threshold  -> few rows survive at sparse stations, but each "daily
    aggregate" really is a day;
  * low threshold   -> lots of rows, but a 2-hour "aggregate" is barely
    distinguishable from an hourly observation, which is exactly what Phase I
    claims the target domain does not have.

    python -m scripts.daily_coverage --batches 400
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import add_common_args, load_config, resolve
from common.utils import setup_logging
from data.dataset import DAILY_WINDOW, daily_occupancy
from data.index import load_index

THRESHOLDS = (1, 4, 6, 8, 12, 16, 18, 20, 24)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Daily-window occupancy statistics."))
    parser.add_argument("--batches", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="index/daily_coverage.csv")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    out_path = resolve(args.out)
    logger = setup_logging(out_path.parent / "daily_coverage.log")

    split_dir = Path(cfg.data.root) / "training"
    frame = load_index(resolve(cfg.data.index_dir), "training")
    frame = frame.loc[frame["kind"] == "regular"]

    rng = np.random.default_rng(args.seed)
    picks = frame.iloc[rng.choice(len(frame), size=min(args.batches, len(frame)), replace=False)]
    logger.info("sampling %d of %d training batches", len(picks), len(frame))

    records = []
    for i, (prefix, station) in enumerate(zip(picks["prefix"], picks["station"]), start=1):
        with open(split_dir / f"{prefix}_metadata.pkl", "rb") as handle:
            meta = pickle.load(handle)
        hours = meta["index"].to_numpy(dtype="datetime64[h]").astype(np.int64)
        order = np.argsort(hours, kind="stable")
        hours = hours[order]
        _, slot_ok = daily_occupancy(hours, np.zeros(len(hours), dtype=np.int64))
        counts = slot_ok.sum(axis=1)
        records.append(
            {
                "station_id": station,
                "source": str(station).split("__")[0],
                "rows": len(counts),
                "median_hours_per_day": float(np.median(counts)),
                "mean_hours_per_day": float(counts.mean()),
                **{f"frac_ge_{k}": float((counts >= k).mean()) for k in THRESHOLDS},
            }
        )
        if i % 100 == 0:
            logger.info("  %d/%d", i, len(picks))

    table = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    logger.info("median occupied hours per 24 h window, by source:\n%s",
                table.groupby("source")["median_hours_per_day"].describe()[["count", "50%", "min", "max"]].to_string())

    overall = pd.DataFrame(
        {
            "threshold": THRESHOLDS,
            "frac_rows_kept": [float(table[f"frac_ge_{k}"].mean()) for k in THRESHOLDS],
            "frac_stations_with_any": [float((table[f"frac_ge_{k}"] > 0).mean()) for k in THRESHOLDS],
            "frac_stations_over_half": [float((table[f"frac_ge_{k}"] > 0.5).mean()) for k in THRESHOLDS],
        }
    )
    logger.info("\n%s", overall.to_string(index=False))
    logger.info(
        "current config transfer.min_daily_hours = %s",
        cfg.get_path("transfer.min_daily_hours", 12),
    )
    overall.to_csv(out_path.with_name("daily_coverage_thresholds.csv"), index=False)


if __name__ == "__main__":
    main()
