"""Write the 5-fold station split (Plan.docx Phase I).

Fold k means: those stations are the TARGET domain (hourly observations treated
as unavailable), all others are the SOURCE domain. Running all five folds gives
every station one turn as a target.

    python -m scripts.make_folds
    python -m scripts.make_folds --require-both-splits    # drop stations missing a split
"""

from __future__ import annotations

import argparse

import pandas as pd

from common.config import add_common_args, load_config, resolve
from common.utils import setup_logging
from data.folds import fold_summary, make_folds
from data.index import load_stations


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Create the 5-fold station split."))
    parser.add_argument("--out", default=None, help="Defaults to cfg.folds.file")
    parser.add_argument("--min-train-batches", type=int, default=1)
    parser.add_argument("--min-val-batches", type=int, default=1)
    parser.add_argument(
        "--require-both-splits",
        action="store_true",
        help="Keep only stations that have batches in BOTH training/ and validation/.",
    )
    parser.add_argument(
        "--sample-per-source",
        type=int,
        default=0,
        help="Keep only N stations per source -- a miniature fold table for integration tests.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    out_path = resolve(args.out or cfg.folds.file)
    logger = setup_logging(out_path.parent / "make_folds.log")

    stations = load_stations(resolve(cfg.data.index_dir))
    logger.info("index has %d stations", len(stations))

    keep = pd.Series(True, index=stations.index)
    if args.require_both_splits or args.min_train_batches or args.min_val_batches:
        keep &= stations["n_training_batches"] >= args.min_train_batches
        keep &= stations["n_validation_batches"] >= args.min_val_batches
    dropped = stations.loc[~keep]
    if len(dropped):
        logger.info(
            "dropping %d stations without enough batches (%s)",
            len(dropped), ", ".join(dropped["station_id"].head(8)),
        )
    stations = stations.loc[keep].reset_index(drop=True)

    if args.sample_per_source > 0:
        stations = pd.concat(
            [group.head(args.sample_per_source) for _, group in stations.groupby("source")]
        ).reset_index(drop=True)
        logger.info("sampled down to %d stations (%d per source) for integration testing",
                    len(stations), args.sample_per_source)

    logger.info("%d stations enter the fold split", len(stations))

    folds = make_folds(stations, out_path, n_folds=int(cfg.folds.n_folds), seed=int(cfg.folds.seed))
    logger.info("wrote %s", out_path)
    logger.info("stations per fold and source:\n%s", fold_summary(folds).to_string())

    per_fold = folds.groupby("fold").size()
    logger.info(
        "target-domain size per fold: min %d, max %d (%.1f%% of %d)",
        per_fold.min(), per_fold.max(), 100.0 * per_fold.mean() / len(folds), len(folds),
    )


if __name__ == "__main__":
    main()
