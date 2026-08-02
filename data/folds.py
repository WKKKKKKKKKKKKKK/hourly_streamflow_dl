"""5-fold station split (Plan.docx Phase I Step 1/4).

Each fold designates 20% of stations as the TARGET domain (their hourly
observations are treated as unavailable) and the other 80% as the SOURCE
domain. Rotating through all five folds gives every station exactly one turn as
a target station, so the final hourly metrics cover the whole station set
rather than a lucky 20%.

The split is stratified so every fold has the same mix of data sources and
record lengths -- CAMELSH alone is 57% of the stations, and an unstratified
draw would let fold composition drive the fold-to-fold spread.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def assign_folds(
    stations: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    strata_cols: tuple[str, ...] = ("source", "size_bin"),
    n_size_bins: int = 5,
) -> pd.DataFrame:
    """Return ``stations`` plus a ``fold`` column in ``0..n_folds-1``."""
    frame = stations.copy()
    if "source" not in frame.columns:
        frame["source"] = frame["station_id"].str.split("__").str[0]

    if "size_bin" in strata_cols:
        total = frame.get("n_training_batches", pd.Series(0, index=frame.index)) + frame.get(
            "n_validation_batches", pd.Series(0, index=frame.index)
        )
        frame["size_bin"] = (
            pd.qcut(total.rank(method="first"), q=n_size_bins, labels=False, duplicates="drop")
            .fillna(0)
            .astype(int)
        )

    rng = np.random.default_rng(seed)
    frame["fold"] = -1
    # Deal each stratum round-robin from a random offset, so strata that are
    # smaller than n_folds still spread across folds instead of piling into 0.
    for _, group in frame.groupby(list(strata_cols), sort=True):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        offset = int(rng.integers(n_folds))
        frame.loc[idx, "fold"] = (np.arange(len(idx)) + offset) % n_folds

    if (frame["fold"] < 0).any():
        raise RuntimeError("some stations were not assigned a fold")
    return frame


def make_folds(
    stations: pd.DataFrame,
    out_path: str | os.PathLike,
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    frame = assign_folds(stations, n_folds=n_folds, seed=seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def load_folds(path: str | os.PathLike) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/make_folds.py first")
    return pd.read_csv(path, dtype={"station_id": str})


def domain_stations(folds: pd.DataFrame, fold: int) -> tuple[set[str], set[str]]:
    """``(source_stations, target_stations)`` for one fold."""
    if fold not in set(folds["fold"].unique()):
        raise ValueError(f"fold {fold} not present in the fold table")
    target = set(folds.loc[folds["fold"] == fold, "station_id"].astype(str))
    source = set(folds.loc[folds["fold"] != fold, "station_id"].astype(str))
    return source, target


def fold_summary(folds: pd.DataFrame) -> pd.DataFrame:
    """Station counts per fold and source, to eyeball the stratification."""
    table = folds.pivot_table(index="fold", columns="source", values="station_id", aggfunc="count").fillna(0)
    table = table.astype(int)
    table["total"] = table.sum(axis=1)
    return table
