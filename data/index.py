"""Batch-file index for the pre-batched hourly_q_dl dataset.

Listing the split directories is expensive (4.9M files under training/, 2.2M
under validation/), so we do it once and cache a compact index.

Two kinds of batch live side by side:

* ``{source}__{id}_w{worker}_{batch}_{x.pt,y.pt,metadata.pkl}`` -- 512 rows from
  one station, so the station is readable straight off the filename.
* ``corrected_{n}_{x.pt,y.pt,metadata.pkl}`` -- leftover per-station tail
  fragments regrouped into full batches; each mixes 2-3 stations and carries no
  station in its name. We read their metadata once (12.6k files) and store the
  station set, then mask rows at load time.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path

import pandas as pd

SPLITS = ("training", "validation")
BATCH_ROWS = 512

# CAMELSH__01200000_w30_0  ->  station CAMELSH__01200000, worker 30, batch 0
_REGULAR = re.compile(r"^(?P<station>.+)_w(?P<worker>\d+)_(?P<batch>\d+)$")
_CORRECTED = re.compile(r"^corrected_(?P<n>\d+)$")


def _scan_prefixes(split_dir: Path) -> list[str]:
    """All batch prefixes in a split directory (the part before ``_x.pt``)."""
    prefixes = []
    with os.scandir(split_dir) as entries:
        for entry in entries:
            name = entry.name
            if name.endswith("_x.pt"):
                prefixes.append(name[: -len("_x.pt")])
    prefixes.sort()
    return prefixes


def build_index(root: str | os.PathLike, out_dir: str | os.PathLike, logger=None) -> dict[str, pd.DataFrame]:
    """Scan ``root/{training,validation}`` and write the index files."""
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {}

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"missing split directory: {split_dir}")
        if logger:
            logger.info("scanning %s ...", split_dir)
        prefixes = _scan_prefixes(split_dir)
        if logger:
            logger.info("  %s: %d batches", split, len(prefixes))

        rows, corrected_prefixes = [], []
        for prefix in prefixes:
            match = _REGULAR.match(prefix)
            if match:
                rows.append(
                    {
                        "prefix": prefix,
                        "station": match.group("station"),
                        "kind": "regular",
                        "worker": int(match.group("worker")),
                        "batch": int(match.group("batch")),
                    }
                )
            elif _CORRECTED.match(prefix):
                corrected_prefixes.append(prefix)
                rows.append({"prefix": prefix, "station": "", "kind": "corrected", "worker": -1, "batch": -1})
            else:
                raise ValueError(f"unrecognised batch filename: {split}/{prefix}_x.pt")

        # corrected_* carry no station in the filename -- read their metadata once.
        corrected_map: dict[str, list[str]] = {}
        for i, prefix in enumerate(corrected_prefixes):
            with open(split_dir / f"{prefix}_metadata.pkl", "rb") as handle:
                meta = pickle.load(handle)
            corrected_map[prefix] = sorted(set(meta["stn"].astype(str)))
            if logger and (i + 1) % 2000 == 0:
                logger.info("  corrected metadata %d/%d", i + 1, len(corrected_prefixes))

        frame = pd.DataFrame(rows)
        frame.to_csv(out_dir / f"batches_{split}.csv.gz", index=False, compression="gzip")
        with open(out_dir / f"corrected_{split}.json", "w", encoding="utf-8") as handle:
            json.dump(corrected_map, handle)
        frames[split] = frame
        if logger:
            logger.info(
                "  %s: %d regular, %d corrected (%d station memberships)",
                split,
                int((frame["kind"] == "regular").sum()),
                len(corrected_map),
                sum(len(v) for v in corrected_map.values()),
            )

    stations = station_table(frames)
    stations.to_csv(out_dir / "stations.csv", index=False)
    meta = {
        "root": str(root),
        "batch_rows": BATCH_ROWS,
        "splits": {split: int(len(frame)) for split, frame in frames.items()},
        "n_stations": int(len(stations)),
    }
    with open(out_dir / "index_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    if logger:
        logger.info("index written to %s", out_dir)
    return frames


def station_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per station with its regular-batch counts per split."""
    counts = {}
    for split, frame in frames.items():
        regular = frame.loc[frame["kind"] == "regular", "station"]
        counts[f"n_{split}_batches"] = regular.value_counts()
    table = pd.DataFrame(counts).fillna(0).astype(int)
    table.index.name = "station_id"
    table = table.reset_index()
    table["source"] = table["station_id"].str.split("__").str[0]
    return table.sort_values("station_id").reset_index(drop=True)


def load_index(index_dir: str | os.PathLike, split: str) -> pd.DataFrame:
    path = Path(index_dir) / f"batches_{split}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/build_index.py first")
    return pd.read_csv(path, dtype={"station": str}).fillna({"station": ""})


def load_corrected_map(index_dir: str | os.PathLike, split: str) -> dict[str, list[str]]:
    path = Path(index_dir) / f"corrected_{split}.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_stations(index_dir: str | os.PathLike) -> pd.DataFrame:
    path = Path(index_dir) / "stations.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/build_index.py first")
    return pd.read_csv(path, dtype={"station_id": str})


def select_prefixes(
    index_dir: str | os.PathLike,
    split: str,
    stations: set[str],
    include_corrected: bool = True,
) -> tuple[list[str], list[str]]:
    """Batch prefixes touching ``stations``.

    Returns ``(regular_prefixes, corrected_prefixes)``. Corrected batches are
    returned when *any* of their rows belong to ``stations``; the dataset masks
    the rest away.
    """
    frame = load_index(index_dir, split)
    regular = frame.loc[
        (frame["kind"] == "regular") & frame["station"].isin(stations), "prefix"
    ].tolist()

    corrected: list[str] = []
    if include_corrected:
        for prefix, members in load_corrected_map(index_dir, split).items():
            if any(member in stations for member in members):
                corrected.append(prefix)
    return regular, sorted(corrected)
