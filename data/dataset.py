"""Dataset over the pre-batched hourly_q_dl files.

One file on disk is already one batch of 512 samples, so one dataset item is
one batch and the DataLoader runs with ``batch_size=None`` (same convention as
MTSLSTM_100stations/code/loder.py).

Branch construction
-------------------
Each sample stores ONE 1000-step sequence covering the previous 8760 hours,
power-law subsampled: 1-hour spacing over the most recent 228 positions,
widening to ~29 hours at the far end (see ``data/lookback_offsets.json``). The
two MTS-LSTM branches are a split of that single sequence along time:

    D (low-frequency / long range) = the full 1000 steps
    H (high-frequency / recent)    = its last ``lookback_hourly`` steps

With ``frequency_factor=1`` the model's transfer index becomes
``len(D) - len(H)``, i.e. the daily state is handed to the hourly branch
exactly where the hourly window starts.

Because the last 228 positions are true hourly spacing, ``H_seq[:, -24:]``
really is the last 24 hours and its mean is the model's predicted daily
aggregate -- that is what Phase I Step 2 supervises.

Daily targets
-------------
``y_daily[i]`` is the mean of the observations inside the 24 hours ending at row
``i``, computed from the batch's own metadata timestamps.

Rows are NOT reliably consecutive hours -- hours whose sample was dropped during
preparation are simply absent, and at sparse stations fewer than half the hours
of a day survive. So instead of demanding a gap-free window, each row gets a
24-slot occupancy mask ``daily_mask[i, o] = "the hour t-23+o is present"``, and

    y_daily[i]  = mean of y over the occupied slots
    prediction  = mean of H_seq[i, -24:] over the SAME slots

Averaging both sides over identical slots keeps the target unbiased -- it is a
partial-day aggregate, not a full-day mean compared against a full-day
prediction. Rows with fewer than ``min_daily_hours`` occupied slots get NaN and
drop out of the loss, which also stops a 1-2 hour "aggregate" from leaking what
is effectively an hourly observation.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

DAILY_WINDOW = 24

# "CAMELSH__01200000_w30_0" -> station "CAMELSH__01200000"; corrected_* never match.
_REGULAR_PREFIX = re.compile(r"^(.+)_w\d+_\d+$")


def load_dataset_config(root: str | os.PathLike) -> dict:
    with open(Path(root) / "config.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_scalers(root: str | os.PathLike) -> dict:
    """``scalers.json`` written by hourly_q_dl, with y_mean/y_std as floats."""
    with open(Path(root) / "scalers.json", "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    scalers = dict(raw)
    scalers["y_mean"] = float(raw["y_mean"])
    scalers["y_std"] = float(raw["y_std"])
    return scalers


def load_lookback_offsets() -> dict:
    with open(Path(__file__).resolve().parent / "lookback_offsets.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_static_columns(root: str | os.PathLike, exclude: list[str] | None) -> tuple[np.ndarray, list[str]]:
    """Column indices of the static features to keep, in dataset order."""
    names = list(load_dataset_config(root)["static_features"])
    exclude = set(exclude or [])
    unknown = exclude - set(names)
    if unknown:
        raise KeyError(f"static_exclude names not present in the dataset: {sorted(unknown)}")
    keep = [i for i, name in enumerate(names) if name not in exclude]
    return np.asarray(keep, dtype=np.int64), [names[i] for i in keep]


def daily_occupancy(
    hours: np.ndarray,
    station_block: np.ndarray,
    window: int = DAILY_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Which of the ``window`` hours ending at each row are present in this batch.

    ``hours``         row timestamps in whole hours,
    ``station_block`` block id per row (rows must be sorted by (block, hour)).

    Returns ``(slot_row, slot_ok)``, both ``(n, window)``. ``slot_ok[i, o]`` says
    the hour ``hours[i] - (window - 1 - o)`` exists in the same station block,
    and ``slot_row[i, o]`` is the row holding it (meaningless where not ok).
    Slot ``window - 1`` is row ``i`` itself, so it is always occupied.
    """
    n = hours.shape[0]
    lag = np.arange(window - 1, -1, -1, dtype=np.int64)          # 23, 22, ..., 0
    slot_row = np.zeros((n, window), dtype=np.int64)
    slot_ok = np.zeros((n, window), dtype=bool)
    if n == 0:
        return slot_row, slot_ok

    boundaries = np.flatnonzero(np.diff(station_block)) + 1
    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, n]):
        block_hours = hours[start:stop]
        wanted = block_hours[:, None] - lag[None, :]
        pos = np.clip(np.searchsorted(block_hours, wanted), 0, len(block_hours) - 1)
        slot_ok[start:stop] = block_hours[pos] == wanted
        slot_row[start:stop] = start + pos
    return slot_row, slot_ok


def partial_daily_mean(
    y: np.ndarray,
    slot_row: np.ndarray,
    slot_ok: np.ndarray,
    min_hours: int = 1,
) -> np.ndarray:
    """Mean of ``y`` over the occupied slots; NaN below ``min_hours`` of them."""
    counts = slot_ok.sum(axis=1)
    gathered = np.where(slot_ok, y[slot_row], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = gathered.sum(axis=1) / counts
    return np.where(counts >= max(1, int(min_hours)), means, np.nan)


class PreparedBatchDataset(Dataset):
    """Map-style dataset where item ``i`` is the whole batch stored in file ``i``."""

    def __init__(
        self,
        root: str | os.PathLike,
        split: str,
        prefixes: list[str],
        lookback_hourly: int,
        static_keep: np.ndarray | None = None,
        allowed_stations: set[str] | None = None,
        with_daily: bool = False,
        min_daily_hours: int = 12,
    ):
        self.split_dir = Path(root) / split
        if not self.split_dir.is_dir():
            raise FileNotFoundError(self.split_dir)
        if not prefixes:
            raise ValueError(f"no batch prefixes given for {self.split_dir}")
        self.prefixes = list(prefixes)
        self.lookback_hourly = int(lookback_hourly)
        self.static_keep = None if static_keep is None else torch.as_tensor(static_keep, dtype=torch.long)
        self.allowed_stations = allowed_stations
        self.with_daily = bool(with_daily)
        self.min_daily_hours = int(min_daily_hours)

    def __len__(self) -> int:
        return len(self.prefixes)

    def __getitem__(self, idx: int) -> dict:
        prefix = self.prefixes[idx]
        stem = str(self.split_dir / prefix)

        x_dyn, x_static = torch.load(stem + "_x.pt", map_location="cpu", weights_only=True)
        y = torch.load(stem + "_y.pt", map_location="cpu", weights_only=True).float().reshape(-1)
        with open(stem + "_metadata.pkl", "rb") as handle:
            meta = pickle.load(handle)

        x_dyn = x_dyn.float()
        x_static = x_static.float()
        stations = meta["stn"].astype(str).to_numpy()
        stn_std = meta["stn_std"].to_numpy(dtype=np.float32)
        hours = meta["index"].to_numpy(dtype="datetime64[h]").astype(np.int64)

        # Rows come out in time order per station, but a corrected_* batch
        # concatenates several of them, so sort by (station, hour) explicitly --
        # the daily windows below need each station's rows contiguous and sorted.
        block_id = np.cumsum(np.concatenate([[False], stations[1:] != stations[:-1]]))
        if not (np.all(np.diff(block_id) >= 0) and np.all(np.diff(hours)[np.diff(block_id) == 0] > 0)):
            order = np.lexsort((hours, stations))
            x_dyn, x_static, y = x_dyn[order], x_static[order], y[order]
            stations, stn_std, hours = stations[order], stn_std[order], hours[order]
            block_id = np.cumsum(np.concatenate([[False], stations[1:] != stations[:-1]]))

        y_daily = daily_mask = None
        if self.with_daily:
            slot_row, slot_ok = daily_occupancy(hours, block_id)
            y_daily = torch.as_tensor(
                partial_daily_mean(y.numpy(), slot_row, slot_ok, self.min_daily_hours),
                dtype=torch.float32,
            )
            daily_mask = torch.as_tensor(slot_ok)

        if self.allowed_stations is not None:
            keep = np.fromiter(
                (station in self.allowed_stations for station in stations), dtype=bool, count=len(stations)
            )
            if not keep.all():
                keep_t = torch.as_tensor(keep)
                x_dyn, x_static, y = x_dyn[keep_t], x_static[keep_t], y[keep_t]
                stations, stn_std, hours = stations[keep], stn_std[keep], hours[keep]
                if y_daily is not None:
                    y_daily, daily_mask = y_daily[keep_t], daily_mask[keep_t]

        if self.static_keep is not None:
            x_static = x_static.index_select(1, self.static_keep)

        seq_len = x_dyn.shape[1]
        k_h = self.lookback_hourly
        if k_h <= 0 or k_h > seq_len:
            k_h = seq_len

        return {
            "x": {
                "D": x_dyn.contiguous(),
                "H": x_dyn[:, -k_h:, :].contiguous(),
                "S": x_static.contiguous(),
            },
            "y": y,
            "y_daily": y_daily,
            "daily_mask": daily_mask,
            "stations": stations.tolist(),
            "stn_std": torch.as_tensor(stn_std, dtype=torch.float32),
            "hours": torch.as_tensor(hours, dtype=torch.int64),
            "prefix": prefix,
        }


def identity_collate(item):
    """DataLoader(batch_size=None) still runs default_convert; keep items untouched."""
    return item


def make_loader(
    dataset: PreparedBatchDataset,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool = False,
    subset: np.ndarray | None = None,
) -> DataLoader:
    """One DataLoader over whole batch files.

    ``subset`` restricts the epoch to those dataset indices (used to sample
    ``batches_per_epoch`` files out of the ~1.3M available).
    """
    from torch.utils.data import SequentialSampler, SubsetRandomSampler

    if subset is not None:
        sampler = SubsetRandomSampler(np.asarray(subset).tolist())
    elif shuffle:
        sampler = SubsetRandomSampler(np.random.permutation(len(dataset)).tolist())
    else:
        sampler = SequentialSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=identity_collate,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def epoch_subset(n_total: int, n_wanted: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n_wanted`` batch indices without replacement (all of them if fewer exist)."""
    if n_wanted <= 0 or n_wanted >= n_total:
        return np.arange(n_total)
    return rng.choice(n_total, size=n_wanted, replace=False)


def describe_batch_pool(prefixes_regular: list[str], prefixes_corrected: list[str]) -> str:
    total = len(prefixes_regular) + len(prefixes_corrected)
    return (
        f"{total} batches (~{total * 512 / 1e6:.1f}M samples): "
        f"{len(prefixes_regular)} regular + {len(prefixes_corrected)} corrected"
    )


def station_of(prefix: str) -> str:
    """Station encoded in a batch prefix; '' for the station-mixing corrected_* batches."""
    match = _REGULAR_PREFIX.match(prefix)
    return match.group(1) if match else ""


def pick_per_station(
    prefixes: list[str],
    per_station: int,
    seed: int = 0,
    max_stations: int = 0,
) -> np.ndarray:
    """Dataset indices holding up to ``per_station`` batches for each station.

    Used to build validation/holdout sets whose median-across-stations metric is
    stable epoch to epoch. A plain random sample of the pool would touch only a
    few hundred of several thousand stations and make the median jump around.
    ``max_stations`` further restricts to a fixed random subsample of stations,
    which is how the per-epoch early-stopping set is kept affordable.
    """
    rng = np.random.default_rng(seed)
    by_station: dict[str, list[int]] = {}
    for index, prefix in enumerate(prefixes):
        station = station_of(prefix)
        if station:
            by_station.setdefault(station, []).append(index)

    stations = sorted(by_station)
    if 0 < max_stations < len(stations):
        chosen = rng.choice(len(stations), size=max_stations, replace=False)
        stations = [stations[i] for i in sorted(chosen)]

    picked: list[int] = []
    for station in stations:
        group = by_station[station]
        if per_station > 0 and len(group) > per_station:
            take = rng.choice(len(group), size=per_station, replace=False)
            group = [group[i] for i in sorted(take)]
        picked.extend(group)
    return np.asarray(sorted(picked), dtype=np.int64)


def cap_per_station(prefixes: list[str], max_per_station: int, seed: int = 0) -> list[str]:
    """Keep at most ``max_per_station`` batches for each station, chosen deterministically.

    Evaluating every batch of a 1,800-station domain means ~150k files (~930 GB).
    Capping keeps per-station coverage even while making a pass affordable --
    a plain random sample of the pool would starve stations with few batches.
    """
    if max_per_station <= 0:
        return prefixes
    rng = np.random.default_rng(seed)
    by_station: dict[str, list[str]] = {}
    for prefix in prefixes:
        by_station.setdefault(station_of(prefix) or prefix, []).append(prefix)

    kept: list[str] = []
    for station in sorted(by_station):
        group = sorted(by_station[station])
        if len(group) > max_per_station:
            picks = rng.choice(len(group), size=max_per_station, replace=False)
            group = [group[i] for i in sorted(picks)]
        kept.extend(group)
    return kept


def build_dataset(
    cfg,
    split: str,
    stations: set[str],
    with_daily: bool = False,
    max_batches_per_station: int = 0,
    logger=None,
) -> PreparedBatchDataset:
    """Assemble a dataset for one temporal split restricted to ``stations``."""
    from common.config import resolve
    from data.index import select_prefixes

    root = cfg.data.root
    index_dir = resolve(cfg.data.index_dir)
    regular, corrected = select_prefixes(
        index_dir, split, stations, include_corrected=bool(cfg.data.include_corrected)
    )
    if max_batches_per_station > 0:
        before = len(regular)
        regular = cap_per_station(regular, max_batches_per_station, seed=0)
        # corrected_* batches are few and mix stations, so cap them as one pool.
        corrected = corrected[: max(1, max_batches_per_station * 20)] if corrected else corrected
        if logger:
            logger.info(
                "%s: capped to <=%d batches/station (%d -> %d regular)",
                split, max_batches_per_station, before, len(regular),
            )
    if logger:
        logger.info("%s: %s", split, describe_batch_pool(regular, corrected))
    static_keep, _ = resolve_static_columns(root, cfg.data.get("static_exclude"))
    return PreparedBatchDataset(
        root=root,
        split=split,
        prefixes=regular + corrected,
        lookback_hourly=int(cfg.data.lookback_hourly),
        static_keep=static_keep,
        # Regular batches were already filtered by station, but corrected ones
        # still carry foreign rows, so the mask has to stay on.
        allowed_stations=stations,
        with_daily=with_daily,
        min_daily_hours=int(cfg.get_path("transfer.min_daily_hours", 12)),
    )
