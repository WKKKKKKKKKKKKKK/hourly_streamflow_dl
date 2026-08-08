"""One place that decides which data path a run uses.

Two representations of the same underlying hourly record are in play:

``prepared``     the 14 TB pre-batched hourly_q_dl files. Their daily branch is a
                 power-law SUBSAMPLE of the past year -- of 365 days, 8 carry 24
                 points, 176 carry one, 7 carry none. Cheap, already on disk.
``hourly_cache`` windows rebuilt from 6sources.nc, with 365 GENUINE daily means.
                 What the 100-station reference does, and the only way the daily
                 branch sees far-past precipitation at all: sampling one hour of a
                 day reports 69% of rainy days as zero.

The training scripts should not care which is in use, so both are assembled here
into the same bundle: a training set, a fixed station-balanced set for early
stopping, and a reporting set disjoint from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from common.config import resolve


@dataclass
class DataBundle:
    """Everything a training script needs, independent of where it came from."""

    train: Any
    val: Any                       # early-stopping set
    val_subset: np.ndarray | None  # indices into `val`; None means "all of it"
    report: Any                    # final metrics, disjoint from the early-stopping set
    dyn_size: int
    static_names: list[str]
    scalers: dict
    source: str
    n_val_stations: int = 0
    extra: dict = field(default_factory=dict)


def _prepared_bundle(cfg, stations, with_daily, logger):
    from data.dataset import (
        build_dataset,
        load_dataset_config,
        load_scalers,
        pick_per_station,
        resolve_static_spec,
        station_of,
    )

    scalers = load_scalers(cfg.data.root)
    ds_config = load_dataset_config(cfg.data.root)
    _, _, static_names = resolve_static_spec(
        cfg.data.root, cfg.data.get("static_exclude"), cfg.data.get("onehot_static")
    )

    train = build_dataset(cfg, "training", stations, with_daily=with_daily, logger=logger)
    val = build_dataset(cfg, "validation", stations, with_daily=False, logger=logger)
    val_subset = pick_per_station(
        val.prefixes,
        per_station=int(cfg.train.val_batches_per_station),
        seed=0,
        max_stations=int(cfg.train.val_max_stations),
    )
    report = build_dataset(
        cfg, "validation", stations, with_daily=False,
        max_batches_per_station=int(cfg.get_path("eval.max_batches_per_station", 0)),
        exclude_prefixes={val.prefixes[i] for i in val_subset},
        logger=logger,
    )
    n_val_stations = len({station_of(val.prefixes[i]) for i in val_subset} - {""})
    return DataBundle(
        train=train, val=val, val_subset=val_subset, report=report,
        dyn_size=len(ds_config["dyn_features"]), static_names=static_names,
        scalers=scalers, source="prepared", n_val_stations=n_val_stations,
    )


def _cache_bundle(cfg, stations, with_daily, logger):
    from data.dataset import load_dataset_config, load_scalers
    from data.hourly_windows import (
        HourlyWindowDataset,
        build_static_matrix,
        load_cache,
        pick_samples_per_station,
    )

    scalers = load_scalers(cfg.data.root)
    ds_config = load_dataset_config(cfg.data.root)
    cache = load_cache(cfg.data.cache_dir, stride=int(cfg.data.get("stride", 24)), logger=logger)
    static, static_names = build_static_matrix(cache, scalers, cfg, logger=logger)

    common = dict(
        cache=cache, static=static, scalers=scalers,
        lookback_hourly=int(cfg.data.lookback_hourly),
        lookback_daily=int(cfg.data.get("lookback_daily", 365)),
        chunk_size=int(cfg.data.get("chunk_size", 512)),
        min_daily_hours=int(cfg.get_path("transfer.min_daily_hours", 18)),
    )

    train = HourlyWindowDataset(stations=stations, split="training",
                                with_daily=with_daily, logger=logger, **common)

    # Station-balanced early-stopping set. One prepared batch is 512 rows, so
    # val_batches_per_station x 512 keeps the two paths' validation sets the same
    # size per station and the metrics comparable.
    per_station = int(cfg.train.val_batches_per_station) * 512
    is_val = ~cache["is_train"]
    val_mask = pick_samples_per_station(
        np.where(is_val, cache["station_idx"], -1), per_station, seed=0
    ) & is_val
    val = HourlyWindowDataset(stations=stations, split="validation",
                              sample_mask=val_mask, logger=logger, **common)

    # Report on everything the early-stopping set did NOT use, capped the same way
    # eval.max_batches_per_station caps the prepared path.
    report_cap = int(cfg.get_path("eval.max_batches_per_station", 0)) * 512
    report_mask = (~val_mask) & is_val
    if report_cap > 0:
        report_mask &= pick_samples_per_station(
            np.where(report_mask, cache["station_idx"], -1), report_cap, seed=1
        )
    report = HourlyWindowDataset(stations=stations, split="validation",
                                 sample_mask=report_mask, logger=logger, **common)

    n_val_stations = int(np.unique(val.station_idx).size)
    return DataBundle(
        train=train, val=val, val_subset=None, report=report,
        dyn_size=len(ds_config["dyn_features"]), static_names=static_names,
        scalers=scalers, source="hourly_cache", n_val_stations=n_val_stations,
        extra={"cache": cache},
    )


def build_bundle(cfg, stations: set[str], with_daily: bool = False, logger=None) -> DataBundle:
    source = str(cfg.data.get("source", "prepared"))
    if logger:
        logger.info("data source: %s", source)
    if source == "prepared":
        return _prepared_bundle(cfg, stations, with_daily, logger)
    if source == "hourly_cache":
        return _cache_bundle(cfg, stations, with_daily, logger)
    raise ValueError(f"data.source must be prepared|hourly_cache, got {source!r}")


def build_eval_set(cfg, stations: set[str], split: str, with_daily: bool = False, logger=None):
    """A single capped evaluation set, for the transfer step's M0/M1/degradation passes."""
    source = str(cfg.data.get("source", "prepared"))
    if source == "prepared":
        from data.dataset import build_dataset

        return build_dataset(
            cfg, split, stations, with_daily=with_daily,
            max_batches_per_station=int(cfg.get_path("eval.max_batches_per_station", 0)),
            logger=logger,
        )

    from data.dataset import load_scalers
    from data.hourly_windows import (
        HourlyWindowDataset,
        build_static_matrix,
        load_cache,
        pick_samples_per_station,
    )

    scalers = load_scalers(cfg.data.root)
    # data.eval_sample_index scores a different sample set than training used. The
    # stride-24 training index is 23:00-only, so leaving it in place reports an
    # "hourly" KGE over one hour of the day; run A's prepared batches cover all 24.
    cache = load_cache(
        cfg.data.cache_dir, stride=int(cfg.data.get("stride", 24)), logger=logger,
        index_name=cfg.data.get("eval_sample_index"),
    )
    static, _ = build_static_matrix(cache, scalers, cfg, logger=logger)
    want_train = split == "training"
    mask = cache["is_train"] == want_train
    cap = int(cfg.get_path("eval.max_batches_per_station", 0)) * 512
    if cap > 0:
        mask &= pick_samples_per_station(np.where(mask, cache["station_idx"], -1), cap, seed=1)
    return HourlyWindowDataset(
        cache=cache, stations=stations, split=split, static=static, scalers=scalers,
        lookback_hourly=int(cfg.data.lookback_hourly),
        lookback_daily=int(cfg.data.get("lookback_daily", 365)),
        chunk_size=int(cfg.data.get("chunk_size", 512)),
        with_daily=with_daily,
        min_daily_hours=int(cfg.get_path("transfer.min_daily_hours", 18)),
        sample_mask=mask, logger=logger,
    )
