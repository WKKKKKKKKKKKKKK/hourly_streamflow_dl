"""True two-branch windows, matching the 100-station reference exactly.

The reference builds each sample as

    x_h = x[t-168 : t]                                    # 168 raw hourly steps
    x_d = x[t-8760 : t].reshape(365, 24, -1).mean(axis=1)  # 365 GENUINE daily means
    y   = y[t]

with ``frequency_factor = 24``, so the model hands the daily state to the hourly
branch at ``transfer_index = 365 - 168//24 = 358`` -- exactly 7 days, i.e. 168
hours, before the target. That alignment is the point of the architecture, and it
is what a positional split of the prepared subsampled sequence cannot give.

The daily branch reads pre-averaged RAW daily means and standardizes afterwards.
That is not an approximation of the reference's standardize-then-average:
standardization is affine and the mean is linear, so the two are identical.

Sampling deviates from the reference in one respect, deliberately: the reference
enumerates every hour, which at 9,181 stations would be ~3.7e9 samples. This uses
stride 24 -- one sample per day at 23:00, per PLAN.md 3.1 and Gauch et al. -- so
the last 24 hourly steps of every sample are one calendar day.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DYN = ("pet", "pcp", "temp")
DAILY_WINDOW = 24


def pick_samples_per_station(
    station_idx: np.ndarray,
    per_station: int,
    seed: int = 0,
    complement: bool = False,
) -> np.ndarray:
    """Boolean mask keeping up to ``per_station`` samples for each station.

    The cache analogue of pick_per_station on the prepared path: a
    median-across-stations metric needs every station represented, not a random
    slice of the pool that happens to cover a few hundred of them.
    ``complement`` returns what the same call would NOT have picked, which is how
    the reported set stays disjoint from the one early stopping selected on.
    """
    rng = np.random.default_rng(seed)
    keep = np.zeros(station_idx.size, dtype=bool)
    order = np.argsort(station_idx, kind="stable")
    ordered = station_idx[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(ordered)) + 1, ordered.size]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        block = order[lo:hi]
        if 0 < per_station < block.size:
            block = block[rng.choice(block.size, size=per_station, replace=False)]
        keep[block] = True
    return ~keep if complement else keep


def load_cache(cache_dir: str | os.PathLike, stride: int = 24, logger=None) -> dict:
    """Open the memmaps and the sample index written by scripts.build_hourly_cache."""
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "cache_meta.json").read_text())
    samples = np.load(cache_dir / f"samples_stride{stride}.npz", allow_pickle=True)

    forcing = np.load(cache_dir / "forcing.f32", mmap_mode="r")
    daily_path = cache_dir / "daily.f32"
    if not daily_path.exists():
        raise FileNotFoundError(
            f"{daily_path} missing -- rerun scripts.build_hourly_cache (it writes the daily "
            "cache during the scan phase; --skip-memmap reuses forcing.f32)"
        )
    # 1.7 GiB: read it into RAM outright, it is touched for every sample.
    daily = np.array(np.load(daily_path, mmap_mode="r"))

    cache = {
        "meta": meta,
        "forcing": forcing,
        "daily": daily,
        "stations": [str(s) for s in samples["stations"]],
        "station_idx": samples["station_idx"],
        "target_idx": samples["target_idx"],
        "is_train": samples["is_train"],
        "station_y_std": samples["station_y_std"],
        "stride": int(samples["stride"]),
    }
    if logger:
        logger.info(
            "cache: %d stations x %d hours | daily %s in RAM (%.2f GiB) | %d samples "
            "(%d train / %d val)",
            meta["n_stations"], meta["n_hours"], daily.shape, daily.nbytes / 1024**3,
            cache["station_idx"].size, int(cache["is_train"].sum()),
            int((~cache["is_train"]).sum()),
        )
    return cache


class HourlyWindowDataset(Dataset):
    """One item is a chunk of samples, so the loader convention matches the rest."""

    def __init__(
        self,
        cache: dict,
        stations: set[str],
        split: str,
        static: np.ndarray,
        scalers: dict,
        lookback_hourly: int = 168,
        lookback_daily: int = 365,
        chunk_size: int = 512,
        with_daily: bool = False,
        min_daily_hours: int = 18,
        rng_seed: int = 0,
        sample_mask: np.ndarray | None = None,
        logger=None,
    ):
        if split not in {"training", "validation"}:
            raise ValueError(f"split must be training|validation, got {split!r}")
        self.cache = cache
        self.forcing = cache["forcing"]
        self.daily = cache["daily"]
        self.static = static
        self.lookback_hourly = int(lookback_hourly)
        self.lookback_daily = int(lookback_daily)
        self.chunk_size = int(chunk_size)
        self.with_daily = bool(with_daily)
        self.min_daily_hours = int(min_daily_hours)

        names = cache["stations"]
        keep_station = np.fromiter((s in stations for s in names), dtype=bool, count=len(names))
        want_train = split == "training"
        mask = keep_station[cache["station_idx"]] & (cache["is_train"] == want_train)
        if sample_mask is not None:
            mask &= sample_mask
        self.sample_mask = mask

        self.station_idx = cache["station_idx"][mask]
        self.target_idx = cache["target_idx"][mask]
        self.station_y_std = cache["station_y_std"]
        self.station_names = names

        if self.station_idx.size == 0:
            raise ValueError(f"no {split} samples for the {len(stations)} requested stations")

        self.dyn_mean = np.array([scalers["x_dyn_mean"][n] for n in DYN], dtype=np.float32)
        self.dyn_std = np.array([scalers["x_dyn_std"][n] for n in DYN], dtype=np.float32)
        self.y_mean = float(scalers["y_mean"])
        self.y_std = float(scalers["y_std"])

        order = np.random.default_rng(rng_seed).permutation(self.station_idx.size)
        self.station_idx = self.station_idx[order]
        self.target_idx = self.target_idx[order]
        self.n_chunks = int(np.ceil(self.station_idx.size / self.chunk_size))

        if logger:
            logger.info(
                "%s: %d samples over %d stations -> %d chunks of <=%d",
                split, self.station_idx.size,
                len(np.unique(self.station_idx)), self.n_chunks, self.chunk_size,
            )

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> dict:
        lo = idx * self.chunk_size
        ks = self.station_idx[lo : lo + self.chunk_size]
        ts = self.target_idx[lo : lo + self.chunk_size]
        n = ks.size

        k_h, k_d = self.lookback_hourly, self.lookback_daily
        x_h = np.empty((n, k_h, 3), dtype=np.float32)
        x_d = np.empty((n, k_d, 3), dtype=np.float32)
        y = np.empty(n, dtype=np.float32)

        for i, (k, t) in enumerate(zip(ks, ts)):
            # H branch: x[t-168 : t], excluding t, exactly as the reference slices it.
            x_h[i] = self.forcing[k, t - k_h : t, :3]
            # D branch: the 365 whole days ending with the day that contains t-1.
            day_end = t // DAILY_WINDOW
            x_d[i] = self.daily[k, day_end - k_d : day_end]
            y[i] = self.forcing[k, t, 3]

        # A non-finite input makes the loss NaN and training continues to "success":
        # a first run exited COMPLETED on all five folds with every metric NaN.
        # Fail here instead, naming the sample so it can be traced.
        for name, arr in (("H", x_h), ("D", x_d)):
            if not np.isfinite(arr).all():
                bad = int(np.argmax(~np.isfinite(arr).all(axis=(1, 2))))
                raise ValueError(
                    f"non-finite values in the {name} branch for station "
                    f"{self.station_names[ks[bad]]} at hour {ts[bad]} -- the sample index "
                    "and the dataset disagree about what gets read; rebuild the cache index"
                )

        x_h = (x_h - self.dyn_mean) / self.dyn_std
        x_d = (x_d - self.dyn_mean) / self.dyn_std
        y = (y - self.y_mean) / self.y_std

        stations = [self.station_names[k] for k in ks]
        out = {
            "x": {
                "H": torch.from_numpy(x_h),
                "D": torch.from_numpy(x_d),
                "S": torch.from_numpy(self.static[ks]),
            },
            "y": torch.from_numpy(y),
            "stn_std": torch.from_numpy(self.station_y_std[ks].astype(np.float32)),
            "stations": stations,
            "hours": torch.from_numpy(np.asarray(ts, dtype=np.int64)),
            "y_daily": None,
            "daily_mask": None,
        }

        if self.with_daily:
            # Targets for the daily-only transfer: the mean of the 24 hourly
            # observations covering the same calendar day as the last 24 hourly
            # predictions. Every sample sits at 23:00 and its window was checked
            # finite, so the day is complete -- the mask is all-ones and only kept
            # so the loss signature matches the prepared-batch path.
            y_daily = np.empty(n, dtype=np.float32)
            ok = np.ones(n, dtype=bool)
            for i, (k, t) in enumerate(zip(ks, ts)):
                hours = self.forcing[k, t - DAILY_WINDOW + 1 : t + 1, 3]
                finite = np.isfinite(hours)
                if finite.sum() < self.min_daily_hours:
                    ok[i] = False
                    y_daily[i] = np.nan
                else:
                    y_daily[i] = hours[finite].mean()
            y_daily = (y_daily - self.y_mean) / self.y_std
            out["y_daily"] = torch.from_numpy(np.where(ok, y_daily, np.nan).astype(np.float32))
            out["daily_mask"] = torch.ones((n, DAILY_WINDOW), dtype=torch.bool)

        return out


def build_static_matrix(cache: dict, scalers: dict, cfg, logger=None) -> tuple[np.ndarray, list[str]]:
    """Standardized static rows for every station in the cache, in cache order."""
    from data.africa import HOURLY_STATIC, build_static_matrix as build_from_table, apply_onehot
    from data.dataset import load_dataset_config, resolve_static_spec

    names_all = list(load_dataset_config(cfg.data.root)["static_features"])
    static_keep, onehot_specs, out_names = resolve_static_spec(
        cfg.data.root, cfg.data.get("static_exclude"), cfg.data.get("onehot_static")
    )
    full, _ = build_from_table(
        cache["stations"], names_all, scalers, static_path=HOURLY_STATIC, logger=logger
    )
    static = apply_onehot(full, static_keep, onehot_specs)
    if logger:
        logger.info("static matrix %s for %d cache stations", static.shape, len(cache["stations"]))
    return static.astype(np.float32), out_names
