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


def load_cache(
    cache_dir: str | os.PathLike,
    stride: int = 24,
    logger=None,
    index_name: str | None = None,
) -> dict:
    """Open the memmaps and the sample index written by scripts.prep.build_hourly_cache.

    ``index_name`` swaps in a different sample index over the same memmaps -- used
    to score an all-hours evaluation set (samples_evalhours.npz) against a model
    trained on the stride-24 index, without touching the forcing cache.
    """
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "cache_meta.json").read_text())
    index_path = cache_dir / (index_name or f"samples_stride{stride}.npz")
    if not index_path.exists():
        raise FileNotFoundError(f"sample index {index_path} missing")
    samples = np.load(index_path, allow_pickle=True)

    forcing = np.load(cache_dir / "forcing.f32", mmap_mode="r")
    daily_path = cache_dir / "daily.f32"
    if not daily_path.exists():
        raise FileNotFoundError(
            f"{daily_path} missing -- rerun scripts.prep.build_hourly_cache (it writes the daily "
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
        hours = np.unique(cache["target_idx"] % 24)
        logger.info(
            "cache: %d stations x %d hours | daily %s in RAM (%.2f GiB) | %d samples "
            "(%d train / %d val) | index %s covering %d hour(s)-of-day%s",
            meta["n_stations"], meta["n_hours"], daily.shape, daily.nbytes / 1024**3,
            cache["station_idx"].size, int(cache["is_train"].sum()),
            int((~cache["is_train"]).sum()), index_path.name, hours.size,
            "" if hours.size > 1 else f" ({hours[0]:02d}:00 only)",
        )
    return cache


WINDOW_ALIGNMENTS = ("lagged", "figure")


def window_bounds(t: int, k_h: int, alignment: str = "lagged", daily_shift: int = 0,
                  day0: int = 0) -> tuple[int, int, int]:
    """Hourly slice ``[h_lo, h_hi)`` and the exclusive end of the daily slice, for target hour t.

    ``"lagged"`` is the layout of every run recorded before 2026-10-18: hourly
    ``x[t-k_h : t]`` (target hour excluded) and daily days ending with the day BEFORE the
    one containing t. For a 23:00 target that leaves the daily branch a day behind the
    hourly one and the hand-off ~23 h early -- an off-by-one against the design, which
    the 100-station reference and Gauch et al. both align.

    ``"figure"`` is the design as drawn in the architecture figure (fig_architecture.py): both
    branches end at the target hour. Hourly ``x[t-k_h+1 : t+1]``; daily ends with the last
    COMPLETE day at or before t, i.e. day d itself for a 23:00 target and day d-1 for any
    earlier hour, so no forcing after t is ever read. With k_h = 14 x 24 and a 23:00
    target the hand-off state (after daily step 365-14) ends exactly where the hourly
    window begins.

    ``day0`` is the index of the first 00:00 in the series (non-zero for Africa).
    """
    if alignment == "figure":
        return t - k_h + 1, t + 1, (t - day0 + 1) // DAILY_WINDOW
    if alignment != "lagged":
        raise ValueError(f"window alignment must be one of {WINDOW_ALIGNMENTS}, got {alignment!r}")
    return t - k_h, t, (t - day0) // DAILY_WINDOW + daily_shift


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
        daily_includes_target_day: bool = False,
        window_alignment: str = "lagged",
    ):
        if split not in {"training", "validation"}:
            raise ValueError(f"split must be training|validation, got {split!r}")
        if window_alignment not in WINDOW_ALIGNMENTS:
            raise ValueError(f"window_alignment must be one of {WINDOW_ALIGNMENTS}")
        if window_alignment == "figure" and daily_includes_target_day:
            raise ValueError("daily_includes_target_day is the lagged layout's baseline "
                             "patch; the figure layout already ends the daily branch on day d")
        self.window_alignment = window_alignment
        # False (the default, and every result recorded before 2026-10-18): the daily
        # branch ends with day d-1, as MTS-LSTM intends -- the hourly branch covers day d.
        # True: the daily branch ends with day d itself. Only meaningful for a standalone
        # daily LSTM read on its daily head, which otherwise never sees the forcing of the
        # day it predicts. The sample set is identical either way.
        self.daily_shift = 1 if daily_includes_target_day else 0
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
            # Slice bounds come from window_bounds(), which documents both layouts. Under
            # "lagged": H = x[t-k_h : t], D = whole days ending the day before t's (or t's
            # own day with daily_includes_target_day). Under "figure": both end at t.
            h_lo, h_hi, day_end = window_bounds(t, k_h, self.window_alignment,
                                                self.daily_shift)
            x_h[i] = self.forcing[k, h_lo:h_hi, :3]
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
            # predictions. The mask is all-ones, so the PREDICTION is averaged over
            # all 24 slots while the TARGET is averaged over the finite ones. Those
            # coincide for 93.7% of valid samples but not for all: valid_targets()
            # checks the three forcing columns over the lookback and q_mm at the
            # target hour only, NOT q_mm over the preceding 23 hours, so a day can
            # carry gaps. Measured over 60k samples: 2.7% have 18-23 finite hours
            # (the rest of the incomplete ones fall below min_daily_hours and are
            # dropped), with a median of 2 missing hours and a median longest gap of
            # 1 hour. Those partial days sit at 0.984 of their station's 30-day
            # median flow against 0.970 for complete days, i.e. no detectable
            # association between missingness and flow level, so averaging the
            # prediction over the full day is comparing two estimates of the same
            # daily mean rather than two different quantities. Left as is
            # deliberately; masking the prediction instead would trade an unbiased
            # full-day prediction for one degraded to match a telemetry artefact.
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
    # The June main experiment takes the 20250630 table (the module default). The
    # December track sets data.static_csv to its own snapshot, whose area and
    # elevation columns differ for about a third of the gauges.
    static_path = Path(cfg.data.get("static_csv") or HOURLY_STATIC)
    full, _ = build_from_table(
        cache["stations"], names_all, scalers, static_path=static_path, logger=logger
    )
    static = apply_onehot(full, static_keep, onehot_specs)
    if logger:
        logger.info("static matrix %s for %d cache stations", static.shape, len(cache["stations"]))
    return static.astype(np.float32), out_names
