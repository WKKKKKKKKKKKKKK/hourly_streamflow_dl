"""An all-hours evaluation index for the cache path, to expose a hidden confound.

``build_hourly_cache`` samples targets at stride 24, and because a target must sit
at hour 23 so the last 24 hourly steps form one calendar day, EVERY sample in
``samples_stride24.npz`` has ``target_idx % 24 == 23``. Training on that is a
deliberate cost/coverage tradeoff. Reporting hourly KGE on it is not: the prepared
path (run A) scores all 24 hours uniformly at ~4.2% each, so the two runs' "hourly
KGE" are not the same quantity. Daily-mean supervision is expected to do its damage
to intra-day shape, which a once-daily snapshot cannot see.

This rebuilds the index with stride 1 -- every valid hour -- then subsamples per
station so the file stays a manageable size and each station keeps a uniform spread
over the clock. Nothing is retrained; only what gets scored changes.

Note what this can and cannot settle. At stride 24 the daily branch always ends at
the midnight 24 h before the target, so the model only ever saw ONE (daily-end,
target) alignment. Scoring at hour 10 presents a 10-hour gap it never trained on.
So a drop at other hours is a real cost of the stride-24 design, but it conflates
"intra-day shape is harder" with "this alignment is out of distribution" -- the
per-hour breakdown this enables is what separates them.

    python -m scripts.build_eval_index --per-station 6144
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.build_hourly_cache import DEFAULT_CACHE, LOOKBACK_HOURS, valid_targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an all-hours evaluation sample index.")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--per-station", type=int, default=6144,
                        help="Validation samples to keep per station (run A reports ~6020).")
    parser.add_argument("--lookback-daily", type=int, default=365)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default=None, help="Defaults to samples_evalhours.npz in the cache.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    base = np.load(cache_dir / "samples_stride24.npz", allow_pickle=True)
    stations = [str(s) for s in base["stations"]]
    boundary = base["split_boundary"]

    forcing = np.load(cache_dir / "forcing.f32", mmap_mode="r")
    daily_all = np.load(cache_dir / "daily.f32", mmap_mode="r")
    n_stations = len(stations)
    if forcing.shape[0] != n_stations:
        raise ValueError(f"forcing has {forcing.shape[0]} stations, index has {n_stations}")

    rng = np.random.default_rng(args.seed)
    keep_station, keep_target = [], []
    n_all_hours = 0
    started = time.time()

    for k in range(n_stations):
        block = np.asarray(forcing[k])
        targets = valid_targets(block, np.asarray(daily_all[k]), stride=1,
                                lookback_daily=args.lookback_daily)
        # Validation period only -- the same split the training run used.
        targets = targets[targets >= boundary[k]]
        n_all_hours += targets.size
        if targets.size == 0:
            continue
        if 0 < args.per_station < targets.size:
            targets = np.sort(rng.choice(targets, size=args.per_station, replace=False))
        keep_station.append(np.full(targets.size, k, dtype=np.int32))
        keep_target.append(targets.astype(np.int64))

        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{n_stations} stations | {sum(a.size for a in keep_target):,} kept "
                  f"| {time.time() - started:.0f}s", flush=True)

    station_idx = np.concatenate(keep_station)
    target_idx = np.concatenate(keep_target)
    hours = target_idx % 24
    counts = np.bincount(hours, minlength=24)

    out = Path(args.out) if args.out else cache_dir / "samples_evalhours.npz"
    np.savez(
        out,
        station_idx=station_idx,
        target_idx=target_idx,
        # Every sample here is a validation target, but the loaders filter on this
        # flag, so it has to be present and False.
        is_train=np.zeros(station_idx.size, dtype=bool),
        split_boundary=boundary,
        station_y_std=base["station_y_std"],
        stations=np.array(stations, dtype=object),
        stride=np.int64(1),
        train_frac=base["train_frac"],
        lookback_hours=np.int64(LOOKBACK_HOURS),
        target_hour_of_day=np.int64(-1),   # -1 = all hours, not a single fixed hour
        n_days=base["n_days"],
        per_station=np.int64(args.per_station),
    )

    print(f"\nwrote {out}")
    print(f"  {station_idx.size:,} samples over {np.unique(station_idx).size} stations "
          f"(of {n_all_hours:,} valid all-hour targets)")
    print(f"  hour-of-day coverage: {counts.min():,} to {counts.max():,} per hour "
          f"({counts.min() / counts.sum():.4f} to {counts.max() / counts.sum():.4f} share)")
    if counts.min() == 0:
        raise RuntimeError("some hour of day has no samples -- the index is not all-hours")

    (cache_dir / "evalhours_meta.json").write_text(json.dumps({
        "n_samples": int(station_idx.size),
        "n_stations": int(np.unique(station_idx).size),
        "per_station": args.per_station,
        "hour_counts": counts.tolist(),
        "n_valid_all_hour_targets": int(n_all_hours),
    }, indent=2))


if __name__ == "__main__":
    main()
