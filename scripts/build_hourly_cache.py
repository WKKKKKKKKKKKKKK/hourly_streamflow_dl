"""Cache 6sources.nc as a memmap so true (365-daily, 168-hourly) windows are cheap.

Why this exists
---------------
The prepared hourly_q_dl batches store ONE power-law-subsampled 1000-step
sequence per sample. The reference 100-station experiment feeds the daily branch
365 GENUINE daily means, i.e. ``x[t-8760:t].reshape(365, 24, -1).mean(axis=1)``,
and that cannot be recovered from the subsample: only 8 of the 365 days carry 24
points, 176 carry exactly one, and 7 carry none. A single instantaneous sample is
a hopeless stand-in for a daily mean of precipitation, where most hours are zero.

So the windows are rebuilt from the raw hourly source the prepared batches were
themselves derived from.

What it writes (default under /ibex/user/kongw0a/hourly_cache)
--------------------------------------------------------------
  forcing.f32          memmap (n_stations, n_hours, 4) float32, RAW units,
                       feature order [pet, pcp, temp, q_mm]
  daily.f32            memmap (n_stations, n_days, 3) float32, RAW daily means of
                       [pet, pcp, temp], aligned so day d covers hours 24d..24d+23
  cache_meta.json      station list, time axis, shape, feature order
  samples_<stride>.npz  valid (station_idx, t_idx) pairs plus the per-station
                       temporal split boundary and the per-station target std

Raw rather than standardized on purpose: the reference standardizes the hourly
series FIRST and averages afterwards, so the dataset applies scalers.json at load
time.

The daily cache is what makes this affordable. Standardization is affine and the
mean is linear, so

    mean((x - mu) / sigma)  ==  (mean(x) - mu) / sigma

i.e. pre-averaging RAW hours and standardizing afterwards is IDENTICAL to the
reference's standardize-then-average, not an approximation. Reading the daily
branch from a 1.7 GiB array that fits in RAM instead of pulling 8760 hourly values
per sample cuts the per-batch read from 51 MiB to 1 MiB -- a 52x reduction, which
matters because the step was already GPU-bound by only a factor of ten.

A sample at ``t`` is valid when every one of the 8760 preceding hours is finite
for pet/pcp/temp and ``q_mm[t]`` is finite -- the same condition the reference
checks, since one NaN in the window makes its daily mean NaN. It is evaluated
with a cumulative-sum difference rather than a per-timestep slice, which is what
makes 9,181 stations tractable.

    python -m scripts.build_hourly_cache --stride 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.config import add_common_args, load_config
from common.utils import setup_logging

SOURCE_NC = "/ibex/project/c2266/abbaa0a/data/input_data/hourly_q_dl/6sources.nc"
DEFAULT_CACHE = "/ibex/user/kongw0a/hourly_cache"
FEATURES = ("pet", "pcp", "temp", "q_mm")
DYN = ("pet", "pcp", "temp")
LOOKBACK_HOURS = 8760          # 365 days, the reference's daily branch span
TARGET_HOUR_OF_DAY = 23        # so the last 24 hourly steps are one calendar day


def first_unwritten(path: Path, n_stations: int, logger) -> int:
    """Where a previous run stopped: the first station row still all zeros.

    403,248 x 4 genuinely-zero values never happens, so an all-zero row means
    unwritten. Lets a killed run resume instead of redoing hours of I/O.
    """
    if not path.exists():
        return 0
    existing = np.load(path, mmap_mode="r")
    if existing.shape[0] != n_stations:
        logger.warning("existing memmap has %d stations, expected %d -- starting over",
                       existing.shape[0], n_stations)
        return 0
    for k in range(n_stations):
        if np.count_nonzero(existing[k, ::20000, :]) == 0:
            return k
    return n_stations


def write_memmap(cache_dir: Path, logger, flush_every: int = 200) -> dict:
    """Stream 6sources.nc into a (station, hour, feature) float32 memmap.

    Flushes periodically: writing 55 GiB without it lets dirty pages pile up, and
    a first attempt on a login node was killed silently around station 5,600.
    """
    import netCDF4 as nc

    handle = nc.Dataset(SOURCE_NC)
    feature_names = [str(x) for x in handle.variables["dynamic_features"][:]]
    order = [feature_names.index(name) for name in FEATURES]
    stations = [k for k in handle.variables if k not in ("time", "dynamic_features")]
    time_var = handle.variables["time"]
    n_hours = len(time_var)

    meta = {
        "stations": stations,
        "n_stations": len(stations),
        "n_hours": n_hours,
        "features": list(FEATURES),
        "time_units": time_var.units,
        "source": SOURCE_NC,
        "dtype": "float32",
    }
    path = cache_dir / "forcing.f32"
    logger.info("writing %s: (%d, %d, %d) float32 = %.1f GiB",
                path, len(stations), n_hours, len(FEATURES),
                len(stations) * n_hours * len(FEATURES) * 4 / 1024**3)

    # Always a single w+ pass. An earlier attempt resumed in place with
    # np.load(mmap_mode="r+"); the log said it wrote stations 5857-9180 and yet
    # every one of them read back as zeros afterwards, on this weka filesystem.
    # The whole write takes ~4 minutes, so resume is not worth that risk.
    memmap = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(stations), n_hours, len(FEATURES))
    )
    for k in range(len(stations)):
        block = np.asarray(handle.variables[stations[k]][:, :], dtype=np.float32)
        memmap[k] = block[:, order]
        if (k + 1) % flush_every == 0:
            memmap.flush()
        if (k + 1) % 500 == 0:
            logger.info("  %d/%d stations", k + 1, len(stations))
    memmap.flush()
    del memmap
    handle.close()

    # Verify after reopening, every station. A partly-written cache is worse than
    # a failed build: zeros are finite, so they pass the validity check and would
    # quietly become thousands of stations of fake all-zero training data.
    check = np.load(path, mmap_mode="r")
    empty = [k for k in range(len(stations)) if np.count_nonzero(check[k, ::20000, :]) == 0]
    del check
    if empty:
        raise RuntimeError(
            f"{len(empty)} stations read back as all zeros after writing, e.g. "
            f"{empty[:5]} ({stations[empty[0]]}). The cache is unusable -- do not "
            "build the sample index from it."
        )
    logger.info("memmap written and verified: all %d stations non-empty", len(stations))

    with open(cache_dir / "cache_meta.json", "w", encoding="utf-8") as out:
        json.dump(meta, out, indent=2)
    return meta


def daily_means(block: np.ndarray, n_days: int) -> np.ndarray:
    """Raw daily means of [pet, pcp, temp]; day d covers hours 24d .. 24d+23.

    NaN propagates, so a day containing any missing hour becomes NaN -- the same
    thing that happens to the reference's daily branch, which is why the validity
    check can be read off either representation.
    """
    usable = n_days * 24
    return block[:usable, :3].reshape(n_days, 24, 3).mean(axis=1)


def valid_targets(block: np.ndarray, stride: int) -> np.ndarray:
    """Target hours whose full 8760-hour window is finite and whose q_mm exists.

    ``block`` is (n_hours, 4) raw. Uses a cumulative-sum difference so the whole
    station is one vectorised pass instead of a slice per timestep.
    """
    n_hours = block.shape[0]
    finite_dyn = np.isfinite(block[:, :3]).all(axis=1)
    csum = np.concatenate([[0], np.cumsum(finite_dyn, dtype=np.int64)])

    # candidate targets: hour-of-day 23, far enough in for a full window
    first = LOOKBACK_HOURS + ((TARGET_HOUR_OF_DAY - LOOKBACK_HOURS) % stride)
    candidates = np.arange(first, n_hours, stride, dtype=np.int64)
    candidates = candidates[candidates >= LOOKBACK_HOURS]
    if candidates.size == 0:
        return candidates

    # window is x[t-8760 : t], i.e. the 8760 hours strictly before t
    window_finite = (csum[candidates] - csum[candidates - LOOKBACK_HOURS]) == LOOKBACK_HOURS
    target_finite = np.isfinite(block[candidates, 3])
    return candidates[window_finite & target_finite]


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Build the hourly memmap cache."))
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--stride", type=int, default=24,
                        help="Hours between candidate targets. 24 = one sample per day "
                             "(Gauch et al.); stride 1 would give ~3.7e9 samples.")
    parser.add_argument("--train-frac", type=float, default=0.7,
                        help="Per-station temporal split, matching hourly_q_dl's 0.7/0.3 local split.")
    parser.add_argument("--skip-memmap", action="store_true", help="Reuse an existing forcing.f32.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(cache_dir / "build_hourly_cache.log")

    meta_path = cache_dir / "cache_meta.json"
    if args.skip_memmap and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        check = np.load(cache_dir / "forcing.f32", mmap_mode="r")
        empty = [k for k in range(meta["n_stations"]) if np.count_nonzero(check[k, ::20000, :]) == 0]
        del check
        if empty:
            raise RuntimeError(
                f"--skip-memmap but {len(empty)} stations are all zeros (e.g. {empty[:5]}). "
                "Rerun without --skip-memmap."
            )
        logger.info("reusing existing memmap for %d stations (verified non-empty)", meta["n_stations"])
    else:
        meta = write_memmap(cache_dir, logger)

    # --- valid samples, per-station split boundary, per-station target std ----
    from data.dataset import load_scalers

    scalers = load_scalers(cfg.data.root)
    y_mean, y_std = scalers["y_mean"], scalers["y_std"]

    forcing = np.load(cache_dir / "forcing.f32", mmap_mode="r")
    stations = meta["stations"]
    n_days = meta["n_hours"] // 24
    logger.info("scanning %d stations for valid targets (stride %d) ...", len(stations), args.stride)

    daily_path = cache_dir / "daily.f32"
    daily = np.lib.format.open_memmap(
        daily_path, mode="w+", dtype=np.float32, shape=(len(stations), n_days, 3)
    )
    logger.info("also writing %s: (%d, %d, 3) = %.2f GiB",
                daily_path, len(stations), n_days, len(stations) * n_days * 3 * 4 / 1024**3)

    station_idx, target_idx, is_train = [], [], []
    boundaries, stds = np.zeros(len(stations), dtype=np.int64), np.zeros(len(stations), dtype=np.float32)
    for k in range(len(stations)):
        block = np.asarray(forcing[k])
        daily[k] = daily_means(block, n_days)
        targets = valid_targets(block, args.stride)
        if targets.size == 0:
            continue

        # Local temporal split: the first train_frac of THIS station's valid
        # samples train, the rest validate -- the same 0.7/0.3 local convention
        # hourly_q_dl used. Derived here rather than read from the prepared
        # batches, so the boundary can differ from theirs by a few samples.
        cut = int(round(args.train_frac * targets.size))
        boundaries[k] = targets[cut] if cut < targets.size else targets[-1] + 1

        # Per-basin std of the STANDARDIZED target over the training part, which
        # is what the basin-averaged NSE loss divides by.
        y_train = (block[targets[:cut], 3] - y_mean) / y_std
        stds[k] = float(np.std(y_train)) if y_train.size >= 2 else 1.0

        station_idx.append(np.full(targets.size, k, dtype=np.int32))
        target_idx.append(targets.astype(np.int64))
        is_train.append(np.arange(targets.size) < cut)

        if (k + 1) % 500 == 0:
            logger.info("  %d/%d stations, %d samples so far",
                        k + 1, len(stations), sum(a.size for a in station_idx))

    daily.flush()
    del daily
    logger.info("daily cache written")

    station_idx = np.concatenate(station_idx)
    target_idx = np.concatenate(target_idx)
    is_train = np.concatenate(is_train)
    out = cache_dir / f"samples_stride{args.stride}.npz"
    np.savez_compressed(
        out,
        station_idx=station_idx,
        target_idx=target_idx,
        is_train=is_train,
        split_boundary=boundaries,
        station_y_std=stds,
        stations=np.array(stations, dtype=object),
        stride=args.stride,
        train_frac=args.train_frac,
        lookback_hours=LOOKBACK_HOURS,
        target_hour_of_day=TARGET_HOUR_OF_DAY,
        n_days=n_days,
    )
    logger.info("wrote %s", out)
    logger.info("valid samples: %d total | %d train / %d validation",
                station_idx.size, int(is_train.sum()), int((~is_train).sum()))
    logger.info("stations with any sample: %d/%d", len(np.unique(station_idx)), len(stations))
    per_station = np.bincount(station_idx, minlength=len(stations))
    nonzero = per_station[per_station > 0]
    logger.info("samples per station: min %d, median %d, max %d",
                nonzero.min(), int(np.median(nonzero)), nonzero.max())
    logger.info("per-basin std of the standardized target: median %.4f, min %.4f, max %.4f",
                float(np.median(stds[stds > 0])), float(stds[stds > 0].min()), float(stds.max()))


if __name__ == "__main__":
    main()
