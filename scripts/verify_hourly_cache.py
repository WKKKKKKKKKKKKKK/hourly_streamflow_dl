"""Check the rebuilt window pipeline against the prepared batches.

The new path is nc -> memmap -> window -> standardize, and every step can be
subtly wrong without raising: the wrong feature order, an hour's offset, the wrong
scaler, a station misalignment. None of those would crash; they would quietly make
the results worse.

The hourly branch has a ground truth to check against. Positions 928..999 of the
prepared 1000-step sequence are hours t-72 .. t-1 at 1-hour spacing, which is
exactly ``x[t-72 : t]`` -- the reference's hourly window. So for a station that
appears in the prepared batches, rebuild its window from the cache and compare
value by value.

Also checked, since they are cheap and equally silent when wrong:
  * y: the cache's q_mm at t against the batch's stored (already standardized) y
  * the daily cache: daily.f32[k, d] against the mean of forcing[k, 24d : 24d+24]
  * the daily branch: 365 means built from the cache against the same means
    computed straight from 6sources.nc

    python -m scripts.verify_hourly_cache --stations 6 --rows-per-batch 4
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.config import add_common_args, load_config, resolve
from common.utils import setup_logging
from data.dataset import load_dataset_config, load_scalers
from data.index import load_index

SOURCE_NC = "/ibex/project/c2266/abbaa0a/data/input_data/hourly_q_dl/6sources.nc"
DEFAULT_CACHE = "/ibex/user/kongw0a/hourly_cache"
DYN = ("pet", "pcp", "temp")


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Verify the rebuilt window pipeline."))
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--stations", type=int, default=6)
    parser.add_argument("--rows-per-batch", type=int, default=4)
    parser.add_argument("--lookback-hourly", type=int, default=None, help="Defaults to the config value.")
    parser.add_argument("--tol", type=float, default=2e-5)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    logger = setup_logging()
    cache_dir = Path(args.cache_dir)
    k_h = int(args.lookback_hourly or cfg.data.lookback_hourly)

    scalers = load_scalers(cfg.data.root)
    dyn_mean = np.array([scalers["x_dyn_mean"][n] for n in DYN], dtype=np.float64)
    dyn_std = np.array([scalers["x_dyn_std"][n] for n in DYN], dtype=np.float64)
    y_mean, y_std = scalers["y_mean"], scalers["y_std"]

    offsets = json.loads((Path("data") / "lookback_offsets.json").read_text())
    hours_ago = np.asarray(offsets["hours_ago"])
    seq_len = int(offsets["seq_len"])
    # x[t-k:t] = hours_ago k..1, which is positions (seq_len-1-k) .. (seq_len-2).
    # The last position is hours_ago 0, the target hour, which the reference excludes.
    start = seq_len - 1 - k_h
    stop = seq_len - 1
    if not np.array_equal(hours_ago[start:stop], np.arange(k_h, 0, -1)):
        raise SystemExit(
            f"positions {start}..{seq_len - 1} are not 1-hour spaced; lookback_hourly={k_h} "
            f"reaches past the hourly tail ({offsets['hourly_tail_positions']})"
        )
    logger.info("comparing prepared positions %d..%d (hours_ago %d..%d) against x[t-%d:t]",
                start, stop - 1, hours_ago[start], hours_ago[stop - 1], k_h)

    meta = json.loads((cache_dir / "cache_meta.json").read_text())
    cache_stations = {name: i for i, name in enumerate(meta["stations"])}
    forcing = np.load(cache_dir / "forcing.f32", mmap_mode="r")
    daily = np.load(cache_dir / "daily.f32", mmap_mode="r")

    frame = load_index(resolve(cfg.data.index_dir), "training")
    regular = frame.loc[frame["kind"] == "regular"]
    first = regular.groupby("station", sort=True)["prefix"].first()
    picks = first.iloc[:: max(1, len(first) // args.stations)][: args.stations]
    logger.info("verifying %d stations from %s",
                len(picks), sorted({s.split("__")[0] for s in picks.index}))

    import netCDF4 as nc

    source = nc.Dataset(SOURCE_NC)
    time_var = source.variables["time"]
    times = pd.DatetimeIndex(
        nc.num2date(time_var[:], time_var.units, only_use_cftime_datetimes=False)
    )

    h_err = y_err = daily_err = d_branch_err = 0.0
    checked = 0
    for station, prefix in picks.items():
        if station not in cache_stations:
            raise SystemExit(f"{station} is in the prepared index but not in the cache")
        k = cache_stations[station]

        x_dyn, _ = torch.load(
            f"{cfg.data.root}/training/{prefix}_x.pt", map_location="cpu", weights_only=True
        )
        y_stored = torch.load(
            f"{cfg.data.root}/training/{prefix}_y.pt", map_location="cpu", weights_only=True
        ).reshape(-1)
        with open(f"{cfg.data.root}/training/{prefix}_metadata.pkl", "rb") as handle:
            md = pickle.load(handle)

        rows = np.linspace(0, len(md) - 1, args.rows_per_batch).astype(int)
        for row in rows:
            t = times.get_loc(md["index"].iloc[row])

            built_h = (np.asarray(forcing[k, t - k_h : t, :3], dtype=np.float64) - dyn_mean) / dyn_std
            h_err = max(h_err, float(np.abs(x_dyn[row, start:stop].numpy() - built_h).max()))

            built_y = (float(forcing[k, t, 3]) - y_mean) / y_std
            y_err = max(y_err, abs(float(y_stored[row]) - built_y))

            # daily branch: 365 whole days ending with the day containing t-1
            day_end = t // 24
            from_cache = np.asarray(daily[k, day_end - 365 : day_end], dtype=np.float64)
            from_nc = np.asarray(
                source.variables[station][(day_end - 365) * 24 : day_end * 24, :3], dtype=np.float64
            ).reshape(365, 24, 3).mean(axis=1)
            both = np.isfinite(from_cache) & np.isfinite(from_nc)
            if both.any():
                d_branch_err = max(d_branch_err, float(np.abs(from_cache[both] - from_nc[both]).max()))
            checked += 1

        # daily cache against the hourly memmap it was derived from
        for day in np.linspace(400, meta["n_hours"] // 24 - 2, 4).astype(int):
            got = np.asarray(daily[k, day], dtype=np.float64)
            want = np.asarray(forcing[k, day * 24 : (day + 1) * 24, :3], dtype=np.float64).mean(axis=0)
            both = np.isfinite(got) & np.isfinite(want)
            if both.any():
                daily_err = max(daily_err, float(np.abs(got[both] - want[both]).max()))
    source.close()

    logger.info("")
    logger.info("checked %d (station, target hour) pairs", checked)
    logger.info("  H branch  x[t-%d:t] vs prepared[%d:%d]        max |err| = %.3g", k_h, start, stop, h_err)
    logger.info("  y         cache q_mm[t] vs stored y          max |err| = %.3g", y_err)
    logger.info("  daily     daily.f32 vs mean of forcing       max |err| = %.3g", daily_err)
    logger.info("  D branch  365 means from cache vs from nc    max |err| = %.3g", d_branch_err)

    worst = max(h_err, y_err, daily_err, d_branch_err)
    if worst < args.tol:
        logger.info("")
        logger.info("VERIFIED to within %.1g: station order, feature order, time alignment, "
                    "scalers and the daily aggregation are all correct.", args.tol)
    else:
        logger.error("MISMATCH: worst %.3g exceeds the %.1g tolerance. Do not train on this cache.",
                     worst, args.tol)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
