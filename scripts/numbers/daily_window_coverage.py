"""How complete are the 24-hour windows behind the daily targets? (Sect. 2.1 and the loss paragraph)

A sample of the hourly cache is a gauge and a target hour t (23:00) whose forcing is complete over
the look-back and whose discharge is finite at t. Its daily observation is the mean of the finite
discharge values over the 24 hours ending at t, and it is used as a training target only if at
least 18 of them are finite (transfer.min_daily_hours). This script counts the finite hours in
randomly drawn windows:

  windows_20k   20,000 samples (rng seed 0): per archive, the share of windows with fewer than 18
                finite hours  -> "fewer than 1 % ... in every archive except CAMELS-AUS (21.8 %)"
  targets_60k   60,000 samples (rng seed 1): share of windows with 24, 18-23 and < 18 finite hours
                -> "about 2.7 % of the daily targets were computed from 18 to 23 valid hours"

Both draw from all samples of the cache (training and test period), as the numbers were first
computed; the same statistics restricted to the training period are written alongside.

    python -m scripts.numbers.daily_window_coverage     # -> reports/methods/daily_window_coverage.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CACHE = "/ibex/user/kongw0a/hourly_cache"
OUT = Path("reports/methods/daily_window_coverage.json")
MIN_HOURS = 18


def finite_hours(forcing, q: int, si: np.ndarray, ti: np.ndarray, pick: np.ndarray) -> np.ndarray:
    """Number of finite discharge hours in the 24 hours ending at each picked target hour."""
    out = np.empty(pick.size, dtype=np.int16)
    # read in (station, hour) order: the counts do not depend on the order, the disk does
    for j in np.lexsort((ti[pick], si[pick])):
        p = pick[j]
        out[j] = int(np.isfinite(np.asarray(forcing[si[p], ti[p] - 23: ti[p] + 1, q])).sum())
    return out


def by_archive(cnt: np.ndarray, src: np.ndarray) -> dict:
    return {s: {"n": int((src == s).sum()), "median_hours": float(np.median(cnt[src == s])),
                "share_below_18": float((cnt[src == s] < MIN_HOURS).mean())}
            for s in sorted(set(src))}


def classes(cnt: np.ndarray) -> dict:
    kept = cnt >= MIN_HOURS
    return {"n": int(cnt.size), "share_24": float((cnt == 24).mean()),
            "share_18_to_23": float(((cnt >= MIN_HOURS) & (cnt < 24)).mean()),
            "share_below_18": float((cnt < MIN_HOURS).mean()),
            "share_18_to_23_of_retained_targets": float((cnt[kept] < 24).mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-dir", default=CACHE)
    ap.add_argument("--out", default=OUT, type=Path)
    args = ap.parse_args()

    meta = json.loads((Path(args.cache_dir) / "cache_meta.json").read_text())
    q = meta["features"].index("q_mm")
    forcing = np.load(Path(args.cache_dir) / "forcing.f32", mmap_mode="r")
    z = np.load(Path(args.cache_dir) / "samples_stride24.npz", allow_pickle=True)
    si, ti, is_train = z["station_idx"], z["target_idx"], z["is_train"]
    archive = np.array([s.split("__")[0] for s in meta["stations"]])
    res = {"n_samples_in_cache": int(si.size), "min_daily_hours": MIN_HOURS}

    for name, n, seed in (("windows_20k", 20000, 0), ("targets_60k", 60000, 1)):
        for pool_name, pool in (("all_samples", np.arange(si.size)), ("training_period", np.flatnonzero(is_train))):
            rng = np.random.default_rng(seed)
            # all_samples reproduces the original draw exactly: rng.choice(si.size, n)
            pick = rng.choice(si.size, n, replace=False) if pool_name == "all_samples" else \
                rng.choice(pool, n, replace=False)
            cnt = finite_hours(forcing, q, si, ti, pick)
            rec = {"seed": seed, **classes(cnt)}
            if name == "windows_20k":
                rec["by_archive"] = by_archive(cnt, archive[si[pick]])
            res.setdefault(name, {})[pool_name] = rec
            print(f"{name:12s} {pool_name:16s} " + json.dumps({k: v for k, v in rec.items() if k != "by_archive"}))
            if name == "windows_20k":
                for s, v in rec["by_archive"].items():
                    print(f"    {s:14s} n={v['n']:6d}  below 18 h: {100 * v['share_below_18']:.2f} %")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
