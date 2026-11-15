"""Rain-gauge density of the African catchments against the training gauges (Appendix D):
"The median rain gauge density is 0.027 for the African basins against 0.787 for the training
sites, and every African basin falls below the training median."

Density is the archive's `rain_gauge_density` static attribute (GHCN-Daily gauges, after Abbas et
al.; see the header of the static tables), from the same 20250630 archive the model inputs come
from (data/africa.py):

  training   the 9,181 gauges of the hourly cache, hourly static table
  africa     the African catchments scored in Appendix D (the union of the gauges in
             outputs/al_v2_africa_m1b_fold*/paired_M0_M1.csv), daily static table;
             all 294 catchments of africa/africa_basins.csv are reported alongside

    python -m scripts.numbers.rain_gauge_density     # -> reports/methods/rain_gauge_density.json
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from data.africa import DAILY_STATIC, HOURLY_STATIC, read_static_table

CACHE_META = "/ibex/user/kongw0a/hourly_cache/cache_meta.json"
SCORED = "outputs/al_v2_africa_m1b_fold*/paired_M0_M1.csv"
OUT = Path("reports/methods/rain_gauge_density.json")


def density(table: pd.DataFrame, ids) -> pd.Series:
    return pd.to_numeric(table.reindex(list(ids))["rain_gauge_density"], errors="coerce").dropna()


def summary(v: pd.Series, ref: float) -> dict:
    return {"n": int(v.size), "median": float(v.median()), "q1": float(v.quantile(0.25)),
            "q3": float(v.quantile(0.75)), "max": float(v.max()),
            "share_below_training_median": float((v < ref).mean())}


def main() -> None:
    hourly_ids = json.loads(Path(CACHE_META).read_text())["stations"]
    d_train = density(read_static_table(HOURLY_STATIC), hourly_ids)
    ref = float(d_train.median())
    daily = read_static_table(DAILY_STATIC)
    scored = set(pd.concat([pd.read_csv(f) for f in sorted(glob.glob(SCORED))])["station_id"].astype(str))
    study = pd.read_csv("africa/africa_basins.csv").station_id.astype(str)
    res = {"training": {"n_hourly_gauges": len(hourly_ids), **summary(d_train, ref)},
           "africa_scored": {"n_scored_basins": len(scored), **summary(density(daily, sorted(scored)), ref)},
           "africa_all_294": summary(density(daily, study), ref)}
    for k, v in res.items():
        print(f"{k:16s} n={v['n']:5d}  median {v['median']:.4f}  max {v['max']:.4f}  "
              f"below training median {100 * v['share_below_training_median']:.0f} %")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
