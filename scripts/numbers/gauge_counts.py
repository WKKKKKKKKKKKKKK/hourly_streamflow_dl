"""From the gauges of the hourly dataset to the gauges every result is computed on.

  dataset         gauges of the hourly dataset (6sources.nc, = the hourly cache)
  indexed         gauges with batches in the prepared (pre-batched) dataset: index/stations.csv
  in_folds        gauges with batches in both the training and the test period: the fold tables;
                  every fold of every run has these as source + target (run_meta.json)
  target_results  gauges with a transfer-stage result as a target (per_station_*_target_hourly.csv)
  scored          score_status ok (>= eval.min_samples_per_station test days, non-constant
                  discharge) and obs_std >= 1e-3: the 8,709 of every hourly statistic
  quantile_mapping  scored gauges with 365 or more training days to fit the correction

Every step is checked on every run of the five designs and three seeds; the counts must agree.

    python -m scripts.numbers.gauge_counts     # -> reports/methods/gauge_counts.json
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.figures.m1_vs_m0_figures import SEED_RUNS, scored_gauges

CACHE_META = "/ibex/user/kongw0a/hourly_cache/cache_meta.json"
CONFIG = "configs/aligned/phase1_runB_v2.yaml"
FOLD_FILES = {"random": "folds/folds_random.csv", "blocked": "folds/folds_blocked.csv",
              "climate": "folds/folds_climate.csv", "longitudinal": "folds/folds_longitudinal.csv",
              "continental": "folds/folds_continental.csv"}
OUT = Path("reports/methods/gauge_counts.json")


def archive(ids) -> dict:
    return {k: int(v) for k, v in pd.Series([s.split("__")[0] for s in ids]).value_counts().items()}


def one(values: list[int], what: str) -> int:
    assert len(set(values)) == 1, f"{what} differs between runs: {sorted(set(values))}"
    return values[0]


def main() -> None:
    dataset = json.loads(Path(CACHE_META).read_text())["stations"]
    index = pd.read_csv("index/stations.csv")
    folds = {d: pd.read_csv(f) for d, f in FOLD_FILES.items()}
    in_folds = one([len(f) for f in folds.values()], "fold table size")
    assert all(set(f.station_id) == set(folds["random"].station_id) for f in folds.values())
    no_split = index[~index.station_id.isin(folds["random"].station_id)]
    min_test_days = yaml.safe_load(open(CONFIG))["eval"]["min_samples_per_station"]

    meta_sum, results, reasons, scored, near_const, qm = [], [], [], [], [], []
    for design, runs in SEED_RUNS.items():
        for run in runs:
            for p in sorted(glob.glob(f"outputs/{run}/fold*/pretrain/run_meta.json")):
                m = json.loads(Path(p).read_text())
                meta_sum.append(m["n_source_stations"] + m["n_target_stations"])
            f = pd.concat(pd.read_csv(p) for p in glob.glob(
                f"outputs/{run}/fold*/transfer/per_station_hourly_fold*_M1_target_hourly.csv"))
            results.append(f.station_id.nunique())
            reasons.append(tuple(sorted(f.exclusion_reason.value_counts().items())))
            ok = f[f.score_status == "ok"]
            near_const.append(int((ok.obs_std < 1e-3).sum()))
            s = scored_gauges(run)
            scored.append(len(s))
            c = pd.concat(pd.read_csv(p) for p in glob.glob(f"outputs/{run}/correction_baseline_v3/correction_fold*.csv"))
            qm.append(len(set(c.station_id) & s))
    reasons = one(reasons, "exclusion reasons")
    res = {
        "dataset": len(dataset),
        "indexed": len(index),
        "not_indexed": len(set(dataset) - set(index.station_id)),
        "in_folds": in_folds,
        "dropped_no_training_batches": int((no_split.n_training_batches == 0).sum()),
        "dropped_no_test_batches": int((no_split.n_validation_batches == 0).sum()),
        "source_plus_target_per_fold": one(meta_sum, "source + target per fold"),
        "target_results": one(results, "target results"),
        "excluded": dict(reasons),
        "min_test_days": int(min_test_days),
        "near_constant_obs_std_below_1e-3": one(near_const, "near-constant gauges"),
        "scored": one(scored, "scored gauges"),
        "quantile_mapping": one(qm, "gauges with a quantile-mapping correction"),
        "by_archive": {"dataset": archive(dataset), "in_folds": archive(folds["random"].station_id),
                       "scored": archive(scored_gauges(SEED_RUNS["random"][0]))},
    }
    assert res["indexed"] + res["not_indexed"] == res["dataset"]
    assert res["in_folds"] + res["dropped_no_training_batches"] + res["dropped_no_test_batches"] == res["indexed"]
    assert res["source_plus_target_per_fold"] == res["in_folds"]
    for k, v in res.items():
        if k != "by_archive":
            print(f"{k:36s} {v}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
