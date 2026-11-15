"""Pretraining cost (Appendix B, Reproducibility): "a fold takes a median of 7.6 h and the five
cross-validation designs together about 175 GPU-hours".

Per fold, the sum of the per-epoch training time (`train_secs` in fold*/pretrain/
training_history.csv: the training pass only, without the per-epoch validation, checkpointing
and queueing). The run is chained over 4 h jobs that resume from a checkpoint, so the history
holds every epoch of the fold once. Per seed: the median over the 24 folds and the total.

    python -m scripts.numbers.pretrain_compute     # -> reports/methods/pretrain_compute.json
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.figures.sec41_figures import DESIGNS, SEED_RUNS

OUT = Path("reports/methods/pretrain_compute.json")


def fold_hours(root: str) -> dict[int, float]:
    out = {}
    for f in sorted(glob.glob(f"outputs/{root}/fold*/pretrain/training_history.csv")):
        k = int(f.split("/fold")[1].split("/")[0])
        out[k] = float(pd.read_csv(f)["train_secs"].sum() / 3600)
    return out


def main() -> None:
    res = {}
    for i, seed in enumerate((42, 43, 44)):
        per = {key: fold_hours(SEED_RUNS[key][i]) for key, _, _ in DESIGNS}
        hours = [h for d in per.values() for h in d.values()]
        res[f"seed{seed}"] = {
            "n_folds": len(hours), "median_hours_per_fold": float(np.median(hours)),
            "min_hours_per_fold": float(min(hours)), "max_hours_per_fold": float(max(hours)),
            "total_gpu_hours": float(sum(hours)),
            "per_design": {k: {"median": float(np.median(list(v.values()))), "total": float(sum(v.values())),
                               "per_fold": v} for k, v in per.items()}}
        print(f"seed {seed}: {len(hours)} folds | median {np.median(hours):.2f} h per fold "
              f"(range {min(hours):.1f}-{max(hours):.1f}) | total {sum(hours):.0f} GPU-hours")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
