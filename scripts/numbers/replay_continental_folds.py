"""Replay in the continental design, per held-out continent: median hourly KGE (23:00) of the target
gauges of each fold, M1 fine-tuned without and with source replay, per seed, and the paired change.

    python -m scripts.numbers.replay_continental_folds     # -> reports/forgetting_replay/continental_per_fold.csv
"""

from __future__ import annotations

import pandas as pd

from scripts.figures.m1_vs_m0_figures import SEED_RUNS
from scripts.figures.replay_figures import OUT, REPLAY_RUNS, scores

CONTINENTS = ["North America", "Oceania", "Europe", "Asia"]   # continental fold 0..3


def main() -> None:
    rows = []
    for seed, plain, rep in zip((42, 43, 44), SEED_RUNS["continental"], REPLAY_RUNS["continental"]):
        m0, m1, mr = (scores(r, m, "target") for r, m in ((plain, "M0"), (plain, "M1"), (rep, "M1")))
        for fold, name in enumerate(CONTINENTS):
            t = pd.DataFrame({"M0": m0.loc[fold], "M1": m1.loc[fold], "M1_replay": mr.loc[fold]}).dropna()
            rows.append({"seed": seed, "fold": fold, "continent": name, "n": len(t),
                         **t.median().to_dict(),
                         "d_M1_replay_vs_M1": float((t.M1_replay - t.M1).median()),
                         "frac_M1_replay_gt_M1": float((t.M1_replay > t.M1).mean())})
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "continental_per_fold.csv", index=False)
    print(t.round(3).to_string(index=False))
    print("wrote", OUT / "continental_per_fold.csv")


if __name__ == "__main__":
    main()
