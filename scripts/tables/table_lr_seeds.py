"""Two tables: the Step-2 learning rate and run-to-run variability across seeds, plus the
pretraining budget (epochs, best epoch) of the seed-42 folds.

Target-domain hourly KGE (23:00) pooled over the 8,709 target gauges of a design, as in the
other results scripts. Seeds: per seed the pooled median of M1 and the paired median M1 - M0; "largest fold
range" is, over the folds of a design, the largest range across seeds of the fold's median M1.

    python -m scripts.tables.table_lr_seeds     # -> reports/tables/{ft_lr.csv, seeds.csv, pretrain_budget.json}
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from scripts.figures.m1_vs_m0_figures import SEED_RUNS

OUT = Path("reports/tables")
LR_RUNS = {  # design -> Step-2 learning rate -> output root
    "random": {"5e-4": "al_v2_runB", "2e-4": "al_v2_lrsweep_daily_lr2e4", "1e-4": "al_v2_lrsweep_daily_lr1e4"},
    "blocked": {"5e-4": "al_v2_blocked", "2e-4": "al_v2lr2_blocked"},
}


def target(root: str, model: str = "M1") -> pd.Series:
    """Hourly KGE per (fold, gauge) on the target domain."""
    parts = []
    for p in sorted(glob.glob(f"outputs/{root}/fold*/transfer/per_station_hourly_fold*_{model}_target_hourly.csv")):
        f = pd.read_csv(p)
        f = f[(f.score_status == "ok") & (f.obs_std >= 1e-3)]
        f["fold"] = int(p.split("/fold")[1].split("/")[0])
        parts.append(f.set_index(["fold", "station_id"]).kge)
    return pd.concat(parts)


def ft_lr() -> pd.DataFrame:
    rows = [{"design": d, "lr": lr, "root": r, "n": len(s := target(r)), "M1": s.median()}
            for d, runs in LR_RUNS.items() for lr, r in runs.items()]
    return pd.DataFrame(rows)


def seeds() -> pd.DataFrame:
    rows = []
    for d, runs in SEED_RUNS.items():
        m1 = [target(r) for r in runs]
        gain = [(a - target(r, "M0")).dropna().median() for a, r in zip(m1, runs)]
        folds = pd.concat([s.groupby(level="fold").median() for s in m1], axis=1)
        rec = {"design": d, **{f"M1_s{k}": s.median() for k, s in zip((42, 43, 44), m1)}}
        rec["range"] = max(s.median() for s in m1) - min(s.median() for s in m1)
        rec["gain_min"], rec["gain_max"] = min(gain), max(gain)
        rec["largest_fold_range"] = (folds.max(axis=1) - folds.min(axis=1)).max()
        rows.append(rec)
    return pd.DataFrame(rows)


def pretrain_budget(max_epochs: int = 30) -> dict:
    """Seed-42 pretraining: folds that ran the full budget, and of those, best epoch in the last three."""
    full = late = n = 0
    for runs in SEED_RUNS.values():
        for p in sorted(glob.glob(f"outputs/{runs[0]}/fold*/pretrain/training_history.csv")):
            n += 1
            last = int(pd.read_csv(p).epoch.max())
            best = int(json.loads((Path(p).parent / "run_meta.json").read_text())["best_epoch"])
            full += last == max_epochs
            late += last == max_epochs and best > max_epochs - 3
    return {"folds": n, "reached_budget": full, "best_in_last_three": late}


def main() -> None:
    pd.set_option("display.width", 200)
    t = ft_lr()
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "ft_lr.csv", index=False)
    print(t.round(3).to_string(index=False), "\n")
    s = seeds()
    s.to_csv(OUT / "seeds.csv", index=False)
    print(s.round(3).to_string(index=False), "\n")
    b = pretrain_budget()
    (OUT / "pretrain_budget.json").write_text(json.dumps(b, indent=1))
    print(b)
    print("wrote", OUT / "ft_lr.csv", OUT / "seeds.csv", OUT / "pretrain_budget.json")


if __name__ == "__main__":
    main()
