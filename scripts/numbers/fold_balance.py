"""Per fold: share of gauges left for pretraining against the held-out fold's M0, M1 and gain.

Asks whether unequal folds (climate zone, continental) explain fold-level skill. Scores are
medians over the fold's scored target gauges, then the median over three seeds (M1 also its
range). Station counts come from the fold files, which match the counts recorded in every
run's run_meta.json.

    python -m scripts.numbers.fold_balance     # -> reports/sec41_m1_vs_m0/fold_balance.csv, fold_balance_spearman.json
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = "reports/sec41_m1_vs_m0/fold_balance.csv"
OUT_RHO = "reports/sec41_m1_vs_m0/fold_balance_spearman.json"
DESIGN_FOLDS = ("random", "blocked", "climate", "longitudinal", "continental")


def fold_stats(root: str, k: int) -> tuple[float, float, float, int]:
    def load(tag):
        f = pd.read_csv(glob.glob(f"outputs/{root}/fold{k}/transfer/per_station_hourly_fold{k}_{tag}_target_hourly.csv")[0])
        return f[(f.score_status == "ok") & (f.obs_std >= 1e-3)].set_index("station_id").kge
    j = pd.concat([load("M0"), load("M1")], axis=1, keys=["M0", "M1"]).dropna()
    return j.M0.median(), j.M1.median(), (j.M1 - j.M0).median(), len(j)


def main() -> None:
    import scripts.figures.sec41_figures as m

    rows = []
    for name in DESIGN_FOLDS:
        f = pd.read_csv(f"folds/folds_{name}.csv")
        for k in sorted(f.fold.unique()):
            st = [fold_stats(r, k) for r in m.SEED_RUNS[name]]
            if name == "continental":
                lab = f.loc[f.fold == k, "continent"].iloc[0]
            elif name == "climate":
                lab = "Koppen " + f.loc[f.fold == k, "kgz_major"].value_counts().idxmax()
            else:
                lab = ""
            rows.append(dict(design=name, fold=k, held_out=lab, source_pct=100 * (f.fold != k).mean(),
                             n_scored=st[0][3], M0=np.median([s[0] for s in st]), M1=np.median([s[1] for s in st]),
                             M1_lo=min(s[1] for s in st), M1_hi=max(s[1] for s in st),
                             gain=np.median([s[2] for s in st])))
    t = pd.DataFrame(rows)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)
    print("wrote", OUT)
    rho = {}
    for col in ("M1", "M0", "gain"):
        r, p = spearmanr(t.source_pct, t[col])
        rho[col] = {"rho": float(r), "p": float(p), "n_folds": len(t)}
        print(f"Spearman(source share, {col}) over {len(t)} folds: {r:+.2f} (p={p:.2f})")
    Path(OUT_RHO).write_text(json.dumps(rho, indent=1))
    print("wrote", OUT_RHO)


if __name__ == "__main__":
    main()
