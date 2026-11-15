"""Dissimilarity of every target gauge to its fold's source domain, with its M0, M1 and gain.

Dissimilarity index after Meyer and Pebesma (2021, Methods Ecol. Evol. 12, 1620-1633), without
variable weighting: for each target gauge, the Euclidean distance in the model's standardised
static attributes to the nearest source gauge of its fold, divided by the mean distance between
source gauges of that fold (estimated from 20,000 random pairs, seed 0). Two versions:

  DI_clim   the four climate attributes (mean precipitation, mean temperature, PET, snowfall
            fraction)
  DI_all    all continuous static attributes; coordinates and categorical attributes
            (Koeppen zones, land-cover, lake/wetland and lithology classes) are excluded

M0, M1 and gain are per-gauge medians over the three seeds of scripts.figures.m1_vs_m0_figures.SEED_RUNS.
One row per gauge and design (each gauge is a target once per design).

    python -m scripts.analysis.dissimilarity_index      # -> reports/m1_vs_m0/dissimilarity_per_gauge.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from common.config import load_config
from data.dataset import load_scalers
from data.hourly_windows import build_static_matrix, load_cache

CONFIG = "configs/aligned/phase1_runB_blocked_v2.yaml"   # any aligned config: same data and statics
OUT = "reports/m1_vs_m0/dissimilarity_per_gauge.csv"
CLIMATE = ["WorldClim_V21_Pmean", "WorldClim_V21_Tmean", "CGIAR_PET_V2", "WorldClim_V2_snowfall_fraction_P.mat"]
NOT_CONTINUOUS = ("long", "lat", "KGZ", "GlobCover", "GLWD", "UMD_1km", "GLiM")
DESIGN_FOLDS = ("random", "blocked", "climate", "longitudinal", "continental")


def statics(data_root: str | None = None) -> pd.DataFrame:
    """Standardised static attributes of every gauge, scaled with the dataset's static scalers.

    data_root overrides cfg.data.root (where scalers.json lives); only the static part of the
    scalers is used here."""
    cfg = load_config(CONFIG, [f"data.root={data_root}"] if data_root else [])
    cache = load_cache(cfg.data.cache_dir, stride=24)
    s, names = build_static_matrix(cache, load_scalers(cfg.data.root), cfg)
    return pd.DataFrame(np.asarray(s), index=list(cache["stations"]), columns=names)


def index(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    dmin, _ = cKDTree(src).query(tgt, k=1)
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(src), 20000), rng.integers(0, len(src), 20000)
    return dmin / np.mean(np.linalg.norm(src[i] - src[j], axis=1))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=None,
                    help="directory holding scalers.json, if not the config's data.root")
    args = ap.parse_args()
    import scripts.figures.m1_vs_m0_figures as m

    s = statics(args.data_root)
    cont = [c for c in s.columns if not any(k in c for k in NOT_CONTINUOUS)]
    rows = []
    for d in DESIGN_FOLDS:
        fs = pd.read_csv(f"folds/folds_{d}.csv").set_index("station_id").fold
        gain = m.seed_gains(d)
        lev = [m.paired(r) for r in m.SEED_RUNS[d]]
        m0 = pd.concat([t.M0 for t in lev], axis=1).median(axis=1)
        m1 = pd.concat([t.M1 for t in lev], axis=1).median(axis=1)
        for k in sorted(fs.unique()):
            src = [x for x in fs.index[fs != k] if x in s.index]
            tgt = [x for x in fs.index[fs == k] if x in s.index]
            di = {tag: pd.Series(index(s.loc[src, cols].to_numpy(float), s.loc[tgt, cols].to_numpy(float)), index=tgt)
                  for tag, cols in (("DI_clim", CLIMATE), ("DI_all", cont))}
            df = (pd.DataFrame(di).join(gain.median(axis=1).rename("gain")).join(m0.rename("M0"))
                  .join(m1.rename("M1")).dropna())
            df["design"], df["fold"] = d, k
            rows.append(df)
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(OUT)
    print("wrote", OUT, f"({len(cont)} continuous attributes)")


if __name__ == "__main__":
    main()
