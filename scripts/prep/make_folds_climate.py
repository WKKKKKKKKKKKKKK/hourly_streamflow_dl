"""Leave-one-climate-zone-out split, filling the supervisor's fourth slot.

He asked for a "sub-region" design without defining it. A Koeppen major zone is the
reading that earns its place, because it also answers a separate remark he made in the
same conversation: he doubted the value of training one model per climate zone, arguing
that a single model plus climate-related predictors should be equivalent. This split is
the test of that claim. If one global model reaches a zone it never saw, the per-zone
models he was sceptical of are unnecessary; if it cannot, they are not.

Withholding a zone is not the same as withholding geography. Fold 2 removes every
temperate gauge on every continent at once, which no spatial design here does, so a
target catchment keeps its nearby neighbours and loses its climatic analogues instead.

The zones are very unequal (6,109 down to 75) and are left that way. The small folds are
kept rather than merged because zone A is the closest in-network analogue to the African
transfer this study also reports, and a noisy answer there is worth more than none. The
per-fold table prints n so the noise stays visible.

    python -m scripts.prep.make_folds_climate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.africa import HOURLY_STATIC, read_static_table
from scripts.prep.make_folds_longitudinal import nearest_other_fold_km, unit_vectors

# Declared, so a fold index means the same thing on every re-run.
ORDER = ["A", "B", "C", "D", "E"]
NAMES = {"A": "tropical", "B": "arid", "C": "temperate",
         "D": "continental", "E": "polar"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="folds/folds_blocked.csv")
    parser.add_argument("--static", default=str(HOURLY_STATIC))
    parser.add_argument("--out", default="folds/folds_climate.csv", type=Path)
    args = parser.parse_args()

    table = pd.read_csv(args.source)
    static = read_static_table(args.static)
    table["kgz_major"] = table["station_id"].map(static["KGZ_major"])

    missing = int(table["kgz_major"].isna().sum())
    if missing:
        raise SystemExit(f"{missing} stations have no KGZ_major in {args.static}")
    unknown = sorted(set(table["kgz_major"]) - set(ORDER))
    if unknown:
        raise SystemExit(f"zone(s) {unknown} are not in ORDER")

    index = {z: i for i, z in enumerate(ORDER)}
    table["fold"] = table["kgz_major"].map(index).astype(int)

    vec = unit_vectors(table["lat"].to_numpy(), table["long"].to_numpy())
    table["nearest_other_fold_km_climate"] = nearest_other_fold_km(
        vec, table["fold"].to_numpy())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print(f"{len(table)} stations | {len(ORDER)} folds -> {args.out}\n")
    print(f"{'fold':>4s} {'zone':>5s} {'name':>13s} {'test':>7s} {'train':>7s} "
          f"{'median nearest-other-fold km':>30s}")
    for f, z in enumerate(ORDER):
        m = table["fold"] == f
        print(f"{f:>4d} {z:>5s} {NAMES[z]:>13s} {int(m.sum()):>7d} {int((~m).sum()):>7d} "
              f"{table.loc[m, 'nearest_other_fold_km_climate'].median():>30.1f}")
    print(f"\nwhole set, median nearest station in another fold: "
          f"{table['nearest_other_fold_km_climate'].median():.1f} km")
    print("\nstations per source and fold:")
    print(pd.crosstab(table["source"], table["kgz_major"]).to_string())


if __name__ == "__main__":
    main()
