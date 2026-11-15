"""Leave-one-continent-out split: the hardest spatial holdout this network allows.

The fourth design the supervisor named, and the only one of his four that was still
missing. It differs from the longitudinal split in that a fold is not a slice of
geometry but a landmass: holding out North America removes every CAMELS-H gauge at
once, leaving the model to reach 5,115 catchments it has no neighbour for, on any
meridian, in any climate zone that continent alone supplies.

The folds are deliberately unequal. Station counts follow the network, not the design,
and forcing them to be equal would mean splitting a continent, which is the one thing
this split exists to avoid. Fold 0 therefore trains on 3,875 stations and tests on
5,115, the reverse of every other split here; that asymmetry is the experiment, not a
flaw in it, and the per-fold table below reports it rather than hiding it in an average.

The station list is taken from the existing fold table, so all four splits cover exactly
the same stations and no comparison is confounded by membership.

    python -m scripts.prep.make_folds_continental
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prep.make_folds_longitudinal import nearest_other_fold_km, unit_vectors

# Declared, not derived: a fold index has to mean the same thing on every re-run, and
# sorting by station count would silently renumber the folds if the archive changed.
CONTINENT = {
    "CAMELSH": "North America",
    "BOMAustralia": "Oceania",
    "LamaHCE": "Europe",
    "Germany": "Europe",
    "LamaHIce": "Europe",
    "CzechRepublic": "Europe",
    "Japan": "Asia",
}
ORDER = ["North America", "Oceania", "Europe", "Asia"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="folds/folds_blocked.csv")
    parser.add_argument("--out", default="folds/folds_continental.csv", type=Path)
    args = parser.parse_args()

    table = pd.read_csv(args.source)
    for col in ("station_id", "lat", "long", "source"):
        if col not in table.columns:
            raise SystemExit(f"{args.source} has no column {col!r}")

    table["continent"] = table["source"].map(CONTINENT)
    unmapped = sorted(table.loc[table["continent"].isna(), "source"].unique())
    if unmapped:
        raise SystemExit(f"no continent for source(s) {unmapped}; add them to CONTINENT")

    index = {name: i for i, name in enumerate(ORDER)}
    missing = sorted(set(table["continent"]) - set(index))
    if missing:
        raise SystemExit(f"{missing} is mapped to a continent that ORDER does not list")
    table["fold"] = table["continent"].map(index).astype(int)

    vec = unit_vectors(table["lat"].to_numpy(), table["long"].to_numpy())
    table["nearest_other_fold_km_continental"] = nearest_other_fold_km(
        vec, table["fold"].to_numpy())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print(f"{len(table)} stations | {len(ORDER)} folds -> {args.out}\n")
    print(f"{'fold':>4s} {'continent':>15s} {'test':>7s} {'train':>7s} "
          f"{'median nearest-other-fold km':>30s}")
    for f, name in enumerate(ORDER):
        m = table["fold"] == f
        print(f"{f:>4d} {name:>15s} {int(m.sum()):>7d} {int((~m).sum()):>7d} "
              f"{table.loc[m, 'nearest_other_fold_km_continental'].median():>30.1f}")
    print(f"\nwhole set, median nearest station in another fold: "
          f"{table['nearest_other_fold_km_continental'].median():.1f} km")
    for col, label in (("nearest_other_fold_km_longitudinal", "longitudinal"),
                       ("nearest_other_fold_km_blocked", "blocked"),
                       ("nearest_other_fold_km_random", "random")):
        if col in table.columns:
            print(f"  for comparison, {label}: {table[col].median():.1f} km")
    print("\nstations per source and fold:")
    print(pd.crosstab(table["source"], table["fold"]).to_string())


if __name__ == "__main__":
    main()
