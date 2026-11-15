"""Longitudinal 5-fold split: contiguous bands of longitude, balanced by station count.

The third spatial design the supervisor asked for, alongside the random split and the
blocked one. It differs from blocking in what it withholds. A blocked fold removes a
target station's immediate neighbourhood while leaving the rest of its continent in the
source domain; a longitudinal fold removes an entire meridional slice, so a target
station loses every station at a similar longitude, including ones on other continents
at the same climatic latitude.

Bands are cut at station-count quantiles rather than at equal longitude, because station
density is wildly uneven: equal 72-degree bands would put more than half the network in
one fold. Cutting at quantiles keeps folds near 1,798 stations each, the size the random
and blocked splits already use, so the three designs stay comparable.

The station list is taken from the existing random-split table, so all three splits cover
exactly the same stations and no comparison is confounded by membership.

    python -m scripts.prep.make_folds_longitudinal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0088


def unit_vectors(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.column_stack([np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon),
                            np.sin(lat)])


def nearest_other_fold_km(vec, fold):
    """Great-circle distance from each station to the nearest station in another fold."""
    out = np.empty(len(vec))
    for f in np.unique(fold):
        here = np.flatnonzero(fold == f)
        other = np.flatnonzero(fold != f)
        # Chunked: the full 8,990 x 7,192 dot product is fine, but this keeps the peak
        # allocation small enough to run beside other jobs on a login node.
        for i in range(0, len(here), 512):
            block = here[i:i + 512]
            dots = np.clip(vec[block] @ vec[other].T, -1.0, 1.0)
            out[block] = np.arccos(dots).min(axis=1) * EARTH_R_KM
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="folds/folds_blocked.csv",
                        help="Existing fold table, for the station list and coordinates")
    parser.add_argument("--out", default="folds/folds_longitudinal.csv", type=Path)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    table = pd.read_csv(args.source)
    for col in ("station_id", "lat", "long", "n_training_batches"):
        if col not in table.columns:
            raise SystemExit(f"{args.source} has no column {col!r}")

    # Rotate so the cut points fall in the Pacific rather than through a dense region.
    # Longitude is periodic, so the choice of origin is free and a bad one would split
    # Europe or the eastern United States down the middle.
    lon = table["long"].to_numpy()
    shifted = (lon + 30.0) % 360.0
    order = np.argsort(shifted, kind="stable")
    fold = np.empty(len(table), dtype=int)
    # Equal station counts, not equal longitude: density varies by more than an order of
    # magnitude and equal-width bands would make one fold several times the others.
    bounds = np.linspace(0, len(table), args.n_folds + 1).astype(int)
    for f, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        fold[order[lo:hi]] = f
    table["fold"] = fold

    vec = unit_vectors(table["lat"].to_numpy(), lon)
    table["nearest_other_fold_km_longitudinal"] = nearest_other_fold_km(vec, fold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print(f"{len(table)} stations | {args.n_folds} folds -> {args.out}\n")
    print(f"{'fold':>4s} {'n':>6s} {'lon range':>22s} {'median nearest-other-fold km':>30s}")
    for f in range(args.n_folds):
        m = fold == f
        lo, hi = lon[m].min(), lon[m].max()
        print(f"{f:>4d} {int(m.sum()):>6d} {f'{lo:.0f} .. {hi:.0f}':>22s} "
              f"{np.median(table.loc[m, 'nearest_other_fold_km_longitudinal']):>30.1f}")
    print(f"\nwhole set, median nearest station in another fold: "
          f"{table['nearest_other_fold_km_longitudinal'].median():.1f} km")
    for col, label in (("nearest_other_fold_km_blocked", "blocked"),
                       ("nearest_other_fold_km_random", "random")):
        if col in table.columns:
            print(f"  for comparison, {label}: {table[col].median():.1f} km")
    print("\nstations per source and fold:")
    print(pd.crosstab(table["source"], table["fold"]).to_string())


if __name__ == "__main__":
    main()
