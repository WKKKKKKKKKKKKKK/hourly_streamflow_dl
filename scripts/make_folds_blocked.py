"""Spatially blocked 5-fold split -- the harder half of PLAN.md 2.

The random split answers "within the same region, can daily data replace hourly
data". It cannot answer the question the whole project is motivated by, because
CAMELSH alone contributes 5,767 US stations: under random assignment a target
station almost always has a hydrological neighbour left in the source domain, so
skill can come from spatial proximity rather than from generalising over basin
attributes. PLAN.md calls the difference between the two splits a headline result.

Why not simply k-means the globe into 5 clusters and call them folds: station
density is wildly uneven, so one fold would hold several thousand target stations
and another a few hundred, and nothing would be comparable with the random split's
~1,798 per fold. Instead cut the sphere into MANY small blocks, then pack whole
blocks into 5 folds balancing station count. A target station's near neighbours sit
in its own block, hence in its own fold, hence NOT in the source domain -- while
fold sizes stay even.

Clustering runs on 3-D unit vectors rather than (lat, lon) degrees, so blocks are
compact in real distance and nothing breaks at the dateline or near the poles.

The station list comes from the existing random-split table, so both splits cover
exactly the same stations and the comparison is not confounded by which stations
each one happened to keep.

    python -m scripts.make_folds_blocked --n-blocks 240
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
STATIC_CSV = (
    "/ibex/project/c2266/abbaa0a/data/gscad_database/processed/20250630/"
    "hourly/dataframes/static.csv"
)


def unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])


def great_circle_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise distances between two sets of unit vectors, (len(a), len(b)) km."""
    cos = np.clip(a @ b.T, -1.0, 1.0)
    return EARTH_RADIUS_KM * np.arccos(cos)


def pack_blocks(block_sizes: pd.Series, n_folds: int, seed: int) -> dict[int, int]:
    """Assign whole blocks to folds, largest first, always to the emptiest fold.

    Greedy longest-processing-time packing. Keeps fold station counts close without
    ever splitting a block, which is what preserves the spatial separation.
    """
    rng = np.random.default_rng(seed)
    order = block_sizes.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    order = order.sort_values(ascending=False, kind="stable")
    totals = np.zeros(n_folds, dtype=np.int64)
    assignment: dict[int, int] = {}
    for block, size in order.items():
        fold = int(np.argmin(totals))
        assignment[block] = fold
        totals[fold] += size
    return assignment


def nearest_other_fold_km(vectors: np.ndarray, folds: np.ndarray, chunk: int = 512) -> np.ndarray:
    """For each station, the great-circle distance to the nearest station NOT in its fold.

    This is the number that decides whether a blocked split earned its name: it is
    exactly "how far away is the closest station the model was allowed to train on".
    """
    out = np.empty(len(vectors), dtype=np.float64)
    for start in range(0, len(vectors), chunk):
        stop = min(start + chunk, len(vectors))
        d = great_circle_km(vectors[start:stop], vectors)
        different = folds[None, start:stop].T != folds[None, :]
        d = np.where(different, d, np.inf)
        out[start:stop] = d.min(axis=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatially blocked 5-fold station split.")
    parser.add_argument("--random-folds", default="folds/folds_random.csv")
    parser.add_argument("--out", default="folds/folds_blocked.csv")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=240,
        help="Spatial blocks before packing. More blocks -> more even folds but weaker "
             "separation; fewer -> stronger separation but lumpier folds.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--static", default=STATIC_CSV)
    args = parser.parse_args()

    base = pd.read_csv(args.random_folds)
    static = pd.read_csv(args.static, comment="#", index_col=0)
    coords = static.loc[:, ["lat", "long"]]

    missing = sorted(set(base["station_id"]) - set(coords.index))
    if missing:
        raise SystemExit(
            f"{len(missing)} stations in {args.random_folds} have no coordinates "
            f"(e.g. {missing[:3]}) -- cannot build a spatial split"
        )
    table = base.merge(coords, left_on="station_id", right_index=True, how="left")
    bad = table[["lat", "long"]].isna().any(axis=1)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} stations have NaN coordinates")

    vectors = unit_vectors(table["lat"].to_numpy(), table["long"].to_numpy())

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=args.n_blocks, n_init=4, random_state=args.seed)
    table["block"] = km.fit_predict(vectors)

    sizes = table.groupby("block").size()
    assignment = pack_blocks(sizes, args.n_folds, args.seed)
    table["fold"] = table["block"].map(assignment).astype(int)

    print(f"{len(table)} stations | {args.n_blocks} blocks | {args.n_folds} folds")
    print("\nfold sizes (target-domain station count per fold)")
    blocked_counts = table["fold"].value_counts().sort_index()
    random_counts = base["fold"].value_counts().sort_index()
    for fold in range(args.n_folds):
        print(f"  fold {fold}: blocked {blocked_counts.get(fold, 0):5d} | "
              f"random {random_counts.get(fold, 0):5d}")

    # The claim under test: a blocked target station's nearest trainable neighbour
    # is far away. Without this the split is "blocked" in name only.
    random_folds = base.set_index("station_id").loc[table["station_id"], "fold"].to_numpy()
    print("\ndistance to the nearest station in a DIFFERENT fold (km)")
    print(f"{'split':>10s} {'median':>9s} {'25%':>9s} {'75%':>9s} {'<10km':>8s} {'<50km':>8s}")
    for name, folds in (("random", random_folds), ("blocked", table["fold"].to_numpy())):
        d = nearest_other_fold_km(vectors, folds)
        print(f"{name:>10s} {np.median(d):9.1f} {np.percentile(d, 25):9.1f} "
              f"{np.percentile(d, 75):9.1f} {(d < 10).mean():8.1%} {(d < 50).mean():8.1%}")
        table[f"nearest_other_fold_km_{name}"] = d

    print("\nfold composition by source agency (blocked; uneven BY DESIGN)")
    comp = pd.crosstab(table["fold"], table["source"])
    print(comp.to_string())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = ["station_id", "n_training_batches", "n_validation_batches", "source",
            "size_bin", "fold", "block", "lat", "long",
            "nearest_other_fold_km_blocked", "nearest_other_fold_km_random"]
    table[keep].to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
