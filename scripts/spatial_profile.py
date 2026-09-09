"""Skill as a function of distance to the nearest trainable gauge, and a buffered variant.

Two things the spatial cross-validation literature converged on, from opposite camps.

Brenning (2023, IJGIS) argues the range of autocorrelation is the WRONG criterion for
building spatial test sets, and that assessment should instead target the intended spatial
prediction horizon: report error as a function of prediction distance. He calls it a spatial
prediction error profile. Mila et al. (2022) and Linnenbrink et al. (2024) arrive at the
same quantity from the other side, matching the test-to-train nearest-neighbour distance
distribution to the one the deployment task faces. Ploton et al. (2020) got there first with
a buffered leave-one-out curve. Wadoux et al. (2021), who reject spatial cross-validation
outright, concede that standard cross-validation is deficient for strongly clustered data
with large differences in sampling density, which describes a network with 5,767 of 8,843
gauges in one country.

So the defensible object is not a single blocked-split scalar. It is the profile, plus a
statement of which part of it the deployment task occupies. That is what this script builds.

It also answers Karasiak et al. (2022), who point out that spatially blocked folds leave
residual dependence at block borders: "Objects belonging to contiguous blocks and located
close to borders may be very almost identical. Finding the ideal block size does not solve
the problem, but the distance-based buffer approach avoids it." Our blocked split has 60
gauges within 10 km of a trainable neighbour and 364 within 20 km, so the buffered variant
re-scores with those removed. No retraining: dropping gauges from the evaluation set changes
which rows are scored, not the model.

    python -m scripts.spatial_profile
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.utils import setup_logging

SPLITS = {
    "random": ("outputs/v2_runB", "nearest_other_fold_km_random"),
    "blocked": ("outputs/v2_blocked", "nearest_other_fold_km_blocked"),
}
FOLDS = "folds/folds_blocked.csv"
# Distance bands rather than quantiles, so the two splits are compared on the same x axis.
# Quantile bands would place the random split's Q5 at 32 km and the blocked split's at
# 180 km and invite exactly the comparison the profile exists to avoid.
EDGES = [0, 5, 10, 20, 40, 80, 160, np.inf]
BUFFER_KM = 20.0


def per_gauge(root: str) -> pd.DataFrame:
    """Paired M0 and M1 per gauge, pooled over folds."""
    path = Path(root) / "diagnostics_allhours" / "kge_components_target.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    table = table[table["obs_std"] >= 1e-3]
    return table[["station_id", "M0_kge", "M1_kge"]].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="outputs/v2_spatial_profile", type=Path)
    parser.add_argument("--buffer-km", type=float, default=BUFFER_KM)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.out_dir / "spatial_profile.log")

    folds = pd.read_csv(FOLDS)
    folds["station_id"] = folds["station_id"].astype(str)

    payload: dict = {"buffer_km": args.buffer_km, "distance_edges": [float(e) for e in EDGES],
                     "splits": {}}
    rows_out = []

    for name, (root, distance_column) in SPLITS.items():
        scores = per_gauge(root)
        if scores.empty:
            logger.info("%s: no diagnostics table, skipping", name)
            continue
        scores["station_id"] = scores["station_id"].astype(str)
        merged = scores.merge(folds[["station_id", distance_column]], on="station_id")
        merged = merged.rename(columns={distance_column: "km"}).dropna(subset=["km"])
        merged["gain"] = merged["M1_kge"] - merged["M0_kge"]

        entry: dict = {
            "n": int(len(merged)),
            "median_km": float(merged["km"].median()),
            "min_km": float(merged["km"].min()),
            "frac_within_10km": float((merged["km"] < 10).mean()),
            "frac_within_buffer": float((merged["km"] < args.buffer_km).mean()),
        }

        # --- the profile ---------------------------------------------------
        # This is Brenning's SPEP: skill against prediction distance, not a single number.
        bands = pd.cut(merged["km"], EDGES, right=False)
        profile = []
        for band, group in merged.groupby(bands, observed=True):
            if len(group) < 20:
                continue
            profile.append({
                "band": str(band), "lo_km": float(band.left), "hi_km": float(band.right),
                "n": int(len(group)),
                "M0": float(group["M0_kge"].median()),
                "M1": float(group["M1_kge"].median()),
                "gain": float(group["gain"].median()),
            })
        entry["profile"] = profile

        # Correlation of zero-shot skill with distance, WITHIN this split. The training set,
        # the hyperparameters and the fold count are identical across the bands, so distance
        # is the only thing that varies and the blocked split's harder regions cannot
        # explain it.
        rho = spearmanr(merged["km"], merged["M0_kge"])
        entry["spearman_M0_vs_km"] = {"rho": float(rho.statistic), "p": float(rho.pvalue)}
        rho_gain = spearmanr(merged["km"], merged["gain"])
        entry["spearman_gain_vs_km"] = {"rho": float(rho_gain.statistic),
                                        "p": float(rho_gain.pvalue)}

        # --- the buffered variant ------------------------------------------
        kept = merged[merged["km"] >= args.buffer_km]
        entry["buffered"] = {
            "n": int(len(kept)), "n_removed": int(len(merged) - len(kept)),
            "M0": float(kept["M0_kge"].median()), "M1": float(kept["M1_kge"].median()),
            "gain": float(kept["gain"].median()),
            "frac_improved": float((kept["gain"] > 0).mean()),
        }
        entry["unbuffered"] = {
            "M0": float(merged["M0_kge"].median()), "M1": float(merged["M1_kge"].median()),
            "gain": float(merged["gain"].median()),
            "frac_improved": float((merged["gain"] > 0).mean()),
        }
        payload["splits"][name] = entry

        merged["split"] = name
        rows_out.append(merged[["station_id", "split", "km", "M0_kge", "M1_kge", "gain"]])

        logger.info("%s split: n=%d, median %.1f km, min %.2f km, %.1f%% within %.0f km",
                    name, entry["n"], entry["median_km"], entry["min_km"],
                    100 * entry["frac_within_buffer"], args.buffer_km)
        logger.info("  profile, skill against distance to the nearest trainable gauge:")
        logger.info("    %-14s %6s %8s %8s %8s", "band (km)", "n", "M0", "M1", "gain")
        for p in profile:
            logger.info("    %-14s %6d %8.4f %8.4f %+8.4f",
                        f"{p['lo_km']:.0f} to {p['hi_km']:.0f}", p["n"], p["M0"], p["M1"],
                        p["gain"])
        logger.info("  Spearman M0 against distance: rho %+.4f (p %.1e)",
                    entry["spearman_M0_vs_km"]["rho"], entry["spearman_M0_vs_km"]["p"])
        logger.info("  buffered at %.0f km: %d gauges removed, M0 %.4f -> M1 %.4f, "
                    "gain %+.4f (unbuffered %+.4f)",
                    args.buffer_km, entry["buffered"]["n_removed"],
                    entry["buffered"]["M0"], entry["buffered"]["M1"],
                    entry["buffered"]["gain"], entry["unbuffered"]["gain"])
        logger.info("")

    if rows_out:
        pd.concat(rows_out, ignore_index=True).to_csv(
            args.out_dir / "per_gauge_distance.csv", index=False)
    (args.out_dir / "spatial_profile.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out_dir / "spatial_profile.json")


if __name__ == "__main__":
    main()
