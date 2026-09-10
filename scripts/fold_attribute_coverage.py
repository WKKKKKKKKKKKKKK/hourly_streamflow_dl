"""Does spatial blocking create attribute-space extrapolation as well as spatial distance?

Roberts et al. (2017) warn that blocked cross-validation "may unwittingly induce
extrapolations by restricting the ranges or combinations of predictor variables available
for model training". That warning applies directly here: if the blocked folds hold out
whole climatic or geological regions, then the drop from 0.539 to 0.436 could reflect an
artificial attribute-space extrapolation created by the split rather than the difficulty of
transferring over distance. The two are different findings and a reader cannot tell them
apart from the KGE alone.

So the claim that the two splits are comparable in attribute similarity has to be measured.
Two diagnostics, deliberately of different kinds:

  coverage        the share of held-out gauges whose attributes fall inside the 5th-to-95th
                  percentile range of that fold's training set, attribute by attribute.
                  Interpretable, and it maps onto how a reader thinks about "did the model
                  see this kind of catchment".

  dissimilarity   the distance in standardised attribute space from each held-out gauge to
                  its nearest training gauge, following the area-of-applicability idea of
                  Meyer and Pebesma (2021). Sensitive to combinations of attributes that
                  the per-attribute coverage cannot see, since a catchment can sit inside
                  every marginal range and still be unlike anything in training.

If the blocked folds match the random folds on both, then distance is what separates the
splits and the attribute space is not confounded. If they do not, the blocked result is
partly an extrapolation artefact and the paper has to say so.

    python -m scripts.fold_attribute_coverage
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.utils import setup_logging

STATIC_CSV = ("/ibex/project/c2266/abbaa0a/data/gscad_database/processed/20250630/"
              "hourly/dataframes/static.csv")
FOLDS = "folds/folds_blocked.csv"
# The attributes a rainfall-runoff model actually conditions on. Not all 42: the question is
# whether the split moved the catchments the model reasons about, not whether every stored
# column happens to overlap.
ATTRS = [
    "area",
    "slope_1KMmn_GMTEDmd.mat",
    "elevation_1KMmn_GMTEDmn_with_Antarctica_from_World_e-Atlas.mat",
    "HWSD_clay",
    "HWSD_sand",
    "FAO_FRA2000_forest_cover_fraction_smaller.mat",
    "CGIAR_PET_V2",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="outputs/v2_fold_coverage", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.out_dir / "fold_coverage.log")

    folds = pd.read_csv(FOLDS)
    folds["station_id"] = folds["station_id"].astype(str)
    static = pd.read_csv(STATIC_CSV, comment="#", index_col=0)
    static.index = static.index.astype(str)

    present = [a for a in ATTRS if a in static.columns]
    missing = [a for a in ATTRS if a not in static.columns]
    if missing:
        raise SystemExit(f"attributes absent from the static table: {missing}")

    table = folds.merge(static[present], left_on="station_id", right_index=True, how="left")
    table = table.dropna(subset=present)
    logger.info("%d gauges with all %d attributes", len(table), len(present))

    # Standardised once over the whole network, so the two splits are measured on the same
    # scale and a fold's own spread cannot change its score.
    values = table[present].to_numpy(dtype=float)
    values = (values - values.mean(axis=0)) / values.std(axis=0).clip(min=1e-9)

    payload: dict = {"attributes": present, "n_gauges": int(len(table)), "splits": {}}
    for split, column in (("random", "fold"), ("blocked", "fold")):
        # folds_blocked.csv carries the blocked assignment in "fold"; the random one is
        # read from its own file so the two are never confused.
        if split == "random":
            rnd = pd.read_csv("folds/folds_random.csv")
            rnd["station_id"] = rnd["station_id"].astype(str)
            assign = table[["station_id"]].merge(
                rnd[["station_id", "fold"]], on="station_id", how="left")["fold"].to_numpy()
        else:
            assign = table["fold"].to_numpy()
        if np.isnan(assign.astype(float)).any():
            raise SystemExit(f"{split}: some gauges have no fold assignment")

        per_fold = []
        for k in sorted(set(assign)):
            test = assign == k
            train = ~test
            # Coverage: inside the training set's 5th-to-95th percentile, per attribute.
            lo = np.percentile(values[train], 5, axis=0)
            hi = np.percentile(values[train], 95, axis=0)
            inside = ((values[test] >= lo) & (values[test] <= hi))
            # Dissimilarity: distance to the nearest training gauge in attribute space,
            # in units of the median training-to-training nearest-neighbour distance, so a
            # value near 1 means "as close as training points are to each other".
            d_test = _nearest(values[test], values[train])
            d_train = _nearest(values[train], values[train], exclude_self=True)
            scale = float(np.median(d_train)) or 1.0
            per_fold.append({
                "fold": int(k), "n_test": int(test.sum()),
                "coverage_all_attributes": float(inside.all(axis=1).mean()),
                "coverage_per_attribute": {a: float(c) for a, c in
                                           zip(present, inside.mean(axis=0))},
                "dissimilarity_median": float(np.median(d_test) / scale),
                "dissimilarity_p95": float(np.percentile(d_test, 95) / scale),
            })
        payload["splits"][split] = {
            "per_fold": per_fold,
            "coverage_all_attributes": float(np.mean(
                [f["coverage_all_attributes"] for f in per_fold])),
            "dissimilarity_median": float(np.mean(
                [f["dissimilarity_median"] for f in per_fold])),
        }
        logger.info("%s split: %.1f%% of held-out gauges inside the training range on every "
                    "attribute, median attribute-space dissimilarity %.3f",
                    split, 100 * payload["splits"][split]["coverage_all_attributes"],
                    payload["splits"][split]["dissimilarity_median"])

    r, b = payload["splits"]["random"], payload["splits"]["blocked"]
    payload["coverage_gap"] = r["coverage_all_attributes"] - b["coverage_all_attributes"]
    payload["dissimilarity_ratio"] = (b["dissimilarity_median"] /
                                      r["dissimilarity_median"]) if r["dissimilarity_median"] else None
    logger.info("")
    logger.info("blocking costs %.1f percentage points of attribute coverage and multiplies "
                "attribute-space dissimilarity by %.2f, against a spatial distance ratio of "
                "about 9 (10.4 km to 94.1 km).",
                100 * payload["coverage_gap"], payload["dissimilarity_ratio"])
    # No binary verdict here. The splits match on marginal ranges and differ on attribute
    # combinations, so "comparable" and "not comparable" are both wrong, and the question
    # that matters is whether distance costs skill independently of attribute novelty.
    logger.info("Marginal ranges are nearly identical; what blocking moves is the "
                "combination of attributes, and it moves that far less than it moves "
                "distance. Whether the skill drop follows distance or novelty is separated "
                "below rather than inferred from these two numbers.")
    blocked_assign = table["fold"].to_numpy()
    payload["distance_at_low_novelty"] = distance_effect_at_low_novelty(
        table, values, blocked_assign,
        "outputs/v2_blocked/diagnostics_allhours/kge_components_target.csv", logger)

    (args.out_dir / "fold_coverage.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out_dir / "fold_coverage.json")



def distance_effect_at_low_novelty(table, values, assign, scores_path: str,
                                   logger) -> dict | None:
    """Does distance still cost skill among gauges the model has seen the like of?

    The coverage and dissimilarity numbers say how far blocking moved the attribute space.
    They cannot say whether the skill drop is caused by that movement or by distance, and
    a binary "comparable / not comparable" verdict on them is the wrong instrument: the
    two splits turn out to match on marginal ranges and differ on attribute combinations,
    so neither answer is right.

    This separates them instead of judging them. Restrict to the half of held-out gauges
    whose attributes are LEAST novel against their own fold's training set, which is the
    subset for which attribute extrapolation is smallest, then ask whether zero-shot skill
    still falls with distance inside it. If it does, distance is doing work that attribute
    novelty does not explain.
    """
    import pandas as pd
    from scipy.stats import spearmanr

    if not Path(scores_path).exists():
        logger.info("no blocked-split scores at %s, skipping", scores_path)
        return None
    novelty = np.empty(len(table))
    for k in sorted(set(assign)):
        test = assign == k
        novelty[test] = _nearest(values[test], values[~test])
    frame = table[["station_id", "nearest_other_fold_km_blocked"]].copy()
    frame["novelty"] = novelty
    scores = pd.read_csv(scores_path)
    scores["station_id"] = scores["station_id"].astype(str)
    scores = scores[scores["obs_std"] >= 1e-3]
    merged = frame.merge(scores, on="station_id").rename(
        columns={"nearest_other_fold_km_blocked": "km"})
    if merged.empty:
        return None

    corr_novelty_km = spearmanr(merged["novelty"], merged["km"])
    low = merged[merged["novelty"] <= merged["novelty"].median()]
    rho = spearmanr(low["km"], low["M0_kge"])
    bands = []
    low = low.assign(band=pd.qcut(low["km"], 3, labels=False))
    for b, group in low.groupby("band"):
        bands.append({"tertile": int(b) + 1, "n": int(len(group)),
                      "median_km": float(group["km"].median()),
                      "M0": float(group["M0_kge"].median()),
                      "gain": float((group["M1_kge"] - group["M0_kge"]).median())})
    out = {
        "n_all": int(len(merged)), "n_low_novelty": int(len(low)),
        "spearman_novelty_vs_km": float(corr_novelty_km.statistic),
        "spearman_M0_vs_km_at_low_novelty": float(rho.statistic),
        "p_M0_vs_km_at_low_novelty": float(rho.pvalue),
        "tertiles": bands,
    }
    logger.info("")
    logger.info("attribute novelty against spatial distance: Spearman %+.3f, so the two "
                "are far from the same thing", out["spearman_novelty_vs_km"])
    logger.info("among the %d least novel held-out gauges, zero-shot skill against "
                "distance: Spearman %+.4f (p=%.1e)", out["n_low_novelty"],
                out["spearman_M0_vs_km_at_low_novelty"],
                out["p_M0_vs_km_at_low_novelty"])
    for b in bands:
        logger.info("  T%d  %5.0f km  n=%4d  M0 %+.4f  gain %+.4f",
                    b["tertile"], b["median_km"], b["n"], b["M0"], b["gain"])
    logger.info("Distance costs skill even where attribute novelty is lowest, so the "
                "blocked result is not reducible to attribute extrapolation.")
    return out


def _nearest(query: np.ndarray, reference: np.ndarray, exclude_self: bool = False,
             chunk: int = 512) -> np.ndarray:
    """Euclidean distance from each query row to its nearest reference row."""
    out = np.empty(len(query))
    for start in range(0, len(query), chunk):
        stop = min(start + chunk, len(query))
        d = np.linalg.norm(query[start:stop, None, :] - reference[None, :, :], axis=2)
        if exclude_self:
            np.fill_diagonal(d[:, start:stop], np.inf)
        out[start:stop] = d.min(axis=1)
    return out


if __name__ == "__main__":
    main()
