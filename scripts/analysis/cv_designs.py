"""The five cross-validation designs, side by side, with the folds kept visible.

The supervisor asked for a number of CV experiments -- random, continental,
longitudinal, sub-region -- and this is the table that answers him. What it shows is
a single ordering: the further a design puts a held-out gauge from anything the model
trained on, the less the zero-shot model is worth and the more the daily fine-tuning
recovers.

The per-fold breakdown is not decoration. On the continental design the pooled median
gain is +0.61, and reading only that would be a mistake: three continents move a
mediocre or useless model to KGE 0.53-0.64, while Australia moves a catastrophic one
(median KGE -7.46) to a merely useless one (-0.03). The pooled figure is inflated by
that fold's enormous gain, which is a failure dressed as a success. Any summary that
hides it is wrong, so every fold is reported.

    python -m scripts.analysis.cv_designs
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.utils import setup_logging

# design -> (output root, fold table, the column holding that design's separation)
DESIGNS = {
    # The random table predates the spatial designs and carries no coordinates, so its
    # separation column lives in the blocked table, which covers exactly the same gauges.
    "random": ("outputs/v2_runB", "folds/folds_blocked.csv",
               "nearest_other_fold_km_random"),
    "blocked": ("outputs/v2_blocked", "folds/folds_blocked.csv",
                "nearest_other_fold_km_blocked"),
    "climate": ("outputs/v2_climate", "folds/folds_climate.csv",
                "nearest_other_fold_km_climate"),
    "longitudinal": ("outputs/v2_longitudinal", "folds/folds_longitudinal.csv",
                     "nearest_other_fold_km_longitudinal"),
    "continental": ("outputs/v2_continental", "folds/folds_continental.csv",
                    "nearest_other_fold_km_continental"),
}
# What a fold means, where the design gives it a name worth printing.
FOLD_NAMES = {
    "continental": {0: "North America", 1: "Oceania", 2: "Europe", 3: "Asia"},
    "climate": {0: "A tropical", 1: "B arid", 2: "C temperate",
                3: "D continental", 4: "E polar"},
}


def per_station(root: str, fold: int, which: str) -> pd.Series | None:
    hits = [f for f in glob.glob(
        f"{root}/fold{fold}/transfer/per_station_hourly_*{which}_target_hourly*.csv")
        if "by_source" not in f]
    if not hits:
        return None
    frame = pd.read_csv(hits[0])
    frame = frame[(frame["obs_std"] >= 1e-3) & (frame["score_status"] == "ok")]
    return frame.set_index("station_id")["kge"]


def fold_table(root: str) -> list[int]:
    return sorted(int(Path(p).name.replace("fold", ""))
                  for p in glob.glob(f"{root}/fold*") if Path(p).is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="outputs/v2_cv_designs", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.out_dir / "cv_designs.log")
    from scipy.stats import wilcoxon

    payload = {"designs": {}}
    for name, (root, folds_path, column) in DESIGNS.items():
        folds = fold_table(root)
        separation = pd.read_csv(folds_path)
        entry = {"n_folds": len(folds), "root": root, "folds": {},
                 "separation_km": {
                     "median": float(separation[column].median()),
                     "p10": float(separation[column].quantile(0.10)),
                     "p90": float(separation[column].quantile(0.90))}}

        pooled = []
        for fold in folds:
            m0, m1 = per_station(root, fold, "M0"), per_station(root, fold, "M1")
            if m0 is None or m1 is None:
                raise SystemExit(f"{root} fold {fold} has no per-station results")
            pair = pd.DataFrame({"M0": m0, "M1": m1}).dropna()
            gain = pair["M1"] - pair["M0"]
            entry["folds"][str(fold)] = {
                "name": FOLD_NAMES.get(name, {}).get(fold),
                "n_gauges": int(len(pair)),
                "M0": float(pair["M0"].median()), "M1": float(pair["M1"].median()),
                "gain": float(gain.median()),
                "frac_improved": float((gain > 0).mean()),
            }
            pooled.append(pair)

        table = pd.concat(pooled)
        if table.index.duplicated().any():
            raise SystemExit(f"{name}: a gauge appears in more than one fold")
        gain = table["M1"] - table["M0"]
        entry["pooled"] = {
            "n_gauges": int(len(table)),
            "M0": float(table["M0"].median()), "M1": float(table["M1"].median()),
            # The median of the paired differences, not the difference of the medians:
            # the two are not equal for a skewed distribution and only the first is a
            # statement about individual gauges.
            "gain": float(gain.median()),
            "frac_improved": float((gain > 0).mean()),
            "wilcoxon_p": float(wilcoxon(table["M1"], table["M0"]).pvalue),
        }
        fold_gains = np.array([f["gain"] for f in entry["folds"].values()])
        entry["fold_spread"] = {
            "std": float(fold_gains.std(ddof=1)),
            "range": float(fold_gains.max() - fold_gains.min()),
        }
        payload["designs"][name] = entry

    order = sorted(payload["designs"], key=lambda k: payload["designs"][k]["separation_km"]["median"])
    payload["order_by_separation"] = order

    logger.info("%-14s %5s %8s %9s %9s %9s %7s", "design", "folds",
                "sep km", "M0", "M1", "gain", "impr")
    for name in order:
        e = payload["designs"][name]
        p = e["pooled"]
        logger.info("%-14s %5d %8.0f %+9.4f %+9.4f %+9.4f %6.0f%%", name,
                    e["n_folds"], e["separation_km"]["median"],
                    p["M0"], p["M1"], p["gain"], 100 * p["frac_improved"])

    for name in order:
        e = payload["designs"][name]
        if not any(f["name"] for f in e["folds"].values()) and name != "continental":
            continue
        logger.info("")
        logger.info("%s, fold by fold:", name)
        for key, f in e["folds"].items():
            label = f["name"] or f"fold {key}"
            logger.info("   %-16s %6d gauges  M0 %+8.4f -> M1 %+8.4f  gain %+8.4f  %3.0f%%",
                        label, f["n_gauges"], f["M0"], f["M1"], f["gain"],
                        100 * f["frac_improved"])

    # The two sentences the section turns on, derived here so the prose cannot drift.
    easiest, hardest = order[0], order[-1]
    e0, e1 = payload["designs"][easiest], payload["designs"][hardest]
    worst_fold = min(payload["designs"]["continental"]["folds"].values(),
                     key=lambda f: f["M1"])
    payload["verdict"] = (
        f"Separation from {e0['separation_km']['median']:.0f} km to "
        f"{e1['separation_km']['median']:,.0f} km takes the zero-shot median from "
        f"{e0['pooled']['M0']:+.4f} to {e1['pooled']['M0']:+.4f} while the fine-tuned "
        f"median only moves from {e0['pooled']['M1']:+.4f} to {e1['pooled']['M1']:+.4f}; "
        f"the paired gain therefore rises from {e0['pooled']['gain']:+.4f} to "
        f"{e1['pooled']['gain']:+.4f}. The exception is the {worst_fold['name']} fold, "
        f"where fine-tuning lifts a median KGE of {worst_fold['M0']:+.2f} only to "
        f"{worst_fold['M1']:+.4f}, which is still no better than predicting the mean."
    )
    logger.info("")
    logger.info("VERDICT: %s", payload["verdict"])

    out = args.out_dir / "cv_designs.json"
    out.write_text(json.dumps(payload, indent=1))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
