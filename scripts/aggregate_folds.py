"""Pool the 5 folds into one global result set (Plan.docx Phase I Step 4).

Every station is a target station in exactly one fold, so concatenating the
per-fold M0/M1 target metrics yields hourly KGE/NSE for the whole station set --
not just a lucky 20%. The paired M1-M0 change per station is the headline
number, tested with a Wilcoxon signed-rank test plus BH-FDR across sources.

    python -m scripts.aggregate_folds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common.config import add_common_args, load_config, resolve
from common.utils import setup_logging


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n, dtype=float)
    running_min = 1.0
    for rank, idx in enumerate(order[::-1]):
        rank_from_top = n - rank
        running_min = min(running_min, pvalues[idx] * n / rank_from_top)
        adjusted[idx] = running_min
    return np.clip(adjusted, 0.0, 1.0)


def _load(out_root: Path, fold: int, tag: str) -> pd.DataFrame | None:
    path = out_root / f"fold{fold}" / "transfer" / f"per_station_hourly_fold{fold}_{tag}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype={"station_id": str})
    frame["fold"] = fold
    return frame


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Pool per-fold Phase I results."))
    parser.add_argument("--out", default="outputs/phase1_summary")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    out_root = resolve(cfg.output_root)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "aggregate_folds.log")

    n_folds = int(cfg.folds.n_folds)
    m0_frames, m1_frames, missing = [], [], []
    for fold in range(n_folds):
        m0, m1 = _load(out_root, fold, "M0_target_hourly"), _load(out_root, fold, "M1_target_hourly")
        if m0 is None or m1 is None:
            missing.append(fold)
            continue
        m0_frames.append(m0)
        m1_frames.append(m1)

    if missing:
        logger.warning("folds with no results yet: %s -- the pooled numbers cover only %d/%d folds",
                       missing, n_folds - len(missing), n_folds)
    if not m1_frames:
        raise SystemExit("no fold results found; run train.transfer_target first")

    m0 = pd.concat(m0_frames, ignore_index=True)
    m1 = pd.concat(m1_frames, ignore_index=True)
    paired = m0[["station_id", "source", "fold", "samples", "kge", "nse", "score_status"]].merge(
        m1[["station_id", "kge", "nse", "score_status"]], on="station_id", suffixes=("_M0", "_M1")
    )
    paired["delta_kge"] = paired["kge_M1"] - paired["kge_M0"]
    paired["delta_nse"] = paired["nse_M1"] - paired["nse_M0"]
    both_ok = paired["score_status_M0"].eq("ok") & paired["score_status_M1"].eq("ok")
    paired["both_scored"] = both_ok
    paired.to_csv(out_dir / "per_station_M0_vs_M1.csv", index=False)

    duplicated = int(paired["station_id"].duplicated().sum())
    if duplicated:
        logger.warning("%d stations appear as a target in more than one fold -- check the fold table", duplicated)

    valid = paired.loc[both_ok]
    logger.info("pooled target stations: %d scored / %d total", len(valid), len(paired))

    # Wilcoxon on a handful of stations says nothing, so small groups are skipped.
    MIN_GROUP = 5
    COLUMNS = [
        "group", "n_stations", "median_kge_M0", "median_kge_M1", "median_delta_kge",
        "frac_improved", "median_nse_M0", "median_nse_M1", "median_delta_nse",
        "wilcoxon_stat", "p_value", "p_value_bh",
    ]
    rows, skipped_groups = [], []
    for name, group in [("ALL", valid)] + list(valid.groupby("source")):
        if len(group) < MIN_GROUP:
            skipped_groups.append((name, len(group)))
            continue
        stat, pvalue = stats.wilcoxon(group["kge_M1"], group["kge_M0"], zero_method="wilcox")
        rows.append({
            "group": name,
            "n_stations": int(len(group)),
            "median_kge_M0": float(group["kge_M0"].median()),
            "median_kge_M1": float(group["kge_M1"].median()),
            "median_delta_kge": float(group["delta_kge"].median()),
            "frac_improved": float((group["delta_kge"] > 0).mean()),
            "median_nse_M0": float(group["nse_M0"].median()),
            "median_nse_M1": float(group["nse_M1"].median()),
            "median_delta_nse": float(group["delta_nse"].median()),
            "wilcoxon_stat": float(stat),
            "p_value": float(pvalue),
        })
    if skipped_groups:
        logger.info("groups with fewer than %d scored stations, not tested: %s", MIN_GROUP, skipped_groups)

    # Declare the columns explicitly: an empty `rows` would otherwise give a
    # frame with no columns at all, and the BH step below would fail on lookup.
    stats_frame = pd.DataFrame(rows, columns=COLUMNS)
    if len(stats_frame):
        per_source = stats_frame["group"].ne("ALL")
        if per_source.any():
            stats_frame.loc[per_source, "p_value_bh"] = benjamini_hochberg(
                stats_frame.loc[per_source, "p_value"].to_numpy()
            )
        logger.info("M1 vs M0 (paired Wilcoxon):\n%s", stats_frame.to_string(index=False))
    else:
        logger.warning(
            "no group had %d+ scored stations, so no significance test was run. "
            "per_station_M0_vs_M1.csv still holds the paired values.", MIN_GROUP,
        )
    stats_frame.to_csv(out_dir / "significance_M1_vs_M0.csv", index=False)

    # Step 3: source-domain degradation, pooled across folds.
    degradation = []
    for fold in range(n_folds):
        path = out_root / f"fold{fold}" / "transfer" / f"source_degradation_fold{fold}.csv"
        if path.exists():
            frame = pd.read_csv(path, dtype={"station_id": str})
            frame["fold"] = fold
            degradation.append(frame)
    if degradation:
        degradation = pd.concat(degradation, ignore_index=True)
        degradation.to_csv(out_dir / "source_degradation_all_folds.csv", index=False)
        ok = degradation["score_status_before"].eq("ok") & degradation["score_status_after"].eq("ok")
        logger.info(
            "STEP 3 pooled source degradation: paired median delta KGE %+.4f over %d station-folds "
            "(%.1f%% got worse)",
            float(degradation.loc[ok, "delta_kge"].median()), int(ok.sum()),
            100.0 * float((degradation.loc[ok, "delta_kge"] < 0).mean()),
        )

    summary = {
        "folds_included": [f for f in range(n_folds) if f not in missing],
        "folds_missing": missing,
        "n_target_stations_pooled": int(len(paired)),
        "n_scored": int(both_ok.sum()),
        "median_kge_M0": float(valid["kge_M0"].median()),
        "median_kge_M1": float(valid["kge_M1"].median()),
        "median_delta_kge": float(valid["delta_kge"].median()),
        "median_nse_M0": float(valid["nse_M0"].median()),
        "median_nse_M1": float(valid["nse_M1"].median()),
    }
    with open(out_dir / "phase1_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
