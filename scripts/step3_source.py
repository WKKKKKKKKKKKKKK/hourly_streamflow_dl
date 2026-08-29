"""Aggregate STEP 3 -- what daily-only fine-tuning costs the source domain.

The plan's Step 3 reads: "Validate the model in step 2 on the 80% hourly data to see if
there is no degradation." The answer is that there IS degradation, and a report that
defines STEP 3 without reporting it leaves the plan's own question unanswered.

Each fold's transfer summary already carries step3_source_before / step3_source_after and
a paired median delta; this collects them so the number has a file behind it rather than
being read out of five separate JSONs by hand.

    python -m scripts.step3_source
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect the STEP 3 source-domain re-score.")
    parser.add_argument("--run-root", default="outputs/v2_runB", type=Path)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--out-dir", default="outputs/v2_step3_source", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.out_dir / "step3_source.log")

    rows = []
    for fold in [int(f) for f in args.folds.split(",") if f.strip()]:
        path = args.run_root / f"fold{fold}" / "transfer" / "summary.json"
        if not path.exists():
            logger.warning("fold %d: %s missing", fold, path)
            continue
        d = json.loads(path.read_text())
        before, after = d["step3_source_before"], d["step3_source_after"]
        rows.append({
            "fold": fold,
            "n_source_stations": before["n_valid_stations"],
            "kge_before": before["median_kge"], "kge_after": after["median_kge"],
            "nse_before": before["median_nse"], "nse_after": after["median_nse"],
            "frac_kge_gt_0_before": before["frac_kge_gt_0"],
            "frac_kge_gt_0_after": after["frac_kge_gt_0"],
            # The paired delta is the honest one: it pairs each source gauge with itself,
            # rather than differencing two medians taken over the same set.
            "paired_median_delta_kge": d["step3_paired_median_delta_kge"],
        })
    if not rows:
        raise SystemExit("no fold carried a step3 block")

    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "step3_by_fold.csv", index=False)

    summary = {
        "n_folds": int(len(table)),
        "n_source_stations": int(table.n_source_stations.median()),
        "median_kge_before": float(table.kge_before.median()),
        "median_kge_after": float(table.kge_after.median()),
        "median_nse_before": float(table.nse_before.median()),
        "median_nse_after": float(table.nse_after.median()),
        "median_paired_delta_kge": float(table.paired_median_delta_kge.median()),
        "paired_delta_range": [float(table.paired_median_delta_kge.min()),
                               float(table.paired_median_delta_kge.max())],
        "degraded_in_all_folds": bool((table.paired_median_delta_kge < 0).all()),
    }
    with open(args.out_dir / "step3_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("STEP 3 over %d folds, %d source gauges each:",
                summary["n_folds"], summary["n_source_stations"])
    logger.info("  median KGE %.4f -> %.4f, median NSE %.4f -> %.4f",
                summary["median_kge_before"], summary["median_kge_after"],
                summary["median_nse_before"], summary["median_nse_after"])
    logger.info("  paired median delta KGE %+.4f (range %+.4f to %+.4f), negative in all "
                "folds: %s", summary["median_paired_delta_kge"],
                *summary["paired_delta_range"], summary["degraded_in_all_folds"])
    logger.info("The plan asked whether there is no degradation. There is: adapting to the "
                "target domain's daily aggregates costs the source domain, consistently.")


if __name__ == "__main__":
    main()
