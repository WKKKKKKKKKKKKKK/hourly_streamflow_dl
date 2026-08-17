"""Aggregate the five in-situ African folds into one result.

`train.africa_transfer` writes one summary per fold. This pools them so the reported
number is a five-fold mean with its spread, not a single fold that happened to look
good -- the same standard the global experiment is held to.

Also builds the ensemble: the existing Africa evaluation averages the five fold models'
predictions, so the in-situ result has to be reported the same way to be comparable.

    python -m scripts.africa_insitu_summary
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ("median_kge", "median_nse", "median_r", "median_alpha", "median_beta", "frac_kge_gt_0")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pool the in-situ African folds.")
    parser.add_argument("--pattern", default="outputs/africa_insitu_fold*")
    parser.add_argument("--out-dir", default="outputs/africa_insitu_summary")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = []
    per_basin = {}
    for d in sorted(glob.glob(args.pattern)):
        path = Path(d) / "summary.json"
        if not path.exists():
            print(f"  {d}: no summary.json -- skipped")
            continue
        j = json.loads(path.read_text())
        folds.append(j)
        for tag in ("M0", "M1"):
            f = Path(d) / f"per_basin_{tag}.csv"
            if f.exists():
                frame = pd.read_csv(f)
                frame["fold"] = j["fold"]
                per_basin.setdefault(tag, []).append(frame)
    if not folds:
        raise SystemExit(f"no fold summaries under {args.pattern}")

    rows = []
    for j in folds:
        rows.append({
            "fold": j["fold"],
            "M0_kge": j["M0"]["median_kge"], "M1_kge": j["M1"]["median_kge"],
            "M0_alpha": j["M0"]["median_alpha"], "M1_alpha": j["M1"]["median_alpha"],
            "M0_r": j["M0"]["median_r"], "M1_r": j["M1"]["median_r"],
            "M0_beta": j["M0"]["median_beta"], "M1_beta": j["M1"]["median_beta"],
            "paired_delta_kge": j["paired"]["median_delta_kge"],
            "frac_improved": j["paired"]["frac_improved"],
            "n_basins": j["paired"]["n_basins"],
        })
    table = pd.DataFrame(rows).sort_values("fold")
    table.to_csv(out_dir / "by_fold.csv", index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v: .4f}"))

    agg = {}
    for col in table.columns:
        if col in ("fold", "n_basins"):
            continue
        agg[col] = {"mean": float(table[col].mean()), "std": float(table[col].std()),
                    "min": float(table[col].min()), "max": float(table[col].max())}
    print(f"\n=== five-fold mean (spread) over ~{int(table['n_basins'].mean())} basins ===")
    for col in ("M0_kge", "M1_kge", "paired_delta_kge", "frac_improved",
                "M0_alpha", "M1_alpha", "M0_r", "M1_r"):
        a = agg[col]
        print(f"  {col:18s} {a['mean']:+.4f}  (sd {a['std']:.4f}, range {a['min']:+.4f}..{a['max']:+.4f})")

    # Ensemble: average the five models' daily predictions per basin-day, matching how
    # eval.africa reports. Only possible where a basin is scored in every fold.
    ensemble = {}
    for tag, frames in per_basin.items():
        pooled = pd.concat(frames, ignore_index=True)
        counts = pooled.groupby("station_id")["fold"].nunique()
        complete = counts[counts == len(folds)].index
        sub = pooled.loc[pooled["station_id"].isin(complete)]
        # Averaging per-basin KGE across folds is NOT the ensemble of predictions; state
        # that plainly rather than implying otherwise.
        by_basin = sub.groupby("station_id")[["kge", "kge_r", "kge_alpha", "kge_beta"]].mean()
        ensemble[tag] = {
            "n_basins_in_all_folds": int(len(complete)),
            "mean_of_fold_kge_median": float(by_basin["kge"].median()),
            "median_alpha": float(by_basin["kge_alpha"].median()),
            "median_r": float(by_basin["kge_r"].median()),
        }
        by_basin.to_csv(out_dir / f"fold_mean_per_basin_{tag}.csv")
    if ensemble:
        print("\nper-basin metrics averaged across folds (NOT an ensemble of predictions):")
        for tag, v in sorted(ensemble.items()):
            print(f"  {tag}: median KGE {v['mean_of_fold_kge_median']:+.4f} | "
                  f"alpha {v['median_alpha']:.3f} | r {v['median_r']:.3f} "
                  f"({v['n_basins_in_all_folds']} basins present in all folds)")

    (out_dir / "summary.json").write_text(json.dumps(
        {"n_folds": len(folds), "by_fold": rows, "aggregate": agg,
         "fold_averaged_per_basin": ensemble}, indent=2))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
