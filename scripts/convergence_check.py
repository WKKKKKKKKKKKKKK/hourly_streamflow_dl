"""Was training long enough, and does the daily-only signal pick the right epoch?

Two questions answerable from the histories already on disk, no GPU needed.

1. PRETRAIN TRUNCATION. `train.epochs` is a cap, but `patience` is what actually stops
   most folds. If the two splits are truncated unequally the headline random-vs-blocked
   gap could be an artefact of that rather than a cost of spatial blocking, so this
   reports each fold's stopping epoch, its best epoch, and the slope of the selection
   metric over the last epochs -- a fold cut early while still improving had headroom.

2. SELECTION LOSS. The premise is that the target domain has no hourly observations, so
   the transfer stage selects its epoch on `holdout/daily_median_kge`. The history also
   records `peek/target_hourly_median_kge`, the hidden hourly truth, for diagnosis only.
   The difference between the peek value at the SELECTED epoch and at the peek-optimal
   epoch is the price of the premise. It is compared against the peek series' own
   epoch-to-epoch variation, because an argmax over a noisy series overstates the gap:
   a "loss" below that floor is not measurable.

    python -m scripts.convergence_check --runs v2_runB,v2_blocked,v2_replay025
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEL_PRETRAIN = "val/median_kge"
SEL_TRANSFER = "holdout/daily_median_kge"
PEEK = "peek/target_hourly_median_kge"


def pretrain_table(runs: list[str], tail: int) -> pd.DataFrame:
    rows = []
    for run in runs:
        for path in sorted(glob.glob(f"outputs/{run}/fold*/pretrain/training_history.csv")):
            hist = pd.read_csv(path)
            if SEL_PRETRAIN not in hist.columns:
                continue
            v = hist[SEL_PRETRAIN].to_numpy(float)
            k = min(tail, len(v))
            rows.append({
                "run": run, "fold": Path(path).parts[2], "epochs": len(v),
                "best_epoch": int(np.nanargmax(v)) + 1, "best_val": float(np.nanmax(v)),
                # positive slope at the end == still improving when it stopped
                "tail_slope": float(np.polyfit(np.arange(k), v[-k:], 1)[0]),
                "epochs_since_best": len(v) - (int(np.nanargmax(v)) + 1),
                "noise": float(np.median(np.abs(np.diff(v)))) if len(v) > 1 else np.nan,
            })
    return pd.DataFrame(rows)


def selection_table(runs: list[str]) -> tuple[pd.DataFrame, float]:
    rows, noise = [], []
    for run in runs:
        for path in sorted(glob.glob(f"outputs/{run}/fold*/transfer/training_history.csv")):
            hist = pd.read_csv(path)
            if PEEK not in hist.columns or SEL_TRANSFER not in hist.columns:
                continue
            peek = hist[PEEK].to_numpy(float)
            sel = int(np.nanargmax(hist[SEL_TRANSFER].to_numpy(float)))
            opt = int(np.nanargmax(peek))
            if len(peek) > 1:
                noise.append(float(np.median(np.abs(np.diff(peek)))))
            rows.append({
                "run": run, "fold": Path(path).parts[2], "epochs": len(peek),
                "selected_epoch": sel + 1, "optimal_epoch": opt + 1,
                "peek_at_selected": float(peek[sel]), "peek_at_optimal": float(peek[opt]),
                "selection_loss": float(peek[opt] - peek[sel]),
            })
    return pd.DataFrame(rows), float(np.median(noise)) if noise else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convergence and selection-loss check.")
    parser.add_argument("--runs", default="v2_runB,v2_blocked,v2_replay025")
    parser.add_argument("--tail", type=int, default=10, help="Epochs for the trend fit.")
    parser.add_argument("--out-dir", default="outputs/convergence_check")
    args = parser.parse_args()

    runs = [r for r in args.runs.split(",") if r.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = pretrain_table(runs, args.tail)
    if pre.empty:
        raise SystemExit(f"no pretrain histories under outputs/{{{args.runs}}}")
    print(f"=== pretrain: what actually stopped each fold (trend over last {args.tail}) ===")
    print(pre.to_string(index=False, float_format=lambda v: f"{v: .5f}"))
    rising = int((pre["tail_slope"] > 0).sum())
    print(f"\n  still improving at the stop: {rising}/{len(pre)} folds "
          f"(median slope {pre['tail_slope'].median():+.5f}/epoch)")
    print(f"  per-epoch trend vs oscillation: {pre['tail_slope'].median():.5f} vs "
          f"{pre['noise'].median():.5f} -- no single stop proves a plateau")
    # Unequal truncation is the thing that would contaminate a cross-split comparison.
    trunc = pre.groupby("run").agg(min_epochs=("epochs", "min"),
                                   max_slope=("tail_slope", "max"),
                                   median_slope=("tail_slope", "median"))
    print("\n  truncation by run (a run cut early WITH a steep slope had headroom):")
    print(trunc.to_string(float_format=lambda v: f"{v: .5f}"))
    pre.to_csv(out_dir / "pretrain_truncation.csv", index=False)

    sel, floor = selection_table(runs)
    summary = {"pretrain_folds_still_improving": rising,
               "pretrain_median_tail_slope": float(pre["tail_slope"].median()),
               "pretrain_truncation_by_run": json.loads(trunc.to_json(orient="index"))}
    if not sel.empty:
        print("\n=== transfer: cost of selecting on daily aggregates vs an hourly oracle ===")
        print(sel.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
        exact = int((sel["selected_epoch"] == sel["optimal_epoch"]).sum())
        mean_loss = float(sel["selection_loss"].mean())
        print(f"\n  by run:")
        for run, group in sel.groupby("run"):
            print(f"    {run:16s} n={len(group)} median {group['selection_loss'].median():+.4f} "
                  f"mean {group['selection_loss'].mean():+.4f} max {group['selection_loss'].max():+.4f}")
        print(f"\n  daily-holdout picks the oracle's epoch in {exact}/{len(sel)} folds")
        print(f"  pooled mean loss {mean_loss:+.4f} vs peek noise floor {floor:.4f} -> "
              f"{'NOT measurable above noise' if mean_loss < floor else 'exceeds the noise floor'}")
        # Only worth a test if a per-run difference is being claimed; n is small either way.
        if sel["run"].nunique() > 1:
            from scipy.stats import kruskal

            groups = [g["selection_loss"].to_numpy() for _, g in sel.groupby("run")]
            p = float(kruskal(*groups).pvalue)
            print(f"  per-run differences: Kruskal-Wallis p={p:.3f} "
                  f"({'not significant -- do not read them as real' if p > 0.05 else 'significant'})")
            summary["selection_loss_between_runs_p"] = p
        sel.to_csv(out_dir / "selection_loss.csv", index=False)
        summary.update({"selection_loss_mean": mean_loss,
                        "selection_loss_median": float(sel["selection_loss"].median()),
                        "peek_noise_floor": floor,
                        "n_folds_exact_match": exact, "n_folds": int(len(sel))})

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
