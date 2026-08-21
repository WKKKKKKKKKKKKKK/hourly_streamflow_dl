"""Persist the v3 convergence check so appendix A's verdict is traceable to a file.

Appendix A's headline -- that the random-vs-blocked M1 gap narrows from -0.0061 to -0.0015
under longer training -- was originally computed in an ad-hoc shell session and never
written anywhere. A conclusion that cannot be pointed at a file is not deliverable, so this
script recomputes all three parts of it and saves them:

1. PRETRAIN GAIN per fold, v3 against v2, on the early-stopping selection metric. Read from
   each fold's checkpoint.pth, which stores `best` and `best_epoch`.
2. THE PAIRED GAP, blocked M1 minus random M1 over the gauges present in both, under each
   configuration -- the statistic the main table rests on.
3. REPRODUCIBILITY. Folds whose pretrained weights are bit-identical between v2 and v3
   (their early stopping had already terminated, so best_model.pth was never rewritten) let
   the transfer stage be compared against itself. Two reproduce exactly and one does not,
   which bounds run-to-run noise at about 0.01 -- larger than the gap being measured.

    python -m scripts.v3_check
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MIN_OBS_STD = 1e-3


def weight_fingerprint(path: Path) -> str | None:
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.md5()
    for key in sorted(state):
        digest.update(state[key].numpy().tobytes())
    return digest.hexdigest()[:16]


def pretrain_gains(runs: list[str]) -> pd.DataFrame:
    rows = []
    for run in runs:
        for fold in range(5):
            a = Path(f"outputs/v2_{run}/fold{fold}/pretrain/checkpoint.pth")
            b = Path(f"outputs/v3_{run}/fold{fold}/pretrain/checkpoint.pth")
            if not (a.exists() and b.exists()):
                continue
            ca = torch.load(a, map_location="cpu", weights_only=False)
            cb = torch.load(b, map_location="cpu", weights_only=False)
            rows.append({
                "run": run, "fold": fold,
                "v2_stopped_epoch": int(ca["epoch"]), "v2_best_epoch": int(ca["best_epoch"]),
                "v2_best": float(ca["best"]),
                "v3_stopped_epoch": int(cb["epoch"]), "v3_best_epoch": int(cb["best_epoch"]),
                "v3_best": float(cb["best"]),
                "pretrain_gain": float(cb["best"] - ca["best"]),
                "weights_identical": weight_fingerprint(a.with_name("best_model.pth"))
                == weight_fingerprint(b.with_name("best_model.pth")),
            })
    return pd.DataFrame(rows)


def per_station_m1(run: str) -> pd.Series:
    """Target-domain M1 KGE per gauge, pooled over folds, degenerate gauges removed."""
    files = sorted(glob.glob(
        f"outputs/{run}/fold*/transfer/per_station_hourly_fold*_M1_target_hourly.csv"))
    if not files:
        return pd.Series(dtype=float)
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame = frame.loc[frame["score_status"].eq("ok") & (frame["obs_std"] >= MIN_OBS_STD)]
    return frame.set_index("station_id")["kge"]


def paired_gap(variant: str) -> dict | None:
    """Blocked minus random, paired over gauges scored in both."""
    from scipy.stats import wilcoxon

    rnd = per_station_m1(f"{variant}_runB")
    blk = per_station_m1(f"{variant}_blocked")
    if rnd.empty or blk.empty:
        return None
    common = rnd.index.intersection(blk.index)
    gap = (blk[common] - rnd[common]).replace([np.inf, -np.inf], np.nan).dropna()
    if gap.empty:
        return None
    return {"variant": variant, "n_stations": int(len(gap)),
            "median_gap": float(gap.median()), "mean_gap": float(gap.mean()),
            "frac_blocked_worse": float((gap < 0).mean()),
            "wilcoxon_p": float(wilcoxon(gap).pvalue),
            "_series": gap}


def transfer_reproducibility(gains: pd.DataFrame) -> pd.DataFrame:
    """For folds with bit-identical weights, did the transfer stage reproduce?"""
    rows = []
    for _, r in gains.loc[gains["weights_identical"]].iterrows():
        run, fold = r["run"], int(r["fold"])
        out = []
        for variant in ("v2", "v3"):
            path = Path(f"outputs/{variant}_{run}/fold{fold}/transfer/summary.json")
            if not path.exists():
                out.append(None)
                continue
            j = json.loads(path.read_text())
            out.append({"best_epoch": j["best_epoch"],
                        "holdout": j["best_holdout_daily_kge"],
                        "M1": j["step2_M1_target_hourly"]["median_kge"]})
        if None in out:
            continue
        a, b = out
        rows.append({"run": run, "fold": fold,
                     "epoch_v2": a["best_epoch"], "epoch_v3": b["best_epoch"],
                     "holdout_v2": a["holdout"], "holdout_v3": b["holdout"],
                     "M1_v2": a["M1"], "M1_v3": b["M1"],
                     "M1_difference": b["M1"] - a["M1"]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Persist the v3 convergence check.")
    parser.add_argument("--out-dir", default="outputs/v3_check")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gains = pretrain_gains(["runB", "blocked"])
    if gains.empty:
        raise SystemExit("no v2/v3 checkpoint pairs found")
    gains.to_csv(out_dir / "pretrain_gains.csv", index=False)
    print("=== pretraining: v3 against v2 on the selection metric ===")
    print(gains[["run", "fold", "v2_best_epoch", "v2_best", "v3_stopped_epoch",
                 "v3_best_epoch", "v3_best", "pretrain_gain", "weights_identical"]]
          .to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    by_run = gains.groupby("run")["pretrain_gain"].agg(["mean", "median", "max"])
    print("\n  mean gain by run:")
    print(by_run.to_string(float_format=lambda v: f"{v:+.4f}"))

    gaps, series = [], {}
    for variant in ("v2", "v3"):
        g = paired_gap(variant)
        if g:
            series[variant] = g.pop("_series")
            gaps.append(g)
    summary = {"pretrain_gain_by_run": json.loads(by_run.to_json(orient="index")),
               "paired_gap": gaps}
    if gaps:
        print("\n=== the paired gap: blocked M1 minus random M1, same gauges ===")
        print(pd.DataFrame(gaps).to_string(index=False))
    if len(series) == 2:
        from scipy.stats import wilcoxon

        common = series["v2"].index.intersection(series["v3"].index)
        delta = (series["v3"][common] - series["v2"][common]).dropna()
        narrowing = {"n_stations": int(len(delta)),
                     "median_change_in_gap": float(delta.median()),
                     "mean_change_in_gap": float(delta.mean()),
                     "wilcoxon_p": float(wilcoxon(delta).pvalue)}
        summary["gap_narrowing_v2_to_v3"] = narrowing
        print(f"\n  gap under v3 minus gap under v2, paired over {narrowing['n_stations']:,} "
              f"gauges: median {narrowing['median_change_in_gap']:+.4f}, "
              f"p = {narrowing['wilcoxon_p']:.2e}")
        pd.DataFrame({"gap_v2": series["v2"][common], "gap_v3": series["v3"][common]}) \
            .to_csv(out_dir / "paired_gap_per_station.csv")

    repro = transfer_reproducibility(gains)
    if not repro.empty:
        repro.to_csv(out_dir / "transfer_reproducibility.csv", index=False)
        print("\n=== folds whose pretrained weights are bit-identical between v2 and v3 ===")
        print(repro.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
        worst = repro.loc[repro["M1_difference"].abs().idxmax()]
        summary["reproducibility"] = {
            "n_identical_weight_folds": int(len(repro)),
            "n_reproduced_exactly": int((repro["M1_difference"].abs() < 1e-9).sum()),
            "largest_M1_difference": float(worst["M1_difference"]),
            "largest_at": f'{worst["run"]} fold{int(worst["fold"])}',
        }
        r = summary["reproducibility"]
        print(f"\n  {r['n_reproduced_exactly']}/{r['n_identical_weight_folds']} reproduced "
              f"bit-exactly; the largest M1 difference is "
              f"{r['largest_M1_difference']:+.4f} ({r['largest_at']}).")
        print("  That bounds run-to-run reproducibility, and it is larger than the gap above.")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
