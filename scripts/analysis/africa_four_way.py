"""Four models on the same African catchments, the same split, the same forcing.

The supervisor asked for one thing plainly: show that the hourly model's daily
performance is at least as good as a typical LSTM's. Answering it needs a comparison in
which only the thing being tested varies, and the obvious candidate fails that test. A
global daily LSTM trained elsewhere by the archive maintainer scores these basins under
a ten-fold ungauged-basin protocol, over its own evaluation window, on a different
precipitation product, having seen roughly 235 other African basins during training. Set
against a model fine-tuned on these basins' own early record it would be four
confounds deep, and the direction of the error changes with which confound dominates.

So the baselines are trained here instead, inside the same framework:

    M0            the pretrained hourly model, zero-shot; no African basin in training
    M1            that model, fine-tuned on African daily observations
    daily_lstm    the same architecture's daily branch, from scratch, scored on its own
                  head: an ordinary daily LSTM given exactly this study's data
    scratch_agg   the same architecture and the same aggregate objective, from scratch:
                  what the method achieves without a pretrained model

Everything else is held: 294 catchments, the same per-basin 70/30 temporal split, the
same held-out slice for choosing the epoch, ERA5-Land forcing, 25 static attributes, and
an equal epoch budget. An unequal budget already invalidated one version of this table,
where M1 had twelve epochs against the baselines' eighty, so the budget is checked here
rather than assumed.

    python -m scripts.analysis.africa_four_way
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.utils import setup_logging

ARMS = {
    "M1": "outputs/v2_africa_m1b_fold*",
    "daily_lstm": "outputs/v2_africa_daily_lstm_fold*",
    "scratch_agg": "outputs/v2_africa_scratch_agg_fold*",
}
M0_FROM = "outputs/v2_africa_m1b_fold*"


def per_basin(pattern: str, name: str) -> pd.Series:
    """Median KGE per basin across the five folds.

    Median rather than the ensemble mean of predictions: the folds differ only in which
    pretrained model they started from, and a per-basin median over five fitted models is
    the statistic all four arms can supply. The published ensemble figure is computed
    elsewhere, from averaged predictions, and is not interchangeable with this.
    """
    files = sorted(glob.glob(f"{pattern}/{name}"))
    if len(files) != 5:
        raise SystemExit(f"{pattern}/{name}: found {len(files)} folds, need 5")
    wide = pd.concat(
        [pd.read_csv(f).set_index("station_id")["kge"].rename(f"fold{i}")
         for i, f in enumerate(files)], axis=1)
    return wide.median(axis=1)


def budgets(pattern: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(f"{pattern}/training_history.csv")):
        h = pd.read_csv(f)
        out.append({"epochs_run": int(len(h)),
                    "best_epoch": int(h.loc[h["median_kge"].idxmax(), "epoch"])})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="outputs/al_v2_africa_five_way", type=Path)
    parser.add_argument("--prefix", default="al_v2",
                        help="run-name prefix of the arms, e.g. al_v2 for the aligned re-run")
    parser.add_argument("--three-way", default=Path("outputs/al_v2_africa_three_way/daily_three_way_per_basin.csv"), type=Path,
                        help="daily_three_way_per_basin.csv; adds ERA5-Land as a fifth arm. "
                             "Its basin-days must equal the arms' own (checked)")
    args = parser.parse_args()
    arms = {a: p.replace("outputs/v2_", f"outputs/{args.prefix}_") for a, p in ARMS.items()}
    m0_from = M0_FROM.replace("outputs/v2_", f"outputs/{args.prefix}_")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.out_dir / "africa_four_way.log")
    from scipy.stats import wilcoxon

    series = {"M0": per_basin(m0_from, "per_basin_M0.csv")}
    for arm, pattern in arms.items():
        series[arm] = per_basin(pattern, "per_basin_M1.csv")
    if args.three_way:
        # ERA5-Land is one deterministic product, so it has no members to take a median over;
        # it enters only if it was scored on exactly the days the arms were.
        tw = pd.read_csv(args.three_way).set_index("station_id")
        days = pd.read_csv(sorted(glob.glob(f"{arms['M1']}/per_basin_M1.csv"))[0]).set_index("station_id")["n_days"]
        both = days.index.intersection(tw.index)
        if not (tw.loc[both, "n_days_era5_land"] == days.loc[both]).all():
            raise SystemExit("ERA5-Land was scored on different basin-days than the arms")
        series["era5_land"] = tw["kge_era5_land"]
    table = pd.DataFrame(series).dropna()

    # The budget check that the first version of this table failed.
    budget = {arm: budgets(pattern) for arm, pattern in arms.items()}
    caps = {arm: max(b["epochs_run"] for b in rows) for arm, rows in budget.items()}
    at_cap = {arm: sum(b["best_epoch"] == b["epochs_run"] for b in rows)
              for arm, rows in budget.items()}
    logger.info("epoch budget per arm (a truncated arm cannot be compared):")
    for arm in arms:
        logger.info("   %-12s ran up to %2d epochs, best at cap in %d of 5 folds",
                    arm, caps[arm], at_cap[arm])
    if len(set(caps.values())) > 1:
        logger.info("   WARNING: the arms did not share a budget")

    payload = {"n_basins": int(len(table)), "budget": budget,
               "epoch_caps": caps, "folds_best_at_cap": at_cap, "medians": {}, "paired": {}}
    logger.info("")
    logger.info("median daily KGE over %d African catchments:", len(table))
    for arm in [a for a in ("M0", "daily_lstm", "scratch_agg", "M1", "era5_land") if a in table]:
        payload["medians"][arm] = float(table[arm].median())
        logger.info("   %-12s %+.4f", arm, table[arm].median())

    logger.info("")
    for a, b in (("M1", "daily_lstm"), ("M1", "scratch_agg"), ("scratch_agg", "daily_lstm"),
                 ("M1", "M0")) + ((("M1", "era5_land"), ("M0", "era5_land"), ("daily_lstm", "era5_land"))
                                   if "era5_land" in table else ()):
        delta = table[a] - table[b]
        payload["paired"][f"{a}_vs_{b}"] = {
            "median": float(delta.median()),
            "frac_a_wins": float((delta > 0).mean()),
            "wilcoxon_p": float(wilcoxon(table[a], table[b]).pvalue),
        }
        e = payload["paired"][f"{a}_vs_{b}"]
        logger.info("   %-12s - %-12s %+.4f   %s wins %.0f%%   p=%.2e",
                    a, b, e["median"], a, 100 * e["frac_a_wins"], e["wilcoxon_p"])

    # The two readings the supervisor's question turns on, stated from the numbers.
    v_lstm = payload["paired"]["M1_vs_daily_lstm"]
    v_pre = payload["paired"]["M1_vs_scratch_agg"]
    payload["verdict"] = (
        f"against an ordinary daily LSTM given the same data, the fine-tuned hourly model "
        f"is {v_lstm['median']:+.4f} in median daily KGE "
        f"(p = {v_lstm['wilcoxon_p']:.1e}); against the same objective without a pretrained "
        f"model it is {v_pre['median']:+.4f} (p = {v_pre['wilcoxon_p']:.1e}), which is what "
        f"the pretraining is worth here."
    )
    logger.info("")
    logger.info("VERDICT: %s", payload["verdict"])
    table.to_csv(args.out_dir / "per_basin_four_way.csv")
    (args.out_dir / "africa_four_way.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
