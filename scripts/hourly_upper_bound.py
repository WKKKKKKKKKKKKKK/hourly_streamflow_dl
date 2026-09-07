"""How much of an hourly-supervised transfer does daily-only supervision recover?

Phase I hides the target stations' hourly observations and fine-tunes on 24 h aggregates
alone. That answers whether daily data helps. It does not answer the question a reader asks
next: how much is given up by not having hourly data at all. Without that number the method
has no position -- "daily data helps" could mean daily is nearly as good as hourly, or that
it captures a small fraction of what hourly would give. Both are publishable and they are
different papers.

Three arms, all starting from the SAME pretrained weights so the comparison is of the
transfer step and not of two pretrainings:

  M0            zero-shot, no target data used at all
  M1_daily      Phase I proper: daily aggregates only          (outputs/v2_runB)
  M1_obj        the objective alone switched to hourly targets  (outputs/v2_hourly_obj)
  M1_upper      objective, selection and the frozen hourly branch all switched, which is
                the configuration a practitioner with hourly gauges would actually use
                                                                (outputs/v2_hourly_upper)

Arm M1_obj differs from M1_daily in exactly one config key, and this script verifies that
rather than trusting it. Arm M1_upper differs in three, and those three are consequences of
a single premise rather than three independent choices: with hourly targets you train on
them, you select on them, and you stop freezing the hourly branch that was frozen only
because daily aggregates cannot inform it. The cost of the selection change on its own is
measured separately, from the per-epoch hourly scores the daily runs already log without
using them.

Every comparison is paired per gauge and reported as a median of per-gauge differences, not
as a difference of medians.

    python -m scripts.hourly_upper_bound
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common.utils import setup_logging

ARMS = (
    ("M1_daily", "outputs/v2_runB", "configs/phase1_runB_v2.yaml"),
    ("M1_obj", "outputs/v2_hourly_obj", "configs/phase1_runB_hourly_obj_v2.yaml"),
    ("M1_upper", "outputs/v2_hourly_upper", "configs/phase1_runB_hourly_upper_v2.yaml"),
)
BASE_CONFIG = "configs/phase1_runB_v2.yaml"


def flatten(node, prefix=""):
    out = {}
    for key, value in (node or {}).items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


def config_delta(path: str) -> dict:
    """Which keys this arm changes against the Phase I config, output_root aside."""
    base = flatten(yaml.safe_load(Path(BASE_CONFIG).read_text()))
    arm = flatten(yaml.safe_load(Path(path).read_text()))
    delta = {k: (base.get(k, "<absent>"), arm.get(k, "<absent>"))
             for k in set(base) | set(arm) if base.get(k) != arm.get(k)}
    delta.pop("output_root", None)
    return delta


def per_gauge(run: Path, which: str) -> pd.DataFrame:
    """Per-gauge hourly KGE for one arm, pooled over folds.

    Reads the per-fold result files the transfer step writes, rather than a diagnostics
    directory, so an arm needs no separate diagnostics pass before it can be read here.
    """
    frames = []
    for fold_dir in sorted(run.glob("fold*/transfer")):
        # per_station_*, explicitly. A looser glob also matches by_source_*, which holds
        # one row per agency, and reading that instead yields 6 rows per fold and a
        # perfectly plausible-looking median. Nothing in the arithmetic downstream would
        # have flagged it, so the prefix is pinned here and the row count is checked below.
        hits = sorted(fold_dir.glob(f"per_station_hourly_*{which}_target_hourly*.csv"))
        if not hits:
            continue
        frame = pd.read_csv(hits[0])
        frame["fold"] = int(fold_dir.parent.name.replace("fold", ""))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if len(out) < 100 * len(frames):
        raise SystemExit(
            f"{run}/{which}: {len(out)} rows over {len(frames)} folds is far too few for a "
            "per-gauge table. An aggregated file was almost certainly read instead."
        )
    id_col = "station_id" if "station_id" in out.columns else out.columns[0]
    return out.rename(columns={id_col: "station_id"})[["station_id", "fold", "kge", "nse"]]


def selection_cost(run: Path) -> dict:
    """What daily-based epoch selection costs, from logs the daily runs already keep.

    Each daily run records the hourly test score every epoch without feeding it to the
    early stopper. Comparing the epoch daily selection chose against the epoch the hourly
    score would have chosen isolates SELECTION from the objective, at no extra compute.
    """
    rows = []
    for history in sorted(run.glob("fold*/transfer/training_history.csv")):
        table = pd.read_csv(history)
        if "peek/target_hourly_median_kge" not in table:
            continue
        chosen = table.loc[table["holdout/daily_median_kge"].idxmax()]
        best = table.loc[table["peek/target_hourly_median_kge"].idxmax()]
        rows.append({
            "fold": int(history.parts[-3].replace("fold", "")),
            "epoch_daily_picked": int(chosen["epoch"]),
            "hourly_there": float(chosen["peek/target_hourly_median_kge"]),
            "epoch_hourly_best": int(best["epoch"]),
            "hourly_best": float(best["peek/target_hourly_median_kge"]),
        })
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    frame["cost"] = frame["hourly_there"] - frame["hourly_best"]
    return {
        "n_folds": len(frame),
        "median_cost_kge": float(frame["cost"].median()),
        "worst_cost_kge": float(frame["cost"].min()),
        "same_epoch_folds": int((frame["epoch_daily_picked"] == frame["epoch_hourly_best"]).sum()),
        "per_fold": frame.to_dict("records"),
    }



# transfer.lr was never in the search grid: every one of the 20-odd search configs carries
# 5e-4, and the ones named lr1e4 or lr5e3 vary the PRETRAIN rate. Sweeping it for the hourly
# arms alone would tilt the comparison the other way, so all three arms get the same values
# and each is credited with its own best.
SWEEP = {
    "M1_daily": {5e-4: "outputs/v2_runB",
                 2e-4: "outputs/v2_lrsweep_daily_lr2e4",
                 1e-4: "outputs/v2_lrsweep_daily_lr1e4"},
    "M1_obj": {5e-4: "outputs/v2_hourly_obj",
               2e-4: "outputs/v2_lrsweep_obj_lr2e4",
               1e-4: "outputs/v2_lrsweep_obj_lr1e4"},
    "M1_upper": {5e-4: "outputs/v2_hourly_upper",
                 2e-4: "outputs/v2_lrsweep_upper_lr2e4",
                 1e-4: "outputs/v2_lrsweep_upper_lr1e4"},
}


def sweep_effect(zero_shot: pd.DataFrame, logger) -> dict:
    """Each arm's paired gain at each learning rate, and the best each arm reaches.

    The point of the sweep is fairness rather than tuning. A result where the arm with LESS
    information wins invites the obvious objection that the better-informed arm was
    misconfigured, and the objection is only answered by giving every arm the same freedom.
    """
    out = {}
    for arm, roots in SWEEP.items():
        per_lr = {}
        for lr, root in sorted(roots.items()):
            frame = per_gauge(Path(root), "M1")
            if frame.empty:
                continue
            merged = zero_shot.merge(frame, on=["station_id", "fold"],
                                     suffixes=("_m0", "_m1"))
            delta = merged["kge_m1"] - merged["kge_m0"]
            per_lr[f"{lr:.0e}"] = {
                "median_M1": float(frame["kge"].median()),
                "paired_gain": float(delta.median()),
                "frac_improved": float((delta > 0).mean()),
                "n": int(len(merged)),
            }
        if not per_lr:
            continue
        best_lr = max(per_lr, key=lambda k: per_lr[k]["paired_gain"])
        out[arm] = {"per_lr": per_lr, "best_lr": best_lr,
                    "best_gain": per_lr[best_lr]["paired_gain"]}
        logger.info("  %-9s best at lr %s, gain %+.4f  (%s)", arm, best_lr,
                    per_lr[best_lr]["paired_gain"],
                    ", ".join(f"{k}: {v['paired_gain']:+.4f}" for k, v in per_lr.items()))
    if {"M1_daily", "M1_obj", "M1_upper"} <= set(out):
        daily = out["M1_daily"]["best_gain"]
        hourly = max(out["M1_obj"]["best_gain"], out["M1_upper"]["best_gain"])
        out["daily_margin_at_best"] = daily - hourly
        logger.info(
            "  each arm at its own best: daily %+.4f against the better hourly arm's "
            "%+.4f, a margin of %+.4f. The ordering does not come from a rate tuned for "
            "the daily objective.", daily, hourly, daily - hourly)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="outputs/v2_hourly_bound", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.out_dir / "hourly_upper_bound.log")

    # Every arm's config delta, checked rather than assumed. Arm M1_obj claims to be a
    # single-variable change and this is where that claim is tested.
    for name, _, config in ARMS:
        if name == "M1_daily":
            continue
        delta = config_delta(config)
        logger.info("%s changes %d key(s) against Phase I: %s", name, len(delta),
                    ", ".join(f"{k} {v[0]} -> {v[1]}" for k, v in sorted(delta.items())))
    obj_delta = config_delta(dict((n, c) for n, _, c in ARMS)["M1_obj"])
    if len(obj_delta) != 1:
        raise SystemExit(
            f"M1_obj is meant to change exactly one key and changes {len(obj_delta)}: "
            f"{sorted(obj_delta)}. Fix the config before reading this comparison as "
            "single-variable."
        )

    zero_shot = per_gauge(Path("outputs/v2_runB"), "M0")
    if zero_shot.empty:
        raise SystemExit("no M0 per-gauge files under outputs/v2_runB")

    arms = {}
    for name, root, _ in ARMS:
        frame = per_gauge(Path(root), "M1")
        if frame.empty:
            logger.info("%s has no results yet under %s, skipping", name, root)
            continue
        arms[name] = frame
        logger.info("%s: %d gauge-folds, median hourly KGE %.4f",
                    name, len(frame), frame["kge"].median())

    if "M1_daily" not in arms:
        raise SystemExit("the Phase I arm is missing, nothing to compare against")

    logger.info("")
    logger.info("gain over the zero-shot model, paired per gauge:")
    gains = {}
    for name, frame in arms.items():
        merged = zero_shot.merge(frame, on=["station_id", "fold"], suffixes=("_m0", "_m1"))
        delta = merged["kge_m1"] - merged["kge_m0"]
        gains[name] = float(delta.median())
        logger.info("  %-9s n=%5d  M0 %.4f -> %.4f  gain %+.4f  improved %.1f%%",
                    name, len(merged), merged["kge_m0"].median(),
                    merged["kge_m1"].median(), delta.median(), 100 * (delta > 0).mean())

    logger.info("")
    for name in ("M1_obj", "M1_upper"):
        if name not in arms:
            continue
        # The headline of this script: the share of an hourly-supervised gain that
        # daily-only supervision already achieves. Computed from the two paired gains,
        # both measured against the same zero-shot baseline on the same gauges.
        paired = arms["M1_daily"].merge(arms[name], on=["station_id", "fold"],
                                        suffixes=("_daily", "_other"))
        head_to_head = paired["kge_daily"] - paired["kge_other"]
        # Reported as a head-to-head difference, not as a percentage of the hourly gain.
        # A share only reads sensibly while the hourly arm is the larger of the two, and it
        # is not: "daily recovers 172% of the hourly gain" is arithmetic noise dressed as a
        # finding. The paired difference says the same thing without the false ceiling.
        share = gains["M1_daily"] / gains[name] if gains[name] else float("nan")
        if gains[name] >= gains["M1_daily"]:
            logger.info(
                "daily-only reaches %.1f%% of the %s gain (%+.4f of %+.4f). Head to head "
                "on %d gauges: median %+.4f, daily ahead at %.1f%%.",
                100 * share, name, gains["M1_daily"], gains[name], len(paired),
                head_to_head.median(), 100 * (head_to_head > 0).mean(),
            )
        else:
            logger.info(
                "daily-only BEATS %s: gain %+.4f against %+.4f. Head to head on %d gauges: "
                "median %+.4f, daily ahead at %.1f%%.",
                name, gains["M1_daily"], gains[name], len(paired),
                head_to_head.median(), 100 * (head_to_head > 0).mean(),
            )

    cost = selection_cost(Path("outputs/v2_runB"))
    if cost:
        logger.info("")
        logger.info(
            "of which epoch SELECTION accounts for at most %+.4f (median over %d folds, "
            "worst %+.4f, %d folds picked the same epoch either way). Measured without "
            "extra compute, from the hourly scores the daily runs log per epoch.",
            cost["median_cost_kge"], cost["n_folds"], cost["worst_cost_kge"],
            cost["same_epoch_folds"],
        )

    logger.info("")
    logger.info("learning-rate sweep, all three arms, paired against the same zero-shot:")
    sweep = sweep_effect(zero_shot, logger)

    payload = {"paired_gain_over_M0": gains, "selection_cost": cost, "lr_sweep": sweep,
               "config_deltas": {n: {k: list(v) for k, v in config_delta(c).items()}
                                 for n, _, c in ARMS if n != "M1_daily"}}
    (args.out_dir / "hourly_upper_bound.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out_dir / "hourly_upper_bound.json")


if __name__ == "__main__":
    main()
