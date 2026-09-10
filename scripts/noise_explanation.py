"""Is daily supervision winning because hourly observations are noisier?

Daily-aggregate supervision beats hourly supervision on the target domain. Before that can
be attributed to anything about the objective, a simpler explanation has to be ruled out:
hourly discharge is usually derived from a rating curve and is noisier at the sub-daily
scale than its own 24-hour mean, so a daily target may simply carry a higher
signal-to-noise ratio while carrying less information. That is an observational-error
account, not a regularisation one, and the two make different predictions.

If the noise account is right, the advantage of daily over hourly supervision should be
LARGER at gauges whose hourly observations are noisier. This script tests that directly.
It needs no new training: both arms are already scored per gauge, and the observed
within-day characteristics were computed by the degeneracy check.

The proxy is the observed flashiness, the relative jump between neighbouring hours,
normalised by the gauge's own mean flow. Its limitation is stated rather than hidden: it
mixes genuine sub-daily hydrological response with rating-curve noise and cannot separate
them. A negative result therefore supports "this proxy detects no such relationship", not
"observational error is irrelevant". Separating the two would need rating-curve uncertainty,
which the dataset does not carry.

    python -m scripts.noise_explanation
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

# Each arm at its own best transfer learning rate, so the comparison is not decided by a
# rate tuned for one of them.
DAILY_ARM = "outputs/v2_lrsweep_daily_lr2e4"
HOURLY_ARM = "outputs/v2_lrsweep_upper_lr2e4"
SHAPE_CSV = "outputs/v2_runB/degenerate/intraday_shape.csv"


def per_gauge(root: str, column: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(
            f"{root}/fold*/transfer/per_station_hourly_*M1_target_hourly*.csv")):
        if "by_source" in path:
            continue
        frame = pd.read_csv(path)
        frame["fold"] = int(Path(path).parts[-3].replace("fold", ""))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[(out["obs_std"] >= 1e-3) & (out["score_status"] == "ok")]
    return out[["station_id", "fold", "kge"]].rename(columns={"kge": column})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="outputs/v2_noise_explanation", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.out_dir / "noise_explanation.log")

    shape_path = Path(SHAPE_CSV)
    if not shape_path.exists():
        raise SystemExit(f"{SHAPE_CSV} is missing. Run the degeneracy check first.")
    shape = pd.read_csv(shape_path)[
        ["station_id", "fold", "obs_flashiness", "obs_intraday_std", "obs_mean", "n_days"]]

    daily, hourly = per_gauge(DAILY_ARM, "daily"), per_gauge(HOURLY_ARM, "hourly")
    if daily.empty or hourly.empty:
        raise SystemExit("one of the two supervision arms has no per-gauge results")

    table = daily.merge(hourly, on=["station_id", "fold"]).merge(
        shape, on=["station_id", "fold"])
    table["advantage"] = table["daily"] - table["hourly"]
    # Normalised by the gauge's own mean flow, so the proxy is a relative jump rather than
    # an absolute one and is comparable between a mountain torrent and a lowland river.
    table["noise_proxy"] = table["obs_flashiness"] / table["obs_mean"].clip(lower=1e-6)
    table = table[np.isfinite(table["noise_proxy"])]

    rho = spearmanr(table["noise_proxy"], table["advantage"])
    payload = {
        "n": int(len(table)),
        "median_advantage": float(table["advantage"].median()),
        "spearman_rho": float(rho.statistic),
        "spearman_p": float(rho.pvalue),
        "quintiles": [],
    }

    logger.info("daily supervision against hourly supervision, each at its own best rate")
    logger.info("  n=%d, median advantage %+.4f", payload["n"], payload["median_advantage"])
    logger.info("  advantage against the observed-noise proxy: Spearman %+.4f (p=%.2e)",
                payload["spearman_rho"], payload["spearman_p"])
    logger.info("")
    logger.info("  by quintile of the noise proxy:")
    table["q"] = pd.qcut(table["noise_proxy"], 5, labels=False)
    for q, group in table.groupby("q"):
        row = {"quintile": int(q) + 1, "n": int(len(group)),
               "median_noise_proxy": float(group["noise_proxy"].median()),
               "median_advantage": float(group["advantage"].median())}
        payload["quintiles"].append(row)
        logger.info("    Q%d  proxy %8.4f  n=%5d  advantage %+.4f",
                    row["quintile"], row["median_noise_proxy"], row["n"],
                    row["median_advantage"])

    # The prediction under the noise account is a monotone rise, so the verdict turns on
    # monotonicity and not only on the correlation coefficient.
    advantages = [q["median_advantage"] for q in payload["quintiles"]]
    payload["monotone_increasing"] = bool(
        all(b >= a for a, b in zip(advantages, advantages[1:])))
    payload["top_quintile_below_overall"] = bool(
        advantages[-1] < payload["median_advantage"])
    logger.info("")
    if payload["spearman_p"] > 0.05 and not payload["monotone_increasing"]:
        logger.info(
            "The noise account predicts the advantage grows with observed noise. It does "
            "not: the correlation is not significant and the bands are not monotone, with "
            "the noisiest quintile at %+.4f against %+.4f overall. The account is not "
            "supported.",
            advantages[-1], payload["median_advantage"])
    else:
        logger.info("The relationship is present. The noise account cannot be dismissed "
                    "and the mechanism claim has to be qualified accordingly.")
    logger.info(
        "This proxy mixes genuine sub-daily response with rating-curve noise and cannot "
        "separate them, so the finding is that no such relationship is detectable, not "
        "that observational error is irrelevant.")

    table.drop(columns=["q"]).to_csv(args.out_dir / "per_gauge_noise.csv", index=False)
    (args.out_dir / "noise_explanation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out_dir / "noise_explanation.json")


if __name__ == "__main__":
    main()
