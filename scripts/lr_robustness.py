"""Does the result depend on the transfer learning rate? Every arm, at every rate run.

transfer.lr was never in the hyperparameter search: all twenty-odd search configurations
carry 5e-4, and the ones named for a rate vary the PRETRAINING rate instead. The sweep that
exposed this was run for a different purpose, to answer the objection that the hourly-
supervision arms might have been misconfigured, and it found every arm peaks at 2e-4.

That raised the obvious follow-up, which is whether the paper should move to 2e-4. It should
not, and this script is why. Comparing the two rates across every arm shows the choice trades
one number against two: the random-split gain rises by 0.005 at 2e-4, while the ratio between
the blocked and random gains falls from 2.19 to 1.95 and the gap between the two fine-tuned
endpoints widens from 0.007 to 0.019. Those last two ARE the central claim of the paper, that
the gain grows with the difficulty of the test while the endpoint barely moves.

The reason is not subtle. 2e-4 is optimal for the random-split target arm, which is the only
arm the sweep selected on. Applying it everywhere reports every arm at a rate tuned for one
of them, which is harder to defend than a single rate applied uniformly, and it happens to
weaken both headline comparisons.

So the re-run becomes a robustness check rather than a configuration change, and a more
useful one than a tuned number would have been: it shows the conclusion does not come from
the choice of rate.

    python -m scripts.lr_robustness
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.utils import setup_logging

# M0 is identical across rates by construction, since the zero-shot model is the pretrained
# checkpoint and the rate only affects the transfer stage. It is read from the 5e-4 runs and
# paired against both.
ARMS = {
    "random": {
        "m0": "outputs/v2_runB",
        "5e-4": "outputs/v2_runB",
        "2e-4": "outputs/v2_lrsweep_daily_lr2e4",
    },
    "blocked": {
        "m0": "outputs/v2_blocked",
        "5e-4": "outputs/v2_blocked",
        "2e-4": "outputs/v2lr2_blocked",
    },
}
AFRICA = {
    "5e-4": "outputs/v2_africa_insitu_summary/ensemble_summary.json",
    "2e-4": "outputs/v2lr2_africa_insitu_summary/ensemble_summary.json",
}
REPLAY = {
    "5e-4": "outputs/v2_replay_effect/replay_effect.json",
    "2e-4": "outputs/v2lr2_replay_effect/replay_effect.json",
}
RATES = ("5e-4", "2e-4")


def per_gauge(root: str, which: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(
            f"{root}/fold*/transfer/per_station_hourly_*{which}_target_hourly*.csv")):
        if "by_source" in path:
            continue
        frame = pd.read_csv(path)
        frame["fold"] = int(Path(path).parts[-3].replace("fold", ""))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[(out["obs_std"] >= 1e-3) & (out["score_status"] == "ok")]
    return out[["station_id", "fold", "kge"]]


def paired(m0_root: str, m1_root: str) -> dict | None:
    m0, m1 = per_gauge(m0_root, "M0"), per_gauge(m1_root, "M1")
    if m0.empty or m1.empty:
        return None
    merged = m0.merge(m1, on=["station_id", "fold"], suffixes=("_0", "_1"))
    delta = merged["kge_1"] - merged["kge_0"]
    # groupby on the difference itself rather than apply over the frame. include_groups is
    # a newer pandas argument than the one installed here, and omitting it emits a
    # deprecation warning on versions that do have it.
    by_fold = delta.groupby(merged["fold"]).median()
    return {
        "n": int(len(merged)),
        "M0": float(merged["kge_0"].median()),
        "M1": float(merged["kge_1"].median()),
        "gain": float(delta.median()),
        "frac_improved": float((delta > 0).mean()),
        "fold_sd_of_gain": float(by_fold.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="outputs/v2_lr_robustness", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.out_dir / "lr_robustness.log")

    payload: dict = {"rates": list(RATES), "arms": {}}
    for arm, roots in ARMS.items():
        payload["arms"][arm] = {}
        for rate in RATES:
            got = paired(roots["m0"], roots[rate])
            if got is None:
                logger.info("%s at %s: missing", arm, rate)
                continue
            payload["arms"][arm][rate] = got

    logger.info("target domain, paired median KGE gain by transfer learning rate:")
    logger.info("  %-10s %12s %12s", "arm", *RATES)
    for arm, byrate in payload["arms"].items():
        logger.info("  %-10s %+12.4f %+12.4f", arm,
                    byrate[RATES[0]]["gain"], byrate[RATES[1]]["gain"])

    # The two quantities the paper's central claim rests on, at each rate. Reported here so
    # the decision to stay at one rate is auditable rather than asserted.
    derived = {}
    for rate in RATES:
        r, b = payload["arms"]["random"][rate], payload["arms"]["blocked"][rate]
        derived[rate] = {
            "gain_ratio_blocked_over_random": b["gain"] / r["gain"],
            "endpoint_gap": abs(r["M1"] - b["M1"]),
        }
    payload["derived"] = derived
    logger.info("")
    logger.info("the two comparisons the central claim rests on:")
    for name, key in (("gain ratio, blocked over random", "gain_ratio_blocked_over_random"),
                      ("gap between fine-tuned endpoints", "endpoint_gap")):
        logger.info("  %-34s %12.4f %12.4f", name,
                    derived[RATES[0]][key], derived[RATES[1]][key])

    africa = {}
    for rate, path in AFRICA.items():
        if Path(path).exists():
            d = json.loads(Path(path).read_text())
            africa[rate] = {"M1": d["M1"]["median_kge"],
                            "gain": d["paired"]["median_delta_kge"],
                            "frac_improved": d["paired"]["frac_improved"]}
    payload["africa"] = africa
    if len(africa) == len(RATES):
        logger.info("")
        logger.info("  %-34s %12.4f %12.4f", "Africa paired gain",
                    africa[RATES[0]]["gain"], africa[RATES[1]]["gain"])

    replay = {}
    for rate, path in REPLAY.items():
        if Path(path).exists():
            e = json.loads(Path(path).read_text())["effects"]
            replay[rate] = {k: {"recovered_share": v["share_of_degradation_recovered"],
                                "degradation": v["source_degradation_without_replay"]}
                            for k, v in e.items()}
    payload["replay"] = replay
    if len(replay) == len(RATES):
        logger.info("  %-34s %11.1f%% %11.1f%%", "replay recovery, blocked split",
                    100 * replay[RATES[0]]["blocked"]["recovered_share"],
                    100 * replay[RATES[1]]["blocked"]["recovered_share"])

    # The span the paper quotes: every arm's gain across every rate that was run.
    spans = {}
    for arm, byrate in payload["arms"].items():
        values = [byrate[r]["gain"] for r in RATES if r in byrate]
        spans[arm] = {"min": min(values), "max": max(values)}
    payload["gain_span"] = spans
    logger.info("")
    for arm, span in spans.items():
        logger.info("  %s gain spans %+.4f to %+.4f across the rates run",
                    arm, span["min"], span["max"])
    logger.info("  ratio stays near %.1f and the endpoint gap stays under %.2f",
                np.mean([derived[r]["gain_ratio_blocked_over_random"] for r in RATES]),
                max(derived[r]["endpoint_gap"] for r in RATES))

    (args.out_dir / "lr_robustness.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out_dir / "lr_robustness.json")


if __name__ == "__main__":
    main()
