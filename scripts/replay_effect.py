"""Does source replay protect the source domain under a blocked split as well?

Replay mixes source-domain batches back into the target-domain fine-tuning, at a rate set
by transfer.source_replay_ratio. Under a random split it reduces the source-domain
degradation that STEP 3 measures, at a small cost to the target-domain gain. That result was
only ever measured under a random split, and this project's own finding is that a random
split overstates both the level and the precision of a result. Whether the protection
survives a blocked split is a separate question, and this script answers it by putting all
four runs on one table.

Pretraining is shared within each split, so the four runs differ only in the split and in
whether replay was on. That makes the two replay effects directly comparable.

    python -m scripts.replay_effect
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.utils import setup_logging

RUNS = (
    ("random", "off", "outputs/v2_runB"),
    ("random", "0.25", "outputs/v2_replay025"),
    ("blocked", "off", "outputs/v2_blocked"),
    ("blocked", "0.25", "outputs/v2_blocked_replay025"),
)


def collect(root: Path) -> dict[int, dict]:
    """Per fold, the three numbers the comparison needs."""
    out = {}
    for path in sorted(root.glob("fold*/transfer/summary.json")):
        fold = int(path.parent.parent.name.replace("fold", ""))
        d = json.loads(path.read_text())
        out[fold] = {
            "target_M1": d["step2_M1_target_hourly"]["median_kge"],
            "source_before": d["step3_source_before"]["median_kge"],
            "source_after": d["step3_source_after"]["median_kge"],
            "source_paired_delta": d["step3_paired_median_delta_kge"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare source replay under both splits.")
    parser.add_argument("--out-dir", default="outputs/v2_replay_effect", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.out_dir / "replay_effect.log")

    per_run = {}
    rows = []
    for split, replay, root in RUNS:
        folds = collect(Path(root))
        if not folds:
            logger.warning("%s / replay %s: no scored fold under %s", split, replay, root)
            continue
        per_run[(split, replay)] = folds
        rows.append({
            "split": split, "replay": replay, "n_folds": len(folds),
            "target_M1": float(np.median([v["target_M1"] for v in folds.values()])),
            "source_after": float(np.median([v["source_after"] for v in folds.values()])),
            "source_paired_delta":
                float(np.median([v["source_paired_delta"] for v in folds.values()])),
        })
    if not rows:
        raise SystemExit("no run produced a scored fold")
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "replay_by_run.csv", index=False)

    logger.info("%-9s %-7s %-6s %10s %12s %12s", "split", "replay", "folds",
                "target M1", "source after", "source delta")
    for r in table.itertuples():
        logger.info("%-9s %-7s %-6d %10.4f %12.4f %12.4f", r.split, r.replay, r.n_folds,
                    r.target_M1, r.source_after, r.source_paired_delta)

    # The effect of replay within each split, paired fold by fold. Differencing the two
    # medians would mix fold variation into the effect, and the folds are shared here, so
    # pairing costs nothing and says more.
    effects = {}
    for split in ("random", "blocked"):
        off, on = per_run.get((split, "off")), per_run.get((split, "0.25"))
        if not off or not on:
            continue
        shared = sorted(set(off) & set(on))
        if not shared:
            continue
        d_src = np.array([on[f]["source_paired_delta"] - off[f]["source_paired_delta"]
                          for f in shared])
        d_tgt = np.array([on[f]["target_M1"] - off[f]["target_M1"] for f in shared])
        base = np.median([off[f]["source_paired_delta"] for f in shared])
        effects[split] = {
            "n_folds": len(shared),
            "source_degradation_without_replay": float(base),
            "source_degradation_recovered": float(np.median(d_src)),
            "share_of_degradation_recovered": float(np.median(d_src) / abs(base)),
            "target_gain_given_up": float(np.median(d_tgt)),
            "recovered_per_unit_given_up":
                float(np.median(d_src) / abs(np.median(d_tgt))) if np.median(d_tgt) else None,
            "same_sign_all_folds": bool((d_src > 0).all() or (d_src < 0).all()),
        }
        e = effects[split]
        logger.info("%s split: replay recovers %+.4f of a %.4f source degradation "
                    "(%.0f%%), and gives up %+.4f on the target. Ratio %.1f to 1.",
                    split, e["source_degradation_recovered"], base,
                    100 * e["share_of_degradation_recovered"], e["target_gain_given_up"],
                    e["recovered_per_unit_given_up"] or float("nan"))
    with open(args.out_dir / "replay_effect.json", "w", encoding="utf-8") as handle:
        json.dump({"per_run": rows, "effects": effects}, handle, indent=2)
    if len(effects) < 2:
        logger.info("Only one split has both runs. The blocked comparison is still pending.")
    logger.info("wrote %s", args.out_dir / "replay_by_run.csv")


if __name__ == "__main__":
    main()
