"""Table: what each term of the fine-tuning objective is worth (random design).

The fine-tuning loss L2 weights two terms, one per output head, against the same daily observation.
Which of them carries the adaptation is not obvious from the loss: removing the daily-branch term
costs almost nothing, while removing the aggregate term costs about 0.07 in median KGE.

All arms start from the SAME pretrained weights -- their M0 agrees to six decimals --
so the only difference between them is the objective. Differences are paired per gauge
against the adopted arm rather than taken between medians, because the arms share their
gauges and pairing removes the between-gauge variance that dominates an unpaired
comparison.

    python -m scripts.tables.table_objective     # -> reports/tables/objective.tex (and a markdown copy to stdout)
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Config stem -> note. The weights are READ FROM THE CONFIG, never written here: a
# first version hard-coded them and swapped two arms, labelling the run that drops the
# aggregate term as the one that drops the daily-branch term.
ARMS = [
    ("phase1_runB_v2", "adopted"),
    ("phase1_runB_path_both_equal", ""),
    ("phase1_runB_path_aggheavy", ""),
    ("phase1_runB_path_aggonly", "daily term off"),
    ("phase1_runB_daily0", "daily term off"),
    # w_H = 0: the hourly aggregate is not supervised, so the epoch is selected on the daily head
    ("phase1_runB_path_stateonly_dailysel", "aggregate term off"),
]
ADOPTED = "phase1_runB_v2"
EXPECT_FOLDS = 5


def arm_spec(stem: str) -> tuple[float, float, str]:
    """(w_D, w_H, output root) read from the config the run was launched with."""
    cfg = yaml.safe_load(Path(f"configs/aligned/{stem}.yaml").read_text())
    t = cfg["transfer"]
    return (float(t.get("daily_branch_weight", 1.0)), float(t["agg_loss_weight"]),
            cfg["output_root"].split("/")[-1])


def per_gauge(root: str, tag: str = "M1") -> pd.Series:
    """Target-domain hourly KGE, one row per gauge; each gauge is a target once."""
    files = sorted(glob.glob(f"outputs/{root}/fold*/transfer/"
                             f"per_station_hourly_fold*_{tag}_target_hourly.csv"))
    if len(files) != EXPECT_FOLDS:
        raise SystemExit(f"{len(files)} of {EXPECT_FOLDS} folds under outputs/{root}")
    parts = []
    for path in files:
        frame = pd.read_csv(path)
        frame = frame[(frame["score_status"] == "ok") & (frame["obs_std"] >= 1e-3)]
        parts.append(frame.set_index("station_id")["kge"])
    joined = pd.concat(parts)
    if joined.index.duplicated().any():
        raise SystemExit(f"{root}: a gauge was a target in more than one fold")
    return joined


def m0_check() -> float:
    """All arms must start from the same weights, or the comparison is not of objectives."""
    values = set()
    for stem, _ in ARMS:
        root = arm_spec(stem)[2]
        payload = json.loads(Path(f"outputs/{root}/fold0/transfer/summary.json").read_text())
        values.add(round(payload["step1_M0_target_hourly"]["median_kge"], 6))
    if len(values) != 1:
        raise SystemExit(f"arms do not share pretrained weights; M0 values {values}")
    return values.pop()


def collect() -> tuple[list[dict], float]:
    m0 = m0_check()
    base = per_gauge(arm_spec(ADOPTED)[2])
    rows = []
    for stem, note in ARMS:
        w_d, w_h, root = arm_spec(stem)
        values = per_gauge(root)
        joined = pd.DataFrame({"arm": values, "base": base}).dropna()
        rows.append({
            "w_d": w_d, "w_h": w_h, "note": note, "n": len(joined),
            "median": float(joined["arm"].median()),
            "paired": float(np.median(joined["arm"] - joined["base"])),
            "adopted": stem == ADOPTED,
        })
    return rows, m0


def fmt_latex(rows: list[dict], m0: float) -> str:  # m0 checked, not printed
    body = []
    for r in rows:
        mark = r"\bfseries " if r["adopted"] else ""
        paired = "---" if r["adopted"] else f"{r['paired']:+.4f}"
        note = r["note"] or ""
        body.append(f"  {mark}{r['w_d']:g} & {mark}{r['w_h']:g} & {mark}{note} & "
                    f"{mark}{r['median']:.4f} & {mark}{paired} \\\\")
    return "\n".join([
        r"\begin{table}[H]",
        r"\caption{Terms of the fine-tuning objective L2. "
        r"$w_D$ weights the daily head, $w_H$ the aggregated hourly output, both against "
        r"the same daily observation. All arms start from identical pretrained weights "
        r"and differ only in the objective. $M_1$ is median hourly KGE over the "
        rf"{rows[0]['n']:,} target gauges of the random design; the last column is the "
        r"median per-gauge difference from the adopted arm, not a difference of medians. "
        r"With $w_H = 0$ the state handoff is the only route to the hourly branch.}",
        r"\label{tab:objective}",
        r"\centering",
        r"\begin{tabular}{rrl rr}",
        r"\toprule",
        r"$w_D$ & $w_H$ & & $M_1$ & paired vs adopted \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


def fmt_markdown(rows: list[dict], m0: float) -> str:
    out = [f"M0 (shared by all arms) = {m0:.4f}", "",
           "| w_D | w_H | | M1 | paired vs adopted |", "|---|---|---|---|---|"]
    for r in rows:
        star = " *" if r["adopted"] else ""
        paired = "—" if r["adopted"] else f"{r['paired']:+.4f}"
        out.append(f"| {r['w_d']:g} | {r['w_h']:g} | {r['note']} | "
                   f"{r['median']:.4f}{star} | {paired} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=Path("reports/tables/objective.tex"), type=Path)
    args = ap.parse_args()

    rows, m0 = collect()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(fmt_latex(rows, m0) + "\n")
    print(fmt_markdown(rows, m0))
    print(f"wrote {args.out}  ({len(rows)} arms, {rows[0]['n']:,} gauges, M0 {m0:.4f})")


if __name__ == "__main__":
    main()
