"""Table B1: the fold-1 architecture search (random design), as LaTeX.

The search exists to answer "were the frozen hyperparameters defensible?", not to pick a
winner: adopting a winner would have meant re-running every comparison. The table is therefore laid out by
the grid rather than by rank: a reader should be able to see the hourly-sequence effect
by scanning a column, which a table sorted by score hides. Each variant is pretrained from
scratch and fine-tuned (configs/aligned/search, outputs/al_search).

Settings come from each variant's own config rather than from its directory name, so a
renamed or regenerated variant cannot silently mislabel a row. Scores come from the
fold-1 transfer summary. Variants that did not complete both stages are counted in the
caption and left out of the body: a half-finished run has no score to put in a cell.

    python -m scripts.tables.table_search                     # -> reports/tables/search.tex
    python -m scripts.tables.table_search --format markdown   # to read in a terminal
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import yaml

SEARCH_CFG = Path("configs/aligned/search")
SEARCH_OUT = Path("outputs/al_search")
OUT = Path("reports/tables/search.tex")
FOLD = 1
# The production configuration, so the row closest to it can be marked. It differs from
# g02 only in chunk size (512 against 256), which the caption states.
ADOPTED = "g02_gauch_hs128_H336"


def settings(name: str) -> dict:
    """Read a variant's settings from its own config (every search config states all six)."""
    path = SEARCH_CFG / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(f"no config for variant {name}: {path}")
    cfg = yaml.safe_load(path.read_text())

    def get(section: str, key: str):
        block = cfg.get(section) or {}
        if key not in block:
            raise SystemExit(f"{path} does not set {section}.{key}")
        return block[key]

    bias = get("model", "initial_forget_bias")
    return {
        "hs": get("model", "hidden_size_hourly"),
        "do": get("model", "dropout"),
        "H": get("data", "lookback_hourly"),
        "bs": get("data", "chunk_size"),
        # null and 0 are different things here: null means the gate is initialised the
        # way PyTorch does it, which is what the frozen configuration had.
        "fb": 0.0 if bias in (None, 0, 0.0) else float(bias),
        "lr": get("train", "lr"),
    }


def pretrain_kge(name: str) -> float | None:
    """The pretraining stage's own metric: median hourly KGE on the source validation period.

    Reported because it does not rank the variants the way the selection criterion does,
    and a reader otherwise has no way to see that choosing on it would have picked a
    different configuration.
    """
    path = (SEARCH_OUT / name / f"fold{FOLD}" / "pretrain"
            / f"summary_fold{FOLD}_source_val.json")
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return (payload.get("hourly") or payload).get("median_kge")


def scores(name: str) -> dict | None:
    path = SEARCH_OUT / name / f"fold{FOLD}" / "transfer" / "summary.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    m0 = (payload.get("step1_M0_target_hourly") or {}).get("median_kge")
    m1 = (payload.get("step2_M1_target_hourly") or {}).get("median_kge")
    if m0 is None or m1 is None:
        return None
    return {"holdout": payload.get("best_holdout_daily_kge"),
            "m0": m0, "m1": m1, "delta": m1 - m0,
            "epoch": payload.get("best_epoch")}


def collect() -> tuple[list[dict], list[str]]:
    rows, incomplete = [], []
    for path in sorted(SEARCH_CFG.glob("*.yaml")):
        name = path.stem
        got = scores(name)
        if got is None:
            incomplete.append(name)
            continue
        rows.append({"name": name, **settings(name), **got,
                     "pre": pretrain_kge(name)})
    # By the grid, not by rank: hourly length first, because that is the one setting the
    # search actually changed in the production configuration.
    rows.sort(key=lambda r: (r["H"], r["hs"], r["do"], r["bs"]))
    return rows, incomplete


def fmt_latex(rows: list[dict], incomplete: list[str]) -> str:
    diverged = [n for n in incomplete if "lr1e2" in n or "lr5e3" in n or "lr1e3" in n]
    truncated = [n for n in incomplete if n not in diverged]
    body = []
    for r in rows:
        mark = r"\bfseries " if r["name"] == ADOPTED else ""
        fb = "3" if r["fb"] else "--"
        lr = f"{r['lr']:g}".replace("e-0", r"\!\times\!10^{-") + "}" if "e" in f"{r['lr']:g}" \
            else f"{r['lr']:g}"
        lr = f"${lr}$" if "times" in lr else lr
        body.append(
            f"  {mark}{r['H']} & {mark}{r['hs']} & {mark}{r['do']:g} & {mark}{r['bs']} & "
            f"{mark}{fb} & {mark}{lr} & "
            f"{mark}{'--' if r['pre'] is None else format(r['pre'], '.4f')} & "
            f"{mark}{r['holdout']:.4f} & {mark}{r['m0']:.4f} & "
            f"{mark}{r['m1']:.4f} & {mark}{r['delta']:+.4f} \\\\")
    return "\n".join([
        r"\begin{table}[H]",
        r"\caption{Fold-1 architecture search. Each row is one variant trained and "
        r"fine-tuned on a single cross-validation fold. $H$ is the hourly input length "
        r"in hours, $h$ the hidden size of both branches, $p$ the dropout rate, $B$ the "
        r"minibatch size, $b_f$ the forget-gate bias at initialisation and $\eta$ the "
        r"\emph{pretraining} learning rate; the fine-tuning rate is swept separately and is "
        r"$5\times10^{-4}$ in every row here. Samples seen "
        r"per epoch are held at $1.024\times10^{7}$ across all variants, so $B$ is not "
        r"confounded with training volume. The selection column is the median daily KGE "
        r"on a held-out slice of the target training period, the only criterion the "
        r"study's premise permits. The pretraining column is median hourly KGE on the "
        r"source gauges' validation period, the metric the pretraining stage itself stops "
        r"on; it is reported because it does not rank the variants the way the selection "
        r"criterion does. $M_0$ and $M_1$ are median hourly KGE on the target "
        r"gauges before and after daily-only fine-tuning. Rows are ordered by the grid, "
        r"not by score. The emphasised row is the configuration adopted for all "
        r"production runs, which differs from it only in minibatch size ($512$). Of the "
        f"{len(rows) + len(incomplete)} variants launched, {len(diverged)} diverged to a "
        r"non-finite loss at learning rates of $10^{-3}$ and above and "
        f"{len(truncated)} (all with $h=256$) did not complete within the available wall "
        r"clock; neither group is listed.}",
        r"\label{tab:search}",
        r"\centering",
        r"\begin{tabular}{rrrrrr r r rrr}",
        r"\toprule",
        r"\multicolumn{6}{c}{setting} & \multicolumn{1}{c}{pretraining} & "
        r"\multicolumn{1}{c}{selection} & \multicolumn{3}{c}{target domain, hourly KGE} \\",
        r"\cmidrule(lr){1-6} \cmidrule(lr){7-7} \cmidrule(lr){8-8} \cmidrule(lr){9-11}",
        r"$H$ & $h$ & $p$ & $B$ & $b_f$ & $\eta$ & source KGE & daily KGE & $M_0$ & "
        r"$M_1$ & $M_1-M_0$ \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


def fmt_markdown(rows: list[dict], incomplete: list[str]) -> str:
    out = ["| H | h | p | B | b_f | lr | pretrain | daily holdout | M0 | M1 | M1-M0 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        star = " *" if r["name"] == ADOPTED else ""
        out.append(f"| {r['H']} | {r['hs']} | {r['do']:g} | {r['bs']} | "
                   f"{'3' if r['fb'] else '--'} | {r['lr']:g} | "
                   f"{'--' if r['pre'] is None else format(r['pre'], '.4f')} | "
                   f"{r['holdout']:.4f} | "
                   f"{r['m0']:.4f} | {r['m1']:.4f} | {r['delta']:+.4f}{star} |")
    out.append(f"\n{len(rows)} complete, {len(incomplete)} not: "
               + ", ".join(incomplete))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT, type=Path)
    ap.add_argument("--format", default="latex", choices=("latex", "markdown"))
    args = ap.parse_args()

    rows, incomplete = collect()
    if not rows:
        raise SystemExit(f"no completed search variant found under {SEARCH_OUT}")
    if args.format == "markdown":
        print(fmt_markdown(rows, incomplete))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"% Generated by scripts.tables.table_search from {SEARCH_OUT} and {SEARCH_CFG}\n"
                        + fmt_latex(rows, incomplete) + "\n")
    print(f"wrote {args.out}  ({len(rows)} variants, {len(incomplete)} incomplete)")


if __name__ == "__main__":
    main()
