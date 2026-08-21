"""Print the experiment inventory: which config produced which output, and its state.

A static table in a README goes stale the moment a config is added -- this repo already
had a README naming `configs/phase1.yaml` as "every knob" while thirteen configs existed.
So the mapping is generated from the configs and the output tree instead of transcribed.

The runtime column comes from each run's own run_meta.json, NOT from the config file,
because the two can disagree: the v1 configs were later given `initial_forget_bias: 3`
and no longer reproduce the v1 results. run_meta.json records the resolved config at
launch and is the authority.

    python -m scripts.inventory            # human-readable table
    python -m scripts.inventory --md       # markdown, for pasting into docs
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import yaml


def _settings(cfg: dict) -> dict:
    return {
        "H": cfg["data"]["lookback_hourly"],
        "forget": cfg["model"].get("initial_forget_bias"),
        "epochs": cfg["train"]["epochs"],
        "patience": cfg["train"]["patience"],
        "folds": cfg["folds"]["file"].split("/")[-1],
    }


def inherited_pretrain(root: Path) -> Path | None:
    """Which run's pretrain a transfer started from, for configs with no pretrain of
    their own -- the replay variants reuse run B's weights on purpose.

    Prefers the transfer's run_meta.json; falls back to the "pretrained <path>" line in
    transfer.log for runs that finished before that file was written.
    """
    for meta in sorted(root.glob("fold*/transfer/run_meta.json")):
        src = json.loads(meta.read_text()).get("pretrained_from")
        if src:
            return Path(src).parents[1]
    for log in sorted(root.glob("fold*/transfer/transfer.log")):
        for line in log.read_text(errors="ignore").splitlines()[:40]:
            if "pretrained " in line:
                token = line.split("pretrained ", 1)[1].split()[0]
                return Path(token).parents[1]
    return None


def runtime_settings(root: Path) -> dict | None:
    """What the run actually used, from the fold it happened to write first.

    A run with no pretrain of its own inherits one, and inherits its pretrain settings
    with it -- reading the config instead would misreport v1 replay as having a forget
    gate, since the v1 configs were edited after those runs finished.
    """
    metas = sorted(root.glob("fold*/pretrain/run_meta.json"))
    if not metas:
        borrowed = inherited_pretrain(root)
        if borrowed is not None:
            inner = sorted(Path(borrowed).glob("pretrain/run_meta.json"))
            if not inner:
                inner = sorted(Path(str(borrowed).rsplit("/fold", 1)[0]).glob("fold*/pretrain/run_meta.json"))
            if inner:
                out = _settings(json.loads(inner[0].read_text())["config"])
                out["inherited_from"] = str(borrowed).replace("outputs/", "").split("/")[0]
                return out
        return None
    cfg = json.loads(metas[0].read_text())["config"]
    return {
        "H": cfg["data"]["lookback_hourly"],
        "forget": cfg["model"].get("initial_forget_bias"),
        "epochs": cfg["train"]["epochs"],
        "patience": cfg["train"]["patience"],
        "folds": cfg["folds"]["file"].split("/")[-1],
    }


def collect() -> list[dict]:
    rows = []
    for path in sorted(glob.glob("configs/phase1*.yaml")):
        cfg = yaml.safe_load(Path(path).read_text())
        root = Path(cfg["output_root"])
        rt = runtime_settings(root)
        declared = {
            "H": cfg["data"]["lookback_hourly"],
            "forget": cfg["model"].get("initial_forget_bias"),
            "epochs": cfg["train"]["epochs"],
            "patience": cfg["train"]["patience"],
            "folds": cfg["folds"]["file"].split("/")[-1],
        }
        drift = sorted(k for k in declared if rt and declared[k] != rt[k])
        rows.append({
            "config": Path(path).name,
            "output": str(root).replace("outputs/", ""),
            "H": (rt or declared)["H"],
            "forget": (rt or declared)["forget"],
            "epochs": (rt or declared)["epochs"],
            "patience": (rt or declared)["patience"],
            "folds": (rt or declared)["folds"].replace("folds_", "").replace(".csv", ""),
            "replay": cfg["transfer"].get("source_replay_ratio", 0),
            "pretrain": f'{len(list(root.glob("fold*/pretrain/DONE")))}/5',
            "transfer": f'{len(list(root.glob("fold*/transfer/summary.json")))}/5',
            "source": (f'继承 {rt["inherited_from"]}' if rt and "inherited_from" in rt
                       else "run_meta" if rt else "config only"),
            "drift": ",".join(drift),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment inventory for Phase I.")
    parser.add_argument("--md", action="store_true", help="Markdown table.")
    args = parser.parse_args()

    rows = collect()
    cols = [("config", "config", 34), ("output", "output_root", 15), ("H", "H", 4),
            ("forget", "forget", 7), ("epochs", "ep", 4), ("patience", "pat", 4),
            ("folds", "folds", 8), ("replay", "replay", 7),
            ("pretrain", "pre", 5), ("transfer", "xfer", 5), ("source", "settings from", 13)]
    if args.md:
        print("| " + " | ".join(h for _, h, _ in cols) + " |")
        print("|" + "|".join("---" for _ in cols) + "|")
        for r in rows:
            print("| " + " | ".join(f'`{r[k]}`' if k in ("config", "output") else str(r[k])
                                    for k, _, _ in cols) + " |")
    else:
        print("".join(h.ljust(w) for _, h, w in cols))
        for r in rows:
            print("".join(str(r[k]).ljust(w) for k, _, w in cols))

    drifted = [r for r in rows if r["drift"]]
    if drifted:
        print("\nCONFIG DRIFT -- these files no longer reproduce the run that used them.")
        print("The table above reports what the RUN used; the file now says something else:")
        for r in drifted:
            print(f"  {r['config']}: differs in {r['drift']}")
        print("Authority is outputs/<run>/fold*/pretrain/run_meta.json.")


if __name__ == "__main__":
    main()
