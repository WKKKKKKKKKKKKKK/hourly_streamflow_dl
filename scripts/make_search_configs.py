"""Hyperparameter search configs for fold 1 (PLAN.md 2).

PLAN.md asks for one search on fold 1, with the other folds reusing the result --
searching every fold is both expensive and leaks selection across folds. Everything
reported so far used hand-set, frozen hyperparameters, so this exists to answer
"were they reasonable?" rather than to replace them; adopting a winner would mean
re-running every comparison, which is a separate decision.

The grid is not generic. The dominant deficit found throughout this work is
under-dispersion -- alpha = std(sim)/std(obs) below 1 at 76.6% of stations at M0,
carrying 59.4% of the KGE gap, and collapsing to 0.162 on Africa. Two of the knobs
here are the ones that could plausibly cause it:

  dropout        0.4 is high. Training with heavy dropout averages over many
                 sub-networks, and averaging suppresses output variance. If alpha
                 rises sharply at lower dropout, the headline deficit is partly a
                 regularisation artefact rather than an inherent limit.
  hidden_size    too little capacity forces the model toward the conditional mean,
                 which also depresses variance.

The rest vary one thing each from the frozen baseline, so every run is interpretable
against it rather than being a point in an uninterpretable joint space.

    python -m scripts.make_search_configs --out-dir configs/search
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# (name, {dotted key: value}) -- one deviation each, except where noted.
VARIANTS = [
    ("baseline", {}),
    # --- the alpha hypothesis -------------------------------------------------
    ("dropout00", {"model.dropout": 0.0}),
    ("dropout02", {"model.dropout": 0.2}),
    ("dropout06", {"model.dropout": 0.6}),
    ("hidden64", {"model.hidden_size_daily": 64, "model.hidden_size_hourly": 64}),
    ("hidden256", {"model.hidden_size_daily": 256, "model.hidden_size_hourly": 256}),
    # --- capacity / depth -----------------------------------------------------
    ("layers2", {"model.num_layers": 2}),
    # --- optimisation ---------------------------------------------------------
    ("lr1e3", {"train.lr": 1.0e-3, "train.lr_schedule": "1:1e-3,12:2e-4,22:1e-4"}),
    ("lr2e4", {"train.lr": 2.0e-4, "train.lr_schedule": "1:2e-4,12:5e-5,22:2e-5"}),
    # --- inputs ---------------------------------------------------------------
    ("hourly168", {"data.lookback_hourly": 168}),
    ("hourly24", {"data.lookback_hourly": 24}),
    # --- the daily-mean regulariser ------------------------------------------
    ("reglambda03", {"train.reg_lambda": 0.3}),
    ("reglambda30", {"train.reg_lambda": 3.0}),
]


def set_path(tree: dict, dotted: str, value):
    node = tree
    parts = dotted.split(".")
    for key in parts[:-1]:
        node = node[key]
    if parts[-1] not in node:
        raise KeyError(f"{dotted} is not in the base config -- typo?")
    node[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the fold-1 search configs.")
    parser.add_argument("--base", default="configs/phase1_runB.yaml")
    parser.add_argument("--out-dir", default="configs/search")
    parser.add_argument("--output-root", default="outputs/search")
    args = parser.parse_args()

    base_text = Path(args.base).read_text()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name, overrides in VARIANTS:
        tree = yaml.safe_load(base_text)
        for dotted, value in overrides.items():
            set_path(tree, dotted, value)
        tree["output_root"] = f"{args.output_root}/{name}"
        # A search run needs its transfer step to start from its OWN pretraining,
        # so leave output_root self-contained.
        path = out_dir / f"{name}.yaml"
        path.write_text(
            f"# Fold-1 hyperparameter search variant: {name}\n"
            f"# Deviation from {args.base}: "
            + (", ".join(f"{k} = {v}" for k, v in overrides.items()) if overrides else "none (baseline)")
            + "\n"
            + yaml.safe_dump(tree, sort_keys=False, allow_unicode=True, width=100)
        )
        written.append((name, overrides, path))

    print(f"wrote {len(written)} configs to {out_dir}")
    for name, overrides, path in written:
        desc = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items()) or "(frozen baseline)"
        print(f"  {name:14s} {desc}")


if __name__ == "__main__":
    main()
