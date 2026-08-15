"""Hyperparameter search configs for fold 1 (PLAN.md 2).

PLAN.md asks for one search on fold 1, with the other folds reusing the result --
searching every fold is both expensive and leaks selection across folds. Everything
reported so far used hand-set, frozen hyperparameters, so this exists to answer
"were they reasonable?" rather than to replace them; adopting a winner would mean
re-running every comparison, which is a separate decision.

The grid mirrors the reference 100-station experiment's joint sweep rather than
varying one knob at a time, so interactions are visible -- a small hidden size may
only work at low dropout, and one-at-a-time deviations cannot show that.

Batches per epoch is rescaled with chunk_size so every variant sees the SAME number
of samples per epoch (10.24M). Without that, "batch 128" would also silently mean
"a quarter of the training data per epoch", and batch size could not be separated
from training volume.

    python -m scripts.make_search_configs --out-dir configs/search
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# The reference 100-station experiment tuned a JOINT grid, not one knob at a time:
# its run directories encode bs {128,256} x do {0.4,0.6} x hs {64,128,256} x
# H {72,168,336} with D fixed at 365, and its daily-LSTM baseline additionally swept
# lr {1e-4, 5e-4, 1e-3}. Mirroring that is the defensible choice -- a joint grid can
# find interactions (small hidden size may only work with low dropout) that
# one-at-a-time deviations cannot.
#
# Two additions to the reference grid, both deliberate:
#   * dropout 0.0 and 0.2. The reference only tried 0.4 and 0.6, but the dominant
#     deficit in this work is under-dispersion (alpha < 1 at 76.6% of stations,
#     59.4% of the KGE gap, 0.162 on Africa). Heavy dropout averages over
#     sub-networks and averaging suppresses output variance, so low dropout is the
#     most direct test of whether that deficit is partly a regularisation artefact.
#   * the frozen baseline itself, so every variant has a like-for-like reference.
#
# In this codebase one dataset item IS a chunk of samples, so chunk_size is the
# effective batch size and the reference's bs maps onto it.

def _spec(bs, do, hs, h, lr=None):
    out = {
        "data.chunk_size": bs,
        "model.dropout": do,
        "model.hidden_size_daily": hs,
        "model.hidden_size_hourly": hs,
        "data.lookback_hourly": h,
    }
    if lr is not None:
        out["train.lr"] = lr
        exp = {1.0e-3: "1:1e-3,12:2e-4,22:1e-4", 1.0e-4: "1:1e-4,12:5e-5,22:2e-5"}[lr]
        out["train.lr_schedule"] = exp
    # Hold samples/epoch fixed: 20000 x 512 in the frozen baseline. A smaller batch
    # therefore takes proportionally more optimiser steps, which is what changing the
    # batch size means -- as opposed to quietly training on less data.
    out["train.batches_per_epoch"] = int(20000 * 512 / bs)
    out["transfer.batches_per_epoch"] = int(4000 * 512 / bs)
    return out


VARIANTS = [
    ("baseline", {}),                                   # frozen: bs512 do0.4 hs128 H72
    # --- the reference's 12 combinations ---------------------------------------
    ("r01_bs128_do4_hs64_H72",   _spec(128, 0.4, 64, 72)),
    ("r02_bs128_do4_hs64_H168",  _spec(128, 0.4, 64, 168)),
    ("r03_bs256_do4_hs64_H168",  _spec(256, 0.4, 64, 168)),
    ("r04_bs128_do6_hs64_H168",  _spec(128, 0.6, 64, 168)),
    ("r05_bs256_do6_hs64_H168",  _spec(256, 0.6, 64, 168)),
    ("r06_bs256_do4_hs64_H336",  _spec(256, 0.4, 64, 336)),
    ("r07_bs128_do4_hs128_H168", _spec(128, 0.4, 128, 168)),
    ("r08_bs256_do4_hs128_H168", _spec(256, 0.4, 128, 168)),
    ("r09_bs128_do6_hs128_H168", _spec(128, 0.6, 128, 168)),
    ("r10_bs128_do4_hs128_H336", _spec(128, 0.4, 128, 336)),
    ("r11_bs128_do4_hs256_H168", _spec(128, 0.4, 256, 168)),
    ("r12_bs256_do4_hs256_H168", _spec(256, 0.4, 256, 168)),
    # --- low dropout, to test the under-dispersion hypothesis -------------------
    ("d00_bs128_hs128_H168", _spec(128, 0.0, 128, 168)),
    ("d02_bs128_hs128_H168", _spec(128, 0.2, 128, 168)),
    ("d02_bs128_hs256_H168", _spec(128, 0.2, 256, 168)),
    # --- learning rate, from the reference's daily-baseline sweep ---------------
    ("lr1e3_bs128_hs128_H168", _spec(128, 0.4, 128, 168, lr=1.0e-3)),
    ("lr1e4_bs128_hs128_H168", _spec(128, 0.4, 128, 168, lr=1.0e-4)),
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
