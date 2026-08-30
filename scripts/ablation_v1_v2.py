"""Separate the two changes that v1 -> v2 made at once.

v2 differs from v1 in two ways: the hourly look-back went from \\SI{72}{\\hour} to
\\SI{336}{\\hour} and the forget gate gained an initial bias of 3. Reporting the pair
together and then attributing the improvement to either one is not supported by that
comparison -- a criticism the report earned and this script answers.

The hyperparameter search already contains two strictly single-variable pairs, differing
in exactly one key and nothing else:

    r06_bs256_do4_hs64_H336  ->  g01_gauch_hs64_H336     only initial_forget_bias
    g03_forgetbias_H72       ->  g02_gauch_hs128_H336    only lookback_hourly

Each search configuration ran ONE fold, so these estimates carry no between-fold spread and
must be read against the noise level established elsewhere in this project -- a fold-level
standard deviation of about 0.008 on the random split. An effect smaller than that is not
distinguishable from noise here, and the script says so rather than ranking the two.

    python -m scripts.ablation_v1_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from common.utils import setup_logging

# The noise floor this project measured for a single fold on the random split.
FOLD_NOISE = 0.0078

PAIRS = (
    ("forget gate 0 -> 3", "initial_forget_bias",
     "r06_bs256_do4_hs64_H336", "g01_gauch_hs64_H336"),
    ("hourly look-back 72 -> 336", "lookback_hourly",
     "g03_forgetbias_H72", "g02_gauch_hs128_H336"),
)


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(flatten(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def scores(root: Path) -> tuple[int, float | None, float | None]:
    m0, m1 = [], []
    for p in sorted(root.glob("fold*/transfer/summary.json")):
        d = json.loads(p.read_text())
        m0.append(d["step1_M0_target_hourly"]["median_kge"])
        m1.append(d["step2_M1_target_hourly"]["median_kge"])
    if not m0:
        return 0, None, None
    return len(m0), float(np.median(m0)), float(np.median(m1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolate the two v1->v2 changes.")
    parser.add_argument("--search-cfg", default="configs/search", type=Path)
    parser.add_argument("--search-out", default="outputs/search", type=Path)
    parser.add_argument("--out-dir", default="outputs/v2_ablation", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.out_dir / "ablation.log")

    rows = []
    for label, key, base, changed in PAIRS:
        cfgs = {}
        for name in (base, changed):
            path = args.search_cfg / f"{name}.yaml"
            if not path.exists():
                raise SystemExit(f"{path} missing")
            cfgs[name] = flatten(yaml.safe_load(path.read_text()))
        # The pair is only an ablation if it differs in one key. Verify rather than trust
        # the file names, which is how a "single-variable" comparison quietly stops being one.
        keys = set(cfgs[base]) | set(cfgs[changed])
        differing = sorted(k for k in keys
                           if str(cfgs[base].get(k)) != str(cfgs[changed].get(k))
                           and k != "output_root")
        if len(differing) != 1:
            raise SystemExit(f"{base} vs {changed} differ in {differing}, not one key")

        n_b, m0_b, m1_b = scores(args.search_out / base)
        n_c, m0_c, m1_c = scores(args.search_out / changed)
        if m1_b is None or m1_c is None:
            logger.warning("%s: no scored folds, skipped", label)
            continue
        rows.append({
            "change": label, "key": differing[0],
            "config_before": base, "config_after": changed,
            "n_folds_before": n_b, "n_folds_after": n_c,
            "M0_before": m0_b, "M0_after": m0_c, "delta_M0": m0_c - m0_b,
            "M1_before": m1_b, "M1_after": m1_c, "delta_M1": m1_c - m1_b,
            "above_fold_noise": abs(m1_c - m1_b) > FOLD_NOISE,
        })

    if not rows:
        raise SystemExit("neither pair had scored folds")
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "ablation_v1_v2.csv", index=False)

    logger.info("Single-variable ablation, %d fold per configuration:", int(table.n_folds_after.max()))
    for r in table.itertuples():
        logger.info("  %-28s M1 %.4f -> %.4f  (%+.4f)  %s", r.change, r.M1_before, r.M1_after,
                    r.delta_M1,
                    "above the fold noise floor" if r.above_fold_noise
                    else f"WITHIN the fold noise floor of {FOLD_NOISE}")
    summary = {
        "fold_noise_floor": FOLD_NOISE,
        "n_folds_per_configuration": int(table.n_folds_after.max()),
        "effects": {r.change: {"delta_M1": r.delta_M1, "delta_M0": r.delta_M0,
                               "above_fold_noise": bool(r.above_fold_noise)}
                    for r in table.itertuples()},
        "caveat": ("One fold per configuration, so these are point estimates with no "
                   "between-fold spread. An effect below the fold noise floor is not "
                   "distinguishable from zero here."),
    }
    with open(args.out_dir / "ablation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("wrote %s", args.out_dir / "ablation_v1_v2.csv")


if __name__ == "__main__":
    main()
