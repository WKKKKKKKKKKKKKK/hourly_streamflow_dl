"""How far each KGE component sits from its ideal, before and after fine-tuning, per domain.

KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2). The three components answer three
different questions -- r is timing, alpha is whether the model swings as much as the river,
beta is whether it carries the right volume -- and the whole mechanism claim of this project
rests on which of them a daily total can repair. A daily aggregate carries magnitude
information, not sub-daily timing information, so it should move alpha and beta and leave r
largely alone. That is a falsifiable prediction and this script measures it.

**Distance, not the raw component.** Alpha and beta are ratios: 0.5 and 2.0 are equally
wrong, and their arithmetic mean is not 1. So their deficit is |log2(x)|, which is zero at
the ideal and symmetric under halving and doubling. r is not a ratio; its deficit is 1 - r.
The two are not comparable in magnitude, which is why the figure that reads this file gives
each component its own axis and compares only the FRACTION of deficit removed.

**Medians of per-gauge distances, never distance of the median.** |log2(median beta)| would
report a domain as unbiased whenever its over- and under-predicting gauges cancel, which is
exactly what happens on the target domain: median beta is 1.02 while half the gauges are off
by more than 20% in one direction or the other.

    python -m scripts.component_deficits
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.utils import setup_logging

COMPONENTS = ("r", "alpha", "beta")


def deficit(values: pd.Series, component: str) -> pd.Series:
    """Distance from the ideal: 1 - r for the correlation, |log2 x| for the two ratios."""
    if component == "r":
        return 1.0 - values
    return np.abs(np.log2(values.clip(lower=1e-6)))


def summarise(m0: pd.DataFrame, m1: pd.DataFrame, domain: str, n_label: str) -> list[dict]:
    rows = []
    for component in COMPONENTS:
        d0 = deficit(m0[f"kge_{component}"], component).dropna()
        d1 = deficit(m1[f"kge_{component}"], component).dropna()
        common = d0.index.intersection(d1.index)
        d0, d1 = d0.loc[common], d1.loc[common]
        removed = float(d0.median() - d1.median())
        rows.append({
            "domain": domain,
            "units": n_label,
            "n": int(len(common)),
            "component": component,
            "median_value_M0": float(m0.loc[common, f"kge_{component}"].median()),
            "median_value_M1": float(m1.loc[common, f"kge_{component}"].median()),
            "median_deficit_M0": float(d0.median()),
            "median_deficit_M1": float(d1.median()),
            "deficit_removed": removed,
            # The one number comparable ACROSS components, since the deficits are not
            # in the same units. This is what the mechanism claim is actually about.
            "fraction_removed": removed / d0.median() if d0.median() > 0 else float("nan"),
            "share_of_gauges_improved": float((d1 < d0).mean()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KGE component deficits before and after fine-tuning, per domain.")
    parser.add_argument("--target-run", default="outputs/v2_runB/diagnostics_allhours", type=Path)
    parser.add_argument("--africa-summary", default="outputs/v2_africa_insitu_summary", type=Path)
    parser.add_argument("--out-dir", default="outputs/v2_component_deficits", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.out_dir / "component_deficits.log")

    rows = []

    target = pd.read_csv(args.target_run / "kge_components_target.csv").set_index("station_id")
    m0 = target[[c for c in target.columns if c.startswith("M0_kge_")]]
    m1 = target[[c for c in target.columns if c.startswith("M1_kge_")]]
    m0.columns = [c.replace("M0_", "") for c in m0.columns]
    m1.columns = [c.replace("M1_", "") for c in m1.columns]
    rows += summarise(m0, m1, "target domain (temperate, hourly truth)", "gauges")

    a0 = pd.read_csv(args.africa_summary / "ensemble_per_basin_M0.csv").set_index("station_id")
    a1 = pd.read_csv(args.africa_summary / "ensemble_per_basin_M1.csv").set_index("station_id")
    rows += summarise(a0, a1, "Africa (external, daily truth only)", "basins")

    table = pd.DataFrame(rows)
    out_csv = args.out_dir / "component_deficits.csv"
    table.to_csv(out_csv, index=False)

    for domain, group in table.groupby("domain", sort=False):
        logger.info("%s -- %d %s", domain, group["n"].iloc[0], group["units"].iloc[0])
        logger.info("  %-7s %10s %10s %10s %9s", "", "deficit M0", "deficit M1",
                    "removed", "fraction")
        for row in group.itertuples():
            logger.info("  %-7s %10.3f %10.3f %10.3f %8.0f%%", row.component,
                        row.median_deficit_M0, row.median_deficit_M1,
                        row.deficit_removed, 100 * row.fraction_removed)

    # The claim in one line, so a reader does not have to derive it from the table.
    verdict = {}
    for domain, group in table.groupby("domain", sort=False):
        g = group.set_index("component")
        magnitude = float(np.mean([g.loc["alpha", "fraction_removed"],
                                   g.loc["beta", "fraction_removed"]]))
        timing = float(g.loc["r", "fraction_removed"])
        verdict[domain] = {"magnitude_fraction_removed": magnitude,
                           "timing_fraction_removed": timing,
                           "ratio": magnitude / timing if timing > 0 else float("nan")}
        logger.info("%s: daily aggregates remove %.0f%% of the magnitude deficit "
                    "(alpha and beta) against %.0f%% of the timing deficit (r) -- %.1fx",
                    domain, 100 * magnitude, 100 * timing, verdict[domain]["ratio"])
    with open(args.out_dir / "component_deficits_summary.json", "w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=2)
    logger.info("wrote %s", out_csv)


if __name__ == "__main__":
    main()
