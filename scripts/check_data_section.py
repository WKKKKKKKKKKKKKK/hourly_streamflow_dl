"""Check the hand-written data section against the files it describes.

Every other number in this report is generated, so it cannot drift. The data section is
written by hand, because its content is structural description rather than results and reads
badly when assembled from format strings. That makes it the one place where a stale number
could survive a rebuild unnoticed, so the quantities that could change are checked here.

    python -m scripts.check_data_section     # exits non-zero on any mismatch
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def facts() -> dict[str, int]:
    """The quantities the section states, read from the files that hold them."""
    out = {}
    meta = json.loads(Path("/ibex/user/kongw0a/hourly_cache/cache_meta.json").read_text())
    out["n_cache_stations"] = int(meta["n_stations"])
    out["n_hours"] = int(meta["n_hours"])

    comp = pd.read_csv("outputs/v2_runB/diagnostics_allhours/kge_components_target.csv")
    out["n_scored"] = int((comp["obs_std"] >= 1e-3).sum())
    out["n_dropped_degenerate"] = int(len(comp) - out["n_scored"])

    index = np.load("/ibex/user/kongw0a/hourly_cache/samples_stride24.npz", allow_pickle=True)
    out["n_samples"] = int(index["station_idx"].shape[0])
    out["n_train_samples"] = int(index["is_train"].sum())
    out["n_valid_samples"] = int((~index["is_train"]).sum())

    basins = pd.read_csv("africa/africa_basins.csv")
    out["n_africa_basins"] = int(len(basins))
    scored = pd.read_csv("outputs/v2_africa_insitu_summary/ensemble_per_basin_M1.csv")
    out["n_africa_scored"] = int(len(scored))
    out["africa_median_days"] = int(scored["n_days"].median())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the data section's numbers.")
    parser.add_argument("--section", default="reports/latex/data_description.tex", type=Path)
    args = parser.parse_args()

    text = args.section.read_text(encoding="utf-8")
    # \num{...} strips the separators LaTeX adds, so compare on digits alone.
    stated = {int(m.replace(",", "")) for m in re.findall(r"\\num\{([\d,]+)\}", text)}

    known = facts()
    missing = {k: v for k, v in known.items() if v not in stated}
    print(f"{args.section}: {len(stated)} distinct integers stated")
    for key, value in sorted(known.items()):
        mark = "ok " if value in stated else "NOT FOUND"
        print(f"  {mark:9s} {key:24s} {value:,}")
    if missing:
        print("\nThe section does not state these values, or states them in a stale form:")
        for k, v in missing.items():
            print(f"  {k} = {v:,}")
        sys.exit(1)
    print("\nevery checked quantity appears in the section")


if __name__ == "__main__":
    main()
