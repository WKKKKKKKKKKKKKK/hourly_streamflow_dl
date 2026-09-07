"""What the prepared dataset contains, and what the source database holds that it does not.

The data section began at 9,181 gauges and documented every filter below that. It never
checked upstream, and the number above it was in two files this project already reads:
PLAN.md and README.md both state 10,423 gauges across seven networks. The difference is
1,242 and includes one network in full.

Three separate reasons account for it, and only one is a choice this study made:

  area cap        The prepared dataset carries max_area 10000, so 660 catchments larger
                  than that are absent. Set at dataset preparation, not here.
  record length   min_data 1095, so a gauge with under three years is absent.
  network         The Czech collection, 437 gauges, is not in the prepared dataset at all.
                  The area cap accounts for 8 of those, so this is a composition choice made
                  when the dataset was built, not a filter this study applied.

The Czech exclusion is the one a reader could mistake for cherry-picking, so this also
measures whether it removes any part of the covered attribute space. It does not: those
catchments sit inside the range already spanned by the German and Austrian gauges that are
included, which is the substantive answer rather than an assurance.

    python -m scripts.check_network_scope
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

STATIC = ("/ibex/project/c2266/abbaa0a/data/gscad_database/processed/20250630/"
          "hourly/dataframes/static.csv")
PREPARED_CONFIG = (
    "/ibex/project/c2266/abbaa0a/data/input_data/hourly_q_dl/"
    "6datasets_H_8760_512_temporal_local_19800101_20251231_0.7_0.3_False_ts_func2_1000_"
    "5.0_False_10000_s_func2_1095_10000_3_42_penman_MSWEP_V316_test_Past_hourly_10_"
    "hourly_9181_42/config.json"
)
# Attributes with a hydrological meaning for how a catchment responds. Deliberately not the
# full 42: the point is whether the excluded network occupies new ground on the properties a
# rainfall-runoff model actually uses, not whether every column happens to overlap.
COMPARE = [
    "area",
    "slope_1KMmn_GMTEDmd.mat",
    "elevation_1KMmn_GMTEDmn_with_Antarctica_from_World_e-Atlas.mat",
    "HWSD_clay",
    "HWSD_sand",
    "FAO_FRA2000_forest_cover_fraction_smaller.mat",
]
COVERED_CENTRAL_EUROPE = ("Germany", "LamaHCE")


def main() -> None:
    out_dir = Path("outputs/v2_network_scope")
    out_dir.mkdir(parents=True, exist_ok=True)

    static = pd.read_csv(STATIC, comment="#", index_col=0)
    static["agency"] = [str(i).split("__")[0] for i in static.index]
    prepared = json.loads(Path(PREPARED_CONFIG).read_text())
    used = set(pd.read_csv("index/stations.csv")["station_id"])
    static["used"] = [i in used for i in static.index]

    max_area = float(prepared["max_area"])
    static["over_area"] = static["area"] > max_area

    payload = {
        "source_gauges": int(len(static)),
        "source_networks": int(static["agency"].nunique()),
        "prepared_gauges": int(prepared["num_stations"]),
        "indexed_gauges": int(len(used)),
        "max_area_km2": max_area,
        "min_record_days": int(prepared["min_data"]),
        "over_area_cap": int(static["over_area"].sum()),
    }

    print(f"source database        {payload['source_gauges']:,} gauges, "
          f"{payload['source_networks']} networks")
    print(f"prepared dataset       {payload['prepared_gauges']:,} gauges "
          f"(max_area {max_area:,.0f} km2, min_data {payload['min_record_days']} days)")
    print(f"batch index            {payload['indexed_gauges']:,} gauges")
    print()
    print("per network, source -> indexed:")
    rows = []
    for agency, group in static.groupby("agency"):
        kept = int(group["used"].sum())
        rows.append({"agency": agency, "source": len(group), "indexed": kept,
                     "dropped": len(group) - kept,
                     "over_area_cap": int(group["over_area"].sum())})
        print(f"  {agency:18s} {len(group):6d} -> {kept:6d}   dropped {len(group) - kept:5d}"
              f"   of which over the area cap {int(group['over_area'].sum()):5d}")
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "per_network.csv", index=False)

    absent = table[table["indexed"] == 0]
    payload["absent_networks"] = absent["agency"].tolist()
    payload["absent_gauges"] = int(absent["source"].sum())

    # Does the absent network occupy attribute space the retained sample does not reach?
    # Measured as the share of its catchments inside the 5th-to-95th-percentile range of the
    # retained central European gauges, attribute by attribute.
    czech = static[static["agency"].isin(absent["agency"])]
    covered = static[static["agency"].isin(COVERED_CENTRAL_EUROPE) & static["used"]]
    overlap = {}
    if len(czech) and len(covered):
        print()
        print(f"attribute overlap: {len(czech)} absent gauges against {len(covered)} "
              "retained central European gauges")
        for column in COMPARE:
            if column not in static.columns:
                continue
            lo, hi = covered[column].quantile(0.05), covered[column].quantile(0.95)
            inside = float(czech[column].between(lo, hi).mean())
            overlap[column] = inside
            print(f"  {column[:44]:46s} {100 * inside:5.1f}% inside "
                  f"[{lo:.2f}, {hi:.2f}]")
        payload["attribute_overlap"] = overlap
        payload["attribute_overlap_min"] = float(min(overlap.values()))
        print(f"  minimum across attributes: {100 * min(overlap.values()):.1f}%")

    # The tested area range, which is the scope statement the paper has to make.
    kept_area = static.loc[static["used"], "area"]
    payload["tested_area_median_km2"] = float(kept_area.median())
    payload["tested_area_p95_km2"] = float(kept_area.quantile(0.95))
    payload["tested_area_max_km2"] = float(kept_area.max())
    print()
    print(f"tested area range      median {kept_area.median():,.0f} km2, "
          f"95th percentile {kept_area.quantile(0.95):,.0f}, max {kept_area.max():,.0f}")

    (out_dir / "network_scope.json").write_text(json.dumps(payload, indent=2),
                                                encoding="utf-8")
    print(f"\nwrote {out_dir / 'network_scope.json'}")


if __name__ == "__main__":
    main()
