"""Length of the test period of the target gauges, for the sentence in Results 3.1
("the median duration of the test period was approximately N years").

Reads the full-series evaluation of the five designs (seed 42, the runs that sec41_figures
uses for fig4/fig13; the test period of a gauge does not depend on the seed) and counts,
per gauge, the days that were evaluated (n_days in full_series.csv: days on which both M0
and M1 have a prediction). Only the scored gauges of the paper are kept (score_status ok and
obs_std >= 1e-3, the same filter as sec41_figures.hourly: 8,709 per design). Writes reports/sec41_m1_vs_m0/test_period_length.json.

    python -m scripts.numbers.test_period_length
"""
import json
from pathlib import Path

import pandas as pd

from scripts.figures.sec41_figures import DESIGNS, paired

OUT = Path("reports/sec41_m1_vs_m0/test_period_length.json")


def main() -> None:
    res, pooled = {}, []
    for key, _, root in DESIGNS:
        f = pd.read_csv(f"outputs/{root}/full_series/full_series.csv")
        keep = paired(root).index
        d = f.drop_duplicates("station_id").set_index("station_id").loc[lambda x: x.index.isin(keep), "n_days"]
        pooled.append(d)
        res[key] = {"n_gauges": int(d.size), "median_days": float(d.median()),
                    "median_years": float(d.median() / 365.25),
                    "p10_years": float(d.quantile(0.1) / 365.25),
                    "p90_years": float(d.quantile(0.9) / 365.25)}
    d = pd.concat(pooled)
    res["all_designs_pooled"] = {"n_rows": int(d.size), "median_days": float(d.median()),
                                 "median_years": float(d.median() / 365.25),
                                 "p10_years": float(d.quantile(0.1) / 365.25),
                                 "p90_years": float(d.quantile(0.9) / 365.25)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{k:20s} {v}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
