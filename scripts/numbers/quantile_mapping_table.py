"""Summary of the quantile-mapping baseline: the daily-observation correction baseline (scripts/analysis/correction_baseline.py, v3).

Per design and seed, the median over target gauges of the full-series hourly metrics of M0, M0
corrected (QM, variance scaling), M1, and M1 corrected; then the median over the three seeds and
the seed range. Every statistic is taken within a seed first (a per-gauge seed median would narrow
the spread). The paired differences are per gauge within a seed, then the median over gauges.
Second table: the five designs pooled and binned into fifths of the climate dissimilarity index
(reports/m1_vs_m0/dissimilarity_per_gauge.csv).

    python -m scripts.numbers.quantile_mapping_table     # -> reports/quantile_mapping/
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT = Path("reports/quantile_mapping")
RUNS = {"random": ["al_v2_runB", "var_al_random_s43", "var_al_random_s44"],
        "blocked": ["al_v2_blocked", "var_al_blocked_s43", "var_al_blocked_s44"],
        "climate": ["al_v2_climate", "var_al_climate_s43", "var_al_climate_s44"],
        "longitudinal": ["al_v2_longitudinal", "var_al_longitudinal_s43", "var_al_longitudinal_s44"],
        "continental": ["al_v2_continental", "var_al_continental_s43", "var_al_continental_s44"]}
SEEDS = (42, 43, 44)
ARMS = ("M0", "qm", "linear", "M1", "M1_qm", "M1_linear")


def load() -> pd.DataFrame:
    parts = []
    for design, runs in RUNS.items():
        for seed, run in zip(SEEDS, runs):
            files = sorted(glob.glob(f"outputs/{run}/correction_baseline_v3/correction_fold*.csv"))
            assert len(files) == (4 if design == "continental" else 5), (run, len(files))
            parts.append(pd.concat(map(pd.read_csv, files)).assign(design=design, seed=seed))
    t = pd.concat(parts, ignore_index=True)
    di = pd.read_csv("reports/m1_vs_m0/dissimilarity_per_gauge.csv").rename(columns={"Unnamed: 0": "station_id"})
    return t.merge(di[["station_id", "design", "DI_clim"]], on=["station_id", "design"], how="left")


def per_seed(t: pd.DataFrame) -> pd.Series:
    r = {}
    for m in ARMS:
        r[f"{m}_kge"] = t[f"{m}_kge_full"].median()
        r[f"{m}_neg"] = (t[f"{m}_kge_full"] < 0).mean()
        for k in ("r", "alpha", "beta", "peak_mape"):
            r[f"{m}_{k}"] = t[f"{m}_{k}"].median()
        r[f"{m}_absfhv"] = t[f"{m}_fhv_pct"].abs().median()
    for a, b in (("M1_qm", "qm"), ("M1", "qm"), ("M1_qm", "M1")):
        d = t[f"{a}_kge_full"] - t[f"{b}_kge_full"]
        r[f"{a}-{b}"], r[f"{a}>{b}"] = d.median(), (d > 0).mean()
        r[f"{a}-{b}_p"] = wilcoxon(t[f"{a}_kge_full"], t[f"{b}_kge_full"], nan_policy="omit").pvalue
    return pd.Series(r)


def summarise(per: pd.DataFrame, by: str) -> pd.DataFrame:
    g = per.groupby(by)
    return pd.concat({"median": g.median(), "min": g.min(), "max": g.max()}, axis=1)


def tex(v: float) -> str:
    return f"{v:.2f}".replace("-", "$-$")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = load()
    per = t.groupby(["design", "seed"]).apply(per_seed).reset_index()
    by_design = summarise(per.drop(columns="seed"), "design").loc[list(RUNS)]
    by_design.to_csv(OUT / "by_design.csv")
    t["DI_fifth"] = pd.qcut(t.DI_clim, 5, labels=False)
    per_di = t.groupby(["DI_fifth", "seed"]).apply(
        lambda x: pd.concat([per_seed(x), pd.Series({"DI_clim": x.DI_clim.median(), "n": len(x)})])).reset_index()
    by_di = summarise(per_di.drop(columns="seed"), "DI_fifth")
    by_di.to_csv(OUT / "by_dissimilarity.csv")

    med = by_design["median"]
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Design & $M_0$ & $M_0$ + QM & $M_1$ & $M_1$ + QM \\", r"\midrule"]
    for d in RUNS:
        cells = []
        for m in ("M0", "qm", "M1", "M1_qm"):
            v, lo, hi = med.loc[d, f"{m}_kge"], by_design["min"].loc[d, f"{m}_kge"], by_design["max"].loc[d, f"{m}_kge"]
            cells.append(tex(v) + (f" ({tex(lo)} to {tex(hi)})" if hi - lo >= 0.05 else ""))
        lines.append(f"{d.capitalize().replace('Climate', 'Climate zone')} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_kge.tex").write_text("\n".join(lines) + "\n")
    (OUT / "numbers.json").write_text(json.dumps(
        {"n_rows": len(t), "by_design_median": med.round(4).to_dict(orient="index"),
         "by_dissimilarity_median": by_di["median"].round(4).to_dict(orient="index"),
         "by_dissimilarity_min": by_di["min"].round(4).to_dict(orient="index"),
         "by_dissimilarity_max": by_di["max"].round(4).to_dict(orient="index")}, indent=1))
    pd.set_option("display.width", 250)
    for cols in (["M0_kge", "qm_kge", "linear_kge", "M1_kge", "M1_qm_kge", "M1_linear_kge"],
                 ["M0_neg", "qm_neg", "M1_neg", "M1_qm_neg"],
                 ["qm_alpha", "M1_alpha", "M1_qm_alpha", "qm_beta", "M1_beta", "M1_qm_beta"],
                 ["qm_r", "M1_r", "M1_qm_r", "qm_peak_mape", "M1_peak_mape", "M1_qm_peak_mape",
                  "qm_absfhv", "M1_absfhv", "M1_qm_absfhv"],
                 ["M1_qm-qm", "M1_qm>qm", "M1_qm-qm_p", "M1-qm", "M1>qm", "M1_qm-M1", "M1_qm>M1"]):
        print(med[cols].round(3).to_string(), "\n")
    print("seed range\n", pd.concat({k: by_design[k][["qm_kge", "M1_kge", "M1_qm_kge", "M1_qm-qm"]]
                                     for k in ("min", "max")}, axis=1).round(3).to_string(), "\n")
    print("by DI fifth\n", pd.concat({k: by_di[k][["DI_clim", "qm_kge", "M1_kge", "M1_qm_kge", "M1_qm-qm", "M1_qm>qm",
                                                   "M1-qm", "qm_r", "M1_r", "qm_alpha", "M1_alpha"]]
                                      for k in ("median", "min", "max")}, axis=1).round(3).to_string())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
