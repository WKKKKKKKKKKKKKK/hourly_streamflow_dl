"""Summary of the quantile-mapping baseline: the daily-observation correction baseline (scripts/analysis/correction_baseline.py, v3).

Gauges: the scored target gauges (score_status ok, obs_std >= 1e-3: the 8,709 of every other
hourly statistic) that have a correction, i.e. 365 or more training days to fit it (8,468).
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
    """Correction results of the scored target gauges (the 8,709 of every other hourly statistic,
    scripts.figures.m1_vs_m0_figures.scored_gauges) that have a correction: 365 or more training
    days to fit it (scripts/analysis/correction_baseline.py)."""
    from scripts.figures.m1_vs_m0_figures import scored_gauges
    parts = []
    for design, runs in RUNS.items():
        for seed, run in zip(SEEDS, runs):
            files = sorted(glob.glob(f"outputs/{run}/correction_baseline_v3/correction_fold*.csv"))
            assert len(files) == (4 if design == "continental" else 5), (run, len(files))
            c = pd.concat(map(pd.read_csv, files))
            c = c[c.station_id.isin(scored_gauges(run))]
            parts.append(c.assign(design=design, seed=seed))
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


LABEL = {"random": "Random", "blocked": "Blocked", "climate": "Climate zone",
         "longitudinal": "Longitudinal", "continental": "Continental"}
PANELS = (("kge", "(a) KGE"), ("r", "(b) Correlation $r$"), ("alpha", r"(c) Variability ratio $\alpha$"))


def qm_table(by_design: pd.DataFrame, n_gauges: int) -> str:
    """The full quantile-mapping table, with caption: KGE, r and alpha of the four models per design,
    median over the target gauges per seed, then the median over the three seeds; the seed range
    is given where it spans 0.05 or more."""
    med, lo, hi = by_design["median"], by_design["min"], by_design["max"]
    rows = []
    for key, title in PANELS:
        rows += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{{title}}} \\"]
        for d in RUNS:
            cells = []
            for m in ("M0", "qm", "M1", "M1_qm"):
                c = f"{m}_{key}"
                cells.append(tex(med.loc[d, c]) + (f" ({tex(lo.loc[d, c])} to {tex(hi.loc[d, c])})"
                                                   if hi.loc[d, c] - lo.loc[d, c] >= 0.05 else ""))
            rows.append(f"\\quad {LABEL[d]} & " + " & ".join(cells) + r" \\")
    caption = (
        r"\caption{Daily fine-tuning against quantile mapping (QM) of the same daily observations. "
        r"QM is fitted, at each target gauge, between the simulated and observed daily discharge "
        r"of the training period and transferred to the hourly series through the daily ratio. "
        r"Values are hourly metrics of the continuous series of the whole test period: per seed, "
        rf"the median over the {n_gauges:,} of the 8{{,}}709 target gauges that have at least 365 "
        r"days of daily observations in the training period to fit the mapping, then "
        r"the median over the three seeds; the range over the seeds is given in parentheses where "
        r"it spans 0.05 or more. These whole-period scores differ slightly from the transfer-stage "
        r"scores (per_station_*_target_hourly.csv).}").replace(f"{n_gauges:,}", f"{n_gauges:,}".replace(",", "{,}"))
    return "\n".join([r"\begin{table}[H]", caption, r"\label{tab:qm}", r"\centering", r"\small",
                      r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l cccc}", r"\toprule",
                      r"Design & $M_0$ & $M_0$ + QM & $M_1$ & $M_1$ + QM \\"] + rows
                     + [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]) + "\n"


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
    n_gauges = int(t.groupby(["design", "seed"]).station_id.nunique().max())
    (OUT / "table_qm.tex").write_text(qm_table(by_design, n_gauges))
    (OUT / "numbers.json").write_text(json.dumps(
        {"n_rows": len(t),
         "n_gauges_per_design_seed": {f"{d}_{s}": int(n) for (d, s), n in
                                      t.groupby(["design", "seed"]).station_id.nunique().items()},
         "by_design_median": med.round(4).to_dict(orient="index"),
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
