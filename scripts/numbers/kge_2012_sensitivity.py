"""Sensitivity of the KGE results to the KGE variant: Gupta et al. (2009) against Kling et al. (2012).

Every KGE in this repository is the 2009 form,

    KGE  = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2),   alpha = sd(sim)/sd(obs),  beta = mean(sim)/mean(obs).

The 2012 form (KGE') replaces alpha by the ratio of the coefficients of variation,

    gamma = (sd(sim)/mean(sim)) / (sd(obs)/mean(obs)) = alpha / beta,

so KGE' follows exactly from the r, alpha and beta already stored per gauge; nothing is re-run.

  transfer_stage  the per-gauge transfer-stage score (per_station_hourly_*_target_hourly.csv), the 8,709 scored
                  target gauges: median KGE of M0 and M1, median paired gain M1 - M0, share of gauges improved,
                  share of gauges below zero; per seed, then the median over the three seeds
  whole_period    the same on the continuous hourly series of every test day (full_series.csv; scored gauges
                  with >= 1,000 valid test hours)
  gain_shares     shares of the KGE gain carried by r, the variability term (alpha or gamma) and beta, computed
                  with the function behind amount_not_timing (m1_vs_m0_figures.gain_shares) on the same tables
  differences     largest absolute change from the 2009 to the 2012 form over the five designs
  quoted          the summary numbers as printed

    python -m scripts.numbers.kge_2012_sensitivity     # -> reports/m1_vs_m0/kge_2012_sensitivity.json
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.figures.m1_vs_m0_figures import DESIGNS, SEED_RUNS, full_tables, gain_shares, scored_gauges

OUT = Path("reports/m1_vs_m0/kge_2012_sensitivity.json")
VARIANTS = ("2009", "2012")


def kge(r, var, beta):
    return 1 - np.sqrt((r - 1) ** 2 + (var - 1) ** 2 + (beta - 1) ** 2)


def transfer_stage(run: str, model: str) -> pd.DataFrame:
    """Per scored gauge: KGE (2009, as stored and recomputed) and KGE' (2012)."""
    f = pd.concat(pd.read_csv(p) for p in glob.glob(
        f"outputs/{run}/fold*/transfer/per_station_hourly_fold*_{model}_target_hourly.csv"))
    f = f[f.station_id.isin(scored_gauges(run))].set_index("station_id")
    recomputed = kge(f.kge_r, f.kge_alpha, f.kge_beta)
    assert (recomputed - f.kge).abs().max() < 1e-4, f"{run} {model}: stored KGE is not the 2009 form"
    return pd.DataFrame({"2009": f.kge, "2012": kge(f.kge_r, f.kge_alpha / f.kge_beta, f.kge_beta)})


def whole_period(table: pd.DataFrame, model: str) -> pd.DataFrame:
    t = table.set_index("station_id")
    recomputed = kge(t[f"{model}_r"], t[f"{model}_alpha"], t[f"{model}_beta"])
    assert (recomputed - t[f"{model}_kge_full"]).abs().max() < 1e-6, "whole-period KGE is not the 2009 form"
    return pd.DataFrame({"2009": t[f"{model}_kge_full"],
                         "2012": kge(t[f"{model}_r"], t[f"{model}_alpha"] / t[f"{model}_beta"], t[f"{model}_beta"])})


def levels(pairs: list[tuple[pd.DataFrame, pd.DataFrame]]) -> dict:
    """pairs: per seed (M0, M1) frames with one column per variant -> medians over seeds."""
    out = {}
    for v in VARIANTS:
        per_seed = []
        for m0, m1 in pairs:
            j = pd.concat([m0[v], m1[v]], axis=1, keys=["M0", "M1"]).dropna()
            per_seed.append({"n": len(j), "M0": j.M0.median(), "M1": j.M1.median(),
                             "gain": (j.M1 - j.M0).median(), "frac_improved": (j.M1 > j.M0).mean(),
                             "frac_M0_below_0": (j.M0 < 0).mean(), "frac_M1_below_0": (j.M1 < 0).mean()})
        out[v] = {k: float(np.median([s[k] for s in per_seed])) for k in per_seed[0] if k != "n"}
        out[v]["n"] = sorted({s["n"] for s in per_seed})
        out[v]["per_seed"] = per_seed
    return out


def as_2012(table: pd.DataFrame) -> pd.DataFrame:
    """The full-series table with alpha -> gamma and kge_full -> KGE', so gain_shares splits KGE'."""
    t = table.copy()
    for m in ("M0", "M1"):
        t[f"{m}_alpha"] = t[f"{m}_alpha"] / t[f"{m}_beta"]
        t[f"{m}_kge_full"] = kge(t[f"{m}_r"], t[f"{m}_alpha"], t[f"{m}_beta"])
    return t


def main() -> None:
    tables = full_tables()
    res = {"transfer_stage": {}, "whole_period": {}, "gain_shares": {}}
    for key, _, _ in DESIGNS:
        res["transfer_stage"][key] = levels(
            [(transfer_stage(r, "M0"), transfer_stage(r, "M1")) for r in SEED_RUNS[key]])
        res["whole_period"][key] = levels([(whole_period(t, "M0"), whole_period(t, "M1")) for t in tables[key]])
        s09, _ = gain_shares(tables, key)
        s12, _ = gain_shares({key: [as_2012(t) for t in tables[key]]}, key)
        res["gain_shares"][key] = {"2009": {"r": s09["r"], "variability": s09["alpha"], "beta": s09["beta"]},
                                   "2012": {"r": s12["r"], "variability": s12["alpha"], "beta": s12["beta"]}}

    designs = [k for k, _, _ in DESIGNS]
    non_cont = [k for k in designs if k != "continental"]

    def change(stage, stat, keys=designs):
        return max(abs(res[stage][k]["2012"][stat] - res[stage][k]["2009"][stat]) for k in keys)

    def rng(stat, v, keys=designs, scale=100):
        vals = [res["gain_shares"][k][v][stat] * scale for k in keys]
        return min(vals), max(vals)

    res["differences"] = {
        stage: {"median_M0": change(stage, "M0"), "median_M1": change(stage, "M1"),
                "median_gain": change(stage, "gain"), "frac_improved_points": 100 * change(stage, "frac_improved")}
        for stage in ("transfer_stage", "whole_period")}
    ts = res["differences"]["transfer_stage"]
    res["quoted"] = {
        "median_kge_change_max": f"{max(ts['median_M0'], ts['median_M1']):.2f}",
        "median_gain_change_max": f"{ts['median_gain']:.2f}",
        "frac_improved_change_max_points": f"{ts['frac_improved_points']:.1f}",
        "gain_2012": "{:.2f}-{:.2f}".format(*(f(res["transfer_stage"][k]["2012"]["gain"] for k in designs)
                                              for f in (min, max))),
        "share_r_2009": "{:.0f}-{:.0f} %".format(*rng("r", "2009")),
        "share_r_2012": "{:.0f}-{:.0f} %".format(*rng("r", "2012")),
        "share_variability_2009": "{:.0f}-{:.0f} %".format(*rng("variability", "2009")),
        "share_variability_2012": "{:.0f}-{:.0f} %".format(*rng("variability", "2012")),
        "share_beta_2009": "{:.0f}-{:.0f} %".format(*rng("beta", "2009")),
        "share_beta_2012": "{:.0f}-{:.0f} %".format(*rng("beta", "2012")),
        "share_r_2012_non_continental": "{:.0f}-{:.0f} %".format(*rng("r", "2012", non_cont)),
    }

    for stage in ("transfer_stage", "whole_period"):
        print(f"\n{stage}")
        for k in designs:
            a, b = res[stage][k]["2009"], res[stage][k]["2012"]
            print(f"  {k:13s} M0 {a['M0']:+.3f} -> {b['M0']:+.3f} | M1 {a['M1']:+.3f} -> {b['M1']:+.3f} | "
                  f"gain {a['gain']:+.3f} -> {b['gain']:+.3f} | improved {a['frac_improved']:.3f} -> "
                  f"{b['frac_improved']:.3f} | n {a['n']}")
    print("\ngain shares (r / variability / beta, %)")
    for k in designs:
        g = res["gain_shares"][k]
        print(f"  {k:13s} " + "  ".join(
            f"{v}: " + " / ".join(f"{100 * g[v][c]:.0f}" for c in ("r", "variability", "beta")) for v in VARIANTS))
    print("\ndifferences", json.dumps(res["differences"], indent=1))
    print("quoted", json.dumps(res["quoted"], indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
