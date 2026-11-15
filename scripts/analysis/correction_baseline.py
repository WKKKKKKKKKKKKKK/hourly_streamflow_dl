"""Post-processing baseline for M1: correct M0 with the target gauge's daily observations, no training.

Most of M1's hourly KGE gain is variability (alpha) and volume (beta). The obvious
question is whether a post-processing of M0 with the same daily observations gets as far. This
fits, per target gauge, a correction of M0's DAILY means against the observed daily means of the
gauge's training period -- the same daily data and period M1 is fine-tuned on (a day counts when
>= transfer.min_daily_hours of its 24 hours are observed, as in fine-tuning) -- and applies it to
M0's hourly output of the validation period. Three corrections, all published practice:

  scale    volume only: d' = d * mean(obs) / mean(sim)                      (linear scaling)
  linear   mean and variance: d' = mean(obs) + (d - mean(sim)) * sd(obs) / sd(sim), >= 0
                                                                            (variance scaling)
  qm       empirical quantile mapping of the daily values, constant ratio beyond the range

The correction is fitted on daily values and carried to the hours of a day by the ratio d'/d, so
M0's within-day shape and timing are kept and only the amount changes (an additive shift when
d is ~0). Scored on the full hourly validation series with scripts.analysis.full_series_eval's metrics; M0
and M1 are rescored and must equal their values in <run>/full_series (the same series).

Two references for what the hourly SOURCE dynamics are worth, built from the same series:

  flat_obs   the observed daily mean spread evenly over its 24 hours: perfect daily information and
             none within the day
  flat_M1    M1's own daily mean spread evenly: M1 without its sub-daily shape

and r_sub, the correlation of the within-day deviations (each hour minus its day's mean, over the
days whose 24 hours are all observed): the sub-daily shape alone, free of magnitude and of
day-to-day variation. flat_* have no within-day variation, so no r_sub.

v3 adds M1_scale, M1_linear and M1_qm: the same three corrections fitted on M1's own daily means of
the training period and applied to M1. If fine-tuning only learned what a correction learns, these
gain nothing over M1; if it learned something else, correcting M1 adds to it.

    python -m scripts.analysis.correction_baseline --config configs/aligned/phase1_runB_blocked_v2.yaml --folds 0
    # -> <output_root>/correction_baseline_v3/correction_fold{k}.csv (v2, without the corrected M1,
    #    is kept in correction_baseline_v2/ (earlier versions, not used here; v1 in correction_baseline/)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.alignment import assert_same_alignment
from common.config import add_common_args, load_config, resolve
from common.utils import get_device, setup_logging
from data.dataset import load_dataset_config, load_scalers, make_loader, resolve_static_spec
from data.folds import domain_stations, load_folds
from data.hourly_windows import load_cache
from data.sources import build_eval_set
from models.mtslstm import build_model
from scripts.analysis.full_series_eval import DAY, collect, series_metrics

EPS = 1e-6   # mm/h; below this a day's mean counts as zero and the correction is additive
METHODS = ("scale", "linear", "qm")
SERIES = ("M0",) + METHODS + ("M1",) + tuple(f"M1_{m}" for m in METHODS) + ("flat_obs", "flat_M1")


def daily_obs(forcing_k: np.ndarray, t: int, min_hours: int) -> float:
    o = forcing_k[t - DAY + 1: t + 1]
    ok = np.isfinite(o)
    return float(o[ok].mean()) if ok.sum() >= min_hours else np.nan


def fit(sim: np.ndarray, obs: np.ndarray) -> dict:
    keep = np.isfinite(sim) & np.isfinite(obs)
    if keep.sum() < 2:
        return {"n": int(keep.sum())}
    s, o = sim[keep], obs[keep]
    q = np.linspace(0, 1, 101)
    return {"n": int(keep.sum()), "ms": s.mean(), "mo": o.mean(), "ss": s.std(), "so": o.std(),
            "qs": np.quantile(s, q), "qo": np.quantile(o, q)}


def corrected_daily(d: np.ndarray, p: dict, method: str) -> np.ndarray:
    if method == "scale":
        return d * (p["mo"] / p["ms"]) if p["ms"] > EPS else d + (p["mo"] - p["ms"])
    if method == "linear":
        k = p["so"] / p["ss"] if p["ss"] > EPS else 1.0
        return np.maximum(p["mo"] + (d - p["ms"]) * k, 0.0)
    # quantile mapping; beyond the fitted range keep the ratio of the end quantiles
    qs, qo = p["qs"], p["qo"]
    out = np.interp(d, qs, qo)
    hi, lo = d > qs[-1], d < qs[0]
    out[hi] = d[hi] * (qo[-1] / qs[-1] if qs[-1] > EPS else 1.0)
    out[lo] = d[lo] * (qo[0] / qs[0] if qs[0] > EPS else 1.0)
    return np.maximum(out, 0.0)


def r_sub(obs: np.ndarray, sim: np.ndarray) -> float:
    """Correlation of within-day deviations, over the days with all 24 hours observed."""
    o, m = obs.reshape(-1, DAY), sim.reshape(-1, DAY)
    full = np.isfinite(o).all(1) & np.isfinite(m).all(1)
    if full.sum() < 30:
        return float("nan")
    do = (o[full] - o[full].mean(1, keepdims=True)).ravel()
    dm = (m[full] - m[full].mean(1, keepdims=True)).ravel()
    return float(np.corrcoef(do, dm)[0, 1]) if do.std() > 0 and dm.std() > 0 else float("nan")


def to_hours(hourly: np.ndarray, d: float, d_new: float) -> np.ndarray:
    """Carry a day's correction to its 24 hours, keeping the within-day shape."""
    return hourly * (d_new / d) if d > EPS else np.maximum(hourly + (d_new - d), 0.0)


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--folds", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, list(args.set or []) + ["eval.max_batches_per_station=0"])
    run_dir = resolve(cfg.output_root)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "correction_baseline_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "correction_baseline.log")
    device = get_device()
    scalers = load_scalers(cfg.data.root)
    y_mean, y_std = float(scalers["y_mean"]), float(scalers["y_std"])
    dyn_size = len(load_dataset_config(cfg.data.root)["dyn_features"])
    _, _, static_names = resolve_static_spec(cfg.data.root, cfg.data.get("static_exclude"),
                                             cfg.data.get("onehot_static"))
    cache = load_cache(cfg.data.cache_dir, stride=int(cfg.data.get("stride", 24)))
    row = {n: i for i, n in enumerate(cache["stations"])}
    min_hours = int(cfg.get_path("transfer.min_daily_hours", 18))
    folds_tab = load_folds(resolve(cfg.folds.file))
    folds = [int(f) for f in args.folds.split(",")] if args.folds else list(range(int(cfg.folds.n_folds)))

    for fold in folds:
        out_csv = out_dir / f"correction_fold{fold}.csv"
        if out_csv.exists():
            logger.info("fold %d already done -- skipped", fold)
            continue
        ref = pd.read_csv(run_dir / "full_series" / f"full_series_fold{fold}.csv").set_index("station_id")
        models = {}
        for tag, ck in (("M0", run_dir / f"fold{fold}" / "pretrain" / "best_model.pth"),
                        ("M1", run_dir / f"fold{fold}" / "transfer" / "best_transfer_model.pth")):
            assert_same_alignment(ck, cfg, logger=logger)
            models[tag] = build_model(cfg, dyn_input_size=dyn_size, static_input_size=len(static_names)).to(device)
            models[tag].load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        _, target = domain_stations(folds_tab, fold)
        got = {}
        for split, tag in (("training", "M0"), ("training", "M1"), ("validation", "M0"), ("validation", "M1")):
            ds = build_eval_set(cfg, target, split, logger=logger)
            loader = make_loader(ds, num_workers=int(cfg.get_path("transfer.num_workers", 4)),
                                 pin_memory=device.type == "cuda")
            logger.info("fold %d %s: predicting the %s period of %d target gauges", fold, tag, split, len(target))
            got[(split, tag)] = collect(models[tag], loader, device, y_mean, y_std)

        rows, checked = [], []
        val0, val1 = got[("validation", "M0")], got[("validation", "M1")]
        for s in sorted(set(got[("training", "M0")]) & set(got[("training", "M1")]) & set(val0) & set(val1)
                        & set(ref.index)):
            fk = np.asarray(cache["forcing"][row[s], :, 3], dtype=np.float64)
            fits = {}
            for tag in ("M0", "M1"):
                tr = got[("training", tag)][s]
                sim_d = np.array([tr[t].mean() for t in sorted(tr)])
                obs_d = np.array([daily_obs(fk, t, min_hours) for t in sorted(tr)])
                fits[tag] = fit(sim_d, obs_d)
            p = fits["M0"]
            if min(p["n"], fits["M1"]["n"]) < 365:   # too few training days to fit a correction
                continue
            days = sorted(set(val0[s]) & set(val1[s]))
            t0, t1 = days[0] - DAY + 1, days[-1] + 1
            n = t1 - t0
            covered = np.zeros(n, dtype=bool)
            sims = {m: np.full(n, np.nan) for m in SERIES}
            for t in days:
                a = t - DAY + 1 - t0
                h, h1 = val0[s][t], val1[s][t]
                d = float(h.mean())
                covered[a:a + DAY] = True
                sims["M0"][a:a + DAY] = h
                sims["M1"][a:a + DAY] = h1
                sims["flat_M1"][a:a + DAY] = h1.mean()
                d1 = float(h1.mean())
                for m in METHODS:
                    sims[m][a:a + DAY] = to_hours(h, d, float(corrected_daily(np.array([d]), p, m)[0]))
                    sims[f"M1_{m}"][a:a + DAY] = to_hours(h1, d1, float(corrected_daily(np.array([d1]), fits["M1"], m)[0]))
            obs = np.where(covered, fk[t0:t1], np.nan)
            for t in days:   # observed daily mean, flat within the day (all observed hours of that day)
                a = t - DAY + 1 - t0
                o = obs[a:a + DAY]
                if np.isfinite(o).sum() >= min_hours:
                    sims["flat_obs"][a:a + DAY] = np.nanmean(o)
            rec = {"station_id": s, "fold": fold, "n_fit_days": p["n"]}
            for m, sim in sims.items():
                for key, v in series_metrics(obs, sim).items():
                    rec[f"{m}_{key}"] = v
                if not m.startswith("flat"):
                    rec[f"{m}_r_sub"] = r_sub(obs, sim)
            checked.append(max(abs(rec.get("M0_kge_full", np.nan) - ref.loc[s, "M0_kge_full"]),
                               abs(rec.get("M1_kge_full", np.nan) - ref.loc[s, "M1_kge_full"])))
            rows.append(rec)
        gap = np.nanmax(checked) if checked else np.nan
        logger.info("fold %d: %d gauges; M0 and M1 KGE here vs full_series, max |diff| %.2e", fold, len(rows), gap)
        if not np.isfinite(gap) or gap > 1e-3:
            raise SystemExit("M0/M1 rescored here differ from full_series: not the same series, refusing to write")
        pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
