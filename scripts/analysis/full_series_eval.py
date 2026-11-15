"""Full-series hourly evaluation: KGE on every hour, and flood-peak metrics on continuous series.

The transfer-stage scores use one sample per day at 23:00 (stride-24) and a random cap of
6,144 days per gauge, so the reported "hourly KGE" sees one hour of each day and no peak
metric can be computed in time order. This script rebuilds each target gauge's hourly
series from ALL of its validation days: every stride-24 sample ends at 23:00, so its last
24 hourly outputs are exactly that calendar day, and consecutive days stitch into a
continuous series. Days without a sample stay missing (never interpolated).

Per gauge and per model (M0, M1), on the full series, every metric with a published
definition so it can be cited rather than defended:
  kge_full, r, alpha, beta, nse   every hour with both values finite (Gupta et al., 2009)
  fhv_pct                         bias over the top 2 % of the flow duration curve
                                  (Yilmaz et al., 2008, Eq. A3; as in Gauch et al., 2021, Table 3)
  peak_timing_h                   Gauch et al. (2021), Appendix A, exactly: observed peaks are local
                                  maxima with topographic prominence >= the observed standard
                                  deviation, thinned (smallest first) to >= 100 steps apart; the
                                  simulated peak is the largest prediction within +/-12 h ("a window
                                  of one day"); the metric is the mean absolute lag. Same code path
                                  as NeuralHydrology's mean_peak_timing (resolution 1h), including
                                  its rule of skipping peaks whose window crosses missing data.
  peak_timing_signed_h            the same lags, signed (positive = simulated peak late)
  peak_mape                       NeuralHydrology's mean_absolute_percentage_peak_error, on the
                                  same observed peaks (simulation read at the observed peak hour)
  missed_peaks                    NeuralHydrology's missed_peaks: observed peaks above the 80th
                                  percentile (>= 30 steps apart) with no simulated peak above the
                                  simulation's own 80th percentile within +/-12 h
  n_peaks                         number of peaks entering peak_timing_h

Evaluation only; writes <output_root>/full_series/ (per-fold CSV, combined CSV, summary).

    python -m scripts.analysis.full_series_eval --config configs/aligned/phase1_runB_v2.yaml --folds 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal

from common.alignment import assert_same_alignment
from common.config import add_common_args, load_config, resolve
from common.metrics import compute_kge, compute_nse, kge_components, peak_flow_bias
from common.utils import get_device, setup_logging
from data.dataset import load_dataset_config, load_scalers, make_loader, resolve_static_spec
from data.folds import domain_stations, load_folds
from data.hourly_windows import load_cache
from data.sources import build_eval_set
from models.mtslstm import build_model

DAY = 24
WINDOW = 12       # +/- hours; NeuralHydrology's default for '1h', Gauch et al. (2021) "one day"


def collect(model, loader, device, y_mean, y_std):
    """station -> {target_hour: 24 simulated hourly values for that calendar day}."""
    out = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = {k: v.to(device) for k, v in batch["x"].items() if k != "S"}
            pred = model(x, batch["x"]["S"].to(device))["H_seq"][:, -DAY:]
            pred = pred.float().cpu().numpy().reshape(len(batch["stations"]), DAY) * y_std + y_mean
            for s, t, p in zip(batch["stations"], batch["hours"].numpy(), pred):
                out.setdefault(s, {})[int(t)] = p
    return out


def _valid(obs: np.ndarray, sim: np.ndarray):
    """Drop hours where either series is missing, keeping each kept hour's original index.

    find_peaks is not NaN-safe, so NeuralHydrology removes invalid steps before searching and
    then rejects any peak whose +/-window span is not contiguous in time. The original index
    plays the role of its datetime check here.
    """
    keep = np.isfinite(obs) & np.isfinite(sim)
    return obs[keep], sim[keep], np.flatnonzero(keep)


def _contiguous(t: np.ndarray, idx: int, n: int) -> bool:
    return idx - WINDOW >= 0 and idx + WINDOW < n and t[idx + WINDOW] - t[idx - WINDOW] == 2 * WINDOW


def peak_metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    o, s, t = _valid(obs, sim)
    if o.size < 1000:
        return {}
    res = {}
    # Gauch et al. (2021), Appendix A / NeuralHydrology mean_peak_timing.
    peaks, _ = signal.find_peaks(o, distance=100, prominence=np.std(o))
    lags = []
    for idx in peaks:
        if not _contiguous(t, idx, o.size):
            continue
        if s[idx] > s[idx - 1] and s[idx] > s[idx + 1]:
            j = idx
        else:
            j = idx - WINDOW + int(np.argmax(s[idx - WINDOW: idx + WINDOW + 1]))
        lags.append(j - idx)
    res["n_peaks"] = len(lags)
    res["peak_timing_h"] = float(np.mean(np.abs(lags))) if lags else float("nan")
    res["peak_timing_signed_h"] = float(np.mean(lags)) if lags else float("nan")
    # NeuralHydrology mean_absolute_percentage_peak_error, same observed peaks.
    res["peak_mape"] = (float(np.sum(np.abs((s[peaks] - o[peaks]) / o[peaks])) / peaks.size * 100)
                        if peaks.size and np.all(o[peaks] != 0) else float("nan"))
    # NeuralHydrology missed_peaks (percentile 80, distance 30).
    po, _ = signal.find_peaks(o, distance=30, height=np.percentile(o, 80))
    ps, _ = signal.find_peaks(s, distance=30, height=np.percentile(s, 80))
    if po.size:
        missed = sum(1 for idx in po if _contiguous(t, idx, o.size)
                     and not np.any(np.abs(ps - idx) <= WINDOW))
        res["missed_peaks"] = missed / po.size
    return res


def series_metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    ok = np.isfinite(obs) & np.isfinite(sim)
    if ok.sum() < 1000 or np.std(obs[ok]) == 0:
        return {}
    o, s = obs[ok], sim[ok]
    kge, r, a, b = kge_components(o, s)
    m = {"n_hours": int(ok.sum()), "kge_full": float(kge), "r": float(r), "alpha": float(a),
         "beta": float(b), "nse": float(compute_nse(o, s)), "fhv_pct": float(peak_flow_bias(o, s))}
    m.update(peak_metrics(np.where(ok, obs, np.nan), np.where(ok, sim, np.nan)))
    return m


def main() -> None:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--folds", default=None)
    ap.add_argument("--pretrain-dir", default=None, help="root holding fold*/pretrain (M0)")
    ap.add_argument("--max-stations", type=int, default=0, help="testing only")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    # Every validation day, not the 6,144-day random cap the transfer scores use.
    cfg = load_config(args.config, list(args.set or []) + ["eval.max_batches_per_station=0"])
    run_dir = resolve(cfg.output_root)
    pre_dir = Path(args.pretrain_dir) if args.pretrain_dir else run_dir
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "full_series"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "full_series.log")
    device = get_device()
    scalers = load_scalers(cfg.data.root)
    y_mean, y_std = float(scalers["y_mean"]), float(scalers["y_std"])
    dyn_size = len(load_dataset_config(cfg.data.root)["dyn_features"])
    _, _, static_names = resolve_static_spec(cfg.data.root, cfg.data.get("static_exclude"),
                                             cfg.data.get("onehot_static"))
    cache = load_cache(cfg.data.cache_dir, stride=int(cfg.data.get("stride", 24)))
    row = {n: i for i, n in enumerate(cache["stations"])}
    folds_tab = load_folds(resolve(cfg.folds.file))
    folds = [int(f) for f in args.folds.split(",")] if args.folds else list(range(int(cfg.folds.n_folds)))

    for fold in folds:
        target_csv = out_dir / f"full_series_fold{fold}.csv"
        if target_csv.exists():
            logger.info("fold %d already done -- skipped", fold)
            continue
        ckpts = {"M0": pre_dir / f"fold{fold}" / "pretrain" / "best_model.pth",
                 "M1": run_dir / f"fold{fold}" / "transfer" / "best_transfer_model.pth"}
        _, target = domain_stations(folds_tab, fold)
        if args.max_stations:
            target = set(sorted(target)[: args.max_stations])
        ds = build_eval_set(cfg, target, "validation", logger=logger)
        loader = make_loader(ds, num_workers=int(cfg.get_path("transfer.num_workers", 4)),
                             pin_memory=device.type == "cuda")
        got = {}
        for tag, ck in ckpts.items():
            assert_same_alignment(ck, cfg, logger=logger)
            model = build_model(cfg, dyn_input_size=dyn_size, static_input_size=len(static_names)).to(device)
            model.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
            logger.info("fold %d %s: predicting full validation period", fold, tag)
            got[tag] = collect(model, loader, device, y_mean, y_std)

        rows = []
        for s in sorted(set(got["M0"]) & set(got["M1"])):
            k = row[s]
            days = sorted(set(got["M0"][s]) & set(got["M1"][s]))
            if len(days) < 100:
                continue
            t0, t1 = days[0] - DAY + 1, days[-1] + 1
            n = t1 - t0
            obs = np.asarray(cache["forcing"][k, t0:t1, 3], dtype=np.float64)
            covered = np.zeros(n, dtype=bool)
            sims = {tag: np.full(n, np.nan) for tag in ("M0", "M1")}
            for t in days:
                a = t - DAY + 1 - t0
                covered[a:a + DAY] = True
                for tag in ("M0", "M1"):
                    sims[tag][a:a + DAY] = got[tag][s][t]
            obs = np.where(covered, obs, np.nan)
            rec = {"station_id": s, "fold": fold, "n_days": len(days)}
            for tag in ("M0", "M1"):
                for key, v in series_metrics(obs, sims[tag]).items():
                    rec[f"{tag}_{key}"] = v
            rows.append(rec)
        pd.DataFrame(rows).to_csv(target_csv, index=False)
        logger.info("fold %d: %d gauges written", fold, len(rows))

    parts = sorted(out_dir.glob("full_series_fold*.csv"))
    table = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    table.to_csv(out_dir / "full_series.csv", index=False)
    summ = {"n_gauges": int(len(table)), "folds": [p.stem for p in parts]}
    for key in ("kge_full", "r", "alpha", "beta", "nse", "fhv_pct", "peak_timing_h",
                "peak_timing_signed_h", "peak_mape", "missed_peaks"):
        if f"M0_{key}" in table and f"M1_{key}" in table:
            d = (table[f"M1_{key}"] - table[f"M0_{key}"]).dropna()
            summ[key] = {"M0": float(table[f"M0_{key}"].median()),
                         "M1": float(table[f"M1_{key}"].median()),
                         "paired_median_M1_minus_M0": float(d.median()) if len(d) else None,
                         "n": int(len(d))}
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1))
    logger.info(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
