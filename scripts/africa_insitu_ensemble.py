"""Ensemble the five in-situ African models the way eval.africa reports.

`africa_insitu_summary` averages each fold's per-basin KGE. That is not the same thing
as the ensemble: `eval.africa` averages the five models' DAILY PREDICTIONS first and
scores once. Averaging metrics and averaging predictions give different numbers -- the
ensemble is usually better, because independent errors partly cancel -- so comparing an
average-of-metrics against eval.africa's average-of-predictions would not be like for
like.

This loads the five fine-tuned checkpoints, averages their predictions per basin-day,
and scores that. M0 is included on the same basin-days so the pair is exact: the five
pretrained checkpoints ensembled the same way.

    python -m scripts.africa_insitu_ensemble
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from common.config import add_common_args, load_config, resolve
from common.metrics import compute_nse, kge_components
from common.utils import get_device, setup_logging
from data.africa import (
    AfricaWindowDataset,
    apply_onehot,
    build_static_matrix,
    load_hourly_forcing,
    load_observed_daily,
)
from data.dataset import load_dataset_config, load_scalers, make_loader, resolve_static_spec
from models.losses import daily_aggregate_prediction
from models.mtslstm import build_model

DAILY_WINDOW = 24
DEFAULT_FORCING = (
    "/ibex/user/kongw0a/era5_land_africa_forcing/"
    "era5_land_africa_hourly_forcing_penman.nc"
)


@torch.no_grad()
def predict(model, loader, device, y_mean: float, y_std: float) -> pd.DataFrame:
    """One row per (basin, date) with the daily prediction in mm/d."""
    model.eval()
    frames = []
    for batch in loader:
        x = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
        out = model({"D": x["D"], "H": x["H"]}, x["S"])
        pred = daily_aggregate_prediction(out, DAILY_WINDOW).float().cpu().numpy()
        frames.append(pd.DataFrame({
            "station_id": batch["stations"],
            "date": batch["dates"],
            "sim": (pred * y_std + y_mean) * DAILY_WINDOW,
            "obs": batch["y_daily_obs"].numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def score(frame: pd.DataFrame, column: str, min_days: int = 100) -> pd.DataFrame:
    rows = []
    for station, group in frame.groupby("station_id"):
        o = group["obs"].to_numpy(np.float64)
        s = group[column].to_numpy(np.float64)
        keep = np.isfinite(o) & np.isfinite(s)
        if keep.sum() < min_days or np.nanstd(o[keep]) == 0:
            continue
        kge, r, alpha, beta = kge_components(o[keep], s[keep])
        rows.append({"station_id": station, "n_days": int(keep.sum()), "kge": kge,
                     "nse": compute_nse(o[keep], s[keep]),
                     "kge_r": r, "kge_alpha": alpha, "kge_beta": beta})
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame) -> dict:
    valid = frame.loc[np.isfinite(frame["kge"])]
    return {"n_basins": int(len(valid)),
            "median_kge": float(valid["kge"].median()),
            "median_nse": float(valid["nse"].median()),
            "median_r": float(valid["kge_r"].median()),
            "median_alpha": float(valid["kge_alpha"].median()),
            "median_beta": float(valid["kge_beta"].median()),
            "frac_kge_gt_0": float((valid["kge"] > 0).mean())}


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(
        description="Ensemble the in-situ African folds by averaging predictions."))
    parser.add_argument("--insitu-glob", default="outputs/africa_insitu_fold")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--forcing", default=DEFAULT_FORCING)
    parser.add_argument("--basins", default="africa/africa_basins.gpkg")
    parser.add_argument("--out-dir", default="outputs/africa_insitu_summary")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "ensemble.log")
    device = get_device()

    scalers = load_scalers(cfg.data.root)
    y_mean, y_std = float(scalers["y_mean"]), float(scalers["y_std"])
    dyn_size = len(load_dataset_config(cfg.data.root)["dyn_features"])
    names_all = list(load_dataset_config(cfg.data.root)["static_features"])
    static_keep, onehot_specs, static_names = resolve_static_spec(
        cfg.data.root, cfg.data.get("static_exclude"), cfg.data.get("onehot_static"))

    basins = gpd.read_file(resolve(args.basins))
    station_ids = basins["station_id"].astype(str).tolist()
    forcing, forcing_times = load_hourly_forcing(args.forcing, station_ids, scalers, logger=logger)
    observed, observed_dates = load_observed_daily(station_ids, logger=logger)
    full, _ = build_static_matrix(station_ids, names_all, scalers, logger=logger)
    static = apply_onehot(full, static_keep, onehot_specs).astype(np.float32)

    valid_ds = AfricaWindowDataset(
        forcing=forcing, forcing_times=forcing_times, static=static, observed=observed,
        observed_dates=observed_dates, station_ids=station_ids,
        lookback_hourly=int(cfg.data.lookback_hourly),
        lookback_daily=int(cfg.data.get("lookback_daily", 365)),
        chunk_size=int(cfg.data.get("chunk_size", 512)),
        split="validation", train_frac=0.7, logger=logger)
    loader = make_loader(valid_ds, num_workers=int(cfg.get_path("transfer.num_workers", 4)),
                         pin_memory=device.type == "cuda")

    folds = [int(f) for f in args.folds.split(",") if f.strip()]
    per_model = {"M0": [], "M1": []}
    for fold in folds:
        m0 = resolve(cfg.output_root) / f"fold{fold}" / "pretrain" / "best_model.pth"
        m1 = Path(f"{args.insitu_glob}{fold}") / "best_africa_model.pth"
        for tag, path in (("M0", m0), ("M1", m1)):
            if not path.exists():
                raise SystemExit(f"fold {fold} {tag}: {path} missing")
            model = build_model(cfg, dyn_input_size=dyn_size,
                                static_input_size=len(static_names)).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            logger.info("fold %d %s: predicting", fold, tag)
            frame = predict(model, loader, device, y_mean, y_std)
            per_model[tag].append(frame.rename(columns={"sim": f"sim_f{fold}"}))

    results = {}
    merged_all = None
    for tag in ("M0", "M1"):
        merged = per_model[tag][0]
        for extra in per_model[tag][1:]:
            merged = merged.merge(extra.drop(columns=["obs"]), on=["station_id", "date"])
        sim_cols = [c for c in merged.columns if c.startswith("sim_f")]
        if len(sim_cols) != len(folds):
            raise SystemExit(f"{tag}: expected {len(folds)} prediction columns, got {len(sim_cols)}")
        merged["ensemble"] = merged[sim_cols].mean(axis=1)
        scored = score(merged, "ensemble")
        results[tag] = summarise(scored)
        scored.to_csv(out_dir / f"ensemble_per_basin_{tag}.csv", index=False)
        merged[["station_id", "date", "obs", "ensemble"]].to_csv(
            out_dir / f"ensemble_series_{tag}.csv.gz", index=False, compression="gzip")
        merged_all = merged if tag == "M0" else merged_all
        logger.info("%s ensemble: median KGE %.4f | r %.3f alpha %.3f beta %.3f | %d basins",
                    tag, results[tag]["median_kge"], results[tag]["median_r"],
                    results[tag]["median_alpha"], results[tag]["median_beta"],
                    results[tag]["n_basins"])

    a = pd.read_csv(out_dir / "ensemble_per_basin_M0.csv")
    b = pd.read_csv(out_dir / "ensemble_per_basin_M1.csv")
    paired = a.merge(b, on="station_id", suffixes=("_M0", "_M1"))
    delta = paired["kge_M1"] - paired["kge_M0"]
    from scipy.stats import wilcoxon

    p = float(wilcoxon(paired["kge_M1"], paired["kge_M0"]).pvalue)
    logger.info("ENSEMBLE paired over %d basins: median dKGE %+.4f | %.1f%% improved | p=%.2e",
                len(paired), float(delta.median()), 100 * float((delta > 0).mean()), p)
    paired.to_csv(out_dir / "ensemble_paired.csv", index=False)

    # State the comparison against the average-of-metrics explicitly, since the two are
    # easy to conflate and this file exists precisely because they differ.
    fold_avg = Path(out_dir / "summary.json")
    note = ""
    if fold_avg.exists():
        prev = json.loads(fold_avg.read_text())["aggregate"]
        note = (f"average-of-fold-metrics gave M1 {prev['M1_kge']['mean']:+.4f}; "
                f"the prediction ensemble gives {results['M1']['median_kge']:+.4f} "
                f"(difference {results['M1']['median_kge'] - prev['M1_kge']['mean']:+.4f})")
        logger.info(note)

    (out_dir / "ensemble_summary.json").write_text(json.dumps(
        {"folds": folds, "M0": results["M0"], "M1": results["M1"],
         "paired": {"n_basins": int(len(paired)),
                    "median_delta_kge": float(delta.median()),
                    "frac_improved": float((delta > 0).mean()), "wilcoxon_p": p},
         "note_vs_metric_average": note}, indent=2))


if __name__ == "__main__":
    main()
