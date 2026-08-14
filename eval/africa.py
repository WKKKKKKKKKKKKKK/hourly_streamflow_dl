"""Phase I Steps 4-5 -- evaluate the fold models on African DAILY streamflow.

Africa is out of sample for every fold: no African basin appears anywhere in the
hourly dataset the models were trained on. So each fold gives an independent
prediction, and their mean is the ensemble.

The model still runs hourly; its daily value is the mean of its last 24 hourly
outputs, which by construction cover the calendar day (see ``data/africa.py``).
Everything is then scored at daily resolution against three references, on
exactly the same station-days so the comparison is fair:

  * observed daily ``q_mm``            -- the truth
  * ERA5-Land daily runoff             -- the physical baseline (Step 5)
  * the continent-holdout PUB baseline -- prior work (Step 4), read from its
    per-basin result file rather than re-run

    python -m eval.africa --checkpoint-kind transfer
    python -m eval.africa --self-test        # synthetic data, no download needed
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from common.config import add_common_args, load_config, resolve
from common.metrics import StationAccumulator, compute_kge, compute_nse, summarize
from common.utils import get_device, setup_logging
from data.africa import (
    AfricaDailyDataset,
    apply_onehot,
    build_static_matrix,
    load_hourly_forcing,
    load_observed_daily,
)
from data.dataset import load_dataset_config, load_scalers, resolve_static_spec
from models.losses import daily_aggregate_prediction
from models.mtslstm import build_model

PUB_BASELINE = Path(
    "/ibex/project/c2266/abbaa0a/results/regionalization/20250630/"
    "gscad_continent_lstm/47515076_africa/2000385_test_performance.csv"
)
DAILY_WINDOW = 24
CHECKPOINTS = {"pretrain": ("pretrain", "best_model.pth"), "transfer": ("transfer", "best_transfer_model.pth")}


@torch.no_grad()
def predict_daily(model, dataset, device, y_mean: float, y_std: float, logger=None) -> pd.DataFrame:
    """One row per (station, date) with the model's daily prediction in mm/d.

    The model is trained on hourly targets in mm/h, so averaging its last 24 hourly
    outputs gives mm/h -- NOT the mm/d the daily observations are in. The two differ
    by exactly the window length. Verified against seven stations present in both
    the hourly cache and the daily file: on matching dates,
    ``daily q_mm == 24 * mean(hourly q_mm/h)`` to within 0.4% (ratio 0.996 over
    4,500-7,100 common days each). Omitting the factor made the model under-predict
    ~24-fold and turned Africa's median KGE into -0.41 while the run reported
    success, so the magnitudes are checked below rather than trusted.
    """
    model.eval()
    frames = []
    for i in range(len(dataset)):
        item = dataset[i]
        x = {key: value.to(device) for key, value in item["x"].items()}
        outputs = model({"D": x["D"], "H": x["H"]}, x["S"])
        # The forcing has no gaps here, so all 24 slots are present: a plain mean.
        pred = daily_aggregate_prediction(outputs, DAILY_WINDOW).float().cpu().numpy()
        frames.append(
            pd.DataFrame(
                {
                    "station_id": item["stations"],
                    "date": item["dates"],
                    # mm/h -> mm/d
                    "sim": (pred * y_std + y_mean) * DAILY_WINDOW,
                    "obs": item["y_daily_obs"].numpy(),
                }
            )
        )
        if logger and (i + 1) % 200 == 0:
            logger.info("  predicted %d/%d chunks", i + 1, len(dataset))
    out = pd.concat(frames, ignore_index=True)

    # A units slip is invisible in KGE (it just looks like a bad model), so state
    # the ratio outright and refuse to report a run that is off by a window length.
    sim_mean = float(np.nanmean(out["sim"]))
    obs_mean = float(np.nanmean(out["obs"]))
    if logger:
        logger.info("magnitude check: sim mean %.4f mm/d | obs mean %.4f mm/d | ratio %.2f",
                    sim_mean, obs_mean, obs_mean / sim_mean if sim_mean else float("nan"))
    if sim_mean > 0 and not 0.05 <= obs_mean / sim_mean <= 20:
        raise SystemExit(
            f"observed/simulated mean ratio is {obs_mean / sim_mean:.1f} -- that is a unit "
            f"error, not a model error (a factor near {DAILY_WINDOW} means the mm/h -> mm/d "
            "conversion is wrong). Refusing to report."
        )
    return out


def score_per_station(frame: pd.DataFrame, sim_column: str, min_days: int = 100) -> pd.DataFrame:
    rows = []
    for station, group in frame.groupby("station_id"):
        both = group[["obs", sim_column]].dropna()
        row = {
            "station_id": station,
            "source": str(station).split("__")[0],
            "n_days": int(len(both)),
            "kge": float("nan"),
            "nse": float("nan"),
            "score_status": "ok",
        }
        if len(both) < max(2, min_days):
            row["score_status"] = "excluded"
        else:
            obs = both["obs"].to_numpy(dtype=np.float64)
            sim = both[sim_column].to_numpy(dtype=np.float64)
            row["kge"], row["nse"] = compute_kge(obs, sim), compute_nse(obs, sim)
            if not (np.isfinite(row["kge"]) and np.isfinite(row["nse"])):
                row["score_status"] = "excluded"
        rows.append(row)
    return pd.DataFrame(rows)


def load_era5_land_runoff(path: Path, logger=None) -> pd.DataFrame:
    ds = xr.open_dataset(path)
    variable = "runoff" if "runoff" in ds else list(ds.data_vars)[0]
    frame = ds[variable].to_dataframe(name="era5_land").reset_index()
    frame = frame.rename(columns={"station": "station_id"})
    ds.close()
    if logger:
        logger.info("ERA5-Land runoff: %d rows, variable %r", len(frame), variable)
    return frame[["station_id", "date", "era5_land"]]


def load_pub_baseline(logger=None) -> pd.DataFrame | None:
    if not PUB_BASELINE.exists():
        if logger:
            logger.warning("PUB baseline not found at %s", PUB_BASELINE)
        return None
    raw = pd.read_csv(PUB_BASELINE, index_col=0)
    stations = [c for c in raw.columns if c not in ("mean", "median")]
    out = pd.DataFrame({"station_id": stations})
    for metric, name in (("kge", "kge"), ("nse", "nse")):
        if metric in raw.index:
            out[f"pub_{name}"] = raw.loc[metric, stations].to_numpy(dtype=float)
    if logger:
        logger.info("PUB baseline: %d basins, median KGE %.4f", len(out), float(out["pub_kge"].median()))
    return out


def write_synthetic(tmp: Path, station_ids: list[str], n_days: int = 500) -> tuple[Path, Path]:
    """Fake forcing + runoff so the whole path can run without the download."""
    rng = np.random.default_rng(0)
    start = pd.Timestamp("1990-01-01")
    hours = pd.date_range(start, periods=(n_days + 366) * 24, freq="h")
    n = len(station_ids)
    forcing = xr.Dataset(
        {
            "pet": (("station", "time"), np.abs(rng.normal(0.1, 0.05, (n, len(hours)))).astype("float32")),
            "pcp": (("station", "time"), np.abs(rng.normal(0.1, 0.4, (n, len(hours)))).astype("float32")),
            "temp": (("station", "time"), rng.normal(22, 5, (n, len(hours))).astype("float32")),
        },
        coords={"station": station_ids, "time": hours},
    )
    forcing_path = tmp / "era5_land_africa_hourly_forcing.nc"
    forcing.to_netcdf(forcing_path)

    dates = pd.date_range(start, periods=n_days + 366, freq="D")
    runoff = xr.Dataset(
        {"runoff": (("station", "date"), np.abs(rng.normal(0.5, 0.3, (n, len(dates)))).astype("float32"))},
        coords={"station": station_ids, "date": dates},
    )
    runoff_path = tmp / "era5_land_africa_daily_runoff.nc"
    runoff.to_netcdf(runoff_path)
    return forcing_path, runoff_path


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Phase I Steps 4-5: African daily evaluation."))
    parser.add_argument("--basins", default="africa/africa_basins.csv")
    parser.add_argument("--forcing", default="/ibex/user/kongw0a/era5_land_africa_forcing/era5_land_africa_hourly_forcing.nc")
    parser.add_argument("--era5-runoff", default="/ibex/user/kongw0a/era5_land_africa/era5_land_africa_daily_runoff.nc")
    parser.add_argument("--checkpoint-kind", choices=sorted(CHECKPOINTS), default="transfer")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="Default: all folds in the config.")
    parser.add_argument("--out-dir", default="outputs/africa")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--min-days", type=int, default=100)
    parser.add_argument("--self-test", action="store_true",
                        help="Run on synthetic forcing/runoff for a small basin subset; proves the path works.")
    parser.add_argument("--limit-basins", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "africa.log")
    device = get_device()

    basins = pd.read_csv(resolve(args.basins), dtype={"station_id": str})
    station_ids = basins["station_id"].tolist()
    if args.self_test:
        station_ids = station_ids[: args.limit_basins or 12]
    elif args.limit_basins:
        station_ids = station_ids[: args.limit_basins]
    logger.info("African basins: %d | device %s | checkpoints: %s",
                len(station_ids), device, args.checkpoint_kind)

    scalers = load_scalers(cfg.data.root)
    ds_config = load_dataset_config(cfg.data.root)
    all_names = list(ds_config["static_features"])
    static_keep, onehot_specs, static_names = resolve_static_spec(
        cfg.data.root, cfg.data.get("static_exclude"), cfg.data.get("onehot_static")
    )

    static_full, kgz_report = build_static_matrix(station_ids, all_names, scalers, logger=logger)
    static = apply_onehot(static_full, static_keep, onehot_specs)
    kgz_report.to_csv(out_dir / "africa_kgz_report.csv", index=False)
    logger.info("static matrix: %s (model expects %d)", static.shape, len(static_names))

    observed, observed_dates = load_observed_daily(station_ids, logger=logger)

    tmp_dir = None
    forcing_path, runoff_path = Path(args.forcing), Path(args.era5_runoff)
    if args.self_test:
        tmp_dir = tempfile.TemporaryDirectory()
        forcing_path, runoff_path = write_synthetic(Path(tmp_dir.name), station_ids)
        logger.warning("SELF-TEST: synthetic forcing and runoff -- the numbers below are meaningless")
    if not forcing_path.exists():
        raise SystemExit(
            f"forcing not found: {forcing_path}\n"
            "Run scripts.download_era5_land_africa --preset forcing, then "
            "scripts.basin_average_era5_forcing. Use --self-test to exercise this script now."
        )

    forcing, forcing_times = load_hourly_forcing(forcing_path, station_ids, scalers, logger=logger)
    dataset = AfricaDailyDataset(
        forcing=forcing,
        forcing_times=forcing_times,
        static=static,
        observed=observed,
        observed_dates=observed_dates,
        station_ids=station_ids,
        lookback_hourly=int(cfg.data.lookback_hourly),
        chunk_size=args.chunk_size,
        logger=logger,
    )

    folds = args.folds if args.folds is not None else list(range(int(cfg.folds.n_folds)))
    subdir, filename = CHECKPOINTS[args.checkpoint_kind]
    per_fold, fold_scores = {}, []
    for fold in folds:
        path = resolve(cfg.output_root) / f"fold{fold}" / subdir / filename
        if not path.exists():
            logger.warning("fold %d: no checkpoint at %s -- skipped", fold, path)
            continue
        model = build_model(cfg, dyn_input_size=len(ds_config["dyn_features"]),
                            static_input_size=len(static_names)).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        logger.info("fold %d: %s", fold, path)

        frame = predict_daily(model, dataset, device, scalers["y_mean"], scalers["y_std"], logger=logger)
        per_fold[fold] = frame.set_index(["station_id", "date"])["sim"]
        scored = score_per_station(frame, "sim", args.min_days)
        scored["fold"] = fold
        fold_scores.append(scored)
        summary = summarize(scored.assign(samples=scored["n_days"]), f"fold{fold}")
        logger.info("  fold %d daily: median KGE %.4f | median NSE %.4f | %d/%d basins scored",
                    fold, summary["median_kge"], summary["median_nse"],
                    summary["n_valid_stations"], summary["n_stations"])

    if not per_fold:
        raise SystemExit("no fold checkpoints found; run train.pretrain_source / train.transfer_target first")

    combined = pd.concat(fold_scores, ignore_index=True)
    combined.to_csv(out_dir / f"per_fold_daily_{args.checkpoint_kind}.csv", index=False)

    # --- ensemble + the two reference series, on identical station-days --------
    base = pd.DataFrame({f"sim_f{fold}": series for fold, series in sorted(per_fold.items())})
    ensemble = base.mean(axis=1).rename("ensemble").reset_index()
    observed_long = pd.DataFrame(
        {
            "station_id": np.repeat(station_ids, observed.shape[1]),
            "date": np.tile(observed_dates.to_numpy(), len(station_ids)),
            "obs": observed.reshape(-1),
        }
    )
    merged = ensemble.merge(observed_long, on=["station_id", "date"], how="left")

    if runoff_path.exists():
        merged = merged.merge(load_era5_land_runoff(runoff_path, logger=logger),
                              on=["station_id", "date"], how="left")
    else:
        logger.warning("ERA5-Land runoff not found at %s -- skipping the physical baseline", runoff_path)
        merged["era5_land"] = np.nan

    merged.to_csv(out_dir / f"daily_series_{args.checkpoint_kind}.csv.gz", index=False, compression="gzip")

    rows = []
    for label, column in (("MTS-LSTM ensemble", "ensemble"), ("ERA5-Land runoff", "era5_land")):
        if merged[column].notna().sum() == 0:
            continue
        scored = score_per_station(merged, column, args.min_days)
        scored.to_csv(out_dir / f"per_basin_{column}_{args.checkpoint_kind}.csv", index=False)
        valid = scored.loc[scored["score_status"].eq("ok")]
        rows.append({
            "method": label,
            "n_basins_scored": int(len(valid)),
            "median_kge": float(valid["kge"].median()) if len(valid) else float("nan"),
            "median_nse": float(valid["nse"].median()) if len(valid) else float("nan"),
            "frac_kge_gt_0": float((valid["kge"] > 0).mean()) if len(valid) else float("nan"),
        })

    pub = load_pub_baseline(logger=logger)
    if pub is not None:
        pub_here = pub.loc[pub["station_id"].isin(station_ids)]
        rows.append({
            "method": "continent-PUB baseline (prior work)",
            "n_basins_scored": int(pub_here["pub_kge"].notna().sum()),
            "median_kge": float(pub_here["pub_kge"].median()),
            "median_nse": float(pub_here["pub_nse"].median()) if "pub_nse" in pub_here else float("nan"),
            "frac_kge_gt_0": float((pub_here["pub_kge"] > 0).mean()),
        })
        pub_here.to_csv(out_dir / "per_basin_pub_baseline.csv", index=False)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"africa_comparison_{args.checkpoint_kind}.csv", index=False)
    logger.info("\n===== Africa daily comparison (%s checkpoints) =====\n%s",
                args.checkpoint_kind, table.to_string(index=False))
    with open(out_dir / f"africa_summary_{args.checkpoint_kind}.json", "w", encoding="utf-8") as handle:
        json.dump({"folds": sorted(per_fold), "n_basins": len(station_ids),
                   "self_test": bool(args.self_test), "comparison": rows}, handle, indent=2)

    if args.self_test:
        logger.info("SELF-TEST PASSED: the whole Africa path runs; numbers are from synthetic data.")
    if tmp_dir is not None:
        tmp_dir.cleanup()


if __name__ == "__main__":
    main()
