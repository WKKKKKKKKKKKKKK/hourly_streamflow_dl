from __future__ import annotations

import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader


ROOT = Path("/home/kongw0a/hourly_streamflow_dl")
RAW_TIMESERIES_DIR = Path(
    "/mnt/datawaha/hyex/atr/gscad_database/raw/CAMELS/CAMELSH/timeseries/Data/CAMELSH/timeseries"
)
SELECTED_STATIONS_CSV = ROOT / "MTSLSTM_100stations" / "outputs" / "s2_random30_idx2_eval" / "selected_stations.csv"
STATIC_PATH = ROOT / "MTSLSTM_100stations" / "metadata" / "static_h_topo_priority27.csv"
OUT_DIR = ROOT / "MTSLSTM_100stations" / "outputs" / "baseline_lstm_daily_s2_random30_tuning"

TRAIN_START = "1990-10-01"
TRAIN_END = "2003-09-30"
VAL_START = "2003-10-01"
VAL_END = "2008-09-30"
TEST_START = "2008-10-01"
TEST_END = "2015-09-30"

DYNAMIC_VARS = ["Rainf", "Tair", "PotEvap"]
TARGET_VAR = "Streamflow"

BASELINE_CODE_DIR = ROOT / "BaselineLSTM" / "code"
if str(BASELINE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_CODE_DIR))

from Modelzoo import LSTM  # noqa: E402
from losses import NSELoss  # noqa: E402
from loder import LSTMDataset, calculate_scalers, handle_extremes, standardize_data  # noqa: E402
from trainer import train_model  # noqa: E402


@dataclass(frozen=True)
class RunSpec:
    idx: int
    lr: float
    dropout: float
    hidden_size: int
    batch_size: int
    lookback: int
    epochs: int
    patience: int = 10
    num_layers: int = 1
    num_workers: int = 4
    loss: str = "nse_loss"

    @property
    def tag(self) -> str:
        return (
            f"idx{self.idx}_lr{self.lr}_bs{self.batch_size}_lb{self.lookback}"
            f"_hs{self.hidden_size}_do{self.dropout}_loss{self.loss}"
        )


GRID = [
    RunSpec(idx=1, lr=5e-4, dropout=0.4, hidden_size=256, batch_size=256, lookback=365, epochs=55),
    RunSpec(idx=2, lr=5e-4, dropout=0.4, hidden_size=256, batch_size=256, lookback=180, epochs=55),
    RunSpec(idx=3, lr=5e-4, dropout=0.4, hidden_size=256, batch_size=512, lookback=365, epochs=55),
    RunSpec(idx=4, lr=5e-4, dropout=0.4, hidden_size=256, batch_size=512, lookback=180, epochs=55),
    RunSpec(idx=5, lr=1e-4, dropout=0.4, hidden_size=256, batch_size=256, lookback=365, epochs=55),
    RunSpec(idx=6, lr=1e-4, dropout=0.4, hidden_size=256, batch_size=256, lookback=180, epochs=55),
    RunSpec(idx=7, lr=1e-4, dropout=0.4, hidden_size=256, batch_size=512, lookback=365, epochs=55),
    RunSpec(idx=8, lr=1e-4, dropout=0.4, hidden_size=256, batch_size=512, lookback=180, epochs=55),
    RunSpec(idx=9, lr=1e-3, dropout=0.2, hidden_size=128, batch_size=128, lookback=90, epochs=55),
    RunSpec(idx=10, lr=1e-3, dropout=0.2, hidden_size=256, batch_size=128, lookback=180, epochs=55),
    RunSpec(idx=11, lr=1e-3, dropout=0.4, hidden_size=256, batch_size=128, lookback=365, epochs=55),
    RunSpec(idx=12, lr=5e-4, dropout=0.2, hidden_size=128, batch_size=128, lookback=90, epochs=55),
    RunSpec(idx=13, lr=5e-4, dropout=0.2, hidden_size=512, batch_size=128, lookback=180, epochs=55),
    RunSpec(idx=14, lr=5e-4, dropout=0.6, hidden_size=512, batch_size=128, lookback=365, epochs=55),
    RunSpec(idx=15, lr=1e-4, dropout=0.2, hidden_size=512, batch_size=256, lookback=90, epochs=55),
    RunSpec(idx=16, lr=1e-4, dropout=0.6, hidden_size=512, batch_size=256, lookback=180, epochs=55),
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    if obs.size < 2:
        return float("nan")
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return float("nan")
    return float(1 - np.sum((sim - obs) ** 2) / denom)


def compute_kge(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    if obs.size < 2:
        return float("nan")
    mean_obs = np.mean(obs)
    std_obs = np.std(obs)
    if std_obs == 0 or mean_obs == 0:
        return float("nan")
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / std_obs
    beta = np.mean(sim) / mean_obs
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def load_station_ids() -> list[str]:
    df = pd.read_csv(SELECTED_STATIONS_CSV, dtype={"station_id": str})
    return df["station_id"].astype(str).str.strip().tolist()


def load_static_df(station_ids: list[str]) -> pd.DataFrame:
    df = pd.read_csv(STATIC_PATH)
    df["station_id"] = df["STAID"].astype(str).str.zfill(8)
    static_cols = [c for c in df.columns if c not in {"STAID", "station_id"}]
    out = df.set_index("station_id")[static_cols].copy()
    out = out.loc[station_ids]
    if out.isna().any().any():
        na_cols = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Static features contain NaNs for selected stations: {na_cols}")
    return out.astype("float32")


def build_daily_dataset(station_ids: list[str]) -> xr.Dataset:
    data_vars: dict[str, xr.DataArray] = {}
    all_features = DYNAMIC_VARS + [TARGET_VAR]

    for station_id in station_ids:
        with xr.open_dataset(RAW_TIMESERIES_DIR / f"{station_id}.nc") as ds:
            frame = ds[all_features].to_dataframe()

        frame.index = pd.to_datetime(frame.index)
        daily = frame.resample("D").mean()
        daily_index = pd.DatetimeIndex(daily.index.to_numpy(), name="time")
        values = daily[all_features].to_numpy(dtype="float32")
        data_vars[station_id] = xr.DataArray(
            values,
            coords={"time": daily_index, "dynamic_forcing": all_features},
            dims=("time", "dynamic_forcing"),
        )

    return xr.Dataset(data_vars)


def build_standardized_splits(
    station_ids: list[str],
    static_df: pd.DataFrame,
    out_dir: Path,
):
    daily_all = build_daily_dataset(station_ids)
    daily_all = handle_extremes(daily_all, min_streamflow=0.0, max_streamflow=1000.0)

    (out_dir / "daily_selected_stn_data.nc").unlink(missing_ok=True)
    daily_all.to_netcdf(out_dir / "daily_selected_stn_data.nc")

    dyn = daily_all.sel(dynamic_forcing=DYNAMIC_VARS)
    target = daily_all.sel(dynamic_forcing=TARGET_VAR)

    train_dyn = dyn.sel(time=slice(TRAIN_START, TRAIN_END))
    train_target = target.sel(time=slice(TRAIN_START, TRAIN_END))
    val_dyn = dyn.sel(time=slice(VAL_START, VAL_END))
    val_target = target.sel(time=slice(VAL_START, VAL_END))
    test_dyn = dyn.sel(time=slice(TEST_START, TEST_END))
    test_target = target.sel(time=slice(TEST_START, TEST_END))

    scalers = calculate_scalers(train_dyn, static_df, train_target)
    train_dyn_std, train_static_std, train_y_std = standardize_data(train_dyn, static_df, train_target, scalers)
    val_dyn_std, val_static_std, val_y_std = standardize_data(val_dyn, static_df, val_target, scalers)
    test_dyn_std, test_static_std, test_y_std = standardize_data(test_dyn, static_df, test_target, scalers)

    station_y_std = {}
    all_finite = []
    for stn in train_y_std.data_vars:
        vals = np.asarray(train_y_std[stn].values, dtype="float64").ravel()
        finite = vals[np.isfinite(vals)]
        if finite.size >= 2:
            station_y_std[str(stn)] = float(np.std(finite))
            all_finite.append(finite)

    global_std = float(np.std(np.concatenate(all_finite))) if all_finite else 1.0
    if (not np.isfinite(global_std)) or global_std <= 0:
        global_std = 1.0
    for stn in train_y_std.data_vars:
        stn_key = str(stn)
        station_y_std[stn_key] = float(station_y_std.get(stn_key, global_std) or global_std)

    scalers["station_y_std"] = station_y_std

    return {
        "train_dyn_std": train_dyn_std,
        "train_y_std": train_y_std,
        "val_dyn_std": val_dyn_std,
        "val_y_std": val_y_std,
        "test_dyn_std": test_dyn_std,
        "test_y_std": test_y_std,
        "train_static_std": train_static_std,
        "val_static_std": val_static_std,
        "test_static_std": test_static_std,
        "scalers": scalers,
    }


def make_loader(dyn_std, y_std, static_std, lookback: int, split: str, batch_size: int, shuffle: bool, num_workers: int):
    start, end = {
        "train": (TRAIN_START, TRAIN_END),
        "val": (VAL_START, VAL_END),
        "test": (TEST_START, TEST_END),
    }[split]
    dataset = LSTMDataset(
        dyn_std,
        y_std,
        static_std,
        lookback=lookback,
        start_date=start,
        end_date=end,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def evaluate_per_station(model: torch.nn.Module, loader: DataLoader, scalers: dict, device: torch.device) -> pd.DataFrame:
    model.eval()
    preds_by_station: dict[str, list[float]] = {}
    trues_by_station: dict[str, list[float]] = {}

    with torch.no_grad():
        for x_dyn_batch, x_static_batch, y_batch, stn_batch in loader:
            x_dyn_batch = x_dyn_batch.to(device)
            x_static_batch = x_static_batch.to(device)
            outputs = model((x_dyn_batch, x_static_batch))
            preds = outputs.detach().cpu().numpy().reshape(-1)
            trues = y_batch.detach().cpu().numpy().reshape(-1)
            for i, stn in enumerate(stn_batch):
                preds_by_station.setdefault(str(stn), []).append(float(preds[i]))
                trues_by_station.setdefault(str(stn), []).append(float(trues[i]))

    y_mean = float(scalers["y_mean"])
    y_std = float(scalers["y_std"])

    rows = []
    for station_id in sorted(preds_by_station):
        sim = np.asarray(preds_by_station[station_id], dtype="float64") * y_std + y_mean
        obs = np.asarray(trues_by_station[station_id], dtype="float64") * y_std + y_mean
        nse = compute_nse(obs, sim)
        kge = compute_kge(obs, sim)
        row = {
            "station_id": station_id,
            "samples": int(len(obs)),
            "score_status": "ok",
            "exclusion_reason": "",
            "nse": float("nan"),
            "kge": float("nan"),
        }
        if not np.isfinite(nse) or not np.isfinite(kge):
            row["score_status"] = "excluded"
            row["exclusion_reason"] = "metric_nonfinite"
        else:
            row["nse"] = float(nse)
            row["kge"] = float(kge)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame, split: str) -> dict[str, float | int | str]:
    valid = metrics.loc[metrics["score_status"].eq("ok")].copy()
    return {
        "split": split,
        "n_total_stations": int(len(metrics)),
        "n_valid_stations": int(len(valid)),
        "n_excluded_stations": int(len(metrics) - len(valid)),
        "median_kge": float(valid["kge"].median()) if len(valid) else float("nan"),
        "median_nse": float(valid["nse"].median()) if len(valid) else float("nan"),
        "mean_kge": float(valid["kge"].mean()) if len(valid) else float("nan"),
        "mean_nse": float(valid["nse"].mean()) if len(valid) else float("nan"),
        "neg_kge_stations": int((valid["kge"] < 0).sum()) if len(valid) else 0,
        "neg_nse_stations": int((valid["nse"] < 0).sum()) if len(valid) else 0,
    }


def save_run_outputs(
    run_dir: Path,
    spec: RunSpec,
    history: dict,
    summaries: list[dict],
    per_station_frames: list[pd.DataFrame],
    scalers: dict,
    selected_meta: pd.DataFrame,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "idx": spec.idx,
                "lr": spec.lr,
                "dropout": spec.dropout,
                "hidden_size": spec.hidden_size,
                "batch_size": spec.batch_size,
                "lookback_days": spec.lookback,
                "epochs": spec.epochs,
                "patience": spec.patience,
                "selection_metric": "val_median_nse",
                "aggregation": {
                    "Rainf": "daily_mean_of_hourly",
                    "Tair": "daily_mean_of_hourly",
                    "PotEvap": "daily_mean_of_hourly",
                    "Streamflow": "daily_mean_of_hourly",
                },
                "splits": {
                    "train": [TRAIN_START, TRAIN_END],
                    "val": [VAL_START, VAL_END],
                    "test": [TEST_START, TEST_END],
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with open(run_dir / "scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    hist_df = pd.DataFrame(
        {
            "epoch": list(range(1, len(history["train_loss"]) + 1)),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
        }
    )
    hist_df.to_csv(run_dir / "training_history.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(run_dir / "summary_metrics.csv", index=False)

    merged = selected_meta.copy()
    for split_name, metrics in zip(["train", "val", "test"], per_station_frames):
        merged = merged.merge(
            metrics.rename(
                columns={
                    "samples": f"{split_name}_samples",
                    "score_status": f"{split_name}_score_status",
                    "exclusion_reason": f"{split_name}_exclusion_reason",
                    "nse": f"{split_name}_nse",
                    "kge": f"{split_name}_kge",
                }
            ),
            on="station_id",
            how="left",
        )
    merged.to_csv(run_dir / "per_station_metrics.csv", index=False)

    lines = [
        f"BaselineLSTM daily aggregation tuning run `{spec.tag}` on S2 random 30 stations",
        "",
        "Metric selection follows BaselineLSTM sweep style: choose best hyperparameters by validation median NSE.",
        "",
        f"Lookback days: {spec.lookback}",
        f"Batch size: {spec.batch_size}",
        f"Hidden size: {spec.hidden_size}",
        f"Dropout: {spec.dropout}",
        f"Learning rate: {spec.lr}",
        f"Epoch cap: {spec.epochs}",
        f"Early stopping patience: {spec.patience}",
        "",
        "Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse,neg_kge_stations,neg_nse_stations",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"{row.split},{row.n_total_stations},{row.n_valid_stations},{row.n_excluded_stations},"
            f"{row.median_kge:.6f},{row.median_nse:.6f},{row.mean_kge:.6f},{row.mean_nse:.6f},"
            f"{row.neg_kge_stations},{row.neg_nse_stations}"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_training(
    spec: RunSpec,
    prepared: dict,
    selected_meta: pd.DataFrame,
    device: torch.device,
) -> dict:
    run_dir = OUT_DIR / "runs" / spec.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, train_loader = make_loader(
        prepared["train_dyn_std"], prepared["train_y_std"], prepared["train_static_std"],
        lookback=spec.lookback, split="train", batch_size=spec.batch_size, shuffle=True, num_workers=spec.num_workers,
    )
    _, train_eval_loader = make_loader(
        prepared["train_dyn_std"], prepared["train_y_std"], prepared["train_static_std"],
        lookback=spec.lookback, split="train", batch_size=spec.batch_size, shuffle=False, num_workers=spec.num_workers,
    )
    _, val_loader = make_loader(
        prepared["val_dyn_std"], prepared["val_y_std"], prepared["val_static_std"],
        lookback=spec.lookback, split="val", batch_size=spec.batch_size, shuffle=False, num_workers=spec.num_workers,
    )
    _, test_loader = make_loader(
        prepared["test_dyn_std"], prepared["test_y_std"], prepared["test_static_std"],
        lookback=spec.lookback, split="test", batch_size=spec.batch_size, shuffle=False, num_workers=spec.num_workers,
    )

    model = LSTM(
        input_size=len(DYNAMIC_VARS) + prepared["train_static_std"].shape[1],
        hidden_size=spec.hidden_size,
        num_layers=spec.num_layers,
        dropout=spec.dropout,
        output_size=1,
    ).to(device)

    criterion = NSELoss(station_std=prepared["scalers"].get("station_y_std", {}), eps=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.lr)

    best_model_path = run_dir / "best_model.pth"
    checkpoint_path = run_dir / "checkpoint.pth"

    start_time = time.time()
    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=spec.epochs,
        patience=spec.patience,
        best_model_path=str(best_model_path),
        early_stopping=True,
        lr_schedule=None,
        checkpoint_path=str(checkpoint_path),
        resume=False,
        save_every=1,
    )
    train_seconds = float(time.time() - start_time)

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    torch.save(model.state_dict(), run_dir / "model.pth")

    train_metrics = evaluate_per_station(model, train_eval_loader, prepared["scalers"], device)
    val_metrics = evaluate_per_station(model, val_loader, prepared["scalers"], device)
    test_metrics = evaluate_per_station(model, test_loader, prepared["scalers"], device)

    summaries = [
        summarize_metrics(train_metrics, "train"),
        summarize_metrics(val_metrics, "val"),
        summarize_metrics(test_metrics, "test"),
    ]
    save_run_outputs(
        run_dir=run_dir,
        spec=spec,
        history=history,
        summaries=summaries,
        per_station_frames=[train_metrics, val_metrics, test_metrics],
        scalers=prepared["scalers"],
        selected_meta=selected_meta,
    )

    val_summary = summaries[1]
    test_summary = summaries[2]
    result = {
        "idx": spec.idx,
        "tag": spec.tag,
        "lr": spec.lr,
        "dropout": spec.dropout,
        "hidden_size": spec.hidden_size,
        "batch_size": spec.batch_size,
        "lookback_days": spec.lookback,
        "epochs_cap": spec.epochs,
        "epochs_ran": len(history["train_loss"]),
        "train_seconds": train_seconds,
        "selection_metric": float(val_summary["median_nse"]),
        "val_median_nse": float(val_summary["median_nse"]),
        "val_median_kge": float(val_summary["median_kge"]),
        "test_median_nse": float(test_summary["median_nse"]),
        "test_median_kge": float(test_summary["median_kge"]),
        "run_dir": str(run_dir),
    }
    return result


def write_final_summary(out_dir: Path, tuning_df: pd.DataFrame, best_row: pd.Series) -> None:
    lines = [
        "BaselineLSTM daily aggregation tuning on S2 random 30 stations",
        "",
        "Training style mirrors the BaselineLSTM repo:",
        "- ordinary LSTM with dynamic and static features concatenated at each time step",
        "- NSE loss",
        "- early stopping on validation loss",
        "- hyperparameter selection by validation median NSE (same as the BaselineLSTM sweep metric)",
        "",
        "Daily aggregation used here:",
        "- Rainf: daily mean of hourly values",
        "- Tair: daily mean of hourly values",
        "- PotEvap: daily mean of hourly values",
        "- Streamflow: daily mean of hourly values",
        "",
        f"Best run: {best_row['tag']}",
        f"Best val median NSE: {best_row['val_median_nse']:.6f}",
        f"Best val median KGE: {best_row['val_median_kge']:.6f}",
        f"Best test median NSE: {best_row['test_median_nse']:.6f}",
        f"Best test median KGE: {best_row['test_median_kge']:.6f}",
        "",
        "All tuned runs:",
    ]
    for row in tuning_df.itertuples(index=False):
        lines.append(
            f"- {row.tag}: val NSE={row.val_median_nse:.6f}, val KGE={row.val_median_kge:.6f}, "
            f"test NSE={row.test_median_nse:.6f}, test KGE={row.test_median_kge:.6f}"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    station_ids = load_station_ids()
    static_df = load_static_df(station_ids)
    selected_meta = pd.read_csv(SELECTED_STATIONS_CSV, dtype={"station_id": str})

    prepared = build_standardized_splits(station_ids, static_df, OUT_DIR)

    tuning_rows = []
    for spec in GRID:
        print(f"=== Running {spec.tag} ===", flush=True)
        row = run_training(spec, prepared, selected_meta, device)
        tuning_rows.append(row)
        pd.DataFrame(tuning_rows).sort_values(["selection_metric", "val_median_kge"], ascending=False).to_csv(
            OUT_DIR / "tuning_summary_partial.csv", index=False
        )
        print(
            f"Completed {spec.tag}: val NSE={row['val_median_nse']:.6f}, "
            f"val KGE={row['val_median_kge']:.6f}, test NSE={row['test_median_nse']:.6f}, "
            f"test KGE={row['test_median_kge']:.6f}",
            flush=True,
        )

    tuning_df = pd.DataFrame(tuning_rows).sort_values(["selection_metric", "val_median_kge"], ascending=False).reset_index(drop=True)
    tuning_df.to_csv(OUT_DIR / "tuning_summary.csv", index=False)
    best_row = tuning_df.iloc[0]
    (OUT_DIR / "best_run.json").write_text(best_row.to_json(indent=2) + "\n", encoding="utf-8")
    write_final_summary(OUT_DIR, tuning_df, best_row)
    print("Best run:", best_row.to_dict(), flush=True)


if __name__ == "__main__":
    main()
