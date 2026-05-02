from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import DataLoader, Dataset

import config
from Modelzoo import sMTSLSTM_daily_hourly
from Train import _add_static_station_aliases, compute_kge, compute_nse


ROOT = Path("/home/kongw0a/hourly_streamflow_dl")
RAW_TIMESERIES_DIR = Path(
    "/mnt/datawaha/hyex/atr/gscad_database/raw/CAMELS/CAMELSH/timeseries/Data/CAMELSH/timeseries"
)
SELECTED_STATIONS_CSV = (
    ROOT / "MTSLSTM_100stations" / "outputs" / "s2_random30_idx2_eval" / "selected_stations.csv"
)
STATIC_MODEL_INPUT_PATH = ROOT / "MTSLSTM_100stations" / "metadata" / "static_h_topo_priority27.csv"
RUN_DIR = (
    ROOT
    / "MTSLSTM_100stations"
    / "training_runs"
    / "20260407_mtslstm_100stations_tuning_topo18_v100"
    / "idx2_bs128_do0.4_hs64_H168_D365"
)
MODEL_PATH = RUN_DIR / "best_model.pth"
SCALER_PATH = RUN_DIR / "scalers.pkl"
OUT_DIR = ROOT / "MTSLSTM_100stations" / "outputs" / "transfer_daily_head_tune_s2_random30"

DYNAMIC_VARS = ["Rainf", "Tair", "PotEvap"]
TARGET_VAR = "Streamflow"
LOOKBACK_D = 365

SPLITS = {
    "train": (config.TRAIN_START, config.TRAIN_END),
    "val": (config.VAL_START, config.VAL_END),
    "test": (config.TEST_START, config.TEST_END),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lookback-daily", type=int, default=LOOKBACK_D)
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 5e-4, 1e-3, 5e-3])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.0, 1e-5, 1e-4])
    return parser.parse_args()


def load_source_scalers() -> dict:
    with SCALER_PATH.open("rb") as fp:
        return pickle.load(fp)


def load_station_ids() -> list[str]:
    selected = pd.read_csv(SELECTED_STATIONS_CSV, dtype={"station_id": str})
    return selected["station_id"].astype(str).tolist()


def load_static_df() -> pd.DataFrame:
    static_df = pd.read_csv(STATIC_MODEL_INPUT_PATH, index_col=0)
    static_df.index = pd.Index(static_df.index.astype(str).str.strip())
    return static_df


def aggregate_station_daily(station_id: str) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    with xr.open_dataset(RAW_TIMESERIES_DIR / f"{station_id}.nc") as ds:
        frame = ds[DYNAMIC_VARS + [TARGET_VAR]].to_dataframe()
    frame.index = pd.to_datetime(frame.index)
    daily = frame.resample("D").mean()
    x = daily[DYNAMIC_VARS].to_numpy(dtype=np.float32)
    y = daily[TARGET_VAR].to_numpy(dtype=np.float32)
    return x, y, daily.index


def standardize_daily_arrays(
    x: np.ndarray,
    y: np.ndarray,
    source_scalers: dict,
) -> tuple[np.ndarray, np.ndarray]:
    x_mean = np.asarray(source_scalers["x_dyn_mean"].values, dtype=np.float32)
    x_std = np.asarray(source_scalers["x_dyn_std"].values, dtype=np.float32)
    y_mean = float(source_scalers["y_mean"])
    y_std = float(source_scalers["y_std"])
    x_stdized = ((x - x_mean) / x_std).astype(np.float32)
    y_stdized = ((y - y_mean) / y_std).astype(np.float32)
    return x_stdized, y_stdized


def build_split_data(
    station_ids: list[str],
    source_scalers: dict,
) -> dict[str, dict[str, dict[str, object]]]:
    split_data: dict[str, dict[str, dict[str, object]]] = {split: {} for split in SPLITS}
    for station_id in station_ids:
        x_daily, y_daily, dates = aggregate_station_daily(station_id)
        x_daily_std, y_daily_std = standardize_daily_arrays(x_daily, y_daily, source_scalers)

        for split_name, (start, end) in SPLITS.items():
            mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            split_data[split_name][station_id] = {
                "x": x_daily_std[mask],
                "y": y_daily_std[mask],
                "dates": dates[mask],
            }
    return split_data


class DailyWindowDataset(Dataset):
    def __init__(
        self,
        split_station_data: dict[str, dict[str, object]],
        static_df_std: pd.DataFrame,
        lookback_daily: int,
    ):
        self.lookback_daily = int(lookback_daily)
        self.static_df_std = static_df_std
        self.samples: list[tuple[str, int]] = []
        self.x_data: dict[str, np.ndarray] = {}
        self.y_data: dict[str, np.ndarray] = {}
        self.dates: dict[str, pd.DatetimeIndex] = {}

        for station_id, payload in split_station_data.items():
            x = np.asarray(payload["x"], dtype=np.float32)
            y = np.asarray(payload["y"], dtype=np.float32)
            dates = payload["dates"]

            if len(x) <= self.lookback_daily:
                continue

            self.x_data[station_id] = x
            self.y_data[station_id] = y
            self.dates[station_id] = dates

            for t in range(self.lookback_daily, len(x)):
                x_win = x[t - self.lookback_daily : t]
                y_t = y[t]
                if np.isnan(x_win).any() or np.isnan(y_t):
                    continue
                self.samples.append((station_id, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        station_id, t = self.samples[idx]
        x_d = self.x_data[station_id][t - self.lookback_daily : t]
        y_t = self.y_data[station_id][t]
        x_s = self.static_df_std.loc[station_id].to_numpy(dtype=np.float32)
        return (
            {
                "D": torch.from_numpy(x_d),
                "S": torch.from_numpy(x_s),
            },
            torch.tensor([y_t], dtype=torch.float32),
            station_id,
        )


def build_station_std(split_station_data: dict[str, dict[str, object]]) -> dict[str, float]:
    station_std: dict[str, float] = {}
    all_vals: list[np.ndarray] = []
    for station_id, payload in split_station_data.items():
        vals = np.asarray(payload["y"], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size >= 2:
            station_std[station_id] = float(np.std(vals))
            all_vals.append(vals)

    global_std = 1.0
    if all_vals:
        global_std = float(np.std(np.concatenate(all_vals)))
        if not np.isfinite(global_std) or global_std <= 0:
            global_std = 1.0

    for station_id in split_station_data:
        if station_id not in station_std or not np.isfinite(station_std[station_id]) or station_std[station_id] <= 0:
            station_std[station_id] = global_std

    return station_std


class DailyNSELoss(nn.Module):
    def __init__(self, station_std: dict[str, float], eps: float = 1e-6):
        super().__init__()
        self.station_std = station_std
        self.eps = float(eps)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, stations: list[str]) -> torch.Tensor:
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        grouped: dict[str, list[int]] = {}
        for idx, station_id in enumerate(stations):
            grouped.setdefault(str(station_id), []).append(idx)
        losses = []
        for station_id, idxs in grouped.items():
            take = torch.as_tensor(idxs, device=y_pred.device)
            err = y_pred.index_select(0, take) - y_true.index_select(0, take)
            mse = (err**2).mean()
            denom = (float(self.station_std[station_id]) + self.eps) ** 2
            losses.append(mse / denom)
        return torch.stack(losses).mean()


def daily_forward(model: sMTSLSTM_daily_hourly, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
    out = model({"D": x_d}, x_s)
    return out["D"].view(-1)


def evaluate_daily_per_station(
    model: sMTSLSTM_daily_hourly,
    loader: DataLoader,
    source_scalers: dict,
    device: torch.device,
    expected_stations: list[str],
) -> pd.DataFrame:
    model.eval()
    preds_by_station: dict[str, list[float]] = {}
    trues_by_station: dict[str, list[float]] = {}

    with torch.no_grad():
        for x_dict, y, stations in loader:
            x_d = x_dict["D"].to(device)
            x_s = x_dict["S"].to(device)
            preds = daily_forward(model, x_d, x_s).detach().cpu().numpy().reshape(-1)
            trues = y.detach().cpu().numpy().reshape(-1)
            for i, station_id in enumerate(stations):
                station_id = str(station_id)
                preds_by_station.setdefault(station_id, []).append(preds[i])
                trues_by_station.setdefault(station_id, []).append(trues[i])

    y_mean = float(source_scalers["y_mean"])
    y_std = float(source_scalers["y_std"])
    rows = []
    for station_id in sorted(set(expected_stations)):
        pred_values = preds_by_station.get(station_id, [])
        true_values = trues_by_station.get(station_id, [])
        row = {
            "station_id": station_id,
            "samples": int(len(true_values)),
            "score_status": "ok",
            "exclusion_reason": "",
            "nse": float("nan"),
            "kge": float("nan"),
        }
        if not true_values:
            row["score_status"] = "excluded"
            row["exclusion_reason"] = "no_valid_windows"
            rows.append(row)
            continue

        sim = np.asarray(pred_values, dtype=np.float64) * y_std + y_mean
        obs = np.asarray(true_values, dtype=np.float64) * y_std + y_mean
        nse = compute_nse(obs, sim)
        kge = compute_kge(obs, sim)
        if not np.isfinite(nse) or not np.isfinite(kge):
            row["score_status"] = "excluded"
            row["exclusion_reason"] = "metric_nonfinite"
        else:
            row["nse"] = float(nse)
            row["kge"] = float(kge)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame, split_name: str) -> dict[str, float | int | str]:
    valid = df.loc[df["score_status"].eq("ok")]
    return {
        "split": split_name,
        "n_total_stations": int(len(df)),
        "n_valid_stations": int(len(valid)),
        "n_excluded_stations": int(len(df) - len(valid)),
        "median_kge": float(valid["kge"].median()) if len(valid) else float("nan"),
        "median_nse": float(valid["nse"].median()) if len(valid) else float("nan"),
    }


def freeze_backbone(model: sMTSLSTM_daily_hourly) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head_daily")


def create_model(device: torch.device) -> sMTSLSTM_daily_hourly:
    model = sMTSLSTM_daily_hourly(
        dyn_input_size=config.DYN_INPUT_SIZE,
        static_input_size=config.STATIC_INPUT_SIZE,
        hidden_size_daily=64,
        hidden_size_hourly=64,
        num_layers=1,
        dropout=0.4,
        frequency_factor=24,
    ).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=True)
    freeze_backbone(model)
    return model


@dataclass
class TrialResult:
    lr: float
    weight_decay: float
    best_epoch: int
    best_val_kge: float
    best_val_nse: float
    model_path: str


def train_one_trial(
    trial_idx: int,
    lr: float,
    weight_decay: float,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    station_ids: list[str],
    source_scalers: dict,
    station_std: dict[str, float],
    device: torch.device,
) -> TrialResult:
    model = create_model(device)
    optimizer = torch.optim.Adam(model.head_daily.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = DailyNSELoss(station_std=station_std)

    best_val_kge = -math.inf
    best_epoch = 0
    best_state = None
    best_val_nse = float("nan")
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x_dict, y, stations in train_loader:
            x_d = x_dict["D"].to(device)
            x_s = x_dict["S"].to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = daily_forward(model, x_d, x_s)
            loss = criterion(pred, y.view(-1), list(stations))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x_d.size(0)

        train_loss = total_loss / max(len(train_loader.dataset), 1)
        val_metrics = evaluate_daily_per_station(model, val_loader, source_scalers, device, station_ids)
        val_summary = summarize_metrics(val_metrics, "val")
        val_kge = float(val_summary["median_kge"])
        val_nse = float(val_summary["median_nse"])
        print(
            f"[trial {trial_idx}] epoch={epoch} lr={lr} wd={weight_decay} "
            f"train_loss={train_loss:.6f} val_kge={val_kge:.6f} val_nse={val_nse:.6f}",
            flush=True,
        )

        if val_kge > best_val_kge:
            best_val_kge = val_kge
            best_val_nse = val_nse
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No best state recorded during training.")

    trial_dir = OUT_DIR / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    model_path = trial_dir / f"trial_{trial_idx:02d}_lr{lr:g}_wd{weight_decay:g}.pth"
    torch.save(best_state, model_path)
    return TrialResult(
        lr=lr,
        weight_decay=weight_decay,
        best_epoch=best_epoch,
        best_val_kge=best_val_kge,
        best_val_nse=best_val_nse,
        model_path=str(model_path),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source_scalers = load_source_scalers()
    station_ids = load_station_ids()
    static_df = load_static_df()
    split_data = build_split_data(station_ids, source_scalers)

    static_std = ((static_df - source_scalers["x_st_mean"]) / source_scalers["x_st_std"]).astype(np.float32)
    static_std, missing_static = _add_static_station_aliases(static_std, station_ids)
    if missing_static:
        preview = ", ".join(missing_static[:10])
        raise KeyError(f"Missing static features for selected stations: {preview}")

    train_dataset = DailyWindowDataset(split_data["train"], static_std, args.lookback_daily)
    val_dataset = DailyWindowDataset(split_data["val"], static_std, args.lookback_daily)
    test_dataset = DailyWindowDataset(split_data["test"], static_std, args.lookback_daily)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(
        f"Daily samples train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}",
        flush=True,
    )

    station_std = build_station_std(split_data["train"])

    trial_results = []
    for trial_idx, (lr, weight_decay) in enumerate(itertools.product(args.lrs, args.weight_decays), start=1):
        trial_results.append(
            train_one_trial(
                trial_idx=trial_idx,
                lr=lr,
                weight_decay=weight_decay,
                args=args,
                train_loader=train_loader,
                val_loader=val_loader,
                station_ids=station_ids,
                source_scalers=source_scalers,
                station_std=station_std,
                device=device,
            )
        )

    trial_df = pd.DataFrame([vars(result) for result in trial_results]).sort_values(
        ["best_val_kge", "best_val_nse"], ascending=[False, False]
    )
    trial_df.to_csv(OUT_DIR / "trial_summary.csv", index=False)

    best_trial = trial_results[int(trial_df.index[0])]
    best_model = create_model(device)
    best_model.load_state_dict(torch.load(best_trial.model_path, map_location=device))

    per_station_frames = []
    split_summaries = []
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        metrics = evaluate_daily_per_station(best_model, loader, source_scalers, device, station_ids)
        metrics = metrics.rename(
            columns={
                "samples": f"{split_name}_samples",
                "score_status": f"{split_name}_score_status",
                "exclusion_reason": f"{split_name}_exclusion_reason",
                "nse": f"{split_name}_nse",
                "kge": f"{split_name}_kge",
            }
        )
        per_station_frames.append(metrics)
        split_summaries.append(summarize_metrics(metrics.rename(columns={
            f"{split_name}_samples": "samples",
            f"{split_name}_score_status": "score_status",
            f"{split_name}_exclusion_reason": "exclusion_reason",
            f"{split_name}_nse": "nse",
            f"{split_name}_kge": "kge",
        }), split_name))

    per_station = pd.read_csv(SELECTED_STATIONS_CSV, dtype={"station_id": str})
    for frame in per_station_frames:
        per_station = per_station.merge(frame, on="station_id", how="left")
    per_station.to_csv(OUT_DIR / "per_station_metrics.csv", index=False)

    summary_df = pd.DataFrame(split_summaries)
    summary_df.to_csv(OUT_DIR / "summary_metrics.csv", index=False)

    best_model_path = OUT_DIR / "best_transfer_model.pth"
    torch.save(best_model.state_dict(), best_model_path)

    metadata = {
        "source_model": str(MODEL_PATH),
        "source_scalers": str(SCALER_PATH),
        "selected_stations": str(SELECTED_STATIONS_CSV),
        "best_trial": vars(best_trial),
        "train_dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "test_dataset_size": len(test_dataset),
        "lookback_daily": args.lookback_daily,
        "daily_aggregation": "24-hour mean for Rainf/Tair/PotEvap/Streamflow to match the pretrained daily branch",
        "frozen_parameters": [
            "lstm_daily",
            "lstm_hourly",
            "transfer_h",
            "transfer_c",
            "head_hourly",
        ],
        "trainable_parameters": ["head_daily"],
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "Daily transfer learning on S2 random 30 stations",
        "",
        f"Source pretrained model: {MODEL_PATH}",
        f"Best transfer trial: lr={best_trial.lr}, weight_decay={best_trial.weight_decay}, best_epoch={best_trial.best_epoch}",
        f"Best validation median KGE: {best_trial.best_val_kge:.6f}",
        f"Best validation median NSE: {best_trial.best_val_nse:.6f}",
        "",
        "Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"{row.split},{row.n_total_stations},{row.n_valid_stations},{row.n_excluded_stations},"
            f"{row.median_kge:.6f},{row.median_nse:.6f}"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
