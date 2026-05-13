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
OUT_DIR = (
    ROOT
    / "MTSLSTM_100stations"
    / "outputs"
    / "head_only_daily_head_fc_transfer_s2_random30"
    / "symbolic_prior_sw0.05"
)

DYNAMIC_VARS = ["Rainf", "Tair", "PotEvap"]
TARGET_VAR = "Streamflow"
LOOKBACK_D = 365

ALPHA = 0.8
BETA = 0.2
TAU = 0.2
SHARPNESS = 0.05

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
    parser.add_argument("--sym-loss-weights", nargs="+", type=float, default=[0.05])
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
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
    x_mean = np.asarray(source_scalers["x_dyn_mean"].sel(dynamic_forcing=DYNAMIC_VARS).values, dtype=np.float32)
    x_std = np.asarray(source_scalers["x_dyn_std"].sel(dynamic_forcing=DYNAMIC_VARS).values, dtype=np.float32)
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


class HeadFeatureDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        log_priors: torch.Tensor,
        stations: list[str],
    ):
        self.features = features.float()
        self.targets = targets.float()
        self.log_priors = log_priors.float()
        self.stations = [str(station) for station in stations]

    def __len__(self) -> int:
        return int(self.targets.numel())

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx], self.log_priors[idx], self.stations[idx]


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

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, stations) -> torch.Tensor:
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


def daily_features(model: sMTSLSTM_daily_hourly, x_d: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = x_d.shape
    x_s_d = x_s.unsqueeze(1).repeat(1, seq_len, 1)
    x_daily = torch.cat([x_d, x_s_d], dim=2)
    out_daily, _ = model.lstm_daily(x_daily)
    return out_daily[:, -1, :]


def freeze_backbone(model: sMTSLSTM_daily_hourly) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("head_daily")


def create_model(device: torch.device, train_head_only: bool) -> sMTSLSTM_daily_hourly:
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
    if train_head_only:
        freeze_backbone(model)
    return model


def build_scaler_tensors(source_scalers: dict, device: torch.device) -> dict[str, object]:
    x_dyn_mean = torch.tensor(
        source_scalers["x_dyn_mean"].sel(dynamic_forcing=DYNAMIC_VARS).values.astype(np.float32),
        device=device,
    ).view(1, 1, len(DYNAMIC_VARS))
    x_dyn_std = torch.tensor(
        source_scalers["x_dyn_std"].sel(dynamic_forcing=DYNAMIC_VARS).values.astype(np.float32),
        device=device,
    ).view(1, 1, len(DYNAMIC_VARS))
    x_st_mean = torch.tensor(source_scalers["x_st_mean"].values.astype(np.float32), device=device).view(1, -1)
    x_st_std = torch.tensor(source_scalers["x_st_std"].values.astype(np.float32), device=device).view(1, -1)
    y_mean = torch.tensor(float(source_scalers["y_mean"]), device=device, dtype=torch.float32)
    y_std = torch.tensor(float(source_scalers["y_std"]), device=device, dtype=torch.float32)
    static_cols = list(source_scalers["x_st_mean"].index)
    static_index = {name: i for i, name in enumerate(static_cols)}
    return {
        "x_dyn_mean": x_dyn_mean,
        "x_dyn_std": x_dyn_std,
        "x_st_mean": x_st_mean,
        "x_st_std": x_st_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "static_index": static_index,
    }


def safe_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.clamp(x, min=0.0))


def safe_divide(num: torch.Tensor, denom: torch.Tensor, min_abs: float = 1e-4) -> torch.Tensor:
    sign = torch.where(denom >= 0, torch.ones_like(denom), -torch.ones_like(denom))
    denom_safe = sign * torch.clamp(denom.abs(), min=min_abs)
    return num / denom_safe


def compute_symbolic_log_prior(
    source_daily_std: torch.Tensor,
    D_batch: torch.Tensor,
    S_batch: torch.Tensor,
    scaler_tensors: dict[str, object],
    station_q75_log: dict[str, float],
    stations,
) -> torch.Tensor:
    D_raw = D_batch[:, :, : len(DYNAMIC_VARS)] * scaler_tensors["x_dyn_std"] + scaler_tensors["x_dyn_mean"]
    S_raw = S_batch * scaler_tensors["x_st_std"] + scaler_tensors["x_st_mean"]

    rain_sum_90 = D_raw[:, -90:, 0].sum(dim=1)
    pet_sum_90 = D_raw[:, -90:, 2].sum(dim=1)
    rain_pet_logratio_90 = torch.log((rain_sum_90 + 1e-6) / (pet_sum_90 + 1e-6))

    idx = scaler_tensors["static_index"]
    slope_pct = S_raw[:, idx["SLOPE_PCT"]]
    p_mean = S_raw[:, idx["p_mean"]]
    for_pc_use = S_raw[:, idx["for_pc_use"]]
    permave = S_raw[:, idx["PERMAVE"]]

    global_corr = (slope_pct + (rain_pet_logratio_90 - p_mean)) * ((0.5582391 - for_pc_use) * 0.06458572)
    denom = torch.cos(permave.pow(2))
    event_corr = safe_divide(torch.full_like(denom, -0.070106834), denom, min_abs=1e-3)

    source_daily_raw = source_daily_std * scaler_tensors["y_std"] + scaler_tensors["y_mean"]
    log_src = safe_log1p(source_daily_raw)
    q75 = torch.tensor(
        [float(station_q75_log.get(str(stn), 0.0)) for stn in stations],
        device=source_daily_std.device,
        dtype=torch.float32,
    )
    gate = torch.sigmoid((log_src - (q75 + TAU)) / SHARPNESS)
    return log_src + ALPHA * global_corr + BETA * gate * event_corr


def compute_source_train_q75_daily(
    source_model: sMTSLSTM_daily_hourly,
    loader: DataLoader,
    source_scalers: dict,
    device: torch.device,
) -> dict[str, float]:
    preds_by_station: dict[str, list[float]] = {}
    y_mean = float(source_scalers["y_mean"])
    y_std = float(source_scalers["y_std"])
    source_model.eval()
    with torch.no_grad():
        for x_dict, _, stations in loader:
            x_d = x_dict["D"].to(device)
            x_s = x_dict["S"].to(device)
            preds = daily_forward(source_model, x_d, x_s).detach().cpu().numpy().reshape(-1) * y_std + y_mean
            for i, station_id in enumerate(stations):
                preds_by_station.setdefault(str(station_id), []).append(float(max(preds[i], 0.0)))
    return {
        station_id: float(np.quantile(np.log1p(np.asarray(vals, dtype=np.float64)), 0.75))
        for station_id, vals in preds_by_station.items()
    }


def extract_head_features_and_priors(
    source_model: sMTSLSTM_daily_hourly,
    loader: DataLoader,
    scaler_tensors: dict[str, object],
    station_q75_log: dict[str, float],
    device: torch.device,
) -> HeadFeatureDataset:
    feature_chunks = []
    target_chunks = []
    prior_chunks = []
    station_values: list[str] = []

    source_model.eval()
    with torch.no_grad():
        for x_dict, y, stations in loader:
            x_d = x_dict["D"].to(device)
            x_s = x_dict["S"].to(device)
            features = daily_features(source_model, x_d, x_s)
            source_daily = source_model.head_daily(features).view(-1)
            log_prior = compute_symbolic_log_prior(
                source_daily_std=source_daily,
                D_batch=x_d,
                S_batch=x_s,
                scaler_tensors=scaler_tensors,
                station_q75_log=station_q75_log,
                stations=stations,
            )
            feature_chunks.append(features.detach().cpu())
            target_chunks.append(y.view(-1).detach().cpu())
            prior_chunks.append(log_prior.detach().cpu())
            station_values.extend(str(station) for station in stations)

    return HeadFeatureDataset(
        features=torch.cat(feature_chunks, dim=0),
        targets=torch.cat(target_chunks, dim=0),
        log_priors=torch.cat(prior_chunks, dim=0),
        stations=station_values,
    )


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


def evaluate_feature_per_station(
    head: nn.Linear,
    dataset: HeadFeatureDataset,
    source_scalers: dict,
    device: torch.device,
    expected_stations: list[str],
    batch_size: int,
) -> pd.DataFrame:
    head.eval()
    preds_by_station: dict[str, list[float]] = {}
    trues_by_station: dict[str, list[float]] = {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for features, y, _, stations in loader:
            pred = head(features.to(device)).view(-1).detach().cpu().numpy()
            true = y.numpy().reshape(-1)
            for i, station_id in enumerate(stations):
                station_id = str(station_id)
                preds_by_station.setdefault(station_id, []).append(float(pred[i]))
                trues_by_station.setdefault(station_id, []).append(float(true[i]))

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


@dataclass
class TrialResult:
    lr: float
    weight_decay: float
    sym_loss_weight: float
    best_epoch: int
    best_val_kge: float
    best_val_nse: float
    model_path: str


def train_one_trial(
    trial_idx: int,
    lr: float,
    weight_decay: float,
    sym_loss_weight: float,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_dataset: HeadFeatureDataset,
    station_ids: list[str],
    source_scalers: dict,
    station_std: dict[str, float],
    scaler_tensors: dict[str, object],
    source_head_state: dict[str, torch.Tensor],
    device: torch.device,
) -> TrialResult:
    head = nn.Linear(64, 1).to(device)
    head.load_state_dict(source_head_state)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = DailyNSELoss(station_std=station_std)

    best_val_kge = -math.inf
    best_epoch = 0
    best_state = None
    best_val_nse = float("nan")
    wait = 0

    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        total_daily = 0.0
        total_sym = 0.0
        total_n = 0
        for features, y, log_prior, stations in train_loader:
            features = features.to(device)
            y = y.to(device).view(-1)
            log_prior = log_prior.to(device).view(-1)
            optimizer.zero_grad()
            pred = head(features).view(-1)
            pred_raw = pred * scaler_tensors["y_std"] + scaler_tensors["y_mean"]
            log_pred = safe_log1p(pred_raw)
            loss_daily = criterion(pred, y, stations)
            loss_sym = torch.nn.functional.smooth_l1_loss(log_pred, log_prior)
            loss = loss_daily + sym_loss_weight * loss_sym
            loss.backward()
            optimizer.step()

            batch_n = features.size(0)
            total_loss += float(loss.item()) * batch_n
            total_daily += float(loss_daily.item()) * batch_n
            total_sym += float(loss_sym.item()) * batch_n
            total_n += batch_n

        train_loss = total_loss / max(total_n, 1)
        train_daily = total_daily / max(total_n, 1)
        train_sym = total_sym / max(total_n, 1)
        val_metrics = evaluate_feature_per_station(head, val_dataset, source_scalers, device, station_ids, args.batch_size)
        val_summary = summarize_metrics(val_metrics, "val")
        val_kge = float(val_summary["median_kge"])
        val_nse = float(val_summary["median_nse"])
        print(
            f"[trial {trial_idx}] epoch={epoch} lr={lr} wd={weight_decay} sw={sym_loss_weight} "
            f"train_loss={train_loss:.6f} daily={train_daily:.6f} sym={train_sym:.6f} "
            f"val_kge={val_kge:.6f} val_nse={val_nse:.6f}",
            flush=True,
        )

        if val_kge > best_val_kge:
            best_val_kge = val_kge
            best_val_nse = val_nse
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No best state recorded during training.")

    trial_dir = Path(args.out_dir) / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    model_path = trial_dir / f"trial_{trial_idx:02d}_lr{lr:g}_wd{weight_decay:g}_sw{sym_loss_weight:g}.pth"
    full_model = create_model(device, train_head_only=True)
    full_model.head_daily.load_state_dict(best_state)
    torch.save(full_model.state_dict(), model_path)
    return TrialResult(
        lr=lr,
        weight_decay=weight_decay,
        sym_loss_weight=sym_loss_weight,
        best_epoch=best_epoch,
        best_val_kge=best_val_kge,
        best_val_nse=best_val_nse,
        model_path=str(model_path),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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

    scaler_tensors = build_scaler_tensors(source_scalers, device)
    station_std = build_station_std(split_data["train"])
    source_model = create_model(device, train_head_only=False)
    source_model.eval()
    for param in source_model.parameters():
        param.requires_grad = False
    station_q75_log = compute_source_train_q75_daily(source_model, train_eval_loader, source_scalers, device)
    source_head_state = {k: v.detach().clone() for k, v in source_model.head_daily.state_dict().items()}

    feature_datasets = {
        "train": extract_head_features_and_priors(
            source_model=source_model,
            loader=train_eval_loader,
            scaler_tensors=scaler_tensors,
            station_q75_log=station_q75_log,
            device=device,
        ),
        "val": extract_head_features_and_priors(
            source_model=source_model,
            loader=val_loader,
            scaler_tensors=scaler_tensors,
            station_q75_log=station_q75_log,
            device=device,
        ),
        "test": extract_head_features_and_priors(
            source_model=source_model,
            loader=test_loader,
            scaler_tensors=scaler_tensors,
            station_q75_log=station_q75_log,
            device=device,
        ),
    }
    feature_loaders = {
        "train": DataLoader(feature_datasets["train"], batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(feature_datasets["val"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(feature_datasets["test"], batch_size=args.batch_size, shuffle=False),
    }
    print(
        "Cached daily-head features "
        f"train={len(feature_datasets['train'])} val={len(feature_datasets['val'])} test={len(feature_datasets['test'])}",
        flush=True,
    )

    trial_results = []
    for trial_idx, (lr, weight_decay, sym_loss_weight) in enumerate(
        itertools.product(args.lrs, args.weight_decays, args.sym_loss_weights),
        start=1,
    ):
        trial_results.append(
            train_one_trial(
                trial_idx=trial_idx,
                lr=lr,
                weight_decay=weight_decay,
                sym_loss_weight=sym_loss_weight,
                args=args,
                train_loader=feature_loaders["train"],
                val_dataset=feature_datasets["val"],
                station_ids=station_ids,
                source_scalers=source_scalers,
                station_std=station_std,
                scaler_tensors=scaler_tensors,
                source_head_state=source_head_state,
                device=device,
            )
        )

    trial_df = pd.DataFrame([vars(result) for result in trial_results]).sort_values(
        ["best_val_kge", "best_val_nse"], ascending=[False, False]
    )
    trial_df.to_csv(out_dir / "trial_summary.csv", index=False)

    best_trial = trial_results[int(trial_df.index[0])]
    best_model = create_model(device, train_head_only=True)
    best_model.load_state_dict(torch.load(best_trial.model_path, map_location=device))
    best_head = best_model.head_daily

    per_station_frames = []
    split_summaries = []
    for split_name in ["train", "val", "test"]:
        metrics = evaluate_feature_per_station(
            best_head,
            feature_datasets[split_name],
            source_scalers,
            device,
            station_ids,
            args.batch_size,
        )
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
        split_summaries.append(
            summarize_metrics(
                metrics.rename(
                    columns={
                        f"{split_name}_samples": "samples",
                        f"{split_name}_score_status": "score_status",
                        f"{split_name}_exclusion_reason": "exclusion_reason",
                        f"{split_name}_nse": "nse",
                        f"{split_name}_kge": "kge",
                    }
                ),
                split_name,
            )
        )

    per_station = pd.read_csv(SELECTED_STATIONS_CSV, dtype={"station_id": str})
    for frame in per_station_frames:
        per_station = per_station.merge(frame, on="station_id", how="left")
    per_station.to_csv(out_dir / "per_station_metrics.csv", index=False)

    summary_df = pd.DataFrame(split_summaries)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)

    best_model_path = out_dir / "best_transfer_model.pth"
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
        "symbolic_prior": {
            "alpha": ALPHA,
            "beta": BETA,
            "tau": TAU,
            "sharpness": SHARPNESS,
            "global_equation": "(SLOPE_PCT + (rain_pet_logratio_90 - p_mean)) * ((0.5582391 - for_pc_use) * 0.06458572)",
            "event_equation": "-0.070106834 / cos(square(PERMAVE))",
            "baseline_for_prior": "frozen source MTSLSTM daily prediction",
            "gate_threshold": "station-specific q75 of source train daily log-flow predictions",
        },
        "selection_metric": "val_daily_kge",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lines = [
        "Daily head-only transfer learning on S2 random 30 stations with hybrid symbolic prior",
        "",
        f"Source pretrained model: {MODEL_PATH}",
        (
            "Best transfer trial: "
            f"lr={best_trial.lr}, weight_decay={best_trial.weight_decay}, "
            f"sym_loss_weight={best_trial.sym_loss_weight}, best_epoch={best_trial.best_epoch}"
        ),
        f"Best validation median KGE: {best_trial.best_val_kge:.6f}",
        f"Best validation median NSE: {best_trial.best_val_nse:.6f}",
        "",
        "Frozen parameters: lstm_daily, lstm_hourly, transfer_h, transfer_c, head_hourly",
        "Trainable parameters: head_daily",
        "Objective: target daily head NSE loss + symbolic hybrid prior loss",
        "Final metrics below are DAILY KGE/NSE.",
        "",
        "Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"{row.split},{row.n_total_stations},{row.n_valid_stations},{row.n_excluded_stations},"
            f"{row.median_kge:.6f},{row.median_nse:.6f}"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
