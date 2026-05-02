from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import DataLoader, Dataset

import config
from Modelzoo import sMTSLSTM
from Train import _add_static_station_aliases, compute_kge, compute_nse
from loder import handle_extremes, standardize_data


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
OUT_DIR = ROOT / "MTSLSTM_100stations" / "outputs" / "transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid"

DYNAMIC_VARS = ["Rainf", "Tair", "PotEvap"]
TARGET_VAR = "Streamflow"
LOOKBACK_H = 168
LOOKBACK_D = 365
FREQ = 24

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
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--agg-loss-weight", type=float, default=0.5)
    parser.add_argument("--sym-loss-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    return parser.parse_args()


def load_station_ids() -> list[str]:
    selected = pd.read_csv(SELECTED_STATIONS_CSV, dtype={"station_id": str})
    return selected["station_id"].astype(str).tolist()


def load_hourly_dataset(station_ids: list[str]) -> xr.Dataset:
    station_arrays = {}
    for station_id in station_ids:
        with xr.open_dataset(RAW_TIMESERIES_DIR / f"{station_id}.nc") as ds:
            da = ds[DYNAMIC_VARS + [TARGET_VAR]].to_array(dim="dynamic_forcing").transpose("DateTime", "dynamic_forcing")
            da = da.rename({"DateTime": "time"})
            da = da.assign_coords(dynamic_forcing=DYNAMIC_VARS + [TARGET_VAR])
            da.name = station_id
            station_arrays[station_id] = da.load()
    full_ds = xr.Dataset(station_arrays)
    return handle_extremes(full_ds, min_streamflow=0.0, max_streamflow=1000.0)


def load_static_df() -> pd.DataFrame:
    static_df = pd.read_csv(STATIC_MODEL_INPUT_PATH, index_col=0)
    static_df.index = pd.Index(static_df.index.astype(str).str.strip())
    return static_df


class TransferTargetDataset(Dataset):
    def __init__(
        self,
        dyn_std: xr.Dataset,
        y_std: xr.Dataset,
        static_std: pd.DataFrame,
        lookback_hourly: int,
        lookback_daily: int,
        frequency_factor: int,
        start_date: str,
        end_date: str,
    ):
        self.lookback_hourly = int(lookback_hourly)
        self.lookback_daily = int(lookback_daily)
        self.frequency_factor = int(frequency_factor)
        self.static_std = static_std

        self.samples: list[tuple[str, int]] = []
        self.x_data: dict[str, np.ndarray] = {}
        self.y_data: dict[str, np.ndarray] = {}

        for station_id in [str(s) for s in dyn_std.data_vars]:
            x = dyn_std[station_id].sel(time=slice(start_date, end_date))
            y = y_std[station_id].sel(time=slice(start_date, end_date))
            x = np.asarray(x.transpose("time", "dynamic_forcing").values, dtype=np.float32)
            y = np.asarray(y.values, dtype=np.float32)
            self.x_data[station_id] = x
            self.y_data[station_id] = y

            t_min = max(self.lookback_hourly, self.lookback_daily * self.frequency_factor)
            for t in range(t_min, len(x)):
                x_h = x[t - self.lookback_hourly : t]
                x_d_full = x[t - self.lookback_daily * self.frequency_factor : t]
                y_t = y[t]
                y_d = y[t - self.frequency_factor : t].mean()
                if np.isnan(x_h).any() or np.isnan(x_d_full).any() or np.isnan(y_t) or np.isnan(y_d):
                    continue
                self.samples.append((station_id, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        station_id, t = self.samples[idx]
        x = self.x_data[station_id]
        y = self.y_data[station_id]

        x_h = x[t - self.lookback_hourly : t]
        x_d_full = x[t - self.lookback_daily * self.frequency_factor : t]
        x_d = x_d_full.reshape(self.lookback_daily, self.frequency_factor, -1).mean(axis=1)
        x_s = self.static_std.loc[station_id].to_numpy(dtype=np.float32)
        y_h = y[t]
        y_d = y[t - self.frequency_factor : t].mean()

        return (
            {
                "H": torch.from_numpy(x_h),
                "D": torch.from_numpy(x_d),
                "S": torch.from_numpy(x_s),
            },
            {
                "hourly": torch.tensor([y_h], dtype=torch.float32),
                "daily": torch.tensor([y_d], dtype=torch.float32),
            },
            station_id,
        )


class StationwiseNSELoss(nn.Module):
    def __init__(self, station_std: dict[str, float], eps: float = 1e-6):
        super().__init__()
        self.station_std = station_std
        self.eps = float(eps)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, stations) -> torch.Tensor:
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        grouped: dict[str, list[int]] = {}
        for i, stn in enumerate(stations):
            grouped.setdefault(str(stn), []).append(i)
        losses = []
        for stn, idxs in grouped.items():
            idx = torch.as_tensor(idxs, device=y_pred.device)
            err = y_pred.index_select(0, idx) - y_true.index_select(0, idx)
            mse = (err**2).mean()
            denom = (float(self.station_std.get(stn, 1.0)) + self.eps) ** 2
            losses.append(mse / denom)
        return torch.stack(losses).mean()


def build_station_std(y_std_split: xr.Dataset) -> dict[str, float]:
    station_std = {}
    vals_all = []
    for stn in y_std_split.data_vars:
        vals = np.asarray(y_std_split[stn].values, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size >= 2:
            station_std[str(stn)] = float(np.std(vals))
            vals_all.append(vals)
    global_std = 1.0
    if vals_all:
        global_std = float(np.std(np.concatenate(vals_all)))
        if not np.isfinite(global_std) or global_std <= 0:
            global_std = 1.0
    for stn in y_std_split.data_vars:
        station_std.setdefault(str(stn), global_std)
    return station_std


def prepare_splits(
    full_ds: xr.Dataset,
    static_df: pd.DataFrame,
    source_scalers: dict,
) -> tuple[dict[str, TransferTargetDataset], dict[str, dict[str, float]], pd.DataFrame]:
    dyn = full_ds.sel(dynamic_forcing=DYNAMIC_VARS)
    y = full_ds.sel(dynamic_forcing=TARGET_VAR)
    dyn_std, static_std, y_std = standardize_data(dyn, static_df, y, source_scalers)
    static_std, missing_static = _add_static_station_aliases(static_std, full_ds.data_vars)
    if missing_static:
        preview = ", ".join(missing_static[:10])
        raise KeyError(f"Missing static features for selected stations: {preview}")

    datasets = {}
    station_stds = {}
    for split_name, (start, end) in SPLITS.items():
        y_split = y_std.sel(time=slice(start, end))
        station_stds[split_name] = build_station_std(y_split)
        datasets[split_name] = TransferTargetDataset(
            dyn_std=dyn_std,
            y_std=y_std,
            static_std=static_std,
            lookback_hourly=LOOKBACK_H,
            lookback_daily=LOOKBACK_D,
            frequency_factor=FREQ,
            start_date=start,
            end_date=end,
        )
    return datasets, station_stds, static_std


def create_model(device: torch.device) -> sMTSLSTM:
    model = sMTSLSTM(
        dyn_input_size=config.DYN_INPUT_SIZE,
        static_input_size=config.STATIC_INPUT_SIZE,
        hidden_size_daily=64,
        hidden_size_hourly=64,
        num_layers=1,
        dropout=0.4,
        frequency_factor=FREQ,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model


def set_trainable_parameters(model: sMTSLSTM) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module in [model.lstm_daily, model.transfer_h, model.transfer_c, model.head_daily, model.head_hourly]:
        for param in module.parameters():
            param.requires_grad = True


def evaluate_hourly_per_station(
    model: sMTSLSTM,
    loader: DataLoader,
    source_scalers: dict,
    device: torch.device,
    expected_stations: list[str],
) -> pd.DataFrame:
    model.eval()
    preds_by_station = {}
    trues_by_station = {}
    with torch.no_grad():
        for x_dict, y_dict, stations in loader:
            H = x_dict["H"].to(device)
            D = x_dict["D"].to(device)
            S = x_dict["S"].to(device)
            outputs = model({"H": H, "D": D}, S)
            preds = outputs["H"].detach().cpu().numpy().reshape(-1)
            trues = y_dict["hourly"].detach().cpu().numpy().reshape(-1)
            for i, stn in enumerate(stations):
                stn = str(stn)
                preds_by_station.setdefault(stn, []).append(preds[i])
                trues_by_station.setdefault(stn, []).append(trues[i])

    y_mean = float(source_scalers["y_mean"])
    y_std = float(source_scalers["y_std"])
    rows = []
    for stn in sorted(set(expected_stations)):
        pred_values = preds_by_station.get(stn, [])
        true_values = trues_by_station.get(stn, [])
        row = {"station_id": stn, "samples": len(true_values), "score_status": "ok", "exclusion_reason": "", "nse": np.nan, "kge": np.nan}
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


def summarize_metrics(df: pd.DataFrame, split_name: str) -> dict[str, object]:
    valid = df.loc[df["score_status"].eq("ok")]
    return {
        "split": split_name,
        "n_total_stations": int(len(df)),
        "n_valid_stations": int(len(valid)),
        "n_excluded_stations": int(len(df) - len(valid)),
        "median_kge": float(valid["kge"].median()) if len(valid) else float("nan"),
        "median_nse": float(valid["nse"].median()) if len(valid) else float("nan"),
    }


def build_scaler_tensors(source_scalers: dict, device: torch.device):
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
    scaler_tensors: dict,
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


def compute_source_train_q75(
    source_model: sMTSLSTM,
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
            H = x_dict["H"].to(device)
            D = x_dict["D"].to(device)
            S = x_dict["S"].to(device)
            outputs = source_model({"H": H, "D": D}, S)
            preds = outputs["D"].detach().cpu().numpy().reshape(-1) * y_std + y_mean
            for i, stn in enumerate(stations):
                preds_by_station.setdefault(str(stn), []).append(float(max(preds[i], 0.0)))
    return {
        stn: float(np.quantile(np.log1p(np.asarray(vals, dtype=np.float64)), 0.75))
        for stn, vals in preds_by_station.items()
    }


def train_epoch(
    model: sMTSLSTM,
    source_model: sMTSLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    daily_loss_fn: StationwiseNSELoss,
    agg_loss_fn: StationwiseNSELoss,
    agg_loss_weight: float,
    sym_loss_weight: float,
    scaler_tensors: dict,
    station_q75_log: dict[str, float],
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total = 0.0
    total_daily = 0.0
    total_agg = 0.0
    total_sym = 0.0
    total_n = 0
    for x_dict, y_dict, stations in loader:
        H = x_dict["H"].to(device)
        D = x_dict["D"].to(device)
        S = x_dict["S"].to(device)
        y_daily = y_dict["daily"].to(device).view(-1)

        optimizer.zero_grad()
        outputs = model({"H": H, "D": D}, S)
        pred_daily = outputs["D"].view(-1)
        pred_hourly_agg = outputs["H_seq"][:, -FREQ:].mean(dim=1)

        with torch.no_grad():
            source_outputs = source_model({"H": H, "D": D}, S)
            source_daily = source_outputs["D"].view(-1)
            log_prior = compute_symbolic_log_prior(
                source_daily_std=source_daily,
                D_batch=D,
                S_batch=S,
                scaler_tensors=scaler_tensors,
                station_q75_log=station_q75_log,
                stations=stations,
            )

        pred_daily_raw = pred_daily * scaler_tensors["y_std"] + scaler_tensors["y_mean"]
        log_pred_daily = safe_log1p(pred_daily_raw)

        loss_daily = daily_loss_fn(pred_daily, y_daily, stations)
        loss_agg = agg_loss_fn(pred_hourly_agg, y_daily, stations)
        loss_sym = torch.nn.functional.smooth_l1_loss(log_pred_daily, log_prior)
        loss = loss_daily + agg_loss_weight * loss_agg + sym_loss_weight * loss_sym
        loss.backward()
        optimizer.step()

        batch_n = H.size(0)
        total += float(loss.item()) * batch_n
        total_daily += float(loss_daily.item()) * batch_n
        total_agg += float(loss_agg.item()) * batch_n
        total_sym += float(loss_sym.item()) * batch_n
        total_n += batch_n
    denom = max(total_n, 1)
    return {
        "total_loss": total / denom,
        "daily_loss": total_daily / denom,
        "agg_loss": total_agg / denom,
        "sym_loss": total_sym / denom,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    station_ids = load_station_ids()
    static_df = load_static_df()
    with SCALER_PATH.open("rb") as fp:
        source_scalers = pickle.load(fp)
    full_ds = load_hourly_dataset(station_ids)
    datasets, station_stds, _ = prepare_splits(full_ds, static_df, source_scalers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(
        f"Hourly-window samples train={len(datasets['train'])} val={len(datasets['val'])} test={len(datasets['test'])}",
        flush=True,
    )

    loaders = {
        split: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, ds in datasets.items()
    }
    train_eval_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_model(device)
    set_trainable_parameters(model)
    source_model = create_model(device)
    source_model.eval()
    for param in source_model.parameters():
        param.requires_grad = False

    scaler_tensors = build_scaler_tensors(source_scalers, device)
    station_q75_log = compute_source_train_q75(source_model, train_eval_loader, source_scalers, device)

    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    daily_loss_fn = StationwiseNSELoss(station_stds["train"])
    agg_loss_fn = StationwiseNSELoss(station_stds["train"])

    best_val_kge = -np.inf
    best_epoch = 0
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(
            model=model,
            source_model=source_model,
            loader=loaders["train"],
            optimizer=optimizer,
            daily_loss_fn=daily_loss_fn,
            agg_loss_fn=agg_loss_fn,
            agg_loss_weight=args.agg_loss_weight,
            sym_loss_weight=args.sym_loss_weight,
            scaler_tensors=scaler_tensors,
            station_q75_log=station_q75_log,
            device=device,
        )
        val_hourly = evaluate_hourly_per_station(model, loaders["val"], source_scalers, device, station_ids)
        val_summary = summarize_metrics(val_hourly, "val")
        val_kge = float(val_summary["median_kge"])
        val_nse = float(val_summary["median_nse"])
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": train_stats["total_loss"],
                "train_daily_loss": train_stats["daily_loss"],
                "train_agg_loss": train_stats["agg_loss"],
                "train_sym_loss": train_stats["sym_loss"],
                "val_hourly_kge": val_kge,
                "val_hourly_nse": val_nse,
            }
        )
        print(
            f"epoch={epoch} total={train_stats['total_loss']:.6f} daily={train_stats['daily_loss']:.6f} "
            f"agg={train_stats['agg_loss']:.6f} sym={train_stats['sym_loss']:.6f} "
            f"val_hourly_kge={val_kge:.6f} val_hourly_nse={val_nse:.6f}",
            flush=True,
        )
        if val_kge > best_val_kge:
            best_val_kge = val_kge
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint.")
    model.load_state_dict(best_state)

    split_frames = []
    split_summaries = []
    for split_name in ["train", "val", "test"]:
        metrics = evaluate_hourly_per_station(model, loaders[split_name], source_scalers, device, station_ids)
        metrics = metrics.rename(
            columns={
                "samples": f"{split_name}_samples",
                "score_status": f"{split_name}_score_status",
                "exclusion_reason": f"{split_name}_exclusion_reason",
                "nse": f"{split_name}_nse",
                "kge": f"{split_name}_kge",
            }
        )
        split_frames.append(metrics)
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
    for frame in split_frames:
        per_station = per_station.merge(frame, on="station_id", how="left")
    per_station.to_csv(out_dir / "per_station_hourly_metrics.csv", index=False)

    summary_df = pd.DataFrame(split_summaries)
    summary_df.to_csv(out_dir / "summary_hourly_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    best_model_path = out_dir / "best_transfer_model.pth"
    torch.save(model.state_dict(), best_model_path)

    run_meta = {
        "source_pretrained_model": str(MODEL_PATH),
        "best_epoch_by_val_hourly_kge": best_epoch,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "agg_loss_weight": args.agg_loss_weight,
        "sym_loss_weight": args.sym_loss_weight,
        "symbolic_hybrid": {
            "alpha": ALPHA,
            "beta": BETA,
            "tau": TAU,
            "sharpness": SHARPNESS,
            "global_equation": "(SLOPE_PCT + (rain_pet_logratio_90 - p_mean)) * ((0.5582391 - for_pc_use) * 0.06458572)",
            "event_equation": "-0.070106834 / cos(square(PERMAVE))",
            "baseline_for_prior": "frozen source MTSLSTM daily prediction",
            "gate_threshold": "station-specific q75 of source train daily log-flow predictions",
        },
        "trainable_modules": ["lstm_daily", "transfer_h", "transfer_c", "head_daily", "head_hourly"],
        "selection_metric": "val_hourly_kge",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

    lines = [
        "Daily-branch transfer learning affecting hourly outputs with hybrid symbolic prior",
        "",
        f"Source pretrained model: {MODEL_PATH}",
        f"Best epoch by hourly val KGE: {best_epoch}",
        f"Learning rate: {args.lr}",
        f"Weight decay: {args.weight_decay}",
        f"Aggregate-hourly daily loss weight: {args.agg_loss_weight}",
        f"Symbolic prior loss weight: {args.sym_loss_weight}",
        "",
        "Trainable modules: lstm_daily, transfer_h, transfer_c, head_daily, head_hourly",
        "Objective: target daily-branch loss + target aggregated-hourly daily loss + symbolic hybrid prior loss",
        "Final metrics below are HOURLY KGE/NSE.",
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
