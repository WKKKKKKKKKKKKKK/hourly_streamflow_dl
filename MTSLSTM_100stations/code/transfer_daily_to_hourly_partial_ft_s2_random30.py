from __future__ import annotations

import argparse
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
OUT_DIR = ROOT / "MTSLSTM_100stations" / "outputs" / "transfer_daily_to_hourly_partial_ft_s2_random30"

DYNAMIC_VARS = ["Rainf", "Tair", "PotEvap"]
TARGET_VAR = "Streamflow"
LOOKBACK_H = 168
LOOKBACK_D = 365
FREQ = 24

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
    parser.add_argument("--num-workers", type=int, default=0)
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


def train_epoch(
    model: sMTSLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    daily_loss_fn: StationwiseNSELoss,
    agg_loss_fn: StationwiseNSELoss,
    agg_loss_weight: float,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
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

        loss_daily = daily_loss_fn(pred_daily, y_daily, stations)
        loss_agg = agg_loss_fn(pred_hourly_agg, y_daily, stations)
        loss = loss_daily + agg_loss_weight * loss_agg
        loss.backward()
        optimizer.step()

        batch_n = H.size(0)
        total += float(loss.item()) * batch_n
        total_n += batch_n
    return total / max(total_n, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    station_ids = load_station_ids()
    static_df = load_static_df()
    with SCALER_PATH.open("rb") as fp:
        source_scalers = pickle.load(fp)
    full_ds = load_hourly_dataset(station_ids)
    datasets, station_stds, _ = prepare_splits(full_ds, static_df, source_scalers)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(
        f"Hourly-window samples train={len(datasets['train'])} val={len(datasets['val'])} test={len(datasets['test'])}",
        flush=True,
    )

    model = create_model(device)
    set_trainable_parameters(model)
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
        train_loss = train_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            daily_loss_fn=daily_loss_fn,
            agg_loss_fn=agg_loss_fn,
            agg_loss_weight=args.agg_loss_weight,
            device=device,
        )
        val_hourly = evaluate_hourly_per_station(model, loaders["val"], source_scalers, device, station_ids)
        val_summary = summarize_metrics(val_hourly, "val")
        val_kge = float(val_summary["median_kge"])
        val_nse = float(val_summary["median_nse"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_hourly_kge": val_kge, "val_hourly_nse": val_nse})
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_hourly_kge={val_kge:.6f} val_hourly_nse={val_nse:.6f}",
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
    per_station.to_csv(OUT_DIR / "per_station_hourly_metrics.csv", index=False)

    summary_df = pd.DataFrame(split_summaries)
    summary_df.to_csv(OUT_DIR / "summary_hourly_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)

    best_model_path = OUT_DIR / "best_transfer_model.pth"
    torch.save(model.state_dict(), best_model_path)

    lines = [
        "Daily-branch transfer learning affecting hourly outputs",
        "",
        f"Source pretrained model: {MODEL_PATH}",
        f"Best epoch by hourly val KGE: {best_epoch}",
        f"Learning rate: {args.lr}",
        f"Weight decay: {args.weight_decay}",
        f"Aggregate-hourly daily loss weight: {args.agg_loss_weight}",
        "",
        "Trainable modules: lstm_daily, transfer_h, transfer_c, head_daily, head_hourly",
        "Objective: target daily-branch loss + target aggregated-hourly daily loss",
        "Final metrics below are HOURLY KGE/NSE.",
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
