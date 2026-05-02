from __future__ import annotations

import argparse
import json
import math
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
from Train import compute_kge, compute_nse


ROOT = Path("/home/kongw0a/hourly_streamflow_dl")
RAW_TIMESERIES_DIR = Path(
    "/mnt/datawaha/hyex/atr/gscad_database/raw/CAMELS/CAMELSH/timeseries/Data/CAMELSH/timeseries"
)
SELECTED_STATIONS_CSV = ROOT / "MTSLSTM_100stations" / "outputs" / "s2_random30_idx2_eval" / "selected_stations.csv"
OUT_DIR = ROOT / "MTSLSTM_100stations" / "outputs" / "symtorch_daily_teacher_s2_random30"

DYNAMIC_VARS = ["Rainf", "Tair", "PotEvap"]
TARGET_VAR = "Streamflow"
STATIC_FEATURES = [
    "aridity_index",
    "BFI_AVE",
    "DRAIN_SQKM",
    "SLOPE_PCT",
    "CLAYAVE",
    "SANDAVE",
    "for_pc_use",
    "urb_pc_use",
]
Z_FEATURES = [
    "rain_1",
    "rain_3",
    "rain_7",
    "rain_30",
    "pet_1",
    "pet_7",
    "pet_30",
    "tair_1",
    "tair_7",
    "tair_30",
    "wetness_7",
    "wetness_30",
    "rain_pet_ratio_30",
    "doy_sin",
    "doy_cos",
]

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
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    return parser.parse_args()


def load_selected_static_df() -> pd.DataFrame:
    df = pd.read_csv(SELECTED_STATIONS_CSV, dtype={"station_id": str})
    df["station_id"] = df["station_id"].astype(str).str.strip()
    missing = [col for col in STATIC_FEATURES if col not in df.columns]
    if missing:
        raise KeyError(f"Missing static features in selected stations csv: {missing}")
    static_df = df.set_index("station_id")[STATIC_FEATURES].copy()
    if static_df.isna().any().any():
        na_counts = static_df.isna().sum()
        raise ValueError(f"Static features contain NaNs:\n{na_counts[na_counts > 0]}")
    return static_df


def build_station_feature_frame(station_id: str, static_row: pd.Series) -> pd.DataFrame:
    with xr.open_dataset(RAW_TIMESERIES_DIR / f"{station_id}.nc") as ds:
        frame = ds[DYNAMIC_VARS + [TARGET_VAR]].to_dataframe()

    frame.index = pd.to_datetime(frame.index)
    daily = frame.resample("D").mean()
    daily.columns = [col.lower() for col in daily.columns]
    rain = daily["rainf"]
    tair = daily["tair"]
    pet = daily["potevap"]
    q = daily["streamflow"]

    out = pd.DataFrame(index=daily.index)
    out["station_id"] = station_id
    out["rain_1"] = rain
    out["rain_3"] = rain.rolling(3, min_periods=3).sum()
    out["rain_7"] = rain.rolling(7, min_periods=7).sum()
    out["rain_30"] = rain.rolling(30, min_periods=30).sum()

    out["pet_1"] = pet
    out["pet_7"] = pet.rolling(7, min_periods=7).sum()
    out["pet_30"] = pet.rolling(30, min_periods=30).sum()

    out["tair_1"] = tair
    out["tair_7"] = tair.rolling(7, min_periods=7).mean()
    out["tair_30"] = tair.rolling(30, min_periods=30).mean()

    out["wetness_7"] = out["rain_7"] - out["pet_7"]
    out["wetness_30"] = out["rain_30"] - out["pet_30"]
    out["rain_pet_ratio_30"] = out["rain_30"] / (out["pet_30"] + 1e-6)

    doy = out.index.dayofyear.to_numpy(dtype=np.float64)
    out["doy_sin"] = np.sin(2.0 * np.pi * doy / 366.0)
    out["doy_cos"] = np.cos(2.0 * np.pi * doy / 366.0)

    out["streamflow_daily"] = q

    for col in STATIC_FEATURES:
        out[col] = float(static_row[col])

    for split_name, (start, end) in SPLITS.items():
        mask = (out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))
        out.loc[mask, "split"] = split_name

    out = out.loc[out["split"].notna()].copy()
    out = out.reset_index(names="date")
    return out


def build_feature_table(static_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for station_id in static_df.index.astype(str):
        frames.append(build_station_feature_frame(station_id, static_df.loc[station_id]))
    feature_df = pd.concat(frames, ignore_index=True)
    feature_df["date"] = pd.to_datetime(feature_df["date"])
    return feature_df


def compute_feature_scalers(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    scalers: dict[str, dict[str, float]] = {}
    for col in Z_FEATURES + STATIC_FEATURES:
        mean = float(train_df[col].mean())
        std = float(train_df[col].std())
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        scalers[col] = {"mean": mean, "std": std}

    y_mean = float(train_df["streamflow_daily"].mean())
    y_std = float(train_df["streamflow_daily"].std())
    if not np.isfinite(y_std) or y_std <= 0:
        y_std = 1.0
    scalers["streamflow_daily"] = {"mean": y_mean, "std": y_std}
    return scalers


def apply_standardization(feature_df: pd.DataFrame, scalers: dict[str, dict[str, float]]) -> pd.DataFrame:
    df = feature_df.copy()
    for col in Z_FEATURES + STATIC_FEATURES:
        df[f"{col}_std"] = (df[col] - scalers[col]["mean"]) / scalers[col]["std"]
    df["streamflow_daily_std"] = (
        (df["streamflow_daily"] - scalers["streamflow_daily"]["mean"]) / scalers["streamflow_daily"]["std"]
    )
    return df


class DailySequenceDataset(Dataset):
    def __init__(self, split_df: pd.DataFrame, lookback: int):
        self.lookback = int(lookback)
        self.samples: list[tuple[str, int]] = []
        self.station_frames: dict[str, pd.DataFrame] = {}

        keep_cols = ["date", "streamflow_daily", "streamflow_daily_std"] + [f"{c}_std" for c in Z_FEATURES + STATIC_FEATURES]
        for station_id, station_df in split_df.groupby("station_id", sort=True):
            station_df = station_df.sort_values("date").reset_index(drop=True).copy()
            station_df = station_df[keep_cols]
            self.station_frames[str(station_id)] = station_df

            if len(station_df) <= self.lookback:
                continue

            dyn_cols = [f"{c}_std" for c in Z_FEATURES]
            for t in range(self.lookback, len(station_df)):
                x_seq = station_df.loc[t - self.lookback : t - 1, dyn_cols].to_numpy(dtype=np.float32)
                y_t = float(station_df.loc[t, "streamflow_daily_std"])
                if np.isnan(x_seq).any() or not np.isfinite(y_t):
                    continue
                self.samples.append((str(station_id), t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        station_id, t = self.samples[idx]
        station_df = self.station_frames[station_id]
        x_dyn = station_df.loc[t - self.lookback : t - 1, [f"{c}_std" for c in Z_FEATURES]].to_numpy(dtype=np.float32)
        x_static = station_df.loc[t, [f"{c}_std" for c in STATIC_FEATURES]].to_numpy(dtype=np.float32)
        y_std = float(station_df.loc[t, "streamflow_daily_std"])
        y_raw = float(station_df.loc[t, "streamflow_daily"])
        date_value = station_df.loc[t, "date"].to_datetime64().astype("datetime64[ns]").astype(np.int64)

        return (
            {
                "Z_seq": torch.from_numpy(x_dyn),
                "S": torch.from_numpy(x_static),
            },
            {
                "y_std": torch.tensor([y_std], dtype=torch.float32),
                "y_raw": torch.tensor([y_raw], dtype=torch.float32),
            },
            station_id,
            np.int64(date_value),
        )


def build_station_std(split_df: pd.DataFrame) -> dict[str, float]:
    station_std: dict[str, float] = {}
    vals_all = []
    for station_id, station_df in split_df.groupby("station_id", sort=True):
        vals = station_df["streamflow_daily_std"].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size >= 2:
            station_std[str(station_id)] = float(np.std(vals))
            vals_all.append(vals)

    global_std = 1.0
    if vals_all:
        global_std = float(np.std(np.concatenate(vals_all)))
        if not np.isfinite(global_std) or global_std <= 0:
            global_std = 1.0

    for station_id in split_df["station_id"].astype(str).unique():
        station_std.setdefault(str(station_id), global_std)
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
            denom = (float(self.station_std.get(station_id, 1.0)) + self.eps) ** 2
            losses.append(mse / denom)
        return torch.stack(losses).mean()


class DailyLSTMTeacher(nn.Module):
    def __init__(
        self,
        dyn_input_size: int,
        static_input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(
            input_size=dyn_input_size + static_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, z_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        x_s = x_static.unsqueeze(1).repeat(1, z_seq.shape[1], 1)
        x = torch.cat([z_seq, x_s], dim=2)
        out, _ = self.lstm(x)
        last = self.dropout(out[:, -1, :])
        return self.head(last).view(-1)


@dataclass
class EvalResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    summary: dict[str, float | int | str]


def inverse_standardize_streamflow(values: np.ndarray, scalers: dict[str, dict[str, float]]) -> np.ndarray:
    y_mean = float(scalers["streamflow_daily"]["mean"])
    y_std = float(scalers["streamflow_daily"]["std"])
    return values * y_std + y_mean


def evaluate_daily_teacher(
    model: DailyLSTMTeacher,
    loader: DataLoader,
    device: torch.device,
    scalers: dict[str, dict[str, float]],
) -> EvalResult:
    model.eval()
    preds_by_station: dict[str, list[float]] = {}
    trues_by_station: dict[str, list[float]] = {}
    pred_rows = []

    with torch.no_grad():
        for x_dict, y_dict, stations, date_ns in loader:
            z_seq = x_dict["Z_seq"].to(device)
            x_s = x_dict["S"].to(device)
            pred_std = model(z_seq, x_s).detach().cpu().numpy().reshape(-1)
            true_std = y_dict["y_std"].detach().cpu().numpy().reshape(-1)
            pred_raw = inverse_standardize_streamflow(pred_std, scalers)
            true_raw = y_dict["y_raw"].detach().cpu().numpy().reshape(-1)
            dates = pd.to_datetime(date_ns.detach().cpu().numpy().reshape(-1), unit="ns")

            for i, station_id in enumerate(stations):
                station_id = str(station_id)
                preds_by_station.setdefault(station_id, []).append(float(pred_raw[i]))
                trues_by_station.setdefault(station_id, []).append(float(true_raw[i]))
                pred_rows.append(
                    {
                        "station_id": station_id,
                        "date": pd.Timestamp(dates[i]),
                        "obs_daily": float(true_raw[i]),
                        "pred_daily": float(pred_raw[i]),
                        "pred_daily_std": float(pred_std[i]),
                        "obs_daily_std": float(true_std[i]),
                    }
                )

    rows = []
    for station_id in sorted(preds_by_station):
        sim = np.asarray(preds_by_station[station_id], dtype=np.float64)
        obs = np.asarray(trues_by_station[station_id], dtype=np.float64)
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

    metrics = pd.DataFrame(rows)
    valid = metrics.loc[metrics["score_status"].eq("ok")]
    summary = {
        "n_total_stations": int(len(metrics)),
        "n_valid_stations": int(len(valid)),
        "n_excluded_stations": int(len(metrics) - len(valid)),
        "median_kge": float(valid["kge"].median()) if len(valid) else float("nan"),
        "median_nse": float(valid["nse"].median()) if len(valid) else float("nan"),
        "mean_kge": float(valid["kge"].mean()) if len(valid) else float("nan"),
        "mean_nse": float(valid["nse"].mean()) if len(valid) else float("nan"),
    }
    predictions = pd.DataFrame(pred_rows).sort_values(["station_id", "date"]).reset_index(drop=True)
    return EvalResult(metrics=metrics, predictions=predictions, summary=summary)


def inverse_standardize_predictions(
    predictions_df: pd.DataFrame,
    scalers: dict[str, dict[str, float]],
) -> pd.DataFrame:
    out = predictions_df.copy()
    if "pred_daily" not in out.columns:
        out["pred_daily"] = inverse_standardize_streamflow(out["pred_daily_std"].to_numpy(dtype=np.float64), scalers)
    if "obs_daily" not in out.columns:
        out["obs_daily"] = inverse_standardize_streamflow(out["obs_daily_std"].to_numpy(dtype=np.float64), scalers)
    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    static_df = load_selected_static_df()
    feature_df = build_feature_table(static_df)
    feature_df.to_csv(out_dir / "daily_feature_rows_raw.csv.gz", index=False)

    train_df = feature_df.loc[feature_df["split"].eq("train")].copy()
    scalers = compute_feature_scalers(train_df)
    scaled_df = apply_standardization(feature_df, scalers)
    scaled_df.to_csv(out_dir / "daily_feature_rows_scaled.csv.gz", index=False)

    run_metadata = {
        "seed": args.seed,
        "lookback": args.lookback,
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "dynamic_features": Z_FEATURES,
        "static_features": STATIC_FEATURES,
    }
    (out_dir / "feature_schema.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    (out_dir / "feature_scalers.json").write_text(json.dumps(scalers, indent=2) + "\n", encoding="utf-8")

    split_dfs = {
        split_name: scaled_df.loc[scaled_df["split"].eq(split_name)].copy()
        for split_name in SPLITS
    }

    train_dataset = DailySequenceDataset(split_dfs["train"], lookback=args.lookback)
    val_dataset = DailySequenceDataset(split_dfs["val"], lookback=args.lookback)
    test_dataset = DailySequenceDataset(split_dfs["test"], lookback=args.lookback)

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
        f"Teacher samples train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}",
        flush=True,
    )

    model = DailyLSTMTeacher(
        dyn_input_size=len(Z_FEATURES),
        static_input_size=len(STATIC_FEATURES),
        hidden_size=args.hidden_size,
        num_layers=1,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = DailyNSELoss(build_station_std(split_dfs["train"]))

    best_state = None
    best_val_kge = -math.inf
    best_epoch = 0
    wait = 0
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for x_dict, y_dict, stations, _ in train_loader:
            z_seq = x_dict["Z_seq"].to(device)
            x_s = x_dict["S"].to(device)
            y = y_dict["y_std"].to(device).view(-1)

            optimizer.zero_grad()
            pred = model(z_seq, x_s)
            loss = criterion(pred, y, list(stations))
            loss.backward()
            optimizer.step()

            batch_n = z_seq.size(0)
            total_loss += float(loss.item()) * batch_n
            total_n += batch_n

        train_loss = total_loss / max(total_n, 1)
        val_eval = evaluate_daily_teacher(model, val_loader, device, scalers)
        val_kge = float(val_eval.summary["median_kge"])
        val_nse = float(val_eval.summary["median_nse"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_median_kge": val_kge,
                "val_median_nse": val_nse,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_median_kge={val_kge:.6f} val_median_nse={val_nse:.6f}",
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
        raise RuntimeError("No best checkpoint found during teacher training.")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "best_daily_teacher_model.pth")
    pd.DataFrame(history_rows).to_csv(out_dir / "teacher_training_history.csv", index=False)

    all_metric_frames = []
    summary_rows = []
    prediction_frames = []
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        eval_result = evaluate_daily_teacher(model, loader, device, scalers)
        metrics = eval_result.metrics.rename(
            columns={
                "samples": f"{split_name}_samples",
                "score_status": f"{split_name}_score_status",
                "exclusion_reason": f"{split_name}_exclusion_reason",
                "nse": f"{split_name}_nse",
                "kge": f"{split_name}_kge",
            }
        )
        all_metric_frames.append(metrics)

        summary = {"split": split_name, **eval_result.summary}
        summary_rows.append(summary)

        preds = inverse_standardize_predictions(eval_result.predictions, scalers)
        preds["split"] = split_name
        prediction_frames.append(preds)

    per_station = pd.read_csv(SELECTED_STATIONS_CSV, dtype={"station_id": str})
    for metrics in all_metric_frames:
        per_station = per_station.merge(metrics, on="station_id", how="left")
    per_station.to_csv(out_dir / "teacher_per_station_metrics.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "teacher_summary_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(out_dir / "teacher_predictions.csv.gz", index=False)

    lines = [
        "Daily LSTM teacher for SymTorch prior extraction on S2 random 30 stations",
        "",
        f"Device: {device}",
        f"Seed: {args.seed}",
        f"Lookback days: {args.lookback}",
        f"Hidden size: {args.hidden_size}",
        f"Dropout: {args.dropout}",
        f"Learning rate: {args.lr}",
        f"Weight decay: {args.weight_decay}",
        f"Best epoch by val median KGE: {best_epoch}",
        "",
        "Dynamic Z features:",
        *[f"- {name}" for name in Z_FEATURES],
        "",
        "Static features:",
        *[f"- {name}" for name in STATIC_FEATURES],
        "",
        "Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"{row.split},{row.n_total_stations},{row.n_valid_stations},{row.n_excluded_stations},"
            f"{row.median_kge:.6f},{row.median_nse:.6f},{row.mean_kge:.6f},{row.mean_nse:.6f}"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
