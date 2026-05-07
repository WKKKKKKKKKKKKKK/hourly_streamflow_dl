from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import config  # noqa: E402
from Modelzoo import sMTSLSTM  # noqa: E402
from loder import _get_static_row  # noqa: E402


EXP_DIR = Path(__file__).resolve().parents[1]
RAW_TIMESERIES_DIR = Path(
    "/mnt/datawaha/hyex/atr/gscad_database/raw/CAMELS/CAMELSH/timeseries/Data/CAMELSH/timeseries"
)
STATIC_MODEL_INPUT_PATH = EXP_DIR / "metadata" / "static_h_topo_priority27.csv"
SOURCE_RUN_DIR = (
    EXP_DIR
    / "training_runs"
    / "20260407_mtslstm_100stations_tuning_topo18_v100"
    / "idx2_bs128_do0.4_hs64_H168_D365"
)
SOURCE_MODEL_PATH = SOURCE_RUN_DIR / "best_model.pth"
SCALER_PATH = SOURCE_RUN_DIR / "scalers.pkl"
TRANSFER_MODEL_PATH = (
    EXP_DIR
    / "outputs"
    / "transfer_daily_to_hourly_partial_ft_s2_random30"
    / "best_transfer_model.pth"
)
SYMBOLIC_TRANSFER_MODEL_PATH = (
    EXP_DIR
    / "outputs"
    / "transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.05"
    / "best_transfer_model.pth"
)
OUT_DIR = EXP_DIR / "outputs" / "source_domain_transfer_retention_eval"

DYNAMIC_VARS = ["Rainf", "Tair", "PotEvap"]
TARGET_VAR = "Streamflow"
ALL_VARS = DYNAMIC_VARS + [TARGET_VAR]
LOOKBACK_H = 168
LOOKBACK_D = 365
FREQ = 24

SPLITS = {
    "train": (config.TRAIN_START, config.TRAIN_END),
    "val": (config.VAL_START, config.VAL_END),
    "test": (config.TEST_START, config.TEST_END),
}


@dataclass
class MetricStats:
    n: int = 0
    sum_obs: float = 0.0
    sum_sim: float = 0.0
    sum_obs2: float = 0.0
    sum_sim2: float = 0.0
    sum_obs_sim: float = 0.0
    sum_sqerr: float = 0.0

    def update(self, obs: np.ndarray, sim: np.ndarray) -> None:
        mask = np.isfinite(obs) & np.isfinite(sim)
        obs = obs[mask].astype(np.float64, copy=False)
        sim = sim[mask].astype(np.float64, copy=False)
        if obs.size == 0:
            return
        self.n += int(obs.size)
        self.sum_obs += float(obs.sum())
        self.sum_sim += float(sim.sum())
        self.sum_obs2 += float(np.dot(obs, obs))
        self.sum_sim2 += float(np.dot(sim, sim))
        self.sum_obs_sim += float(np.dot(obs, sim))
        err = sim - obs
        self.sum_sqerr += float(np.dot(err, err))

    def metrics(self) -> tuple[str, str, float, float]:
        if self.n < 2:
            return "excluded", "too_few_samples", float("nan"), float("nan")

        obs_mean = self.sum_obs / self.n
        sim_mean = self.sum_sim / self.n
        obs_var_sum = self.sum_obs2 - (self.sum_obs * self.sum_obs) / self.n
        sim_var_sum = self.sum_sim2 - (self.sum_sim * self.sum_sim) / self.n
        cov_sum = self.sum_obs_sim - (self.sum_obs * self.sum_sim) / self.n

        if obs_var_sum <= 0:
            return "excluded", "obs_std_zero", float("nan"), float("nan")
        if obs_mean == 0:
            return "excluded", "obs_mean_zero", float("nan"), float("nan")
        if sim_var_sum <= 0:
            return "excluded", "sim_std_zero", float("nan"), float("nan")

        nse = 1.0 - self.sum_sqerr / obs_var_sum
        corr = cov_sum / np.sqrt(obs_var_sum * sim_var_sum)
        alpha = np.sqrt(sim_var_sum / self.n) / np.sqrt(obs_var_sum / self.n)
        beta = sim_mean / obs_mean
        kge = 1.0 - np.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

        if not np.isfinite(nse) or not np.isfinite(kge):
            return "excluded", "metric_nonfinite", float("nan"), float("nan")
        return "ok", "", float(nse), float(kge)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit-stations", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_scalers() -> dict:
    with SCALER_PATH.open("rb") as fp:
        return pickle.load(fp)


def load_static_df() -> pd.DataFrame:
    static_df = pd.read_csv(STATIC_MODEL_INPUT_PATH, index_col=0)
    static_df.index = pd.Index(static_df.index.astype(str).str.strip())
    return static_df


def create_model(model_path: Path, device: torch.device) -> sMTSLSTM:
    model = sMTSLSTM(
        dyn_input_size=config.DYN_INPUT_SIZE,
        static_input_size=config.STATIC_INPUT_SIZE,
        hidden_size_daily=64,
        hidden_size_hourly=64,
        num_layers=1,
        dropout=0.4,
        frequency_factor=FREQ,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_station_split(station_id: str, start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    path = RAW_TIMESERIES_DIR / f"{station_id}.nc"
    with xr.open_dataset(path) as ds:
        ds = ds[ALL_VARS].sel(DateTime=slice(start, end)).load()
        x_raw = ds[DYNAMIC_VARS].to_array(dim="dynamic_forcing").transpose("DateTime", "dynamic_forcing")
        y_raw = ds[TARGET_VAR]
        x = np.asarray(x_raw.values, dtype=np.float32)
        y = np.asarray(y_raw.values, dtype=np.float32)
    y = np.where((y >= 0.0) & (y <= 1000.0), y, np.nan).astype(np.float32)
    return x, y


def standardize_dynamic(x_raw: np.ndarray, scalers: dict) -> np.ndarray:
    mean = scalers["x_dyn_mean"].sel(dynamic_forcing=DYNAMIC_VARS).values.astype(np.float32)
    std = scalers["x_dyn_std"].sel(dynamic_forcing=DYNAMIC_VARS).values.astype(np.float32)
    return ((x_raw - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)


def standardize_static(station_id: str, static_df: pd.DataFrame, scalers: dict) -> np.ndarray:
    row = _get_static_row(static_df, station_id).reindex(scalers["x_st_mean"].index)
    values = row.to_numpy(dtype=np.float32)
    mean = scalers["x_st_mean"].to_numpy(dtype=np.float32)
    std = scalers["x_st_std"].to_numpy(dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def rolling_24_hour_mean(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    filled = np.where(finite, x, 0.0).astype(np.float32)
    sums = np.vstack([np.zeros((1, x.shape[1]), dtype=np.float32), np.cumsum(filled, axis=0, dtype=np.float32)])
    counts = np.vstack([np.zeros((1, x.shape[1]), dtype=np.int32), np.cumsum(finite.astype(np.int32), axis=0)])
    out = np.full((x.shape[0] + 1, x.shape[1]), np.nan, dtype=np.float32)
    window_sum = sums[24:] - sums[:-24]
    window_count = counts[24:] - counts[:-24]
    means = window_sum / 24.0
    means[window_count != 24] = np.nan
    out[24:] = means
    return out


def predict_hourly_last(
    model: sMTSLSTM,
    h_tensor: torch.Tensor,
    d_tensor: torch.Tensor,
    s_tensor: torch.Tensor,
) -> torch.Tensor:
    """Equivalent to model(...)[\"H\"], but skips unused daily/H sequence outputs."""
    batch_size, seq_len_d, _ = d_tensor.shape
    seq_len_h = h_tensor.shape[1]

    s_daily = s_tensor.unsqueeze(1).repeat(1, seq_len_d, 1)
    x_daily = torch.cat([d_tensor, s_daily], dim=2)

    offset_days = seq_len_h // model.frequency_factor
    transfer_index = seq_len_d - offset_days
    if transfer_index <= 0:
        raise ValueError("Daily sequence too short for hourly alignment.")

    _, (h_mid, c_mid) = model.lstm_daily(x_daily[:, :transfer_index, :])
    h_transfer = h_mid[-1]
    c_transfer = c_mid[-1]

    h0 = model.transfer_h(h_transfer).unsqueeze(0).repeat(model.num_layers, 1, 1)
    c0 = model.transfer_c(c_transfer).unsqueeze(0).repeat(model.num_layers, 1, 1)

    s_hourly = s_tensor.unsqueeze(1).repeat(1, seq_len_h, 1)
    x_hourly = torch.cat([h_tensor, s_hourly], dim=2)
    _, (h_hourly, _) = model.lstm_hourly(x_hourly, (h0, c0))
    return model.head_hourly(h_hourly[-1]).squeeze(-1)


def evaluate_station_split(
    models: dict[str, sMTSLSTM],
    x_std: np.ndarray,
    y_raw: np.ndarray,
    static_std: np.ndarray,
    scalers: dict,
    device: torch.device,
    batch_size: int,
) -> dict[str, MetricStats]:
    stats = {name: MetricStats() for name in models}
    t_min = max(LOOKBACK_H, LOOKBACK_D * FREQ)
    if len(y_raw) <= t_min:
        return stats

    h_view = sliding_window_view(x_std, LOOKBACK_H, axis=0).transpose(0, 2, 1)
    d_roll24 = rolling_24_hour_mean(x_std)
    daily_offsets = FREQ * np.arange(1, LOOKBACK_D + 1, dtype=np.int64)

    y_mean = float(scalers["y_mean"])
    y_std = float(scalers["y_std"])
    candidate_t = np.arange(t_min, len(y_raw), dtype=np.int64)

    with torch.inference_mode():
        for start in range(0, len(candidate_t), batch_size):
            t = candidate_t[start : start + batch_size]
            obs = y_raw[t].astype(np.float64, copy=False)
            h_batch = h_view[t - LOOKBACK_H].astype(np.float32, copy=False)
            d_end_idx = t[:, None] - LOOKBACK_D * FREQ + daily_offsets[None, :]
            d_batch = d_roll24[d_end_idx].astype(np.float32, copy=False)

            valid = np.isfinite(obs)
            valid &= np.isfinite(h_batch).all(axis=(1, 2))
            valid &= np.isfinite(d_batch).all(axis=(1, 2))
            if not valid.any():
                continue

            h_tensor = torch.from_numpy(h_batch[valid]).to(device)
            d_tensor = torch.from_numpy(d_batch[valid]).to(device)
            s_tensor = torch.from_numpy(np.repeat(static_std[None, :], int(valid.sum()), axis=0)).to(device)
            obs_valid = obs[valid]

            for name, model in models.items():
                pred_std = predict_hourly_last(model, h_tensor, d_tensor, s_tensor).detach().cpu().numpy().reshape(-1)
                pred_raw = pred_std.astype(np.float64) * y_std + y_mean
                stats[name].update(obs_valid, pred_raw)

    return stats


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    summaries = []
    for (model, split), group in df.groupby(["model", "split"], sort=False):
        valid = group.loc[group["score_status"].eq("ok")].copy()
        summaries.append(
            {
                "model": model,
                "split": split,
                "n_total_stations": int(len(group)),
                "n_valid_stations": int(len(valid)),
                "n_excluded_stations": int(len(group) - len(valid)),
                "median_kge": float(valid["kge"].median()) if len(valid) else float("nan"),
                "median_nse": float(valid["nse"].median()) if len(valid) else float("nan"),
                "mean_kge": float(valid["kge"].mean()) if len(valid) else float("nan"),
                "mean_nse": float(valid["nse"].mean()) if len(valid) else float("nan"),
            }
        )
    return pd.DataFrame(summaries)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.threads))

    scalers = load_scalers()
    station_ids = list(scalers["station_y_std"].keys())
    if args.limit_stations > 0:
        station_ids = station_ids[: args.limit_stations]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_specs = {
        "source_pretransfer": SOURCE_MODEL_PATH,
        "transfer": TRANSFER_MODEL_PATH,
        "symbolic_transfer_sw0.05": SYMBOLIC_TRANSFER_MODEL_PATH,
    }
    models = {name: create_model(path, device) for name, path in model_specs.items()}
    static_df = load_static_df()

    per_station_path = out_dir / "per_station_source_domain_metrics.csv"
    summary_path = out_dir / "summary_source_domain_metrics.csv"
    if per_station_path.exists() and not args.force:
        raise FileExistsError(f"{per_station_path} already exists. Pass --force to overwrite.")

    rows: list[dict[str, object]] = []
    t0 = time.time()
    print(f"Evaluating {len(station_ids)} source-domain stations on {device}", flush=True)
    for station_idx, station_id in enumerate(station_ids, start=1):
        static_std = standardize_static(station_id, static_df, scalers)
        print(f"[{station_idx:03d}/{len(station_ids)}] {station_id} start", flush=True)
        for split_name, (start_date, end_date) in SPLITS.items():
            x_raw, y_raw = load_station_split(station_id, start_date, end_date)
            x_std = standardize_dynamic(x_raw, scalers)
            stats_by_model = evaluate_station_split(
                models=models,
                x_std=x_std,
                y_raw=y_raw,
                static_std=static_std,
                scalers=scalers,
                device=device,
                batch_size=args.batch_size,
            )
            for model_name, stats in stats_by_model.items():
                status, reason, nse, kge = stats.metrics()
                rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "station_id": station_id,
                        "samples": stats.n,
                        "score_status": status,
                        "exclusion_reason": reason,
                        "nse": nse,
                        "kge": kge,
                    }
                )
            split_preview = ", ".join(
                f"{name}:KGE={stats.metrics()[3]:.4f}" if stats.metrics()[0] == "ok" else f"{name}:excluded"
                for name, stats in stats_by_model.items()
            )
            print(f"[{station_idx:03d}/{len(station_ids)}] {station_id} {split_name} {split_preview}", flush=True)

        write_rows(per_station_path, rows)
        summary_df = summarize(rows)
        summary_df.to_csv(summary_path, index=False)

    summary_df = summarize(rows)
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "source_station_count": len(station_ids),
        "station_source": str(SCALER_PATH) + "::station_y_std.keys()",
        "splits": SPLITS,
        "models": {name: str(path) for name, path in model_specs.items()},
        "batch_size": args.batch_size,
        "threads": args.threads,
        "elapsed_seconds": time.time() - t0,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lines = [
        "Source-domain retention evaluation for transfer models",
        "",
        f"Stations: {len(station_ids)} from source scaler station_y_std keys",
        f"Elapsed seconds: {metadata['elapsed_seconds']:.1f}",
        "",
        "model,split,n_valid,median_kge,median_nse",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"{row.model},{row.split},{row.n_valid_stations},{row.median_kge:.6f},{row.median_nse:.6f}"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
