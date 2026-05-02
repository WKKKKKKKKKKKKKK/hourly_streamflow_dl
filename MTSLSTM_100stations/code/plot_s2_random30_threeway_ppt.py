from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, Dataset

import config
from Modelzoo import sMTSLSTM
from Train import _add_static_station_aliases
from loder import handle_extremes, standardize_data


ROOT = Path("/home/kongw0a/hourly_streamflow_dl")
RAW_TIMESERIES_DIR = Path(
    "/mnt/datawaha/hyex/atr/gscad_database/raw/CAMELS/CAMELSH/timeseries/Data/CAMELSH/timeseries"
)
SELECTED_STATIONS_CSV = ROOT / "MTSLSTM_100stations" / "outputs" / "s2_random30_idx2_eval" / "selected_stations.csv"
STATIC_MODEL_INPUT_PATH = ROOT / "MTSLSTM_100stations" / "metadata" / "static_h_topo_priority27.csv"

RUN_DIR = (
    ROOT
    / "MTSLSTM_100stations"
    / "training_runs"
    / "20260407_mtslstm_100stations_tuning_topo18_v100"
    / "idx2_bs128_do0.4_hs64_H168_D365"
)
BASELINE_MODEL_PATH = RUN_DIR / "best_model.pth"
BASELINE_METRICS_CSV = ROOT / "MTSLSTM_100stations" / "outputs" / "s2_random30_idx2_eval" / "per_station_metrics.csv"

TRANSFER_OUT_DIR = ROOT / "MTSLSTM_100stations" / "outputs" / "transfer_daily_to_hourly_partial_ft_s2_random30"
TRANSFER_MODEL_PATH = TRANSFER_OUT_DIR / "best_transfer_model.pth"
TRANSFER_METRICS_CSV = TRANSFER_OUT_DIR / "per_station_hourly_metrics.csv"

SYMBOLIC_OUT_DIR = ROOT / "MTSLSTM_100stations" / "outputs" / "transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.05"
SYMBOLIC_MODEL_PATH = SYMBOLIC_OUT_DIR / "best_transfer_model.pth"
SYMBOLIC_METRICS_CSV = SYMBOLIC_OUT_DIR / "per_station_hourly_metrics.csv"

SCALER_PATH = RUN_DIR / "scalers.pkl"
DEFAULT_OUT_DIR = ROOT / "MTSLSTM_100stations" / "outputs" / "s2_random30_threeway_ppt_plots"

DYNAMIC_VARS = ["Rainf", "Tair", "PotEvap"]
TARGET_VAR = "Streamflow"
LOOKBACK_H = 168
LOOKBACK_D = 365
FREQ = 24
BATCH_SIZE = 1024
SEED = 42
SCATTER_SAMPLE_SIZE = 10000

TEST_START = config.TEST_START
TEST_END = config.TEST_END
HYDRO_START = pd.Timestamp("2013-10-01 00:00:00")
HYDRO_END = pd.Timestamp("2015-09-30 23:00:00")
HYDRO_CONTEXT_START = HYDRO_START - pd.Timedelta(days=LOOKBACK_D)
FIXED_HYDRO_STATIONS = ["02191300", "07058000", "07071500"]

HYDRO_SUBTITLE_FONTSIZE = 24
HYDRO_AXISLABEL_FONTSIZE = 22
HYDRO_TICK_FONTSIZE = 18
HYDRO_LEGEND_FONTSIZE = 17

SCATTER_TITLE_FONTSIZE = 24
SCATTER_AXISLABEL_FONTSIZE = 21
SCATTER_TICK_FONTSIZE = 18
SCATTER_LEGEND_FONTSIZE = 17
SCATTER_TEXT_FONTSIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


@dataclass
class StationSeries:
    time: np.ndarray
    obs: np.ndarray
    pred: np.ndarray


class PredictionDataset(Dataset):
    def __init__(
        self,
        dataset: xr.Dataset,
        target: xr.Dataset,
        static_df: pd.DataFrame,
        lookback_hourly: int,
        lookback_daily: int,
        frequency_factor: int,
        start_date: str,
        end_date: str,
    ):
        self.lookback_hourly = int(lookback_hourly)
        self.lookback_daily = int(lookback_daily)
        self.frequency_factor = int(frequency_factor)
        self.static_df = static_df

        self.x_data: dict[str, np.ndarray] = {}
        self.y_data: dict[str, np.ndarray] = {}
        self.time_ns: dict[str, np.ndarray] = {}
        self.samples: list[tuple[str, int]] = []

        for stn in [str(s) for s in dataset.data_vars]:
            x = dataset[stn].sel(time=slice(start_date, end_date))
            y = target[stn].sel(time=slice(start_date, end_date))
            time = pd.to_datetime(x["time"].values)

            x_np = np.asarray(x.transpose("time", "dynamic_forcing").values, dtype=np.float32)
            y_np = np.asarray(y.values, dtype=np.float32)
            time_ns = time.to_numpy(dtype="datetime64[ns]").astype(np.int64)

            self.x_data[stn] = x_np
            self.y_data[stn] = y_np
            self.time_ns[stn] = time_ns

            total = x_np.shape[0]
            min_t = max(self.lookback_hourly, self.lookback_daily * self.frequency_factor)
            for t in range(min_t, total):
                x_h = x_np[t - self.lookback_hourly : t]
                x_d_full = x_np[t - self.lookback_daily * self.frequency_factor : t]
                y_t = y_np[t]
                if np.isnan(x_h).any() or np.isnan(x_d_full).any() or np.isnan(y_t):
                    continue
                self.samples.append((stn, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        stn, t = self.samples[idx]
        x = self.x_data[stn]
        y = self.y_data[stn]
        x_h = x[t - self.lookback_hourly : t]
        x_d_full = x[t - self.lookback_daily * self.frequency_factor : t]
        x_d = x_d_full.reshape(self.lookback_daily, self.frequency_factor, -1).mean(axis=1)
        x_s = self.static_df.loc[stn].to_numpy(dtype=np.float32)
        y_t = y[t]
        time_ns = np.int64(self.time_ns[stn][t])

        return (
            {
                "H": torch.from_numpy(x_h),
                "D": torch.from_numpy(x_d),
                "S": torch.from_numpy(x_s),
            },
            torch.tensor([y_t], dtype=torch.float32),
            stn,
            time_ns,
        )


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


def build_loader(
    full_ds: xr.Dataset,
    static_df: pd.DataFrame,
    scalers: dict,
    start_date: str,
    end_date: str,
) -> DataLoader:
    dyn = full_ds.sel(dynamic_forcing=DYNAMIC_VARS)
    y = full_ds.sel(dynamic_forcing=TARGET_VAR)
    dyn_std, static_std, y_std = standardize_data(dyn, static_df, y, scalers)
    static_std, missing_static = _add_static_station_aliases(static_std, full_ds.data_vars)
    if missing_static:
        preview = ", ".join(missing_static[:10])
        raise KeyError(f"Missing static features for selected stations: {preview}")

    dataset = PredictionDataset(
        dataset=dyn_std,
        target=y_std,
        static_df=static_std,
        lookback_hourly=LOOKBACK_H,
        lookback_daily=LOOKBACK_D,
        frequency_factor=FREQ,
        start_date=start_date,
        end_date=end_date,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader


def create_model(device: torch.device, model_path: Path) -> sMTSLSTM:
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


def collect_predictions(
    model: sMTSLSTM,
    loader: DataLoader,
    scalers: dict,
    device: torch.device,
) -> dict[str, StationSeries]:
    y_mean = float(scalers["y_mean"])
    y_std = float(scalers["y_std"])

    cache: dict[str, dict[str, list[float]]] = {}
    with torch.no_grad():
        for x_dict, y, stations, time_ns in loader:
            H = x_dict["H"].to(device)
            D = x_dict["D"].to(device)
            S = x_dict["S"].to(device)

            outputs = model({"H": H, "D": D}, S)
            preds = outputs["H"].detach().cpu().numpy().reshape(-1) * y_std + y_mean
            obs = y.detach().cpu().numpy().reshape(-1) * y_std + y_mean
            time_values = time_ns.detach().cpu().numpy().reshape(-1)

            for i, stn in enumerate(stations):
                key = str(stn)
                d = cache.setdefault(key, {"time_ns": [], "obs": [], "pred": []})
                d["time_ns"].append(int(time_values[i]))
                d["obs"].append(float(obs[i]))
                d["pred"].append(float(preds[i]))

    out: dict[str, StationSeries] = {}
    for stn, d in cache.items():
        time_ns = np.asarray(d["time_ns"], dtype=np.int64)
        order = np.argsort(time_ns)
        out[stn] = StationSeries(
            time=pd.to_datetime(time_ns[order], unit="ns").to_numpy(),
            obs=np.asarray(d["obs"], dtype=np.float64)[order],
            pred=np.asarray(d["pred"], dtype=np.float64)[order],
        )
    return out


def detect_local_peaks(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return np.array([], dtype=int)
    prominence = max(5.0, float(np.nanstd(values)) * 0.10, float(np.nanmax(values)) * 0.03)
    height = max(0.0, float(np.nanpercentile(values, 85)))
    peaks, _ = find_peaks(values, prominence=prominence, height=height, distance=12)
    return peaks.astype(int)


def compute_mean_peak_lag_hours(obs_peaks: np.ndarray, pred_peaks: np.ndarray, max_lag_hours: int = 72) -> tuple[float, int]:
    if len(obs_peaks) == 0 or len(pred_peaks) == 0:
        return float("nan"), 0

    used = set()
    lags = []
    for obs_idx in obs_peaks:
        diffs = pred_peaks - obs_idx
        nearest = np.argsort(np.abs(diffs))
        for pos in nearest:
            if int(pos) in used:
                continue
            lag = int(diffs[pos])
            if abs(lag) <= max_lag_hours:
                used.add(int(pos))
                lags.append(float(lag))
                break

    if not lags:
        return float("nan"), 0
    return float(np.mean(lags)), int(len(lags))


def compute_shared_hydro_y_limits(
    prediction_groups: list[dict[str, StationSeries]],
    station_ids: list[str],
) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for stn in station_ids:
        values = []
        for predictions in prediction_groups:
            series = predictions[stn]
            mask = (series.time >= HYDRO_START.to_datetime64()) & (series.time <= HYDRO_END.to_datetime64())
            values.append(series.obs[mask])
            values.append(series.pred[mask])
        combined = np.concatenate(values).astype(np.float64)
        combined = combined[np.isfinite(combined)]
        if combined.size == 0:
            limits[stn] = (0.0, 1.0)
            continue
        y_min = float(np.nanmin(combined))
        y_max = float(np.nanmax(combined))
        y_min = min(0.0, y_min)
        span = y_max - y_min
        pad = max(1.0, span * 0.06 if span > 0 else abs(y_max) * 0.08)
        limits[stn] = (y_min - pad * 0.08, y_max + pad)
    return limits


def flatten_predictions(predictions: dict[str, StationSeries], station_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    obs_blocks = []
    pred_blocks = []
    for stn in station_ids:
        if stn not in predictions:
            continue
        series = predictions[stn]
        obs_blocks.append(series.obs)
        pred_blocks.append(series.pred)
    if not obs_blocks:
        raise ValueError("No overlapping stations available for scatter flattening.")
    return np.concatenate(obs_blocks), np.concatenate(pred_blocks)


def compute_shared_scatter_limits(
    prediction_groups: list[dict[str, StationSeries]],
    station_ids: list[str],
) -> tuple[float, float]:
    values = []
    for predictions in prediction_groups:
        obs, pred = flatten_predictions(predictions, station_ids)
        values.append(obs)
        values.append(pred)
    combined = np.concatenate(values).astype(np.float64)
    combined = combined[np.isfinite(combined)]
    low = min(float(np.nanmin(combined)), 0.0)
    high = max(float(np.nanpercentile(combined, 99.5)), 50.0)
    high = min(1020.0, high)
    low = min(low, -20.0)
    return low, high


def plot_hydrographs(
    predictions: dict[str, StationSeries],
    station_ids: list[str],
    output_path: Path,
    y_limits: dict[str, tuple[float, float]],
    label: str,
) -> pd.DataFrame:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(station_ids), 1, figsize=(19, 14), sharex=True)
    if len(station_ids) == 1:
        axes = [axes]

    rows = []
    for ax, stn in zip(axes, station_ids):
        series = predictions[stn]
        mask = (series.time >= HYDRO_START.to_datetime64()) & (series.time <= HYDRO_END.to_datetime64())
        time = pd.to_datetime(series.time[mask])
        obs = series.obs[mask]
        pred = series.pred[mask]

        obs_peaks = detect_local_peaks(obs)
        pred_peaks = detect_local_peaks(pred)
        mean_lag, n_matches = compute_mean_peak_lag_hours(obs_peaks, pred_peaks)
        lag_text = "n/a" if not np.isfinite(mean_lag) else f"{mean_lag:+.1f} h"

        ax.plot(time, obs, color="#1f77b4", linewidth=1.6, label="Observed")
        ax.plot(time, pred, color="#d62728", linewidth=1.4, alpha=0.95, label="Predicted")
        if len(obs_peaks):
            ax.scatter(time[obs_peaks], obs[obs_peaks], color="#1f77b4", s=28, label="Obs local peaks", zorder=4)
        if len(pred_peaks):
            ax.scatter(time[pred_peaks], pred[pred_peaks], color="#d62728", marker="x", s=34, label="Pred local peaks", zorder=4)

        ax.set_ylabel("Streamflow", fontsize=HYDRO_AXISLABEL_FONTSIZE)
        ax.set_ylim(*y_limits[stn])
        ax.set_title(
            f"Station {stn} | avg local peak lag = {lag_text} (n={n_matches})",
            fontsize=HYDRO_SUBTITLE_FONTSIZE,
        )
        ax.legend(loc="upper right", fontsize=HYDRO_LEGEND_FONTSIZE, ncol=2, framealpha=0.95)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="both", labelsize=HYDRO_TICK_FONTSIZE)

        rows.append(
            {
                "model": label,
                "station_id": stn,
                "hydro_start": str(HYDRO_START),
                "hydro_end": str(HYDRO_END),
                "mean_peak_lag_hours": mean_lag,
                "matched_peak_count": n_matches,
                "obs_peak_count": int(len(obs_peaks)),
                "pred_peak_count": int(len(pred_peaks)),
            }
        )

    axes[-1].set_xlabel(f"Time ({HYDRO_START} to {HYDRO_END})", fontsize=HYDRO_AXISLABEL_FONTSIZE)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def build_cdf(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0
    negative = int(np.sum(finite < 0))
    nonnegative = np.sort(finite[finite >= 0])
    if nonnegative.size == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 1.0]), negative
    x = np.concatenate(([0.0], nonnegative))
    y = np.concatenate(([negative / finite.size], (negative + np.arange(1, nonnegative.size + 1)) / finite.size))
    return x, y, negative


def plot_scatter_and_cdfs(
    predictions: dict[str, StationSeries],
    metrics_df: pd.DataFrame,
    station_ids: list[str],
    sample_idx: np.ndarray,
    scatter_limits: tuple[float, float],
    output_path: Path,
    label: str,
) -> pd.DataFrame:
    plt.style.use("seaborn-v0_8-whitegrid")
    all_obs, all_pred = flatten_predictions(predictions, station_ids)
    obs_sample = all_obs[sample_idx]
    pred_sample = all_pred[sample_idx]

    fig, axes = plt.subplots(1, 3, figsize=(23, 7.2))

    scatter_ax = axes[0]
    scatter_ax.scatter(obs_sample, pred_sample, s=10, alpha=0.20, color="#79d57c", edgecolors="none")
    low, high = scatter_limits
    scatter_ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1.6, label="y = x")
    scatter_ax.set_xlim(low, high)
    scatter_ax.set_ylim(low, high)
    scatter_ax.set_xlabel("Observed streamflow", fontsize=SCATTER_AXISLABEL_FONTSIZE)
    scatter_ax.set_ylabel("Predicted streamflow", fontsize=SCATTER_AXISLABEL_FONTSIZE)
    scatter_ax.set_title(f"Observed vs predicted scatter (n={len(sample_idx):,})", fontsize=SCATTER_TITLE_FONTSIZE)
    scatter_ax.legend(loc="lower right", fontsize=SCATTER_LEGEND_FONTSIZE, framealpha=0.95)
    scatter_ax.tick_params(axis="both", labelsize=SCATTER_TICK_FONTSIZE)

    test_kge = metrics_df["test_kge"].to_numpy(dtype=np.float64)
    test_nse = metrics_df["test_nse"].to_numpy(dtype=np.float64)
    excluded_kge = int(np.sum(~np.isfinite(test_kge)))
    excluded_nse = int(np.sum(~np.isfinite(test_nse)))

    kge_x, kge_y, kge_negative = build_cdf(test_kge)
    nse_x, nse_y, nse_negative = build_cdf(test_nse)

    kge_ax = axes[1]
    kge_ax.step(kge_x, kge_y, where="post", color="#ff7f0e", linewidth=2.6, label=f"CDF (n={np.isfinite(test_kge).sum()} valid)")
    kge_ax.set_xlim(0.0, 1.0)
    kge_ax.set_ylim(0.0, 1.03)
    kge_ax.set_xlabel("Test KGE", fontsize=SCATTER_AXISLABEL_FONTSIZE)
    kge_ax.set_ylabel("Cumulative probability", fontsize=SCATTER_AXISLABEL_FONTSIZE)
    kge_ax.set_title("CDF of test KGE", fontsize=SCATTER_TITLE_FONTSIZE)
    kge_ax.legend(loc="lower right", fontsize=SCATTER_LEGEND_FONTSIZE, framealpha=0.95)
    kge_ax.tick_params(axis="both", labelsize=SCATTER_TICK_FONTSIZE)
    kge_ax.text(
        0.03,
        0.96,
        f"negative stations = {kge_negative}\nexcluded = {excluded_kge}",
        transform=kge_ax.transAxes,
        va="top",
        fontsize=SCATTER_TEXT_FONTSIZE,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#bbbbbb"},
    )

    nse_ax = axes[2]
    nse_ax.step(nse_x, nse_y, where="post", color="#1f77b4", linewidth=2.6, label=f"CDF (n={np.isfinite(test_nse).sum()} valid)")
    nse_ax.set_xlim(0.0, 1.0)
    nse_ax.set_ylim(0.0, 1.03)
    nse_ax.set_xlabel("Test NSE", fontsize=SCATTER_AXISLABEL_FONTSIZE)
    nse_ax.set_ylabel("Cumulative probability", fontsize=SCATTER_AXISLABEL_FONTSIZE)
    nse_ax.set_title("CDF of test NSE", fontsize=SCATTER_TITLE_FONTSIZE)
    nse_ax.legend(loc="lower right", fontsize=SCATTER_LEGEND_FONTSIZE, framealpha=0.95)
    nse_ax.tick_params(axis="both", labelsize=SCATTER_TICK_FONTSIZE)
    nse_ax.text(
        0.03,
        0.96,
        f"negative stations = {nse_negative}\nexcluded = {excluded_nse}",
        transform=nse_ax.transAxes,
        va="top",
        fontsize=SCATTER_TEXT_FONTSIZE,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#bbbbbb"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(
        [
            {
                "model": label,
                "scatter_sample_n": int(len(sample_idx)),
                "test_kge_valid_n": int(np.isfinite(test_kge).sum()),
                "test_kge_negative_n": int(kge_negative),
                "test_kge_excluded_n": int(excluded_kge),
                "test_nse_valid_n": int(np.isfinite(test_nse).sum()),
                "test_nse_negative_n": int(nse_negative),
                "test_nse_excluded_n": int(excluded_nse),
            }
        ]
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    station_ids = load_station_ids()
    hydro_station_ids = FIXED_HYDRO_STATIONS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print("Loading scalers and source data...", flush=True)
    static_df = load_static_df()
    with SCALER_PATH.open("rb") as fp:
        scalers = pickle.load(fp)
    full_ds = load_hourly_dataset(station_ids)
    print("Building test loader...", flush=True)
    test_loader = build_loader(full_ds, static_df, scalers, TEST_START, TEST_END)
    print("Building hydrograph loader...", flush=True)
    hydro_loader = build_loader(
        full_ds,
        static_df,
        scalers,
        HYDRO_CONTEXT_START.strftime("%Y-%m-%d %H:%M:%S"),
        HYDRO_END.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print(
        f"Prediction windows loaded: test={len(test_loader.dataset)} hydro={len(hydro_loader.dataset)}",
        flush=True,
    )

    model_specs = [
        {
            "key": "baseline",
            "label": "Best 100-station MTSLSTM",
            "model_path": BASELINE_MODEL_PATH,
            "metrics_csv": BASELINE_METRICS_CSV,
            "hydro_png": "baseline_hydrographs_shared_two_year_window_ppt.png",
            "scatter_png": "baseline_scatter_kge_nse_cdf_ppt.png",
        },
        {
            "key": "transfer",
            "label": "Daily-supervised transfer MTSLSTM",
            "model_path": TRANSFER_MODEL_PATH,
            "metrics_csv": TRANSFER_METRICS_CSV,
            "hydro_png": "transfer_hydrographs_shared_two_year_window_ppt.png",
            "scatter_png": "transfer_scatter_kge_nse_cdf_ppt.png",
        },
        {
            "key": "symbolic_transfer",
            "label": "Transfer + symbolic regression prior",
            "model_path": SYMBOLIC_MODEL_PATH,
            "metrics_csv": SYMBOLIC_METRICS_CSV,
            "hydro_png": "symbolic_transfer_hydrographs_shared_two_year_window_ppt.png",
            "scatter_png": "symbolic_transfer_scatter_kge_nse_cdf_ppt.png",
        },
    ]

    test_predictions: dict[str, dict[str, StationSeries]] = {}
    hydro_predictions: dict[str, dict[str, StationSeries]] = {}
    metrics_tables: dict[str, pd.DataFrame] = {}

    for spec in model_specs:
        print(f"Collecting predictions for {spec['key']}...", flush=True)
        model = create_model(device, spec["model_path"])
        test_predictions[spec["key"]] = collect_predictions(model, test_loader, scalers, device)
        hydro_predictions[spec["key"]] = collect_predictions(model, hydro_loader, scalers, device)
        metrics_tables[spec["key"]] = pd.read_csv(spec["metrics_csv"], dtype={"station_id": str})

    hydro_y_limits = compute_shared_hydro_y_limits(list(hydro_predictions.values()), hydro_station_ids)
    common_scatter_stations = sorted(
        set(station_ids).intersection(*[set(preds.keys()) for preds in test_predictions.values()])
    )
    if not common_scatter_stations:
        raise RuntimeError("No common stations with valid prediction windows across the three models.")
    scatter_limits = compute_shared_scatter_limits(list(test_predictions.values()), common_scatter_stations)

    baseline_obs, _ = flatten_predictions(test_predictions["baseline"], common_scatter_stations)
    rng = np.random.default_rng(SEED)
    sample_n = min(SCATTER_SAMPLE_SIZE, baseline_obs.size)
    sample_idx = rng.choice(baseline_obs.size, size=sample_n, replace=False)

    hydro_rows = []
    scatter_rows = []
    for spec in model_specs:
        print(f"Rendering plots for {spec['key']}...", flush=True)
        hydro_rows.append(
            plot_hydrographs(
                predictions=hydro_predictions[spec["key"]],
                station_ids=hydro_station_ids,
                output_path=out_dir / spec["hydro_png"],
                y_limits=hydro_y_limits,
                label=spec["label"],
            )
        )
        scatter_rows.append(
            plot_scatter_and_cdfs(
                predictions=test_predictions[spec["key"]],
                metrics_df=metrics_tables[spec["key"]],
                station_ids=common_scatter_stations,
                sample_idx=sample_idx,
                scatter_limits=scatter_limits,
                output_path=out_dir / spec["scatter_png"],
                label=spec["label"],
            )
        )

    pd.concat(hydro_rows, ignore_index=True).to_csv(out_dir / "hydrograph_peak_lag_summary.csv", index=False)
    pd.concat(scatter_rows, ignore_index=True).to_csv(out_dir / "scatter_cdf_summary.csv", index=False)
    pd.DataFrame({"station_id": hydro_station_ids, "selection_note": "fixed stations reused for cross-model PPT comparison"}).to_csv(
        out_dir / "hydrograph_selected_stations.csv",
        index=False,
    )

    summary_lines = [
        "Three-model S2 Cfa-SE 30-station PPT plots with enlarged fonts",
        "",
        f"Baseline model: {BASELINE_MODEL_PATH}",
        f"Transfer model: {TRANSFER_MODEL_PATH}",
        f"Symbolic transfer model: {SYMBOLIC_MODEL_PATH}",
        f"Shared hydrograph window: {HYDRO_START} to {HYDRO_END}",
        f"Hydrograph context start: {HYDRO_CONTEXT_START}",
        f"Hydrograph stations: {', '.join(hydro_station_ids)}",
        f"Common scatter stations: {len(common_scatter_stations)}",
        f"Shared scatter axis limits: {scatter_limits}",
        "",
        "Generated files:",
    ]
    for spec in model_specs:
        summary_lines.append(f"- {spec['hydro_png']}")
        summary_lines.append(f"- {spec['scatter_png']}")
    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
