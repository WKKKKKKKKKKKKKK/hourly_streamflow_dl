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
CODE_DIR = ROOT / "MTSLSTM_100stations" / "code"
RAW_TIMESERIES_DIR = Path(
    "/mnt/datawaha/hyex/atr/gscad_database/raw/CAMELS/CAMELSH/timeseries/Data/CAMELSH/timeseries"
)
SELECTED_STATIONS_CSV = ROOT / "MTSLSTM_100stations" / "outputs" / "s2_random30_idx2_eval" / "selected_stations.csv"
BASELINE_METRICS_CSV = ROOT / "MTSLSTM_100stations" / "outputs" / "s2_random30_idx2_eval" / "per_station_metrics.csv"
TRANSFER_OUT_DIR = ROOT / "MTSLSTM_100stations" / "outputs" / "transfer_daily_to_hourly_partial_ft_s2_random30"
TRANSFER_METRICS_CSV = TRANSFER_OUT_DIR / "per_station_hourly_metrics.csv"
PLOTS_OUT_DIR = TRANSFER_OUT_DIR / "baseline_vs_transfer_plots"
STATIC_MODEL_INPUT_PATH = ROOT / "MTSLSTM_100stations" / "metadata" / "static_h_topo_priority27.csv"
RUN_DIR = (
    ROOT
    / "MTSLSTM_100stations"
    / "training_runs"
    / "20260407_mtslstm_100stations_tuning_topo18_v100"
    / "idx2_bs128_do0.4_hs64_H168_D365"
)
BASELINE_MODEL_PATH = RUN_DIR / "best_model.pth"
TRANSFER_MODEL_PATH = TRANSFER_OUT_DIR / "best_transfer_model.pth"
SCALER_PATH = RUN_DIR / "scalers.pkl"

DYNAMIC_VARS = ["Rainf", "Tair", "PotEvap"]
TARGET_VAR = "Streamflow"
LOOKBACK_H = 168
LOOKBACK_D = 365
FREQ = 24
BATCH_SIZE = 1024
SEED = 42
N_HYDRO_STATIONS = 3
SCATTER_SAMPLE_SIZE = 10000
TEST_START = config.TEST_START
TEST_END = config.TEST_END
HYDRO_START = pd.Timestamp("2013-10-01 00:00:00")
HYDRO_END = pd.Timestamp("2015-09-30 23:00:00")
HYDRO_CONTEXT_START = HYDRO_START - pd.Timedelta(days=LOOKBACK_D)
FIXED_HYDRO_STATIONS = ["02191300", "07058000", "07071500"]

HYDRO_SUPTITLE_FONTSIZE = 20
HYDRO_SUBTITLE_FONTSIZE = 17
HYDRO_AXISLABEL_FONTSIZE = 15
HYDRO_TICK_FONTSIZE = 13
HYDRO_LEGEND_FONTSIZE = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hydro-only", action="store_true")
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


def prepare_loader(station_ids: list[str], start_date: str, end_date: str) -> tuple[DataLoader, dict]:
    static_df = load_static_df()
    with SCALER_PATH.open("rb") as fp:
        scalers = pickle.load(fp)
    full_ds = load_hourly_dataset(station_ids)
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
    return loader, scalers


def prepare_test_loader(station_ids: list[str]) -> tuple[DataLoader, dict]:
    return prepare_loader(station_ids, TEST_START, TEST_END)


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


def load_valid_hydrograph_station_candidates() -> list[str]:
    baseline = pd.read_csv(BASELINE_METRICS_CSV, dtype={"station_id": str})
    transfer = pd.read_csv(TRANSFER_METRICS_CSV, dtype={"station_id": str})

    valid_baseline = set(baseline.loc[baseline["test_score_status"].eq("ok"), "station_id"].astype(str))
    valid_transfer = set(transfer.loc[transfer["test_score_status"].eq("ok"), "station_id"].astype(str))
    return sorted(valid_baseline & valid_transfer)


def detect_local_peaks(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return np.array([], dtype=int)
    prominence = max(5.0, float(np.nanstd(values)) * 0.10, float(np.nanmax(values)) * 0.03)
    height = max(0.0, float(np.nanpercentile(values, 85)))
    peaks, _ = find_peaks(values, prominence=prominence, height=height, distance=12)
    return peaks.astype(int)


def choose_informative_hydrograph_stations(predictions: dict[str, StationSeries], candidate_ids: list[str]) -> list[str]:
    informative = []
    for stn in candidate_ids:
        series = predictions[stn]
        mask = (series.time >= HYDRO_START.to_datetime64()) & (series.time <= HYDRO_END.to_datetime64())
        obs = series.obs[mask]
        if obs.size == 0:
            continue
        if not np.isfinite(obs).all():
            continue
        if float(np.nanmax(obs)) < 20.0:
            continue
        if float(np.nanstd(obs)) < 1.0:
            continue
        if len(detect_local_peaks(obs)) < 5:
            continue
        informative.append(stn)

    pool = informative if len(informative) >= N_HYDRO_STATIONS else candidate_ids
    if len(pool) < N_HYDRO_STATIONS:
        raise ValueError(f"Only {len(pool)} eligible hydrograph stations available.")

    rng = np.random.default_rng(SEED)
    chosen = rng.choice(np.asarray(sorted(pool)), size=N_HYDRO_STATIONS, replace=False)
    return sorted(str(x) for x in chosen.tolist())


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
    baseline_predictions: dict[str, StationSeries],
    transfer_predictions: dict[str, StationSeries],
    station_ids: list[str],
) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for stn in station_ids:
        values = []
        for predictions in (baseline_predictions, transfer_predictions):
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
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            limits[stn] = (0.0, 1.0)
            continue

        y_min = min(0.0, y_min)
        span = y_max - y_min
        if span <= 0:
            pad = max(1.0, abs(y_max) * 0.05)
        else:
            pad = span * 0.05
        limits[stn] = (y_min - pad * 0.1, y_max + pad)
    return limits


def plot_hydrographs(
    predictions: dict[str, StationSeries],
    station_ids: list[str],
    model_title: str,
    output_path: Path,
    y_limits: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(station_ids), 1, figsize=(16, 12), sharex=True)
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

        ax.plot(time, obs, color="#1f77b4", linewidth=1.2, label="Observed")
        ax.plot(time, pred, color="#d62728", linewidth=1.0, alpha=0.95, label="Predicted")
        if len(obs_peaks):
            ax.scatter(time[obs_peaks], obs[obs_peaks], color="#1f77b4", s=18, label="Obs local peaks", zorder=4)
        if len(pred_peaks):
            ax.scatter(time[pred_peaks], pred[pred_peaks], color="#d62728", marker="x", s=22, label="Pred local peaks", zorder=4)

        ax.set_ylabel("Streamflow", fontsize=HYDRO_AXISLABEL_FONTSIZE)
        if y_limits is not None and stn in y_limits:
            ax.set_ylim(*y_limits[stn])
        ax.set_title(
            f"Station {stn} | avg local peak lag = {lag_text} (n={n_matches})",
            fontsize=HYDRO_SUBTITLE_FONTSIZE,
        )
        ax.legend(loc="upper right", fontsize=HYDRO_LEGEND_FONTSIZE, ncol=2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="both", labelsize=HYDRO_TICK_FONTSIZE)

        rows.append(
            {
                "model": model_title,
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
    fig.savefig(output_path, dpi=200)
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
    model_title: str,
    output_path: Path,
) -> pd.DataFrame:
    plt.style.use("seaborn-v0_8-whitegrid")
    all_obs = np.concatenate([series.obs for series in predictions.values()])
    all_pred = np.concatenate([series.pred for series in predictions.values()])

    rng = np.random.default_rng(SEED)
    sample_n = min(SCATTER_SAMPLE_SIZE, all_obs.size)
    sample_idx = rng.choice(all_obs.size, size=sample_n, replace=False)
    obs_sample = all_obs[sample_idx]
    pred_sample = all_pred[sample_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    scatter_ax = axes[0]
    scatter_ax.scatter(obs_sample, pred_sample, s=6, alpha=0.18, color="#79d57c", edgecolors="none")
    low = min(float(obs_sample.min()), float(pred_sample.min()), 0.0)
    high = max(float(np.quantile(obs_sample, 0.995)), float(np.quantile(pred_sample, 0.995)), 1.0)
    high = min(1020.0, max(high, 50.0))
    low = min(low, -20.0)
    scatter_ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1.2, label="y = x")
    scatter_ax.set_xlim(low, high)
    scatter_ax.set_ylim(low, high)
    scatter_ax.set_xlabel("Observed streamflow")
    scatter_ax.set_ylabel("Predicted streamflow")
    scatter_ax.set_title(f"Observed vs predicted scatter (n={sample_n:,})")
    scatter_ax.legend(loc="lower right")

    test_kge = metrics_df["test_kge"].to_numpy(dtype=np.float64)
    test_nse = metrics_df["test_nse"].to_numpy(dtype=np.float64)
    excluded_kge = int(np.sum(~np.isfinite(test_kge)))
    excluded_nse = int(np.sum(~np.isfinite(test_nse)))

    kge_x, kge_y, kge_negative = build_cdf(test_kge)
    nse_x, nse_y, nse_negative = build_cdf(test_nse)

    kge_ax = axes[1]
    kge_ax.step(kge_x, kge_y, where="post", color="#ff7f0e", linewidth=2.0, label=f"CDF (n={np.isfinite(test_kge).sum()} valid)")
    kge_ax.set_xlim(0.0, 1.0)
    kge_ax.set_ylim(0.0, 1.03)
    kge_ax.set_xlabel("Test KGE")
    kge_ax.set_ylabel("Cumulative probability")
    kge_ax.set_title("CDF of test KGE")
    kge_ax.legend(loc="lower right")
    kge_ax.text(
        0.02,
        0.96,
        f"negative stations = {kge_negative}\nexcluded = {excluded_kge}",
        transform=kge_ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
    )

    nse_ax = axes[2]
    nse_ax.step(nse_x, nse_y, where="post", color="#1f77b4", linewidth=2.0, label=f"CDF (n={np.isfinite(test_nse).sum()} valid)")
    nse_ax.set_xlim(0.0, 1.0)
    nse_ax.set_ylim(0.0, 1.03)
    nse_ax.set_xlabel("Test NSE")
    nse_ax.set_ylabel("Cumulative probability")
    nse_ax.set_title("CDF of test NSE")
    nse_ax.legend(loc="lower right")
    nse_ax.text(
        0.02,
        0.96,
        f"negative stations = {nse_negative}\nexcluded = {excluded_nse}",
        transform=nse_ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(
        [
            {
                "model": model_title,
                "scatter_sample_n": int(sample_n),
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
    PLOTS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.hydro_only:
        station_ids = FIXED_HYDRO_STATIONS
        hydro_station_ids = FIXED_HYDRO_STATIONS
        loader, scalers = prepare_loader(
            station_ids,
            HYDRO_CONTEXT_START.strftime("%Y-%m-%d %H:%M:%S"),
            HYDRO_END.strftime("%Y-%m-%d %H:%M:%S"),
        )
    else:
        station_ids = load_station_ids()
        hydro_candidate_ids = load_valid_hydrograph_station_candidates()
        loader, scalers = prepare_test_loader(station_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"Prediction windows: {len(loader.dataset)}", flush=True)

    baseline_model = create_model(device, BASELINE_MODEL_PATH)
    transfer_model = create_model(device, TRANSFER_MODEL_PATH)

    baseline_predictions = collect_predictions(baseline_model, loader, scalers, device)
    transfer_predictions = collect_predictions(transfer_model, loader, scalers, device)
    if not args.hydro_only:
        hydro_station_ids = choose_informative_hydrograph_stations(baseline_predictions, hydro_candidate_ids)
    hydro_y_limits = compute_shared_hydro_y_limits(baseline_predictions, transfer_predictions, hydro_station_ids)
    pd.DataFrame({"station_id": hydro_station_ids, "selection_seed": SEED}).to_csv(
        PLOTS_OUT_DIR / "hydrograph_selected_stations.csv",
        index=False,
    )
    print(f"Hydrograph stations: {', '.join(hydro_station_ids)}", flush=True)

    baseline_metrics = pd.read_csv(BASELINE_METRICS_CSV, dtype={"station_id": str})
    transfer_metrics = pd.read_csv(TRANSFER_METRICS_CSV, dtype={"station_id": str})

    hydro_rows = []
    hydro_rows.append(
        plot_hydrographs(
            predictions=baseline_predictions,
            station_ids=hydro_station_ids,
            model_title="Baseline best 100-station MTSLSTM",
            output_path=PLOTS_OUT_DIR / "baseline_hydrographs_shared_two_year_window.png",
            y_limits=hydro_y_limits,
        )
    )
    hydro_rows.append(
        plot_hydrographs(
            predictions=transfer_predictions,
            station_ids=hydro_station_ids,
            model_title="Daily-supervised transfer MTSLSTM",
            output_path=PLOTS_OUT_DIR / "transfer_hydrographs_shared_two_year_window.png",
            y_limits=hydro_y_limits,
        )
    )
    pd.concat(hydro_rows, ignore_index=True).to_csv(PLOTS_OUT_DIR / "hydrograph_peak_lag_summary.csv", index=False)

    if args.hydro_only:
        summary_lines = [
            "Hydrographs refreshed with a fixed station order and larger label fonts",
            "",
            f"Baseline model: {BASELINE_MODEL_PATH}",
            f"Transfer model: {TRANSFER_MODEL_PATH}",
            f"Shared hydrograph window: {HYDRO_START} to {HYDRO_END}",
            f"Hydrograph context start: {HYDRO_CONTEXT_START}",
            f"Hydrograph stations: {', '.join(hydro_station_ids)}",
            "",
            "Updated files:",
            "- baseline_hydrographs_shared_two_year_window.png",
            "- transfer_hydrographs_shared_two_year_window.png",
            "- hydrograph_selected_stations.csv",
            "- hydrograph_peak_lag_summary.csv",
        ]
        (PLOTS_OUT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        return

    scatter_rows = []
    scatter_rows.append(
        plot_scatter_and_cdfs(
            predictions=baseline_predictions,
            metrics_df=baseline_metrics,
            model_title="Baseline best 100-station MTSLSTM",
            output_path=PLOTS_OUT_DIR / "baseline_scatter_kge_nse_cdf.png",
        )
    )
    scatter_rows.append(
        plot_scatter_and_cdfs(
            predictions=transfer_predictions,
            metrics_df=transfer_metrics,
            model_title="Daily-supervised transfer MTSLSTM",
            output_path=PLOTS_OUT_DIR / "transfer_scatter_kge_nse_cdf.png",
        )
    )
    pd.concat(scatter_rows, ignore_index=True).to_csv(PLOTS_OUT_DIR / "scatter_cdf_summary.csv", index=False)

    summary_lines = [
        "Baseline vs transfer plots for the S2 Cfa-SE random-30 station test set",
        "",
        f"Baseline model: {BASELINE_MODEL_PATH}",
        f"Transfer model: {TRANSFER_MODEL_PATH}",
        f"Shared hydrograph window: {HYDRO_START} to {HYDRO_END}",
        f"Random hydrograph station seed: {SEED}",
        f"Hydrograph stations: {', '.join(hydro_station_ids)}",
        "",
        "Output files:",
        "- baseline_hydrographs_shared_two_year_window.png",
        "- transfer_hydrographs_shared_two_year_window.png",
        "- baseline_scatter_kge_nse_cdf.png",
        "- transfer_scatter_kge_nse_cdf.png",
        "- hydrograph_selected_stations.csv",
        "- hydrograph_peak_lag_summary.csv",
        "- scatter_cdf_summary.csv",
    ]
    (PLOTS_OUT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
