from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sympy
import torch
import torch.nn as nn
import xarray as xr


JULIA_EXE = "/home/kongw0a/miniconda3/envs/mtslstm_symtorch/julia_env/pyjuliapkg/install/bin/julia"
os.environ.setdefault("PYTHON_JULIACALL_EXE", JULIA_EXE)

from symtorch import SymbolicModel  # noqa: E402


ROOT = Path("/home/kongw0a/hourly_streamflow_dl")
CODE_DIR = ROOT / "MTSLSTM_100stations" / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import tune_baseline_lstm_daily_s2_random30 as daily_tune  # noqa: E402
from Modelzoo import LSTM  # noqa: E402


BEST_RUN_JSON = (
    ROOT
    / "MTSLSTM_100stations"
    / "outputs"
    / "baseline_lstm_daily_s2_random30_tuning"
    / "best_run_by_val_kge.json"
)
OUT_ROOT = (
    ROOT
    / "MTSLSTM_100stations"
    / "outputs"
    / "baseline_lstm_daily_s2_random30_symtorch_hydro_valkge"
)

EXPERIMENTS = [
    {"name": "hydro_rawstd_residual", "target_mode": "rawstd", "event_quantile": None},
    {"name": "hydro_log_residual", "target_mode": "log", "event_quantile": None},
    {"name": "hydro_log_residual_eventq75", "target_mode": "log", "event_quantile": 0.75},
]

ROLLING_SUM_WINDOWS = [1, 3, 7, 14, 30, 60, 90]
ROLLING_MEAN_WINDOWS = [1, 3, 7, 14, 30, 60, 90]


class DummyModel(nn.Module):
    def __init__(self, y: torch.Tensor):
        super().__init__()
        if y.ndim == 1:
            y = y.unsqueeze(1)
        self.register_buffer("y", y)

    def forward(self, x):
        return self.y


def sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")


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
    std_obs = np.std(obs)
    mean_obs = np.mean(obs)
    if std_obs == 0 or mean_obs == 0:
        return float("nan")
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / std_obs
    beta = np.mean(sim) / mean_obs
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def build_numpy_symbolic_function(
    equations_df: pd.DataFrame,
    best_equation_payload: dict,
    variable_names: list[str],
):
    locals_map = {
        "square": lambda x: x**2,
        "sqrt_abs": lambda x: sympy.sqrt(sympy.Abs(x) + sympy.Float("1e-6")),
        "log_abs": lambda x: sympy.log(sympy.Abs(x) + sympy.Float("1e-6")),
    }
    match = equations_df.loc[equations_df["equation"].eq(best_equation_payload["equation"])].copy()
    if match.empty:
        raise ValueError("Could not find the best equation row in equations table.")
    expr = sympy.sympify(str(match.iloc[0]["sympy_format"]), locals=locals_map)
    vars_sorted = sorted(expr.free_symbols, key=lambda s: str(s))
    func = sympy.lambdify(vars_sorted, expr, modules=["numpy"])
    var_indices = [variable_names.index(str(var)) for var in vars_sorted]

    def symbolic_func(x: np.ndarray) -> np.ndarray:
        cols = [x[:, idx] for idx in var_indices]
        out = func(*cols)
        return np.asarray(out, dtype=np.float64).reshape(-1)

    return symbolic_func


def summarize_target_fidelity(split: str, y_ref: np.ndarray, y_hat: np.ndarray) -> dict[str, float | str]:
    mask = np.isfinite(y_ref) & np.isfinite(y_hat)
    y_ref = y_ref[mask]
    y_hat = y_hat[mask]
    if y_ref.size == 0:
        return {
            "split": split,
            "rmse_target": float("nan"),
            "mae_target": float("nan"),
            "r2_target": float("nan"),
            "corr_target": float("nan"),
        }
    err = y_hat - y_ref
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_ref - np.mean(y_ref)) ** 2))
    r2 = float("nan") if denom == 0 else float(1 - np.sum(err**2) / denom)
    corr = float(np.corrcoef(y_ref, y_hat)[0, 1]) if len(y_ref) > 1 else float("nan")
    return {
        "split": split,
        "rmse_target": rmse,
        "mae_target": mae,
        "r2_target": r2,
        "corr_target": corr,
    }


def summarize_hydrology(
    split: str,
    station_ids: np.ndarray,
    y_true_raw: np.ndarray,
    y_lstm_raw: np.ndarray,
    y_corrected_raw: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    rows = []
    for station_id in sorted(np.unique(station_ids)):
        mask = station_ids == station_id
        obs = y_true_raw[mask]
        lstm = y_lstm_raw[mask]
        corrected = y_corrected_raw[mask]
        rows.append(
            {
                "split": split,
                "station_id": str(station_id),
                "samples": int(mask.sum()),
                "lstm_nse": compute_nse(obs, lstm),
                "lstm_kge": compute_kge(obs, lstm),
                "corrected_nse": compute_nse(obs, corrected),
                "corrected_kge": compute_kge(obs, corrected),
            }
        )
    per_station = pd.DataFrame(rows)
    valid = per_station.copy()
    summary = {
        "split": split,
        "n_stations": int(len(valid)),
        "lstm_median_nse": float(valid["lstm_nse"].median()),
        "lstm_median_kge": float(valid["lstm_kge"].median()),
        "corrected_median_nse": float(valid["corrected_nse"].median()),
        "corrected_median_kge": float(valid["corrected_kge"].median()),
        "lstm_negative_nse_stations": int((valid["lstm_nse"] < 0).sum()),
        "lstm_negative_kge_stations": int((valid["lstm_kge"] < 0).sum()),
        "corrected_negative_nse_stations": int((valid["corrected_nse"] < 0).sum()),
        "corrected_negative_kge_stations": int((valid["corrected_kge"] < 0).sum()),
    }
    return per_station, summary


def load_best_daily_lstm():
    best_run = json.loads(BEST_RUN_JSON.read_text(encoding="utf-8"))
    run_dir = Path(best_run["run_dir"])
    lookback = int(best_run["lookback_days"])
    hidden_size = int(best_run["hidden_size"])
    dropout = float(best_run["dropout"])

    station_ids = daily_tune.load_station_ids()
    static_df = daily_tune.load_static_df(station_ids)
    prep_dir = OUT_ROOT / "prep_cache"
    prep_dir.mkdir(parents=True, exist_ok=True)
    prepared = daily_tune.build_standardized_splits(station_ids, static_df, prep_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM(
        input_size=len(daily_tune.DYNAMIC_VARS) + prepared["train_static_std"].shape[1],
        hidden_size=hidden_size,
        num_layers=1,
        dropout=dropout,
        output_size=1,
    ).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pth", map_location=device))
    model.eval()
    return best_run, model, prepared, static_df, lookback, device


def collect_split_predictions(
    split_name: str,
    dyn_std: xr.Dataset,
    y_std_ds: xr.Dataset,
    static_std: pd.DataFrame,
    model: torch.nn.Module,
    lookback: int,
    device: torch.device,
    y_mean: float,
    y_std: float,
    batch_size: int = 2048,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for station_id in [str(s) for s in dyn_std.data_vars]:
            x = np.asarray(dyn_std[station_id].transpose("time", "dynamic_forcing").values, dtype=np.float32)
            y = np.asarray(y_std_ds[station_id].values, dtype=np.float32).reshape(-1)
            dates = pd.to_datetime(dyn_std[station_id].coords["time"].values)
            x_static = np.asarray(static_std.loc[station_id].values, dtype=np.float32)

            valid_t: list[int] = []
            windows: list[np.ndarray] = []
            for t in range(lookback - 1, len(dates)):
                x_win = x[t - lookback + 1 : t + 1]
                y_t = y[t]
                if np.isnan(x_win).any() or not np.isfinite(y_t):
                    continue
                valid_t.append(t)
                windows.append(x_win)

            if not valid_t:
                continue

            preds_std_parts = []
            for start in range(0, len(valid_t), batch_size):
                end = min(start + batch_size, len(valid_t))
                x_batch = np.stack(windows[start:end], axis=0)
                s_batch = np.repeat(x_static[np.newaxis, :], len(x_batch), axis=0)
                preds = model(
                    (
                        torch.from_numpy(x_batch).to(device),
                        torch.from_numpy(s_batch).to(device),
                    )
                ).detach().cpu().numpy().reshape(-1)
                preds_std_parts.append(preds)
            preds_std = np.concatenate(preds_std_parts, axis=0)

            for pred_std, t in zip(preds_std, valid_t):
                y_std_t = float(y[t])
                rows.append(
                    {
                        "split": split_name,
                        "station_id": station_id,
                        "date": pd.Timestamp(dates[t]),
                        "y_true_std": y_std_t,
                        "y_lstm_std": float(pred_std),
                        "y_true_raw": float(y_std_t * y_std + y_mean),
                        "y_lstm_raw": float(pred_std * y_std + y_mean),
                    }
                )
    return pd.DataFrame(rows)


def build_hydrologic_feature_table(daily_data_path: Path, static_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    with xr.open_dataset(daily_data_path) as ds:
        for station_id in static_df.index.astype(str):
            station_df = ds[station_id].to_pandas()
            station_df.index = pd.to_datetime(station_df.index)
            rain = station_df["Rainf"].astype(np.float64)
            tair = station_df["Tair"].astype(np.float64)
            pet = station_df["PotEvap"].astype(np.float64)

            frame = pd.DataFrame(index=station_df.index)
            frame["station_id"] = station_id

            for w in ROLLING_SUM_WINDOWS:
                frame[f"rain_sum_{w}"] = rain.rolling(w, min_periods=w).sum()
                frame[f"pet_sum_{w}"] = pet.rolling(w, min_periods=w).sum()
            for w in ROLLING_MEAN_WINDOWS:
                frame[f"tair_mean_{w}"] = tair.rolling(w, min_periods=w).mean()

            frame["wetness_7"] = frame["rain_sum_7"] - frame["pet_sum_7"]
            frame["wetness_30"] = frame["rain_sum_30"] - frame["pet_sum_30"]
            frame["wetness_90"] = frame["rain_sum_90"] - frame["pet_sum_90"]

            frame["api_7"] = rain.ewm(span=7, adjust=False).mean()
            frame["api_30"] = rain.ewm(span=30, adjust=False).mean()

            frame["rain_pet_logratio_30"] = np.log((frame["rain_sum_30"] + 1e-6) / (frame["pet_sum_30"] + 1e-6))
            frame["rain_pet_logratio_90"] = np.log((frame["rain_sum_90"] + 1e-6) / (frame["pet_sum_90"] + 1e-6))
            frame["storminess_logratio_3_30"] = np.log((frame["rain_sum_3"] + 1e-6) / (frame["rain_sum_30"] + 1e-6))
            frame["storminess_logratio_7_90"] = np.log((frame["rain_sum_7"] + 1e-6) / (frame["rain_sum_90"] + 1e-6))
            frame["tair_diff_7_30"] = frame["tair_mean_7"] - frame["tair_mean_30"]
            frame["tair_diff_30_90"] = frame["tair_mean_30"] - frame["tair_mean_90"]

            doy = frame.index.dayofyear.to_numpy(dtype=np.float64)
            frame["doy_sin"] = np.sin(2.0 * np.pi * doy / 366.0)
            frame["doy_cos"] = np.cos(2.0 * np.pi * doy / 366.0)

            for col in static_df.columns:
                frame[sanitize_name(col)] = float(static_df.loc[station_id, col])

            frame = frame.reset_index(names="date")
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def standardize_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    out = df.copy()
    train_df = out.loc[out["split"].eq("train")].copy()
    scalers: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        values = train_df[col].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        mean = float(values.mean()) if values.size else 0.0
        std = float(values.std()) if values.size else 1.0
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        scalers[col] = {"mean": mean, "std": std}
        out[col] = (out[col] - mean) / std
    return out, scalers


def build_merged_table() -> tuple[pd.DataFrame, list[str], dict]:
    best_run, model, prepared, static_df, lookback, device = load_best_daily_lstm()
    y_mean = float(prepared["scalers"]["y_mean"])
    y_std = float(prepared["scalers"]["y_std"])

    split_frames = []
    for split_name in ["train", "val", "test"]:
        split_frames.append(
            collect_split_predictions(
                split_name=split_name,
                dyn_std=prepared[f"{split_name}_dyn_std"],
                y_std_ds=prepared[f"{split_name}_y_std"],
                static_std=prepared[f"{split_name}_static_std"],
                model=model,
                lookback=lookback,
                device=device,
                y_mean=y_mean,
                y_std=y_std,
            )
        )
    sample_df = pd.concat(split_frames, ignore_index=True)

    daily_data_path = OUT_ROOT / "prep_cache" / "daily_selected_stn_data.nc"
    feature_df = build_hydrologic_feature_table(daily_data_path, static_df)
    merged = sample_df.merge(feature_df, on=["station_id", "date"], how="left", validate="one_to_one")
    merged["lstm_pred_raw"] = merged["y_lstm_raw"]
    merged["lstm_pred_log1p"] = np.log1p(np.clip(merged["y_lstm_raw"], a_min=0.0, a_max=None))

    static_feature_cols = [sanitize_name(col) for col in static_df.columns]
    dynamic_feature_cols = [
        f"rain_sum_{w}" for w in ROLLING_SUM_WINDOWS
    ] + [
        f"pet_sum_{w}" for w in ROLLING_SUM_WINDOWS
    ] + [
        f"tair_mean_{w}" for w in ROLLING_MEAN_WINDOWS
    ] + [
        "wetness_7",
        "wetness_30",
        "wetness_90",
        "api_7",
        "api_30",
        "rain_pet_logratio_30",
        "rain_pet_logratio_90",
        "storminess_logratio_3_30",
        "storminess_logratio_7_90",
        "tair_diff_7_30",
        "tair_diff_30_90",
        "doy_sin",
        "doy_cos",
        "lstm_pred_raw",
        "lstm_pred_log1p",
    ]
    feature_cols = dynamic_feature_cols + static_feature_cols

    merged, feature_scalers = standardize_feature_columns(merged, feature_cols)
    meta = {
        "best_run": best_run,
        "lookback_days": lookback,
        "y_mean": y_mean,
        "y_std": y_std,
        "feature_cols": feature_cols,
        "feature_scalers": feature_scalers,
        "n_rows": int(len(merged)),
    }
    return merged, feature_cols, meta


def train_station_thresholds(train_df: pd.DataFrame, quantile: float) -> dict[str, float]:
    series = train_df.groupby("station_id", sort=True)["y_lstm_raw"].quantile(quantile)
    return {str(idx): float(val) for idx, val in series.items()}


def build_experiment_targets(df: pd.DataFrame, target_mode: str) -> tuple[pd.Series, pd.Series]:
    if target_mode == "rawstd":
        target = df["y_true_std"] - df["y_lstm_std"]
        correction_scale = pd.Series(1.0, index=df.index, dtype=np.float64)
        return target.astype(np.float64), correction_scale
    if target_mode == "log":
        log_obs = np.log1p(np.clip(df["y_true_raw"].to_numpy(dtype=np.float64), a_min=0.0, a_max=None))
        log_lstm = np.log1p(np.clip(df["y_lstm_raw"].to_numpy(dtype=np.float64), a_min=0.0, a_max=None))
        target = log_obs - log_lstm
        correction_scale = pd.Series(1.0, index=df.index, dtype=np.float64)
        return pd.Series(target, index=df.index, dtype=np.float64), correction_scale
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def evaluate_experiment(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_mode: str,
    best_equation_payload: dict,
    equations_df: pd.DataFrame,
    y_std_scale: float,
    event_thresholds: dict[str, float] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sym_func = build_numpy_symbolic_function(equations_df, best_equation_payload, feature_cols)

    fidelity_rows = []
    hydrology_rows = []
    hydrology_frames = []
    prediction_frames = []

    for split_name in ["train", "val", "test"]:
        split_df = df.loc[df["split"].eq(split_name)].copy().reset_index(drop=True)
        X = split_df[feature_cols].to_numpy(dtype=np.float64)
        target_true, _ = build_experiment_targets(split_df, target_mode)
        pred_target = np.zeros(len(split_df), dtype=np.float64)
        finite_feature_mask = np.isfinite(X).all(axis=1)
        if finite_feature_mask.any():
            pred_target[finite_feature_mask] = np.asarray(
                sym_func(X[finite_feature_mask]),
                dtype=np.float64,
            ).reshape(-1)

        apply_mask = np.ones(len(split_df), dtype=bool)
        if event_thresholds is not None:
            thresholds = split_df["station_id"].map(event_thresholds).to_numpy(dtype=np.float64)
            apply_mask = split_df["y_lstm_raw"].to_numpy(dtype=np.float64) >= thresholds
            pred_target = np.where(apply_mask, pred_target, 0.0)

        fidelity_rows.append(summarize_target_fidelity(split_name, target_true.to_numpy(dtype=np.float64), pred_target))

        y_true_raw = split_df["y_true_raw"].to_numpy(dtype=np.float64)
        y_lstm_raw = split_df["y_lstm_raw"].to_numpy(dtype=np.float64)
        if target_mode == "rawstd":
            y_corrected_raw = np.clip(y_lstm_raw + pred_target * y_std_scale, a_min=0.0, a_max=None)
        else:
            log_lstm = np.log1p(np.clip(y_lstm_raw, a_min=0.0, a_max=None))
            y_corrected_raw = np.clip(np.expm1(log_lstm + pred_target), a_min=0.0, a_max=None)

        frame, summary = summarize_hydrology(
            split_name,
            split_df["station_id"].to_numpy(dtype=str),
            y_true_raw,
            y_lstm_raw,
            y_corrected_raw,
        )
        hydrology_frames.append(frame)
        hydrology_rows.append(summary)

        prediction_frames.append(
            pd.DataFrame(
                {
                    "split": split_name,
                    "station_id": split_df["station_id"].to_numpy(dtype=str),
                    "date": split_df["date"].to_numpy(),
                    "y_true_raw": y_true_raw,
                    "y_lstm_raw": y_lstm_raw,
                    "target_true": target_true.to_numpy(dtype=np.float64),
                    "target_symbolic": pred_target,
                    "y_corrected_raw": y_corrected_raw,
                    "apply_mask": apply_mask.astype(np.int8),
                }
            )
        )

    fidelity_df = pd.DataFrame(fidelity_rows)
    hydrology_summary_df = pd.DataFrame(hydrology_rows)
    hydrology_per_station_df = pd.concat(hydrology_frames, ignore_index=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    return fidelity_df, hydrology_summary_df, hydrology_per_station_df, prediction_df


def run_single_experiment(
    merged_df: pd.DataFrame,
    feature_cols: list[str],
    meta: dict,
    name: str,
    target_mode: str,
    event_quantile: float | None,
    select_k_features: int = 28,
    niterations: int = 220,
    maxsize: int = 45,
    per_station_cap: int = 500,
) -> dict[str, object]:
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = merged_df.loc[merged_df["split"].eq("train")].copy()
    train_target, _ = build_experiment_targets(train_df, target_mode)
    train_df["target_value"] = train_target

    event_thresholds = None
    if event_quantile is not None:
        event_thresholds = train_station_thresholds(train_df, quantile=float(event_quantile))
        train_df = train_df.loc[
            train_df["y_lstm_raw"] >= train_df["station_id"].map(event_thresholds).to_numpy(dtype=np.float64)
        ].copy()

    finite_mask = np.isfinite(train_df["target_value"].to_numpy(dtype=np.float64))
    finite_mask &= np.isfinite(train_df[feature_cols].to_numpy(dtype=np.float64)).all(axis=1)
    train_df = train_df.loc[finite_mask].copy()

    sampled = []
    rng = np.random.default_rng(42)
    for station_id, station_df in train_df.groupby("station_id", sort=True):
        if len(station_df) > per_station_cap:
            keep_idx = rng.choice(station_df.index.to_numpy(), size=per_station_cap, replace=False)
            station_df = station_df.loc[np.sort(keep_idx)].copy()
        sampled.append(station_df)
    train_df = pd.concat(sampled, ignore_index=True)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["target_value"].to_numpy(dtype=np.float32)

    equations_path = out_dir / "equations_dim0.csv"
    best_equation_path = out_dir / "best_equation.json"
    selected_features_path = out_dir / "selected_features.json"

    if equations_path.exists() and best_equation_path.exists() and selected_features_path.exists():
        equations_df = pd.read_csv(equations_path)
        best_equation_payload = json.loads(best_equation_path.read_text(encoding="utf-8"))
        selected_feature_names = json.loads(selected_features_path.read_text(encoding="utf-8"))
    else:
        dummy_model = DummyModel(torch.from_numpy(y_train))
        symbolic_model = SymbolicModel(dummy_model, name)

        sr_params = {
            "niterations": niterations,
            "maxsize": maxsize,
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": [
                "sin",
                "cos",
                "exp",
                "tanh",
                "square(x)=x^2",
                "sqrt_abs(x)=sqrt(abs(x) + 1.0f-6)",
                "log_abs(x)=log(abs(x) + 1.0f-6)",
            ],
            "extra_sympy_mappings": {
                "square": lambda x: x**2,
                "sqrt_abs": lambda x: sympy.sqrt(sympy.Abs(x) + sympy.Float("1e-6")),
                "log_abs": lambda x: sympy.log(sympy.Abs(x) + sympy.Float("1e-6")),
            },
            "complexity_of_constants": 1,
            "complexity_of_operators": {
                "sin": 3,
                "cos": 3,
                "exp": 4,
                "tanh": 3,
                "square": 2,
                "sqrt_abs": 4,
                "log_abs": 4,
            },
            "elementwise_loss": "loss(prediction, target) = (prediction - target)^2",
            "parsimony": 0.0025,
            "batching": True,
            "batch_size": 256,
            "random_state": 42,
            "model_selection": "best",
            "select_k_features": select_k_features,
            "temp_equation_file": False,
        }
        fit_params = {"variable_names": feature_cols}

        symbolic_model.distill(
            torch.from_numpy(X_train),
            sr_params=sr_params,
            fit_params=fit_params,
            save_path=str(out_dir),
        )

        regressor = symbolic_model.pysr_regressor[0]
        equations_df = regressor.equations_.copy()
        equations_df.to_csv(equations_path, index=False)

        selection_mask = getattr(regressor, "selection_mask_", None)
        selected_feature_names = []
        if selection_mask is not None:
            selected_feature_names = [name for name, keep in zip(feature_cols, selection_mask) if keep]
        selected_features_path.write_text(json.dumps(selected_feature_names, indent=2) + "\n", encoding="utf-8")

        best_equation = regressor.get_best()
        best_equation_payload = {
            "equation": str(best_equation["equation"]),
            "complexity": int(best_equation["complexity"]),
            "loss": float(best_equation["loss"]),
            "score": float(best_equation["score"]) if "score" in best_equation else None,
            "selected_feature_names": selected_feature_names,
        }
        best_equation_path.write_text(json.dumps(best_equation_payload, indent=2) + "\n", encoding="utf-8")

    fidelity_df, hydrology_summary_df, hydrology_per_station_df, prediction_df = evaluate_experiment(
        df=merged_df,
        feature_cols=feature_cols,
        target_mode=target_mode,
        best_equation_payload=best_equation_payload,
        equations_df=equations_df,
        y_std_scale=float(meta["y_std"]),
        event_thresholds=event_thresholds,
    )

    fidelity_df.to_csv(out_dir / "target_fidelity_summary.csv", index=False)
    hydrology_summary_df.to_csv(out_dir / "corrected_hydrology_summary.csv", index=False)
    hydrology_per_station_df.to_csv(out_dir / "corrected_hydrology_per_station.csv", index=False)
    prediction_df.to_csv(out_dir / "residual_symbolic_predictions.csv.gz", index=False, compression="gzip")

    exp_meta = {
        "name": name,
        "target_mode": target_mode,
        "event_quantile": event_quantile,
        "niterations": niterations,
        "maxsize": maxsize,
        "select_k_features": select_k_features,
        "per_station_cap": per_station_cap,
        "train_distill_samples": int(len(train_df)),
        "n_total_features": len(feature_cols),
        "selected_feature_names": best_equation_payload["selected_feature_names"],
    }
    (out_dir / "experiment_config.json").write_text(json.dumps(exp_meta, indent=2) + "\n", encoding="utf-8")

    summary_lines = [
        f"Hydrology-informed SymTorch experiment `{name}`",
        "",
        f"Target mode: {target_mode}",
        f"Event quantile gate: {event_quantile}",
        f"Train distill sample size: {len(train_df)}",
        f"Total candidate features: {len(feature_cols)}",
        f"PySR select_k_features: {select_k_features}",
        "",
        "Best symbolic equation:",
        best_equation_payload["equation"],
        "",
        f"Equation complexity: {best_equation_payload['complexity']}",
        f"Equation loss: {best_equation_payload['loss']:.8f}",
        "",
        "Selected features:",
    ]
    summary_lines.extend(f"- {name}" for name in best_equation_payload["selected_feature_names"])
    summary_lines.append("")
    summary_lines.append("Target fidelity:")
    for row in fidelity_df.itertuples(index=False):
        summary_lines.append(
            f"- {row.split}: RMSE={row.rmse_target:.6f}, MAE={row.mae_target:.6f}, "
            f"R2={row.r2_target:.6f}, corr={row.corr_target:.6f}"
        )
    summary_lines.append("")
    summary_lines.append("Hydrology metrics against observed daily streamflow:")
    for row in hydrology_summary_df.itertuples(index=False):
        summary_lines.append(
            f"- {row.split}: baseline dailyLSTM KGE={row.lstm_median_kge:.6f}, NSE={row.lstm_median_nse:.6f}; "
            f"corrected KGE={row.corrected_median_kge:.6f}, NSE={row.corrected_median_nse:.6f}"
        )
    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    result = {
        "name": name,
        "target_mode": target_mode,
        "event_quantile": event_quantile,
        "equation": best_equation_payload["equation"],
        "equation_complexity": best_equation_payload["complexity"],
        "equation_loss": best_equation_payload["loss"],
        "val_corrected_kge": float(
            hydrology_summary_df.loc[hydrology_summary_df["split"].eq("val"), "corrected_median_kge"].iloc[0]
        ),
        "val_corrected_nse": float(
            hydrology_summary_df.loc[hydrology_summary_df["split"].eq("val"), "corrected_median_nse"].iloc[0]
        ),
        "test_corrected_kge": float(
            hydrology_summary_df.loc[hydrology_summary_df["split"].eq("test"), "corrected_median_kge"].iloc[0]
        ),
        "test_corrected_nse": float(
            hydrology_summary_df.loc[hydrology_summary_df["split"].eq("test"), "corrected_median_nse"].iloc[0]
        ),
        "out_dir": str(out_dir),
    }
    return result


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    merged_df, feature_cols, meta = build_merged_table()

    merged_df.to_csv(OUT_ROOT / "merged_sample_feature_table.csv.gz", index=False, compression="gzip")
    (OUT_ROOT / "feature_cols.json").write_text(json.dumps(feature_cols, indent=2) + "\n", encoding="utf-8")
    (OUT_ROOT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    results = []
    for exp in EXPERIMENTS:
        result = run_single_experiment(
            merged_df=merged_df,
            feature_cols=feature_cols,
            meta=meta,
            name=exp["name"],
            target_mode=exp["target_mode"],
            event_quantile=exp["event_quantile"],
        )
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_ROOT / "experiment_comparison.csv", index=False)

    lines = [
        "Hydrology-informed SymTorch residual experiments on the validation-KGE-best daily BaselineLSTM",
        "",
        f"Total merged rows: {meta['n_rows']}",
        f"Total candidate features: {len(feature_cols)}",
        "",
        "Experiments:",
    ]
    for row in results_df.itertuples(index=False):
        lines.append(
            f"- {row.name}: val KGE={row.val_corrected_kge:.6f}, val NSE={row.val_corrected_nse:.6f}, "
            f"test KGE={row.test_corrected_kge:.6f}, test NSE={row.test_corrected_nse:.6f}"
        )
    (OUT_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
