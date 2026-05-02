from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sympy
import torch
import torch.nn as nn


JULIA_EXE = "/home/kongw0a/miniconda3/envs/mtslstm_symtorch/julia_env/pyjuliapkg/install/bin/julia"
os.environ.setdefault("PYTHON_JULIACALL_EXE", JULIA_EXE)

from symtorch import SymbolicModel  # noqa: E402


ROOT = Path("/home/kongw0a/hourly_streamflow_dl")
INPUT_DIR = (
    ROOT
    / "MTSLSTM_100stations"
    / "outputs"
    / "baseline_lstm_daily_s2_random30_symtorch_direct_valkge"
)
OUT_DIR = (
    ROOT
    / "MTSLSTM_100stations"
    / "outputs"
    / "baseline_lstm_daily_s2_random30_symtorch_direct_valkge"
    / "symtorch_direct_distill"
)


class DummyModel(nn.Module):
    def __init__(self, y: torch.Tensor):
        super().__init__()
        if y.ndim == 1:
            y = y.unsqueeze(1)
        self.register_buffer("y", y)

    def forward(self, x):
        return self.y


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


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


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
        raise ValueError("Could not find the best equation row in equations_dim0.csv.")
    expr_str = str(match.iloc[0]["sympy_format"])
    expr = sympy.sympify(expr_str, locals=locals_map)
    vars_sorted = sorted(expr.free_symbols, key=lambda s: str(s))
    func = sympy.lambdify(vars_sorted, expr, modules=["numpy"])
    var_indices = [variable_names.index(str(var)) for var in vars_sorted]

    def symbolic_func(x: np.ndarray) -> np.ndarray:
        cols = [x[:, idx] for idx in var_indices]
        out = func(*cols)
        return np.asarray(out, dtype=np.float64).reshape(-1)

    return symbolic_func


def summarize_distill_fidelity(split: str, y_ref: np.ndarray, y_hat: np.ndarray) -> dict[str, float | str]:
    err = y_hat - y_ref
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_ref - np.mean(y_ref)) ** 2))
    r2 = float("nan") if denom == 0 else float(1 - np.sum(err ** 2) / denom)
    corr = float(np.corrcoef(y_ref, y_hat)[0, 1]) if len(y_ref) > 1 else float("nan")
    return {
        "split": split,
        "rmse_std_vs_lstm": rmse,
        "mae_std_vs_lstm": mae,
        "r2_std_vs_lstm": r2,
        "corr_std_vs_lstm": corr,
    }


def summarize_hydrology(
    split: str,
    station_ids: np.ndarray,
    y_true_raw: np.ndarray,
    y_lstm_raw: np.ndarray,
    y_sym_raw: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    rows = []
    for station_id in sorted(np.unique(station_ids)):
        mask = station_ids == station_id
        obs = y_true_raw[mask]
        lstm = y_lstm_raw[mask]
        sym = y_sym_raw[mask]
        rows.append(
            {
                "split": split,
                "station_id": str(station_id),
                "samples": int(mask.sum()),
                "lstm_nse": compute_nse(obs, lstm),
                "lstm_kge": compute_kge(obs, lstm),
                "symbolic_nse": compute_nse(obs, sym),
                "symbolic_kge": compute_kge(obs, sym),
            }
        )

    per_station = pd.DataFrame(rows)
    valid = per_station.copy()
    summary = {
        "split": split,
        "n_stations": int(len(valid)),
        "lstm_median_nse": float(valid["lstm_nse"].median()),
        "lstm_median_kge": float(valid["lstm_kge"].median()),
        "symbolic_median_nse": float(valid["symbolic_nse"].median()),
        "symbolic_median_kge": float(valid["symbolic_kge"].median()),
        "lstm_negative_nse_stations": int((valid["lstm_nse"] < 0).sum()),
        "lstm_negative_kge_stations": int((valid["lstm_kge"] < 0).sum()),
        "symbolic_negative_nse_stations": int((valid["symbolic_nse"] < 0).sum()),
        "symbolic_negative_kge_stations": int((valid["symbolic_kge"] < 0).sum()),
    }
    return per_station, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = json.loads((INPUT_DIR / "export_metadata.json").read_text(encoding="utf-8"))
    variable_names = json.loads((INPUT_DIR / "variable_names.json").read_text(encoding="utf-8"))
    train_sample = load_npz(INPUT_DIR / "train_distill_sample.npz")
    val = load_npz(INPUT_DIR / "val_full.npz")
    test = load_npz(INPUT_DIR / "test_full.npz")
    train_full = load_npz(INPUT_DIR / "train_full.npz")

    X_train = train_sample["X_flat"].astype(np.float32)
    y_train_pred = train_sample["y_pred_std"].astype(np.float32)

    equations_path = OUT_DIR / "equations_dim0.csv"
    best_equation_path = OUT_DIR / "best_equation.json"
    selected_features_path = OUT_DIR / "selected_features.json"
    distill_config = {
        "niterations": 180,
        "maxsize": 35,
        "select_k_features": 16,
    }

    if equations_path.exists() and best_equation_path.exists() and selected_features_path.exists():
        equations_df = pd.read_csv(equations_path)
        best_equation_payload = json.loads(best_equation_path.read_text(encoding="utf-8"))
        selected_feature_names = json.loads(selected_features_path.read_text(encoding="utf-8"))
    else:
        dummy_model = DummyModel(torch.from_numpy(y_train_pred))
        symbolic_model = SymbolicModel(dummy_model, "daily_lstm_direct_timewindow_distill")

        sr_params = {
            "niterations": distill_config["niterations"],
            "maxsize": distill_config["maxsize"],
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
            "parsimony": 0.003,
            "batching": True,
            "batch_size": 256,
            "random_state": 42,
            "model_selection": "best",
            "select_k_features": distill_config["select_k_features"],
            "temp_equation_file": False,
        }
        fit_params = {"variable_names": variable_names}

        symbolic_model.distill(
            torch.from_numpy(X_train),
            sr_params=sr_params,
            fit_params=fit_params,
            save_path=str(OUT_DIR),
        )

        regressor = symbolic_model.pysr_regressor[0]
        equations_df = regressor.equations_.copy()
        equations_df.to_csv(equations_path, index=False)

        selection_mask = getattr(regressor, "selection_mask_", None)
        selected_feature_names = []
        if selection_mask is not None:
            selected_feature_names = [name for name, keep in zip(variable_names, selection_mask) if keep]
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

    sym_func = build_numpy_symbolic_function(equations_df, best_equation_payload, variable_names)

    fidelity_rows = []
    hydro_summaries = []
    hydro_frames = []
    prediction_frames = []

    y_mean = float(meta["scalers"]["y_mean"])
    y_std = float(meta["scalers"]["y_std"])

    for split_name, split_data in [("train", train_full), ("val", val), ("test", test)]:
        X = split_data["X_flat"].astype(np.float32)
        y_true_std = split_data["y_true_std"].astype(np.float64)
        y_lstm_std = split_data["y_pred_std"].astype(np.float64)
        station_ids = split_data["station_ids"].astype(str)

        y_sym_std = np.asarray(sym_func(X), dtype=np.float64).reshape(-1)

        fidelity_rows.append(summarize_distill_fidelity(split_name, y_lstm_std, y_sym_std))

        y_true_raw = y_true_std * y_std + y_mean
        y_lstm_raw = y_lstm_std * y_std + y_mean
        y_sym_raw = y_sym_std * y_std + y_mean

        hydro_frame, hydro_summary = summarize_hydrology(
            split_name,
            station_ids,
            y_true_raw,
            y_lstm_raw,
            y_sym_raw,
        )
        hydro_frames.append(hydro_frame)
        hydro_summaries.append(hydro_summary)

        prediction_frames.append(
            pd.DataFrame(
                {
                    "split": split_name,
                    "station_id": station_ids,
                    "y_true_std": y_true_std,
                    "y_lstm_std": y_lstm_std,
                    "y_symbolic_std": y_sym_std,
                    "y_true_raw": y_true_raw,
                    "y_lstm_raw": y_lstm_raw,
                    "y_symbolic_raw": y_sym_raw,
                }
            )
        )

    fidelity_df = pd.DataFrame(fidelity_rows)
    hydro_summary_df = pd.DataFrame(hydro_summaries)
    hydro_station_df = pd.concat(hydro_frames, ignore_index=True)
    pred_df = pd.concat(prediction_frames, ignore_index=True)

    fidelity_df.to_csv(OUT_DIR / "fidelity_summary.csv", index=False)
    hydro_summary_df.to_csv(OUT_DIR / "hydrology_summary.csv", index=False)
    hydro_station_df.to_csv(OUT_DIR / "hydrology_per_station.csv", index=False)
    pred_df.to_csv(OUT_DIR / "symbolic_vs_lstm_predictions.csv.gz", index=False, compression="gzip")

    summary_lines = [
        "Direct SymTorch distillation of the validation-KGE-best daily BaselineLSTM",
        "",
        "Workflow:",
        "- Follow the MJO_prediction.ipynb logic",
        "- Use flattened daily input windows directly",
        "- Use a DummyModel whose output is the trained dailyLSTM prediction on the distillation sample",
        "- Distill one symbolic equation for the dailyLSTM output",
        "",
        f"Julia executable: {JULIA_EXE}",
        f"Train distill sample size: {len(X_train)}",
        f"Total flattened input dimension: {len(variable_names)}",
        f"PySR select_k_features: {distill_config['select_k_features']}",
        "",
        "Best symbolic equation:",
        best_equation_payload["equation"],
        "",
        f"Equation complexity: {best_equation_payload['complexity']}",
        f"Equation loss: {best_equation_payload['loss']:.8f}",
        "",
        "Selected features used by feature selection:",
    ]
    summary_lines.extend(f"- {name}" for name in selected_feature_names)
    summary_lines.append("")
    summary_lines.append("Fidelity to dailyLSTM predictions:")
    for row in fidelity_df.itertuples(index=False):
        summary_lines.append(
            f"- {row.split}: RMSE(std)={row.rmse_std_vs_lstm:.6f}, "
            f"MAE(std)={row.mae_std_vs_lstm:.6f}, R2(std)={row.r2_std_vs_lstm:.6f}, "
            f"corr(std)={row.corr_std_vs_lstm:.6f}"
        )
    summary_lines.append("")
    summary_lines.append("Hydrology metrics against observed daily streamflow:")
    for row in hydro_summary_df.itertuples(index=False):
        summary_lines.append(
            f"- {row.split}: LSTM median KGE={row.lstm_median_kge:.6f}, "
            f"LSTM median NSE={row.lstm_median_nse:.6f}, "
            f"Symbolic median KGE={row.symbolic_median_kge:.6f}, "
            f"Symbolic median NSE={row.symbolic_median_nse:.6f}"
        )

    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps(best_equation_payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
