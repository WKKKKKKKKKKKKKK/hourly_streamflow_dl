from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sympy


ROOT = Path("/home/kongw0a/hourly_streamflow_dl")
HYDRO_ROOT = (
    ROOT
    / "MTSLSTM_100stations"
    / "outputs"
    / "baseline_lstm_daily_s2_random30_symtorch_hydro_valkge"
)
GLOBAL_DIR = HYDRO_ROOT / "hydro_log_residual"
EVENT_DIR = HYDRO_ROOT / "hydro_log_residual_eventq75"
OUT_DIR = HYDRO_ROOT / "hybrid_log_residual_smoothgate"


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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def summarize_split(split_df: pd.DataFrame, pred_raw: np.ndarray) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    rows = []
    station_ids = split_df["station_id"].to_numpy(dtype=str)
    obs_all = split_df["y_true_raw"].to_numpy(dtype=np.float64)
    lstm_all = split_df["y_lstm_raw"].to_numpy(dtype=np.float64)
    for station_id in sorted(np.unique(station_ids)):
        mask = station_ids == station_id
        obs = obs_all[mask]
        lstm = lstm_all[mask]
        corrected = pred_raw[mask]
        rows.append(
            {
                "station_id": station_id,
                "samples": int(mask.sum()),
                "lstm_nse": compute_nse(obs, lstm),
                "lstm_kge": compute_kge(obs, lstm),
                "corrected_nse": compute_nse(obs, corrected),
                "corrected_kge": compute_kge(obs, corrected),
            }
        )
    per_station = pd.DataFrame(rows)
    summary = {
        "n_stations": int(len(per_station)),
        "lstm_median_nse": float(per_station["lstm_nse"].median()),
        "lstm_median_kge": float(per_station["lstm_kge"].median()),
        "corrected_median_nse": float(per_station["corrected_nse"].median()),
        "corrected_median_kge": float(per_station["corrected_kge"].median()),
        "lstm_negative_nse_stations": int((per_station["lstm_nse"] < 0).sum()),
        "lstm_negative_kge_stations": int((per_station["lstm_kge"] < 0).sum()),
        "corrected_negative_nse_stations": int((per_station["corrected_nse"] < 0).sum()),
        "corrected_negative_kge_stations": int((per_station["corrected_kge"] < 0).sum()),
    }
    return summary, per_station


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = json.loads((HYDRO_ROOT / "feature_cols.json").read_text(encoding="utf-8"))
    merged = pd.read_csv(HYDRO_ROOT / "merged_sample_feature_table.csv.gz", parse_dates=["date"])

    global_equations = pd.read_csv(GLOBAL_DIR / "equations_dim0.csv")
    global_best = json.loads((GLOBAL_DIR / "best_equation.json").read_text(encoding="utf-8"))
    event_equations = pd.read_csv(EVENT_DIR / "equations_dim0.csv")
    event_best = json.loads((EVENT_DIR / "best_equation.json").read_text(encoding="utf-8"))

    global_func = build_numpy_symbolic_function(global_equations, global_best, feature_cols)
    event_func = build_numpy_symbolic_function(event_equations, event_best, feature_cols)

    X = merged[feature_cols].to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(X).all(axis=1)
    global_pred = np.zeros(len(merged), dtype=np.float64)
    event_pred = np.zeros(len(merged), dtype=np.float64)
    if finite_mask.any():
        global_pred[finite_mask] = global_func(X[finite_mask])
        event_pred[finite_mask] = event_func(X[finite_mask])

    merged["global_log_residual"] = global_pred
    merged["event_log_residual_raw"] = event_pred
    merged["log_lstm"] = np.log1p(np.clip(merged["y_lstm_raw"].to_numpy(dtype=np.float64), a_min=0.0, a_max=None))

    train_df = merged.loc[merged["split"].eq("train")].copy()
    thresholds = train_df.groupby("station_id", sort=True)["log_lstm"].quantile(0.75).to_dict()
    merged["log_threshold_q75"] = merged["station_id"].map(thresholds).astype(np.float64)

    val_df = merged.loc[merged["split"].eq("val")].copy().reset_index(drop=True)
    test_df = merged.loc[merged["split"].eq("test")].copy().reset_index(drop=True)

    alpha_grid = [0.4, 0.6, 0.8, 1.0, 1.2]
    beta_grid = [0.2, 0.4, 0.6, 0.8, 1.0]
    tau_grid = [-0.4, -0.2, 0.0, 0.2, 0.4]
    sharpness_grid = [0.05, 0.1, 0.2, 0.35]

    search_rows = []
    best_payload = None

    for alpha in alpha_grid:
        for beta in beta_grid:
            for tau in tau_grid:
                for sharpness in sharpness_grid:
                    gate = sigmoid((val_df["log_lstm"].to_numpy(dtype=np.float64) - (val_df["log_threshold_q75"].to_numpy(dtype=np.float64) + tau)) / sharpness)
                    corrected_log = (
                        val_df["log_lstm"].to_numpy(dtype=np.float64)
                        + alpha * val_df["global_log_residual"].to_numpy(dtype=np.float64)
                        + beta * gate * val_df["event_log_residual_raw"].to_numpy(dtype=np.float64)
                    )
                    corrected_raw = np.clip(np.expm1(corrected_log), a_min=0.0, a_max=None)
                    summary, _ = summarize_split(val_df, corrected_raw)
                    row = {
                        "alpha": alpha,
                        "beta": beta,
                        "tau": tau,
                        "sharpness": sharpness,
                        "val_corrected_kge": summary["corrected_median_kge"],
                        "val_corrected_nse": summary["corrected_median_nse"],
                        "val_negative_kge_stations": summary["corrected_negative_kge_stations"],
                        "val_negative_nse_stations": summary["corrected_negative_nse_stations"],
                    }
                    search_rows.append(row)
                    key = (
                        row["val_corrected_kge"],
                        row["val_corrected_nse"],
                        -row["val_negative_kge_stations"],
                        -row["val_negative_nse_stations"],
                    )
                    if best_payload is None or key > best_payload["key"]:
                        best_payload = {"key": key, **row}

    search_df = pd.DataFrame(search_rows).sort_values(
        ["val_corrected_kge", "val_corrected_nse", "val_negative_kge_stations", "val_negative_nse_stations"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    search_df.to_csv(OUT_DIR / "grid_search.csv", index=False)

    alpha = float(best_payload["alpha"])
    beta = float(best_payload["beta"])
    tau = float(best_payload["tau"])
    sharpness = float(best_payload["sharpness"])

    all_prediction_frames = []
    summary_rows = []
    per_station_frames = []
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        gate = sigmoid((split_df["log_lstm"].to_numpy(dtype=np.float64) - (split_df["log_threshold_q75"].to_numpy(dtype=np.float64) + tau)) / sharpness)
        corrected_log = (
            split_df["log_lstm"].to_numpy(dtype=np.float64)
            + alpha * split_df["global_log_residual"].to_numpy(dtype=np.float64)
            + beta * gate * split_df["event_log_residual_raw"].to_numpy(dtype=np.float64)
        )
        corrected_raw = np.clip(np.expm1(corrected_log), a_min=0.0, a_max=None)

        summary, per_station = summarize_split(split_df, corrected_raw)
        summary["split"] = split_name
        summary_rows.append(summary)

        per_station["split"] = split_name
        per_station_frames.append(per_station)

        all_prediction_frames.append(
            pd.DataFrame(
                {
                    "split": split_name,
                    "station_id": split_df["station_id"].to_numpy(dtype=str),
                    "date": split_df["date"].to_numpy(),
                    "y_true_raw": split_df["y_true_raw"].to_numpy(dtype=np.float64),
                    "y_lstm_raw": split_df["y_lstm_raw"].to_numpy(dtype=np.float64),
                    "log_lstm": split_df["log_lstm"].to_numpy(dtype=np.float64),
                    "global_log_residual": split_df["global_log_residual"].to_numpy(dtype=np.float64),
                    "event_log_residual_raw": split_df["event_log_residual_raw"].to_numpy(dtype=np.float64),
                    "gate": gate,
                    "y_corrected_raw": corrected_raw,
                }
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    per_station_df = pd.concat(per_station_frames, ignore_index=True)
    predictions_df = pd.concat(all_prediction_frames, ignore_index=True)

    summary_df.to_csv(OUT_DIR / "corrected_hydrology_summary.csv", index=False)
    per_station_df.to_csv(OUT_DIR / "corrected_hydrology_per_station.csv", index=False)
    predictions_df.to_csv(OUT_DIR / "hybrid_predictions.csv.gz", index=False, compression="gzip")

    best_meta = {
        "alpha": alpha,
        "beta": beta,
        "tau": tau,
        "sharpness": sharpness,
        "global_equation": global_best["equation"],
        "event_equation": event_best["equation"],
    }
    (OUT_DIR / "best_params.json").write_text(json.dumps(best_meta, indent=2) + "\n", encoding="utf-8")

    lines = [
        "Hybrid smooth-gated fusion of hydrology-informed symbolic log-residual corrections",
        "",
        "Correction form:",
        "log(1+q_corr) = log(1+q_lstm) + alpha * g_global(x) + beta * sigmoid((log(1+q_lstm) - (log_q75 + tau))/sharpness) * g_event(x)",
        "",
        f"Best alpha: {alpha}",
        f"Best beta: {beta}",
        f"Best tau: {tau}",
        f"Best sharpness: {sharpness}",
        "",
        f"Global equation: {global_best['equation']}",
        f"Event equation: {event_best['equation']}",
        "",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"- {row.split}: baseline KGE={row.lstm_median_kge:.6f}, baseline NSE={row.lstm_median_nse:.6f}, "
            f"hybrid KGE={row.corrected_median_kge:.6f}, hybrid NSE={row.corrected_median_nse:.6f}"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
