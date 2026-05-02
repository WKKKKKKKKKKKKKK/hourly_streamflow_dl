from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


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
OUT_DIR = (
    ROOT
    / "MTSLSTM_100stations"
    / "outputs"
    / "baseline_lstm_daily_s2_random30_symtorch_direct_valkge"
)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")


def build_variable_names(lookback: int, dynamic_vars: list[str], static_cols: list[str]) -> list[str]:
    names: list[str] = []
    for time_idx in range(lookback):
        lag = lookback - 1 - time_idx
        for var in dynamic_vars:
            names.append(f"{sanitize_name(var)}_lag{lag}")
    names.extend(sanitize_name(col) for col in static_cols)
    return names


def collect_split_arrays(loader, model: torch.nn.Module, device: torch.device) -> dict[str, np.ndarray]:
    flat_inputs: list[np.ndarray] = []
    y_true_std: list[np.ndarray] = []
    y_pred_std: list[np.ndarray] = []
    stations: list[str] = []

    model.eval()
    with torch.no_grad():
        for x_dyn_batch, x_static_batch, y_batch, stn_batch in loader:
            x_dyn_batch = x_dyn_batch.to(device)
            x_static_batch = x_static_batch.to(device)
            outputs = model((x_dyn_batch, x_static_batch)).reshape(-1)

            flat = torch.cat(
                [x_dyn_batch.reshape(x_dyn_batch.shape[0], -1), x_static_batch],
                dim=1,
            )

            flat_inputs.append(flat.detach().cpu().numpy().astype(np.float32))
            y_true_std.append(y_batch.detach().cpu().numpy().reshape(-1).astype(np.float32))
            y_pred_std.append(outputs.detach().cpu().numpy().reshape(-1).astype(np.float32))
            stations.extend(str(stn) for stn in stn_batch)

    return {
        "X_flat": np.concatenate(flat_inputs, axis=0),
        "y_true_std": np.concatenate(y_true_std, axis=0),
        "y_pred_std": np.concatenate(y_pred_std, axis=0),
        "station_ids": np.asarray(stations, dtype="<U16"),
    }


def balanced_subsample_indices(station_ids: np.ndarray, per_station_cap: int = 300, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for station_id in sorted(np.unique(station_ids)):
        idx = np.where(station_ids == station_id)[0]
        if len(idx) > per_station_cap:
            idx = rng.choice(idx, size=per_station_cap, replace=False)
        selected.append(np.sort(idx))
    return np.sort(np.concatenate(selected)).astype(np.int64)


def save_split_npz(path: Path, split_arrays: dict[str, np.ndarray], indices: np.ndarray | None = None) -> None:
    payload = split_arrays
    if indices is not None:
        payload = {
            "X_flat": split_arrays["X_flat"][indices],
            "y_true_std": split_arrays["y_true_std"][indices],
            "y_pred_std": split_arrays["y_pred_std"][indices],
            "station_ids": split_arrays["station_ids"][indices],
            "sample_indices": indices,
        }
    np.savez_compressed(path, **payload)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_tune.set_seed(42)

    best_run = json.loads(BEST_RUN_JSON.read_text(encoding="utf-8"))
    run_dir = Path(best_run["run_dir"])
    lookback = int(best_run["lookback_days"])
    hidden_size = int(best_run["hidden_size"])
    dropout = float(best_run["dropout"])

    station_ids = daily_tune.load_station_ids()
    static_df = daily_tune.load_static_df(station_ids)
    selected_meta = pd.read_csv(daily_tune.SELECTED_STATIONS_CSV, dtype={"station_id": str})
    prepared = daily_tune.build_standardized_splits(station_ids, static_df, OUT_DIR)

    _, train_loader = daily_tune.make_loader(
        prepared["train_dyn_std"],
        prepared["train_y_std"],
        prepared["train_static_std"],
        lookback=lookback,
        split="train",
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )
    _, val_loader = daily_tune.make_loader(
        prepared["val_dyn_std"],
        prepared["val_y_std"],
        prepared["val_static_std"],
        lookback=lookback,
        split="val",
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )
    _, test_loader = daily_tune.make_loader(
        prepared["test_dyn_std"],
        prepared["test_y_std"],
        prepared["test_static_std"],
        lookback=lookback,
        split="test",
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM(
        input_size=len(daily_tune.DYNAMIC_VARS) + prepared["train_static_std"].shape[1],
        hidden_size=hidden_size,
        num_layers=1,
        dropout=dropout,
        output_size=1,
    ).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pth", map_location=device))

    train_arrays = collect_split_arrays(train_loader, model, device)
    val_arrays = collect_split_arrays(val_loader, model, device)
    test_arrays = collect_split_arrays(test_loader, model, device)

    static_cols = static_df.columns.tolist()
    variable_names = build_variable_names(lookback, list(daily_tune.DYNAMIC_VARS), static_cols)

    save_split_npz(OUT_DIR / "train_full.npz", train_arrays)
    save_split_npz(OUT_DIR / "val_full.npz", val_arrays)
    save_split_npz(OUT_DIR / "test_full.npz", test_arrays)

    distill_indices = balanced_subsample_indices(train_arrays["station_ids"], per_station_cap=300, seed=42)
    save_split_npz(OUT_DIR / "train_distill_sample.npz", train_arrays, indices=distill_indices)

    export_meta = {
        "source_run": best_run,
        "splits": {
            "train": [daily_tune.TRAIN_START, daily_tune.TRAIN_END],
            "val": [daily_tune.VAL_START, daily_tune.VAL_END],
            "test": [daily_tune.TEST_START, daily_tune.TEST_END],
        },
        "lookback_days": lookback,
        "dynamic_vars": list(daily_tune.DYNAMIC_VARS),
        "static_vars": static_cols,
        "n_total_features": len(variable_names),
        "n_train_samples": int(len(train_arrays["y_true_std"])),
        "n_val_samples": int(len(val_arrays["y_true_std"])),
        "n_test_samples": int(len(test_arrays["y_true_std"])),
        "n_distill_samples": int(len(distill_indices)),
        "distill_sampling": {
            "strategy": "balanced_per_station_cap",
            "per_station_cap": 300,
            "seed": 42,
        },
        "scalers": {
            "y_mean": float(prepared["scalers"]["y_mean"]),
            "y_std": float(prepared["scalers"]["y_std"]),
        },
        "selected_stations_csv": str(daily_tune.SELECTED_STATIONS_CSV),
        "variable_names_json": str(OUT_DIR / "variable_names.json"),
    }
    (OUT_DIR / "variable_names.json").write_text(json.dumps(variable_names, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "export_metadata.json").write_text(json.dumps(export_meta, indent=2) + "\n", encoding="utf-8")
    selected_meta.to_csv(OUT_DIR / "selected_stations.csv", index=False)

    summary_lines = [
        "Export of best validation-KGE daily BaselineLSTM inputs/predictions for SymTorch direct distillation",
        "",
        f"Best run tag: {best_run['tag']}",
        f"Lookback days: {lookback}",
        f"Dynamic variables: {', '.join(daily_tune.DYNAMIC_VARS)}",
        f"Static variables: {len(static_cols)}",
        f"Flattened feature count: {len(variable_names)}",
        f"Train samples: {len(train_arrays['y_true_std'])}",
        f"Val samples: {len(val_arrays['y_true_std'])}",
        f"Test samples: {len(test_arrays['y_true_std'])}",
        f"Distill train sample size: {len(distill_indices)}",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps(export_meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
