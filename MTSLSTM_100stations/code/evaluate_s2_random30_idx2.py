from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

import config
from Modelzoo import sMTSLSTM
from Train import _add_static_station_aliases, evaluate_per_station
from loder import MultiscaleLSTMDataset, handle_extremes, standardize_data


ROOT = Path("/home/kongw0a/hourly_streamflow_dl")
RAW_TIMESERIES_DIR = Path(
    "/mnt/datawaha/hyex/atr/gscad_database/raw/CAMELS/CAMELSH/timeseries/Data/CAMELSH/timeseries"
)
SELECTION_META_PATH = Path(
    "/mnt/datawaha/hyex/atr/gscad_database/raw/CAMELS/CAMELSH/attributes/attributes_gageii_BasinID.csv"
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
OUT_DIR = ROOT / "MTSLSTM_100stations" / "outputs" / "s2_random30_idx2_eval"

DYNAMIC_VARS = ["Rainf", "Tair", "PotEvap"]
TARGET_VAR = "Streamflow"
LOOKBACK_H = 168
LOOKBACK_D = 365
FREQ = 24
BATCH_SIZE = 256

# These bounds are inferred from the uploaded S2 Cfa-SE proposal box in the figure.
S2_LON_MIN = -95.0
S2_LON_MAX = -75.0
S2_LAT_MIN = 28.0
S2_LAT_MAX = 36.8

SPLITS = {
    "train": (config.TRAIN_START, config.TRAIN_END),
    "val": (config.VAL_START, config.VAL_END),
    "test": (config.TEST_START, config.TEST_END),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-stations", type=int, default=30)
    parser.add_argument("--force-resample", action="store_true")
    return parser.parse_args()


def choose_stations(seed: int, n_stations: int) -> pd.DataFrame:
    selection_meta = pd.read_csv(SELECTION_META_PATH, dtype={"STAID": str})
    selection_meta["STAID"] = selection_meta["STAID"].str.strip()

    static_df = pd.read_csv(STATIC_MODEL_INPUT_PATH, index_col=0)
    static_df.index = static_df.index.astype(str).str.strip()

    available = {path.stem for path in RAW_TIMESERIES_DIR.glob("*.nc")}

    candidates = selection_meta.loc[
        selection_meta["LNG_GAGE"].between(S2_LON_MIN, S2_LON_MAX)
        & selection_meta["LAT_GAGE"].between(S2_LAT_MIN, S2_LAT_MAX)
    ].copy()
    candidates = candidates.rename(columns={"STAID": "station_id", "LAT_GAGE": "lat", "LNG_GAGE": "lon"})
    candidates = candidates.loc[candidates["station_id"].isin(available)].copy()

    static_augmented, missing_static = _add_static_station_aliases(static_df, candidates["station_id"].tolist())
    if missing_static:
        candidates = candidates.loc[~candidates["station_id"].isin(set(missing_static))].copy()

    candidates = candidates.sort_values("station_id").reset_index(drop=True)
    if len(candidates) < n_stations:
        raise ValueError(f"Only {len(candidates)} S2 candidates available, fewer than requested {n_stations}.")

    selected = candidates.sample(n=n_stations, random_state=seed).sort_values("station_id").reset_index(drop=True)
    selected["selection_seed"] = seed
    selected["selection_region"] = "S2 Cfa-SE"
    selected["selection_note"] = (
        f"Figure-inferred box lon=[{S2_LON_MIN}, {S2_LON_MAX}], lat=[{S2_LAT_MIN}, {S2_LAT_MAX}]"
    )
    return selected


def prepare_split(
    full_ds: xr.Dataset,
    static_df: pd.DataFrame,
    scalers: dict,
    start: str,
    end: str,
):
    dyn = full_ds.sel(time=slice(start, end))
    dyn_forcing = dyn.sel(dynamic_forcing=DYNAMIC_VARS)
    target = dyn.sel(dynamic_forcing=TARGET_VAR)
    dyn_std, static_std, y_std = standardize_data(dyn_forcing, static_df, target, scalers)
    static_std, missing_static = _add_static_station_aliases(static_std, full_ds.data_vars)
    if missing_static:
        preview = ", ".join(missing_static[:10])
        suffix = " ..." if len(missing_static) > 10 else ""
        raise KeyError(f"Missing static features for {len(missing_static)} stations: {preview}{suffix}")
    return dyn_std, static_std, y_std


def build_loader(dyn_std, y_std, static_std, start: str, end: str) -> DataLoader:
    dataset = MultiscaleLSTMDataset(
        dyn_std,
        y_std,
        static_std,
        lookback_hourly=LOOKBACK_H,
        lookback_daily=LOOKBACK_D,
        frequency_factor=FREQ,
        start_date=start,
        end_date=end,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def load_model(device: torch.device) -> tuple[torch.nn.Module, dict]:
    with SCALER_PATH.open("rb") as fp:
        scalers = pickle.load(fp)

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
    model.eval()
    return model, scalers


def load_station_dataset(station_ids: list[str]) -> xr.Dataset:
    station_arrays = {}
    for station_id in station_ids:
        with xr.open_dataset(RAW_TIMESERIES_DIR / f"{station_id}.nc") as ds:
            da = ds[DYNAMIC_VARS + [TARGET_VAR]].to_array(dim="dynamic_forcing").transpose("DateTime", "dynamic_forcing")
            da = da.rename({"DateTime": "time"})
            da = da.assign_coords(dynamic_forcing=DYNAMIC_VARS + [TARGET_VAR])
            da.name = station_id
            station_arrays[station_id] = da.load()
    full_ds = xr.Dataset(station_arrays)
    full_ds = handle_extremes(full_ds, min_streamflow=0.0, max_streamflow=1000.0)
    return full_ds


def summarize_metrics(metrics_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    valid = metrics_df.loc[metrics_df["score_status"].eq("ok")].copy()
    summary = pd.DataFrame(
        [
            {
                "split": split_name,
                "n_total_stations": int(len(metrics_df)),
                "n_valid_stations": int(len(valid)),
                "n_excluded_stations": int(len(metrics_df) - len(valid)),
                "median_kge": float(valid["kge"].median()) if len(valid) else np.nan,
                "median_nse": float(valid["nse"].median()) if len(valid) else np.nan,
                "mean_kge": float(valid["kge"].mean()) if len(valid) else np.nan,
                "mean_nse": float(valid["nse"].mean()) if len(valid) else np.nan,
            }
        ]
    )
    return summary


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_csv = OUT_DIR / "selected_stations.csv"
    if selected_csv.exists() and not args.force_resample:
        selected = pd.read_csv(selected_csv, dtype={"station_id": str})
    else:
        selected = choose_stations(seed=args.seed, n_stations=args.n_stations)
        selected.to_csv(selected_csv, index=False)

    station_ids = selected["station_id"].astype(str).tolist()
    static_df = pd.read_csv(STATIC_MODEL_INPUT_PATH, index_col=0)
    static_df.index = static_df.index.astype(str).str.strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model, scalers = load_model(device)
    full_ds = load_station_dataset(station_ids)

    all_metrics = selected.copy()
    split_summaries = []

    for split_name, (start, end) in SPLITS.items():
        print(f"Evaluating split={split_name} range=[{start}, {end}]", flush=True)
        dyn_std, static_std, y_std = prepare_split(full_ds, static_df, scalers, start, end)
        loader = build_loader(dyn_std, y_std, static_std, start, end)
        split_metrics = evaluate_per_station(model, loader, scalers, device, expected_stations=station_ids)
        split_metrics = split_metrics.rename(
            columns={
                "samples": f"{split_name}_samples",
                "score_status": f"{split_name}_score_status",
                "exclusion_reason": f"{split_name}_exclusion_reason",
                "nse": f"{split_name}_nse",
                "kge": f"{split_name}_kge",
            }
        )
        all_metrics = all_metrics.merge(split_metrics, on="station_id", how="left")
        split_summaries.append(summarize_metrics(split_metrics.rename(columns={
            f"{split_name}_samples": "samples",
            f"{split_name}_score_status": "score_status",
            f"{split_name}_exclusion_reason": "exclusion_reason",
            f"{split_name}_nse": "nse",
            f"{split_name}_kge": "kge",
        }), split_name))

    summary_df = pd.concat(split_summaries, ignore_index=True)
    all_metrics.to_csv(OUT_DIR / "per_station_metrics.csv", index=False)
    summary_df.to_csv(OUT_DIR / "summary_metrics.csv", index=False)

    lines = [
        "S2 random-30 evaluation with best MTSLSTM 100-station model",
        "",
        f"Model run: {RUN_DIR.name}",
        f"Model path: {MODEL_PATH}",
        f"Scaler path: {SCALER_PATH}",
        f"Selection seed: {int(selected['selection_seed'].iloc[0])}",
        f"Selected stations: {len(selected)}",
        f"S2 box inference: lon=[{S2_LON_MIN}, {S2_LON_MAX}], lat=[{S2_LAT_MIN}, {S2_LAT_MAX}]",
        "",
        "Split,n_total_stations,n_valid_stations,n_excluded_stations,median_kge,median_nse,mean_kge,mean_nse",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"{row.split},{row.n_total_stations},{row.n_valid_stations},{row.n_excluded_stations},"
            f"{row.median_kge:.6f},{row.median_nse:.6f},{row.mean_kge:.6f},{row.mean_nse:.6f}"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
