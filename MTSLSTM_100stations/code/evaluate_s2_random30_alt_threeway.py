from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch

CODE_DIR = Path("/home/kongw0a/hourly_streamflow_dl/MTSLSTM_100stations/code")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import plot_s2_random30_threeway_ppt as p
from Train import compute_kge, compute_nse
from plot_s2_random30_baseline_vs_transfer import choose_informative_hydrograph_stations
from plot_three_method_peak_lag_cdfs import plot_peak_lag_cdfs


ROOT = Path("/home/kongw0a/hourly_streamflow_dl")
SELECTION_META_PATH = Path(
    "/mnt/datawaha/hyex/atr/gscad_database/raw/CAMELS/CAMELSH/attributes/attributes_gageii_BasinID.csv"
)
PREVIOUS_SELECTED_PATH = ROOT / "MTSLSTM_100stations" / "outputs" / "s2_random30_idx2_eval" / "selected_stations.csv"
DEFAULT_OUT_DIR = ROOT / "MTSLSTM_100stations" / "outputs" / "s2_random30_alt_threeway_ppt_plots"

S2_LON_MIN = -95.0
S2_LON_MAX = -75.0
S2_LAT_MIN = 28.0
S2_LAT_MAX = 36.8
SELECTION_SEED = 43
N_STATIONS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def load_s2_alt_station_table() -> pd.DataFrame:
    selection_meta = pd.read_csv(SELECTION_META_PATH, dtype={"STAID": str})
    selection_meta["STAID"] = selection_meta["STAID"].str.strip()
    static_df = p.load_static_df()
    available = {path.stem for path in p.RAW_TIMESERIES_DIR.glob("*.nc")}

    previous = pd.read_csv(PREVIOUS_SELECTED_PATH, dtype={"station_id": str})
    previous_ids = set(previous["station_id"].astype(str))

    candidates = selection_meta.loc[
        selection_meta["LNG_GAGE"].between(S2_LON_MIN, S2_LON_MAX)
        & selection_meta["LAT_GAGE"].between(S2_LAT_MIN, S2_LAT_MAX)
    ].copy()
    candidates = candidates.rename(columns={"STAID": "station_id", "LAT_GAGE": "lat", "LNG_GAGE": "lon"})
    candidates = candidates.loc[candidates["station_id"].isin(available)].copy()
    candidates = candidates.loc[~candidates["station_id"].isin(previous_ids)].copy()

    static_augmented, missing_static = p._add_static_station_aliases(static_df, candidates["station_id"].tolist())
    if missing_static:
        candidates = candidates.loc[~candidates["station_id"].isin(set(missing_static))].copy()

    candidates = candidates.sort_values("station_id").reset_index(drop=True)
    if len(candidates) < N_STATIONS:
        raise ValueError(
            f"Only {len(candidates)} S2 candidates remain after excluding the original 30 stations; fewer than requested {N_STATIONS}."
        )

    selected = candidates.sample(n=N_STATIONS, random_state=SELECTION_SEED).sort_values("station_id").reset_index(drop=True)
    static_reset = static_df.reset_index()
    static_reset = static_reset.rename(columns={static_reset.columns[0]: "station_id"})
    selected = selected.merge(static_reset, on="station_id", how="left")
    selected["selection_seed"] = SELECTION_SEED
    selected["selection_region"] = "S2 Cfa-SE"
    selected["selection_note"] = (
        f"Figure-inferred box lon=[{S2_LON_MIN}, {S2_LON_MAX}], lat=[{S2_LAT_MIN}, {S2_LAT_MAX}]; "
        f"disjoint resample excluding original S2 30 stations from {PREVIOUS_SELECTED_PATH.name}; "
        f"requested {N_STATIONS}, selected {N_STATIONS}"
    )
    return selected


def evaluate_predictions(predictions: dict[str, p.StationSeries], expected_stations: list[str]) -> pd.DataFrame:
    rows = []
    for stn in expected_stations:
        row = {
            "station_id": stn,
            "samples": 0,
            "score_status": "ok",
            "exclusion_reason": "",
            "nse": np.nan,
            "kge": np.nan,
        }
        series = predictions.get(stn)
        if series is None:
            row["score_status"] = "excluded"
            row["exclusion_reason"] = "no_valid_windows"
            rows.append(row)
            continue
        obs = np.asarray(series.obs, dtype=np.float64)
        pred = np.asarray(series.pred, dtype=np.float64)
        row["samples"] = int(obs.size)
        nse = compute_nse(obs, pred)
        kge = compute_kge(obs, pred)
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


def compute_per_station_peak_lag(
    predictions: dict[str, p.StationSeries], split_name: str, model_name: str, expected_stations: list[str]
) -> pd.DataFrame:
    rows = []
    for stn in expected_stations:
        series = predictions.get(stn)
        if series is None:
            rows.append(
                {
                    "station_id": stn,
                    "split": split_name,
                    "model": model_name,
                    "mean_peak_lag_hours": np.nan,
                    "matched_peak_count": 0,
                    "obs_peak_count": 0,
                    "pred_peak_count": 0,
                }
            )
            continue
        obs_peaks = p.detect_local_peaks(series.obs)
        pred_peaks = p.detect_local_peaks(series.pred)
        mean_lag, n_matches = p.compute_mean_peak_lag_hours(obs_peaks, pred_peaks)
        rows.append(
            {
                "station_id": stn,
                "split": split_name,
                "model": model_name,
                "mean_peak_lag_hours": mean_lag,
                "matched_peak_count": int(n_matches),
                "obs_peak_count": int(len(obs_peaks)),
                "pred_peak_count": int(len(pred_peaks)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = load_s2_alt_station_table()
    selected.to_csv(out_dir / "selected_stations.csv", index=False)

    station_ids = selected["station_id"].astype(str).tolist()
    static_df = p.load_static_df()
    with p.SCALER_PATH.open("rb") as fp:
        scalers = pickle.load(fp)
    full_ds = p.load_hourly_dataset(station_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    model_specs = [
        {
            "key": "baseline",
            "label": "Best 100-station MTSLSTM",
            "model_path": p.BASELINE_MODEL_PATH,
            "hydro_png": "baseline_hydrographs_shared_two_year_window_ppt.png",
            "scatter_png": "baseline_scatter_kge_nse_cdf_ppt.png",
        },
        {
            "key": "transfer",
            "label": "Daily-supervised transfer MTSLSTM",
            "model_path": p.TRANSFER_MODEL_PATH,
            "hydro_png": "transfer_hydrographs_shared_two_year_window_ppt.png",
            "scatter_png": "transfer_scatter_kge_nse_cdf_ppt.png",
        },
        {
            "key": "symbolic",
            "label": "Transfer + symbolic regression prior",
            "model_path": p.SYMBOLIC_MODEL_PATH,
            "hydro_png": "symbolic_transfer_hydrographs_shared_two_year_window_ppt.png",
            "scatter_png": "symbolic_transfer_scatter_kge_nse_cdf_ppt.png",
        },
    ]

    models = {spec["key"]: p.create_model(device, spec["model_path"]) for spec in model_specs}

    split_dates = {
        "train": (p.config.TRAIN_START, p.config.TRAIN_END),
        "val": (p.config.VAL_START, p.config.VAL_END),
        "test": (p.config.TEST_START, p.config.TEST_END),
    }

    metrics_long_rows = []
    predictions_by_split: dict[str, dict[str, dict[str, p.StationSeries]]] = {}
    peak_lag_frames = []

    for split_name, (start, end) in split_dates.items():
        print(f"Building loader for {split_name}...", flush=True)
        loader = p.build_loader(full_ds, static_df, scalers, start, end)
        predictions_by_split[split_name] = {}
        for spec in model_specs:
            print(f"Collecting {spec['key']} predictions for {split_name}...", flush=True)
            preds = p.collect_predictions(models[spec["key"]], loader, scalers, device)
            predictions_by_split[split_name][spec["key"]] = preds
            station_metrics = evaluate_predictions(preds, station_ids)
            station_metrics = station_metrics.rename(
                columns={
                    "samples": f"{split_name}_samples",
                    "score_status": f"{split_name}_score_status",
                    "exclusion_reason": f"{split_name}_exclusion_reason",
                    "nse": f"{split_name}_nse",
                    "kge": f"{split_name}_kge",
                }
            )
            station_metrics["model"] = spec["key"]
            metrics_long_rows.append(station_metrics)
            peak_lag_frames.append(compute_per_station_peak_lag(preds, split_name, spec["key"], station_ids))

    comparison_rows = []
    summary_tables = {}
    for spec in model_specs:
        merged = selected.copy()
        split_summaries = []
        for split_name in ["train", "val", "test"]:
            frame = [
                df
                for df in metrics_long_rows
                if df["model"].iloc[0] == spec["key"] and f"{split_name}_samples" in df.columns
            ][0]
            cols = [
                "station_id",
                f"{split_name}_samples",
                f"{split_name}_score_status",
                f"{split_name}_exclusion_reason",
                f"{split_name}_nse",
                f"{split_name}_kge",
            ]
            merged = merged.merge(frame[cols], on="station_id", how="left")
            summary = summarize_metrics(
                frame.rename(
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
            split_summaries.append(summary)
        merged.to_csv(out_dir / f"{spec['key']}_per_station_metrics.csv", index=False)
        pd.DataFrame(split_summaries).to_csv(out_dir / f"{spec['key']}_summary_metrics.csv", index=False)
        summary_tables[spec["key"]] = {row["split"]: row for row in split_summaries}

    for split_name in ["train", "val", "test"]:
        b = summary_tables["baseline"][split_name]
        t = summary_tables["transfer"][split_name]
        s = summary_tables["symbolic"][split_name]
        comparison_rows.append(
            {
                "split": split_name,
                "baseline_kge": b["median_kge"],
                "baseline_nse": b["median_nse"],
                "transfer_kge": t["median_kge"],
                "transfer_nse": t["median_nse"],
                "transfer_kge_gain_vs_baseline": t["median_kge"] - b["median_kge"],
                "transfer_nse_gain_vs_baseline": t["median_nse"] - b["median_nse"],
                "symbolic_kge": s["median_kge"],
                "symbolic_nse": s["median_nse"],
                "symbolic_kge_gain_vs_baseline": s["median_kge"] - b["median_kge"],
                "symbolic_nse_gain_vs_baseline": s["median_nse"] - b["median_nse"],
            }
        )
    pd.DataFrame(comparison_rows).to_csv(out_dir / "three_method_metrics_comparison.csv", index=False)

    test_predictions = predictions_by_split["test"]
    common_valid_test = set(station_ids)
    for spec in model_specs:
        df = pd.read_csv(out_dir / f"{spec['key']}_per_station_metrics.csv", dtype={"station_id": str})
        ok = set(df.loc[df["test_score_status"].eq("ok"), "station_id"].astype(str))
        common_valid_test &= ok
    common_valid_test = sorted(common_valid_test)

    hydro_station_ids = choose_informative_hydrograph_stations(test_predictions["baseline"], common_valid_test)
    pd.DataFrame(
        {
            "station_id": hydro_station_ids,
            "selection_note": "informative common-valid test stations for alternate S2 Cfa-SE three-model comparison",
        }
    ).to_csv(out_dir / "hydrograph_selected_stations.csv", index=False)
    hydro_y_limits = p.compute_shared_hydro_y_limits(list(test_predictions.values()), hydro_station_ids)
    scatter_limits = p.compute_shared_scatter_limits(list(test_predictions.values()), common_valid_test)

    baseline_obs, _ = p.flatten_predictions(test_predictions["baseline"], common_valid_test)
    rng = np.random.default_rng(p.SEED)
    sample_n = min(p.SCATTER_SAMPLE_SIZE, baseline_obs.size)
    sample_idx = rng.choice(baseline_obs.size, size=sample_n, replace=False)

    hydro_rows = []
    scatter_rows = []
    for spec in model_specs:
        hydro_rows.append(
            p.plot_hydrographs(
                predictions=test_predictions[spec["key"]],
                station_ids=hydro_station_ids,
                output_path=out_dir / spec["hydro_png"],
                y_limits=hydro_y_limits,
                label=spec["label"],
            )
        )
        metrics_df = pd.read_csv(out_dir / f"{spec['key']}_per_station_metrics.csv", dtype={"station_id": str})
        scatter_rows.append(
            p.plot_scatter_and_cdfs(
                predictions=test_predictions[spec["key"]],
                metrics_df=metrics_df,
                station_ids=common_valid_test,
                sample_idx=sample_idx,
                scatter_limits=scatter_limits,
                output_path=out_dir / spec["scatter_png"],
                label=spec["label"],
            )
        )
    pd.concat(hydro_rows, ignore_index=True).to_csv(out_dir / "hydrograph_peak_lag_summary.csv", index=False)
    pd.concat(scatter_rows, ignore_index=True).to_csv(out_dir / "scatter_cdf_summary.csv", index=False)

    peak_lag_df = pd.concat(peak_lag_frames, ignore_index=True)
    peak_lag_df.to_csv(out_dir / "three_method_peak_lag_per_station.csv", index=False)

    peak_summary_rows = []
    for split_name in ["train", "val", "test"]:
        row = {"split": split_name}
        for key in ["baseline", "symbolic", "transfer"]:
            vals = peak_lag_df.loc[
                peak_lag_df["split"].eq(split_name) & peak_lag_df["model"].eq(key), "mean_peak_lag_hours"
            ].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[key] = float(np.median(vals)) if vals.size else np.nan
            row[f"{key}_valid_n"] = int(vals.size)
        peak_summary_rows.append(row)
    pd.DataFrame(peak_summary_rows).to_csv(out_dir / "three_method_peak_lag_comparison.csv", index=False)

    plot_peak_lag_cdfs(peak_lag_df, out_dir / "three_method_peak_lag_cdfs.png")

    summary_lines = [
        "Alternate S2 Cfa-SE three-model evaluation",
        "",
        "Station selection:",
        f"- Source: {SELECTION_META_PATH.name} + static_h_topo_priority27.csv",
        "- Region: S2 Cfa-SE",
        f"- Figure-inferred box lon=[{S2_LON_MIN}, {S2_LON_MAX}], lat=[{S2_LAT_MIN}, {S2_LAT_MAX}]",
        f"- Exclusion: all stations listed in {PREVIOUS_SELECTED_PATH.name}",
        f"- Requested target count: {N_STATIONS}; selected stations: {N_STATIONS}",
        "",
        "Generated files:",
        "- selected_stations.csv",
        "- baseline_summary_metrics.csv / baseline_per_station_metrics.csv",
        "- transfer_summary_metrics.csv / transfer_per_station_metrics.csv",
        "- symbolic_summary_metrics.csv / symbolic_per_station_metrics.csv",
        "- three_method_metrics_comparison.csv",
        "- baseline/transfer/symbolic hydrograph PNGs",
        "- baseline/transfer/symbolic scatter-KGE-NSE-CDF PNGs",
        "- three_method_peak_lag_comparison.csv",
        "- three_method_peak_lag_cdfs.png",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
