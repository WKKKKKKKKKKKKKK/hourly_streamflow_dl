from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

CODE_DIR = Path("/home/kongw0a/hourly_streamflow_dl/MTSLSTM_100stations/code")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import plot_s2_random30_threeway_ppt as p


ROOT = Path("/home/kongw0a/hourly_streamflow_dl")
DEFAULT_OUT_DIR = ROOT / "MTSLSTM_100stations" / "outputs" / "s2_random30_threeway_ppt_plots"

TITLE_FS = 22
AXIS_FS = 20
TICK_FS = 16
LEGEND_FS = 15
TEXT_FS = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.array([0.0]), np.array([0.0])
    xs = np.sort(finite)
    ys = np.arange(1, xs.size + 1, dtype=np.float64) / xs.size
    return xs, ys


def compute_per_station_peak_lag(predictions: dict[str, p.StationSeries], split_name: str, model_name: str) -> pd.DataFrame:
    rows = []
    for stn, series in sorted(predictions.items()):
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


def plot_peak_lag_cdfs(per_station_lags: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), sharey=True)

    split_order = ["train", "val", "test"]
    split_titles = {"train": "Train", "val": "Validation", "test": "Test"}
    model_specs = [
        ("baseline", "Best 100-station MTSLSTM", "#1f77b4"),
        ("transfer", "Daily-supervised transfer", "#ff7f0e"),
        ("symbolic", "Transfer + symbolic prior", "#2ca02c"),
    ]

    abs_lags = per_station_lags.copy()
    abs_lags["abs_mean_peak_lag_hours"] = abs_lags["mean_peak_lag_hours"].abs()

    all_finite = abs_lags["abs_mean_peak_lag_hours"].to_numpy(dtype=np.float64)
    all_finite = all_finite[np.isfinite(all_finite)]
    if all_finite.size:
        x_min = 0.0
        x_max = float(np.nanmax(all_finite))
        span = max(x_max - x_min, 1.0)
        x_limits = (x_min, x_max + 0.06 * span)
    else:
        x_limits = (0.0, 5.0)

    for ax, split in zip(axes, split_order):
        split_df = abs_lags.loc[abs_lags["split"].eq(split)]
        for model_key, label, color in model_specs:
            vals = split_df.loc[split_df["model"].eq(model_key), "abs_mean_peak_lag_hours"].to_numpy(dtype=np.float64)
            xs, ys = empirical_cdf(vals)
            valid_n = int(np.isfinite(vals).sum())
            vals = vals[np.isfinite(vals)]
            med = float(np.median(vals)) if vals.size else float("nan")
            if np.isfinite(med):
                legend_label = f"{label} (n={valid_n}, median={med:.2f} h)"
            else:
                legend_label = f"{label} (n={valid_n}, median=n/a)"
            ax.step(xs, ys, where="post", linewidth=2.6, color=color, label=legend_label)

        ax.set_title(split_titles[split], fontsize=TITLE_FS)
        ax.set_xlabel("|Average local peak lag| (hours)", fontsize=AXIS_FS)
        ax.set_xlim(*x_limits)
        ax.set_ylim(0.0, 1.03)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.legend(loc="lower right", fontsize=LEGEND_FS, framealpha=0.95)

    axes[0].set_ylabel("Cumulative probability", fontsize=AXIS_FS)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_station_path = out_dir / "three_method_peak_lag_per_station.csv"
    if per_station_path.exists():
        print(f"Reusing existing per-station peak lag file: {per_station_path}", flush=True)
        per_station_lags = pd.read_csv(per_station_path)
    else:
        station_ids = p.load_station_ids()
        static_df = p.load_static_df()
        with p.SCALER_PATH.open("rb") as fp:
            scalers = pickle.load(fp)
        full_ds = p.load_hourly_dataset(station_ids)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}", flush=True)

        models = {
            "baseline": p.create_model(device, p.BASELINE_MODEL_PATH),
            "transfer": p.create_model(device, p.TRANSFER_MODEL_PATH),
            "symbolic": p.create_model(device, p.SYMBOLIC_MODEL_PATH),
        }
        split_dates = {
            "train": (p.config.TRAIN_START, p.config.TRAIN_END),
            "val": (p.config.VAL_START, p.config.VAL_END),
            "test": (p.config.TEST_START, p.config.TEST_END),
        }

        lag_frames = []
        for split_name, (start, end) in split_dates.items():
            print(f"Building loader for {split_name}...", flush=True)
            loader = p.build_loader(full_ds, static_df, scalers, start, end)
            for model_name, model in models.items():
                print(f"Collecting {model_name} predictions for {split_name}...", flush=True)
                preds = p.collect_predictions(model, loader, scalers, device)
                lag_frames.append(compute_per_station_peak_lag(preds, split_name, model_name))

        per_station_lags = pd.concat(lag_frames, ignore_index=True)
        per_station_lags.to_csv(per_station_path, index=False)

    summary = (
        per_station_lags.dropna(subset=["mean_peak_lag_hours"])
        .groupby(["split", "model"], as_index=False)
        .agg(
            median_avg_local_peak_lag_hours=("mean_peak_lag_hours", "median"),
            valid_station_count=("mean_peak_lag_hours", "size"),
        )
    )
    summary_pivot = summary.pivot(index="split", columns="model", values="median_avg_local_peak_lag_hours").reset_index()
    count_pivot = summary.pivot(index="split", columns="model", values="valid_station_count").reset_index()
    summary_pivot.columns.name = None
    count_pivot.columns.name = None
    merged = summary_pivot.merge(count_pivot, on="split", suffixes=("", "_valid_n"))
    merged.to_csv(out_dir / "three_method_peak_lag_comparison.csv", index=False)

    plot_peak_lag_cdfs(per_station_lags, out_dir / "three_method_peak_lag_cdfs.png")
    (out_dir / "three_method_peak_lag_cdfs_summary.md").write_text(
        "\n".join(
            [
                "CDFs of absolute average local peak lag for three methods across train/validation/test",
                "",
                f"Output figure: {out_dir / 'three_method_peak_lag_cdfs.png'}",
                f"Per-station lags: {out_dir / 'three_method_peak_lag_per_station.csv'}",
                f"Summary table: {out_dir / 'three_method_peak_lag_comparison.csv'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
