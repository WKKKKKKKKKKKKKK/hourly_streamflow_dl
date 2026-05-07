from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = EXP_DIR / "outputs" / "source_domain_transfer_retention_eval"

AXIS_LABEL_FS = 22
TICK_FS = 18
LEGEND_FS = 20
ANNOTATION_FS = 18
FOOTNOTE_FS = 17

MODEL_STYLE = {
    "source_pretransfer": {
        "label": "Original source model",
        "color": "#1f77b4",
        "marker": "o",
    },
    "transfer": {
        "label": "Transfer learning",
        "color": "#e66101",
        "marker": "s",
    },
    "symbolic_transfer_sw0.05": {
        "label": "Symbolic + transfer learning",
        "color": "#2ca02c",
        "marker": "^",
    },
}

METRIC_CONFIG = {
    "kge": {
        "upper": "KGE",
        "figure_name": "source_domain_test_kge_vs_mean_flow_three_models",
        "binned_summary_name": "kge_vs_flow_test_binned_summary.csv",
        "ylabel": "KGE on source-domain stations",
        "ylim": (-8.5, 1.15),
        "yticks": [-8, -6, -4, -2, 0],
        "zero_label": "KGE = 0",
        "threshold_label": "catastrophic threshold (KGE = -1)",
        "threshold_y": -0.92,
    },
    "nse": {
        "upper": "NSE",
        "figure_name": "source_domain_test_nse_vs_mean_flow_three_models",
        "binned_summary_name": "nse_vs_flow_test_binned_summary.csv",
        "ylabel": "NSE on source-domain stations",
        "ylim": (-12.5, 1.15),
        "yticks": [-12, -10, -8, -6, -4, -2, 0],
        "zero_label": "NSE = 0",
        "threshold_label": "catastrophic threshold (NSE = -1)",
        "threshold_y": -0.92,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--metric", choices=sorted(METRIC_CONFIG), default="kge")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def build_binned_summary(test_df: pd.DataFrame, metric: str, n_bins: int = 8) -> pd.DataFrame:
    station_bins = (
        test_df[["station_id", "obs_mean"]]
        .drop_duplicates("station_id")
        .sort_values("obs_mean")
        .copy()
    )
    station_bins["flow_bin"] = pd.qcut(
        station_bins["obs_mean"].rank(method="first"),
        q=n_bins,
        labels=False,
    )
    test_df = test_df.merge(station_bins[["station_id", "flow_bin"]], on="station_id")
    rows = []
    for (model, flow_bin), group in test_df.groupby(["model", "flow_bin"], sort=True):
        rows.append(
            {
                "model": model,
                "flow_bin": int(flow_bin),
                "n": int(len(group)),
                "median_obs_mean": float(group["obs_mean"].median()),
                f"median_{metric}": float(group[metric].median()),
                f"q25_{metric}": float(group[metric].quantile(0.25)),
                f"q75_{metric}": float(group[metric].quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = METRIC_CONFIG[args.metric]
    out_dir = Path(args.out_dir)
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    station_df = pd.read_csv(out_dir / "per_station_source_domain_metrics_with_flow.csv")
    test_df = station_df.loc[
        station_df["split"].eq("test")
        & station_df["score_status"].eq("ok")
        & station_df["flow_status"].eq("ok")
        & station_df["obs_mean"].gt(0)
    ].copy()
    bin_df = build_binned_summary(test_df, args.metric)
    bin_df.to_csv(out_dir / config["binned_summary_name"], index=False)

    q1_cutoff = test_df.drop_duplicates("station_id")["obs_mean"].quantile(0.25)
    x_min = test_df["obs_mean"].min() * 0.85
    x_max = test_df["obs_mean"].max() * 1.35

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(16.5, 10.0))

    ax.axvspan(x_min, q1_cutoff, color="#e9e9e9", alpha=0.75, zorder=0)
    ax.axhline(0.0, color="#7f7f7f", linestyle="--", linewidth=1.7, zorder=1)
    ax.axhline(-1.0, color="#9b3a3a", linestyle=":", linewidth=1.8, zorder=1)

    for model_name, style in MODEL_STYLE.items():
        model_points = test_df.loc[test_df["model"].eq(model_name)]
        ax.scatter(
            model_points["obs_mean"],
            model_points[args.metric],
            s=56,
            marker=style["marker"],
            color=style["color"],
            alpha=0.16,
            edgecolors="none",
            zorder=2,
        )

    for model_name, style in MODEL_STYLE.items():
        model_bins = bin_df.loc[bin_df["model"].eq(model_name)].sort_values("flow_bin")
        x = model_bins["median_obs_mean"].to_numpy()
        median = model_bins[f"median_{args.metric}"].to_numpy()
        q25 = model_bins[f"q25_{args.metric}"].to_numpy()
        q75 = model_bins[f"q75_{args.metric}"].to_numpy()
        ax.fill_between(x, q25, q75, color=style["color"], alpha=0.12, linewidth=0, zorder=3)
        ax.plot(
            x,
            median,
            color=style["color"],
            marker=style["marker"],
            linewidth=4.2,
            markersize=10.5,
            label=style["label"],
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(*config["ylim"])
    ax.set_yticks(config["yticks"])

    ax.set_xlabel("Mean observed streamflow in test period (log scale)", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel(config["ylabel"], fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS, length=6, width=1.3)
    ax.tick_params(axis="both", which="minor", length=3, width=1.0)
    ax.grid(True, which="major", color="#d7d7d7", linewidth=1.1)
    ax.grid(True, which="minor", axis="x", color="#eeeeee", linewidth=0.8, alpha=0.75)

    ax.text(
        x_min * 1.35,
        0.88,
        "Lowest 25% by mean flow",
        fontsize=ANNOTATION_FS,
        color="#555555",
    )
    ax.text(
        x_max / 1.35,
        0.08,
        config["zero_label"],
        fontsize=ANNOTATION_FS,
        color="#3f3f3f",
        ha="right",
    )
    ax.text(
        x_max / 1.35,
        config["threshold_y"],
        config["threshold_label"],
        fontsize=ANNOTATION_FS,
        color="#7f1515",
        ha="right",
    )

    legend = ax.legend(
        loc="lower right",
        fontsize=LEGEND_FS,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#cfcfcf",
    )
    legend.get_frame().set_linewidth(1.6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.subplots_adjust(left=0.08, right=0.995, top=0.96, bottom=0.18)
    fig.text(
        0.075,
        0.045,
        f"Points: station-level test {config['upper']}. Lines: median {config['upper']} "
        "in 8 equal-count flow bins; "
        "bands: 25th-75th percentile within each bin.",
        fontsize=FOOTNOTE_FS,
        color="#555555",
    )

    fig.savefig(figure_dir / f"{config['figure_name']}.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(figure_dir / f"{config['figure_name']}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
