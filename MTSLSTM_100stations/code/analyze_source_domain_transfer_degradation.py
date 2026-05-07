from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = EXP_DIR / "outputs" / "source_domain_transfer_retention_eval"
TRANSFER_MODELS = ("transfer", "symbolic_transfer_sw0.05")
SOURCE_MODEL = "source_pretransfer"
QUARTILE_LABELS = ("Q1-low", "Q2", "Q3", "Q4-high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess source-domain transfer-retention metrics to quantify "
            "KGE/NSE degradation, low-flow failure rates, and binned flow responses."
        )
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--metrics-with-flow",
        default="",
        help=(
            "Optional flow-enriched metrics CSV. Defaults to "
            "<out-dir>/per_station_source_domain_metrics_with_flow.csv."
        ),
    )
    parser.add_argument("--flow-bins", type=int, default=8)
    return parser.parse_args()


def assign_flow_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["obs_mean_pct_rank"] = np.nan
    df["flow_mean_quartile"] = ""
    for split, split_df in df.groupby("split"):
        station_flow = (
            split_df.loc[
                split_df["flow_status"].eq("ok") & split_df["obs_mean"].gt(0),
                ["station_id", "obs_mean"],
            ]
            .drop_duplicates("station_id")
            .sort_values("obs_mean")
        )
        if station_flow.empty:
            continue
        ranks = station_flow["obs_mean"].rank(method="average", pct=True)
        quartiles = pd.qcut(
            station_flow["obs_mean"].rank(method="first"),
            4,
            labels=QUARTILE_LABELS,
        )
        rank_map = dict(zip(station_flow["station_id"], ranks))
        quartile_map = dict(zip(station_flow["station_id"], quartiles.astype(str)))
        mask = df["split"].eq(split)
        df.loc[mask, "obs_mean_pct_rank"] = df.loc[mask, "station_id"].map(rank_map)
        df.loc[mask, "flow_mean_quartile"] = df.loc[mask, "station_id"].map(quartile_map)
    return df


def add_degradation_flags(df: pd.DataFrame) -> pd.DataFrame:
    derived_cols = [
        "source_score_status",
        "source_kge",
        "source_nse",
        "delta_kge_vs_source",
        "delta_nse_vs_source",
        "retained_close_kge",
        "retained_close_nse",
        "retained_close_both",
        "retained_usable",
        "failure_kge_lt_0",
        "catastrophic_kge_lt_minus1",
        "big_forgetting_delta_lt_minus05",
    ]
    df = df.drop(columns=[col for col in derived_cols if col in df.columns]).copy()
    source = df.loc[df["model"].eq(SOURCE_MODEL), ["split", "station_id", "score_status", "kge", "nse"]]
    source = source.rename(
        columns={
            "score_status": "source_score_status",
            "kge": "source_kge",
            "nse": "source_nse",
        }
    )
    out = df.merge(source, on=["split", "station_id"], how="left")
    out["delta_kge_vs_source"] = out["kge"] - out["source_kge"]
    out["delta_nse_vs_source"] = out["nse"] - out["source_nse"]
    out["retained_close_kge"] = out["delta_kge_vs_source"].ge(-0.1)
    out["retained_close_nse"] = out["delta_nse_vs_source"].ge(-0.1)
    out["retained_close_both"] = out["retained_close_kge"] & out["retained_close_nse"]
    out["retained_usable"] = out["kge"].ge(0.5) & out["nse"].ge(0.5)
    out["failure_kge_lt_0"] = out["kge"].lt(0.0)
    out["catastrophic_kge_lt_minus1"] = out["kge"].lt(-1.0)
    out["big_forgetting_delta_lt_minus05"] = out["delta_kge_vs_source"].lt(-0.5)
    return out


def summarize_lowflow_failure(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = df.loc[df["score_status"].eq("ok") & df["flow_status"].eq("ok")].copy()
    for (model, split, quartile), group in valid.groupby(
        ["model", "split", "flow_mean_quartile"],
        sort=False,
    ):
        n = len(group)
        rows.append(
            {
                "model": model,
                "split": split,
                "flow_mean_quartile": quartile,
                "n": n,
                "median_obs_mean": float(group["obs_mean"].median()),
                "median_kge": float(group["kge"].median()),
                "median_nse": float(group["nse"].median()),
                "kge_lt_0_count": int(group["kge"].lt(0.0).sum()),
                "kge_lt_0_rate": float(group["kge"].lt(0.0).mean()),
                "kge_lt_minus1_count": int(group["kge"].lt(-1.0).sum()),
                "kge_lt_minus1_rate": float(group["kge"].lt(-1.0).mean()),
                "nse_lt_minus1_count": int(group["nse"].lt(-1.0).sum()),
                "nse_lt_minus1_rate": float(group["nse"].lt(-1.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_metric_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = df.loc[
        df["score_status"].eq("ok") & df["flow_status"].eq("ok") & df["obs_mean"].gt(0)
    ].copy()
    valid["log_obs_mean"] = np.log10(valid["obs_mean"])
    for (model, split), group in valid.groupby(["model", "split"], sort=False):
        for metric in ("kge", "nse"):
            finite = group[np.isfinite(group["log_obs_mean"]) & np.isfinite(group[metric])]
            if len(finite) < 3:
                continue
            low = finite.loc[finite["flow_mean_quartile"].eq("Q1-low"), metric]
            high = finite.loc[finite["flow_mean_quartile"].eq("Q4-high"), metric]
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "metric": metric,
                    "n": int(len(finite)),
                    "spearman_log_meanflow_vs_metric": float(
                        stats.spearmanr(finite["log_obs_mean"], finite[metric]).statistic
                    ),
                    "pearson_log_meanflow_vs_metric": float(
                        stats.pearsonr(finite["log_obs_mean"], finite[metric]).statistic
                    ),
                    "median_metric_lowest_flow_quartile": float(low.median()),
                    "median_metric_highest_flow_quartile": float(high.median()),
                }
            )
    return pd.DataFrame(rows)


def summarize_retention_failures(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = df.loc[
        df["model"].isin(TRANSFER_MODELS)
        & df["score_status"].eq("ok")
        & df["source_score_status"].eq("ok")
        & df["flow_status"].eq("ok")
    ].copy()
    for (model, split), group in valid.groupby(["model", "split"], sort=False):
        n = len(group)
        catastrophic = group.loc[group["catastrophic_kge_lt_minus1"]]
        retained = group.loc[group["retained_close_kge"]]
        rows.append(
            {
                "model": model,
                "split": split,
                "valid_station_splits": n,
                "retained_close_kge_count": int(group["retained_close_kge"].sum()),
                "retained_close_kge_rate": float(group["retained_close_kge"].mean()),
                "retained_usable_count": int(group["retained_usable"].sum()),
                "retained_usable_rate": float(group["retained_usable"].mean()),
                "failure_kge_lt_0_count": int(group["failure_kge_lt_0"].sum()),
                "failure_kge_lt_0_rate": float(group["failure_kge_lt_0"].mean()),
                "catastrophic_kge_lt_minus1_count": int(
                    group["catastrophic_kge_lt_minus1"].sum()
                ),
                "catastrophic_kge_lt_minus1_rate": float(
                    group["catastrophic_kge_lt_minus1"].mean()
                ),
                "big_forgetting_delta_lt_minus05_count": int(
                    group["big_forgetting_delta_lt_minus05"].sum()
                ),
                "big_forgetting_delta_lt_minus05_rate": float(
                    group["big_forgetting_delta_lt_minus05"].mean()
                ),
                "median_delta_kge": float(group["delta_kge_vs_source"].median()),
                "median_source_kge": float(group["source_kge"].median()),
                "median_transfer_kge": float(group["kge"].median()),
                "median_obs_mean_retained_close": float(retained["obs_mean"].median()),
                "median_obs_mean_catastrophic": float(catastrophic["obs_mean"].median()),
            }
        )
    return pd.DataFrame(rows)


def summarize_station_classes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    valid = df.loc[
        df["model"].isin(TRANSFER_MODELS)
        & df["score_status"].eq("ok")
        & df["source_score_status"].eq("ok")
        & df["flow_status"].eq("ok")
    ].copy()
    for (model, station_id), group in valid.groupby(["model", "station_id"], sort=False):
        test = group.loc[group["split"].eq("test")]
        rows.append(
            {
                "model": model,
                "station_id": station_id,
                "valid_splits": int(len(group)),
                "all_valid_splits_retained_close_kge": bool(group["retained_close_kge"].all()),
                "any_valid_split_catastrophic_kge_lt_minus1": bool(
                    group["catastrophic_kge_lt_minus1"].any()
                ),
                "test_retained_close_kge": bool(test["retained_close_kge"].iloc[0])
                if len(test)
                else np.nan,
                "test_failure_kge_lt_0": bool(test["failure_kge_lt_0"].iloc[0])
                if len(test)
                else np.nan,
                "test_catastrophic_kge_lt_minus1": bool(
                    test["catastrophic_kge_lt_minus1"].iloc[0]
                )
                if len(test)
                else np.nan,
                "median_obs_mean_across_valid_splits": float(group["obs_mean"].median()),
                "median_flow_pct_rank_across_valid_splits": float(
                    group["obs_mean_pct_rank"].median()
                ),
            }
        )
    classes = pd.DataFrame(rows)
    counts = []
    for model, group in classes.groupby("model", sort=False):
        counts.append(
            {
                "model": model,
                "stations": int(len(group)),
                "all_valid_splits_retained": int(
                    group["all_valid_splits_retained_close_kge"].sum()
                ),
                "any_split_catastrophic": int(
                    group["any_valid_split_catastrophic_kge_lt_minus1"].sum()
                ),
                "test_retained": int(group["test_retained_close_kge"].fillna(False).sum()),
                "test_failure": int(group["test_failure_kge_lt_0"].fillna(False).sum()),
                "test_catastrophic": int(
                    group["test_catastrophic_kge_lt_minus1"].fillna(False).sum()
                ),
            }
        )
    return classes, pd.DataFrame(counts)


def summarize_without_lowflow(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = df.loc[df["score_status"].eq("ok") & df["flow_status"].eq("ok")].copy()
    filters = {
        "all_valid": valid,
        "without_lowest_flow_quartile": valid.loc[
            ~valid["flow_mean_quartile"].eq("Q1-low")
        ],
    }
    for filter_name, sub in filters.items():
        for (model, split), group in sub.groupby(["model", "split"], sort=False):
            n = len(group)
            rows.append(
                {
                    "filter": filter_name,
                    "model": model,
                    "split": split,
                    "n_valid_stations": n,
                    "median_kge": float(group["kge"].median()),
                    "median_nse": float(group["nse"].median()),
                    "kge_lt_0_count": int(group["kge"].lt(0.0).sum()),
                    "kge_lt_0_rate": float(group["kge"].lt(0.0).mean()),
                    "kge_lt_minus1_count": int(group["kge"].lt(-1.0).sum()),
                    "kge_lt_minus1_rate": float(group["kge"].lt(-1.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_kge_flow_bins(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    test = df.loc[
        df["split"].eq("test")
        & df["score_status"].eq("ok")
        & df["flow_status"].eq("ok")
        & df["obs_mean"].gt(0)
    ].copy()
    station_bins = (
        test[["station_id", "obs_mean"]]
        .drop_duplicates("station_id")
        .sort_values("obs_mean")
        .copy()
    )
    station_bins["flow_bin"] = pd.qcut(
        station_bins["obs_mean"].rank(method="first"),
        q=n_bins,
        labels=False,
    )
    bin_map = dict(zip(station_bins["station_id"], station_bins["flow_bin"]))
    test["flow_bin"] = test["station_id"].map(bin_map)
    rows = []
    for (model, flow_bin), group in test.groupby(["model", "flow_bin"], sort=False):
        rows.append(
            {
                "model": model,
                "flow_bin": int(flow_bin),
                "n": int(len(group)),
                "median_obs_mean": float(group["obs_mean"].median()),
                "median_kge": float(group["kge"].median()),
                "q25_kge": float(group["kge"].quantile(0.25)),
                "q75_kge": float(group["kge"].quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    test = df.loc[df["split"].eq("test") & df["model"].isin(TRANSFER_MODELS)]
    lines = [
        "Source-domain transfer degradation analysis",
        "",
        "Definitions:",
        "- retained_close_kge: transfer KGE is within 0.1 of the original source model",
        "- retained_usable: KGE >= 0.5 and NSE >= 0.5",
        "- catastrophic_kge_lt_minus1: KGE < -1",
        "- big_forgetting_delta_lt_minus05: transfer KGE - source KGE < -0.5",
        "",
        "Test split medians:",
    ]
    for (model, _split), group in test.groupby(["model", "split"], sort=False):
        lines.append(
            f"- {model}: median KGE={group['kge'].median():.4f}, "
            f"median NSE={group['nse'].median():.4f}, "
            f"KGE<0 rate={group['failure_kge_lt_0'].mean():.3f}, "
            f"KGE<-1 rate={group['catastrophic_kge_lt_minus1'].mean():.3f}"
        )
    (out_dir / "source_domain_degradation_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    metrics_path = (
        Path(args.metrics_with_flow)
        if args.metrics_with_flow
        else out_dir / "per_station_source_domain_metrics_with_flow.csv"
    )
    df = pd.read_csv(metrics_path, dtype={"station_id": str})
    if "flow_mean_quartile" not in df.columns or df["flow_mean_quartile"].isna().any():
        df = assign_flow_ranks(df)
    df = add_degradation_flags(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_station_source_domain_metrics_with_flow.csv", index=False)
    summarize_lowflow_failure(df).to_csv(out_dir / "lowflow_failure_by_quartile.csv", index=False)
    summarize_metric_correlations(df).to_csv(out_dir / "lowflow_metric_correlations.csv", index=False)
    summarize_retention_failures(df).to_csv(out_dir / "retention_failure_counts.csv", index=False)
    classes, counts = summarize_station_classes(df)
    classes.to_csv(out_dir / "station_level_retention_classes.csv", index=False)
    counts.to_csv(out_dir / "station_level_retention_counts.csv", index=False)
    summarize_without_lowflow(df).to_csv(out_dir / "summary_without_lowflow_stations.csv", index=False)
    summarize_kge_flow_bins(df, args.flow_bins).to_csv(
        out_dir / "kge_vs_flow_test_binned_summary.csv",
        index=False,
    )
    write_summary(out_dir, df)
    print(f"Wrote source-domain degradation summaries to {out_dir}")


if __name__ == "__main__":
    main()
