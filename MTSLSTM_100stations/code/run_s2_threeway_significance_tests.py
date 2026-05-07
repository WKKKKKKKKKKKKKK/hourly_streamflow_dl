from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = EXP_DIR / "outputs" / "s2_random30_threeway_ppt_plots"
DEFAULT_INPUTS = {
    "baseline": EXP_DIR / "outputs" / "s2_random30_idx2_eval" / "per_station_metrics.csv",
    "transfer": EXP_DIR
    / "outputs"
    / "transfer_daily_to_hourly_partial_ft_s2_random30"
    / "per_station_hourly_metrics.csv",
    "symbolic": EXP_DIR
    / "outputs"
    / "transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.05"
    / "per_station_hourly_metrics.csv",
}
METHOD_LABELS = {
    "baseline": "Baseline",
    "transfer": "Transfer learning",
    "symbolic": "Transfer learning + symbolic prior",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired station-level Wilcoxon tests for the S2 Cfa-SE baseline, "
            "transfer-learning, and symbolic-prior models."
        )
    )
    parser.add_argument("--baseline", default=str(DEFAULT_INPUTS["baseline"]))
    parser.add_argument("--transfer", default=str(DEFAULT_INPUTS["transfer"]))
    parser.add_argument("--symbolic", default=str(DEFAULT_INPUTS["symbolic"]))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = p_values.to_numpy(dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running_min = 1.0
    n = len(p)
    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = n - rank_from_end + 1
        candidate = p[idx] * n / rank
        running_min = min(running_min, candidate)
        adjusted[idx] = min(running_min, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def paired_metric_table(
    dfs: dict[str, pd.DataFrame],
    comparison: str,
    numerator: str,
    denominator: str,
    alpha: float,
) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        for metric in ("kge", "nse"):
            left = dfs[denominator][
                ["station_id", f"{split}_score_status", f"{split}_{metric}"]
            ].rename(
                columns={
                    f"{split}_score_status": "denominator_status",
                    f"{split}_{metric}": "denominator_value",
                }
            )
            right = dfs[numerator][
                ["station_id", f"{split}_score_status", f"{split}_{metric}"]
            ].rename(
                columns={
                    f"{split}_score_status": "numerator_status",
                    f"{split}_{metric}": "numerator_value",
                }
            )
            paired = left.merge(right, on="station_id", how="inner")
            paired = paired.loc[
                paired["denominator_status"].eq("ok") & paired["numerator_status"].eq("ok")
            ].copy()
            paired = paired.loc[
                np.isfinite(paired["denominator_value"])
                & np.isfinite(paired["numerator_value"])
            ].copy()

            diff = paired["numerator_value"].to_numpy(dtype=float) - paired[
                "denominator_value"
            ].to_numpy(dtype=float)
            wilcoxon = stats.wilcoxon(
                diff,
                alternative="greater",
                zero_method="wilcox",
                method="auto",
            )
            paired_t = stats.ttest_rel(
                paired["numerator_value"],
                paired["denominator_value"],
                alternative="greater",
            )

            rows.append(
                {
                    "comparison": comparison,
                    "numerator": numerator,
                    "denominator": denominator,
                    "numerator_label": METHOD_LABELS[numerator],
                    "denominator_label": METHOD_LABELS[denominator],
                    "split": split,
                    "metric": metric.upper(),
                    "n_pairs": int(len(paired)),
                    "denominator_median": float(paired["denominator_value"].median()),
                    "numerator_median": float(paired["numerator_value"].median()),
                    "median_delta_numerator_minus_denominator": float(np.median(diff)),
                    "mean_delta_numerator_minus_denominator": float(np.mean(diff)),
                    "positive_pairs": int((diff > 0).sum()),
                    "negative_pairs": int((diff < 0).sum()),
                    "wilcoxon_stat": float(wilcoxon.statistic),
                    "wilcoxon_p_one_sided_greater": float(wilcoxon.pvalue),
                    "paired_t_p_one_sided_greater": float(paired_t.pvalue),
                }
            )

    results = pd.DataFrame(rows)
    fdr_col = f"wilcoxon_p_fdr_bh_{len(results)}_tests"
    results[fdr_col] = benjamini_hochberg(results["wilcoxon_p_one_sided_greater"])
    results["significant_raw"] = results["wilcoxon_p_one_sided_greater"] < alpha
    results["significant_fdr"] = results[fdr_col] < alpha
    return results


def main() -> None:
    args = parse_args()
    paths = {
        "baseline": Path(args.baseline),
        "transfer": Path(args.transfer),
        "symbolic": Path(args.symbolic),
    }
    dfs = {name: pd.read_csv(path, dtype={"station_id": str}) for name, path in paths.items()}

    transfer_vs_baseline = paired_metric_table(
        dfs,
        comparison="transfer_vs_baseline",
        numerator="transfer",
        denominator="baseline",
        alpha=args.alpha,
    )
    symbolic_vs_baseline = paired_metric_table(
        dfs,
        comparison="symbolic_vs_baseline",
        numerator="symbolic",
        denominator="baseline",
        alpha=args.alpha,
    )
    symbolic_vs_transfer = paired_metric_table(
        dfs,
        comparison="symbolic_vs_transfer",
        numerator="symbolic",
        denominator="transfer",
        alpha=args.alpha,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vs_baseline = pd.concat([transfer_vs_baseline, symbolic_vs_baseline], ignore_index=True)
    all_tests = pd.concat([vs_baseline, symbolic_vs_transfer], ignore_index=True)

    vs_baseline.to_csv(out_dir / "significance_tests_vs_baseline.csv", index=False)
    symbolic_vs_transfer.to_csv(
        out_dir / "symbolic_vs_transfer_significance_tests.csv",
        index=False,
    )
    all_tests.to_csv(out_dir / "s2_threeway_all_paired_significance_tests.csv", index=False)
    print(all_tests.to_string(index=False))


if __name__ == "__main__":
    main()
