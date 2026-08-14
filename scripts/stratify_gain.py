"""Where is daily-only supervision actually useful? (PLAN.md 5, items 1-3)

The headline ΔKGE is one number over ~8,900 stations. It hides the question the
global scale-up was for: which catchments does daily-aggregate supervision help, and
which does it not? This cuts the per-station gain by

  1. climate and region -- Koeppen-Geiger major/detailed zone, source agency
     (a continent proxy), catchment area, record length
  2. **distance to the nearest source-domain station** -- the direct quantification
     of "a random split is easy". The blocked-split experiment already showed the
     aggregate cost (M0 fell 0.128); this turns it into a curve, so the reader can
     see how skill decays as the nearest trainable neighbour recedes
  3. hydrological character -- flashiness, best_lag, max_lag_corr, q95 event rate.
     Snow- and reservoir-dominated catchments (low max_lag_corr) are expected to
     gain least, since their storage dynamics are not recoverable from a daily total

Everything runs off the per-station diagnostic CSVs, so no retraining and no GPU.

Degenerate stations are dropped exactly as in diagnose_kge: alpha and beta divide by
std(obs) and mean(obs), so a near-constant record produces meaningless ratios.

    python -m scripts.stratify_gain --run outputs/runB_truedaily/diagnostics_allhours
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STATIC_CSV = (
    "/ibex/project/c2266/abbaa0a/data/gscad_database/processed/20250630/"
    "hourly/dataframes/static.csv"
)
BASIN_FEATURES = "/home/kongw0a/EDA_global_hourly_runoff/tables/basin_features.csv"
MIN_GROUP = 30          # below this a group median is noise, not a stratum
MIN_OBS_STD = 1e-3      # mm/h


def load_covariates(station_ids: list[str], folds_file: str, logger=None) -> pd.DataFrame:
    """Static attributes, hydrological features and neighbour distance, per station."""
    static = pd.read_csv(STATIC_CSV, comment="#", index_col=0)
    keep = {
        "area": "area_km2",
        "lat": "lat",
        "long": "long",
        "KGZ_major": "kgz_major",
        "KGZ_detailed": "kgz_detailed",
        "WorldClim_V21_Pmean": "pmean",
        "WorldClim_V2_snowfall_fraction_P.mat": "snow_fraction",
        "reservoir_impact_GRanD_v1_3": "reservoir_impact",
        "elevation_SRTM_v2.1_GTOPO30": "elevation",
    }
    present = {k: v for k, v in keep.items() if k in static.columns}
    out = static.loc[static.index.intersection(station_ids), list(present)].rename(columns=present)

    try:
        feats = pd.read_csv(BASIN_FEATURES).set_index("station_id")
        wanted = ["years_q_valid", "best_lag", "max_lag_corr", "flashiness_log1p",
                  "q95_events_per_year", "q95_median_sharpness"]
        out = out.join(feats[[c for c in wanted if c in feats.columns]], how="left")
    except FileNotFoundError:
        if logger:
            logger.warning("%s not found -- skipping hydrological features", BASIN_FEATURES)

    # Distance to the nearest station in another fold == nearest station the model was
    # NOT allowed to train on for that fold. Written by scripts.make_folds_blocked.
    try:
        folds = pd.read_csv(folds_file).set_index("station_id")
        for column in ("nearest_other_fold_km_random", "nearest_other_fold_km_blocked"):
            if column in folds.columns:
                out[column] = folds[column]
    except FileNotFoundError:
        if logger:
            logger.warning("%s not found -- skipping neighbour distance", folds_file)
    return out


def stratify(table: pd.DataFrame, column: str, gain: str = "gain",
             bins: int = 5, categorical: bool = False) -> pd.DataFrame:
    """Median gain within each level (or quantile bin) of ``column``."""
    frame = table.dropna(subset=[column, gain]).copy()
    if frame.empty:
        return pd.DataFrame()
    if categorical:
        frame["_group"] = frame[column].astype(str)
    else:
        try:
            frame["_group"] = pd.qcut(frame[column], bins, duplicates="drop")
        except ValueError:
            return pd.DataFrame()

    rows = []
    for level, block in frame.groupby("_group", observed=True):
        if len(block) < MIN_GROUP:
            continue
        rows.append({
            "variable": column,
            "group": str(level),
            "n_stations": len(block),
            "covariate_median": float(np.nanmedian(block[column])) if not categorical else float("nan"),
            "M0_kge": float(block["M0_kge"].median()),
            "M1_kge": float(block["M1_kge"].median()),
            "gain": float(block[gain].median()),
            "frac_improved": float((block[gain] > 0).mean()),
            "M0_alpha": float(block["M0_kge_alpha"].median()),
            "M1_alpha": float(block["M1_kge_alpha"].median()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratify the M1-M0 gain by catchment properties.")
    parser.add_argument("--run", default="outputs/runB_truedaily/diagnostics_allhours",
                        help="Directory holding kge_components_target.csv")
    parser.add_argument("--domain", default="target")
    parser.add_argument("--folds-file", default="folds/folds_blocked.csv")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger("stratify")

    run = Path(args.run)
    out_dir = Path(args.out_dir) if args.out_dir else run / "stratified"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(run / f"kge_components_{args.domain}.csv")
    before = len(table)
    table = table.loc[table["obs_std"] >= MIN_OBS_STD].copy()
    table["gain"] = table["M1_kge"] - table["M0_kge"]
    logger.info("%d stations (%d dropped as numerically degenerate) | median gain %+.4f",
                len(table), before - len(table), table["gain"].median())

    cov = load_covariates(table["station_id"].tolist(), args.folds_file, logger)
    table = table.merge(cov, left_on="station_id", right_index=True, how="left")
    table.to_csv(out_dir / f"gain_with_covariates_{args.domain}.csv", index=False)

    specs = [
        ("source", True), ("kgz_major", True), ("kgz_detailed", True),
        ("area_km2", False), ("years_q_valid", False), ("pmean", False),
        ("snow_fraction", False), ("reservoir_impact", False), ("elevation", False),
        ("best_lag", False), ("max_lag_corr", False), ("flashiness_log1p", False),
        ("q95_events_per_year", False), ("q95_median_sharpness", False),
        ("nearest_other_fold_km_random", False), ("nearest_other_fold_km_blocked", False),
    ]
    pieces = []
    for column, categorical in specs:
        if column not in table.columns:
            logger.warning("no column %s -- skipped", column)
            continue
        piece = stratify(table, column, categorical=categorical)
        if piece.empty:
            logger.warning("%s: every group below %d stations -- skipped", column, MIN_GROUP)
            continue
        pieces.append(piece)
        logger.info("\n%s", piece.to_string(index=False, float_format=lambda v: f"{v: .4f}"))

    if not pieces:
        raise SystemExit("no covariate produced a usable stratification")
    combined = pd.concat(pieces, ignore_index=True)
    combined.to_csv(out_dir / f"stratified_gain_{args.domain}.csv", index=False)

    # Rank the covariates by how much the gain actually varies across their levels --
    # a covariate whose strata all agree explains nothing, however plausible it looked.
    spread = (
        combined.groupby("variable")
        .agg(n_groups=("group", "size"), gain_min=("gain", "min"), gain_max=("gain", "max"))
        .assign(gain_spread=lambda d: d["gain_max"] - d["gain_min"])
        .sort_values("gain_spread", ascending=False)
    )
    logger.info("\ncovariates ranked by how much the gain varies across their strata:\n%s",
                spread.to_string(float_format=lambda v: f"{v: .4f}"))
    spread.to_csv(out_dir / f"covariate_ranking_{args.domain}.csv")

    # Spearman on the continuous ones: monotone trend, robust to the binning above.
    from scipy.stats import spearmanr

    trends = []
    for column, categorical in specs:
        if categorical or column not in table.columns:
            continue
        pair = table[[column, "gain"]].dropna()
        if len(pair) < MIN_GROUP:
            continue
        result = spearmanr(pair[column], pair["gain"])
        trends.append({"variable": column, "n": len(pair),
                       "spearman_rho": float(result.statistic), "p": float(result.pvalue)})
    trend_frame = pd.DataFrame(trends).sort_values("spearman_rho", key=abs, ascending=False)
    logger.info("\nmonotone trend of the gain (Spearman):\n%s",
                trend_frame.to_string(index=False, float_format=lambda v: f"{v: .4g}"))
    trend_frame.to_csv(out_dir / f"gain_trends_{args.domain}.csv", index=False)

    (out_dir / f"stratified_summary_{args.domain}.json").write_text(json.dumps({
        "run": str(run), "domain": args.domain,
        "n_stations": int(len(table)),
        "median_gain": float(table["gain"].median()),
        "covariate_ranking": spread.reset_index().to_dict(orient="records"),
        "trends": trends,
    }, indent=2))


if __name__ == "__main__":
    main()
