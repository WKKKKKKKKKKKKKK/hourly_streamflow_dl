"""Save the model's HOURLY African predictions for a handful of catchments.

``scripts.africa_insitu_ensemble`` keeps only the daily aggregate: it computes
``daily_aggregate_prediction(out, 24)``, the mean of the last 24 hourly steps, and
discards the 24 values themselves. That is the right thing for scoring -- African
discharge is observed daily, so daily is the only resolution at which a score can be
computed -- but it leaves the project with no hourly African series to plot at all.

This script re-runs the same five folds through the same validation windows and keeps
the hourly tail instead of averaging it. Nothing about the models or the windows
changes; the only difference is what is written out.

The mapping from tensor to wall clock is exact and comes from AfricaWindowDataset:
the target hour is 23:00 of the observed day, so ``H_seq``'s last 24 steps are hours
00:00..23:00 of that calendar date, in the forcing's own UTC.

Units: the hourly values are mm/h (the trained target unit). The daily observation is
mm/d. ``24 * mean(hourly)`` is what the daily score used, so an hourly curve whose
daily mean lands on the observed daily value is the same statement the KGE table makes.

IMPORTANT -- there is no hourly observation for any African catchment. That absence is
why Africa is the external test in this experiment. The hourly curves written here can
be compared with each other, but they cannot be scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from common.config import add_common_args, load_config, resolve
from common.utils import get_device, setup_logging
from data.africa import (
    AfricaWindowDataset,
    apply_onehot,
    build_static_matrix,
    load_hourly_forcing,
    load_observed_daily,
)
from data.dataset import load_dataset_config, load_scalers, make_loader, resolve_static_spec
from models.mtslstm import build_model

DAILY_WINDOW = 24
DEFAULT_FORCING = (
    "/ibex/user/kongw0a/era5_land_africa_forcing/"
    "era5_land_africa_hourly_forcing_penman.nc"
)


def pick_stations(per_basin: Path, quantiles=(0.25, 0.50, 0.75)) -> list[str]:
    """The same three catchments figure 7 shows: quartiles of the M1 KGE distribution.

    Reproduced rather than hard-coded so the hourly figure cannot drift away from the
    daily one, and so the choice stays 'spanning the outcome' instead of 'three good ones'.
    """
    scored = pd.read_csv(per_basin)
    scored = scored.loc[np.isfinite(scored["kge"])].sort_values("kge").reset_index(drop=True)
    return [str(scored.iloc[int(len(scored) * q)]["station_id"]) for q in quantiles]


@torch.no_grad()
def predict_hourly(model, loader, device, y_mean: float, y_std: float,
                   codes: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flat arrays in loader order: station code, hour, hourly prediction, daily observation.

    Arrays rather than a DataFrame, and codes rather than strings, because the all-basin
    pass is 7.7 million rows and ten model passes: merging ten frames of that size on
    (station, hour) costs far more than it buys. ``make_loader`` uses a SequentialSampler
    when shuffle is off, so the row order is identical for every fold and the ensemble can
    be accumulated in place. The order is verified against fold 0 rather than assumed.
    """
    model.eval()
    station, hour, sim, obs = [], [], [], []
    for batch in loader:
        x = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
        out = model({"D": x["D"], "H": x["H"]}, x["S"])
        tail = out["H_seq"][:, -DAILY_WINDOW:].squeeze(-1).float().cpu().numpy()
        dates = pd.to_datetime(pd.Series(batch["dates"]).values).values
        n = len(dates)
        station.append(np.repeat(
            np.fromiter((codes[s] for s in batch["stations"]), dtype=np.int32, count=n),
            DAILY_WINDOW))
        # hour h of the target date: the 24 slots are 00:00..23:00 UTC of that day
        hour.append(np.repeat(dates, DAILY_WINDOW)
                    + np.tile(np.arange(DAILY_WINDOW), n) * np.timedelta64(1, "h"))
        sim.append((tail.reshape(-1) * y_std + y_mean).astype(np.float32))
        obs.append(np.repeat(batch["y_daily_obs"].numpy().astype(np.float32), DAILY_WINDOW))
    return (np.concatenate(station), np.concatenate(hour),
            np.concatenate(sim), np.concatenate(obs))


def within_day_stats(frame: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Per basin: how much sub-daily shape the hourly curve carries.

    The within-day coefficient of variation -- sd of a day's 24 values over that day's
    own mean -- is used rather than the sd itself so the number does not simply track how
    wet the catchment is. Taking the median over days keeps a handful of near-zero-flow
    days from dominating, since dividing by a tiny mean can produce an arbitrarily large
    ratio.
    """
    # observed=True is not cosmetic here: station_id is a Categorical, and the default
    # would form the full 294-basin x every-date product, almost all of it empty.
    by_day = frame.groupby(["station_id", "date"], observed=True)[f"ensemble_{tag}"]
    cv = (by_day.std() / by_day.mean().replace(0, np.nan)).rename("cv")
    out = cv.groupby("station_id", observed=True).agg(["median", "count"])
    return out.rename(columns={"median": f"cv_{tag}", "count": f"n_days_{tag}"})


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(
        description="Write the hourly African predictions for a few catchments."))
    parser.add_argument("--insitu-glob", default="outputs/v2_africa_insitu_fold")
    parser.add_argument("--pretrain-root", default="outputs/v2_runB",
                        help="Where fold{k}/pretrain/best_model.pth (the M0 models) live.")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--forcing", default=DEFAULT_FORCING)
    parser.add_argument("--basins", default="africa/africa_basins.gpkg")
    parser.add_argument("--per-basin", default="outputs/v2_africa_insitu_summary/ensemble_per_basin_M1.csv")
    parser.add_argument("--stations", default=None,
                        help="Comma-separated station ids whose full hourly series is saved; "
                             "default = figure 7's three quartile picks.")
    parser.add_argument("--all-basins", action="store_true",
                        help="Run every African basin and write the within-day statistics for "
                             "all of them. The full hourly series is still saved only for "
                             "--stations: three catchments cannot support a claim about the "
                             "model, and 7.7 million rows of hourly output is not a "
                             "deliverable. Costs ~20 min on one GPU.")
    parser.add_argument("--out-dir", default="outputs/v2_africa_hourly")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "hourly_series.log")
    device = get_device()

    wanted = ([s.strip() for s in args.stations.split(",") if s.strip()]
              if args.stations else pick_stations(Path(args.per_basin)))
    logger.info("full hourly series for: %s", ", ".join(wanted))

    scalers = load_scalers(cfg.data.root)
    y_mean, y_std = float(scalers["y_mean"]), float(scalers["y_std"])
    dyn_size = len(load_dataset_config(cfg.data.root)["dyn_features"])
    names_all = list(load_dataset_config(cfg.data.root)["static_features"])
    static_keep, onehot_specs, static_names = resolve_static_spec(
        cfg.data.root, cfg.data.get("static_exclude"), cfg.data.get("onehot_static"))

    # Without --all-basins only the requested basins are loaded. The split is temporal per
    # basin, so dropping the others cannot move the validation window -- the same dates are
    # scored either way, which is what lets the two modes be compared.
    basins = gpd.read_file(resolve(args.basins))
    basins["station_id"] = basins["station_id"].astype(str)
    if args.all_basins:
        station_ids = basins["station_id"].tolist()
    else:
        keep = basins.loc[basins.station_id.isin(wanted)]
        station_ids = keep["station_id"].tolist()
    missing = sorted(set(wanted) - set(station_ids))
    if missing:
        raise SystemExit(f"not in {args.basins}: {missing}")
    logger.info("basins in this pass: %d", len(station_ids))

    forcing, forcing_times = load_hourly_forcing(args.forcing, station_ids, scalers, logger=logger)
    observed, observed_dates = load_observed_daily(station_ids, logger=logger)
    full, _ = build_static_matrix(station_ids, names_all, scalers, logger=logger)
    static = apply_onehot(full, static_keep, onehot_specs).astype(np.float32)

    valid_ds = AfricaWindowDataset(
        forcing=forcing, forcing_times=forcing_times, static=static, observed=observed,
        observed_dates=observed_dates, station_ids=station_ids,
        lookback_hourly=int(cfg.data.lookback_hourly),
        lookback_daily=int(cfg.data.get("lookback_daily", 365)),
        chunk_size=int(cfg.data.get("chunk_size", 512)),
        split="validation", train_frac=0.7, logger=logger)
    loader = make_loader(valid_ds, num_workers=int(cfg.get_path("transfer.num_workers", 4)),
                         pin_memory=device.type == "cuda")

    codes = {sid: i for i, sid in enumerate(station_ids)}
    names = np.asarray(station_ids, dtype=object)
    folds = [int(f) for f in args.folds.split(",") if f.strip()]
    index: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    total = {"M0": None, "M1": None}
    for fold in folds:
        m0 = Path(args.pretrain_root) / f"fold{fold}" / "pretrain" / "best_model.pth"
        m1 = Path(f"{args.insitu_glob}{fold}") / "best_africa_model.pth"
        for tag, path in (("M0", m0), ("M1", m1)):
            if not path.exists():
                raise SystemExit(f"fold {fold} {tag}: {path} missing")
            model = build_model(cfg, dyn_input_size=dyn_size,
                                static_input_size=len(static_names)).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            logger.info("fold %d %s: predicting hourly", fold, tag)
            station, hour, sim, obs = predict_hourly(model, loader, device, y_mean, y_std, codes)
            if index is None:
                index = (station, hour, obs)
            elif not (len(station) == len(index[0])
                      and np.array_equal(station, index[0])
                      and np.array_equal(hour, index[1])):
                # If this ever fires, accumulating across folds would silently add up
                # predictions for different basin-hours.
                raise SystemExit(f"fold {fold} {tag}: row order differs from fold "
                                 f"{folds[0]}; the ensemble cannot be accumulated")
            total[tag] = sim if total[tag] is None else total[tag] + sim

    station, hour, obs = index
    frame = pd.DataFrame({
        "station_id": pd.Categorical.from_codes(station, categories=names.tolist()),
        "time": hour,
        "obs_daily": obs,
        # The ensemble averages PREDICTIONS, matching how the scored daily table was built.
        "ensemble_M0": total["M0"] / len(folds),
        "ensemble_M1": total["M1"] / len(folds),
    })
    frame["date"] = frame["time"].values.astype("datetime64[D]").astype("datetime64[ns]")

    series = frame.loc[frame.station_id.astype(str).isin(wanted)].copy()
    series["station_id"] = series.station_id.astype(str)
    series = series[["station_id", "time", "date", "obs_daily", "ensemble_M0", "ensemble_M1"]]
    series = series.sort_values(["station_id", "time"]).reset_index(drop=True)
    series_path = out_dir / "hourly_series.csv.gz"
    series.to_csv(series_path, index=False, compression="gzip")
    logger.info("wrote %s: %d rows, %d basins, %s .. %s", series_path, len(series),
                series.station_id.nunique(), series.time.min(), series.time.max())

    # The check that matters: 24 * mean(hourly) must reproduce the daily prediction the
    # scored table was built from. If it does, this hourly series is the same models on the
    # same windows and inherits their KGE; if it does not, it is a different run and cannot
    # be plotted beside figure 7. Comparing the hourly mean to the OBSERVATION instead would
    # only re-measure model error, and as a mean-of-ratios it is dominated by near-zero
    # observation days -- not a check on anything.
    for tag in ("M0", "M1"):
        published = Path(args.per_basin).parent / f"ensemble_series_{tag}.csv.gz"
        if not published.exists():
            logger.warning("%s: %s absent, cannot cross-check against the scored table",
                           tag, published)
            continue
        daily = (series.groupby(["station_id", "date"], as_index=False)[f"ensemble_{tag}"]
                 .mean().rename(columns={f"ensemble_{tag}": "from_hourly"}))
        daily["from_hourly"] *= DAILY_WINDOW
        ref = pd.read_csv(published, parse_dates=["date"])
        both = daily.merge(ref[["station_id", "date", "ensemble"]],
                           on=["station_id", "date"], how="inner")
        gap = (both.from_hourly - both.ensemble).abs()
        logger.info("%s: 24 x mean(hourly) vs the scored daily prediction over %d "
                    "basin-days -- max |diff| %.2e mm/d (float32 rounding is ~1e-6)",
                    tag, len(both), float(gap.max()))

    # How much sub-daily shape survives daily-only supervision. Over every basin when
    # --all-basins is set, so this is a distribution and not three anecdotes.
    stats = within_day_stats(frame, "M0").join(within_day_stats(frame, "M1"), how="inner")
    stats = stats.reset_index()
    stats["station_id"] = stats.station_id.astype(str)
    stats["ratio_M1_over_M0"] = stats.cv_M1 / stats.cv_M0.replace(0, np.nan)
    stats_path = out_dir / "within_day_cv_per_basin.csv"
    stats.to_csv(stats_path, index=False)

    paired = stats.loc[np.isfinite(stats.cv_M0) & np.isfinite(stats.cv_M1)]
    summary = {
        "n_basins": int(len(paired)),
        "median_cv_M0": float(paired.cv_M0.median()),
        "median_cv_M1": float(paired.cv_M1.median()),
        "median_paired_difference": float((paired.cv_M1 - paired.cv_M0).median()),
        "share_of_basins_with_higher_cv_after_finetuning":
            float((paired.cv_M1 > paired.cv_M0).mean()),
    }
    if len(paired) >= 8:
        from scipy.stats import wilcoxon
        stat, pval = wilcoxon(paired.cv_M1, paired.cv_M0)
        # Paired over basins, because the two models are scored on the same catchments and
        # the same days; an unpaired test would throw that away.
        summary["wilcoxon_statistic"] = float(stat)
        summary["wilcoxon_p"] = float(pval)
    with open(out_dir / "within_day_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("within-day CV over %d basins: M0 median %.4f, M1 median %.4f, "
                "median paired difference %+.4f, higher after fine-tuning in %.1f%% of basins%s",
                summary["n_basins"], summary["median_cv_M0"], summary["median_cv_M1"],
                summary["median_paired_difference"],
                100 * summary["share_of_basins_with_higher_cv_after_finetuning"],
                f", Wilcoxon p = {summary['wilcoxon_p']:.3g}" if "wilcoxon_p" in summary else "")
    logger.info("wrote %s", stats_path)


if __name__ == "__main__":
    main()
