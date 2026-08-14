"""Does daily-aggregate supervision produce a flat line inside each day? (PLAN.md 5, item 4)

``loss_agg`` constrains only the mean of 24 hourly outputs. Nothing in it constrains
how the day is shaped, so there is a degenerate optimum: emit a constant value all
day. The daily aggregate would be perfect and the hourly series worthless. KGE on
hourly values would catch it eventually, but only after the fact and mixed together
with every other error, so measure the intra-day shape directly.

The stride-24 sampling makes this clean. Every sample's target sits at 23:00, so the
model's last 24 hourly outputs are exactly that calendar day; consecutive sample days
stitch into a continuous hourly series with no overlap and no gaps to interpolate.

Reported per station, for M0, M1 and the observations themselves:

  flashiness      Richards-Baker index, sum|q_t - q_(t-1)| / sum q_t -- the standard
                  measure of how jagged a hydrograph is. A within-day constant drives
                  the numerator toward only the day-to-day steps, so this collapses.
  intraday_std    mean over days of the within-day standard deviation. The most direct
                  statement of "is it a flat line": it goes to zero for the degenerate
                  solution regardless of whether the daily means are right.
  intraday_range  mean over days of (max - min) within the day.
  q95_events      exceedances per year of the OBSERVED q95, counted as runs, so a
                  smoothed prediction that never crosses the threshold shows up.

    python -m scripts.degenerate_check --config configs/phase1_runB.yaml --folds 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.config import add_common_args, load_config, resolve
from common.utils import get_device, setup_logging
from data.dataset import load_dataset_config, load_scalers, make_loader, resolve_static_spec
from data.folds import domain_stations, load_folds
from data.sources import build_eval_set
from models.mtslstm import build_model

DAILY_WINDOW = 24
MIN_DAYS = 200


def flashiness(series: np.ndarray) -> float:
    """Richards-Baker index. Undefined for a zero-volume series."""
    total = np.nansum(series)
    if not np.isfinite(total) or total <= 0:
        return float("nan")
    return float(np.nansum(np.abs(np.diff(series))) / total)


def count_events(series: np.ndarray, threshold: float) -> int:
    """Runs above ``threshold``, so one long peak counts once rather than per hour."""
    above = series > threshold
    if above.size == 0:
        return 0
    return int(np.count_nonzero(above[1:] & ~above[:-1]) + int(above[0]))


def shape_stats(daily_blocks: np.ndarray, threshold: float, days_per_year: float) -> dict:
    """``daily_blocks`` is (n_days, 24) in physical units, days consecutive."""
    flat = daily_blocks.reshape(-1)
    return {
        "flashiness": flashiness(flat),
        "intraday_std": float(np.nanmean(np.nanstd(daily_blocks, axis=1))),
        "intraday_range": float(np.nanmean(np.nanmax(daily_blocks, axis=1) - np.nanmin(daily_blocks, axis=1))),
        "q95_events_per_year": count_events(flat, threshold) / max(days_per_year, 1e-9),
        "mean": float(np.nanmean(flat)),
    }


@torch.no_grad()
def collect_hourly(model, loader, device, y_mean: float, y_std: float, logger=None):
    """(station -> list of (day_index, 24 predicted hours, 24 observed hours)).

    The observed hours come from the loader's own hourly targets where available; the
    aggregate path only carries the target hour, so the observed day is reconstructed
    from the 24 consecutive samples that share a calendar day when present. Here the
    per-sample target IS the 23:00 hour, so observations are handled separately by the
    caller from the cache.
    """
    model.eval()
    per_station: dict[str, list] = {}
    for batch in loader:
        stations = batch["stations"]
        if not stations:
            continue
        x = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
        out = model({"D": x["D"], "H": x["H"]}, x["S"])
        seq = out["H_seq"][:, -DAILY_WINDOW:].float().cpu().numpy() * y_std + y_mean
        hours = batch["hours"].numpy()
        for station, hour, block in zip(np.asarray(stations, dtype=object), hours, seq):
            per_station.setdefault(str(station), []).append((int(hour), block))
    if logger:
        logger.info("collected hourly blocks for %d stations", len(per_station))
    return per_station


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Intra-day shape check (degenerate-solution diagnostic)."))
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--pretrain-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-samples-per-station", type=int, default=1500)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if str(cfg.data.get("source", "prepared")) != "hourly_cache":
        raise SystemExit(
            "this diagnostic needs the hourly-cache path: it stitches consecutive "
            "stride-24 days into a continuous hourly series, which the prepared "
            "subsampled layout cannot provide"
        )
    run_dir = Path(args.run_dir) if args.run_dir else resolve(cfg.output_root)
    pre_dir = Path(args.pretrain_dir) if args.pretrain_dir else run_dir
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "degenerate"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "degenerate_check.log")
    device = get_device()

    scalers = load_scalers(cfg.data.root)
    y_mean, y_std = float(scalers["y_mean"]), float(scalers["y_std"])
    dyn_size = len(load_dataset_config(cfg.data.root)["dyn_features"])
    _, _, static_names = resolve_static_spec(
        cfg.data.root, cfg.data.get("static_exclude"), cfg.data.get("onehot_static")
    )

    from data.hourly_windows import load_cache

    cache = load_cache(cfg.data.cache_dir, stride=int(cfg.data.get("stride", 24)), logger=logger)
    forcing = cache["forcing"]
    station_row = {name: i for i, name in enumerate(cache["stations"])}

    folds = load_folds(resolve(cfg.folds.file))
    rows = []
    for fold in [int(f) for f in args.folds.split(",") if f.strip()]:
        m0 = pre_dir / f"fold{fold}" / "pretrain" / "best_model.pth"
        m1 = run_dir / f"fold{fold}" / "transfer" / "best_transfer_model.pth"
        if not (m0.exists() and m1.exists()):
            logger.warning("fold %d: missing checkpoint (%s / %s) -- skipped", fold, m0.name, m1.name)
            continue
        _, target_stations = domain_stations(folds, fold)
        dataset = build_eval_set(cfg, target_stations, "validation", logger=logger)
        loader = make_loader(dataset, num_workers=int(cfg.get_path("transfer.num_workers", 4)),
                            pin_memory=device.type == "cuda")

        collected = {}
        for tag, path in (("M0", m0), ("M1", m1)):
            model = build_model(cfg, dyn_input_size=dyn_size, static_input_size=len(static_names)).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            logger.info("fold %d %s: predicting hourly shape", fold, tag)
            collected[tag] = collect_hourly(model, loader, device, y_mean, y_std, logger)

        common = set(collected["M0"]) & set(collected["M1"])
        logger.info("fold %d: %d stations with both passes", fold, len(common))
        for station in sorted(common):
            k = station_row.get(station)
            if k is None:
                continue
            entries = {tag: dict(collected[tag][station]) for tag in ("M0", "M1")}
            hours = sorted(set(entries["M0"]) & set(entries["M1"]))
            if len(hours) < MIN_DAYS:
                continue
            if args.max_samples_per_station:
                hours = hours[: args.max_samples_per_station]

            # Observed day: the 24 hours ending at the 23:00 target, from the cache.
            obs_blocks = np.stack([forcing[k, t - DAILY_WINDOW + 1 : t + 1, 3] for t in hours])
            keep = np.isfinite(obs_blocks).all(axis=1)
            if keep.sum() < MIN_DAYS:
                continue
            obs_blocks = obs_blocks[keep]
            kept_hours = [h for h, ok in zip(hours, keep) if ok]
            days_per_year = len(kept_hours) / 365.25
            threshold = float(np.nanpercentile(obs_blocks, 95))

            row = {"station_id": station, "fold": fold, "n_days": int(len(kept_hours))}
            for tag, prefix in (("obs", "obs"), ("M0", "M0"), ("M1", "M1")):
                if tag == "obs":
                    blocks = obs_blocks
                else:
                    blocks = np.stack([entries[tag][h] for h in kept_hours])
                stats = shape_stats(blocks, threshold, days_per_year)
                for name, value in stats.items():
                    row[f"{prefix}_{name}"] = value
            rows.append(row)

    if not rows:
        raise SystemExit("no station produced a usable hourly series")
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "intraday_shape.csv", index=False)

    logger.info("%d stations over %d folds", len(table), table["fold"].nunique())
    summary = {}
    logger.info("%-16s %>10s", "", "")
    header = f'{"metric":22s} {"observed":>10s} {"M0":>10s} {"M1":>10s} {"M1/obs":>8s} {"M1/M0":>8s}'
    logger.info(header)
    for metric in ("flashiness", "intraday_std", "intraday_range", "q95_events_per_year", "mean"):
        o = table[f"obs_{metric}"].median()
        a = table[f"M0_{metric}"].median()
        b = table[f"M1_{metric}"].median()
        summary[metric] = {"observed": float(o), "M0": float(a), "M1": float(b)}
        logger.info("%-22s %10.4f %10.4f %10.4f %8.3f %8.3f", metric, o, a, b,
                    b / o if o else float("nan"), b / a if a else float("nan"))

    # The degenerate solution is "intra-day variability collapses while the daily mean
    # stays right". State that combination explicitly instead of leaving it to be read
    # off the table.
    std_ratio = summary["intraday_std"]["M1"] / summary["intraday_std"]["observed"]
    flash_ratio = summary["flashiness"]["M1"] / summary["flashiness"]["observed"]
    verdict = (
        f"M1 keeps {std_ratio:.1%} of the observed within-day standard deviation and "
        f"{flash_ratio:.1%} of its flashiness. "
        + ("Intra-day shape has largely collapsed -- the daily-aggregate term is being "
           "satisfied by flattening the day, which is the degenerate solution PLAN.md 5.4 "
           "warns about." if std_ratio < 0.5 else
           "Intra-day variability survives, so the daily-aggregate term is not being "
           "satisfied by flattening the day.")
    )
    logger.info("VERDICT: %s", verdict)
    (out_dir / "degenerate_summary.json").write_text(json.dumps(
        {"n_stations": int(len(table)), "medians": summary, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
