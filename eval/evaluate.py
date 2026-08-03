"""Per-station evaluation, shared by every Phase I step.

``evaluate_model`` returns hourly metrics always, and daily-aggregate metrics
when the loader was built with ``with_daily=True``. Phase I Step 2 needs the
daily metrics for model selection precisely because the hourly ones must stay
unseen until the very end.

Also usable directly:

    python -m eval.evaluate --checkpoint outputs/fold0/pretrain/best_model.pth \
        --fold 0 --domain target --split validation --tag M0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.config import add_common_args, load_config, resolve
from common.metrics import StationAccumulator, summarize
from common.utils import get_device, setup_logging
from data.dataset import build_dataset, load_dataset_config, load_scalers, make_loader, resolve_static_spec
from data.folds import domain_stations, load_folds
from models.losses import daily_aggregate_prediction
from models.mtslstm import build_model


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    y_mean: float,
    y_std: float,
    min_samples: int = 1,
    daily_window: int | None = None,
    criterion=None,
    logger=None,
) -> dict[str, pd.DataFrame]:
    """Per-station hourly (and optionally daily) NSE/KGE in physical units.

    Pass ``criterion`` to also accumulate the objective on this split, so the
    validation loss is directly comparable with the training loss instead of
    only sharing an epoch axis with it.
    """
    model.eval()
    hourly = StationAccumulator()
    daily = StationAccumulator() if daily_window else None
    n_batches = n_rows = 0
    loss_totals: dict[str, float] = {}

    for batch in loader:
        stations = batch["stations"]
        if not stations:
            continue
        x = {key: value.to(device, non_blocking=True) for key, value in batch["x"].items()}
        outputs = model({"D": x["D"], "H": x["H"]}, x["S"])

        hourly.update(stations, outputs["H"].float().cpu().numpy(), batch["y"].numpy())

        if criterion is not None:
            parts = criterion(outputs, batch["y"].to(device), batch["stn_std"].to(device))
            for key, value in parts.items():
                loss_totals[key] = loss_totals.get(key, 0.0) + float(value.item()) * len(stations)

        if daily is not None:
            y_daily = batch.get("y_daily")
            if y_daily is None:
                raise ValueError("daily metrics requested but the dataset was built with with_daily=False")
            y_daily_np = y_daily.numpy()
            keep = np.isfinite(y_daily_np)
            if keep.any():
                mask = batch.get("daily_mask")
                if mask is not None:
                    mask = mask.to(device, non_blocking=True)
                pred = daily_aggregate_prediction(outputs, daily_window, mask).float().cpu().numpy()
                daily.update(
                    [s for s, k in zip(stations, keep) if k],
                    pred[keep],
                    y_daily_np[keep],
                )

        n_batches += 1
        n_rows += len(stations)
        if logger and n_batches % 200 == 0:
            logger.info("  evaluated %d batches / %d samples", n_batches, n_rows)

    result = {"hourly": hourly.to_frame(y_mean, y_std, min_samples)}
    if daily is not None:
        result["daily"] = daily.to_frame(y_mean, y_std, min_samples)
    if criterion is not None:
        result["losses"] = {k: v / max(n_rows, 1) for k, v in loss_totals.items()}
    return result


def build_eval_loader(cfg, stations: set[str], split: str, with_daily: bool, logger=None):
    dataset = build_dataset(
        cfg, split, stations, with_daily=with_daily,
        max_batches_per_station=int(cfg.get_path("eval.max_batches_per_station", 0)),
        logger=logger,
    )
    loader = make_loader(
        dataset,
        num_workers=int(cfg.get_path("train.num_workers", 4)),
        pin_memory=bool(cfg.get_path("train.pin_memory", False)) and torch.cuda.is_available(),
    )
    return dataset, loader


def write_results(results: dict[str, pd.DataFrame], out_dir: Path, tag: str) -> dict[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for scale, frame in results.items():
        frame.to_csv(out_dir / f"per_station_{scale}_{tag}.csv", index=False)
        summary = summarize(frame, split=f"{tag}_{scale}")
        summaries[scale] = summary
        by_source = (
            frame.loc[frame["score_status"].eq("ok")]
            .groupby("source")[["kge", "nse"]]
            .median()
            .reset_index()
        )
        by_source.to_csv(out_dir / f"by_source_{scale}_{tag}.csv", index=False)
    with open(out_dir / f"summary_{tag}.json", "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    return summaries


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Evaluate a Phase I checkpoint."))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--domain", choices=["source", "target"], required=True)
    parser.add_argument("--split", choices=["training", "validation"], default="validation")
    parser.add_argument("--tag", default=None, help="Name for the output files (default: derived).")
    parser.add_argument("--daily", action="store_true", help="Also report daily-aggregate metrics.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    tag = args.tag or f"fold{args.fold}_{args.domain}_{args.split}"
    out_dir = Path(args.out_dir) if args.out_dir else resolve(cfg.output_root) / f"fold{args.fold}" / "eval"
    logger = setup_logging(out_dir / f"evaluate_{tag}.log")

    folds = load_folds(resolve(cfg.folds.file))
    source_stations, target_stations = domain_stations(folds, args.fold)
    stations = source_stations if args.domain == "source" else target_stations
    logger.info("fold %d %s domain: %d stations", args.fold, args.domain, len(stations))

    device = get_device()
    scalers = load_scalers(cfg.data.root)
    static_keep, onehot_specs, static_names = resolve_static_spec(
        cfg.data.root, cfg.data.get("static_exclude"), cfg.data.get("onehot_static")
    )
    dyn_size = len(load_dataset_config(cfg.data.root)["dyn_features"])

    model = build_model(cfg, dyn_input_size=dyn_size, static_input_size=len(static_names)).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    logger.info("loaded checkpoint %s", args.checkpoint)

    _, loader = build_eval_loader(cfg, stations, args.split, with_daily=args.daily, logger=logger)
    results = evaluate_model(
        model,
        loader,
        device,
        scalers["y_mean"],
        scalers["y_std"],
        min_samples=int(cfg.get_path("eval.min_samples_per_station", 1)),
        daily_window=24 if args.daily else None,
        logger=logger,
    )
    summaries = write_results(results, out_dir, tag)
    for scale, summary in summaries.items():
        logger.info("%s %s: median KGE %.4f | median NSE %.4f | %d/%d stations scored",
                    tag, scale, summary["median_kge"], summary["median_nse"],
                    summary["n_valid_stations"], summary["n_stations"])


if __name__ == "__main__":
    main()
