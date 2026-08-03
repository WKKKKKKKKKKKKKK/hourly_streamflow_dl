"""Phase I Step 1 -- pretrain MTS-LSTM on the 80% source domain (hourly targets).

For one fold:

  * train on the source stations' ``training/`` batches with hourly supervision,
  * early-stop on median hourly KGE over the source stations' ``validation/``
    batches,
  * write ``best_model.pth`` plus per-station source-domain metrics.

The target stations are never loaded here -- evaluating this checkpoint on them
(``eval.evaluate --domain target``) is the zero-shot M0 baseline that Step 2 has
to beat.

    python -m train.pretrain_source --fold 0

The source pool is ~1.3M batch files (~8 TB), so an "epoch" is a random
``train.batches_per_epoch`` sample of it, drawn without replacement and
re-drawn each epoch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.config import add_common_args, load_config, resolve
from common.metrics import summarize
from common.utils import (
    EarlyStopping,
    apply_lr_schedule,
    atomic_save,
    get_device,
    init_wandb,
    parse_lr_schedule,
    set_seed,
    setup_logging,
    wandb_finish,
    wandb_log,
)
from data.dataset import (
    build_dataset,
    epoch_subset,
    load_dataset_config,
    load_scalers,
    make_loader,
    pick_per_station,
    resolve_static_spec,
    station_of,
)
from data.folds import domain_stations, load_folds
from eval.evaluate import evaluate_model, write_results
from models.losses import MTSBasinNSELoss
from models.mtslstm import build_model


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip, logger, log_every=100):
    model.train()
    totals: dict[str, float] = {}
    n_rows = 0
    n_batches = 0
    started = time.time()

    for batch in loader:
        if not batch["stations"]:
            continue
        x = {key: value.to(device, non_blocking=True) for key, value in batch["x"].items()}
        y = batch["y"].to(device, non_blocking=True)
        stn_std = batch["stn_std"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model({"D": x["D"], "H": x["H"]}, x["S"])
        parts = criterion(outputs, y, stn_std)
        loss = parts["loss"]
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()

        rows = y.shape[0]
        n_rows += rows
        n_batches += 1
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * rows

        if logger and n_batches % log_every == 0:
            rate = n_batches / max(time.time() - started, 1e-9)
            logger.info(
                "    batch %d/%d | loss %.5f | %.2f batch/s",
                n_batches,
                len(loader),
                totals["loss"] / max(n_rows, 1),
                rate,
            )

    return {key: value / max(n_rows, 1) for key, value in totals.items()}, n_rows


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Phase I Step 1: source-domain pretraining."))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint.pth if present.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    out_dir = Path(args.out_dir) if args.out_dir else resolve(cfg.output_root) / f"fold{args.fold}" / "pretrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "pretrain.log")
    set_seed(int(cfg.train.seed) + args.fold)

    device = get_device()
    logger.info("fold %d | device %s", args.fold, device)
    if device.type != "cuda":
        logger.warning("CUDA not available -- this will be far too slow for the full dataset.")

    # --- data -----------------------------------------------------------
    folds = load_folds(resolve(cfg.folds.file))
    source_stations, target_stations = domain_stations(folds, args.fold)
    logger.info("source domain %d stations | target domain %d stations (held out)",
                len(source_stations), len(target_stations))

    train_ds = build_dataset(cfg, "training", source_stations, with_daily=False, logger=logger)
    val_ds = build_dataset(cfg, "validation", source_stations, with_daily=False, logger=logger)
    report_ds = build_dataset(
        cfg, "validation", source_stations, with_daily=False,
        max_batches_per_station=int(cfg.get_path("eval.max_batches_per_station", 0)), logger=logger,
    )

    scalers = load_scalers(cfg.data.root)
    ds_config = load_dataset_config(cfg.data.root)
    static_keep, onehot_specs, static_names = resolve_static_spec(
        cfg.data.root, cfg.data.get("static_exclude"), cfg.data.get("onehot_static")
    )
    dyn_size = len(ds_config["dyn_features"])
    logger.info("inputs: %d dynamic (%s) + %d static", dyn_size, ",".join(ds_config["dyn_features"]), len(static_names))

    num_workers = int(cfg.train.num_workers)
    pin_memory = bool(cfg.train.pin_memory) and device.type == "cuda"
    rng = np.random.default_rng(int(cfg.train.seed) + args.fold)

    # Fixed, station-balanced early-stopping set: the same batches every epoch,
    # spread over many stations, so the median KGE is comparable epoch to epoch.
    val_subset = pick_per_station(
        val_ds.prefixes,
        per_station=int(cfg.train.val_batches_per_station),
        seed=0,
        max_stations=int(cfg.train.val_max_stations),
    )
    val_loader = make_loader(val_ds, num_workers=num_workers, pin_memory=pin_memory, subset=val_subset)
    # Report the station count actually covered, not the config value: 0 means
    # "all of them", and echoing the 0 read as if nothing were selected.
    n_val_stations = len({station_of(val_ds.prefixes[i]) for i in val_subset} - {""})
    logger.info(
        "early stopping on %d source validation batches (~%d samples) over %d stations",
        len(val_subset), len(val_subset) * 512, n_val_stations,
    )

    # --- model / optim ---------------------------------------------------
    model = build_model(cfg, dyn_input_size=dyn_size, static_input_size=len(static_names)).to(device)
    logger.info("model params: %d", sum(p.numel() for p in model.parameters()))

    if cfg.train.loss == "nse_basin":
        criterion = MTSBasinNSELoss(
            frequency_factor=int(cfg.model.frequency_factor),
            reg_lambda=float(cfg.train.reg_lambda),
        )
    elif cfg.train.loss == "mse":
        class _MSE(torch.nn.Module):
            def forward(self, outputs, y, stn_std):
                loss = torch.nn.functional.mse_loss(outputs["H"].reshape(-1), y.reshape(-1))
                return {"loss": loss, "loss_hourly": loss.detach()}

        criterion = _MSE()
    else:
        raise ValueError(f"unknown loss {cfg.train.loss!r}")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))
    lr_schedule = parse_lr_schedule(cfg.train.get("lr_schedule"))
    stopper = EarlyStopping(patience=int(cfg.train.patience), mode="max")

    run = init_wandb(
        cfg,
        run_name=f"pretrain_fold{args.fold}",
        extra={"fold": args.fold, "step": "phase1_step1", "n_source_stations": len(source_stations)},
    )

    ckpt_path = out_dir / "checkpoint.pth"
    best_path = out_dir / "best_model.pth"
    start_epoch = 1
    history: list[dict] = []
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        stopper.best = ckpt["best"]
        stopper.best_epoch = ckpt["best_epoch"]
        stopper.counter = ckpt["counter"]
        history = ckpt.get("history", [])
        start_epoch = int(ckpt["epoch"]) + 1
        logger.info("resumed from %s at epoch %d", ckpt_path, start_epoch)

    # --- loop -------------------------------------------------------------
    for epoch in range(start_epoch, int(cfg.train.epochs) + 1):
        apply_lr_schedule(optimizer, lr_schedule, epoch)
        subset = epoch_subset(len(train_ds), int(cfg.train.batches_per_epoch), rng)
        train_loader = make_loader(train_ds, num_workers=num_workers, pin_memory=pin_memory, subset=subset)

        t0 = time.time()
        losses, n_rows = train_one_epoch(
            model, train_loader, optimizer, criterion, device, float(cfg.train.grad_clip), logger
        )
        train_secs = time.time() - t0

        results = evaluate_model(
            model, val_loader, device, scalers["y_mean"], scalers["y_std"],
            min_samples=int(cfg.get_path("eval.min_samples_per_station", 1)),
        )
        val_summary = summarize(results["hourly"], "source_val")

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_samples": n_rows,
            "train_secs": round(train_secs, 1),
            **{f"train/{k}": v for k, v in losses.items()},
            "val/median_kge": val_summary["median_kge"],
            "val/median_nse": val_summary["median_nse"],
            "val/n_valid_stations": val_summary["n_valid_stations"],
        }
        history.append(row)
        logger.info(
            "epoch %d/%d | loss %.5f | val median KGE %.4f NSE %.4f | %d samples in %.0fs",
            epoch, int(cfg.train.epochs), losses["loss"],
            val_summary["median_kge"], val_summary["median_nse"], n_rows, train_secs,
        )
        wandb_log(run, row, step=epoch)

        improved = stopper.step(val_summary["median_kge"], epoch)
        if improved:
            atomic_save(model.state_dict(), best_path)
            logger.info("  new best (median KGE %.4f) -> %s", stopper.best, best_path)
        atomic_save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best": stopper.best,
                "best_epoch": stopper.best_epoch,
                "counter": stopper.counter,
                "history": history,
            },
            ckpt_path,
        )
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

        if bool(cfg.train.early_stopping) and stopper.should_stop:
            logger.info("early stopping at epoch %d (best epoch %d)", epoch, stopper.best_epoch)
            break

    # --- final source-domain metrics with the best weights ---------------
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        logger.info("reloaded best weights from epoch %d", stopper.best_epoch)

    # Report over the FULL source domain (capped per station), not just the
    # early-stopping subsample.
    report_loader = make_loader(report_ds, num_workers=num_workers, pin_memory=pin_memory)
    results = evaluate_model(
        model, report_loader, device, scalers["y_mean"], scalers["y_std"],
        min_samples=int(cfg.get_path("eval.min_samples_per_station", 1)), logger=logger,
    )
    summaries = write_results(results, out_dir, tag=f"fold{args.fold}_source_val")
    logger.info("SOURCE val | median KGE %.4f | median NSE %.4f",
                summaries["hourly"]["median_kge"], summaries["hourly"]["median_nse"])

    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "fold": args.fold,
                "best_epoch": stopper.best_epoch,
                "best_val_median_kge": stopper.best,
                "n_source_stations": len(source_stations),
                "n_target_stations": len(target_stations),
                "static_features": static_names,
                "config": dict(cfg),
            },
            handle,
            indent=2,
            default=str,
        )
    wandb_log(run, {"final/source_val_median_kge": summaries["hourly"]["median_kge"],
                    "final/source_val_median_nse": summaries["hourly"]["median_nse"]})
    wandb_finish(run)
    (out_dir / "DONE").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
