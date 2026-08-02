"""Phase I Steps 2 and 3 -- daily-only fine-tune on the 20% target domain.

The premise is that the target stations have no hourly observations, only 24 h
aggregates. So for one fold:

  * start from the fold's pretrained ``best_model.pth`` (M_src),
  * fine-tune on the target stations' ``training/`` batches with
    ``loss = NSE_d(D, y_d) + w_agg * NSE_d(mean_24h(H_seq), y_d)`` where
    ``y_d = mean(y_h[t-23..t])`` -- no hourly term anywhere,
  * freeze ``lstm_hourly`` so the source-learned sub-daily dynamics survive,
  * early-stop on DAILY KGE over a held-out slice of the target training period.

That last point is the fix flagged in PLAN.md 3.2 #4: selecting the epoch by
hourly KGE would use observations the premise says do not exist. For the
robustness check the script also records, each epoch, what the hourly test KGE
would have been -- reported but never used for selection.

Then it evaluates, all with the same checkpoint:

  * M0 = M_src on the target validation period (Step 1's "validate on 20%"),
  * M1 = M_transfer on the target validation period  (Step 2),
  * M_src and M_transfer on the source validation period (Step 3, degradation).

    python -m train.transfer_target --fold 0
"""

from __future__ import annotations

import argparse
import copy
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
    atomic_save,
    get_device,
    init_wandb,
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
    resolve_static_columns,
)
from data.folds import domain_stations, load_folds
from eval.evaluate import evaluate_model, write_results
from models.losses import DailyAggregateTransferLoss
from models.mtslstm import build_model, set_trainable

DAILY_WINDOW = 24


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip, logger, log_every=100):
    model.train()
    totals: dict[str, float] = {}
    n_rows = n_batches = n_skipped = 0
    started = time.time()

    for batch in loader:
        if not batch["stations"]:
            continue
        y_daily = batch["y_daily"]
        if y_daily is None:
            raise ValueError("transfer training needs a dataset built with with_daily=True")
        finite = torch.isfinite(y_daily)
        if not bool(finite.any()):
            # Every row in this batch sits inside a time gap wider than 24 h.
            n_skipped += 1
            continue

        x = {key: value[finite].to(device, non_blocking=True) for key, value in batch["x"].items()}
        y_daily = y_daily[finite].to(device, non_blocking=True)
        stn_std = batch["stn_std"][finite].to(device, non_blocking=True)
        daily_mask = batch["daily_mask"][finite].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model({"D": x["D"], "H": x["H"]}, x["S"])
        parts = criterion(outputs, y_daily, stn_std, daily_mask)
        parts["loss"].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()

        rows = y_daily.shape[0]
        n_rows += rows
        n_batches += 1
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * rows

        if logger and n_batches % log_every == 0:
            logger.info(
                "    batch %d | loss %.5f | %.2f batch/s",
                n_batches, totals["loss"] / max(n_rows, 1), n_batches / max(time.time() - started, 1e-9),
            )

    if logger and n_skipped:
        logger.info("    skipped %d batches with no complete 24 h window", n_skipped)
    return {key: value / max(n_rows, 1) for key, value in totals.items()}, n_rows


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Phase I Steps 2-3: daily-only target transfer."))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--pretrained", default=None, help="Defaults to outputs/fold{N}/pretrain/best_model.pth")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--peek-hourly",
        action="store_true",
        help="Log the target hourly test KGE each epoch (robustness check only; never used for selection).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    out_dir = Path(args.out_dir) if args.out_dir else resolve(cfg.output_root) / f"fold{args.fold}" / "transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "transfer.log")
    set_seed(int(cfg.transfer.seed) + args.fold)

    pretrained = Path(args.pretrained) if args.pretrained else (
        resolve(cfg.output_root) / f"fold{args.fold}" / "pretrain" / "best_model.pth"
    )
    if not pretrained.exists():
        raise FileNotFoundError(f"pretrained checkpoint not found: {pretrained}")

    device = get_device()
    logger.info("fold %d | device %s | pretrained %s", args.fold, device, pretrained)

    folds = load_folds(resolve(cfg.folds.file))
    source_stations, target_stations = domain_stations(folds, args.fold)
    logger.info("target domain %d stations | source domain %d stations", len(target_stations), len(source_stations))

    scalers = load_scalers(cfg.data.root)
    ds_config = load_dataset_config(cfg.data.root)
    static_keep, static_names = resolve_static_columns(cfg.data.root, cfg.data.get("static_exclude"))
    dyn_size = len(ds_config["dyn_features"])

    num_workers = int(cfg.transfer.num_workers)
    pin_memory = bool(cfg.get_path("train.pin_memory", False)) and device.type == "cuda"
    min_samples = int(cfg.get_path("eval.min_samples_per_station", 1))

    # --- data --------------------------------------------------------------
    # Fine-tuning and its early-stopping holdout both live in the target
    # TRAINING period; the target validation period is the untouched test set.
    target_train_ds = build_dataset(cfg, "training", target_stations, with_daily=True, logger=logger)
    target_test_ds = build_dataset(
        cfg, "validation", target_stations, with_daily=False,
        max_batches_per_station=int(cfg.get_path("eval.max_batches_per_station", 0)), logger=logger,
    )

    rng = np.random.default_rng(int(cfg.transfer.seed) + args.fold)
    n_train_batches = len(target_train_ds)
    # Station-balanced holdout: a fixed couple of batches from EVERY target
    # station, so the daily median KGE driving early stopping is stable across
    # epochs. (It shares the training period with the fit pool and its input
    # windows overlap, so it is a selection set, not a test set -- the target
    # validation period below is the untouched test.)
    holdout_idx = pick_per_station(
        target_train_ds.prefixes,
        per_station=int(cfg.transfer.holdout_batches_per_station),
        seed=0,
    )
    fit_idx = np.setdiff1d(np.arange(n_train_batches), holdout_idx)
    logger.info("target training pool: %d batches -> %d fit / %d daily-KGE holdout",
                n_train_batches, len(fit_idx), len(holdout_idx))

    holdout_loader = make_loader(target_train_ds, num_workers=num_workers, pin_memory=pin_memory, subset=holdout_idx)

    # --- model ---------------------------------------------------------------
    model = build_model(cfg, dyn_input_size=dyn_size, static_input_size=len(static_names)).to(device)
    model.load_state_dict(torch.load(pretrained, map_location=device, weights_only=True))
    source_state = copy.deepcopy(model.state_dict())

    n_trainable, n_frozen = set_trainable(model, list(cfg.transfer.freeze_modules or []))
    logger.info("frozen modules %s | %d trainable / %d frozen params",
                list(cfg.transfer.freeze_modules or []), n_trainable, n_frozen)

    criterion = DailyAggregateTransferLoss(
        daily_window=DAILY_WINDOW, agg_loss_weight=float(cfg.transfer.agg_loss_weight)
    )
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.transfer.lr),
        weight_decay=float(cfg.transfer.weight_decay),
    )
    stopper = EarlyStopping(patience=int(cfg.transfer.patience), mode="max")

    run = init_wandb(
        cfg,
        run_name=f"transfer_fold{args.fold}",
        extra={"fold": args.fold, "step": "phase1_step2", "n_target_stations": len(target_stations)},
    )

    # --- M0: the pretrained model on the target domain, zero-shot -----------
    logger.info("evaluating M0 (zero-shot M_src on the target validation period) ...")
    m0_loader = make_loader(target_test_ds, num_workers=num_workers, pin_memory=pin_memory)
    m0 = evaluate_model(model, m0_loader, device, scalers["y_mean"], scalers["y_std"], min_samples, logger=logger)
    m0_summary = write_results(m0, out_dir, tag=f"fold{args.fold}_M0_target_hourly")["hourly"]
    logger.info("M0 target hourly: median KGE %.4f | median NSE %.4f",
                m0_summary["median_kge"], m0_summary["median_nse"])

    # --- fine-tune ------------------------------------------------------------
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []
    peek_loader = None
    if args.peek_hourly:
        peek_idx = epoch_subset(len(target_test_ds), 600, np.random.default_rng(0))
        peek_loader = make_loader(target_test_ds, num_workers=num_workers, pin_memory=pin_memory, subset=peek_idx)

    for epoch in range(1, int(cfg.transfer.epochs) + 1):
        subset = fit_idx[rng.permutation(len(fit_idx))[: int(cfg.transfer.batches_per_epoch)]]
        fit_loader = make_loader(target_train_ds, num_workers=num_workers, pin_memory=pin_memory, subset=subset)

        t0 = time.time()
        losses, n_rows = train_one_epoch(
            model, fit_loader, optimizer, criterion, device, float(cfg.transfer.grad_clip), logger
        )
        train_secs = time.time() - t0

        holdout = evaluate_model(
            model, holdout_loader, device, scalers["y_mean"], scalers["y_std"],
            min_samples=min_samples, daily_window=DAILY_WINDOW,
        )
        daily_summary = summarize(holdout["daily"], "target_holdout_daily")

        row = {
            "epoch": epoch,
            "train_samples": n_rows,
            "train_secs": round(train_secs, 1),
            **{f"train/{k}": v for k, v in losses.items()},
            "holdout/daily_median_kge": daily_summary["median_kge"],
            "holdout/daily_median_nse": daily_summary["median_nse"],
        }
        if peek_loader is not None:
            peek = evaluate_model(model, peek_loader, device, scalers["y_mean"], scalers["y_std"], min_samples)
            peek_summary = summarize(peek["hourly"], "target_test_hourly_peek")
            # Recorded for the "does daily-based selection cost anything?" check
            # in PLAN.md 3.2 #4; deliberately not fed to the early stopper.
            row["peek/target_hourly_median_kge"] = peek_summary["median_kge"]

        history.append(row)
        logger.info(
            "epoch %d/%d | loss %.5f | holdout daily KGE %.4f%s | %d samples in %.0fs",
            epoch, int(cfg.transfer.epochs), losses["loss"], daily_summary["median_kge"],
            f" | (peek hourly KGE {row['peek/target_hourly_median_kge']:.4f})" if peek_loader is not None else "",
            n_rows, train_secs,
        )
        wandb_log(run, row, step=epoch)

        if stopper.step(daily_summary["median_kge"], epoch):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            atomic_save(best_state, out_dir / "best_transfer_model.pth")
            logger.info("  new best daily KGE %.4f", stopper.best)
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

        if stopper.should_stop:
            logger.info("early stopping at epoch %d (best epoch %d)", epoch, stopper.best_epoch)
            break

    model.load_state_dict(best_state)
    logger.info("restored best transfer weights (epoch %d, holdout daily KGE %.4f)",
                stopper.best_epoch, stopper.best)

    # --- M1: Step 2 result -------------------------------------------------
    logger.info("evaluating M1 (M_transfer on the target validation period) ...")
    m1 = evaluate_model(model, m0_loader, device, scalers["y_mean"], scalers["y_std"], min_samples, logger=logger)
    m1_summary = write_results(m1, out_dir, tag=f"fold{args.fold}_M1_target_hourly")["hourly"]
    logger.info("M1 target hourly: median KGE %.4f | median NSE %.4f",
                m1_summary["median_kge"], m1_summary["median_nse"])

    # --- Step 3: no degradation on the source domain -----------------------
    logger.info("evaluating Step 3 (source-domain degradation) ...")
    source_test_ds = build_dataset(
        cfg, "validation", source_stations, with_daily=False,
        max_batches_per_station=int(cfg.get_path("eval.max_batches_per_station", 0)), logger=logger,
    )
    source_loader = make_loader(source_test_ds, num_workers=num_workers, pin_memory=pin_memory)

    after = evaluate_model(model, source_loader, device, scalers["y_mean"], scalers["y_std"], min_samples, logger=logger)
    after_summary = write_results(after, out_dir, tag=f"fold{args.fold}_M1_source_hourly")["hourly"]

    model.load_state_dict(source_state)
    before = evaluate_model(model, source_loader, device, scalers["y_mean"], scalers["y_std"], min_samples, logger=logger)
    before_summary = write_results(before, out_dir, tag=f"fold{args.fold}_M0_source_hourly")["hourly"]

    # Paired per-station change is the number that matters, not the two medians.
    merged = before["hourly"][["station_id", "kge", "nse", "score_status"]].merge(
        after["hourly"][["station_id", "kge", "nse", "score_status"]],
        on="station_id", suffixes=("_before", "_after"),
    )
    both_ok = merged["score_status_before"].eq("ok") & merged["score_status_after"].eq("ok")
    merged["delta_kge"] = merged["kge_after"] - merged["kge_before"]
    merged["delta_nse"] = merged["nse_after"] - merged["nse_before"]
    merged.to_csv(out_dir / f"source_degradation_fold{args.fold}.csv", index=False)
    delta_kge = float(merged.loc[both_ok, "delta_kge"].median())
    logger.info(
        "STEP 3 source domain: median KGE %.4f -> %.4f (paired median delta %+.4f over %d stations)",
        before_summary["median_kge"], after_summary["median_kge"], delta_kge, int(both_ok.sum()),
    )

    summary = {
        "fold": args.fold,
        "best_epoch": stopper.best_epoch,
        "best_holdout_daily_kge": stopper.best,
        "n_target_stations": len(target_stations),
        "n_source_stations": len(source_stations),
        "step1_M0_target_hourly": m0_summary,
        "step2_M1_target_hourly": m1_summary,
        "step2_gain_median_kge": m1_summary["median_kge"] - m0_summary["median_kge"],
        "step3_source_before": before_summary,
        "step3_source_after": after_summary,
        "step3_paired_median_delta_kge": delta_kge,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    logger.info("STEP 2 gain on target hourly KGE: %+.4f (M0 %.4f -> M1 %.4f)",
                summary["step2_gain_median_kge"], m0_summary["median_kge"], m1_summary["median_kge"])
    wandb_log(run, {
        "final/M0_target_hourly_kge": m0_summary["median_kge"],
        "final/M1_target_hourly_kge": m1_summary["median_kge"],
        "final/target_kge_gain": summary["step2_gain_median_kge"],
        "final/source_paired_delta_kge": delta_kge,
    })
    wandb_finish(run)
    (out_dir / "DONE").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
