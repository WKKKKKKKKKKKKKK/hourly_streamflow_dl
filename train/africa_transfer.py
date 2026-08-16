"""Phase I's premise where it actually occurs: fine-tune on AFRICAN daily observations.

Everything reported for Africa so far applies a model fine-tuned on temperate target
stations to African basins. That is an extrapolation test, not a validation of the
method: it asks "does a model tuned elsewhere transfer?", when the question the project
exists to answer is "given a region with daily discharge and no hourly discharge, can
daily-only supervision recover hourly skill there?".

Africa is that region, unarranged. 294 basins with daily `q_mm`, no hourly record, none
of them anywhere in training. So run the actual Phase I protocol on them:

    M0   the pretrained source model, zero-shot on the African validation period
    M1   after fine-tuning on the African TRAINING period using daily observations only
         -- the same DailyAggregateTransferLoss and the same frozen lstm_hourly as the
         global transfer step

Each basin's own record is split 70/30 in time, matching the global convention; a
global date boundary would give water-rich and sparse basins different splits. Skill is
scored daily, because Africa has no hourly observations to score against -- which is
the entire point.

    python -m train.africa_transfer --config configs/phase1_runB.yaml --fold 0
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from common.config import add_common_args, load_config, resolve
from common.metrics import compute_kge, compute_nse, kge_components
from common.utils import EarlyStopping, atomic_save, get_device, set_seed, setup_logging
from data.africa import (
    AfricaWindowDataset,
    apply_onehot,
    build_static_matrix,
    load_hourly_forcing,
    load_observed_daily,
)
from data.dataset import load_dataset_config, load_scalers, make_loader, resolve_static_spec
from models.losses import DailyAggregateTransferLoss, daily_aggregate_prediction
from models.mtslstm import build_model, set_trainable

DAILY_WINDOW = 24
DEFAULT_FORCING = (
    "/ibex/user/kongw0a/era5_land_africa_forcing/"
    "era5_land_africa_hourly_forcing_penman.nc"
)


@torch.no_grad()
def score(model, loader, device, y_mean: float, y_std: float, min_days: int = 100):
    """Per-basin daily KGE/NSE in mm/d, plus the r/alpha/beta split."""
    model.eval()
    sim: dict[str, list] = {}
    obs: dict[str, list] = {}
    for batch in loader:
        x = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
        outputs = model({"D": x["D"], "H": x["H"]}, x["S"])
        pred = daily_aggregate_prediction(outputs, DAILY_WINDOW).float().cpu().numpy()
        # mm/h -> mm/d, the same conversion eval.africa makes.
        pred = (pred * y_std + y_mean) * DAILY_WINDOW
        truth = batch["y_daily_obs"].numpy()
        for station, s, o in zip(np.asarray(batch["stations"], dtype=object), pred, truth):
            sim.setdefault(str(station), []).append(s)
            obs.setdefault(str(station), []).append(o)

    rows = []
    for station in sorted(sim):
        s = np.asarray(sim[station], dtype=np.float64)
        o = np.asarray(obs[station], dtype=np.float64)
        keep = np.isfinite(s) & np.isfinite(o)
        if keep.sum() < min_days or np.nanstd(o[keep]) == 0:
            continue
        kge, r, alpha, beta = kge_components(o[keep], s[keep])
        rows.append({"station_id": station, "n_days": int(keep.sum()),
                     "kge": kge, "nse": compute_nse(o[keep], s[keep]),
                     "kge_r": r, "kge_alpha": alpha, "kge_beta": beta})
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame, tag: str) -> dict:
    valid = frame.loc[np.isfinite(frame["kge"])]
    return {"tag": tag, "n_basins": int(len(valid)),
            "median_kge": float(valid["kge"].median()),
            "median_nse": float(valid["nse"].median()),
            "median_r": float(valid["kge_r"].median()),
            "median_alpha": float(valid["kge_alpha"].median()),
            "median_beta": float(valid["kge_beta"].median()),
            "frac_kge_gt_0": float((valid["kge"] > 0).mean())}


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(
        description="Fine-tune on African daily observations (Phase I protocol, in situ)."))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--forcing", default=DEFAULT_FORCING)
    parser.add_argument("--basins", default="africa/africa_basins.gpkg")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.7)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    out_dir = Path(args.out_dir) if args.out_dir else resolve(cfg.output_root) / f"fold{args.fold}" / "africa_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "africa_transfer.log")
    set_seed(int(cfg.transfer.seed) + args.fold)
    device = get_device()

    pretrained = Path(args.pretrained) if args.pretrained else (
        resolve(cfg.output_root) / f"fold{args.fold}" / "pretrain" / "best_model.pth"
    )
    if not pretrained.exists():
        raise SystemExit(f"pretrained checkpoint not found: {pretrained}")
    logger.info("fold %d | device %s | starting from %s", args.fold, device, pretrained)

    scalers = load_scalers(cfg.data.root)
    y_mean, y_std = float(scalers["y_mean"]), float(scalers["y_std"])
    dyn_size = len(load_dataset_config(cfg.data.root)["dyn_features"])
    names_all = list(load_dataset_config(cfg.data.root)["static_features"])
    static_keep, onehot_specs, static_names = resolve_static_spec(
        cfg.data.root, cfg.data.get("static_exclude"), cfg.data.get("onehot_static")
    )

    basins = gpd.read_file(resolve(args.basins))
    station_ids = basins["station_id"].astype(str).tolist()
    forcing, forcing_times = load_hourly_forcing(args.forcing, station_ids, scalers, logger=logger)
    observed, observed_dates = load_observed_daily(station_ids, logger=logger)
    full, _ = build_static_matrix(station_ids, names_all, scalers, logger=logger)
    static = apply_onehot(full, static_keep, onehot_specs).astype(np.float32)

    common = dict(forcing=forcing, forcing_times=forcing_times, static=static,
                  observed=observed, observed_dates=observed_dates, station_ids=station_ids,
                  lookback_hourly=int(cfg.data.lookback_hourly),
                  lookback_daily=int(cfg.data.get("lookback_daily", 365)),
                  chunk_size=int(cfg.data.get("chunk_size", 512)),
                  train_frac=args.train_frac, scalers=scalers, logger=logger)
    train_ds = AfricaWindowDataset(split="training", with_daily=True, **common)
    valid_ds = AfricaWindowDataset(split="validation", **common)

    num_workers = int(cfg.get_path("transfer.num_workers", 4))
    pin = device.type == "cuda"
    train_loader = make_loader(train_ds, num_workers=num_workers, pin_memory=pin, shuffle=True)
    valid_loader = make_loader(valid_ds, num_workers=num_workers, pin_memory=pin)

    model = build_model(cfg, dyn_input_size=dyn_size, static_input_size=len(static_names)).to(device)
    model.load_state_dict(torch.load(pretrained, map_location=device, weights_only=True))

    # M0 first: the same checkpoint, zero-shot on the African validation period, so M1
    # is compared against the model it actually started from on the same basins/days.
    logger.info("scoring M0 (zero-shot) on the African validation period ...")
    m0 = score(model, valid_loader, device, y_mean, y_std)
    m0_summary = summarise(m0, "M0")
    logger.info("M0: median KGE %.4f | r %.3f alpha %.3f beta %.3f | %d basins",
                m0_summary["median_kge"], m0_summary["median_r"],
                m0_summary["median_alpha"], m0_summary["median_beta"], m0_summary["n_basins"])
    m0.to_csv(out_dir / "per_basin_M0.csv", index=False)

    n_trainable, n_frozen = set_trainable(model, list(cfg.transfer.freeze_modules or []))
    logger.info("frozen %s | %d trainable / %d frozen params",
                list(cfg.transfer.freeze_modules or []), n_trainable, n_frozen)
    criterion = DailyAggregateTransferLoss(
        daily_window=DAILY_WINDOW, agg_loss_weight=float(cfg.transfer.agg_loss_weight))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=float(cfg.transfer.lr),
                                 weight_decay=float(cfg.transfer.weight_decay))
    epochs = args.epochs if args.epochs is not None else int(cfg.transfer.epochs)
    stopper = EarlyStopping(patience=int(cfg.transfer.patience), mode="max")
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        totals, rows = {}, 0
        for batch in train_loader:
            y_daily = batch["y_daily"]
            finite = torch.isfinite(y_daily)
            if not bool(finite.any()):
                continue
            x = {k: v[finite].to(device, non_blocking=True) for k, v in batch["x"].items()}
            outputs = model({"D": x["D"], "H": x["H"]}, x["S"])
            parts = criterion(outputs, y_daily[finite].to(device),
                              batch["stn_std"][finite].to(device),
                              batch["daily_mask"][finite].to(device))
            optimizer.zero_grad(set_to_none=True)
            parts["loss"].backward()
            if float(cfg.transfer.grad_clip):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], float(cfg.transfer.grad_clip))
            optimizer.step()
            n = int(finite.sum())
            rows += n
            for k, v in parts.items():
                totals[k] = totals.get(k, 0.0) + float(v.item()) * n
        if rows == 0:
            raise SystemExit("no finite daily targets in the African training split")
        loss = totals["loss"] / rows
        if not np.isfinite(loss):
            raise SystemExit(f"training loss is {loss} at epoch {epoch} -- refusing to continue")

        # Selection uses DAILY skill on the African validation period, which is all the
        # premise allows: these basins have no hourly observations at all.
        current = score(model, valid_loader, device, y_mean, y_std)
        summary = summarise(current, f"epoch{epoch}")
        history.append({"epoch": epoch, "train_loss": loss, **summary})
        logger.info("epoch %d/%d | loss %.5f | val daily KGE %.4f | %d basins",
                    epoch, epochs, loss, summary["median_kge"], summary["n_basins"])
        if stopper.step(summary["median_kge"], epoch):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            atomic_save(best_state, out_dir / "best_africa_model.pth")
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        if stopper.should_stop:
            logger.info("early stopping at epoch %d (best %d)", epoch, stopper.best_epoch)
            break

    model.load_state_dict(best_state)
    m1 = score(model, valid_loader, device, y_mean, y_std)
    m1_summary = summarise(m1, "M1")
    m1.to_csv(out_dir / "per_basin_M1.csv", index=False)
    logger.info("M1: median KGE %.4f | r %.3f alpha %.3f beta %.3f",
                m1_summary["median_kge"], m1_summary["median_r"],
                m1_summary["median_alpha"], m1_summary["median_beta"])

    paired = m0.merge(m1, on="station_id", suffixes=("_M0", "_M1"))
    delta = paired["kge_M1"] - paired["kge_M0"]
    from scipy.stats import wilcoxon

    p = float(wilcoxon(paired["kge_M1"], paired["kge_M0"]).pvalue) if len(paired) > 10 else float("nan")
    logger.info("PAIRED over %d basins: median dKGE %+.4f | %.1f%% improved | p=%.2e",
                len(paired), float(delta.median()), 100 * float((delta > 0).mean()), p)
    paired.to_csv(out_dir / "paired_M0_M1.csv", index=False)

    (out_dir / "summary.json").write_text(json.dumps({
        "fold": args.fold, "pretrained": str(pretrained),
        "n_train_chunks": len(train_ds), "n_valid_chunks": len(valid_ds),
        "M0": m0_summary, "M1": m1_summary,
        "paired": {"n_basins": int(len(paired)), "median_delta_kge": float(delta.median()),
                   "frac_improved": float((delta > 0).mean()), "wilcoxon_p": p},
    }, indent=2))
    (out_dir / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
