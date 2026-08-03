"""End-to-end smoke test on a handful of stations and batches.

Runs the whole Phase I chain -- index lookup, dataset, model, both losses,
metrics -- on a tiny slice, so a broken assumption surfaces in a minute instead
of after a night on a GPU. Safe to run on a login node.

    python -m scripts.smoke_test --stations 40 --batches 6
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from common.config import add_common_args, load_config, resolve
from common.metrics import summarize
from common.utils import get_device, set_seed, setup_logging
from data.dataset import (
    PreparedBatchDataset,
    load_dataset_config,
    load_lookback_offsets,
    load_scalers,
    make_loader,
    resolve_static_spec,
)
from data.index import load_stations, select_prefixes
from eval.evaluate import evaluate_model
from models.losses import DailyAggregateTransferLoss, MTSBasinNSELoss
from models.mtslstm import build_model, set_trainable


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Phase I smoke test."))
    parser.add_argument("--stations", type=int, default=40)
    parser.add_argument("--batches", type=int, default=6)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    logger = setup_logging()
    set_seed(0)
    device = get_device()
    logger.info("device: %s", device)

    # --- 1. sanity-check the branch split against the recorded offsets ------
    offsets = load_lookback_offsets()
    k_h = int(cfg.data.lookback_hourly)
    logger.info("sequence length %d | hourly tail %d positions | lookback_hourly %d",
                offsets["seq_len"], offsets["hourly_tail_positions"], k_h)
    if k_h > offsets["hourly_tail_positions"]:
        logger.warning(
            "lookback_hourly=%d reaches past the 1-hour-spaced tail (%d); the hourly branch "
            "would mix spacings and H_seq[:, -24:] would no longer be exactly 24 hours.",
            k_h, offsets["hourly_tail_positions"],
        )
    hours_ago = np.asarray(offsets["hours_ago"])
    assert hours_ago[-1] == 0 and (np.diff(hours_ago[-24:]) == -1).all(), "hourly tail is not 1-hour spaced"
    logger.info("OK: the last 24 positions are consecutive hours ending at the target hour")

    # --- 2. dataset ---------------------------------------------------------
    stations_all = load_stations(resolve(cfg.data.index_dir))
    stations_all = stations_all.loc[stations_all["n_training_batches"] > 0]
    # Spread the picks across the table so every source is represented -- the
    # first N alphabetically are all BOMAustralia, whose hourly records are the
    # sparse ones, and that would hide problems with the dense sources.
    step = max(1, len(stations_all) // args.stations)
    stations = set(stations_all["station_id"].iloc[::step].head(args.stations))
    logger.info("using %d stations from %s", len(stations), sorted({s.split("__")[0] for s in stations}))

    regular, corrected = select_prefixes(
        resolve(cfg.data.index_dir), "training", stations, include_corrected=bool(cfg.data.include_corrected)
    )
    logger.info("matched %d regular + %d corrected batches", len(regular), len(corrected))
    prefixes = (regular[: args.batches] + corrected[:2]) or regular[: args.batches]

    static_keep, onehot_specs, static_names = resolve_static_spec(
        cfg.data.root, cfg.data.get("static_exclude"), cfg.data.get("onehot_static")
    )
    dataset = PreparedBatchDataset(
        root=cfg.data.root,
        split="training",
        prefixes=prefixes,
        lookback_hourly=k_h,
        static_keep=static_keep,
        allowed_stations=stations,
        with_daily=True,
        min_daily_hours=int(cfg.get_path("transfer.min_daily_hours", 18)),
        onehot_specs=onehot_specs,
    )
    item = dataset[0]
    occupancy = item["daily_mask"].sum(dim=1)
    logger.info(
        "item shapes: D %s H %s S %s | y %s | %d distinct stations",
        tuple(item["x"]["D"].shape), tuple(item["x"]["H"].shape), tuple(item["x"]["S"].shape),
        tuple(item["y"].shape), len(set(item["stations"])),
    )
    logger.info(
        "daily targets: %d/%d rows usable at min_daily_hours=%s | observed hours per window "
        "min %d median %d max %d",
        int(torch.isfinite(item["y_daily"]).sum()), len(item["y_daily"]),
        cfg.get_path("transfer.min_daily_hours", 18),
        int(occupancy.min()), int(occupancy.median()), int(occupancy.max()),
    )
    assert bool(item["daily_mask"][:, -1].all()), "the target hour itself must always be occupied"
    assert item["x"]["H"].shape[1] == k_h
    assert item["x"]["D"].shape[1] == offsets["seq_len"]
    assert item["x"]["S"].shape[1] == len(static_names)
    if onehot_specs:
        # Every row must land in exactly one category of each one-hot block.
        offset = len(static_keep)
        for spec in onehot_specs:
            width = len(spec.categories)
            block = item["x"]["S"][:, offset : offset + width]
            assert bool((block.sum(dim=1) == 1).all()), (
                f"{spec.name}: {int((block.sum(dim=1) != 1).sum())} rows fell outside "
                f"the declared categories {list(spec.categories)}"
            )
            offset += width
        logger.info("OK: one-hot blocks %s sum to 1 per row",
                    {s.name: len(s.categories) for s in onehot_specs})
    assert torch.allclose(item["x"]["H"], item["x"]["D"][:, -k_h:, :]), "H must be the tail of D"

    # --- 3. model + losses ---------------------------------------------------
    ds_config = load_dataset_config(cfg.data.root)
    scalers = load_scalers(cfg.data.root)
    model = build_model(cfg, dyn_input_size=len(ds_config["dyn_features"]), static_input_size=len(static_names)).to(device)
    logger.info("model params: %d", sum(p.numel() for p in model.parameters()))

    x = {key: value.to(device) for key, value in item["x"].items()}
    outputs = model({"D": x["D"], "H": x["H"]}, x["S"])
    logger.info("outputs: %s", {key: tuple(value.shape) for key, value in outputs.items()})
    assert outputs["H_seq"].shape[1] == k_h

    pretrain_loss = MTSBasinNSELoss(frequency_factor=int(cfg.model.frequency_factor),
                                    reg_lambda=float(cfg.train.reg_lambda))
    parts = pretrain_loss(outputs, item["y"].to(device), item["stn_std"].to(device))
    logger.info("step-1 loss: %s", {k: round(float(v), 5) for k, v in parts.items()})
    parts["loss"].backward()
    logger.info("OK: backward through the step-1 loss")

    finite = torch.isfinite(item["y_daily"])
    if bool(finite.any()):
        model.zero_grad(set_to_none=True)
        n_trainable, n_frozen = set_trainable(model, list(cfg.transfer.freeze_modules or []))
        logger.info("transfer freeze: %d trainable / %d frozen params", n_trainable, n_frozen)
        xf = {key: value[finite] for key, value in x.items()}
        out_f = model({"D": xf["D"], "H": xf["H"]}, xf["S"])
        transfer_loss = DailyAggregateTransferLoss(agg_loss_weight=float(cfg.transfer.agg_loss_weight))
        parts = transfer_loss(
            out_f,
            item["y_daily"][finite].to(device),
            item["stn_std"][finite].to(device),
            item["daily_mask"][finite].to(device),
        )
        logger.info("step-2 loss: %s", {k: round(float(v), 5) for k, v in parts.items()})
        parts["loss"].backward()
        frozen_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.lstm_hourly.parameters()
        )
        assert not frozen_grad, "lstm_hourly received gradients despite being frozen"
        logger.info("OK: backward through the step-2 loss, lstm_hourly stayed frozen")
    else:
        logger.warning("no complete 24 h window in this batch; skipped the step-2 loss check")

    # --- 4. metrics ------------------------------------------------------------
    loader = make_loader(dataset, num_workers=0)
    results = evaluate_model(model, loader, device, scalers["y_mean"], scalers["y_std"],
                             min_samples=1, daily_window=24)
    for scale, frame in results.items():
        logger.info("%s metrics: %s", scale, summarize(frame, scale))

    logger.info("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
