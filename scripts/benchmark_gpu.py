"""Measure what the sizing knobs in configs/phase1.yaml should actually be.

Three things are guesses until this runs:

  * does the model even fit? hidden 128, batch 512, sequence 1000, and a daily
    branch that keeps all 1000 outputs to form d_seq -- never tried on a GPU;
  * is the bottleneck the GPU or reading 6 MB batch files off Lustre? that decides
    whether raising ``batches_per_epoch`` is nearly free or the dominant cost;
  * how long does an epoch take, hence ``epochs``, ``batches_per_epoch`` and the
    SLURM ``--time`` for the real 5-fold array.

Measured separately, because the answer differs:
    IO only     -- iterate the loader, no model
    GPU only    -- forward+backward on synthetic tensors already on the device
    combined    -- the real training step
Then projected to epoch and 5-fold wall clock.

    python -m scripts.benchmark_gpu --batches 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common.config import add_common_args, load_config, resolve
from common.utils import get_device, set_seed, setup_logging
from data.dataset import (
    build_dataset,
    epoch_subset,
    load_dataset_config,
    make_loader,
    resolve_static_spec,
)
from data.folds import domain_stations, load_folds
from models.losses import MTSBasinNSELoss
from models.mtslstm import build_model


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_gpu(cfg, device, dyn_size, static_size, hidden, batch_rows, seq_len, steps, logger):
    """Forward+backward on synthetic tensors: pure GPU cost and peak memory."""
    cfg = dict(cfg)
    cfg["model"] = {**dict(cfg["model"]), "hidden_size_daily": hidden, "hidden_size_hourly": hidden}
    from common.config import Config

    model = build_model(Config(cfg), dyn_input_size=dyn_size, static_input_size=static_size).to(device)
    criterion = MTSBasinNSELoss(frequency_factor=1, reg_lambda=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    k_h = int(cfg["data"]["lookback_hourly"])

    x_d = torch.randn(batch_rows, seq_len, dyn_size, device=device)
    x_s = torch.randn(batch_rows, static_size, device=device)
    y = torch.randn(batch_rows, device=device)
    std = torch.rand(batch_rows, device=device) + 0.1

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 3):                      # first 3 are warm-up
        if i == 3:
            sync(device)
            started = time.time()
        optimizer.zero_grad(set_to_none=True)
        outputs = model({"D": x_d, "H": x_d[:, -k_h:, :]}, x_s)
        criterion(outputs, y, std)["loss"].backward()
        optimizer.step()
    sync(device)
    per_step = (time.time() - started) / steps
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else float("nan")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  hidden %3d | %6.3f s/batch | peak GPU mem %5.2f GiB | %d params",
                hidden, per_step, peak, n_params)
    del model, optimizer, x_d, x_s
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"hidden": hidden, "gpu_s_per_batch": per_step, "peak_gib": peak, "params": n_params}


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Benchmark the training step."))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batches", type=int, default=60, help="Batch files to read for the IO measurement.")
    parser.add_argument("--gpu-steps", type=int, default=25)
    parser.add_argument("--hidden-sweep", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--out", default="outputs/benchmark_gpu.json")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_path.with_suffix(".log"))
    set_seed(0)

    device = get_device()
    logger.info("device: %s", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s | %.0f GiB | %d SMs | torch %s / cuda %s",
                    props.name, props.total_memory / 1024**3, props.multi_processor_count,
                    torch.__version__, torch.version.cuda)
    else:
        logger.warning("no CUDA -- the numbers below say nothing about the real run")

    ds_config = load_dataset_config(cfg.data.root)
    dyn_size = len(ds_config["dyn_features"])
    _, _, static_names = resolve_static_spec(
        cfg.data.root, cfg.data.get("static_exclude"), cfg.data.get("onehot_static")
    )
    logger.info("inputs: %d dynamic + %d static", dyn_size, len(static_names))

    # ---- 1. GPU cost and memory, per hidden size ---------------------------
    logger.info("GPU forward+backward on synthetic tensors (batch 512, seq 1000):")
    gpu_results = []
    for hidden in args.hidden_sweep:
        try:
            gpu_results.append(
                bench_gpu(cfg, device, dyn_size, len(static_names), hidden, 512, 1000, args.gpu_steps, logger)
            )
        except torch.cuda.OutOfMemoryError:
            logger.error("  hidden %3d | OUT OF MEMORY -- this size does not fit", hidden)
            torch.cuda.empty_cache()
            gpu_results.append({"hidden": hidden, "gpu_s_per_batch": None, "oom": True})

    # ---- 2. IO cost: read real batch files, no model -----------------------
    folds = load_folds(resolve(cfg.folds.file))
    source_stations, _ = domain_stations(folds, args.fold)
    train_ds = build_dataset(cfg, "training", source_stations, with_daily=False, logger=logger)
    subset = epoch_subset(len(train_ds), args.batches, np.random.default_rng(0))
    loader = make_loader(
        train_ds,
        num_workers=int(cfg.train.num_workers),
        pin_memory=bool(cfg.train.pin_memory) and device.type == "cuda",
        subset=subset,
    )

    logger.info("reading %d real batch files with num_workers=%d ...", len(subset), int(cfg.train.num_workers))
    started = time.time()
    n_batches = n_rows = 0
    for batch in loader:
        n_rows += len(batch["stations"])
        n_batches += 1
    io_elapsed = time.time() - started
    io_per_batch = io_elapsed / max(n_batches, 1)
    logger.info("IO ONLY: %d batches in %.1f s -> %.3f s/batch (%.0f MiB/s effective)",
                n_batches, io_elapsed, io_per_batch, n_batches * 6.2 / io_elapsed)

    # ---- 3. combined: the real training step -------------------------------
    hidden = int(cfg.model.hidden_size_daily)
    model = build_model(cfg, dyn_input_size=dyn_size, static_input_size=len(static_names)).to(device)
    criterion = MTSBasinNSELoss(frequency_factor=int(cfg.model.frequency_factor),
                                reg_lambda=float(cfg.train.reg_lambda))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))
    loader = make_loader(
        train_ds,
        num_workers=int(cfg.train.num_workers),
        pin_memory=bool(cfg.train.pin_memory) and device.type == "cuda",
        subset=subset,
    )
    model.train()
    sync(device)
    started = time.time()
    done = 0
    for batch in loader:
        if not batch["stations"]:
            continue
        x = {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()}
        y = batch["y"].to(device, non_blocking=True)
        std = batch["stn_std"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        criterion(model({"D": x["D"], "H": x["H"]}, x["S"]), y, std)["loss"].backward()
        optimizer.step()
        done += 1
    sync(device)
    combined_per_batch = (time.time() - started) / max(done, 1)
    logger.info("COMBINED: %d real training steps -> %.3f s/batch", done, combined_per_batch)

    # ---- 4. what the config should say ------------------------------------
    gpu_at_config = next((r["gpu_s_per_batch"] for r in gpu_results if r["hidden"] == hidden), None)
    logger.info("")
    logger.info("===== breakdown at hidden=%d =====", hidden)
    logger.info("  IO only        %.3f s/batch", io_per_batch)
    if gpu_at_config:
        logger.info("  GPU only       %.3f s/batch", gpu_at_config)
        bound = "IO-bound" if io_per_batch > gpu_at_config else "GPU-bound"
        logger.info("  combined       %.3f s/batch  -> %s", combined_per_batch, bound)
        overlap = (io_per_batch + gpu_at_config - combined_per_batch) / max(combined_per_batch, 1e-9)
        logger.info("  prefetch hides %.0f%% of the smaller cost", 100 * max(0.0, overlap))

    bpe = int(cfg.train.batches_per_epoch)
    val_batches = int(cfg.train.val_max_stations) * int(cfg.train.val_batches_per_station)
    train_s = bpe * combined_per_batch
    val_s = val_batches * io_per_batch          # eval is forward-only, IO dominates
    epoch_s = train_s + val_s
    epochs = int(cfg.train.epochs)
    logger.info("")
    logger.info("===== projection with the current config =====")
    logger.info("  train %d batches      -> %5.1f min", bpe, train_s / 60)
    logger.info("  validate %d batches   -> %5.1f min", val_batches, val_s / 60)
    logger.info("  one epoch             -> %5.1f min", epoch_s / 60)
    logger.info("  %d epochs (one fold)  -> %5.1f h", epochs, epochs * epoch_s / 3600)
    logger.info("  sbatch --time is 1-00:00:00; fold needs %.1f h -> %s",
                epochs * epoch_s / 3600, "OK" if epochs * epoch_s / 3600 < 22 else "TOO SHORT, raise it")
    budget_h = 20.0
    suggest = int((budget_h * 3600 - val_s * epochs) / (epochs * combined_per_batch))
    logger.info("  to fit %d epochs inside %.0f h, batches_per_epoch <= %d",
                epochs, budget_h, max(0, suggest))

    payload = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_properties(0).name if device.type == "cuda" else None,
        "dyn_size": dyn_size, "static_size": len(static_names),
        "io_s_per_batch": io_per_batch,
        "combined_s_per_batch": combined_per_batch,
        "hidden_sweep": gpu_results,
        "config": {"batches_per_epoch": bpe, "val_batches": val_batches, "epochs": epochs,
                   "num_workers": int(cfg.train.num_workers), "hidden": hidden},
        "projection_hours_per_fold": epochs * epoch_s / 3600,
        "suggested_batches_per_epoch_for_20h": max(0, suggest),
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
