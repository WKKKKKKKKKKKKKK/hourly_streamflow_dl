"""Seeding, logging, W&B and checkpoint helpers shared by every entry point."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# `import wandb` costs ~4 minutes on a cold ibex home directory, so it is
# deferred to init_wandb() -- scripts that never log (build_index, make_folds,
# build_station_table, evaluate) should not pay for it.


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: str | os.PathLike | None = None) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("global_mtslstm")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_wandb(cfg, run_name: str, extra: dict[str, Any] | None = None):
    """Start a W&B run, or return None when disabled/unavailable.

    Falls back to offline mode instead of crashing the job if the API key is
    missing -- a compute node without credentials should still train.
    """
    wandb_cfg = cfg.get("wandb", {}) or {}
    mode = str(wandb_cfg.get("mode", "online")).lower()
    if mode == "disabled":
        logging.getLogger("global_mtslstm").info("W&B disabled.")
        return None

    try:
        import wandb
    except Exception as exc:  # pragma: no cover
        logging.getLogger("global_mtslstm").warning("wandb unavailable (%s); continuing without it.", exc)
        return None

    if mode == "online" and not (os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").exists()):
        logging.getLogger("global_mtslstm").warning(
            "No W&B credentials found (WANDB_API_KEY / ~/.netrc); falling back to offline mode. "
            "Run `wandb login` once on a login node, or `wandb sync` the offline run later."
        )
        mode = "offline"

    try:
        run = wandb.init(
            project=wandb_cfg.get("project", "global_mtslstm_phase1"),
            entity=wandb_cfg.get("entity") or None,
            group=wandb_cfg.get("group") or None,
            name=run_name,
            mode=mode,
            config={**dict(cfg), **(extra or {})},
        )
        return run
    except Exception as exc:  # pragma: no cover
        logging.getLogger("global_mtslstm").warning("wandb.init failed (%s); continuing without W&B.", exc)
        return None


def wandb_log(run, payload: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    try:
        run.log(payload, step=step)
    except Exception:  # pragma: no cover
        pass


def wandb_finish(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:  # pragma: no cover
        pass


def atomic_save(obj: Any, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def parse_lr_schedule(schedule: str | None) -> list[tuple[int, float]] | None:
    """'1:5e-4,10:1e-4' -> [(1, 5e-4), (10, 1e-4)]."""
    schedule = (schedule or "").strip().replace("−", "-").replace(" ", "")
    if not schedule:
        return None
    items: list[tuple[int, float]] = []
    for part in schedule.split(","):
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid lr schedule segment: {part!r}")
        epoch_s, lr_s = part.split(":", 1)
        items.append((int(epoch_s), float(lr_s)))
    items.sort(key=lambda pair: pair[0])
    return items


def apply_lr_schedule(optimizer: torch.optim.Optimizer, schedule, epoch: int) -> None:
    if not schedule:
        return
    current = None
    for start_epoch, lr in schedule:
        if epoch >= start_epoch:
            current = lr
        else:
            break
    if current is None:
        return
    for group in optimizer.param_groups:
        group["lr"] = current


class EarlyStopping:
    """Tracks a metric that should go up (mode='max') or down (mode='min')."""

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 0.0):
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best = -np.inf if mode == "max" else np.inf
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def step(self, value: float, epoch: int) -> bool:
        if not np.isfinite(value):
            self.counter += 1
            self.should_stop = self.counter >= self.patience
            return False
        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best = float(value)
            self.best_epoch = int(epoch)
            self.counter = 0
            return True
        self.counter += 1
        self.should_stop = self.counter >= self.patience
        return False
