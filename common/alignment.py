"""Which input-window layout a checkpoint was trained with, and a refusal when it differs.

Two layouts exist (see data.hourly_windows.window_bounds). ``"lagged"`` is every run
recorded before 2026-10-18; ``"figure"`` is the design as drawn in the architecture figure
(scripts/figures/fig_architecture.py). A model trained on one layout and evaluated on the other does not
fail: it predicts, it scores, and the score is simply wrong. So every place that loads a
checkpoint and builds inputs from a config calls ``assert_same_alignment``.

The layout is read from the run's own record, never inferred from a path name:
``run_meta.json`` (pretraining, transfer) or ``alignment.json`` (Africa transfer), in the
checkpoint's directory or up to two levels above it. A checkpoint with no record at all
predates the key and is therefore ``"lagged"``.
"""

from __future__ import annotations

import json
from pathlib import Path

KEY = "window_alignment"
DEFAULT = "lagged"


def alignment_of_cfg(cfg) -> str:
    return str(cfg.data.get(KEY, DEFAULT))


def _from_record(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if KEY in payload:
        return str(payload[KEY])
    data = (payload.get("config") or {}).get("data") or {}
    if isinstance(data, dict):
        return str(data.get(KEY, DEFAULT))
    return None


def alignment_of_checkpoint(checkpoint) -> str:
    here = Path(checkpoint).resolve().parent
    for directory in (here, here.parent, here.parent.parent):
        for name in ("alignment.json", "run_meta.json", "pretrain/run_meta.json"):
            found = _from_record(directory / name)
            if found is not None:
                return found
    return DEFAULT


def assert_same_alignment(checkpoint, cfg, logger=None) -> str:
    want = alignment_of_cfg(cfg)
    have = alignment_of_checkpoint(checkpoint)
    if have != want:
        raise SystemExit(
            f"window layout mismatch: {checkpoint} was trained with {KEY}={have!r} but the "
            f"config builds inputs with {want!r}. Evaluating it this way would score the "
            "model on inputs shifted from the ones it learned.")
    if logger:
        logger.info("window layout: %s (checkpoint and config agree)", want)
    return want


def write_alignment(out_dir, cfg) -> None:
    """For runs that write no run_meta.json (the Africa transfer)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "alignment.json").write_text(
        json.dumps({KEY: alignment_of_cfg(cfg)}, indent=2))
