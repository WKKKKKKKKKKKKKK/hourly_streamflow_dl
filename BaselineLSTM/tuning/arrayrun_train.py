#!/usr/bin/env python3
"""HydroDeepAI-style SLURM array wrapper for BaselineLSTM.

Called by runme_array.sh with positional arguments extracted from config_whole.txt.

It builds a deterministic output directory for this hyperparameter combo, enables
resume/checkpointing, and skips finished runs (DONE marker).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path


def _fmt(v: str) -> str:
    # Shorten floats like 0.00050 -> 0.0005, keep scientific notation if present.
    try:
        if re.fullmatch(r"[+-]?(\d+\.?\d*|\d*\.?\d+)([eE][+-]?\d+)?", v.strip()):
            f = float(v)
            return format(f, '.10g')
    except Exception:
        pass
    return v.strip()


def _sanitize(s: str) -> str:
    s = str(s)
    s = s.replace(' ', '')
    s = re.sub(r"[^A-Za-z0-9._+-]+", '-', s)
    s = re.sub(r"-+", '-', s).strip('-')
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('job_id')
    ap.add_argument('mem_per_node')
    ap.add_argument('ntasks')
    ap.add_argument('array_task_id', type=int)
    ap.add_argument('lr')
    ap.add_argument('dropout')
    ap.add_argument('hidden_size')
    ap.add_argument('batch_size')
    ap.add_argument('lookback')
    ap.add_argument('epochs')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    lr = _fmt(args.lr)
    dropout = _fmt(args.dropout)
    hidden_size = str(int(float(args.hidden_size)))
    batch_size = str(int(float(args.batch_size)))
    lookback = str(int(float(args.lookback)))
    epochs = str(int(float(args.epochs)))

    run_group = os.environ.get('RUN_GROUP', 'grid')
    base = Path('runs') / run_group / 'BaselineLSTM'

    tag = _sanitize(f"idx{args.array_task_id:04d}_lb{lookback}_bs{batch_size}_hs{hidden_size}_do{dropout}_lr{lr}_ep{epochs}")
    out_dir = base / tag

    done_path = out_dir / 'DONE'
    if done_path.exists() and os.environ.get('FORCE_RERUN', '0') != '1':
        print(f"DONE exists, skipping: {done_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_project = os.environ.get('WANDB_PROJECT', 'LSTM_Streamflow')
    wandb_mode = os.environ.get('WANDB_MODE', 'online')
    wandb_run_name = os.environ.get('WANDB_RUN_NAME', f"BaselineLSTM_{tag}")

    cmd = [
        'python', 'Train.py',
        '--output-dir', str(out_dir),
        '--lr', lr,
        '--dropout', dropout,
        '--hidden-size', hidden_size,
        '--batch-size', batch_size,
        '--lookback', lookback,
        '--epochs', epochs,
        '--loss', 'nse_loss',
        '--early-stopping',
        '--patience', os.environ.get('PATIENCE', '10'),
        '--resume',
        '--wandb',
        '--wandb-project', wandb_project,
        '--wandb-run-name', wandb_run_name,
        '--wandb-mode', wandb_mode,
    ]

    print('Running:', ' '.join(cmd))
    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
