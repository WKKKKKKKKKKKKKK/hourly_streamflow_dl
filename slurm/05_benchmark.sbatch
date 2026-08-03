#!/bin/bash
# One-off: measure the sizing knobs before committing to the 5-fold array.
# Answers three things the config currently only guesses: does hidden=128 /
# batch=512 / seq=1000 fit in GPU memory, is the step IO- or GPU-bound, and how
# long is an epoch (hence epochs, batches_per_epoch and --time for the real run).
#   sbatch slurm/05_benchmark.sbatch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint=v100
#SBATCH -J gmts_bench
#SBATCH -o logs/benchmark_%j.out
#SBATCH -e logs/benchmark_%j.err

set -eo pipefail
# SLURM copies this script to a node-local spool directory, so BASH_SOURCE does
# NOT point at the repo. Use the submit directory, or GMTS_REPO if you submit from
# elsewhere, and verify it really is the repo before touching anything.
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$REPO_ROOT/configs/phase1.yaml" ]]; then
  echo "REPO_ROOT=$REPO_ROOT is not the global_mtslstm repo (no configs/phase1.yaml)." >&2
  echo "Submit from the repo root, or export GMTS_REPO=/path/to/global_mtslstm." >&2
  exit 2
fi
cd "$REPO_ROOT"
mkdir -p logs

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
echo "node=$(hostname) python=$PYTHON_BIN"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=================== 1/2  benchmark ==================="
"$PYTHON_BIN" -u -m scripts.benchmark_gpu --fold 0 --batches 80 --gpu-steps 25

echo "=================== 2/2  two real epochs ==================="
"$PYTHON_BIN" -u -m train.pretrain_source --fold 0 \
  --out-dir outputs/_gpucheck/fold0/pretrain \
  --set train.epochs=2 train.batches_per_epoch=200 \
        train.val_max_stations=150 train.val_batches_per_station=1 \
        eval.max_batches_per_station=1 wandb.mode=disabled
