#!/bin/bash
# One-off: list the 7.1M batch files and cache the index (~10-20 min on Lustre).
#   sbatch slurm/00_build_index.sbatch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -J gmts_index
#SBATCH -o logs/index_%j.out
#SBATCH -e logs/index_%j.err

set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

source "${CONDA_ENV_SETUP:-/dev/null}" 2>/dev/null || true
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
echo "python: $PYTHON_BIN"

"$PYTHON_BIN" -u -m scripts.build_index
"$PYTHON_BIN" -u -m scripts.make_folds --require-both-splits
"$PYTHON_BIN" -u -m scripts.smoke_test
