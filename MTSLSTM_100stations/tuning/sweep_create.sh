#!/bin/bash
set -eo pipefail

# Creates a sweep on this directory's sweep.yaml.
# Writes the full sweep id to sweep_id.txt for easy copy/paste.

export BASHRCSOURCED="${BASHRCSOURCED:-1}"
source ~/.bashrc
conda activate /ibex/user/kongw0a/conda-environments/hydroenv
cd "/home/kongw0a/MTS_LSTM/experiment_withcursor/MTSLSTM"

mkdir -p logs

out=$(wandb sweep sweep.yaml 2>&1 | tee logs/sweep_create_$(date +%Y%m%d_%H%M%S).log)

# Try to extract the full sweep id: <entity>/<project>/<sweep_id>
SWEEP_ID=$(echo "$out" | grep -Eo "[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9]+" | tail -n 1 || true)

if [[ -z "${SWEEP_ID}" ]]; then
  echo "WARNING: Could not auto-detect sweep id. Please copy it from the output above."
  exit 0
fi

echo "${SWEEP_ID}" > sweep_id.txt
echo "Saved sweep id to $(pwd)/sweep_id.txt: ${SWEEP_ID}"
