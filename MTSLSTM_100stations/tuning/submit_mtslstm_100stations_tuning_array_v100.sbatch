#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --constraint=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --array=1-12%3
#SBATCH -J mts100_tune
#SBATCH -o /home/kongw0a/MTS_LSTM/experiment_withcursor/MTSLSTM/logs/mts100_tune_%A_%a.out
#SBATCH -e /home/kongw0a/MTS_LSTM/experiment_withcursor/MTSLSTM/logs/mts100_tune_%A_%a.err
#SBATCH --chdir=/home/kongw0a/MTS_LSTM/experiment_withcursor/MTSLSTM

set -eo pipefail

HYDROENV=/ibex/user/kongw0a/conda-environments/hydroenv/bin
"$HYDROENV/wandb" online || true

export WANDB_DIR=/ibex/project/c2266/wkkong/data/CAEMLSH/data_workstation/CAMELSH/training_runs/wandb
mkdir -p "$WANDB_DIR"
BASE_OUT=/ibex/project/c2266/wkkong/data/CAEMLSH/data_workstation/CAMELSH/training_runs/20260407_mtslstm_100stations_tuning_topo18_v100
mkdir -p "$BASE_OUT"

CONFIG=/home/kongw0a/MTS_LSTM/experiment_withcursor/MTSLSTM/tuning_500stations_prevbest.tsv
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is required" >&2
  exit 2
fi

line=$(awk -F '\t' -v idx="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==idx {print $0}' "$CONFIG")
if [[ -z "$line" ]]; then
  echo "No config for idx=$SLURM_ARRAY_TASK_ID" >&2
  exit 3
fi

IFS=$'\t' read -r idx dropout hidden_size batch_size lookback_hourly lookback_daily lr_schedule epochs patience <<< "$line"
run_tag="idx${idx}_bs${batch_size}_do${dropout}_hs${hidden_size}_H${lookback_hourly}_D${lookback_daily}"
out_dir="$BASE_OUT/$run_tag"
mkdir -p "$out_dir"

"$HYDROENV/python" -u Train.py \
  --data-path '/ibex/project/c2266/wkkong/data/CAEMLSH/data_workstation/CAMELSH/training_subsets/selected_stn_data_100stations_west10_east90_proposal_boxes_part*.nc' \
  --static-path /home/kongw0a/MTS_LSTM/experiment_withcursor/data/static_h_topo_priority27.csv \
  --batch-size "$batch_size" \
  --num-workers 3 \
  --dropout "$dropout" \
  --hidden-size-daily "$hidden_size" \
  --hidden-size-hourly "$hidden_size" \
  --lookback-hourly "$lookback_hourly" \
  --lookback-daily "$lookback_daily" \
  --lr 5e-4 \
  --lr-schedule "$lr_schedule" \
  --reg-lambda 1.0 \
  --loss nse_loss \
  --num-layers 1 \
  --epochs "$epochs" \
  --early-stopping \
  --patience "$patience" \
  --seed 42 \
  --output-dir "$out_dir" \
  --no-resume \
  --wandb true \
  --wandb-project MTSLSTM_Streamflow \
  --wandb-run-name "mtslstm_100stations_tune_${run_tag}" \
  --wandb-mode online
