#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10-00:00:00
#SBATCH --array=1-1%1
#SBATCH -J BaselineLSTM_grid
#SBATCH -o logs/array_%A_%a.out
#SBATCH -e logs/array_%A_%a.err
#SBATCH --signal=B:SIGUSR1@120

set -eo pipefail

# HydroDeepAI-style logic:
# 1) Generate config_whole.txt (tab-separated) from sweep.yaml
# 2) Each SLURM array task reads its own hyperparameters (by ArrayTaskID)
# 3) Launch one training run for that combination (resume + DONE-skip)

source ~/.bashrc
conda activate /ibex/user/kongw0a/conda-environments/hydroenv

wandb online || true

cd /home/kongw0a/MTS_LSTM/experiment_withcursor/BaselineLSTM

if [ ! -s config_whole.txt ]; then
  python make_config_whole.py
fi

config=/home/kongw0a/MTS_LSTM/experiment_withcursor/BaselineLSTM/config_whole.txt

lr=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
dropout=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
hidden_size=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
batch_size=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)
lookback=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)
epochs=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $7}' $config)

python arrayrun_train.py "$SLURM_JOB_ID" "$SLURM_MEM_PER_NODE" "$SLURM_NTASKS" "$SLURM_ARRAY_TASK_ID" "$lr" "$dropout" "$hidden_size" "$batch_size" "$lookback" "$epochs"