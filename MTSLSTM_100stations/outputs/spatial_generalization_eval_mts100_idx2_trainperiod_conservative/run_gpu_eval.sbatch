#!/bin/bash
#SBATCH -J mts100_spatial_train
#SBATCH --chdir=/home/kongw0a/MTS_LSTM/experiment_withcursor
#SBATCH -o /home/kongw0a/MTS_LSTM/experiment_withcursor/MTSLSTM/logs/spatial_eval_mts100_trainperiod_%j.out
#SBATCH -e /home/kongw0a/MTS_LSTM/experiment_withcursor/MTSLSTM/logs/spatial_eval_mts100_trainperiod_%j.err
#SBATCH -t 12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -euo pipefail

source /ibex/user/kongw0a/miniforge/etc/profile.d/conda.sh
conda activate hydroenv

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

/ibex/user/kongw0a/conda-environments/hydroenv/bin/python -u /home/kongw0a/MTS_LSTM/experiment_withcursor/MTSLSTM/outputs/spatial_generalization_eval_mts100_idx2_trainperiod_conservative/evaluate_selected_stations_gpu.py
