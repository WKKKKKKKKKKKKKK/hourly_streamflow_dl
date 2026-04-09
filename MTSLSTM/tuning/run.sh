#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH -J mtslstm_train
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --mem=40G

# 激活环境
source ~/.bashrc
conda activate /ibex/user/kongw0a/conda-environments/hydroenv

# WandB上线
wandb online

# Ensure we run from the submission directory (SLURM copies scripts to /var/spool)
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# ====== 进入项目目录 ======

# ====== 运行 ======
# Optional: set TRAIN_ARGS to pass CLI hyperparameters (e.g., --lr 1e-4 --output-dir runs/$SLURM_JOB_ID)
python Train.py ${TRAIN_ARGS}