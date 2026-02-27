#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH -J mtslstm_train
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --mem=40G

# 
source ~/.bashrc
conda activate /ibex/user/kongw0a/conda-environments/hydroenv

# W
wandb online

# 
cd ./hourly_streamflow_dl/MTSLSTM

# 
python Train.py