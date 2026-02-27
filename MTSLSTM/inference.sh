#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=inf_%j.out
#SBATCH --error=inf_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate /ibex/user/kongw0a/conda-environments/hydroenv
python inference.py