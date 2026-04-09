"""
====================================================
Configuration File for sMTSLSTM Streamflow Experiments
====================================================
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


# ====================================================
# 1️⃣ Data Paths
# ====================================================

DATA_PATH = str((BASE_DIR.parent / "data" / "selected_stn_data.nc").resolve())
#DATA_PATH = "/ibex/project/c2266/wkkong/data/CAEMLSH/all_stations.nc"
STATIC_PATH = str((BASE_DIR.parent / "data" / "static_h.csv").resolve())
MODEL_SAVE_PATH = str((BASE_DIR / "checkpoints" / "best_model_mtslstm.pth").resolve())
#MODEL_SAVE_PATH = "/home/kongw0a/MTS_LSTM/experiment/save_model/MTSLSTM/best_model_mtslstm.pth"

# ====================================================
# 2️⃣ Time Splits
# ====================================================

TRAIN_START = "1990-10-01"
TRAIN_END   = "2003-09-30"

VAL_START   = "2003-10-01"
VAL_END     = "2008-09-30"

TEST_START  = "2008-10-01"
TEST_END    = "2015-09-30"

# ====================================================
# 3️⃣ Variables
# ====================================================

DYNAMIC_VARS = ['Rainf', 'Tair', 'PotEvap']
TARGET_VAR   = 'Streamflow'

# ====================================================
# 4️⃣ Data Settings
# ====================================================

BATCH_SIZE = 256

NUM_WORKERS = 4
PIN_MEMORY = True

# Multi-timescale window lengths
LOOKBACK_HOURLY = 72
LOOKBACK_DAILY = 365

# ====================================================
# 5️⃣ Model Hyperparameters
# ====================================================

DYN_INPUT_SIZE = 3
STATIC_INPUT_SIZE = 27

HIDDEN_SIZE_DAILY  = 64
HIDDEN_SIZE_HOURLY = 64

NUM_LAYERS = 1
DROPOUT = 0.4
FREQUENCY_FACTOR = 24

# ====================================================
# 6️⃣ Training Settings
# ====================================================

LEARNING_RATE = 5e-4
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# ====================================================
# 7️⃣ Reproducibility
# ====================================================

SEED = 42

# ====================================================
# 8️⃣ WandB Settings
# ====================================================

PROJECT_NAME = "MTSLSTM_Streamflow"
RUN_NAME = "sMTSLSTM_experiment"

# ====================================================
# 9️⃣ Saving & Logging
# ====================================================
USE_EARLY_STOPPING = True
CHECKPOINT_DIR = str((BASE_DIR / "checkpoints").resolve()) + "/"
BEST_MODEL_PATH = str((BASE_DIR / "checkpoints" / "best_model_mtslstm.pth").resolve())
LOG_DIR = str((BASE_DIR / "logs").resolve()) + "/"