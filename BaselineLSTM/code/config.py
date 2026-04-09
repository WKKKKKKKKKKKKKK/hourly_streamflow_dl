"""
====================================================
Configuration File for LSTM Streamflow Experiments
====================================================
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


# ====================================================
# 1. Data Paths
# ====================================================
DATA_PATH = str((BASE_DIR.parent / "data" / "selected_stn_data.nc").resolve())
STATIC_PATH = str((BASE_DIR.parent / "data" / "static_h.csv").resolve())
#MODEL_SAVE_PATH = "/home/kongw0a/MTS_LSTM/experiment/save_model/BaselineLSTM/"
MODEL_SAVE_PATH = str((BASE_DIR / "hist_model" / "best_model.pth").resolve())

# ====================================================
# 2. Time Splits
# ====================================================
TRAIN_START = "1990-10-01"
TRAIN_END   = "2003-09-30"

VAL_START   = "2003-10-01"
VAL_END     = "2008-09-30"

TEST_START  = "2008-10-01"
TEST_END    = "2015-09-30"


# ====================================================
# 3. Variables
# ====================================================
DYNAMIC_VARS = ['Rainf', 'Tair', 'PotEvap']
TARGET_VAR   = 'Streamflow'


# ====================================================
# 4. Data Settings
# ====================================================
LOOKBACK     = 365 * 24   # 1 year (hourly data)
BATCH_SIZE   = 256
NUM_WORKERS  = 4
PIN_MEMORY   = True


# ====================================================
# 5. Model Hyperparameters
# ====================================================
HIDDEN_SIZE  = 256
NUM_LAYERS   = 1
DROPOUT      = 0.4
OUTPUT_SIZE  = 1
INPUT_SIZE = 3+27
DYN_INPUT_SIZE = 3
STATIC_INPUT_SIZE = 27


# ====================================================
# 6. Training Settings
# ====================================================
LEARNING_RATE = 1e-4
NUM_EPOCHS    = 55
EARLY_STOPPING_PATIENCE = 10
USE_EARLY_STOPPING = True


# ====================================================
# 7. Reproducibility
# ====================================================
SEED = 42


# ====================================================
# 8. WandB Settings
# ====================================================
PROJECT_NAME = "LSTM_Streamflow"
RUN_NAME     = "experiment"


# ====================================================
# 9. Saving & Logging
# ====================================================
CHECKPOINT_DIR   = str((BASE_DIR / "checkpoints").resolve()) + "/"
BEST_MODEL_PATH  = str((BASE_DIR / "checkpoints" / "best_model.pth").resolve())
LOG_DIR          = str((BASE_DIR / "logs").resolve()) + "/"
