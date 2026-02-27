"""
====================================================
Configuration File for sMTSLSTM Streamflow Experiments
====================================================
"""

# ====================================================
# 1️⃣ Data Paths(change these to your local paths)
# ====================================================

DATA_PATH = "/home/kongw0a/MTS_LSTM/experiment/data/selected_stn_data.nc"
STATIC_PATH = "/home/kongw0a/MTS_LSTM/experiment/data/static_h.csv"
MODEL_SAVE_PATH = "/home/kongw0a/MTS_LSTM/experiment/save_model/MTSLSTM/best_model_mtslstm.pth"

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
USE_EARLY_STOPPING = False
CHECKPOINT_DIR = "checkpoints/"
BEST_MODEL_PATH = "checkpoints/best_model_mtslstm.pth"

LOG_DIR = "logs/"