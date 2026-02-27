import os
import random
import time
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from loder import (
    handle_extremes,
    standardize_data,
    calculate_scalers,
    LSTMDataset
)

from Modelzoo import LSTM
from trainer import train_model
import config


# =====================================================
# 🔒 0️⃣ Reproducibility 
# =====================================================

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =====================================================
# 1️⃣ Device
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =====================================================
# 2️⃣ Load Data
# =====================================================

selected_stn_data = xr.open_dataset(config.DATA_PATH)
dyn = handle_extremes(selected_stn_data)

static_df = pd.read_csv(config.STATIC_PATH, index_col=0)
static_df.index = static_df.index.astype(str).str.zfill(8)


# =====================================================
# 3️⃣ Time Split
# =====================================================

dyn_forcing = dyn.sel(dynamic_forcing=config.DYNAMIC_VARS)
target_var_da = dyn.sel(dynamic_forcing=config.TARGET_VAR)

train_dyn = dyn_forcing.sel(time=slice(config.TRAIN_START, config.TRAIN_END))
train_target = target_var_da.sel(time=slice(config.TRAIN_START, config.TRAIN_END))

val_dyn = dyn_forcing.sel(time=slice(config.VAL_START, config.VAL_END))
val_target = target_var_da.sel(time=slice(config.VAL_START, config.VAL_END))


# =====================================================
# 4️⃣ Standardization
# =====================================================

train_scalers = calculate_scalers(train_dyn, static_df, train_target)

train_dyn_std, train_static_std, train_y_std = standardize_data(
    train_dyn, static_df, train_target, train_scalers
)

val_dyn_std, val_static_std, val_y_std = standardize_data(
    val_dyn, static_df, val_target, train_scalers
)


# =====================================================
# 5️⃣ Dataset & DataLoader
# =====================================================

train_dataset = LSTMDataset(
    dataset=train_dyn_std,
    target=train_y_std,
    static_df=train_static_std,
    lookback=config.LOOKBACK,
    start_date=config.TRAIN_START,
    end_date=config.TRAIN_END
)

validation_dataset = LSTMDataset(
    dataset=val_dyn_std,
    target=val_y_std,
    static_df=val_static_std,
    lookback=config.LOOKBACK,
    start_date=config.VAL_START,
    end_date=config.VAL_END
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY
)


# =====================================================
# 6️⃣ Model
# =====================================================

dynamic_size = train_dyn_std[list(train_dyn_std.data_vars)[0]].shape[1]
static_size = train_static_std.shape[1]
input_size = dynamic_size + static_size

model = LSTM(
    input_size=input_size,
    hidden_size=config.HIDDEN_SIZE,
    num_layers=config.NUM_LAYERS,
    dropout=config.DROPOUT,
    output_size=config.OUTPUT_SIZE
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


# =====================================================
# 7️⃣ WandB
# =====================================================

wandb.init(
    project=config.PROJECT_NAME,
    name=f"{config.RUN_NAME}_{int(time.time())}",
    config={
        "data_path": config.DATA_PATH,
        "static_path": config.STATIC_PATH,
        "train_start": config.TRAIN_START,
        "train_end": config.TRAIN_END,
        "val_start": config.VAL_START,
        "val_end": config.VAL_END,
        "lookback": config.LOOKBACK,
        "batch_size": config.BATCH_SIZE,
        "hidden_size": config.HIDDEN_SIZE,
        "num_layers": config.NUM_LAYERS,
        "dropout": config.DROPOUT,
        "learning_rate": config.LEARNING_RATE,
        "num_epochs": config.NUM_EPOCHS,
        "seed": config.SEED
    }
)

wandb.watch(model, log="gradients", log_freq=200)


# =====================================================
# 8️⃣ Train
# =====================================================

history = train_model(
    model=model,
    train_loader=train_loader,
    validation_loader=validation_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    num_epochs=config.NUM_EPOCHS
)


# =====================================================
# 9️⃣ Save Model
# =====================================================

os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
print("Model saved to", config.MODEL_SAVE_PATH)

wandb.finish()