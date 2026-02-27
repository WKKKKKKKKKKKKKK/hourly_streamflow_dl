# ===============================
# Train.py
# ===============================

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
    MultiscaleLSTMDataset
)

from Modelzoo import sMTSLSTM
from trainer import train_model
import config


# =====================================================
# 🔒 0️⃣ Reproducibility
# =====================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================
# 1️⃣ Dataloader 
# =====================================================
def get_dataloaders():

    selected_stn_data = xr.open_dataset(config.DATA_PATH)
    dyn = handle_extremes(selected_stn_data)

    static_df = pd.read_csv(config.STATIC_PATH, index_col=0)
    static_df.index = static_df.index.astype(str).str.zfill(8)

    dyn_forcing = dyn.sel(dynamic_forcing=config.DYNAMIC_VARS)
    target_var_da = dyn.sel(dynamic_forcing=config.TARGET_VAR)

    train_dyn = dyn_forcing.sel(time=slice(config.TRAIN_START, config.TRAIN_END))
    train_target = target_var_da.sel(time=slice(config.TRAIN_START, config.TRAIN_END))

    val_dyn = dyn_forcing.sel(time=slice(config.VAL_START, config.VAL_END))
    val_target = target_var_da.sel(time=slice(config.VAL_START, config.VAL_END))

    train_scalers = calculate_scalers(train_dyn, static_df, train_target)

    train_dyn_std, train_static_std, train_y_std = standardize_data(
        train_dyn, static_df, train_target, train_scalers
    )

    val_dyn_std, val_static_std, val_y_std = standardize_data(
        val_dyn, static_df, val_target, train_scalers
    )

    train_dataset = MultiscaleLSTMDataset(
        dataset=train_dyn_std,
        target=train_y_std,
        static_df=train_static_std,
        start_date=config.TRAIN_START,
        end_date=config.TRAIN_END
    )

    val_dataset = MultiscaleLSTMDataset(
        dataset=val_dyn_std,
        target=val_y_std,
        static_df=val_static_std,
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

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    return train_loader, val_loader, train_scalers


# =====================================================
# 2️⃣ train_model 
# =====================================================
def main():

    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, _ = get_dataloaders()

    model = sMTSLSTM(
        dyn_input_size=config.DYN_INPUT_SIZE,
        static_input_size=config.STATIC_INPUT_SIZE,
        hidden_size_daily=config.HIDDEN_SIZE_DAILY,
        hidden_size_hourly=config.HIDDEN_SIZE_HOURLY,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        frequency_factor=config.FREQUENCY_FACTOR
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    wandb.init(
        project=config.PROJECT_NAME,
        name=f"{config.RUN_NAME}_{int(time.time())}"
    )

    trainer = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        config
    )

    trainer.fit()

    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    print("Model saved to", config.MODEL_SAVE_PATH)
    wandb.finish()



if __name__ == "__main__":
    main()