# =====================================================
# Train.py  
# =====================================================

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================
# 1️⃣ dataloader
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

    test_dyn = dyn_forcing.sel(time=slice(config.TEST_START, config.TEST_END))
    test_target = target_var_da.sel(time=slice(config.TEST_START, config.TEST_END))

    # 
    train_scalers = calculate_scalers(train_dyn, static_df, train_target)

    train_dyn_std, train_static_std, train_y_std = standardize_data(
        train_dyn, static_df, train_target, train_scalers
    )

    val_dyn_std, val_static_std, val_y_std = standardize_data(
        val_dyn, static_df, val_target, train_scalers
    )

    test_dyn_std, test_static_std, test_y_std = standardize_data(
        test_dyn, static_df, test_target, train_scalers
    )

    # Dataset
    train_dataset = LSTMDataset(
        dataset=train_dyn_std,
        target=train_y_std,
        static_df=train_static_std,
        lookback=config.LOOKBACK,
        start_date=config.TRAIN_START,
        end_date=config.TRAIN_END
    )

    val_dataset = LSTMDataset(
        dataset=val_dyn_std,
        target=val_y_std,
        static_df=val_static_std,
        lookback=config.LOOKBACK,
        start_date=config.VAL_START,
        end_date=config.VAL_END
    )

    test_dataset = LSTMDataset(
        dataset=test_dyn_std,
        target=test_y_std,
        static_df=test_static_std,
        lookback=config.LOOKBACK,
        start_date=config.TEST_START,
        end_date=config.TEST_END
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    return train_loader, val_loader, test_loader, train_scalers


# =====================================================
# 2️⃣ mian 
# =====================================================

def main():

    set_seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, _ = get_dataloaders()

   
    dynamic_size = config.DYN_INPUT_SIZE
    static_size = config.STATIC_INPUT_SIZE
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

    # WandB
    wandb.init(
        project=config.PROJECT_NAME,
        name=f"{config.RUN_NAME}_{int(time.time())}",
    )

    wandb.config.update({
        "DATA_PATH": config.DATA_PATH,
        "STATIC_PATH": config.STATIC_PATH,
        "LOOKBACK": config.LOOKBACK,
        "BATCH_SIZE": config.BATCH_SIZE,
        "NUM_WORKERS": config.NUM_WORKERS,
        "HIDDEN_SIZE": config.HIDDEN_SIZE,
        "NUM_LAYERS": config.NUM_LAYERS,
        "DROPOUT": config.DROPOUT,
        "LEARNING_RATE": config.LEARNING_RATE,
        "NUM_EPOCHS": config.NUM_EPOCHS,
        "SEED": config.SEED,
    })

    wandb.watch(model, log="gradients", log_freq=200)

    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=config.NUM_EPOCHS
    )

    # Save model
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    print("Model saved to", config.MODEL_SAVE_PATH)
    wandb.finish()



if __name__ == "__main__":
    main()