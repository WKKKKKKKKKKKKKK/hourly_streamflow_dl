# ===============================
# inference.py
# ===============================

import torch
import numpy as np
from Train import get_dataloaders
from Modelzoo import sMTSLSTM
import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1️⃣  dataloader & scaler
_, val_loader, train_scalers = get_dataloaders()

# 2️⃣ model
model = sMTSLSTM(
    dyn_input_size=config.DYN_INPUT_SIZE,
    static_input_size=config.STATIC_INPUT_SIZE,
    hidden_size_daily=config.HIDDEN_SIZE_DAILY,
    hidden_size_hourly=config.HIDDEN_SIZE_HOURLY,
    num_layers=config.NUM_LAYERS,
    dropout=config.DROPOUT,
    frequency_factor=config.FREQUENCY_FACTOR
).to(device)

model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
model.eval()

print("Model loaded successfully.")


# =====================================================
# metric functions
# =====================================================
def compute_nse(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) < 2:
        return np.nan

    denom = np.sum((obs - np.mean(obs))**2)
    return 1 - np.sum((sim - obs)**2) / denom


def compute_kge(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) < 2:
        return np.nan

    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)

    return 1 - np.sqrt((r - 1)**2 +
                       (alpha - 1)**2 +
                       (beta - 1)**2)


# =====================================================
# Inference
# =====================================================
preds_by_station = {}
trues_by_station = {}

with torch.no_grad():
    for batch in val_loader:

        x_dict, y, stn = batch

        H = x_dict["H"].to(device)
        D = x_dict["D"].to(device)
        S = x_dict["S"].to(device)

        outputs = model({"H": H, "D": D}, S)

        preds = outputs["H"].cpu().numpy()
        y = y.cpu().numpy()

        for i, station in enumerate(stn):

            if station not in preds_by_station:
                preds_by_station[station] = []
                trues_by_station[station] = []

            preds_by_station[station].append(preds[i])
            trues_by_station[station].append(y[i, 0])


# =====================================================
# inverse transform & compute metrics
# =====================================================
y_mean = train_scalers["y_mean"]
y_std = train_scalers["y_std"]

nse_dict = {}
kge_dict = {}

for station in preds_by_station:

    sim = np.array(preds_by_station[station]) * y_std + y_mean
    obs = np.array(trues_by_station[station]) * y_std + y_mean

    nse_dict[station] = compute_nse(obs, sim)
    kge_dict[station] = compute_kge(obs, sim)

print("================================")
print("Median NSE:", np.nanmedian(list(nse_dict.values())))
print("Median KGE:", np.nanmedian(list(kge_dict.values())))
print("================================")