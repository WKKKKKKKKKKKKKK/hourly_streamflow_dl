# ===============================
# inference.py  (LSTM version)
# ===============================

import torch
import numpy as np
from Train import get_dataloaders
from Modelzoo import LSTM
import config


# =====================================================
# Device
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# 1️⃣  dataloader & scaler
# =====================================================
_, _, test_loader, train_scalers = get_dataloaders()


# =====================================================
# 2️⃣ model
# =====================================================
model = LSTM(
    input_size=config.INPUT_SIZE,
    hidden_size=config.HIDDEN_SIZE,
    num_layers=config.NUM_LAYERS,
    dropout=config.DROPOUT,
    output_size=config.OUTPUT_SIZE
).to(device)

model.load_state_dict(
    torch.load(config.MODEL_SAVE_PATH, map_location=device)
)

model.eval()

print("Model loaded successfully.")


# =====================================================
# Metrics
# =====================================================
def compute_nse(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) < 2:
        return np.nan

    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return np.nan

    return 1 - np.sum((sim - obs) ** 2) / denom


def compute_kge(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[mask]
    sim = sim[mask]

    if len(obs) < 2:
        return np.nan

    mean_obs = np.mean(obs)
    std_obs = np.std(obs)

    if std_obs == 0 or mean_obs == 0:
        return np.nan

    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / std_obs
    beta = np.mean(sim) / mean_obs

    return 1 - np.sqrt(
        (r - 1) ** 2 +
        (alpha - 1) ** 2 +
        (beta - 1) ** 2
    )


# =====================================================
# 3️⃣ Inference
# =====================================================
preds_by_station = {}
trues_by_station = {}

with torch.no_grad():
    for x_dyn_batch, x_static_batch, y_batch, stn_batch in test_loader:

        x_dyn_batch = x_dyn_batch.to(device)
        x_static_batch = x_static_batch.to(device)

        outputs = model((x_dyn_batch, x_static_batch))

        preds = outputs.cpu().numpy()
        y = y_batch.cpu().numpy()
        stn_batch = np.array(stn_batch)

        for i, station in enumerate(stn_batch):

            if station not in preds_by_station:
                preds_by_station[station] = []
                trues_by_station[station] = []

            preds_by_station[station].append(preds[i, 0])
            trues_by_station[station].append(y[i, 0])


# =====================================================
# 4️⃣ inverse transform & compute metrics
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