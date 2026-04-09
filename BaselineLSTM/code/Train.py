# =====================================================
# Train.py (Baseline LSTM)
# - Supports CLI hyperparameter overrides (manual tuning or W&B sweeps)
# - Trains, saves model + scalers, and evaluates on val/test (NSE/KGE)
# =====================================================

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    s = str(v).strip().lower()
    if s in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if s in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {v!r}')

import os
import pickle
import random
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from torch.utils.data import DataLoader

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

import config
from Modelzoo import LSTM
from loder import handle_extremes, standardize_data, calculate_scalers, LSTMDataset
from trainer import train_model
from losses import NSELoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data / preprocessing
    p.add_argument('--data-path', default=config.DATA_PATH)
    p.add_argument('--static-path', default=config.STATIC_PATH)
    p.add_argument('--streamflow-min', type=float, default=0.0)
    p.add_argument('--streamflow-max', type=float, default=1000.0)

    # Time splits
    p.add_argument('--train-start', default=config.TRAIN_START)
    p.add_argument('--train-end', default=config.TRAIN_END)
    p.add_argument('--val-start', default=config.VAL_START)
    p.add_argument('--val-end', default=config.VAL_END)
    p.add_argument('--test-start', default=config.TEST_START)
    p.add_argument('--test-end', default=config.TEST_END)

    # Variables
    p.add_argument('--dynamic-vars', nargs='+', default=config.DYNAMIC_VARS)
    p.add_argument('--target-var', default=config.TARGET_VAR)

    # Dataloader
    p.add_argument('--lookback', type=int, default=config.LOOKBACK)
    p.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    p.add_argument('--num-workers', type=int, default=config.NUM_WORKERS)
    p.add_argument('--pin-memory', type=str2bool, nargs='?', const=True, default=config.PIN_MEMORY)
    p.add_argument('--no-pin-memory', dest='pin_memory', action='store_false')

    # Model
    p.add_argument('--hidden-size', type=int, default=config.HIDDEN_SIZE)
    p.add_argument('--num-layers', type=int, default=config.NUM_LAYERS)
    p.add_argument('--dropout', type=float, default=config.DROPOUT)

    # Training
    p.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    p.add_argument('--lr-schedule', default='', help='e.g. 1:5e-4,10:1e-4,25:5e-5')
    p.add_argument('--epochs', type=int, default=config.NUM_EPOCHS)
    p.add_argument('--loss', choices=['mse', 'nse_loss'], default='nse_loss')
    p.add_argument('--nse-eps', type=float, default=1e-6)
    p.add_argument('--early-stopping', action='store_true', default=config.USE_EARLY_STOPPING)
    p.add_argument('--no-early-stopping', dest='early_stopping', action='store_false')
    p.add_argument('--patience', type=int, default=config.EARLY_STOPPING_PATIENCE)

    # Reproducibility
    p.add_argument('--seed', type=int, default=config.SEED)

    # Outputs
    p.add_argument('--output-dir', default=None, help='If set, write model/scalers/checkpoints into this directory.')
    p.add_argument('--model-save-path', default=config.MODEL_SAVE_PATH)
    p.add_argument('--best-model-path', default=config.BEST_MODEL_PATH)
    p.add_argument('--scaler-save-path', default=os.path.join(os.path.dirname(config.MODEL_SAVE_PATH), 'scalers.pkl'))

    # Checkpointing / resume
    p.add_argument('--resume', type=str2bool, nargs='?', const=True, default=True)
    p.add_argument('--no-resume', dest='resume', action='store_false')
    p.add_argument('--checkpoint-path', default=None)
    p.add_argument('--save-every', type=int, default=1)

    
    # Underscore aliases (W&B sweeps use these flag names)
    p.add_argument('--batch_size', dest='batch_size', type=int)
    p.add_argument('--hidden_size', dest='hidden_size', type=int)
    p.add_argument('--num_workers', dest='num_workers', type=int)
    p.add_argument('--pin_memory', dest='pin_memory', type=str2bool, nargs='?', const=True)
    p.add_argument('--lr_schedule', dest='lr_schedule')

# W&B
    p.add_argument('--wandb', type=str2bool, nargs='?', const=True, default=True)
    p.add_argument('--no-wandb', dest='wandb', action='store_false')
    p.add_argument('--wandb-project', default=config.PROJECT_NAME)
    p.add_argument('--wandb-run-name', default=config.RUN_NAME)
    p.add_argument('--wandb-mode', default=os.environ.get('WANDB_MODE', 'online'))

    return p.parse_args()


def compute_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[mask]
    sim = sim[mask]
    if obs.size < 2:
        return float('nan')
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return float('nan')
    return float(1 - np.sum((sim - obs) ** 2) / denom)


def compute_kge(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[mask]
    sim = sim[mask]
    if obs.size < 2:
        return float('nan')

    mean_obs = np.mean(obs)
    std_obs = np.std(obs)
    if std_obs == 0 or mean_obs == 0:
        return float('nan')

    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / std_obs
    beta = np.mean(sim) / mean_obs

    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def evaluate(model: torch.nn.Module, loader: DataLoader, scalers: dict, device: torch.device) -> dict:
    model.eval()

    preds_by_station = {}
    trues_by_station = {}

    with torch.no_grad():
        for x_dyn_batch, x_static_batch, y_batch, stn_batch in loader:
            x_dyn_batch = x_dyn_batch.to(device)
            x_static_batch = x_static_batch.to(device)
            out = model((x_dyn_batch, x_static_batch))

            preds = out.detach().cpu().numpy().reshape(-1)
            trues = y_batch.detach().cpu().numpy().reshape(-1)

            for i, stn in enumerate(stn_batch):
                preds_by_station.setdefault(stn, []).append(preds[i])
                trues_by_station.setdefault(stn, []).append(trues[i])

    y_mean = scalers['y_mean']
    y_std = scalers['y_std']

    nse = []
    kge = []
    for stn in preds_by_station:
        sim = np.asarray(preds_by_station[stn], dtype='float64') * y_std + y_mean
        obs = np.asarray(trues_by_station[stn], dtype='float64') * y_std + y_mean
        nse.append(compute_nse(obs, sim))
        kge.append(compute_kge(obs, sim))

    return {
        'median_nse': float(np.nanmedian(nse)) if len(nse) else float('nan'),
        'median_kge': float(np.nanmedian(kge)) if len(kge) else float('nan'),
    }


def get_dataloaders(cfg: SimpleNamespace):
    selected_stn_data = xr.open_dataset(cfg.data_path)
    dyn = handle_extremes(selected_stn_data, min_streamflow=cfg.streamflow_min, max_streamflow=cfg.streamflow_max)

    static_df = pd.read_csv(cfg.static_path, index_col=0)
    static_df.index = static_df.index.astype(str).str.zfill(8)

    dyn_forcing = dyn.sel(dynamic_forcing=cfg.dynamic_vars)
    target_var_da = dyn.sel(dynamic_forcing=cfg.target_var)

    train_dyn = dyn_forcing.sel(time=slice(cfg.train_start, cfg.train_end))
    train_target = target_var_da.sel(time=slice(cfg.train_start, cfg.train_end))

    val_dyn = dyn_forcing.sel(time=slice(cfg.val_start, cfg.val_end))
    val_target = target_var_da.sel(time=slice(cfg.val_start, cfg.val_end))

    test_dyn = dyn_forcing.sel(time=slice(cfg.test_start, cfg.test_end))
    test_target = target_var_da.sel(time=slice(cfg.test_start, cfg.test_end))

    train_scalers = calculate_scalers(train_dyn, static_df, train_target)

    train_dyn_std, train_static_std, train_y_std = standardize_data(train_dyn, static_df, train_target, train_scalers)
    val_dyn_std, val_static_std, val_y_std = standardize_data(val_dyn, static_df, val_target, train_scalers)
    test_dyn_std, test_static_std, test_y_std = standardize_data(test_dyn, static_df, test_target, train_scalers)

    # Per-station std (computed on standardized y, training period)
    # Some stations may have too few finite y values in the training split;
    # in that case fall back to the global std to avoid NaNs in NSE loss.
    station_y_std = {}
    all_finite = []
    for stn in train_y_std.data_vars:
        vals = np.asarray(train_y_std[stn].values).astype('float64').ravel()
        finite = vals[np.isfinite(vals)]
        if finite.size >= 2:
            station_y_std[str(stn)] = float(np.std(finite))
            all_finite.append(finite)

    if len(all_finite):
        global_std = float(np.std(np.concatenate(all_finite)))
        if not np.isfinite(global_std) or global_std <= 0:
            global_std = 1.0
    else:
        global_std = 1.0

    for stn in train_y_std.data_vars:
        stn_key = str(stn)
        s = station_y_std.get(stn_key, global_std)
        if (not np.isfinite(s)) or s <= 0:
            s = global_std
        station_y_std[stn_key] = float(s)

    train_scalers['station_y_std'] = station_y_std


    train_dataset = LSTMDataset(train_dyn_std, train_y_std, train_static_std, lookback=cfg.lookback,
                               start_date=cfg.train_start, end_date=cfg.train_end)
    val_dataset = LSTMDataset(val_dyn_std, val_y_std, val_static_std, lookback=cfg.lookback,
                              start_date=cfg.val_start, end_date=cfg.val_end)
    test_dataset = LSTMDataset(test_dyn_std, test_y_std, test_static_std, lookback=cfg.lookback,
                               start_date=cfg.test_start, end_date=cfg.test_end)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    return train_loader, val_loader, test_loader, train_scalers




def parse_lr_schedule(schedule: str):
    schedule = (schedule or '').strip()
    if not schedule:
        return None

    # Normalize unicode minus and remove spaces
    schedule = schedule.replace('−', '-').replace(' ', '')

    items = []
    for part in schedule.split(','):
        if not part:
            continue
        if ':' not in part:
            raise ValueError(f'Invalid lr schedule segment: {part!r}')
        epoch_s, lr_s = part.split(':', 1)
        epoch = int(epoch_s)
        lr = float(lr_s)
        items.append((epoch, lr))

    items.sort(key=lambda x: x[0])
    return items



def _sanitize_tag(s: str) -> str:
    s = str(s)
    for ch in [':', ',', ' ', '/', '\\', '(', ')', '[', ']', '{', '}', '|', ';']:
        s = s.replace(ch, '-')
    while '--' in s:
        s = s.replace('--', '-')
    return s.strip('-')


def _maybe_autotag_output_dir(cfg) -> str:
    run_id = os.environ.get('WANDB_RUN_ID', '')
    if not run_id or not cfg.output_dir:
        return cfg.output_dir

    # Only auto-tag when output-dir ends with the run id (sweep.yaml uses that pattern).
    if not cfg.output_dir.rstrip('/').endswith(run_id):
        return cfg.output_dir

    if "Baseline" in "BaselineLSTM":
        tag = f"lr{cfg.lr}_bs{cfg.batch_size}_lb{cfg.lookback}_hs{cfg.hidden_size}_do{cfg.dropout}_loss{cfg.loss}"
    else:
        tag = "run"

    tag = _sanitize_tag(tag)
    # Keep run id, but add readable suffix.
    return cfg.output_dir + '_' + tag


def resolve_outputs(cfg: SimpleNamespace) -> SimpleNamespace:
    if not cfg.output_dir:
        return cfg

    # Replace placeholder used in sweep.yaml with the actual W&B run id (available after wandb.init).
    run_id = os.environ.get('WANDB_RUN_ID', '')
    if run_id and cfg.output_dir.rstrip('/').endswith('WANDB_RUN_ID'):
        cfg.output_dir = cfg.output_dir.rstrip('/')
        cfg.output_dir = cfg.output_dir[: -len('WANDB_RUN_ID')] + run_id

    cfg.output_dir = _maybe_autotag_output_dir(cfg)

    out_dir = os.path.abspath(cfg.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    cfg.model_save_path = os.path.join(out_dir, 'model.pth')
    cfg.best_model_path = os.path.join(out_dir, 'best_model.pth')
    cfg.scaler_save_path = os.path.join(out_dir, 'scalers.pkl')
    if getattr(cfg, 'checkpoint_path', None) is None:
        cfg.checkpoint_path = os.path.join(out_dir, 'checkpoint.pth')
    return cfg


def main() -> None:
    args = parse_args()
    cfg = SimpleNamespace(**vars(args))

    set_seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print('Run hparams:', {
        'lookback': cfg.lookback,
        'batch_size': cfg.batch_size,
        'hidden_size': cfg.hidden_size,
        'num_layers': cfg.num_layers,
        'dropout': cfg.dropout,
        'lr': cfg.lr,
        'epochs': cfg.epochs,
        'loss': cfg.loss,
        'early_stopping': cfg.early_stopping,
        'patience': cfg.patience,
    })

    train_loader, val_loader, test_loader, train_scalers = get_dataloaders(cfg)

    input_size = config.DYN_INPUT_SIZE + config.STATIC_INPUT_SIZE
    model = LSTM(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        output_size=config.OUTPUT_SIZE,
    ).to(device)

    if cfg.loss == 'nse_loss':
        criterion = NSELoss(station_std=train_scalers.get('station_y_std', {}), eps=cfg.nse_eps)
    else:
        criterion = nn.MSELoss()
    lr_schedule = parse_lr_schedule(cfg.lr_schedule)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    os.makedirs(os.path.dirname(cfg.model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.best_model_path), exist_ok=True)

    use_wandb = bool(cfg.wandb) and (wandb is not None)
    if use_wandb:
        wandb.init(project=cfg.wandb_project, name=f"{cfg.wandb_run_name}_{int(time.time())}", mode=cfg.wandb_mode)
        # In sweeps, sweep parameters are locked; only add keys that are not set by the sweep.
        for k, v in vars(cfg).items():
            if k not in wandb.config:
                wandb.config[k] = v
        wandb.watch(model, log='gradients', log_freq=200)


    cfg = resolve_outputs(cfg)

    # Train (optionally with early stopping + best checkpoint)
    train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=cfg.epochs,
        patience=cfg.patience,
        best_model_path=cfg.best_model_path,
        early_stopping=cfg.early_stopping,
        lr_schedule=lr_schedule,
        checkpoint_path=cfg.checkpoint_path,
        resume=cfg.resume,
        save_every=cfg.save_every,
    )

    # Save scalers for reproducible inference
    with open(cfg.scaler_save_path, 'wb') as f:
        pickle.dump(train_scalers, f)

    # Prefer best model if early stopping was enabled and checkpoint exists
    if cfg.early_stopping and os.path.exists(cfg.best_model_path):
        model.load_state_dict(torch.load(cfg.best_model_path, map_location=device))

    # Always save final (or best-loaded) model
    torch.save(model.state_dict(), cfg.model_save_path)
    print('Model saved to', cfg.model_save_path)

    # Evaluate
    val_metrics = evaluate(model, val_loader, train_scalers, device)
    test_metrics = evaluate(model, test_loader, train_scalers, device)

    print('VAL  median NSE:', val_metrics['median_nse'], 'median KGE:', val_metrics['median_kge'])
    print('TEST median NSE:', test_metrics['median_nse'], 'median KGE:', test_metrics['median_kge'])

    # Mark run as completed so grid resubmits can skip finished indices
    try:
        out_dir = os.path.abspath(cfg.output_dir) if getattr(cfg, 'output_dir', None) else os.path.dirname(cfg.model_save_path)
        done_path = os.path.join(out_dir, 'DONE')
        with open(done_path, 'w', encoding='utf-8') as f:
            f.write('ok\n')
    except Exception as e:
        print('WARNING: failed to write DONE marker:', e)

    if use_wandb:
        wandb.log({
            'val/median_nse': val_metrics['median_nse'],
            'val/median_kge': val_metrics['median_kge'],
            'test/median_nse': test_metrics['median_nse'],
            'test/median_kge': test_metrics['median_kge'],
        })
        wandb.finish()


if __name__ == '__main__':
    main()
