# ===============================
# Train.py (sMTSLSTM)
# - Supports CLI hyperparameter overrides (manual tuning or W&B sweeps)
# - Trains, saves model + scalers, and evaluates on val/test (NSE/KGE)
# ===============================

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
from Modelzoo import sMTSLSTM
from loder import handle_extremes, standardize_data, calculate_scalers, MultiscaleLSTMDataset
from trainer import train_model
from losses import NSELoss, MTSNSERegularizedLoss


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
    p.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    p.add_argument('--num-workers', type=int, default=config.NUM_WORKERS)
    p.add_argument('--pin-memory', type=str2bool, nargs='?', const=True, default=config.PIN_MEMORY)
    p.add_argument('--no-pin-memory', dest='pin_memory', action='store_false')

    p.add_argument('--lookback-hourly', type=int, default=config.LOOKBACK_HOURLY)
    p.add_argument('--lookback-daily', type=int, default=config.LOOKBACK_DAILY)
    p.add_argument('--frequency-factor', type=int, default=config.FREQUENCY_FACTOR)

    # Model
    p.add_argument('--hidden-size-daily', type=int, default=config.HIDDEN_SIZE_DAILY)
    p.add_argument('--hidden-size-hourly', type=int, default=config.HIDDEN_SIZE_HOURLY)
    p.add_argument('--num-layers', type=int, default=config.NUM_LAYERS)
    p.add_argument('--dropout', type=float, default=config.DROPOUT)

    # Training
    p.add_argument('--lr', type=float, default=config.LEARNING_RATE)
    p.add_argument('--lr-schedule', default='', help='e.g. 1:5e-4,10:1e-4,25:5e-5')
    p.add_argument('--epochs', type=int, default=config.NUM_EPOCHS)
    p.add_argument('--loss', choices=['mse', 'nse_loss'], default='nse_loss')
    p.add_argument('--reg-lambda', type=float, default=1.0)
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
    p.add_argument('--lookback_hourly', dest='lookback_hourly', type=int)
    p.add_argument('--lookback_daily', dest='lookback_daily', type=int)
    p.add_argument('--frequency_factor', dest='frequency_factor', type=int)
    p.add_argument('--lr_schedule', dest='lr_schedule')
    p.add_argument('--hidden_size_daily', dest='hidden_size_daily', type=int)
    p.add_argument('--hidden_size_hourly', dest='hidden_size_hourly', type=int)

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


def evaluate_per_station(
    model: torch.nn.Module,
    loader: DataLoader,
    scalers: dict,
    device: torch.device,
    expected_stations=None,
) -> pd.DataFrame:
    model.eval()

    preds_by_station = {}
    trues_by_station = {}

    with torch.no_grad():
        for x_dict, y, stn in loader:
            H = x_dict['H'].to(device)
            D = x_dict['D'].to(device)
            S = x_dict['S'].to(device)

            out = model({'H': H, 'D': D}, S)
            preds = out['H'].detach().cpu().numpy().reshape(-1)
            trues = y.detach().cpu().numpy().reshape(-1)

            for i, station in enumerate(stn):
                station = str(station)
                preds_by_station.setdefault(station, []).append(preds[i])
                trues_by_station.setdefault(station, []).append(trues[i])

    y_mean = scalers['y_mean']
    y_std = scalers['y_std']

    station_ids = set(preds_by_station)
    if expected_stations is not None:
        station_ids.update(str(station) for station in expected_stations)

    rows = []
    for station in sorted(station_ids):
        pred_values = preds_by_station.get(station, [])
        true_values = trues_by_station.get(station, [])
        row = {
            'station_id': station,
            'samples': int(len(true_values)),
            'score_status': 'ok',
            'exclusion_reason': '',
            'nse': float('nan'),
            'kge': float('nan'),
        }

        if not true_values:
            row['score_status'] = 'excluded'
            row['exclusion_reason'] = 'no_valid_windows'
            rows.append(row)
            continue

        sim = np.asarray(pred_values, dtype='float64') * y_std + y_mean
        obs = np.asarray(true_values, dtype='float64') * y_std + y_mean
        nse = compute_nse(obs, sim)
        kge = compute_kge(obs, sim)

        if not np.isfinite(nse) or not np.isfinite(kge):
            reasons = []
            if obs.size < 2:
                reasons.append('too_few_samples')
            obs_std = float(np.nanstd(obs)) if obs.size else float('nan')
            obs_mean = float(np.nanmean(obs)) if obs.size else float('nan')
            if not np.isfinite(obs_std):
                reasons.append('obs_std_nonfinite')
            elif obs_std == 0:
                reasons.append('obs_std_zero')
            if not np.isfinite(obs_mean):
                reasons.append('obs_mean_nonfinite')
            elif obs_mean == 0:
                reasons.append('obs_mean_zero')
            if not reasons:
                reasons.append('metric_nonfinite')
            row['score_status'] = 'excluded'
            row['exclusion_reason'] = '+'.join(reasons)
            rows.append(row)
            continue

        row['nse'] = float(nse)
        row['kge'] = float(kge)
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate(model: torch.nn.Module, loader: DataLoader, scalers: dict, device: torch.device) -> dict:
    station_metrics = evaluate_per_station(model, loader, scalers, device)
    nse = station_metrics['nse'].to_numpy(dtype='float64') if len(station_metrics) else np.array([], dtype='float64')
    kge = station_metrics['kge'].to_numpy(dtype='float64') if len(station_metrics) else np.array([], dtype='float64')
    return {
        'median_nse': float(np.nanmedian(nse)) if len(nse) else float('nan'),
        'median_kge': float(np.nanmedian(kge)) if len(kge) else float('nan'),
    }




def parse_lr_schedule(schedule: str):
    schedule = (schedule or '').strip()
    if not schedule:
        return None

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

def _normalize_station_id(station_id) -> str:
    station_id = str(station_id).strip()
    if station_id.endswith('.0'):
        whole, frac = station_id.rsplit('.', 1)
        if frac == '0':
            station_id = whole
    return station_id.lstrip('0') or '0'


def _add_static_station_aliases(static_df: pd.DataFrame, station_ids):
    static_df = static_df.copy()
    static_df.index = pd.Index([str(idx).strip() for idx in static_df.index])

    canonical_to_index = {}
    ambiguous = set()
    for idx in static_df.index:
        key = _normalize_station_id(idx)
        if key in canonical_to_index and canonical_to_index[key] != idx:
            ambiguous.add(key)
        else:
            canonical_to_index[key] = idx
    for key in ambiguous:
        canonical_to_index.pop(key, None)

    alias_rows = {}
    missing = []
    aliased = []
    for station_id in station_ids:
        station_id = str(station_id).strip()
        if station_id in static_df.index:
            continue

        match = canonical_to_index.get(_normalize_station_id(station_id))
        if match is None:
            missing.append(station_id)
            continue

        alias_rows[station_id] = static_df.loc[match].copy()
        aliased.append(station_id)

    if alias_rows:
        alias_df = pd.DataFrame.from_dict(alias_rows, orient='index')
        alias_df = alias_df.reindex(columns=static_df.columns)
        static_df = pd.concat([static_df, alias_df], axis=0)

    static_df.attrs['aliased_station_ids'] = aliased
    return static_df, missing


def open_selected_dataset(data_path: str):
    path_str = str(data_path)
    if ',' in path_str:
        parts = [part.strip() for part in path_str.split(',') if part.strip()]
    elif any(ch in path_str for ch in '*?[]'):
        import glob
        parts = sorted(glob.glob(path_str))
    else:
        parts = [path_str]

    if not parts:
        raise FileNotFoundError(f'No dataset files matched: {data_path}')

    datasets = [xr.open_dataset(part, decode_times=False) for part in parts]
    merged = datasets[0] if len(datasets) == 1 else xr.merge(datasets, compat='override', combine_attrs='override')

    if 'time' in merged.dims:
        time_index = pd.date_range('1980-01-01 00:00:00', periods=int(merged.sizes['time']), freq='h')
        merged = merged.assign_coords(time=time_index)

    return merged


def get_dataloaders(cfg: SimpleNamespace):
    selected_stn_data = open_selected_dataset(cfg.data_path)
    dyn = handle_extremes(selected_stn_data, min_streamflow=cfg.streamflow_min, max_streamflow=cfg.streamflow_max)

    station_ids = [str(station_id).strip() for station_id in selected_stn_data.data_vars]

    static_df = pd.read_csv(cfg.static_path, index_col=0)
    static_df.index = pd.Index([str(idx).strip() for idx in static_df.index])

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

    train_static_std, missing_static = _add_static_station_aliases(train_static_std, station_ids)
    val_static_std, _ = _add_static_station_aliases(val_static_std, station_ids)
    test_static_std, _ = _add_static_station_aliases(test_static_std, station_ids)

    aliased_station_ids = train_static_std.attrs.get('aliased_station_ids', [])
    if aliased_station_ids:
        preview = ', '.join(aliased_station_ids[:10])
        suffix = ' ...' if len(aliased_station_ids) > 10 else ''
        print(f"Aliased {len(aliased_station_ids)} station IDs in static features: {preview}{suffix}")

    if missing_static:
        preview = ', '.join(missing_static[:10])
        suffix = ' ...' if len(missing_static) > 10 else ''
        raise KeyError(f"Missing static features for {len(missing_static)} selected stations: {preview}{suffix}")

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


    train_dataset = MultiscaleLSTMDataset(train_dyn_std, train_y_std, train_static_std,
                                         lookback_hourly=cfg.lookback_hourly,
                                         lookback_daily=cfg.lookback_daily,
                                         frequency_factor=cfg.frequency_factor,
                                         start_date=cfg.train_start, end_date=cfg.train_end)

    val_dataset = MultiscaleLSTMDataset(val_dyn_std, val_y_std, val_static_std,
                                        lookback_hourly=cfg.lookback_hourly,
                                        lookback_daily=cfg.lookback_daily,
                                        frequency_factor=cfg.frequency_factor,
                                        start_date=cfg.val_start, end_date=cfg.val_end)

    test_dataset = MultiscaleLSTMDataset(test_dyn_std, test_y_std, test_static_std,
                                         lookback_hourly=cfg.lookback_hourly,
                                         lookback_daily=cfg.lookback_daily,
                                         frequency_factor=cfg.frequency_factor,
                                         start_date=cfg.test_start, end_date=cfg.test_end)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    return train_loader, val_loader, test_loader, train_scalers




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

    if "Baseline" in "MTSLSTM":
        tag = f"lr{cfg.lr}_bs{cfg.batch_size}_lb{cfg.lookback}_hs{cfg.hidden_size}_do{cfg.dropout}_loss{cfg.loss}"
    else:
        sched = getattr(cfg, 'lr_schedule', '')
        if not sched:
            sched = getattr(cfg, 'lr', '')
        tag = f"hd{cfg.hidden_size_daily}-{cfg.hidden_size_hourly}_bs{cfg.batch_size}_H{cfg.lookback_hourly}_D{cfg.lookback_daily}_do{cfg.dropout}_loss{cfg.loss}_reg{getattr(cfg,'reg_lambda',0)}_sch{sched}"

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
    if getattr(cfg, 'hidden_size', None) is not None:
        cfg.hidden_size_daily = cfg.hidden_size
        cfg.hidden_size_hourly = cfg.hidden_size

    set_seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print('Run hparams:', {
        'lookback_hourly': cfg.lookback_hourly,
        'lookback_daily': cfg.lookback_daily,
        'batch_size': cfg.batch_size,
        'hidden_size_daily': cfg.hidden_size_daily,
        'hidden_size_hourly': cfg.hidden_size_hourly,
        'num_layers': cfg.num_layers,
        'dropout': cfg.dropout,
        'lr': cfg.lr,
        'lr_schedule': cfg.lr_schedule,
        'epochs': cfg.epochs,
        'loss': cfg.loss,
        'reg_lambda': cfg.reg_lambda,
        'early_stopping': cfg.early_stopping,
        'patience': cfg.patience,
    })

    train_loader, val_loader, test_loader, train_scalers = get_dataloaders(cfg)

    model = sMTSLSTM(
        dyn_input_size=config.DYN_INPUT_SIZE,
        static_input_size=config.STATIC_INPUT_SIZE,
        hidden_size_daily=cfg.hidden_size_daily,
        hidden_size_hourly=cfg.hidden_size_hourly,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        frequency_factor=cfg.frequency_factor,
    ).to(device)

    if cfg.loss == 'nse_loss':
        criterion = MTSNSERegularizedLoss(
            station_std=train_scalers.get('station_y_std', {}),
            frequency_factor=cfg.frequency_factor,
            reg_lambda=cfg.reg_lambda,
            eps=cfg.nse_eps,
        )
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


    cfg = resolve_outputs(cfg)

    # Pass a config-like object into the existing trainer
    trainer_cfg = SimpleNamespace(
        NUM_EPOCHS=cfg.epochs,
        USE_EARLY_STOPPING=cfg.early_stopping,
        EARLY_STOPPING_PATIENCE=cfg.patience,
        BEST_MODEL_PATH=cfg.best_model_path,
        LR_SCHEDULE=lr_schedule,
        REG_LAMBDA=cfg.reg_lambda,
        FREQUENCY_FACTOR=cfg.frequency_factor,
        CHECKPOINT_PATH=cfg.checkpoint_path,
        RESUME=cfg.resume,
        SAVE_EVERY=cfg.save_every,
    )

    trainer = train_model(model, train_loader, val_loader, criterion, optimizer, device, trainer_cfg)
    trainer.fit()

    # Save scalers
    with open(cfg.scaler_save_path, 'wb') as f:
        pickle.dump(train_scalers, f)

    # Prefer best checkpoint if available
    if cfg.early_stopping and os.path.exists(cfg.best_model_path):
        model.load_state_dict(torch.load(cfg.best_model_path, map_location=device))

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
