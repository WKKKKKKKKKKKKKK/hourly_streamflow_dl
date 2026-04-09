# ===============================
# inference.py (sMTSLSTM)
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

from types import SimpleNamespace

import torch

import config
from Modelzoo import sMTSLSTM
from Train import get_dataloaders, evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['train', 'val', 'test'], default='val')

    # Keep these aligned with Train.py so we can reuse its dataloader builder.
    p.add_argument('--data-path', default=config.DATA_PATH)
    p.add_argument('--static-path', default=config.STATIC_PATH)
    p.add_argument('--streamflow-min', type=float, default=0.0)
    p.add_argument('--streamflow-max', type=float, default=1000.0)

    p.add_argument('--train-start', default=config.TRAIN_START)
    p.add_argument('--train-end', default=config.TRAIN_END)
    p.add_argument('--val-start', default=config.VAL_START)
    p.add_argument('--val-end', default=config.VAL_END)
    p.add_argument('--test-start', default=config.TEST_START)
    p.add_argument('--test-end', default=config.TEST_END)

    p.add_argument('--dynamic-vars', nargs='+', default=config.DYNAMIC_VARS)
    p.add_argument('--target-var', default=config.TARGET_VAR)

    p.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    p.add_argument('--num-workers', type=int, default=config.NUM_WORKERS)
    p.add_argument('--pin-memory', type=str2bool, nargs='?', const=True, default=config.PIN_MEMORY)
    p.add_argument('--no-pin-memory', dest='pin_memory', action='store_false')

    p.add_argument('--lookback-hourly', type=int, default=config.LOOKBACK_HOURLY)
    p.add_argument('--lookback-daily', type=int, default=config.LOOKBACK_DAILY)
    p.add_argument('--frequency-factor', type=int, default=config.FREQUENCY_FACTOR)

    p.add_argument('--hidden-size-daily', type=int, default=config.HIDDEN_SIZE_DAILY)
    p.add_argument('--hidden-size-hourly', type=int, default=config.HIDDEN_SIZE_HOURLY)
    p.add_argument('--num-layers', type=int, default=config.NUM_LAYERS)
    p.add_argument('--dropout', type=float, default=config.DROPOUT)

    p.add_argument('--model-path', default=config.MODEL_SAVE_PATH)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimpleNamespace(**vars(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader, scalers = get_dataloaders(cfg)
    if cfg.split == 'train':
        loader = train_loader
    elif cfg.split == 'val':
        loader = val_loader
    else:
        loader = test_loader

    model = sMTSLSTM(
        dyn_input_size=config.DYN_INPUT_SIZE,
        static_input_size=config.STATIC_INPUT_SIZE,
        hidden_size_daily=cfg.hidden_size_daily,
        hidden_size_hourly=cfg.hidden_size_hourly,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        frequency_factor=cfg.frequency_factor,
    ).to(device)

    state = torch.load(cfg.model_path, map_location=device)
    model.load_state_dict(state)

    metrics = evaluate(model, loader, scalers, device)

    print('================================')
    print('Split:', cfg.split)
    print('Median NSE:', metrics['median_nse'])
    print('Median KGE:', metrics['median_kge'])
    print('================================')


if __name__ == '__main__':
    main()
