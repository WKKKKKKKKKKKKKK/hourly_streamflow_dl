from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

ROOT = Path('/home/kongw0a/MTS_LSTM/experiment_withcursor')
MTS_DIR = ROOT / 'MTSLSTM'
sys.path.insert(0, str(MTS_DIR))

import config
from Modelzoo import sMTSLSTM
from Train import _add_static_station_aliases, evaluate_per_station
from loder import MultiscaleLSTMDataset, handle_extremes, standardize_data

RUN_DIR = Path(
    '/ibex/project/c2266/wkkong/data/CAEMLSH/data_workstation/CAMELSH/training_runs/'
    '20260407_mtslstm_100stations_tuning_topo18_v100/idx2_bs128_do0.4_hs64_H168_D365'
)
MODEL_PATH = RUN_DIR / 'best_model.pth'
SCALER_PATH = RUN_DIR / 'scalers.pkl'
SAMPLES_CSV = ROOT / 'station_maps' / 'outputs' / 'spatial_generalization_station_samples_conservative.csv'
STATIC_CSV = ROOT / 'data' / 'static_h_topo_priority27.csv'
RAW_DIR = Path('/ibex/project/c2266/wkkong/data/CAEMLSH/data_workstation/CAMELSH/timeseries/Data/CAMELSH/timeseries')
OUT_DIR = MTS_DIR / 'outputs' / 'spatial_generalization_eval_mts100_idx2_trainperiod_conservative'

TRAIN_START, TRAIN_END = config.TRAIN_START, config.TRAIN_END
DYNAMIC_VARS = ['Rainf', 'Tair', 'PotEvap']
TARGET_VAR = 'Streamflow'
ALL_VARS = DYNAMIC_VARS + [TARGET_VAR]
LOOKBACK_H = 168
LOOKBACK_D = 365
FREQ = 24
BATCH_SIZE = 256
NUM_WORKERS = config.NUM_WORKERS
PIN_MEMORY = config.PIN_MEMORY
BEST_VAL_KGE = 0.782548238948517
BEST_TEST_KGE = 0.7211588814761928


def prepare_split(full_ds: xr.Dataset, static_df: pd.DataFrame, scalers: dict, start: str, end: str):
    dyn = full_ds.sel(time=slice(start, end))
    dyn_forcing = dyn.sel(dynamic_forcing=DYNAMIC_VARS)
    target = dyn.sel(dynamic_forcing=TARGET_VAR)
    dyn_std, static_std, y_std = standardize_data(dyn_forcing, static_df, target, scalers)
    static_std, missing_static = _add_static_station_aliases(static_std, full_ds.data_vars)
    if missing_static:
        preview = ', '.join(missing_static[:10])
        suffix = ' ...' if len(missing_static) > 10 else ''
        raise KeyError(f'Missing static features for {len(missing_static)} stations: {preview}{suffix}')
    return dyn_std, static_std, y_std


def build_loader(dyn_std, y_std, static_std, start: str, end: str) -> DataLoader:
    dataset = MultiscaleLSTMDataset(
        dyn_std,
        y_std,
        static_std,
        lookback_hourly=LOOKBACK_H,
        lookback_daily=LOOKBACK_D,
        frequency_factor=FREQ,
        start_date=start,
        end_date=end,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def rename_metrics(metrics_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return metrics_df.rename(
        columns={
            'samples': f'{prefix}_samples',
            'score_status': f'{prefix}_score_status',
            'exclusion_reason': f'{prefix}_exclusion_reason',
            'nse': f'{prefix}_nse',
            'kge': f'{prefix}_kge',
        }
    )


def format_metric(value) -> str:
    if value == '' or pd.isna(value):
        return ''
    return f'{float(value):.6f}'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))))

    samples_df = pd.read_csv(
        SAMPLES_CSV,
        dtype={'station_id': str, 'scheme_code': str, 'scheme_label': str, 'koppen_class': str},
    )
    station_ids = samples_df['station_id'].tolist()

    static_df = pd.read_csv(STATIC_CSV, index_col=0)
    static_df.index = static_df.index.astype(str)

    with SCALER_PATH.open('rb') as fp:
        scalers = pickle.load(fp)

    print(f'[{time.strftime("%F %T")}] loading {len(station_ids)} stations', flush=True)
    station_arrays = {}
    for idx, station_id in enumerate(station_ids, start=1):
        with xr.open_dataset(RAW_DIR / f'{station_id}.nc') as ds:
            da = ds[ALL_VARS].to_array(dim='dynamic_forcing').transpose('DateTime', 'dynamic_forcing')
            da = da.rename({'DateTime': 'time'})
            da = da.assign_coords(dynamic_forcing=ALL_VARS)
            da.name = station_id
            station_arrays[station_id] = da.load()
        if idx % 10 == 0 or idx == len(station_ids):
            print(f'  loaded {idx}/{len(station_ids)} stations', flush=True)

    full_ds = xr.Dataset(station_arrays)
    full_ds = handle_extremes(full_ds, min_streamflow=0.0, max_streamflow=1000.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[{time.strftime("%F %T")}] device={device}', flush=True)

    model = sMTSLSTM(
        dyn_input_size=config.DYN_INPUT_SIZE,
        static_input_size=config.STATIC_INPUT_SIZE,
        hidden_size_daily=64,
        hidden_size_hourly=64,
        num_layers=1,
        dropout=0.4,
        frequency_factor=FREQ,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print(f'[{time.strftime("%F %T")}] preparing training split', flush=True)
    train_dyn_std, train_static_std, train_y_std = prepare_split(full_ds, static_df, scalers, TRAIN_START, TRAIN_END)
    print(f'[{time.strftime("%F %T")}] building training loader', flush=True)
    train_loader = build_loader(train_dyn_std, train_y_std, train_static_std, TRAIN_START, TRAIN_END)
    print(f'[{time.strftime("%F %T")}] evaluating training split', flush=True)
    train_df = rename_metrics(
        evaluate_per_station(model, train_loader, scalers, device, expected_stations=station_ids),
        'train',
    )

    preexisting_eval_cols = [
        col
        for col in [
            'train_samples',
            'train_reason',
            'train_score_status',
            'train_exclusion_reason',
            'train_nse',
            'train_kge',
        ]
        if col in samples_df.columns
    ]
    samples_meta = samples_df.drop(columns=preexisting_eval_cols)
    station_metrics = samples_meta.merge(train_df, on='station_id', how='left')

    for idx, row in enumerate(station_metrics.itertuples(index=False), start=1):
        if row.train_score_status == 'ok':
            print(
                f'[{idx:02d}/{len(station_metrics)}] {row.station_id} '
                f'trainKGE={row.train_kge:.4f} trainNSE={row.train_nse:.4f} '
                f'train_samples={row.train_samples}',
                flush=True,
            )
        else:
            print(
                f'[{idx:02d}/{len(station_metrics)}] {row.station_id} '
                f'train_status={row.train_score_status}:{row.train_exclusion_reason or "ok"} '
                f'train_samples={row.train_samples}',
                flush=True,
            )

    valid_station_metrics = station_metrics[station_metrics['train_score_status'].eq('ok')].copy()
    excluded_station_metrics = station_metrics[station_metrics['train_score_status'].ne('ok')].copy()

    region_counts = station_metrics.groupby(['scheme_code', 'scheme_label'], as_index=False).agg(
        n_total_stations=('station_id', 'count'),
    )
    region_valid_counts = valid_station_metrics.groupby(['scheme_code', 'scheme_label'], as_index=False).agg(
        n_valid_stations=('station_id', 'count'),
    )
    region_summary = valid_station_metrics.groupby(['scheme_code', 'scheme_label'], as_index=False).agg(
        median_train_kge=('train_kge', 'median'),
        median_train_nse=('train_nse', 'median'),
    )
    region_summary = region_counts.merge(region_valid_counts, on=['scheme_code', 'scheme_label'], how='left').merge(
        region_summary,
        on=['scheme_code', 'scheme_label'],
        how='left',
    )
    region_summary['n_valid_stations'] = region_summary['n_valid_stations'].fillna(0).astype(int)
    region_summary['n_excluded_stations'] = region_summary['n_total_stations'] - region_summary['n_valid_stations']

    region_climate_counts = station_metrics.groupby(['scheme_code', 'scheme_label', 'koppen_class'], as_index=False).agg(
        n_total_stations=('station_id', 'count'),
    )
    region_climate_valid_counts = valid_station_metrics.groupby(
        ['scheme_code', 'scheme_label', 'koppen_class'],
        as_index=False,
    ).agg(
        n_valid_stations=('station_id', 'count'),
    )
    region_climate_summary = valid_station_metrics.groupby(
        ['scheme_code', 'scheme_label', 'koppen_class'],
        as_index=False,
    ).agg(
        median_train_kge=('train_kge', 'median'),
        median_train_nse=('train_nse', 'median'),
    )
    region_climate_summary = region_climate_counts.merge(
        region_climate_valid_counts,
        on=['scheme_code', 'scheme_label', 'koppen_class'],
        how='left',
    ).merge(
        region_climate_summary,
        on=['scheme_code', 'scheme_label', 'koppen_class'],
        how='left',
    )
    region_climate_summary['n_valid_stations'] = region_climate_summary['n_valid_stations'].fillna(0).astype(int)
    region_climate_summary['n_excluded_stations'] = (
        region_climate_summary['n_total_stations'] - region_climate_summary['n_valid_stations']
    )

    for col in ['median_train_kge', 'median_train_nse']:
        region_summary[col] = region_summary[col].where(region_summary[col].notna(), '')
        region_climate_summary[col] = region_climate_summary[col].where(region_climate_summary[col].notna(), '')

    station_csv = OUT_DIR / 'per_station_metrics.csv'
    excluded_station_csv = OUT_DIR / 'per_station_exclusions.csv'
    region_csv = OUT_DIR / 'per_region_medians.csv'
    region_climate_csv = OUT_DIR / 'per_region_climate_medians.csv'
    summary_md = OUT_DIR / 'summary.md'

    valid_station_metrics.to_csv(station_csv, index=False)
    excluded_station_metrics.to_csv(excluded_station_csv, index=False)
    region_summary.to_csv(region_csv, index=False)
    region_climate_summary.to_csv(region_climate_csv, index=False)

    lines = [
        'Spatial Generalization Evaluation on Training Period Using Best 100 Station Tuned MTSLSTM',
        '',
        'Selected by validation median KGE from mts100_tune.',
        f'Best run: {RUN_DIR.name}',
        f'Tuned validation median KGE: {BEST_VAL_KGE:.15f}',
        f'Tuned test median KGE: {BEST_TEST_KGE:.15f}',
        f'Model path: {MODEL_PATH}',
        f'Scaler path: {SCALER_PATH}',
        f'Samples csv: {SAMPLES_CSV}',
        f'Static csv: {STATIC_CSV}',
        f'Training period: {TRAIN_START} to {TRAIN_END}',
        'Evaluation path: MultiscaleLSTMDataset plus Train.evaluate_per_station',
        '',
        'Regional medians',
        '',
        'scheme_code,scheme_label,n_total_stations,n_valid_stations,n_excluded_stations,median_train_kge,median_train_nse',
    ]
    for row in region_summary.itertuples(index=False):
        lines.append(
            f'{row.scheme_code},{row.scheme_label},{row.n_total_stations},{row.n_valid_stations},{row.n_excluded_stations},{format_metric(row.median_train_kge)},{format_metric(row.median_train_nse)}'
        )
    summary_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('\nRegional medians:', flush=True)
    print(region_summary.to_string(index=False), flush=True)
    print(f'\nSaved per station metrics to {station_csv}', flush=True)
    print(f'Saved per station exclusions to {excluded_station_csv}', flush=True)
    print(f'Saved per region medians to {region_csv}', flush=True)
    print(f'Saved per region climate medians to {region_climate_csv}', flush=True)
    print(f'Saved summary to {summary_md}', flush=True)


if __name__ == '__main__':
    main()
