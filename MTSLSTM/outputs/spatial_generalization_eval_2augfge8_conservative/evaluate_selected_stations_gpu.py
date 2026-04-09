from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

ROOT = Path('/home/kongw0a/MTS_LSTM/experiment_withcursor')
MTS_DIR = ROOT / 'MTSLSTM'
sys.path.insert(0, str(MTS_DIR))

import config
from Modelzoo import sMTSLSTM
from Train import evaluate_per_station
from loder import MultiscaleLSTMDataset, handle_extremes, standardize_data

RUN_DIR = MTS_DIR / 'runs' / '20260314' / 'MTSLSTM' / '2augfge8_hd64-64_bs128_H72_D365_do0.4_lossnse_loss_reg1.0_sch1-5e-4-10-1e-4-25-5e-5'
MODEL_PATH = RUN_DIR / 'best_model.pth'
SCALER_PATH = RUN_DIR / 'scalers.pkl'
SAMPLES_CSV = ROOT / 'station_maps' / 'outputs' / 'spatial_generalization_station_samples_conservative.csv'
STATIC_CSV = ROOT / 'data' / 'static_h.csv'
RAW_DIR = Path('/ibex/project/c2266/wkkong/data/CAEMLSH/data_workstation/CAMELSH/timeseries/Data/CAMELSH/timeseries')
OUT_DIR = MTS_DIR / 'outputs' / 'spatial_generalization_eval_2augfge8_conservative'

VAL_START, VAL_END = '2003-10-01', '2008-09-30'
TEST_START, TEST_END = '2008-10-01', '2015-09-30'
DYNAMIC_VARS = ['Rainf', 'Tair', 'PotEvap']
TARGET_VAR = 'Streamflow'
ALL_VARS = DYNAMIC_VARS + [TARGET_VAR]
LOOKBACK_H = 72
LOOKBACK_D = 365
FREQ = 24
BATCH_SIZE = 256
NUM_WORKERS = config.NUM_WORKERS
PIN_MEMORY = config.PIN_MEMORY


def prepare_split(full_ds: xr.Dataset, static_df: pd.DataFrame, scalers: dict, start: str, end: str):
    dyn = full_ds.sel(time=slice(start, end))
    dyn_forcing = dyn.sel(dynamic_forcing=DYNAMIC_VARS)
    target = dyn.sel(dynamic_forcing=TARGET_VAR)
    return standardize_data(dyn_forcing, static_df, target, scalers)


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
    if value == '':
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

    static_df = pd.read_csv(STATIC_CSV, dtype={'STAID': str}).set_index('STAID')
    static_df.index = static_df.index.astype(str)
    static_df = static_df.loc[station_ids]

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

    print(f'[{time.strftime("%F %T")}] preparing validation split', flush=True)
    val_dyn_std, val_static_std, val_y_std = prepare_split(full_ds, static_df, scalers, VAL_START, VAL_END)
    print(f'[{time.strftime("%F %T")}] building validation loader', flush=True)
    val_loader = build_loader(val_dyn_std, val_y_std, val_static_std, VAL_START, VAL_END)
    print(f'[{time.strftime("%F %T")}] evaluating validation split', flush=True)
    val_df = rename_metrics(
        evaluate_per_station(model, val_loader, scalers, device, expected_stations=station_ids),
        'val',
    )

    print(f'[{time.strftime("%F %T")}] preparing test split', flush=True)
    test_dyn_std, test_static_std, test_y_std = prepare_split(full_ds, static_df, scalers, TEST_START, TEST_END)
    print(f'[{time.strftime("%F %T")}] building test loader', flush=True)
    test_loader = build_loader(test_dyn_std, test_y_std, test_static_std, TEST_START, TEST_END)
    print(f'[{time.strftime("%F %T")}] evaluating test split', flush=True)
    test_df = rename_metrics(
        evaluate_per_station(model, test_loader, scalers, device, expected_stations=station_ids),
        'test',
    )

    preexisting_eval_cols = [
        col
        for col in [
            'val_samples',
            'test_samples',
            'val_reason',
            'test_reason',
            'val_score_status',
            'test_score_status',
            'val_exclusion_reason',
            'test_exclusion_reason',
            'val_nse',
            'val_kge',
            'test_nse',
            'test_kge',
        ]
        if col in samples_df.columns
    ]
    samples_meta = samples_df.drop(columns=preexisting_eval_cols)
    station_metrics = samples_meta.merge(val_df, on='station_id', how='left').merge(test_df, on='station_id', how='left')

    for idx, row in enumerate(station_metrics.itertuples(index=False), start=1):
        if row.val_score_status == 'ok' and row.test_score_status == 'ok':
            print(
                f'[{idx:02d}/{len(station_metrics)}] {row.station_id} '
                f'valKGE={row.val_kge:.4f} valNSE={row.val_nse:.4f} '
                f'testKGE={row.test_kge:.4f} testNSE={row.test_nse:.4f} '
                f'val_samples={row.val_samples} test_samples={row.test_samples}',
                flush=True,
            )
        else:
            print(
                f'[{idx:02d}/{len(station_metrics)}] {row.station_id} '
                f'val_status={row.val_score_status}:{row.val_exclusion_reason or "ok"} '
                f'test_status={row.test_score_status}:{row.test_exclusion_reason or "ok"} '
                f'val_samples={row.val_samples} test_samples={row.test_samples}',
                flush=True,
            )

    valid_station_metrics = station_metrics[
        (station_metrics['val_score_status'] == 'ok') & (station_metrics['test_score_status'] == 'ok')
    ].copy()
    excluded_station_metrics = station_metrics[
        (station_metrics['val_score_status'] != 'ok') | (station_metrics['test_score_status'] != 'ok')
    ].copy()

    region_counts = station_metrics.groupby(['scheme_code', 'scheme_label'], as_index=False).agg(
        n_total_stations=('station_id', 'count'),
    )
    region_valid_counts = valid_station_metrics.groupby(['scheme_code', 'scheme_label'], as_index=False).agg(
        n_valid_stations=('station_id', 'count'),
    )
    region_summary = valid_station_metrics.groupby(['scheme_code', 'scheme_label'], as_index=False).agg(
        median_val_kge=('val_kge', 'median'),
        median_val_nse=('val_nse', 'median'),
        median_test_kge=('test_kge', 'median'),
        median_test_nse=('test_nse', 'median'),
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
        median_val_kge=('val_kge', 'median'),
        median_val_nse=('val_nse', 'median'),
        median_test_kge=('test_kge', 'median'),
        median_test_nse=('test_nse', 'median'),
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

    metric_cols = ['median_val_kge', 'median_val_nse', 'median_test_kge', 'median_test_nse']
    for col in metric_cols:
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
        '# Spatial Generalization Evaluation Using Best Tuned MTSLSTM',
        '',
        f'- best run: `{RUN_DIR.name}`',
        f'- model path: `{MODEL_PATH}`',
        f'- samples csv: `{SAMPLES_CSV}`',
        f'- validation period: `{VAL_START}` to `{VAL_END}`',
        f'- test period: `{TEST_START}` to `{TEST_END}`',
        '- evaluation path: reuse `MultiscaleLSTMDataset` + `Train.evaluate_per_station`',
        '',
        '## Regional medians',
        '',
        '| scheme_code | scheme_label | n_total_stations | n_valid_stations | n_excluded_stations | median_val_kge | median_val_nse | median_test_kge | median_test_nse |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in region_summary.itertuples(index=False):
        lines.append(
            f'| {row.scheme_code} | {row.scheme_label} | {row.n_total_stations} | {row.n_valid_stations} | {row.n_excluded_stations} | {format_metric(row.median_val_kge)} | {format_metric(row.median_val_nse)} | {format_metric(row.median_test_kge)} | {format_metric(row.median_test_nse)} |'
        )
    summary_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('\nRegional medians:', flush=True)
    print(region_summary.to_string(index=False), flush=True)
    print(f'\nSaved per-station metrics to {station_csv}', flush=True)
    print(f'Saved per-station exclusions to {excluded_station_csv}', flush=True)
    print(f'Saved per-region medians to {region_csv}', flush=True)
    print(f'Saved per-region climate medians to {region_climate_csv}', flush=True)
    print(f'Saved markdown summary to {summary_md}', flush=True)


if __name__ == '__main__':
    main()
