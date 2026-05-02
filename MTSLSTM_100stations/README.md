# MTSLSTM 100-station archive plus 30-station follow-up experiments

This directory now contains two related groups of artifacts:

1. the original 100-station MTSLSTM tuning archive and spatial generalization evaluation
2. the later 30-station follow-up experiments built around daily LSTM, SymTorch distillation, transfer learning, and transfer learning with symbolic priors

## 100-station archive

This part is the reorganized copy of the local 100-station MTSLSTM experiment artifacts from `/home/kongw0a/MTS_LSTM/experiment_withcursor`.
The original files under `experiment_withcursor` were not modified.

Best 100-station run:
- selected by `valKGE`: `idx2`
- hyperparameters: `hidden_size=64`, `batch_size=128`, `lookback_hourly=168`, `lookback_daily=365`, `dropout=0.4`
- validation: `valKGE=0.782548238948517`, `valNSE=0.7179507274017882`
- test: `testKGE=0.7211588814761928`, `testNSE=0.6938193376278703`

100-station folders:
- `code/`: core training, model, loader, loss, and inference code used by the 100-station jobs
- `tuning/`: 100-station submission script plus the parameter table and supporting scripts
- `logs/tuning/`: local `mts100_tune_*.out/.err` logs
- `logs/spatial_eval/`: local `spatial_eval_mts100_*.out/.err` logs
- `metadata/`: copied local static attribute csv and conservative station sample csv, plus external path notes
- `summaries/`: generated summary table for the completed `46452569` tuning array
- `training_runs/`: stored checkpoints and final models for the 100-station tuning jobs

Notes:
- `tuning/tuning_500stations_prevbest.tsv` is kept because it is the parameter table referenced by `submit_mtslstm_100stations_tuning_array_v100.sbatch` for this 100-station run.
- The actual 100-station training subset NetCDF files and the centralized training run artifacts live outside `experiment_withcursor`; their paths are recorded in `metadata/external_paths.txt`.

## 30-station follow-up experiments

The repository now also keeps the full 30-station experiment chain under `outputs/` and the matching scripts under `code/`.

Main experiment families:
- daily baseline LSTM on the selected 30 stations
- daily teacher LSTM for symbolic-prior work
- SymTorch direct and residual distillation
- daily-to-hourly transfer learning
- daily-to-hourly transfer learning with symbolic priors
- three-way evaluation and presentation plots

Start here for the curated map:
- `EXPERIMENTS_30STATION_GUIDE.md`

Most important 30-station output folders:
- `outputs/s2_random30_idx2_eval/`
- `outputs/baseline_lstm_daily_s2_random30_tuning/`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_hydro_valkge/`
- `outputs/symtorch_daily_teacher_s2_random30/`
- `outputs/transfer_daily_head_tune_s2_random30/`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30/`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid/`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.05/`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.1/`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.2/`

Most important 30-station scripts:
- `code/tune_baseline_lstm_daily_s2_random30.py`
- `code/train_symtorch_daily_teacher_s2_random30.py`
- `code/export_best_daily_lstm_for_symtorch.py`
- `code/distill_best_daily_lstm_with_symtorch.py`
- `code/distill_daily_lstm_obs_residual_with_symtorch.py`
- `code/distill_daily_lstm_hydro_residual_with_symtorch.py`
- `code/transfer_daily_head_tune_s2_random30.py`
- `code/transfer_daily_to_hourly_partial_ft_s2_random30.py`
- `code/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid.py`
