# MTSLSTM 100-station archive

This directory is a reorganized copy of the local 100-station MTSLSTM experiment artifacts from `/home/kongw0a/MTS_LSTM/experiment_withcursor`.
The original files under `experiment_withcursor` were not modified.

## Scope

This archive only covers the 100-station tuning and the follow-up spatial generalization evaluations.
The 500-station experiments are intentionally not included here.

## Best run

- selected by `valKGE`: `idx2`
- hyperparameters: `hidden_size=64`, `batch_size=128`, `lookback_hourly=168`, `lookback_daily=365`, `dropout=0.4`
- validation: `valKGE=0.782548238948517`, `valNSE=0.7179507274017882`
- test: `testKGE=0.7211588814761928`, `testNSE=0.6938193376278703`

## Folder guide

- `code/`: core training, model, loader, loss, and inference code used by the 100-station jobs
- `tuning/`: 100-station submission script plus the parameter table and supporting scripts
- `logs/tuning/`: local `mts100_tune_*.out/.err` logs
- `logs/spatial_eval/`: local `spatial_eval_mts100_*.out/.err` logs
- `outputs/`: copied spatial generalization outputs for the selected `idx2` model
- `metadata/`: copied local static attribute csv and conservative station sample csv, plus external path notes
- `summaries/`: generated summary table for the completed `46452569` tuning array

## Notes

- `tuning/tuning_500stations_prevbest.tsv` is kept because it is the parameter table referenced by `submit_mtslstm_100stations_tuning_array_v100.sbatch` for this 100-station run.
- The actual 100-station training subset NetCDF files and the centralized training run artifacts live outside `experiment_withcursor`; their paths are recorded in `metadata/external_paths.txt`.
