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
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30/`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid/`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.05/`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.1/`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.2/`

Most important 30-station scripts:
- `code/tune_baseline_lstm_daily_s2_random30.py`
- `code/export_best_daily_lstm_for_symtorch.py`
- `code/distill_best_daily_lstm_with_symtorch.py`
- `code/distill_daily_lstm_obs_residual_with_symtorch.py`
- `code/distill_daily_lstm_hydro_residual_with_symtorch.py`
- `code/transfer_daily_to_hourly_partial_ft_s2_random30.py`
- `code/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid.py`

## Significance and degradation analyses

The S2 Cfa-SE three-model comparison now includes two reproducible analysis steps.

### Station-level paired significance tests

Run:

```bash
python code/run_s2_threeway_significance_tests.py
```

This script compares the station-level KGE/NSE scores for:
- ordinary transfer learning vs. baseline
- symbolic-prior transfer learning vs. baseline
- symbolic-prior transfer learning vs. ordinary transfer learning

The test is a one-sided paired Wilcoxon signed-rank test on matched station-level
differences, e.g. `transfer - baseline > 0`. Paired t-test p-values are also
reported as a sensitivity check. Benjamini-Hochberg FDR correction is applied
within each comparison family.

Inputs:
- `outputs/s2_random30_idx2_eval/per_station_metrics.csv`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30/per_station_hourly_metrics.csv`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.05/per_station_hourly_metrics.csv`

Outputs:
- `outputs/s2_random30_threeway_ppt_plots/significance_tests_vs_baseline.csv`
- `outputs/s2_random30_threeway_ppt_plots/symbolic_vs_transfer_significance_tests.csv`
- `outputs/s2_random30_threeway_ppt_plots/s2_threeway_all_paired_significance_tests.csv`

### Source-domain transfer degradation tests

First evaluate the original source model, the ordinary transfer model, and the
symbolic-prior transfer model on the source-domain stations:

```bash
python code/evaluate_transfer_on_source_domain.py --force
```

Then run the degradation post-processing:

```bash
python code/analyze_source_domain_transfer_degradation.py
python code/plot_source_domain_kge_vs_mean_flow.py
```

The degradation analysis quantifies source-domain forgetting after transfer,
especially at low-flow stations. Key derived flags include:
- `retained_close_kge`: transfer KGE is within 0.1 of the original source model
- `retained_usable`: KGE >= 0.5 and NSE >= 0.5
- `failure_kge_lt_0`: KGE < 0
- `catastrophic_kge_lt_minus1`: KGE < -1
- `big_forgetting_delta_lt_minus05`: transfer KGE - source KGE < -0.5

Important outputs:
- `outputs/source_domain_transfer_retention_eval/per_station_source_domain_metrics.csv`
- `outputs/source_domain_transfer_retention_eval/per_station_source_domain_metrics_with_flow.csv`
- `outputs/source_domain_transfer_retention_eval/lowflow_failure_by_quartile.csv`
- `outputs/source_domain_transfer_retention_eval/retention_failure_counts.csv`
- `outputs/source_domain_transfer_retention_eval/kge_vs_flow_test_binned_summary.csv`
- `outputs/source_domain_transfer_retention_eval/figures/source_domain_test_kge_vs_mean_flow_three_models.png`
