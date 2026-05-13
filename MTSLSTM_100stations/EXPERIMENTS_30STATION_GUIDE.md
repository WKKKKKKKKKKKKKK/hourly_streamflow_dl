# 30-station Daily LSTM, SymTorch, Transfer, and Symbolic-Prior Guide

This guide organizes the 30-station follow-up experiments stored under `MTSLSTM_100stations`.
It is meant to answer one practical question quickly: where are the code, trained models, and result folders for each experiment family?

## 1. Daily LSTM on the selected 30 stations

Purpose:
Train the daily baseline LSTM used later by SymTorch and transfer-learning experiments.

Station subset:
- `outputs/s2_random30_idx2_eval/selected_stations.csv`
- `outputs/s2_random30_idx2_eval/summary.md`

Primary code:
- `code/tune_baseline_lstm_daily_s2_random30.py`

Key outputs:
- `outputs/baseline_lstm_daily_s2_random30_tuning/`
- `outputs/baseline_lstm_daily_s2_random30_tuning/best_run_by_val_kge.json`
- `outputs/baseline_lstm_daily_s2_random30_tuning/summary_by_val_kge.md`
- `outputs/baseline_lstm_daily_s2_random30_tuning/runs/`

Best model selected for downstream use:
- `outputs/baseline_lstm_daily_s2_random30_tuning/runs/idx9_lr0.001_bs128_lb90_hs128_do0.2_lossnse_loss/best_model.pth`

Supporting artifacts kept in the run folders:
- `model.pth`
- `checkpoint.pth`
- `scalers.pkl`
- `training_history.csv`
- `summary.md`
- `per_station_metrics.csv`

## 2. SymTorch distillation of the daily LSTM

Purpose:
Export the best daily LSTM, then distill symbolic equations against either the raw daily-LSTM output or residual targets.

Reference notebook:
- `Sym_lstm/MJO_prediction.ipynb`

Primary code:
- `code/export_best_daily_lstm_for_symtorch.py`
- `code/distill_best_daily_lstm_with_symtorch.py`
- `code/distill_daily_lstm_obs_residual_with_symtorch.py`
- `code/distill_daily_lstm_hydro_residual_with_symtorch.py`
- `code/fuse_hydro_log_symbolic_residuals.py`

Direct-export folder used by SymTorch:
- `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/`

Key export artifacts:
- `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/export_metadata.json`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/variable_names.json`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/train_distill_sample.npz`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/train_full.npz`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/val_full.npz`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/test_full.npz`

Direct symbolic distillation results:
- `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/symtorch_direct_distill/`

Observed-minus-dailyLSTM residual distillation results:
- `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/symtorch_obs_minus_dailylstm_residual/`

Hydrology-informed symbolic residual results:
- `outputs/baseline_lstm_daily_s2_random30_symtorch_hydro_valkge/`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_hydro_valkge/hydro_rawstd_residual/`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_hydro_valkge/hydro_log_residual/`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_hydro_valkge/hydro_log_residual_eventq75/`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_hydro_valkge/hybrid_log_residual_smoothgate/`

Useful summary files:
- `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/symtorch_direct_distill/summary.md`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/symtorch_obs_minus_dailylstm_residual/summary.md`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_hydro_valkge/summary.md`
- `outputs/baseline_lstm_daily_s2_random30_symtorch_hydro_valkge/experiment_comparison.csv`

## 3. Transfer learning from daily to hourly

Purpose:
Use the daily-model knowledge to initialize and fine-tune the hourly model on the same 30-station subset.

Primary code:
- `code/transfer_daily_to_hourly_partial_ft_s2_random30.py`
- `code/evaluate_s2_random30_idx2.py`
- `code/plot_s2_random30_baseline_vs_transfer.py`

Main output folder:
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30/`

Key model file:
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30/best_transfer_model.pth`

Transfer-learning artifacts:
- `method_notes.md`
- `training_history.csv`
- `summary.md`
- `summary_hourly_metrics.csv`
- `per_station_hourly_metrics.csv`

Related comparison outputs:
- `outputs/s2_random30_idx2_eval/`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30/baseline_vs_transfer_plots/`

## 4. Transfer learning with symbolic prior

Purpose:
Inject symbolic-prior information into the daily-to-hourly transfer model and compare different symbolic weights.

Primary code:
- `code/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid.py`

Main result folders:
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid/`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.05/`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.1/`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.2/`

Key model files:
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid/best_transfer_model.pth`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.05/best_transfer_model.pth`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.1/best_transfer_model.pth`
- `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.2/best_transfer_model.pth`

Typical artifacts in each symbolic-transfer folder:
- `run_metadata.json`
- `training_history.csv`
- `summary.md`
- `summary_hourly_metrics.csv`
- `per_station_hourly_metrics.csv`

## 5. Daily-head-only transfer-learning ablation

Purpose:
Compare ordinary transfer learning and symbolic-prior transfer learning when only the daily output head is fine-tuned.
This is a daily-branch ablation, not the main hourly forecast transfer experiment.

Frozen parameters:
- `lstm_daily`
- `lstm_hourly`
- `transfer_h`
- `transfer_c`
- `head_hourly`

Trainable parameters:
- `head_daily`

Primary code:
- `code/transfer_daily_head_tune_s2_random30.py`
- `code/transfer_daily_head_tune_symbolic_s2_random30.py`

Main output folder:
- `outputs/head_only_daily_head_fc_transfer_s2_random30/`

Result folders:
- `outputs/head_only_daily_head_fc_transfer_s2_random30/ordinary_transfer/`
- `outputs/head_only_daily_head_fc_transfer_s2_random30/symbolic_prior_sw0.05/`

Key model files:
- `outputs/head_only_daily_head_fc_transfer_s2_random30/ordinary_transfer/best_transfer_model.pth`
- `outputs/head_only_daily_head_fc_transfer_s2_random30/symbolic_prior_sw0.05/best_transfer_model.pth`
- `outputs/head_only_daily_head_fc_transfer_s2_random30/ordinary_transfer/trials/*.pth`
- `outputs/head_only_daily_head_fc_transfer_s2_random30/symbolic_prior_sw0.05/trials/*.pth`

Summary files:
- `outputs/head_only_daily_head_fc_transfer_s2_random30/README.md`
- `outputs/head_only_daily_head_fc_transfer_s2_random30/comparison_summary.csv`
- `outputs/head_only_daily_head_fc_transfer_s2_random30/ordinary_transfer/summary.md`
- `outputs/head_only_daily_head_fc_transfer_s2_random30/symbolic_prior_sw0.05/summary.md`

Best DAILY metrics:

| Method | Best lr | Weight decay | Symbolic weight | Best epoch | Val KGE | Val NSE | Test KGE | Test NSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordinary head-only transfer | 0.005 | 0.0001 | NA | 13 | 0.184845 | -0.037005 | 0.206447 | 0.107114 |
| Symbolic-prior head-only transfer | 0.001 | 0.0 | 0.05 | 14 | 0.183022 | -0.011178 | 0.134518 | 0.029555 |

Reference daily-branch transfer result from the original daily-to-hourly experiment:

| Method | Output folder | Trainable modules | Frozen modules | Metric scale | Best epoch | Val KGE | Val NSE | Test KGE | Test NSE |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| Daily-branch transfer, hourly branch frozen | `outputs/transfer_daily_to_hourly_partial_ft_s2_random30/` | `lstm_daily`, `transfer_h`, `transfer_c`, `head_daily`, `head_hourly` | `lstm_hourly` | HOURLY | 10 | 0.524135 | 0.224185 | 0.489140 | 0.252204 |

## 6. Evaluation and presentation plots for three-way comparisons

Purpose:
Compare baseline hourly, transfer-learning, and symbolic-transfer results.

Primary code:
- `code/plot_s2_random30_threeway_ppt.py`
- `code/evaluate_s2_random30_alt_threeway.py`
- `code/evaluate_s1_csb_ca_threeway.py`
- `code/evaluate_u2_dfb_e_gl_threeway.py`
- `code/plot_three_method_peak_lag_cdfs.py`

Plot output folders:
- `outputs/s2_random30_threeway_ppt_plots/`
- `outputs/s2_random30_alt_threeway_ppt_plots/`
- `outputs/s1_csb_ca_threeway_ppt_plots/`
- `outputs/u2_dfb_e_gl_threeway_ppt_plots/`

## 7. Fast navigation by experiment family

If you only need the most important entry points:

- Daily baseline LSTM:
  `code/tune_baseline_lstm_daily_s2_random30.py`
  `outputs/baseline_lstm_daily_s2_random30_tuning/`

- SymTorch direct distillation:
  `code/export_best_daily_lstm_for_symtorch.py`
  `code/distill_best_daily_lstm_with_symtorch.py`
  `outputs/baseline_lstm_daily_s2_random30_symtorch_direct_valkge/`

- SymTorch residual distillation:
  `code/distill_daily_lstm_obs_residual_with_symtorch.py`
  `code/distill_daily_lstm_hydro_residual_with_symtorch.py`
  `outputs/baseline_lstm_daily_s2_random30_symtorch_hydro_valkge/`

- Transfer learning:
  `code/transfer_daily_to_hourly_partial_ft_s2_random30.py`
  `outputs/transfer_daily_to_hourly_partial_ft_s2_random30/`

- Transfer learning with symbolic prior:
  `code/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid.py`
  `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.05/`
  `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.1/`
  `outputs/transfer_daily_to_hourly_partial_ft_s2_random30_symbolic_hybrid_sw0.2/`

- Daily-head-only transfer ablation:
  `code/transfer_daily_head_tune_s2_random30.py`
  `code/transfer_daily_head_tune_symbolic_s2_random30.py`
  `outputs/head_only_daily_head_fc_transfer_s2_random30/`
