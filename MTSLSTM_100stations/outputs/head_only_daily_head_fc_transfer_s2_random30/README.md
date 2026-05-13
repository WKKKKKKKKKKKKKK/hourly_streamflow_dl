# Daily-Head-Only Transfer Learning on S2 Random 30 Stations

This folder keeps the transfer-learning ablation where only the daily branch output head is trainable.
It is intentionally separate from the full daily-to-hourly transfer folders so the head-only results and models are not mixed with the main transfer experiments.

Shared setup:
- source model: `training_runs/20260407_mtslstm_100stations_tuning_topo18_v100/idx2_bs128_do0.4_hs64_H168_D365/best_model.pth`
- selected stations: `outputs/s2_random30_idx2_eval/selected_stations.csv`
- frozen parameters: `lstm_daily`, `lstm_hourly`, `transfer_h`, `transfer_c`, `head_hourly`
- trainable parameters: `head_daily`
- final metrics: daily KGE/NSE

## Folder Layout

- `ordinary_transfer/`: ordinary transfer learning with only `head_daily` fine-tuned
- `symbolic_prior_sw0.05/`: same head-only transfer with the hybrid symbolic prior loss, `sym_loss_weight=0.05`
- `comparison_summary.csv`: compact side-by-side summary of best trials and train/val/test scores, including the original daily-branch transfer reference row

Each result subfolder includes:
- `best_transfer_model.pth`
- `trials/*.pth`
- `trial_summary.csv`
- `summary_metrics.csv`
- `per_station_metrics.csv`
- `run_metadata.json`
- `summary.md`

The generated `trial_summary.csv` and `run_metadata.json` files preserve the original absolute run paths for provenance. The uploaded model checkpoints are stored in the neighboring `best_transfer_model.pth` and `trials/` files inside each subfolder.

## Best Results

| Method | Metric scale | Best lr | Weight decay | Symbolic weight | Best epoch | Train KGE | Train NSE | Val KGE | Val NSE | Test KGE | Test NSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordinary head-only transfer | DAILY | 0.005 | 0.0001 | NA | 13 | 0.267977 | 0.189181 | 0.184845 | -0.037005 | 0.206447 | 0.107114 |
| Symbolic-prior head-only transfer | DAILY | 0.001 | 0.0 | 0.05 | 14 | 0.220719 | 0.091654 | 0.183022 | -0.011178 | 0.134518 | 0.029555 |

In this head-only setting, the ordinary transfer run has the better test KGE/NSE. The symbolic-prior run has a slightly lower validation KGE but a less negative validation NSE.

## Reference Daily-Branch Transfer Result

The original daily-to-hourly transfer experiment is kept in `outputs/transfer_daily_to_hourly_partial_ft_s2_random30/`.
It freezes `lstm_hourly` and fine-tunes `lstm_daily`, `transfer_h`, `transfer_c`, `head_daily`, and `head_hourly`.
Its final scores are hourly KGE/NSE, so the row below is a transfer-strategy reference rather than another daily-head score.

| Method | Metric scale | Best lr | Weight decay | Best epoch | Train KGE | Train NSE | Val KGE | Val NSE | Test KGE | Test NSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Daily-branch transfer, hourly branch frozen | HOURLY | 0.0005 | 0.00001 | 10 | 0.784762 | 0.720489 | 0.524135 | 0.224185 | 0.489140 | 0.252204 |
