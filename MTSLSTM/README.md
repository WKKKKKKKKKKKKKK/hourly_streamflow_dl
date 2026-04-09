# MTSLSTM 15-station archive

## Best run
- run id: `2augfge8`
- run dir: `runs/20260314/MTSLSTM/2augfge8_hd64-64_bs128_H72_D365_do0.4_lossnse_loss_reg1.0_sch1-5e-4-10-1e-4-25-5e-5`
- train median NSE: `0.912072488841589`
- train median KGE: `0.9257960694949389`

## Folder guide
- `code/`: core model, training, loss, data-loading, and inference code
- `tuning/`: grid/sweep scripts and parameter files used for the 15-station search
- `runs/`: copied model artifacts from the 15-station tuning runs
- `logs/tuning/`: grid, array, and sweep submission logs
- `logs/metrics/`: best-run train metric logs
- `logs/inference_jobs/`: copied inference job logs from the original directory
- `logs/spatial_eval/`: copied spatial generalization job logs for `2augfge8`
- `outputs/`: copied spatial generalization scripts and csv outputs for `2augfge8` and `2augfge8_conservative`
- `wandb_best_run/`: W&B config, summary, and raw run files for the selected best run
- `metadata/data_references.txt`: original dataset and static-attribute paths
