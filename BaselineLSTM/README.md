# BaselineLSTM 15-station archive

## Best run
- run id: `6bz7arav`
- run dir: `runs/20260314/BaselineLSTM/6bz7arav_lr0.0001_bs256_lb8760_hs256_do0.4_lossnse_loss`
- train median NSE: `0.8641952099561759`
- train median KGE: `0.8527882610216142`

## Folder guide
- `code/`: core model, training, loss, data-loading, and inference code
- `tuning/`: grid/sweep scripts and parameter files used for the 15-station search
- `runs/`: copied model artifacts from the 15-station tuning runs
- `logs/tuning/`: grid, array, and sweep submission logs
- `logs/metrics/`: best-run train metric logs
- `logs/inference_jobs/`: copied inference job logs from the original directory
- `wandb_best_run/`: W&B config, summary, and raw run files for the selected best run
- `metadata/data_references.txt`: original dataset and static-attribute paths
