# Daily-to-Hourly Transfer Learning Notes

## Goal

This experiment adapts the best 100-station `MTSLSTM` source model to the selected 30 S2 stations under the assumption that the target stations do **not** have hourly observations during transfer learning.

The target stations are treated as having only **daily observations**, which are obtained by aggregating their hourly streamflow series.  
Even though the transfer supervision is daily, the final model must still produce **hourly outputs**, and final evaluation is reported with **hourly KGE/NSE**.

## Source Model

- Source model: `idx2_bs128_do0.4_hs64_H168_D365`
- Path:
  `/home/kongw0a/hourly_streamflow_dl/MTSLSTM_100stations/training_runs/20260407_mtslstm_100stations_tuning_topo18_v100/idx2_bs128_do0.4_hs64_H168_D365/best_model.pth`

## Target Stations

- Target subset: 30 randomly selected S2 stations
- Station list:
  `/home/kongw0a/hourly_streamflow_dl/MTSLSTM_100stations/outputs/s2_random30_idx2_eval/selected_stations.csv`

## Main Idea

The transfer learning logic is:

1. Keep the original `MTSLSTM` architecture with both daily and hourly branches.
2. Build target-domain daily supervision from hourly streamflow by aggregating 24 hours into one daily value.
3. Fine-tune the **daily branch** so that it learns the target-domain daily behavior.
4. Allow that daily adaptation to propagate to the hourly branch through the model's internal state-transfer layers.
5. Evaluate the adapted model only on **hourly outputs**.

This is not a pure `head_daily` fine-tuning setup.  
If only `head_daily` were updated, hourly outputs would remain unchanged.  
Instead, this experiment updates the daily pathway and the daily-to-hourly connection so that hourly predictions can change.

## How Daily Supervision Is Constructed

Inside `transfer_daily_to_hourly_partial_ft_s2_random30.py`:

- A window of hourly inputs is extracted for the hourly branch.
- A longer hourly block is extracted and reshaped into `(365 days, 24 hours, features)`.
- The 24 hours inside each day are averaged to create daily inputs for the daily branch.

Conceptually:

- `x_h`: hourly branch input
- `x_d_full`: hourly-resolution block used to construct daily input
- `x_d`: day-averaged input actually fed into the daily branch

For the labels:

- `y_h`: hourly streamflow at the prediction time step
- `y_d`: daily streamflow label formed by averaging the last 24 hourly streamflow values

So the target-domain daily supervision is derived from hourly observations, but during transfer learning it is treated as if only the daily label were available.

## Trainable Modules

The model is initialized from the source pretrained weights and then partially fine-tuned.

Frozen:

- `lstm_hourly`

Trainable:

- `lstm_daily`
- `transfer_h`
- `transfer_c`
- `head_daily`
- `head_hourly`

Why:

- `lstm_daily` must adapt to target-domain daily dynamics.
- `transfer_h` and `transfer_c` must propagate that adaptation into the hourly branch.
- `head_daily` must fit the target daily supervision.
- `head_hourly` is allowed to adjust the final hourly readout.
- `lstm_hourly` is kept frozen to preserve the source-domain hourly representation and reduce overfitting risk.

## Loss Design

The training loss has two parts.

### 1. Daily-branch supervision

The daily output of the model is directly supervised by the target daily label:

- `pred_daily = outputs["D"]`
- `loss_daily = daily_loss_fn(pred_daily, y_daily, stations)`

This is the main target-domain supervision term.

### 2. Aggregated-hourly consistency term

The hourly sequence output is also aggregated to daily scale:

- `pred_hourly_agg = outputs["H_seq"][:, -24:].mean(dim=1)`

Then it is constrained to match the same target daily label:

- `loss_agg = agg_loss_fn(pred_hourly_agg, y_daily, stations)`

This auxiliary loss encourages the hourly branch to remain consistent with the daily supervision.

### Total loss

The final objective is:

`loss = loss_daily + agg_loss_weight * loss_agg`

In this experiment:

- `agg_loss_weight = 0.5`

## Why This Affects Hourly Output

Daily supervision affects hourly predictions because gradients flow through:

`daily loss -> lstm_daily -> transfer_h/transfer_c -> hourly initial state -> hourly outputs`

In addition, the aggregated-hourly consistency loss directly pushes the hourly sequence output toward target-domain daily behavior.

Therefore, even though the target-domain supervision is daily, the final hourly outputs can change substantially.

## Model Selection

Training supervision is daily, but model selection is based on **hourly validation performance**.

At the end of each epoch:

- the model is evaluated on the target validation split,
- hourly predictions are compared against hourly ground truth,
- `median hourly KGE` is used to determine the best epoch.

This ensures the final selected checkpoint is the one that best improves hourly performance.

## Final Evaluation Protocol

Final reporting is always done on **hourly predictions**:

- hourly train KGE/NSE
- hourly validation KGE/NSE
- hourly test KGE/NSE

So the experiment should be interpreted as:

- **training supervision**: target daily scale
- **output of interest**: hourly streamflow
- **final metrics**: hourly KGE/NSE

## Key Output Files

- Best adapted model:
  `best_transfer_model.pth`
- Hourly summary metrics:
  `summary_hourly_metrics.csv`
- Per-station hourly metrics:
  `per_station_hourly_metrics.csv`
- Training history:
  `training_history.csv`
- Short experiment summary:
  `summary.md`

All of these are stored in:

`/home/kongw0a/hourly_streamflow_dl/MTSLSTM_100stations/outputs/transfer_daily_to_hourly_partial_ft_s2_random30`
