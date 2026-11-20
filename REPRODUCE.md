# Output -> command -> file

All commands run from the repository root as `python -m <module>`; `bash make_reports.sh` runs every
CPU step in order. Hourly KGE of a target gauge is the transfer-stage score
(`per_station_hourly_*_target_hourly.csv`) unless a whole-hydrograph metric is named; statistics
are taken per seed, then the median over the three seeds.

## Data, designs, model

| Output | Command | File |
|---|---|---|
| map of daily-only and hourly gauges, gauge counts per network | `scripts.figures.label_availability --out-dir reports/label_availability` | `reports/figures/label_availability.png`, `reports/label_availability/label_availability.json` |
| the 294 African catchments and their sources | `scripts.prep.build_africa_basins` | `africa/africa_basins.csv` (column `source`) |
| distance from a held-out gauge to the nearest trainable gauge, five designs | `scripts.analysis.cv_designs`, then `scripts.figures.fig_cv_designs` | `outputs/v2_cv_designs/cv_designs.json`, `reports/figures/cv_designs.png` |
| protocol schematic | `scripts.figures.fig_protocol` | `reports/figures/protocol.png` |
| architecture schematic; total, trainable and frozen parameters | `scripts.figures.fig_architecture` (prints the counts) | `reports/figures/architecture.png` |
| temporal split, model and training settings | the configs (`data`, `model`, `train`, `transfer` blocks) | `configs/aligned/phase1_runB_v2.yaml` |
| gauges from the dataset (9,181) to the cross-validation (8,990), the scored targets (8,709) and the quantile-mapping table (8,468) | `scripts.numbers.gauge_counts` | `reports/methods/gauge_counts.json` |
| median length of the test period | `scripts.numbers.test_period_length` | `reports/m1_vs_m0/test_period_length.json` |
| completeness of the 24-hour windows behind the daily targets | `scripts.numbers.daily_window_coverage` | `reports/methods/daily_window_coverage.json` |
| pretraining time per fold, GPU hours per seed | `scripts.numbers.pretrain_compute` | `reports/methods/pretrain_compute.json` |

## M1 against M0

Files in `reports/m1_vs_m0/` unless named otherwise.

| Output | Command | File |
|---|---|---|
| hourly KGE of M0 and M1 against distance (five designs) | `scripts.figures.m1_vs_m0_figures` | `distance_and_dissimilarity.png`; `numbers.json` -> `ladder` |
| the same against quintiles of the dissimilarity index | `scripts.analysis.dissimilarity_index --data-root <dir with scalers.json>`, then `m1_vs_m0_figures` | `dissimilarity_per_gauge.csv`; `numbers.json` -> `dissimilarity_quintiles` |
| pretraining share of a fold against its skill (Spearman) | `scripts.numbers.fold_balance` | `fold_balance.csv`, `fold_balance_spearman.json` |
| robustness of the gain over seeds, per 4 degree cell | `m1_vs_m0_figures` | `gain_robustness_cells.png`; `numbers.json` -> `gain_robustness_cells` |
| KGE gain split into r, alpha, beta; flood-peak errors | `m1_vs_m0_figures` | `amount_not_timing.png`; `numbers.json` -> `amount_not_timing` |
| whole-hydrograph levels (KGE parts, FHV, peak magnitude and timing) | `m1_vs_m0_figures` | `numbers.json` -> `kge_decomposition`, `flood_peaks` |
| distributions over the target gauges of KGE, \|FHV\| and peak-timing error, M0 and M1, and their summary numbers | `scripts.figures.magnitude_timing_cdf` | `magnitude_timing_cdf.png`; `magnitude_timing_cdf.json` -> `text` (`quoted`: as printed) |
| hydrograph examples, one gauge per design (>= 3 peaks, test period >= 2 years, M1 KGE > 0; gain nearest the 75th percentile) | `m1_vs_m0_figures` (`scripts.figures.fig_hydrograph_examples`; series of new gauges: GPU `slurm/96_hydrograph_examples.sbatch`) | `hydrographs.png`; `numbers.json` -> `hydrograph_examples` (`summary` -> `quoted`) |
| quantile-mapping baseline (M0+QM, M1+QM) | GPU: `slurm/95_correction_baseline.sbatch`; then `scripts.numbers.quantile_mapping_table` | `reports/quantile_mapping/{by_design,by_dissimilarity}.csv`, `numbers.json`, `table_kge.tex`, `table_qm.tex` (KGE, r, alpha with caption); scored target gauges with 365 or more training days only |

## Forgetting and replay

Files in `reports/forgetting_replay/`.

| Output | Command | File |
|---|---|---|
| distributions of source and target KGE: M0, M1, M1 with replay | `scripts.figures.replay_figures` | `cdf_replay.png`, `numbers.json` |
| levels, KGE parts and paired changes behind that figure | `replay_figures` | `replay_summary.csv` (per seed: `replay_summary_per_seed.csv`) |
| continental design, per held-out continent | `scripts.numbers.replay_continental_folds` | `continental_per_fold.csv` |

## Daily scale and Africa

Files in `reports/daily_scale/` unless named otherwise.

| Output | Command | File |
|---|---|---|
| daily KGE of M1 against a daily LSTM, maps per design and Africa | GPU: `slurm/70_score_daily.sbatch`; then `scripts.figures.daily_scale_figures` | `daily_lstm_maps.png`; `numbers.json` -> `m1_vs_daily_lstm` |
| African catchments: daily LSTM, M1 and ERA5-Land | `scripts.analysis.africa_daily_three_way`, `africa_four_way`, then `daily_scale_figures` | `africa_era5.png`; `numbers.json` -> `africa_era5` |
| African flood examples and daily peak metrics | GPU: `slurm/94_africa_examples.sbatch`; then `daily_scale_figures` | `africa_examples.png`; `numbers.json` -> `africa_examples`, `africa_peaks` |
| rain-gauge density, African vs training catchments | `scripts.numbers.rain_gauge_density` | `reports/methods/rain_gauge_density.json` |

## Tables

Files in `reports/tables/`.

| Output | Command | File |
|---|---|---|
| static attributes | the list is `static_features` in any run's `fold*/pretrain/run_meta.json` | - |
| fold-1 architecture search | `scripts.tables.table_search` | `search.tex` |
| Step-2 learning rate | `scripts.tables.table_lr_seeds` | `ft_lr.csv` |
| terms of the fine-tuning objective | `scripts.tables.table_objective` | `objective.tex` |
| run-to-run variability over seeds; pretraining budget | `scripts.tables.table_lr_seeds` | `seeds.csv`, `pretrain_budget.json` |

## Definitions of the summary numbers

* `daily_window_coverage`: windows are drawn from all samples of the hourly cache (training and
  test period); the same statistics restricted to the training period are written alongside.
  The window is the 24 hours ending at the target hour (`data.window_alignment: figure`).
* `pretrain_compute`: the sum of the per-epoch training time (`train_secs`), without
  validation, checkpointing and queueing, for the seed-42 runs.
* `rain_gauge_density`: the `rain_gauge_density` static attribute; training = the 9,181 gauges
  of the hourly cache, Africa = the scored African catchments (all 294 written alongside).
