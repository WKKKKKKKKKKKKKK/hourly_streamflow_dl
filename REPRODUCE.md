# Manuscript item -> command -> file

All commands run from the repository root; `bash make_paper.sh` runs every one of the CPU steps in
order. Figure and table numbers follow `manuscript_hess.tex` (appendices: A static attributes,
B hyperparameters, C forgetting and replay, D daily scale). Hourly KGE of a target gauge is the
transfer-stage score (`per_station_hourly_*_target_hourly.csv`) unless a whole-hydrograph metric
is named; statistics are taken per seed, then the median over the three seeds.

## Methods

| Item | Command | Output |
|---|---|---|
| Fig. 1, gauge counts (27,992 daily, 9,755 hourly, 294 African) | `scripts.figures.label_availability --out-dir reports/label_availability` | `reports/figures/fig17_label_availability.png`, `reports/label_availability/label_availability.json` |
| 294 African catchments: 126 GRDC-Caravan, 119 GRDC, 49 ADHI | `scripts.prep.build_africa_basins` | `africa/africa_basins.csv` (column `source`) |
| Fig. 2, distance to the nearest trainable gauge | `scripts.analysis.cv_designs`, then `scripts.figures.fig_cv_designs` | `outputs/v2_cv_designs/cv_designs.json`, `reports/figures/fig19_cv_designs.png` |
| Fig. 3 | `scripts.figures.fig_protocol` | `reports/figures/fig25_protocol.png` |
| Fig. 4; 195,074 / 114,178 / 80,896 parameters | `scripts.figures.fig_architecture` (prints the counts) | `reports/figures/fig26_architecture.png` |
| 70/30 temporal split, 18 of 24 hours, model and training settings | the configs (`data`, `model`, `train`, `transfer` blocks) | `configs/aligned/phase1_runB_v2.yaml` |
| Median test period about 7 years (6.95 y) | `scripts.numbers.test_period_length` | `reports/sec41_m1_vs_m0/test_period_length.json` |

## Results (Sect. 3)

| Item | Command | Output |
|---|---|---|
| Fig. 5a; gains 0.05-0.21, M1 about 0.6, continental gain 0.36 and M1 about 0.1 | `scripts.figures.sec41_figures` | `fig11_distance_and_dissimilarity.png`; `numbers.json` -> `ladder` |
| Fig. 5b-c; DI quintiles (0.6 -> 0, 0.6 -> 0.4, 0.7 -> 0.28, gains 0.05-0.34) | `scripts.analysis.dissimilarity_index --data-root <dir with scalers.json>`, then `sec41_figures` | `dissimilarity_per_gauge.csv`; `numbers.json` -> `dissimilarity_quintiles` |
| Pretraining share vs fold skill: rho = 0.23; temperate 32 % / 0.55, arid 91 % / 0.10, Oceania 80 % / 0.00 | `scripts.numbers.fold_balance` | `fold_balance.csv`, `fold_balance_spearman.json` |
| Fig. 6; robust-gain shares, 116 cells, about 99 % of gauges | `sec41_figures` | `fig9_gain_robustness_cells.png`; `numbers.json` -> `gain_robustness_cells` |
| Fig. 7; shares of alpha, beta, r (39-51 %, at most 14 %); reductions of FHV (24-49 %), peak magnitude (6-18 %), timing (1-8 %) | `sec41_figures` | `fig13_amount_not_timing.png`; `numbers.json` -> `amount_not_timing` |
| Peak-timing error about 5-6 h, other whole-hydrograph levels | `sec41_figures` | `numbers.json` -> `flood_peaks`, `kge_decomposition` |
| Fig. 8; 79-91 % eligible gauges, FHV and timing of the examples | `sec41_figures` (`scripts.figures.fig_hydrograph_examples`) | `fig7a_hydrographs_main.png`; `numbers.json` -> `hydrograph_examples` |
| Quantile-mapping paragraph (M0+QM 0.58-0.67, M1+QM 0.64-0.68, margin 0.01 -> 0.06, alpha 0.84-0.87) | GPU: `slurm/95_correction_baseline.sbatch`; then `scripts.numbers.sec41_correction_table` | `reports/sec41_correction_baseline/{by_design,by_dissimilarity}.csv`, `numbers.json`, `table_kge.tex` |

Files of this section are in `reports/sec41_m1_vs_m0/` unless named otherwise.

## Appendices

| Item | Command | Output |
|---|---|---|
| Table A1 (static attributes) | written by hand; the list is `static_features` in any run's `fold*/pretrain/run_meta.json` | `appendix_static_attributes.tex` (manuscript) |
| Table B1 (architecture search) | `scripts.tables.table_search` | `reports/tables/search.tex` |
| Table B2 (Step-2 learning rate) | `scripts.tables.table_appendix` | `reports/tables/ft_lr.csv` |
| Table B3 (objective terms) | `scripts.tables.table_objective` | `reports/tables/objective.tex` |
| Table B4 (seeds); "17 of the 24 seed-42 folds, 6 with the best epoch among the last three" | `scripts.tables.table_appendix` | `reports/tables/seeds.csv`, `pretrain_budget.json` |
| Fig. C1 and the medians on it | `scripts.figures.sec42_figures` | `reports/sec42_forgetting_replay/fig2_cdf_replay.png`, `numbers.json` |
| Appendix C text: source KGE 0.65 -> 0.55 ... -0.04, KGE < 0 9 % -> 15-57 %, ln beta, ln alpha, r, replay levels, win shares 61-97 %, target changes | `sec42_figures` | `fig2_text_numbers.csv` (per seed: `fig2_text_numbers_per_seed.csv`) |
| Appendix C, per held-out continent (Oceania, North America) | `scripts.numbers.sec42_continental_folds` | `reports/sec42_forgetting_replay/continental_per_fold.csv` |
| Fig. D1 (8,574 gauges) | GPU: `slurm/70_score_daily.sbatch`; then `scripts.figures.sec43_figures` | `reports/sec43_daily_scale/fig4_maps.png`; `numbers.json` -> `m1_vs_daily_lstm` |
| Fig. D2; Africa 0.55 / 0.52 / -0.36, r about 0.38 vs 0.8 | `scripts.analysis.africa_daily_three_way`, `africa_four_way`, then `sec43_figures` | `fig2_africa_era5.png`; `numbers.json` -> `africa_era5` |
| Fig. D3 | GPU: `slurm/94_africa_examples.sbatch`; then `sec43_figures` | `fig3_africa_examples.png`; `numbers.json` -> `africa_examples`, `africa_peaks` |

## Data and compute numbers

First computed ad hoc, scripted on 2026-11-15 with the original definitions (recovered from the
session logs) and checked against the manuscript.

| Number in the manuscript | Where | Command / output | Check |
|---|---|---|---|
| fewer than 1 % of 20,000 sampled windows below 18 valid hours in every archive except CAMELS-AUS (21.8 %) | Sect. 2.1 | `scripts.numbers.daily_window_coverage` -> `reports/methods/daily_window_coverage.json`, `windows_20k` | reproduced exactly (21.79 %; others at most 0.82 %). The windows are drawn from all samples, training and test period; "training windows" would give 28.5 % for CAMELS-AUS |
| about 2.7 % of the daily targets from 18 to 23 valid hours | Sect. 2.2, loss | same file, `targets_60k` | not reproducible: with the window the paper describes (24 h ending at the target hour) it is 3.1 % of the sampled windows, 3.2 % of the retained targets (3.4 % in the training period). The 2.7 % was measured in the main line before the window alignment (the previous day's window gives 2.8 %) |
| 7.6 h median per fold, about 175 GPU-hours | Appendix B | `scripts.numbers.pretrain_compute` -> `reports/methods/pretrain_compute.json` | aligned seed-42 runs: 7.65 h, 173.9 GPU-hours (training passes only). The 7.6 h came from the pre-alignment v2 runs; 7.7 h would round correctly |
| rain gauge density 0.027 vs 0.787, all African catchments below the training median | Appendix D | `scripts.numbers.rain_gauge_density` -> `reports/methods/rain_gauge_density.json` | reproduced exactly: 0.787 over 9,174 training gauges, 0.027 over 281 of the 282 scored African catchments (one has no value), 100 % below; over all 294 study catchments 0.036 |
