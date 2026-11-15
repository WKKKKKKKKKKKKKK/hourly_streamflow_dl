# Daily-supervised fine-tuning of an hourly MTS-LSTM

Code, configurations and fold definitions behind *Can Daily Observations Improve Hourly
Streamflow Prediction?* (HESS submission). An MTS-LSTM (Gauch et al., 2021) is pretrained on
hourly discharge of source gauges (M0) and fine-tuned at held-out target gauges with their daily
means only, the hourly LSTM frozen (M1). Five cross-validation designs (random, blocked,
climate zone, longitudinal, continental), three seeds each, plus 294 African catchments that have
daily records only.

This repository contains only what the manuscript uses: the model and training code, the
configurations of the runs that enter a figure, table or number, the jobs that ran them, and one
script per figure, table or group of quoted numbers. It was extracted on 2026-11-15 from the
development repository `~/global_mtslstm_aligned`, which is unchanged.
[REPRODUCE.md](REPRODUCE.md) maps every item of the manuscript to the command and the file that
produce it.

## Layout

```
common/ data/ models/ train/ eval/   model, data pipeline, training and scoring (library code)
configs/                             one YAML per run; see configs/README.md
folds/                               the five fold assignments (station -> fold), as used
index/stations.csv                   station list the random folds were drawn from
africa/                              the 294 African catchments (attributes, polygons, ERA5 tiles)
slurm/                               job scripts and the four submission stages
scripts/prep/                        building the inputs: hourly cache, folds, African forcing
scripts/analysis/                    evaluation of trained models and intermediate analyses
scripts/figures/                     manuscript figures and the numbers quoted from them
scripts/tables/                      appendix tables
scripts/numbers/                     quoted numbers that belong to no figure
make_paper.sh                        every figure, table and number, in order (CPU only)
reports/                             what make_paper.sh writes (regenerated and verified, see below)
outputs -> /ibex/user/kongw0a/global_mtslstm_outputs_aligned   the finished runs (symlink)
```

Run every command from the repository root as `python -m scripts.<group>.<name>`.

## Environment

Python 3.10 with the packages in `requirements.txt` (results were produced with torch 2.7.1+cu118
on NVIDIA V100/A100 cards of the KAUST Ibex cluster). Maps need the Natural Earth 1:110m
coastline and admin-0 boundary shapefiles in the cartopy cache (`~/.local/share/cartopy`).
Training logs to Weights & Biases; set `wandb.mode: offline` in a config to run without an account.

## Data

| Input | Location on ibex | Made by |
|---|---|---|
| hourly discharge and forcing of the 9,181 gauges (six archives, MSWEP V3.16, ERA5-HRES, Penman PET), 1980-2025 | `/ibex/project/c2266/abbaa0a/data/input_data/hourly_q_dl/6sources.nc` | data provider |
| static attributes | `/ibex/project/c2266/abbaa0a/data/gscad_database/processed/20250630/hourly/dataframes/static.csv` | data provider |
| dataset `config.json` and `scalers.json` (normalisation) | `data.root` of every config: `.../hourly_q_dl/6datasets_H_8760_..._9181_42` | data provider; **no longer on ibex** (see below) |
| hourly cache: memmap of 6sources.nc, sample index, per-gauge 70/30 temporal split | `/ibex/user/kongw0a/hourly_cache` | `scripts/prep/build_hourly_cache.py` |
| daily archive (Fig. 1 only) | `.../gscad_database/processed/20250630/daily/dataframes/daily_q.nc` | data provider |
| African daily discharge and the daily training products for the 294 catchments | `/ibex/project/c2266/abbaa0a/data/input_data/hydrodeepai/` | data provider |
| African hourly forcing: ERA5-Land, rescaled to the daily training products | `/ibex/user/kongw0a/era5_land_africa_forcing/era5_land_africa_hourly_forcing_penman.nc` | download, basin average, rescale (below) |
| ERA5-Land daily runoff of the African catchments | `/ibex/user/kongw0a/era5_land_africa/era5_land_africa_daily_runoff.nc` | download, basin average (below) |

**Not distributed.** Several inputs come from archives with restricted access (ADHI, GRDC
and other national networks). No discharge series of any gauge is part of this repository:
it holds code, gauge metadata and results. One file is also left out because the paper does
not need it: the inventory of every daily-only gauge in the archive
(`reports/label_availability/daily_only_gauges.csv`), which lists gauges of non-public
networks unrelated to this study; it is written locally by
`scripts/figures/label_availability.py`.

**Missing normalisation file.** The prepared-dataset directory named in `data.root` was removed
from the project space by the data provider (replaced by a newer dataset on 2026-11-07). Its
`scalers.json` and `config.json` are needed to train or run any model. A copy recomputed from
`6sources.nc` with the same method exists at `/ibex/user/kongw0a/dec_track/data/june_check/`
(job `valsc_51855630`, 2026-09-13): the 42 static scalers are identical, the dynamic and discharge
scalers differ from the original by at most 5.4e-5 (relative). It reproduces the dissimilarity
index exactly (only the static scalers enter) but would not retrain the models bit for bit. To
retrain exactly, ask the data provider for the original two files and point `data.root` at them.

## Pipeline

**0. Inputs** (once). The folds are included in `folds/` and need not be rebuilt.

```bash
sbatch slurm/01_build_cache.sbatch                       # hourly cache (~55 GiB) from 6sources.nc
python -m scripts.prep.make_folds                        # folds/folds_random.csv (from index/stations.csv)
python -m scripts.prep.make_folds_blocked                # blocked, then the three derived designs:
python -m scripts.prep.make_folds_climate
python -m scripts.prep.make_folds_longitudinal
python -m scripts.prep.make_folds_continental
python -m scripts.prep.build_africa_basins               # africa/africa_basins.{csv,gpkg}, ERA5 tiles
PRESET=forcing sbatch slurm/02_download_era5.sbatch      # ERA5-Land hourly t2m, tp, pev (needs ~/.cdsapirc)
PRESET=runoff  sbatch slurm/02_download_era5.sbatch      # ERA5-Land daily runoff
PRESET=forcing sbatch slurm/40_basin_average.sbatch      # catchment means -> era5_land_africa_hourly_forcing.nc
PRESET=runoff  sbatch slurm/40_basin_average.sbatch      # -> era5_land_africa_daily_runoff.nc
python -m scripts.prep.rescale_africa_forcing            # -> ..._hourly_forcing_penman.nc: ERA5-Land's
                                                         #    sub-daily shape, daily totals of the training products
```

**1-4. Training and evaluation on GPUs.** Each stage prints its plan without `SUBMIT=1`.

```bash
SUBMIT=1 bash slurm/submit_1_main_runs.sh               # seed 42: 5 designs (M0, M1), replay, learning rate, objective
SUBMIT=1 bash slurm/submit_2_seed_runs.sh               # seeds 43, 44 of every design, and their replay arms
SUBMIT=1 bash slurm/submit_3_search_baselines_africa.sh # architecture search, daily LSTMs, African arms
SUBMIT=1 bash slurm/submit_4_evaluation.sh              # daily scores, whole hydrographs, quantile mapping, Africa ensemble
bash make_paper.sh all                                   # (CPU) intermediate analyses, then step 5
sbatch slurm/94_africa_examples.sbatch                   # African example series (needs the four-way table)
```

Pretraining a fold takes a median of 7.6 h on one GPU (chained 4 h chunks that resume from the
last checkpoint); the five designs take about 175 GPU-hours per seed.

**5. Figures, tables, numbers** (CPU, minutes): `bash make_paper.sh` writes everything into
`reports/` from the finished runs under `outputs/`.

## Verification

On 2026-11-15 the original scripts of the development repository and the scripts here were run
on the same finished runs. All twelve figures are pixel-identical; every table and number file
is identical (the search table differs in its first, comment line). The CV-design distances,
the dissimilarity index (with the recomputed scalers, see Data) and the two African summary
tables were also rebuilt and are identical to the files the figures were drawn from. Differences
from the original scripts, all intended:

* `fig_architecture.py` built the model with 4 dynamic inputs instead of 3 and therefore printed
  114,690 trainable and 81,408 frozen parameters in the legend of the architecture figure. Every
  trained checkpoint has 28 inputs (3 forcings + 25 statics): 195,074 parameters, 114,178
  trainable, 80,896 frozen, as the manuscript text states. Fixed here; the figure in the
  manuscript (`image_5.png`) still shows the old counts.
* `reports/sec41_m1_vs_m0/numbers.json` now also records the dissimilarity quintiles (Fig. 5b-c),
  the hydrograph examples (Fig. 8) and the share of gauges inside the grid cells of Fig. 6; the
  keys of figures not in the manuscript are gone. Text numbers of Appendix C come from
  `fig2_text_numbers.csv`, as before.
* `scripts/figures/fig_protocol.py` is the main-line version (`~/global_mtslstm`), which the
  manuscript's Fig. 3 was drawn with: it has the "outside the CV (Africa only)" ERA5-Land item
  that the development branch's copy lacks. The only change is "held-out period" -> "test
  period" in the Step 3 and evaluate boxes.
* `scripts/numbers/fold_balance.py` saves its Spearman correlations
  (`fold_balance_spearman.json`) instead of only printing them.
* Table files are named `search.tex`, `objective.tex`, `ft_lr.csv`, `seeds.csv`,
  `pretrain_budget.json` (formerly with an `_aligned` suffix).

## Names

Output roots keep their names from the development repository (`al_v2_<design>`,
`var_al_<design>_s43`, ...), so this code reads the finished runs as they are. `al_` marks the
figure-aligned re-run (both branches end at the target hour, as in the architecture figure);
`v2` the configuration after the hyperparameter check; `runB` is the random design. Analysis
directories `outputs/v2_cv_designs`, `outputs/v2_design_daily` and the `v2_<design>` entries are
links to or products of the aligned runs. The code and the outputs call the late 30 % of each
gauge's record the "validation" split; the manuscript calls it the test period.
