#!/bin/bash
# Every figure, table and summary number, from the finished runs under outputs/.
# CPU only (a login node is enough). Writes only into reports/, except steps 1-2, which write the
# intermediate analysis files the figures read (outputs/v2_cv_designs, outputs/al_v2_africa_*).
#
#   bash make_reports.sh              # steps 3-5: figures, tables, numbers from the existing intermediate files
#   bash make_reports.sh all          # steps 1-5: also rebuild the intermediate files under outputs/
#                                     # (outputs/ may be a link to the original results: step 1-2 then
#                                     #  rewrites files there, with identical content)
#
# DATA_ROOT: directory with the dataset's scalers.json, needed only by the dissimilarity index
# (static part). The config's data.root no longer exists on ibex; see README.md, "Data".
set -euo pipefail
cd "$(dirname "$0")"
export DATA_ROOT="${DATA_ROOT:-/ibex/user/kongw0a/dec_track/data/june_check}"   # also used to predict missing example series
run() { echo "== python -m $*"; python -m "$@"; }

if [[ "${1:-figures}" == all ]]; then
  # 1. Distances between held-out and trainable gauges
  run scripts.analysis.cv_designs                        # outputs/v2_cv_designs/cv_designs.json
  # 2. Africa: ERA5-Land and the five-member medians (after the GPU ensemble job, stage 4)
  run scripts.analysis.africa_daily_three_way            # outputs/al_v2_africa_three_way/
  run scripts.analysis.africa_four_way                   # outputs/al_v2_africa_five_way/
fi
# 3. Data and design figures
run scripts.figures.label_availability --out-dir reports/label_availability   # daily vs hourly gauges (+ counts json)
run scripts.figures.fig_cv_designs                       # the five CV designs
run scripts.figures.fig_protocol                         # protocol schematic
run scripts.figures.fig_architecture                     # architecture (and the parameter counts)
# 4. Results
run scripts.analysis.dissimilarity_index --data-root "$DATA_ROOT"   # input of distance_and_dissimilarity and the QM table
run scripts.figures.m1_vs_m0_figures                     # M1 vs M0 figures + reports/m1_vs_m0/numbers.json
run scripts.figures.magnitude_timing_cdf                 # distributions of KGE, |FHV|, peak timing (+ text numbers)
run scripts.numbers.kge_2012_sensitivity                 # the KGE results under KGE' (Kling et al., 2012)
run scripts.numbers.fold_balance                         # pretraining share vs fold skill (Spearman)
run scripts.numbers.test_period_length                   # median test-period length
run scripts.numbers.gauge_counts                         # dataset -> cross-validation -> scored gauges
run scripts.numbers.daily_window_coverage                # valid hours per 24-h window
run scripts.numbers.pretrain_compute                     # GPU hours of pretraining
run scripts.numbers.rain_gauge_density                   # African vs training gauge density
run scripts.numbers.quantile_mapping_table               # quantile-mapping baseline
run scripts.figures.replay_figures                       # forgetting and replay
run scripts.numbers.replay_continental_folds             # replay, per held-out continent
run scripts.figures.daily_scale_figures                  # daily scale: daily LSTM, Africa, ERA5-Land
# 5. Tables
run scripts.tables.table_search                          # architecture search
run scripts.tables.table_lr_seeds                        # Step-2 learning rate, seeds (+ pretraining budget)
run scripts.tables.table_objective                       # terms of the fine-tuning objective
echo "done: reports/"
