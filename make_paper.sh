#!/bin/bash
# Every figure, table and quoted number of the manuscript, from the finished runs under outputs/.
# CPU only (a login node is enough). Writes only into reports/, except steps 1-2, which write the
# intermediate analysis files the figures read (outputs/v2_cv_designs, outputs/al_v2_africa_*).
#
#   bash make_paper.sh              # steps 3-5: figures, tables, numbers from the existing intermediate files
#   bash make_paper.sh all          # steps 1-5: also rebuild the intermediate files under outputs/
#                                   # (outputs/ may be a link to the original results: step 1-2 then
#                                   #  rewrites files there, with identical content)
#
# DATA_ROOT: directory with the dataset's scalers.json, needed only by the dissimilarity index
# (static part). The config's data.root no longer exists on ibex; see README.md, "Data".
set -euo pipefail
cd "$(dirname "$0")"
DATA_ROOT="${DATA_ROOT:-/ibex/user/kongw0a/dec_track/data/june_check}"
run() { echo "== python -m $*"; python -m "$@"; }

if [[ "${1:-figures}" == all ]]; then
  # 1. Methods: distances between held-out and trainable gauges
  run scripts.analysis.cv_designs                        # outputs/v2_cv_designs/cv_designs.json
  # 2. Africa: ERA5-Land and the five-member medians (after the GPU ensemble job, stage 4)
  run scripts.analysis.africa_daily_three_way            # outputs/al_v2_africa_three_way/
  run scripts.analysis.africa_four_way                   # outputs/al_v2_africa_five_way/
fi
# 3. Methods figures
run scripts.figures.label_availability --out-dir reports/label_availability   # Fig. 1 (+ gauge counts json)
run scripts.figures.fig_cv_designs                       # Fig. 2
run scripts.figures.fig_protocol                         # Fig. 3
run scripts.figures.fig_architecture                     # Fig. 4 (and the parameter counts)
# 4. Results (Sect. 3) and appendices
run scripts.analysis.dissimilarity_index --data-root "$DATA_ROOT"   # input of Fig. 5b-c and the QM table
run scripts.figures.sec41_figures                        # Figs. 5-8 + reports/sec41_m1_vs_m0/numbers.json
run scripts.numbers.fold_balance                         # pretraining share vs fold skill (Spearman)
run scripts.numbers.test_period_length                   # median test-period length
run scripts.numbers.daily_window_coverage                # valid hours per 24-h window (Sect. 2.1, loss)
run scripts.numbers.pretrain_compute                     # GPU hours of pretraining (Appendix B)
run scripts.numbers.rain_gauge_density                   # African vs training gauge density (Appendix D)
run scripts.numbers.sec41_correction_table               # quantile-mapping paragraph
run scripts.figures.sec42_figures                        # Fig. C1 + appendix C numbers
run scripts.numbers.sec42_continental_folds              # appendix C, per held-out continent
run scripts.figures.sec43_figures                        # Figs. D1-D3 + appendix D numbers
# 5. Appendix tables
run scripts.tables.table_search                          # Table B1
run scripts.tables.table_appendix                        # Tables B2, B4 (+ pretraining budget)
run scripts.tables.table_objective                       # Table B3
echo "done: reports/"
