#!/bin/bash
# Stage 4: GPU evaluation of the trained models, after stages 1-3 are done.
#
#   bash slurm/submit_4_evaluation.sh            # dry run
#   SUBMIT=1 bash slurm/submit_4_evaluation.sh   # submit
#
#   70_score_daily        daily KGE of every target gauge (M1 of every seed, daily LSTM)
#                         -> outputs/v2_design_daily/<run>/                       (Fig. D1)
#   92_full_series        continuous hourly series of every test day, M0 and M1, all seeds
#                         -> <run>/full_series/                                    (Figs. 7, 8)
#   95_correction_baseline  quantile mapping of M0 and M1 on the daily observations, all seeds
#                         -> <run>/correction_baseline_v3/                         (QM paragraph)
#   43_africa_ensemble    African M0/M1 daily predictions of the five members -> al_v2_africa_m1b_summary
#   94_africa_examples    daily and hourly series of the African example catchments (Fig. D3);
#                         needs the CPU steps africa_daily_three_way and africa_four_way first (README)
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f train/pretrain_source.py ]] || { echo "run from the repository root" >&2; exit 2; }
SUBMIT="${SUBMIT:-0}"; mkdir -p logs
LOG=logs/submission_stage4_$(date +%Y%m%d_%H%M%S).txt
sub() {
  if [[ $SUBMIT == 1 ]]; then local id; id=$(sbatch --parsable "$@"); echo "$id  sbatch $*" >> "$LOG"; echo "$id"
  else echo "DRY  sbatch $*" >&2; echo "DRY"; fi
}
A=configs/aligned
declare -A BASE=( [random]=phase1_runB_v2 [blocked]=phase1_runB_blocked_v2 [climate]=phase1_runB_climate
                  [longitudinal]=phase1_runB_longitudinal [continental]=phase1_runB_continental )
for d in random blocked climate longitudinal continental; do
  arr=0-4; n=5; [[ $d == continental ]] && { arr=0-3; n=4; }
  c=$A/${BASE[$d]}.yaml
  seeds=("$c" configs/variance/al_${d}_s43.yaml configs/variance/al_${d}_s44.yaml)
  # daily scoring: M1 of every seed (the 24-h mean of its hourly output) and the daily LSTM
  for cfg in "${seeds[@]}"; do
    sub --export=ALL,CONFIG=$cfg,TAG=M1,DOMAIN=target,SCORE_BRANCH=agg -J scoreD --array=$arr slurm/70_score_daily.sbatch >/dev/null
  done
  sub --export=ALL,CONFIG=configs/phase1_dailylstm_aligned_$d.yaml,TAG=daily_lstm,DOMAIN=target,SCORE_BRANCH=daily \
      -J scoreD --array=$arr slurm/70_score_daily.sbatch >/dev/null
  # whole hydrographs and quantile mapping: one job per fold
  for k in $(seq 0 $((n - 1))); do
    for cfg in "${seeds[@]}"; do
      sub --export=ALL,CONFIG=$cfg,FOLDS=$k -J fullser slurm/92_full_series.sbatch >/dev/null
      sub --export=ALL,CONFIG=$cfg,FOLDS=$k -J corr slurm/95_correction_baseline.sbatch >/dev/null
    done
  done
done
# Africa: ensemble of the five M1 members (and of the pretrained M0)
sub --export=ALL,CONFIG=$A/phase1_runB_v2.yaml,INSITU_GLOB=outputs/al_v2_africa_m1b_fold,PRETRAIN_ROOT=outputs/al_v2_runB,OUT_DIR=outputs/al_v2_africa_m1b_summary \
    -J afens slurm/43_africa_ensemble.sbatch >/dev/null
echo "after the CPU steps africa_daily_three_way and africa_four_way: sbatch slurm/94_africa_examples.sbatch" >&2
[[ $SUBMIT == 1 ]] && echo "job ids in $LOG"
