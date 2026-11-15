#!/bin/bash
# Stage 3: the architecture search, the daily-LSTM baselines and the African arms.
# Every line reproduces a recorded launch.
#
#   bash slurm/submit_3_search_baselines_africa.sh            # dry run
#   SUBMIT=1 bash slurm/submit_3_search_baselines_africa.sh   # submit (after stage 1 for Africa)
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f train/pretrain_source.py ]] || { echo "run from the repository root" >&2; exit 2; }
SUBMIT="${SUBMIT:-0}"; mkdir -p logs
LOG=logs/submission_stage3_$(date +%Y%m%d_%H%M%S).txt
sub() {
  if [[ $SUBMIT == 1 ]]; then local id; id=$(sbatch --parsable "$@"); echo "$id  sbatch $*" >> "$LOG"; echo "$id"
  else echo "DRY  sbatch $*" >&2; echo "DRY"; fi
}

# ---- architecture search: fold 1 of the random design; four variants also on all five folds.
# Each fold is pretrained and fine-tuned in 4 x 4 h chunks (slurm/50_search.sbatch).
FIVE="g01_gauch_hs64_H336 g02_gauch_hs128_H336 g03_forgetbias_H72 r06_bs256_do4_hs64_H336"
for cfg in configs/aligned/search/*.yaml; do
  name=$(basename "$cfg" .yaml); folds=1; [[ " $FIVE " == *" $name "* ]] && folds="0 1 2 3 4"
  for k in $folds; do
    prev=""
    for i in 1 2 3 4; do
      dep=(); [[ -n $prev && $prev != DRY ]] && dep=(--dependency=afterany:$prev)
      prev=$(sub --export=ALL,CONFIG=$cfg,FOLD=$k -J search ${dep[@]+"${dep[@]}"} slurm/50_search.sbatch)
    done
  done
done

# ---- daily LSTM trained from scratch on the target gauges' daily observations
for d in random blocked climate longitudinal continental; do
  arr=0-4; [[ $d == continental ]] && arr=0-3
  sub --export=ALL,CONFIG=configs/phase1_dailylstm_aligned_$d.yaml -J dLSTM --array=$arr --time=08:00:00 \
      slurm/71_train_daily_lstm.sbatch >/dev/null
done

# ---- Africa: five temporal folds per arm, each its own job.
#   m1b          M1 fine-tuned from each of the five random-design pretrained models
#   daily_lstm   daily LSTM from scratch
#   scratch_agg  the M1 architecture and objective from scratch (enters the common basin set)
africa() {  # arm config env
  for k in 0 1 2 3 4; do
    sub --export=ALL,CONFIG=$2,FOLD=$k,OUT_DIR=outputs/al_v2_africa_$1_fold$k,$3 -J afxfer --time=08:00:00 \
        slurm/42_africa_transfer.sbatch >/dev/null
  done
}
africa m1b        configs/aligned/phase1_runB_v2.yaml            "SELECT_FRAC=0.15,EPOCHS=80,PATIENCE=15"
africa daily_lstm configs/phase1_africa_daily_lstm_aligned.yaml  "SCRATCH=1,SCORE_BRANCH=daily,SELECT_FRAC=0.15"
africa scratch_agg configs/aligned/phase1_africa_scratch_agg.yaml "SCRATCH=1,SCORE_BRANCH=agg,SELECT_FRAC=0.15"
[[ $SUBMIT == 1 ]] && echo "job ids in $LOG"
