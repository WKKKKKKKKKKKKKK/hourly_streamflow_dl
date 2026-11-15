#!/bin/bash
# Stage 1: the seed-42 runs of every CV design, and every fine-tuning variant that starts from
# them. These are the recorded launches of the runs under outputs/al_*, one per line.
#
#   bash slurm/submit_1_main_runs.sh            # dry run: print the plan
#   SUBMIT=1 bash slurm/submit_1_main_runs.sh   # submit
#
# Pretraining: one array task per fold, chained as 5 x 4 h chunks (each chunk resumes from the
# last checkpoint; the gpu4 partition ends at 4 h). Fine-tuning waits for the last chunk.
#   base runs        M0 and M1 of the five designs
#   replay           M1 with source replay, ratio 0.25
#   learning rate    Step-2 learning rate 2e-4 and 1e-4
#   objective        weights of the two terms of L2, random design
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f train/pretrain_source.py ]] || { echo "run from the repository root" >&2; exit 2; }
SUBMIT="${SUBMIT:-0}"; CHUNKS=5; mkdir -p logs
LOG=logs/submission_stage1_$(date +%Y%m%d_%H%M%S).txt
root_of() { grep -m1 "^output_root:" "$1" | awk '{print $2}'; }
sub() {  # print the call; submit it when SUBMIT=1; echo the job id
  if [[ $SUBMIT == 1 ]]; then local id; id=$(sbatch --parsable "$@"); echo "$id  sbatch $*" >> "$LOG"; echo "$id"
  else echo "DRY  sbatch $*" >&2; echo "DRY"; fi
}
A=configs/aligned
declare -A BASE=( [random]=phase1_runB_v2 [blocked]=phase1_runB_blocked_v2 [climate]=phase1_runB_climate
                  [longitudinal]=phase1_runB_longitudinal [continental]=phase1_runB_continental )
declare -A ARR=( [random]=0-4 [blocked]=0-4 [climate]=0-4 [longitudinal]=0-4 [continental]=0-3 )
# fine-tuning variant -> design whose pretrained weights it starts from ; extra environment
declare -A FROM=(
  [phase1_runB_replay_v2]=random [phase1_runB_blocked_replay_v2]=blocked
  [phase1_runB_climate_replay025]=climate [phase1_runB_longitudinal_replay025]=longitudinal
  [phase1_runB_continental_replay025]=continental
  [phase1_runB_daily_lr2e4]=random [phase1_runB_daily_lr1e4]=random [phase1_runB_blocked_lr2e4]=blocked
  [phase1_runB_path_both_equal]=random [phase1_runB_path_aggheavy]=random
  [phase1_runB_path_aggonly]=random [phase1_runB_daily0]=random [phase1_runB_path_stateonly_dailysel]=random
)
declare -A ENV=( [phase1_runB_path_stateonly_dailysel]=",SCORE_BRANCH=daily" )   # w_H = 0: select on the daily head

declare -A LAST
for d in random blocked climate longitudinal continental; do
  cfg=$A/${BASE[$d]}.yaml; prev=""
  for i in $(seq 1 $CHUNKS); do
    dep=(); [[ -n $prev && $prev != DRY ]] && dep=(--dependency=afterany:$prev)
    prev=$(sub --export=ALL,CONFIG=$cfg,WANDB_GROUP=aligned_pretrain --array=${ARR[$d]} -J pre_$d \
           ${dep[@]+"${dep[@]}"} slurm/10_pretrain_runB.sbatch)
  done
  LAST[$d]=$prev
  # M1 of the design itself (PRETRAIN_ROOT empty: its own output_root)
  dep=(); [[ $prev != DRY ]] && dep=(--dependency=afterok:$prev)
  sub --export=ALL,CONFIG=$cfg,PRETRAIN_ROOT=,WANDB_GROUP=aligned_transfer --array=${ARR[$d]} -J xfer \
      ${dep[@]+"${dep[@]}"} slurm/20_transfer_runB.sbatch >/dev/null
done
for c in "${!FROM[@]}"; do
  d=${FROM[$c]}; pre=$(root_of $A/${BASE[$d]}.yaml)
  dep=(); [[ ${LAST[$d]} != DRY ]] && dep=(--dependency=afterok:${LAST[$d]})
  sub --export=ALL,CONFIG=$A/$c.yaml,PRETRAIN_ROOT=$pre${ENV[$c]:-},WANDB_GROUP=aligned_transfer --array=${ARR[$d]} \
      -J xfer ${dep[@]+"${dep[@]}"} slurm/20_transfer_runB.sbatch >/dev/null
done
[[ $SUBMIT == 1 ]] && echo "job ids in $LOG"
