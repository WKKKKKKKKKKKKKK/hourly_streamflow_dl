#!/bin/bash
# Stage 2: seeds 43 and 44 of every design (pretraining and fine-tuning repeated), and their
# replay arms. With the seed-42 runs of stage 1 every design has three seeds.
#
#   bash slurm/submit_2_seed_runs.sh            # dry run
#   SUBMIT=1 bash slurm/submit_2_seed_runs.sh   # submit
#
# The configs of the four non-continental designs carry their seed; the two continental configs
# keep seed 42 in the file and receive it at launch (EXTRA_SET, passed through the environment),
# as they were run. The recorded
# seeds are in each run's fold*/pretrain/run_meta.json.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f train/pretrain_source.py ]] || { echo "run from the repository root" >&2; exit 2; }
SUBMIT="${SUBMIT:-0}"; CHUNKS=5; mkdir -p logs
LOG=logs/submission_stage2_$(date +%Y%m%d_%H%M%S).txt
sub() {
  if [[ $SUBMIT == 1 ]]; then local id; id=$(sbatch --parsable "$@"); echo "$id  sbatch $*" >> "$LOG"; echo "$id"
  else echo "DRY  sbatch $*" >&2; echo "DRY"; fi
}
for d in random blocked climate longitudinal continental; do
  arr=0-4; [[ $d == continental ]] && arr=0-3
  for s in 43 44; do
    cfg=configs/variance/al_${d}_s${s}.yaml
    extra=""; [[ $d == continental ]] && extra="train.seed=$s transfer.seed=$s"
    prev=""
    for i in $(seq 1 $CHUNKS); do
      dep=(); [[ -n $prev && $prev != DRY ]] && dep=(--dependency=afterany:$prev)
      prev=$(EXTRA_SET="$extra" sub --export=ALL,CONFIG=$cfg --array=$arr -J var_pre ${dep[@]+"${dep[@]}"} \
             slurm/10_pretrain_runB.sbatch)
    done
    dep=(); [[ $prev != DRY ]] && dep=(--dependency=afterok:$prev)
    EXTRA_SET="$extra" sub --export=ALL,CONFIG=$cfg --array=$arr -J var_xfer ${dep[@]+"${dep[@]}"} \
        slurm/20_transfer_runB.sbatch >/dev/null
    # replay arm of the same seed, from the same pretrained weights
    EXTRA_SET="" sub --export=ALL,CONFIG=configs/variance/al_${d}_replay025_s${s}.yaml,PRETRAIN_ROOT=outputs/var_al_${d}_s${s} \
        --array=$arr -J var_replay ${dep[@]+"${dep[@]}"} slurm/20_transfer_runB.sbatch >/dev/null
  done
done
[[ $SUBMIT == 1 ]] && echo "job ids in $LOG"
