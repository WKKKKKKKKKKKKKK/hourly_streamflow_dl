#!/bin/bash
# ERA5-Land download, as a chained job rather than nohup on a login node.
# ibex compute nodes do reach CDS (verified: a batch job on cn604-16 got 202 from
# cds.climate.copernicus.eu), so there is no reason to depend on a login node
# staying up, or on anyone's laptop.
#
# CDS serialises per user at a measured 1.4 requests/hour, so this takes days. The
# job asks for 7 days and still chains a successor BEFORE starting work, with
# --dependency=afterany on itself.
#
# The long walltime is not just convenience: every time the job is killed, the CDS
# requests currently queued for it are abandoned, and those may have been waiting
# hours. At 24 h a ten-day download meant roughly ten restarts and about seventy
# wasted request slots.
# Queueing the successor at the END does not work: the job is expected to be killed
# at the time limit, and a killed script never reaches its last line. The first
# attempt did exactly that -- both jobs hit TIMEOUT and the chain died silently.
# The successor exits immediately if the manifest turns out to be complete.
#
#   PRESET=runoff  sbatch slurm/02_download_era5.sbatch
#   PRESET=forcing sbatch slurm/02_download_era5.sbatch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH -p batch
#SBATCH -J gmts_cds
#SBATCH -o logs/cds_%x_%j.out
#SBATCH -e logs/cds_%x_%j.err

set -eo pipefail
REPO_ROOT="${GMTS_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[[ -f "$REPO_ROOT/configs/phase1.yaml" ]] || { echo "not the repo: $REPO_ROOT" >&2; exit 2; }
cd "$REPO_ROOT"
mkdir -p logs
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

PRESET="${PRESET:-runoff}"
case "$PRESET" in
  runoff)
    OUT=/ibex/user/kongw0a/era5_land_africa
    ARGS=(--preset runoff --years-per-request 5 --start-year 1980 --end-year 2024 --workers 4)
    TOTAL=171 ;;
  forcing)
    OUT=/ibex/user/kongw0a/era5_land_africa_forcing
    ARGS=(--preset forcing --tiles africa/africa_era5_tiles_45deg.csv
          --start-year 1980 --end-year 1995 --years-per-request 1
          --months-per-request 2 --workers 3)
    TOTAL=288 ;;
  *) echo "PRESET must be runoff or forcing" >&2; exit 2 ;;
esac

CHAIN="${CDS_CHAIN:-1}"
echo "node=$(hostname) preset=$PRESET out=$OUT chain_link=$CHAIN"
[[ -f "$HOME/.cdsapirc" ]] || { echo "~/.cdsapirc missing" >&2; exit 2; }

count_done () {
  "$PYTHON_BIN" -c "
import json, pathlib
p = pathlib.Path('$OUT/manifest.json')
m = json.loads(p.read_text()) if p.exists() else {}
print(sum(1 for v in m.values() if v.get('complete')))
"
}

DONE=$(count_done)
echo "already complete: $DONE / $TOTAL"
if [[ "$DONE" -ge "$TOTAL" ]]; then
  echo "download already complete -- nothing to do"
  exit 0
fi

# Chain the successor NOW, before any work, so a TIMEOUT or a lost node cannot
# break the chain. afterany fires whatever the exit state. Capped so a script that
# fails instantly cannot spawn an unbounded chain of jobs. With 7-day links the cap
# is 140 days of coverage, far more than needed.
if (( CHAIN < 20 )); then
  NEXT=$(PRESET="$PRESET" CDS_CHAIN=$((CHAIN + 1)) sbatch --parsable \
           --dependency=afterany:"$SLURM_JOB_ID" \
           --job-name="gmts_cds_$PRESET" "$REPO_ROOT/slurm/02_download_era5.sbatch")
  echo "chained successor $NEXT (link $((CHAIN + 1))/20, afterany:$SLURM_JOB_ID)"
else
  echo "chain limit reached at link $CHAIN -- not chaining further" >&2
fi

# A non-zero exit only means some requests failed; the manifest decides what is left.
set +e
"$PYTHON_BIN" -u -m scripts.download_era5_land_africa "${ARGS[@]}" --out-dir "$OUT"
set -e

echo "complete at exit: $(count_done) / $TOTAL"
