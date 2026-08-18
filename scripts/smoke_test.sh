#!/bin/bash
# Submit one short, minimal-fraction job per model type as a smoke test of the
# current environment (uv-resolved venv for BLT/BPE-LCM/BPE-Transformer/Llama-8B,
# apptainer container for SONAR). Intended to be run once per cluster, from that
# cluster's own login node, with CLUSTER_PARTITION already set correctly in .env
# (a100-galvani on Galvani, h100-ferranti on Ferranti).
#
# This only checks that each job starts and gets through its imports/setup --
# it does not need to run to completion. Jobs are capped at CLUSTER_TIME
# (default 30 min) so a healthy-but-slow job gets cut off by Slurm instead of
# burning its full multi-hour default allocation; override by exporting
# CLUSTER_TIME (e.g. "01:00:00") before running this script.
#
# Job IDs are written to logs/smoke_test_<tag>.jobids for
# scripts/check_smoke_test.sh to pick up afterwards.
#
# Usage:
#   scripts/smoke_test.sh [tag]
#   scripts/smoke_test.sh galvani
#   scripts/smoke_test.sh ferranti
set -uo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
REPO_DIR=$(realpath "$SCRIPT_DIR/..")
cd "$REPO_DIR"

TAG=${1:-smoketest}
FRACTION=0.25
export CLUSTER_TIME=${CLUSTER_TIME:-00:30:00}
mkdir -p logs
STATE_FILE="logs/smoke_test_${TAG}.jobids"
: > "$STATE_FILE"

echo "Submitting smoke-test jobs (tag=$TAG, fraction=$FRACTION) ..."
echo ""

declare -A JOB_IDS
declare -a ORDER=(blt bpe_lcm bpe_transformer llama8b sonar)

submit() {
    local key="$1"; shift
    local out
    out=$("$@" 2>&1)
    local id
    id=$(echo "$out" | awk '/Submitted batch job/{print $NF}')
    if [ -n "$id" ]; then
        JOB_IDS[$key]="$id"
        echo "$key=$id" >> "$STATE_FILE"
        echo "  [$key] submitted: job $id"
    else
        echo "  [$key] FAILED to submit:"
        echo "$out" | sed 's/^/    /'
    fi
}

submit blt             scripts/sbatch.sh scripts/submit_blt.sh         "$FRACTION" "blt_${TAG}"
submit bpe_lcm         scripts/sbatch.sh scripts/submit_bpe_lcm.sh     "$FRACTION" "bpe_lcm_${TAG}"
submit bpe_transformer scripts/sbatch.sh scripts/submit_transformer.sh "$FRACTION" "bpe_transformer_${TAG}"
submit llama8b         scripts/sbatch.sh scripts/submit_llama8b.sh     "$FRACTION" "llama8b_${TAG}" qlora
submit sonar           scripts/sbatch.sh scripts/submit_sonar.sh       "$FRACTION" "sonar_${TAG}"

echo ""
echo "Submitted ${#JOB_IDS[@]} / ${#ORDER[@]} jobs."
echo ""
echo "Check status:   squeue -u \$USER"
if [ "${#JOB_IDS[@]}" -gt 0 ]; then
    ids="${JOB_IDS[@]}"
    echo "Tail all logs:  tail -f logs/*_{${ids// /,}}.out"
    echo "Cancel all:     scancel $ids"
    echo "Check outcome:  scripts/check_smoke_test.sh $TAG   (once jobs finish, time out, or you cancel them)"
fi
