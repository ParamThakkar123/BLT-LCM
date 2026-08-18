#!/bin/bash
# Cluster-aware sbatch wrapper.
# Reads CLUSTER_PARTITION/CLUSTER_MEM from .env and passes them to sbatch, so
# the same submission scripts work on Galvani, Ferranti, and any other
# cluster by changing variables in .env instead of editing each script.
# CLUSTER_TIME optionally overrides a job's own #SBATCH --time (a command-line
# --time always wins over the one baked into the script) -- unset by default,
# so normal full-length runs are unaffected; scripts/smoke_test.sh sets it.
#
# Usage (from repo root):
#   scripts/sbatch.sh scripts/submit_blt.sh 0.25 lcm_blt_25
#   scripts/sbatch.sh scripts/submit_sonar.sh 0.50 lcm_sonar_50
set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
REPO_DIR=$(realpath "$SCRIPT_DIR/..")

[ -f "$REPO_DIR/.env" ] && set -a && source "$REPO_DIR/.env" && set +a

PARTITION=${CLUSTER_PARTITION:-a100-galvani}
MEM=${CLUSTER_MEM:-40G}
mkdir -p "$REPO_DIR/logs"

TIME_ARGS=()
[ -n "${CLUSTER_TIME:-}" ] && TIME_ARGS=(--time="$CLUSTER_TIME")

exec sbatch --partition="$PARTITION" --mem="$MEM" "${TIME_ARGS[@]}" "$@"
