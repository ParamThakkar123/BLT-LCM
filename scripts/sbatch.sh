#!/bin/bash
# Cluster-aware sbatch wrapper.
# Reads CLUSTER_PARTITION from .env and passes it to sbatch, so the
# same submission scripts work on Galvani, Ferranti, and any other cluster
# by changing one variable in .env.
#
# Usage (from repo root):
#   scripts/sbatch.sh scripts/submit_blt.sh 0.25 lcm_blt_25 11:00:00
#   scripts/sbatch.sh scripts/submit_sonar.sh 0.50 lcm_sonar_50
set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
REPO_DIR=$(realpath "$SCRIPT_DIR/..")

[ -f "$REPO_DIR/.env" ] && set -a && source "$REPO_DIR/.env" && set +a

PARTITION=${CLUSTER_PARTITION:-a100-galvani}
MEM=${CLUSTER_MEM:-40G}
mkdir -p "$REPO_DIR/logs"

exec sbatch --partition="$PARTITION" --mem="$MEM" "$@"
