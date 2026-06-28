#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
REPO_DIR=$(realpath "$SCRIPT_DIR/..")

cd "$REPO_DIR"

echo "Building lcm-sonar.sif from apptainer.def ..."
echo "This requires Apptainer/Singularity to be installed and may take 15-30 minutes."
echo ""

APPTAINER_CMD=$(command -v apptainer || command -v singularity || { echo "Neither apptainer nor singularity found" >&2; exit 1; })
"$APPTAINER_CMD" build lcm-sonar.sif apptainer.def

echo ""
echo "Build complete: $REPO_DIR/lcm-sonar.sif"
echo ""
echo "To use with Slurm:"
echo "  export APPTAINER_IMAGE=$REPO_DIR/lcm-sonar.sif"
echo "  scripts/sbatch.sh scripts/submit_sonar.sh 0.25 lcm_sonar_25"
echo ""
echo "Or set APPTAINER_IMAGE in your .env to a shared path."
