#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(realpath "$(dirname "$0")")
REPO_DIR=$(realpath "$SCRIPT_DIR/..")

cd "$REPO_DIR"

[ -f .env ] && { set -a && source .env && set +a; }

echo "Building lcm-sonar.sif from apptainer.def ..."
echo "This requires Apptainer/Singularity to be installed and may take 15-30 minutes."
echo ""

APPTAINER_CMD=$(command -v apptainer || command -v singularity || { echo "Neither apptainer nor singularity found" >&2; exit 1; })

# mksquashfs defaults to one compression thread per detected CPU thread, each
# holding its own buffer -- on a high-core-count login node (e.g. Ferranti's
# 128-thread nodes) that can exceed a per-user memory cap and get SIGKILL'd by
# the OOM killer (exit status 137) well before the machine's total RAM is
# actually exhausted. Cap it; override via MKSQUASHFS_ARGS in .env if needed.
MKSQUASHFS_ARGS=${MKSQUASHFS_ARGS:-"-processors 4 -mem 2048M"}
"$APPTAINER_CMD" build --mksquashfs-args="$MKSQUASHFS_ARGS" lcm-sonar.sif apptainer.def

echo ""
echo "Build complete: $REPO_DIR/lcm-sonar.sif"
echo ""
echo "To use with Slurm:"
echo "  export APPTAINER_IMAGE=$REPO_DIR/lcm-sonar.sif"
echo "  scripts/sbatch.sh scripts/submit_sonar.sh 0.25 lcm_sonar_25"
echo ""
echo "Or set APPTAINER_IMAGE in your .env to a shared path."
