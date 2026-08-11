#!/bin/bash
#SBATCH -J eval-sonar
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Evaluate an existing SONAR-LCM checkpoint at all noise levels.
#
# Usage (via wrapper — recommended):
#   scripts/sbatch.sh scripts/eval_sonar.sh [fraction] [run_name] [epochs]
#   scripts/sbatch.sh scripts/eval_sonar.sh 0.25 lcm_sonar_25 2

FRACTION=${1:-0.25}
RUN_NAME=${2:-"lcm_sonar_$(echo $FRACTION | tr -d '.')"}
EPOCHS=${3:-2}
FRAC_TAG=$(echo "$FRACTION" | tr -d '.')

REPO_DIR=${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")/..")}
APPTAINER_IMAGE=${APPTAINER_IMAGE:-"$REPO_DIR/lcm-sonar.sif"}

cd "$REPO_DIR"

mkdir -p logs

set -a && source .env && set +a

# Ensure libcudart.so.12 is visible to fairseq2n native extension
_cuda_lib=$(ldconfig -p 2>/dev/null | awk '/libcudart\.so\.12/{print $NF}' | head -1 | xargs dirname 2>/dev/null)
[ -n "$_cuda_lib" ] && export LD_LIBRARY_PATH="$_cuda_lib:${LD_LIBRARY_PATH:-}"

echo "Job started:  $(date)"
START=$(date +%s)

source "$REPO_DIR/scripts/report_gpu.sh"

apptainer exec --nv \
    --bind "$REPO_DIR:/workspace" \
    --pwd /workspace \
    "$APPTAINER_IMAGE" \
    python -u lcm_scripts/eval_lcm_sonar.py \
    --checkpoint "runs/${RUN_NAME}/lcm_sonar_fraction${FRACTION}_epoch${EPOCHS}.pth" \
    --fraction "$FRACTION" \
    --eval_docs 100 \
    --noise_levels 0.0 0.1 0.2 \
    --out_csv "runs/${RUN_NAME}/metrics_fraction${FRACTION}.csv"

echo "Job finished: $(date)"
echo "Total elapsed: $(( $(date +%s) - START ))s"
