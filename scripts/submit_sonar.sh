#!/bin/bash
#SBATCH -J lcm-sonar
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage (via wrapper — recommended):
#   scripts/sbatch.sh scripts/submit_sonar.sh [fraction] [run_name] [time_limit]
#   scripts/sbatch.sh scripts/submit_sonar.sh 0.25 lcm_sonar_25 11:00:00
#   scripts/sbatch.sh scripts/submit_sonar.sh 0.50 lcm_sonar_50 21:00:00
#   scripts/sbatch.sh scripts/submit_sonar.sh 0.80 lcm_sonar_80 1-09:00

FRACTION=${1:-0.25}
RUN_NAME=${2:-"lcm_sonar_$(echo $FRACTION | tr -d '.')"}

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

apptainer exec --nv \
    --bind "$REPO_DIR:/workspace" \
    --pwd /workspace \
    "$APPTAINER_IMAGE" \
    python -u lcm_scripts/train_lcm_sonar.py \
    --fraction "$FRACTION" \
    --epochs 2 \
    --batch_size 8 \
    --noise_levels 0.0 0.1 0.2 \
    --out_dir "runs/${RUN_NAME}" \
    --wandb \
    --wandb_project "BLT-LCM" \
    --wandb_name "$RUN_NAME"

echo "Job finished: $(date)"
echo "Total elapsed: $(( $(date +%s) - START ))s"
