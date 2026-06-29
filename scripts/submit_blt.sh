#!/bin/bash
set -euo pipefail
#SBATCH -J lcm-blt
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
#   scripts/sbatch.sh scripts/submit_blt.sh [fraction] [run_name] [time_limit]
#   scripts/sbatch.sh scripts/submit_blt.sh 0.25 lcm_blt_25 11:00:00
#   scripts/sbatch.sh scripts/submit_blt.sh 0.50 lcm_blt_50 21:00:00
#   scripts/sbatch.sh scripts/submit_blt.sh 0.80 lcm_blt_80 1-09:00

FRACTION=${1:-0.25}
RUN_NAME=${2:-"lcm_blt_$(echo $FRACTION | tr -d '.')"}
FRAC_TAG=$(echo "$FRACTION" | tr -d '.')
CACHE_NAME="blt_embeddings_frac${FRAC_TAG}.pth"

REPO_DIR=${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")/..")}
cd "$REPO_DIR"

mkdir -p logs

set -a && source .env && set +a

echo "Job started:  $(date)"
START=$(date +%s)

uv run lcm_scripts/train_lcm_blt.py \
    --entropy_model patching_scratch/entropy_model_marathi.pt \
    --fraction "$FRACTION" \
    --epochs 2 \
    --batch_size 32 \
    --embed_cache "embeddings/${CACHE_NAME}" \
    --model_dir "lcm_models/${RUN_NAME}" \
    --wandb \
    --wandb_project "BLT-LCM" \
    --wandb_name "$RUN_NAME"

echo "Job finished: $(date)"
echo "Total elapsed: $(( $(date +%s) - START ))s"
