#!/bin/bash
#SBATCH -J bpe-lcm
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
#   scripts/sbatch.sh scripts/submit_bpe_lcm.sh [fraction] [run_name]
#   scripts/sbatch.sh scripts/submit_bpe_lcm.sh 0.25 lcm_bpe_25
#   scripts/sbatch.sh scripts/submit_bpe_lcm.sh 0.50 lcm_bpe_50
#   scripts/sbatch.sh scripts/submit_bpe_lcm.sh 0.80 lcm_bpe_80

FRACTION=${1:-0.25}
RUN_NAME=${2:-"lcm_bpe_$(echo $FRACTION | tr -d '.')"}

REPO_DIR=${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")/..")}
cd "$REPO_DIR"

mkdir -p logs

set -a && source .env && set +a

echo "Job started:  $(date)"
START=$(date +%s)

uv run lcm_scripts/train_lcm_bpe.py \
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
