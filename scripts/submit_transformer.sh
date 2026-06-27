#!/bin/bash
#SBATCH -J bpe-transformer
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
#   scripts/sbatch.sh scripts/submit_transformer.sh [fraction] [run_name]
#   scripts/sbatch.sh scripts/submit_transformer.sh 0.25 bpe_transformer_25
#   scripts/sbatch.sh scripts/submit_transformer.sh 0.50 bpe_transformer_50
#   scripts/sbatch.sh scripts/submit_transformer.sh 0.80 bpe_transformer_80

FRACTION=${1:-0.25}
RUN_NAME=${2:-"bpe_transformer_$(echo $FRACTION | tr -d '.')"}

REPO_DIR=$(realpath "$(dirname "$0")/..")
cd "$REPO_DIR"

mkdir -p logs

set -a && source .env && set +a

echo "Job started:  $(date)"
START=$(date +%s)

uv run lcm_scripts/train_bpe_transformer.py \
    --fraction "$FRACTION" \
    --epochs 3 \
    --batch_size 32 \
    --noise_levels 0.0 0.1 0.2 \
    --out_dir "runs/${RUN_NAME}"

echo "Job finished: $(date)"
echo "Total elapsed: $(( $(date +%s) - START ))s"
