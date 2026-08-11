#!/bin/bash
#SBATCH -J blt-encode
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Pre-compute and cache BLT embeddings for a given dataset fraction.
# Run once per fraction before submitting training jobs.
#
# Usage (via wrapper — recommended):
#   scripts/sbatch.sh scripts/encode_blt.sh [fraction]
#   scripts/sbatch.sh scripts/encode_blt.sh 0.25
#   scripts/sbatch.sh scripts/encode_blt.sh 0.50
#   scripts/sbatch.sh scripts/encode_blt.sh 0.80

FRACTION=${1:-1.0}
CACHE_NAME="blt_embeddings_frac$(echo $FRACTION | tr -d '.').pth"

REPO_DIR=${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")/..")}

cd "$REPO_DIR"

mkdir -p logs embeddings

set -a && source .env && set +a

echo "Job started:  $(date)"
START=$(date +%s)

source "$REPO_DIR/scripts/report_gpu.sh"

uv run --frozen lcm_scripts/train_lcm_blt.py \
    --entropy_model patching_scratch/entropy_model_marathi.pt \
    --fraction "$FRACTION" \
    --epochs 0 \
    --batch_size 8 \
    --embed_cache "embeddings/${CACHE_NAME}"

echo "Job finished: $(date)"
echo "Total elapsed: $(( $(date +%s) - START ))s"
echo "Cache saved to: embeddings/${CACHE_NAME}"
