#!/bin/bash
#SBATCH -J blt-encode
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=/mnt/lustre/home/gehler/gfh098/dev/BLT-LCM/logs/encode_%j.out
#SBATCH --error=/mnt/lustre/home/gehler/gfh098/dev/BLT-LCM/logs/encode_%j.err

# Pre-compute and cache BLT embeddings for a given dataset fraction.
# Run this once per fraction before submitting training jobs.
#
# Usage:
#   sbatch scripts/encode_blt.sh [fraction]
#   sbatch scripts/encode_blt.sh 0.25
#   sbatch scripts/encode_blt.sh 1.0

FRACTION=${1:-1.0}
CACHE_NAME="blt_embeddings_frac$(echo $FRACTION | tr -d '.').pth"

REPO_DIR=/mnt/lustre/home/gehler/gfh098/dev/BLT-LCM
cd "$REPO_DIR"

mkdir -p logs embeddings

set -a && source .env && set +a

echo "Job started:  $(date)"
START=$(date +%s)

uv run lcm_scripts/train_lcm_blt.py \
    --entropy_model patching_scratch/entropy_model_marathi.pt \
    --fraction "$FRACTION" \
    --epochs 0 \
    --batch_size 8 \
    --embed_cache "embeddings/${CACHE_NAME}"

echo "Job finished: $(date)"
echo "Total elapsed: $(( $(date +%s) - START ))s"
echo "Cache saved to: embeddings/${CACHE_NAME}"
