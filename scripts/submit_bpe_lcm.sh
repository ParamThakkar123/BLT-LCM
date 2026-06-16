#!/bin/bash
#SBATCH -J bpe-lcm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=/mnt/lustre/home/gehler/gfh098/dev/BLT-LCM/logs/bpe_lcm_%j.out
#SBATCH --error=/mnt/lustre/home/gehler/gfh098/dev/BLT-LCM/logs/bpe_lcm_%j.err

# Usage:
#   sbatch scripts/submit_bpe_lcm.sh [fraction] [run_name]
#   sbatch scripts/submit_bpe_lcm.sh 0.25 lcm_bpe_25
#   sbatch scripts/submit_bpe_lcm.sh 0.50 lcm_bpe_50
#   sbatch scripts/submit_bpe_lcm.sh 0.80 lcm_bpe_80

FRACTION=${1:-0.25}
RUN_NAME=${2:-"lcm_bpe_$(echo $FRACTION | tr -d '.')"}

REPO_DIR=/mnt/lustre/home/gehler/gfh098/dev/BLT-LCM
cd "$REPO_DIR"

mkdir -p logs

set -a && source .env && set +a

echo "Job started:  $(date)"
START=$(date +%s)

uv run lcm_scripts/train_lcm_bpe.py \
    --fraction "$FRACTION" \
    --epochs 1 \
    --batch_size 32 \
    --log_dir "runs/${RUN_NAME}" \
    --wandb \
    --wandb_project "BLT-LCM" \
    --wandb_name "$RUN_NAME"

echo "Job finished: $(date)"
echo "Total elapsed: $(( $(date +%s) - START ))s"
