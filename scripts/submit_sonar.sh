#!/bin/bash
#SBATCH -J lcm-sonar
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --output=/mnt/lustre/home/gehler/gfh098/dev/BLT-LCM/logs/sonar_%j.out
#SBATCH --error=/mnt/lustre/home/gehler/gfh098/dev/BLT-LCM/logs/sonar_%j.err

# Usage:
#   sbatch scripts/submit_sonar.sh [fraction] [run_name]
#   sbatch scripts/submit_sonar.sh 0.25 lcm_sonar_25

FRACTION=${1:-0.25}
RUN_NAME=${2:-"lcm_sonar_$(echo $FRACTION | tr -d '.')"}

REPO_DIR=/mnt/lustre/home/gehler/gfh098/dev/BLT-LCM
cd "$REPO_DIR"

mkdir -p logs

set -a && source .env && set +a

uv run lcm_scripts/train_lcm_sonar.py \
    --fraction "$FRACTION" \
    --epochs 1 \
    --batch_size 8 \
    --log_dir "runs/${RUN_NAME}" \
    --wandb \
    --wandb_project "BLT-LCM" \
    --wandb_name "$RUN_NAME"
