#!/bin/bash
#SBATCH -J eval-blt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Evaluate a trained BLT-LCM checkpoint (NN retrieval → BLEU / chrF++ / TER / METEOR / COMET).
#
# Usage (via wrapper — recommended):
#   scripts/sbatch.sh scripts/eval_blt.sh [fraction] [run_name]
#   scripts/sbatch.sh scripts/eval_blt.sh 0.25 lcm_blt_25
#   scripts/sbatch.sh scripts/eval_blt.sh 0.50 lcm_blt_50
#   scripts/sbatch.sh scripts/eval_blt.sh 0.80 lcm_blt_80

FRACTION=${1:-0.25}
RUN_NAME=${2:-"lcm_blt_$(echo $FRACTION | tr -d '.')"}
FRAC_TAG=$(echo "$FRACTION" | tr -d '.')

REPO_DIR=${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")/..")}

cd "$REPO_DIR"

mkdir -p logs results

set -a && source .env && set +a

echo "Job started:  $(date)"
START=$(date +%s)

source "$REPO_DIR/scripts/report_gpu.sh"

uv run --frozen lcm_scripts/eval_lcm_blt.py \
    --lcm_checkpoint "lcm_models/${RUN_NAME}/lcm_blt_best.pth" \
    --entropy_model patching_scratch/entropy_model_marathi.pt \
    --embed_cache "embeddings/blt_embeddings_frac${FRAC_TAG}.pth" \
    --fraction "$FRACTION" \
    --out_csv "results/blt_lcm_${FRAC_TAG}_metrics.csv"

echo "Job finished: $(date)"
echo "Total elapsed: $(( $(date +%s) - START ))s"
