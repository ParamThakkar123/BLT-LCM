#!/bin/bash
#SBATCH -J metric-suite
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Run the full BLT-LCM metric suite (BLEU / chrF++ / METEOR / COMET / TER).
# Requires hypothesis files to already exist in HYP_DIR.
#
# Usage (via wrapper — recommended):
#   scripts/sbatch.sh scripts/metric_suite.sh [fraction] [run_name]
#   scripts/sbatch.sh scripts/metric_suite.sh 0.25 lcm_blt_25
#   scripts/sbatch.sh scripts/metric_suite.sh 0.50 lcm_blt_50
#   scripts/sbatch.sh scripts/metric_suite.sh 0.80 lcm_blt_80

FRACTION=${1:-0.25}
RUN_NAME=${2:-"lcm_blt_$(echo $FRACTION | tr -d '.')"}
FRAC_TAG=$(echo "$FRACTION" | tr -d '.')

REPO_DIR=${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")/..")}
cd "$REPO_DIR"

mkdir -p logs results

set -a && source .env && set +a

echo "Job started:  $(date)"
START=$(date +%s)

uv run lcm_scripts/run_metric_suite.py \
    --checkpoints_dir "lcm_models/${RUN_NAME}" \
    --checkpoint_glob "lcm_blt_best.pth" \
    --hyp_dir "outputs/${RUN_NAME}_hyps" \
    --ref_file "outputs/ref_${FRAC_TAG}.txt" \
    --out_csv "results/blt_lcm_${FRAC_TAG}_full_metrics.csv" \
    --comet_model Unbabel/wmt22-comet-da

echo "Job finished: $(date)"
echo "Total elapsed: $(( $(date +%s) - START ))s"
