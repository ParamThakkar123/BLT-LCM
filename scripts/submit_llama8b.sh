#!/bin/bash
#SBATCH -J bpe-llama8b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Train BPE + Llama-8B with LoRA (default: QLoRA 4-bit, ~16GB VRAM).
# Without --qlora the model needs ~48GB VRAM; mem=48G covers both cases.
#
# Usage (via wrapper — recommended):
#   scripts/sbatch.sh scripts/submit_llama8b.sh [fraction] [run_name] [qlora]
#   scripts/sbatch.sh scripts/submit_llama8b.sh 0.25 bpe_llama8b_25 qlora
#   scripts/sbatch.sh scripts/submit_llama8b.sh 0.25 bpe_llama8b_25_full full
#   scripts/sbatch.sh scripts/submit_llama8b.sh 0.50 bpe_llama8b_50_qlora qlora
#   scripts/sbatch.sh scripts/submit_llama8b.sh 0.80 bpe_llama8b_80_qlora qlora

FRACTION=${1:-0.25}
RUN_NAME=${2:-"bpe_llama8b_$(echo $FRACTION | tr -d '.')"}
MODE=${3:-qlora}    # "qlora" (default) or "full"

REPO_DIR=${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "$0")/..")}
cd "$REPO_DIR"

mkdir -p logs

set -a && source .env && set +a

QLORA_FLAG=""
if [ "$MODE" = "qlora" ]; then
    QLORA_FLAG="--qlora"
fi

echo "Job started:  $(date)"
START=$(date +%s)

uv run lcm_scripts/train_bpe_llama8b.py \
    --fraction "$FRACTION" \
    --epochs 1 \
    --batch_size 1 \
    --grad_accum 16 \
    --noise_levels 0.0 0.1 0.2 \
    --out_dir "runs/${RUN_NAME}" \
    $QLORA_FLAG

echo "Job finished: $(date)"
echo "Total elapsed: $(( $(date +%s) - START ))s"
