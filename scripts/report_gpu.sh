#!/bin/bash
# Print what the scheduler actually allocated, before the job does any work.
#
# Sourced by every scripts/*.sh that runs a model:
#   source "$REPO_DIR/scripts/report_gpu.sh"
#
# The python scripts report the device torch resolved ("[device] cuda:0 | ...").
# This reports the device the *node* is offering, so the two can be compared: a
# `--gres=gpu:1` that silently did not take shows up here as no visible GPU,
# long before the job is twenty hours slower than expected.

echo "GPU allocation:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader 2>/dev/null | sed 's/^/  /' \
        || echo "  nvidia-smi failed -- no GPU visible to this job (CPU fallback)"
else
    echo "  nvidia-smi not on PATH -- no GPU visible to this job (CPU fallback)"
fi
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)}"
echo "  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-(unset)}"
