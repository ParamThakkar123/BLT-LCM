# Apptainer Container for BLT-LCM

An [Apptainer](https://apptainer.org/) (formerly Singularity) definition for running lcm-sonar on HPC systems without `uv` or a local Python environment. The container packages CUDA 12.1, Python 3.10, PyTorch, and all project dependencies. Source code is bind-mounted at runtime so the container doesn't need rebuilding for code changes.

## Quickstart

```bash
# 1. Build the container (needs internet; ~15-30 min; run on a login node)
./scripts/build_apptainer.sh

# 2. Submit Slurm jobs as usual — the container is used automatically
scripts/sbatch.sh scripts/submit_sonar.sh 0.25 lcm_sonar_25
scripts/sbatch.sh scripts/submit_blt.sh 0.25 lcm_blt_25
```

## Building

```bash
# From the repo root:
apptainer build lcm-sonar.sif apptainer.def
```

Or use the helper:

```bash
./scripts/build_apptainer.sh
```

The output is `lcm-sonar.sif` in the repo root (also ignored by git via `*.sif`).

## How It Works

The Slurm scripts in `scripts/` have been updated to call `apptainer exec` instead of `uv run`. Each script:

1. Sources `.env` for API tokens (W&B, HF)
2. Mounts the repo at `/workspace` inside the container
3. Sets the working directory to `/workspace`
4. Passes through the NVIDIA GPU via `--nv`

The container itself contains only system packages and Python libraries — the source code, checkpoints, embeddings, and logs all live on the host filesystem via bind mount.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `APPTAINER_IMAGE` | `$REPO_DIR/lcm-sonar.sif` | Path to the `.sif` container image |

Set `APPTAINER_IMAGE` in `.env` to point to a shared filesystem location:

```bash
APPTAINER_IMAGE=/shared/containers/lcm-sonar.sif
```

## Caching Directories

Cache dirs (`HF_HOME`, `TORCH_HOME`, etc.) default to subdirectories of `/workspace/.cache` inside the container. To use a shared cluster-wide cache instead, set the relevant `*_HOME` variables in `.env` — Apptainer inherits host environment variables by default.

## Running Outside Slurm

```bash
export APPTAINER_IMAGE=./lcm-sonar.sif
apptainer exec --nv \
    --bind "$PWD:/workspace" \
    --pwd /workspace \
    "$APPTAINER_IMAGE" \
    python -u lcm_scripts/train_lcm_sonar.py \
    --fraction 0.25 --epochs 2 --batch_size 8 \
    --noise_levels 0.0 0.1 0.2 \
    --out_dir runs/lcm_sonar_25 --wandb --wandb_project BLT-LCM --wandb_name lcm_sonar_25
```

Or set `APPTAINER_IMAGE` in `.env` and keep using the Slurm scripts directly:

```bash
# These now run via Apptainer instead of uv:
./scripts/sbatch.sh scripts/submit_sonar.sh 0.25 lcm_sonar_25
./scripts/sbatch.sh scripts/submit_blt.sh 0.25 lcm_blt_25
```

## Reverting to `uv run`

To go back to using `uv run` directly (e.g. for local development), set `APPTAINER_IMAGE` to an empty value or unset it:

```bash
unset APPTAINER_IMAGE
```

Or comment it out in `.env`. The Slurm scripts check `APPTAINER_IMAGE` — if unset or empty, they fall back to `$REPO_DIR/lcm-sonar.sif`. To permanently revert, edit the scripts and change `apptainer exec` back to `uv run`.

## Container Contents

- **Base:** `nvidia/cuda:12.1.0-runtime-ubuntu22.04`
- **Python:** 3.10 (from deadsnakes PPA)
- **ML:** PyTorch 2.x (CUDA 12.1), transformers, peft, bitsandbytes
- **Metrics:** sacrebleu, unbabel-comet, nltk (METEOR)
- **Embeddings:** sonar-space, fairseq2, sentencepiece, tiktoken
- **Tracking:** wandb, tensorboard
- **Test:** pytest
