# BLT-LCM Marathi Machine Translation

This project implements BLT-LCM (Byte Latent Transformer - Large Concept Model) for script-agnostic concept extraction in Marathi machine translation, and provides tooling to train and evaluate two concept sources:

- SONAR-lite (in-repo SONAR-like byte encoder → bottleneck → decoder)
- BLT-derived patch embeddings (using the entropy model in `patching_scratch`)

Key scripts live under `lcm_scripts/`.

## Installation

1. Clone the repository:

```bash
git clone <repo-url>
cd BLT-LCM
```

2. Install with [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Or with plain pip:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

## Environment setup

Copy the template and fill in your credentials:

```bash
cp .env.example .env
```

Required variables in `.env`:

| Variable | Description |
|---|---|
| `WANDB_API_KEY` | Your W&B API key |
| `HF_TOKEN` | Your Hugging Face token (for higher rate limits) |
| `WANDB_ENTITY` | Your W&B team or username (e.g. `fyp-team-2513`) |
| `HF_HOME` | HuggingFace cache directory (use a shared path on HPC, e.g. `$SCRATCH/.cache/huggingface`) |
| `HF_DATASETS_OFFLINE` | Set to `1` on compute nodes without internet, `0` on login nodes |

All training scripts load `.env` automatically at startup.

## Dataset

The training scripts use the Hugging Face dataset `ParamTh/BhashaSetu` (2,170,000 Marathi sentences, train split).

**On HPC clusters** (compute nodes typically have no internet access): download the dataset once from a login node, then run training offline.

```bash
# On the login node (internet access required):
uv run scripts/download_dataset.py

# Then set HF_DATASETS_OFFLINE=1 in your .env before running on compute nodes.
```

If you want to use your own data, pass `--data_path` to the training scripts.

## Testing

```bash
uv pip install -e ".[test]"
pytest tests/
```

## Important files / scripts

- `lcm_scripts/train_sonar.py` — pretrain SONAR-lite (denoising AE + MSE bottleneck distillation). Supports `--wandb` logging. Streams sentences from `ParamTh/BhashaSetu` by default.
- `lcm_scripts/train_lcm_sonar.py` — encode documents with SONAR-lite and train BaseLCM on SONAR embeddings.
- `lcm_scripts/train_lcm_blt.py` — compute BLT sentence embeddings (via `lcm_scripts/blt_loader.py` and `patching_scratch/entropy_model_marathi.pt`) and train BaseLCM on BLT embeddings.
- `lcm_scripts/blt_loader.py` — BLT loader that uses the entropy model checkpoint to extract patch embeddings and aggregate them to sentence embeddings (mean pooling by default).
- `lcm_scripts/eval_metrics.py` and `lcm_scripts/eval_runner.py` — evaluation metrics wrappers (BLEU/chrF/TER via sacrebleu, METEOR via NLTK, optional COMET) and runner for noisy-input tests.
- `lcm_scripts/experiment_config.py` — helpers for YAML configs, TensorBoard (`setup_logging`), W&B (`setup_wandb`), and a VRAM check helper.
- `patching_scratch/entropy_model_marathi.pt` — example entropy model checkpoint used by the BLT loader.

## Training on Dataset Subsets

To train on specific percentages of the full dataset (~2.78M rows from `ParamTh/BhashaSetu`), use the following commands. These use `--fraction` to select the subset and `--epochs 1` for a single epoch approximation of max_steps.

Note: If your dataset contains one sentence per row (sentence-level corpus) the script will automatically group consecutive sentences into pseudo-documents. To disable this automatic grouping and keep strict per-row documents, pass `--no_grouping`. The default group size is 4.

### 25% Subset
- LCM on SONAR embeddings:
  ```bash
  uv run lcm_scripts/train_lcm_sonar.py --fraction 0.25 --epochs 1 --batch_size 8 --log_dir runs/lcm_sonar_25 --wandb --wandb_project "BLT-LCM" --wandb_name "lcm_sonar_25"
  ```
- LCM on BLT embeddings:
  ```bash
  uv run lcm_scripts/train_lcm_blt.py --entropy_model patching_scratch/entropy_model_marathi.pt --fraction 0.25 --epochs 1 --batch_size 8 --log_dir runs/lcm_blt_25 --wandb --wandb_project "BLT-LCM" --wandb_name "lcm_blt_25"
  ```

### 50% Subset
- LCM on SONAR embeddings:
  ```bash
  uv run lcm_scripts/train_lcm_sonar.py --fraction 0.5 --epochs 1 --batch_size 8 --log_dir runs/lcm_sonar_50 --wandb --wandb_project "BLT-LCM" --wandb_name "lcm_sonar_50"
  ```
- LCM on BLT embeddings:
  ```bash
  uv run lcm_scripts/train_lcm_blt.py --entropy_model patching_scratch/entropy_model_marathi.pt --fraction 0.5 --epochs 1 --batch_size 8 --log_dir runs/lcm_blt_50 --wandb --wandb_project "BLT-LCM" --wandb_name "lcm_blt_50"
  ```

### 80% Subset
- LCM on SONAR embeddings:
  ```bash
  uv run lcm_scripts/train_lcm_sonar.py --fraction 0.8 --epochs 1 --batch_size 8 --log_dir runs/lcm_sonar_80 --wandb --wandb_project "BLT-LCM" --wandb_name "lcm_sonar_80"
  ```
- LCM on BLT embeddings:
  ```bash
  uv run lcm_scripts/train_lcm_blt.py --entropy_model patching_scratch/entropy_model_marathi.pt --fraction 0.8 --epochs 1 --batch_size 8 --log_dir runs/lcm_blt_80 --wandb --wandb_project "BLT-LCM" --wandb_name "lcm_blt_80"
  ```

4) Fine-tune LCM on BLT embeddings — requires a pre-trained checkpoint:
```bash
uv run lcm_scripts/finetune_lcm.py --checkpoint lcm_models/lcm_blt_best.pth --entropy_model patching_scratch/entropy_model_marathi.pt --num_docs 100 --epochs 3 --lr 1e-5 --log_dir runs/lcm_finetune
```
Options for fine-tuning:
- `--freeze_prenet`: Freeze the input projection layers
- `--freeze_postnet`: Freeze the output projection layers  
- `--freeze_layers N`: Freeze the first N transformer layers
- `--lr`: Set a lower learning rate (default 1e-5 for fine-tuning)
- `--lora`: Enable LoRA fine-tuning for parameter efficiency
- `--qlora`: Enable QLoRA (4-bit quantization + LoRA) for memory efficiency
- `--lora_rank`: LoRA rank (default 8)
- `--lora_alpha`: LoRA alpha (default 32)
- `--target_modules`: Modules to apply LoRA (default ["linear"] for all Linear layers)

5) Run evaluation (generate hypothesis/reference files first, one sentence per line):

```bash
uv run lcm_scripts/eval_runner.py --hyp_file outputs/hyp.txt --ref_file outputs/ref.txt --out_csv results/eval_results.csv --seeds 42 43 44 --noisy_probs 0.0 0.1 0.2
```

5) Fertility audit (example interactive usage):

```py
from lcm_scripts.fertility_audit import compute_fertility
avg, counts = compute_fertility(open('data/sentences.txt').read().splitlines())
print(avg)
```

## Notes and caveats

- The repository provides a SONAR-lite implementation (in `lcm_scripts/sonar_module.py`) because the original SONAR package may not be available in all environments.
- COMET metric requires a COMET model name or local checkpoint and the `comet` package; if not available COMET scores will be NaN and a warning is printed.
- W&B integration is implemented: call training scripts with `--wandb` and set `--wandb_project`/`--wandb_name` as needed. `--wandb_entity` defaults to the `WANDB_ENTITY` environment variable (set it in `.env`). No need to run `wandb login` separately if `WANDB_API_KEY` is in `.env`.
- The sentence splitting in the training scripts is simple and may be suboptimal for publication-grade experiments; replace it with a more robust sentence splitter if needed.
