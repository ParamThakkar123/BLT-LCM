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

2. Create a Python environment and install the package (editable):

```bash
python -m venv .venv
# Activate: source .venv/bin/activate (Linux/macOS) or .venv\Scripts\activate (Windows)
pip install -e .
```

3. Optional dependencies for full evaluation and logging:

```bash
pip install sacrebleu nltk wandb
# COMET is optional and only required if you want COMET scores
pip install comet-ml
```

## Testing

Install test dependencies:
```bash
pip install -e ".[test]"
```

Run tests:
```bash
pytest tests/
```

## Dataset

The training scripts stream Marathi text from the Hugging Face dataset `ParamTh/BhashaSetu` (train split) and read the `marathi` column. Sentence splitting is a simple period/newline split performed inside the scripts. See `lcm_scripts/train_sonar.py` and `lcm_scripts/train_lcm_*.py` for details.

If you want to use your own data, supply files and update the scripts or modify the dataset loading logic.

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
  python lcm_scripts/train_lcm_sonar.py --fraction 0.25 --epochs 1 --batch_size 8 --log_dir runs/lcm_sonar_25 --wandb --wandb_project "BLT-LCM" --wandb_name "lcm_sonar_25"
  ```
- LCM on BLT embeddings:
  ```bash
  python lcm_scripts/train_lcm_blt.py --entropy_model ../patching_scratch/entropy_model_marathi.pt --fraction 0.25 --epochs 1 --batch_size 8 --log_dir runs/lcm_blt_25 --wandb --wandb_project "BLT-LCM" --wandb_name "lcm_blt_25"
  ```

### 50% Subset
- LCM on SONAR embeddings:
  ```bash
  python lcm_scripts/train_lcm_sonar.py --fraction 0.5 --epochs 1 --batch_size 8 --log_dir runs/lcm_sonar_50 --wandb --wandb_project "BLT-LCM" --wandb_name "lcm_sonar_50"
  ```
- LCM on BLT embeddings:
  ```bash
  python lcm_scripts/train_lcm_blt.py --entropy_model ../patching_scratch/entropy_model_marathi.pt --fraction 0.5 --epochs 1 --batch_size 8 --log_dir runs/lcm_blt_50 --wandb --wandb_project "BLT-LCM" --wandb_name "lcm_blt_50"
  ```

### 80% Subset
- LCM on SONAR embeddings:
  ```bash
  python lcm_scripts/train_lcm_sonar.py --fraction 0.8 --epochs 1 --batch_size 8 --log_dir runs/lcm_sonar_80 --wandb --wandb_project "BLT-LCM" --wandb_name "lcm_sonar_80"
  ```
- LCM on BLT embeddings:
  ```bash
  python lcm_scripts/train_lcm_blt.py --entropy_model ../patching_scratch/entropy_model_marathi.pt --fraction 0.8 --epochs 1 --batch_size 8 --log_dir runs/lcm_blt_80 --wandb --wandb_project "BLT-LCM" --wandb_name "lcm_blt_80"
  ```

4) Fine-tune LCM on BLT embeddings — requires a pre-trained checkpoint:
```bash
python lcm_scripts/finetune_lcm.py --checkpoint lcm_models/lcm_blt_best.pth --entropy_model patching_scratch/entropy_model_marathi.pt --num_docs 100 --epochs 3 --lr 1e-5 --log_dir runs/lcm_finetune
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
python lcm_scripts/eval_runner.py --hyp_file outputs/hyp.txt --ref_file outputs/ref.txt --out_csv results/eval_results.csv --seeds 42 43 44 --noisy_probs 0.0 0.1 0.2
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
- W&B integration is implemented: call training scripts with `--wandb` and set `--wandb_project`/`--wandb_name`/`--wandb_entity` as needed (defaults to team workspace "fyp-team-2513"). You must run `wandb login` before uploading artifacts.
- The sentence splitting in the training scripts is simple and may be suboptimal for publication-grade experiments; replace it with a more robust sentence splitter if needed.
