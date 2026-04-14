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

Note: If you use W&B, run `wandb login` in your environment before enabling `--wandb`.

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

## Quick reproducible smoke run (recommended)

1) Pretrain SONAR-lite (small smoke test):

```bash
python lcm_scripts/train_sonar.py --num_samples 2000 --epochs 2 --batch_size 16 --noise_prob 0.1 --lambda_mse 1.0 --robust_update --log_dir runs/sonar_smoke
```

2) Train LCM on SONAR embeddings (dev):

```bash
python lcm_scripts/train_lcm_sonar.py --num_docs 200 --epochs 2 --batch_size 8 --log_dir runs/lcm_sonar
```

3) Train LCM on BLT embeddings (dev) — requires `patching_scratch/entropy_model_marathi.pt`:

```bash
python lcm_scripts/train_lcm_blt.py --entropy_model patching_scratch/entropy_model_marathi.pt --num_docs 200 --epochs 2 --batch_size 8 --log_dir runs/lcm_blt
```

4) Run evaluation (generate hypothesis/reference files first, one sentence per line):

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
- W&B integration is implemented: call training scripts with `--wandb` and set `--wandb_project`/`--wandb_name`/`--wandb_entity` as needed. You must run `wandb login` before uploading artifacts.
- The sentence splitting in the training scripts is simple and may be suboptimal for publication-grade experiments; replace it with a more robust sentence splitter if needed.
