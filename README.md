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
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Or with plain pip:

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

This project currently supports Python 3.10 through 3.12. On Windows,
using Python 3.13 can force some dependencies such as NumPy 1.x to build
from source and fail if the Microsoft C++ build tools are not available.

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

Scripts that use Indic NLP morphology (for example `lcm_scripts/fertility_audit.py`) also require the external Indic NLP resource files. Download the resources once and set `INDIC_RESOURCES_PATH` before running those scripts.

On Linux/macOS or Git Bash, from the repository root:

```bash
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
export INDIC_RESOURCES_PATH="$PWD/indic_nlp_resources"
uv run lcm_scripts/fertility_audit.py
```

On Windows PowerShell, from the repository root:

```powershell
git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
$env:INDIC_RESOURCES_PATH = "$PWD\indic_nlp_resources"
uv run lcm_scripts/fertility_audit.py
```

After the variable is set, the other morphology scripts can be run the same way:

```bash
uv run morpheme_alignment/morpheme_boundary_alignment.py
uv run fixed_chunk_ablation/fixed_chunk_ablation.py
uv run sweep_threshold/sweep_entropy_threshold.py
```

If `INDIC_RESOURCES_PATH` is not set, these scripts look for `./indic_nlp_resources`, `./resources/indic_nlp_resources`, and `~/indic_nlp_resources` before trying the legacy `D:/phase2/indic_nlp_resources` path.

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


## Tokenization and Statistics

### Patch compression ratio by morpheme class

Use `tokenization_statistics/patch_compression_by_morpheme_class.py` to compare per-sentence BLT patch counts with BPE token counts, then aggregate the comparison by morpheme class from the fertility audit detail file. The script writes a per-sentence CSV, a summary JSON, and a publication-ready PNG.

With a BLT patch JSONL and tokenizer file:

```bash
python tokenization_statistics/patch_compression_by_morpheme_class.py \
  --blt-jsonl blt_marathi_patched.jsonl \
  --morpheme-detail results/fertility_by_class_detail.jsonl \
  --bpe-tokenizer ../Tokeniser_Retrained/tokenizer.json
```

With the existing model-comparison scores file:

```bash
python tokenization_statistics/patch_compression_by_morpheme_class.py \
  --scores-jsonl error_analysis/all_sentences_scores.jsonl \
  --morpheme-detail results/fertility_by_class_detail.jsonl \
  --bpe-column tok_ret
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

## Quick Start: BLT LCM Full Training Command

Use this command to run a full training job with periodic autosave and checkpoint cleanup. It sets a model output directory, saves periodic checkpoints every 1000 steps and keeps the 5 most recent periodic checkpoints:

Single-line example:

```bash
python lcm_scripts/train_lcm_blt.py --entropy_model patching_scratch/entropy_model_marathi.pt --fraction 0.5 --epochs 15 --batch_size 32 --log_dir runs/lcm_blt_50 --wandb --wandb_project "BLT-LCM" --wandb_name "lcm_blt_50" --model_dir lcm_models --save_interval_steps 1000 --max_checkpoints 5
```

Notes:
- `--model_dir` controls where per-epoch/best and periodic checkpoints are saved (default `lcm_models`).
- `--save_interval_steps` (default 1000) enables periodic autosaves; set to `0` to disable.
- `--max_checkpoints` (default 5) limits how many periodic `lcm_blt_step*.pth` files are kept.
- If you see errors about temporary directories (`No usable temporary directory found`), set a writable temp directory before running, e.g. `export TMPDIR=~/tmp && mkdir -p $TMPDIR`.

If you want numeric-sorted cleanup, save-on-interrupt behavior, compression, or sharded checkpoints, check the script or ask to enable those options.

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
uv run lcm_scripts/finetune_lcm.py --checkpoint lcm_models/lcm_blt_best.pth --entropy_model patching_scratch/entropy_model_marathi.pt --fraction 0.25 --epochs 3 --lr 1e-5 --log_dir runs/lcm_finetune
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

5) Fertility audit and fertility-vs-Δ chrF++ scatter plot

Run the fertility audit first to create the per-class fertility summary and word-level sentence/class detail files:

```bash
uv run lcm_scripts/fertility_audit.py
```

Then generate the Day 6 fertility-vs-gain table and scatter plot. The BPE-LCM, BLT-LCM, and reference files must be aligned one sentence per line, and their line order must match the sentence IDs in `results/fertility_by_class_detail.jsonl`.

```bash
uv run lcm_scripts/fertility_chrf_scatter.py \
  --fertility_json results/fertility_by_class.json \
  --fertility_detail_jsonl results/fertility_by_class_detail.jsonl \
  --bpe_hyp_file outputs/bpe_lcm.hyp.txt \
  --blt_hyp_file outputs/blt_lcm.hyp.txt \
  --ref_file outputs/ref.txt \
  --out_csv results/fertility_chrf_delta_by_class.csv \
  --out_summary results/fertility_chrf_delta_summary.json \
  --out_plot results/fertility_chrf_delta_scatter.png
```

This writes:

- `results/fertility_chrf_delta_by_class.csv` — per morpheme class λ, BPE-LCM chrF++, BLT-LCM chrF++, Δ chrF++, chrF error ratio, and the empirical bound diagnostic.
- `results/fertility_chrf_delta_summary.json` — empirical validation summary with bound-satisfaction rate plus Pearson/Spearman correlations between λ and Δ chrF++.
- `results/fertility_chrf_delta_scatter.png` — scatter plot with λ on the x-axis and BLT-LCM minus BPE-LCM Δ chrF++ on the y-axis.

Optional flags:

- `--include_other` adds the catch-all `other` morpheme class to the CSV and plot.
- `--bound_exponent 0.5` changes α in the bound diagnostic `E(BLT) ≤ λ^-α · E(BPE)`; the default is `1.0`.

## Notes and caveats

- The repository provides a SONAR-lite implementation (in `lcm_scripts/sonar_module.py`) because the original SONAR package may not be available in all environments.
- COMET metric requires a COMET model name or local checkpoint and the `comet` package; if not available COMET scores will be NaN and a warning is printed.
- W&B integration is implemented: call training scripts with `--wandb` and set `--wandb_project`/`--wandb_name` as needed. `--wandb_entity` defaults to the `WANDB_ENTITY` environment variable (set it in `.env`). No need to run `wandb login` separately if `WANDB_API_KEY` is in `.env`.
- The sentence splitting in the training scripts is simple and may be suboptimal for publication-grade experiments; replace it with a more robust sentence splitter if needed.

## BhashaSetu baseline training and noisy benchmarking

The repository now includes end-to-end scripts for the requested BhashaSetu subset experiments. Each script trains on a deterministic shuffled fraction of `ParamTh/BhashaSetu` and writes BLEU, chrF++ and TER results for clean, 10% noisy and 20% noisy inputs.

### One-command benchmark sweep

Run the BPE Transformer, BPE-LCM, BPE Llama 8B, and SONAR-LCM baselines on 25%, 50% and 80% of the dataset:

```bash
uv run lcm_scripts/benchmark_bhashasetu_models.py \
  --models bpe_transformer bpe_lcm bpe_llama8b sonar_lcm \
  --fractions 0.25 0.50 0.80 \
  --noise_levels 0.0 0.10 0.20 \
  --epochs 1 \
  --out_dir runs/bhashasetu_benchmarks
```

The orchestrator creates one metrics CSV per model/fraction and a combined `runs/bhashasetu_benchmarks/summary_metrics.csv` with columns `model`, `fraction`, `noise`, `BLEU`, `chrF++` and `TER`.

### Individual baselines

- **BPE + Transformer** trains a SentencePiece BPE tokenizer and a PyTorch encoder-decoder Transformer from scratch:

  ```bash
  uv run lcm_scripts/train_bpe_transformer.py \
    --fraction 0.25 \
    --epochs 3 \
    --noise_levels 0.0 0.10 0.20 \
    --out_dir runs/bpe_transformer_25
  ```


- **BPE + LCM baseline** trains a SentencePiece BPE tokenizer, mean-pools learned BPE token embeddings into sentence vectors, trains `BaseLCM` for next-sentence embedding prediction, and decodes with nearest-neighbor retrieval:

  ```bash
  uv run lcm_scripts/train_lcm_bpe.py \
    --fraction 0.25 \
    --epochs 2 \
    --noise_levels 0.0 0.10 0.20 \
    --out_dir runs/lcm_bpe_25
  ```

  Slurm/W&B launcher:

  ```bash
  sbatch scripts/submit_bpe_lcm.sh 0.25 lcm_bpe_25
  ```

- **BPE + Llama 8B** fine-tunes a Llama-family 8B causal LM with LoRA/QLoRA and evaluates generated translations:

  ```bash
  uv run lcm_scripts/train_bpe_llama8b.py \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --fraction 0.25 \
    --qlora \
    --epochs 1 \
    --noise_levels 0.0 0.10 0.20 \
    --out_dir runs/bpe_llama8b_25
  ```

- **SONAR embedding + LCM baseline** encodes Marathi sentence documents with the SONAR-like loader, trains `BaseLCM`, decodes with nearest-neighbor retrieval and computes the same metrics:

  ```bash
  uv run lcm_scripts/train_lcm_sonar.py \
    --fraction 0.25 \
    --epochs 2 \
    --noise_levels 0.0 0.10 0.20 \
    --out_dir runs/lcm_sonar_25
  ```

If your local BhashaSetu export uses non-default parallel column names, pass `--src_col` and `--tgt_col` to the parallel-text BPE scripts or the benchmark orchestrator.

### Recover SONAR-LCM baseline metrics from an existing checkpoint

If a SONAR-LCM run produced checkpoints or TensorBoard event files but did not write a metrics CSV, evaluate the saved checkpoint directly. Use the same fraction and model dimensions that were used during training so the deterministic BhashaSetu split and `BaseLCM` shape match the checkpoint.

```bash
uv run lcm_scripts/eval_lcm_sonar.py \
  --checkpoint runs/lcm_sonar/lcm_sonar_fraction0.25_epoch1.pth \
  --fraction 0.25 \
  --eval_docs 100 \
  --noise_levels 0.0 0.10 0.20 \
  --out_csv runs/lcm_sonar/metrics_fraction0.25.csv
```

The output CSV contains `model`, `fraction`, `noise`, `checkpoint`, `num_predictions`, `BLEU`, `chrF++` and `TER`, which makes the SONAR-LCM baseline comparable to the BPE and BLT-LCM CSVs at the same data fraction.

### Three-way BLT-LCM vs BPE-LCM vs SONAR-LCM comparison

After all per-model CSVs exist for the same fraction, combine just the headline metrics into one comparison table:

```bash
uv run lcm_scripts/compare_bhashasetu_metrics.py \
  --inputs \
    runs/blt_lcm_25/metrics_fraction0.25.csv \
    runs/lcm_bpe_25/metrics_fraction0.25.csv \
    runs/lcm_sonar/metrics_fraction0.25.csv \
  --fraction 0.25 \
  --models blt_lcm bpe_lcm sonar_lcm \
  --metrics chrF++ BLEU \
  --strict \
  --out_csv results/three_way_lcm_fraction0.25_chrf_bleu.csv
```

Use `--strict` for the paper table: it fails if any requested model is missing for a noise level, preventing an accidental two-way comparison from being reported as a three-way comparison.
