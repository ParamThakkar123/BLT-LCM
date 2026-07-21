# BLT-LCM Training & Evaluation Scripts

All commands assume the repository root as working directory and use `uv run` or `python`.

---

## 0. Prerequisites

```bash
# Restore deleted BLT-LCM training script (lost in merge conflic

# Set up environment
cp .env.example .env   # edit with your HF/W&B tokens
uv sync                 # install dependencies
```

---

## 1. BLT-LCM

### 1.0 Learn the concept space (pooler + generative decoder) — do this first

BLT concept embeddings are produced by a cross-attention **pooler** over byte
hidden states, and predicted concepts are turned back into text by a generative
**BLTDecoder** (not nearest-neighbor retrieval). Both are trained jointly with a
byte-reconstruction objective (the byte backbone/entropy boundaries stay frozen):

```bash
python lcm_scripts/blt_decoder.py \
  --entropy_model patching_scratch/entropy_model_marathi.pt \
  --num_sentences 50000 --epochs 10 \
  --pooler_save_path lcm_models/blt_pooler.pth \
  --save_path lcm_models/blt_decoder.pth
# Outputs → lcm_models/blt_pooler.pth  (loaded by all BLT training/eval via --pooler)
#           lcm_models/blt_decoder.pth (generative decoder for eval)
```

Train the pooler+decoder BEFORE encoding LCM embeddings, and regenerate any
`--embed_cache` afterwards (concept vectors change when the pooler changes).

### 1.1 Pre-encode BLT embeddings (cache step, ~hours per fraction)

```bash
# 25%  —  ~6h
uv run lcm_scripts/train_lcm_blt.py \
  --entropy_model patching_scratch/entropy_model_marathi.pt \
  --fraction 0.25 --epochs 0 --batch_size 8 \
  --embed_cache embeddings/blt_embeddings_frac025.pth

# 50%  —  ~12h
uv run lcm_scripts/train_lcm_blt.py \
  --entropy_model patching_scratch/entropy_model_marathi.pt \
  --fraction 0.50 --epochs 0 --batch_size 8 \
  --embed_cache embeddings/blt_embeddings_frac050.pth

# 80%  —  ~20h
uv run lcm_scripts/train_lcm_blt.py \
  --entropy_model patching_scratch/entropy_model_marathi.pt \
  --fraction 0.80 --epochs 0 --batch_size 8 \
  --embed_cache embeddings/blt_embeddings_frac080.pth
```

### 1.2 Train BLT-LCM

```bash
# 25%
uv run lcm_scripts/train_lcm_blt.py \
  --entropy_model patching_scratch/entropy_model_marathi.pt \
  --fraction 0.25 --epochs 2 --batch_size 32 \
  --embed_cache embeddings/blt_embeddings_frac025.pth \
  --model_dir lcm_models/blt_lcm_25 --wandb --wandb_project BLT-LCM \
  --wandb_name blt_lcm_25
# Output → lcm_models/blt_lcm_25/lcm_blt_best.pth

# 50%
uv run lcm_scripts/train_lcm_blt.py \
  --entropy_model patching_scratch/entropy_model_marathi.pt \
  --fraction 0.50 --epochs 2 --batch_size 32 \
  --embed_cache embeddings/blt_embeddings_frac050.pth \
  --model_dir lcm_models/blt_lcm_50 --wandb --wandb_project BLT-LCM \
  --wandb_name blt_lcm_50

# 80%
uv run lcm_scripts/train_lcm_blt.py \
  --entropy_model patching_scratch/entropy_model_marathi.pt \
  --fraction 0.80 --epochs 2 --batch_size 32 \
  --embed_cache embeddings/blt_embeddings_frac080.pth \
  --model_dir lcm_models/blt_lcm_80 --wandb --wandb_project BLT-LCM \
  --wandb_name blt_lcm_80
```

### 1.3 Evaluate BLT-LCM (generative decoding → BLEU / chrF++ / TER / METEOR / COMET)

Predicted concepts are decoded to text with the trained `BLTDecoder`
(`--decode_method generative`, the default). Pass the same `--pooler`/`--decoder`
produced in step 1.0 and used to train the LCM. `--decode_method retrieval` is
available only as a nearest-neighbor baseline.

```bash
# 25%  (concept next-sentence task)
uv run lcm_scripts/eval_lcm_blt.py \
  --lcm_checkpoint lcm_models/blt_lcm_25/lcm_blt_best.pth \
  --entropy_model patching_scratch/entropy_model_marathi.pt \
  --pooler lcm_models/blt_pooler.pth \
  --decoder lcm_models/blt_decoder.pth \
  --fraction 0.25 --out_csv results/blt_lcm_25_metrics.csv
```

### 1.5 English → Marathi translation (the MT task, with source-noise levels)

This is the actual translation task (English source → Marathi target), evaluated
across 0/10/20% source-side character noise. It trains `BaseLCM` to map the
source concept to the target concept, then decodes to Marathi generatively.

```bash
uv run lcm_scripts/train_lcm_blt_mt.py \
  --entropy_model patching_scratch/entropy_model_marathi.pt \
  --pooler lcm_models/blt_pooler.pth \
  --decoder lcm_models/blt_decoder.pth \
  --fraction 0.25 --epochs 3 \
  --noise_levels 0.0 0.1 0.2 \
  --out_csv results/blt_lcm_mt_25.csv
# CSV columns: model, fraction, noise, BLEU, chrF++, TER, METEOR, COMET
```

### 1.4 BLT-LCM full evaluation suite (BLEU / chrF++ / METEOR / COMET / TER)

```bash
# First generate hypotheses using evaluate_blt_lcm.py, then run metric suite:

# 25%
uv run lcm_scripts/run_metric_suite.py \
  --checkpoints_dir lcm_models/blt_lcm_25 \
  --checkpoint_glob "lcm_blt_best.pth" \
  --hyp_dir outputs/blt_lcm_25_hyps \
  --ref_file outputs/ref_25.txt \
  --out_csv results/blt_lcm_25_full_metrics.csv \
  --comet_model Unbabel/wmt22-comet-da

# 50%
uv run lcm_scripts/run_metric_suite.py \
  --checkpoints_dir lcm_models/blt_lcm_50 \
  --checkpoint_glob "lcm_blt_best.pth" \
  --hyp_dir outputs/blt_lcm_50_hyps \
  --ref_file outputs/ref_50.txt \
  --out_csv results/blt_lcm_50_full_metrics.csv \
  --comet_model Unbabel/wmt22-comet-da

# 80%
uv run lcm_scripts/run_metric_suite.py \
  --checkpoints_dir lcm_models/blt_lcm_80 \
  --checkpoint_glob "lcm_blt_best.pth" \
  --hyp_dir outputs/blt_lcm_80_hyps \
  --ref_file outputs/ref_80.txt \
  --out_csv results/blt_lcm_80_full_metrics.csv \
  --comet_model Unbabel/wmt22-comet-da
```

---

## 2. BPE + LCM

Each script trains the model and evaluates at all noise levels, saving `metrics_fraction{frac}.csv`.

### 2.1 Train & Evaluate

```bash
# 25%
uv run lcm_scripts/train_lcm_bpe.py \
  --fraction 0.25 --epochs 2 --batch_size 8 \
  --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/lcm_bpe_25 --wandb --wandb_project BLT-LCM --wandb_name lcm_bpe_25
# → runs/lcm_bpe_25/metrics_fraction0.25.csv

# 50%
uv run lcm_scripts/train_lcm_bpe.py \
  --fraction 0.50 --epochs 2 --batch_size 8 \
  --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/lcm_bpe_50 --wandb --wandb_project BLT-LCM --wandb_name lcm_bpe_50

# 80%
uv run lcm_scripts/train_lcm_bpe.py \
  --fraction 0.80 --epochs 2 --batch_size 8 \
  --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/lcm_bpe_80 --wandb --wandb_project BLT-LCM --wandb_name lcm_bpe_80
```

### 2.2 Output columns (per CSV)

| model | fraction | noise | num_predictions | BLEU | chrF++ | TER |
|-------|----------|-------|-----------------|------|--------|-----|

---

## 3. SONAR + LCM

### 3.1 Train & Evaluate

```bash
# 25%
uv run lcm_scripts/train_lcm_sonar.py \
  --fraction 0.25 --epochs 2 --batch_size 8 \
  --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/lcm_sonar_25 --wandb --wandb_project BLT-LCM --wandb_name lcm_sonar_25
# → runs/lcm_sonar_25/metrics_fraction0.25.csv

# 50%
uv run lcm_scripts/train_lcm_sonar.py \
  --fraction 0.50 --epochs 2 --batch_size 8 \
  --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/lcm_sonar_50 --wandb --wandb_project BLT-LCM --wandb_name lcm_sonar_50

# 80%
uv run lcm_scripts/train_lcm_sonar.py \
  --fraction 0.80 --epochs 2 --batch_size 8 \
  --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/lcm_sonar_80 --wandb --wandb_project BLT-LCM --wandb_name lcm_sonar_80
```

### 3.2 Evaluate existing checkpoint separately

```bash
uv run lcm_scripts/eval_lcm_sonar.py \
  --checkpoint runs/lcm_sonar_25/lcm_sonar_fraction0.25_epoch2.pth \
  --fraction 0.25 --eval_docs 100 \
  --noise_levels 0.0 0.1 0.2 \
  --out_csv runs/lcm_sonar_25/metrics_fraction0.25.csv
```

### 3.3 Output columns (per CSV)

| model | fraction | noise | BLEU | chrF++ | TER |
|-------|----------|-------|------|--------|-----|

---

## 4. BPE + Llama-8B

### 4.1 Train & Evaluate (with LoRA)

```bash
# 25%  (without QLoRA — needs ~48GB VRAM)
uv run lcm_scripts/train_bpe_llama8b.py \
  --fraction 0.25 --epochs 1 --batch_size 1 --grad_accum 16 \
  --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/bpe_llama8b_25
# → runs/bpe_llama8b_25/metrics_fraction0.25.csv

# 25%  (with QLoRA 4-bit — fits ~16GB VRAM)
uv run lcm_scripts/train_bpe_llama8b.py \
  --fraction 0.25 --epochs 1 --batch_size 1 --grad_accum 16 \
  --qlora --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/bpe_llama8b_25_qlora

# 50%  (with QLoRA)
uv run lcm_scripts/train_bpe_llama8b.py \
  --fraction 0.50 --epochs 1 --batch_size 1 --grad_accum 16 \
  --qlora --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/bpe_llama8b_50_qlora

# 80%  (with QLoRA)
uv run lcm_scripts/train_bpe_llama8b.py \
  --fraction 0.80 --epochs 1 --batch_size 1 --grad_accum 16 \
  --qlora --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/bpe_llama8b_80_qlora
```

### 4.2 Output columns (per CSV)

| model | fraction | noise | BLEU | chrF++ | TER |
|-------|----------|-------|------|--------|-----|

---

## 5. BPE + Transformer

### 5.1 Train & Evaluate

```bash
# 25%
uv run lcm_scripts/train_bpe_transformer.py \
  --fraction 0.25 --epochs 3 --batch_size 32 \
  --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/bpe_transformer_25
# → runs/bpe_transformer_25/metrics_fraction0.25.csv

# 50%
uv run lcm_scripts/train_bpe_transformer.py \
  --fraction 0.50 --epochs 3 --batch_size 32 \
  --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/bpe_transformer_50

# 80%
uv run lcm_scripts/train_bpe_transformer.py \
  --fraction 0.80 --epochs 3 --batch_size 32 \
  --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/bpe_transformer_80
```

### 5.2 Output columns (per CSV)

| model | fraction | noise | BLEU | chrF++ | TER |
|-------|----------|-------|------|--------|-----|

---

## 6. All Models at Once (Benchmark Orchestrator)

```bash
# Run BPE-Transformer + BPE-LCM + SONAR-LCM in one go (excludes BLT-LCM and Llama)
uv run lcm_scripts/benchmark_bhashasetu_models.py \
  --models bpe_transformer bpe_lcm sonar_lcm \
  --fractions 0.25 0.50 0.80 \
  --noise_levels 0.0 0.1 0.2 \
  --eval_docs 100 --epochs 1 \
  --out_dir runs/bhashasetu_benchmarks

# Include Llama-8B (requires ~48GB VRAM or use --llama_qlora)
uv run lcm_scripts/benchmark_bhashasetu_models.py \
  --models bpe_transformer bpe_lcm bpe_llama8b sonar_lcm \
  --fractions 0.25 0.50 0.80 \
  --noise_levels 0.0 0.1 0.2 \
  --epochs 1 --eval_docs 100 --eval_examples 500 \
  --llama_qlora --out_dir runs/bhashasetu_benchmarks

# Final summary
# → runs/bhashasetu_benchmarks/summary_metrics.csv
```

---

## 7. Research Paper Data Collection

After all runs complete, aggregate all CSVs into a single paper-ready table:

```bash
# Collect all metric CSVs
python -c "
import csv, glob, os

rows = []
for f in glob.glob('runs/**/metrics_fraction*.csv', recursive=True):
    with open(f) as fh:
        rows.extend(list(csv.DictReader(fh)))
for f in glob.glob('results/blt_lcm_*_metrics.csv', recursive=True):
    with open(f) as fh:
        rows.extend(list(csv.DictReader(fh)))

fieldnames = ['model', 'fraction', 'noise', 'BLEU', 'chrF++', 'TER', 'METEOR', 'COMET', 'num_predictions']
with open('paper_results/all_metrics.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader(); w.writerows(rows)
print(f'Aggregated {len(rows)} rows → paper_results/all_metrics.csv')
"

# Generate LaTeX table
python -c "
import csv, collections

with open('paper_results/all_metrics.csv') as f:
    rows = list(csv.DictReader(f))

# Group by model and fraction
groups = collections.defaultdict(list)
for r in rows:
    groups[(r['model'], r['fraction'], r['noise'])].append(r)

print('\\\\begin{table}[t]')
print('\\\\centering')
print('\\\\begin{tabular}{lcccc}')

# Output table...
"
```

---

## 8. Slurm Cluster Submission

Existing Slurm scripts in `scripts/`:

```bash
# BLT-LCM
sbatch scripts/submit_blt.sh 0.25 lcm_blt_25 11:00:00
sbatch scripts/submit_blt.sh 0.50 lcm_blt_50 21:00:00
sbatch scripts/submit_blt.sh 0.80 lcm_blt_80 1-09:00

# Pre-encode BLT embeddings
sbatch scripts/encode_blt.sh 0.25
sbatch scripts/encode_blt.sh 0.50
sbatch scripts/encode_blt.sh 0.80

# BPE-LCM
sbatch scripts/submit_bpe_lcm.sh 0.25 lcm_bpe_25
sbatch scripts/submit_bpe_lcm.sh 0.50 lcm_bpe_50
sbatch scripts/submit_bpe_lcm.sh 0.80 lcm_bpe_80

# SONAR-LCM
sbatch scripts/submit_sonar.sh 0.25 lcm_sonar_25
sbatch scripts/submit_sonar.sh 0.50 lcm_sonar_50
sbatch scripts/submit_sonar.sh 0.80 lcm_sonar_80
```
