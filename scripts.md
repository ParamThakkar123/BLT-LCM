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

## 0.5 Checkpointing & resume

Every long-running script in this repo can be killed and restarted, and will
continue from where it stopped. **Just re-run the exact same command** — resume
is on by default.

```bash
# Start a run
python lcm_scripts/train_lcm_blt.py --entropy_model patching_scratch/entropy_model_marathi.pt \
  --fraction 0.25 --epochs 5 --model_dir lcm_models/blt_lcm_25

# ... job is preempted / node dies / you hit Ctrl-C ...

# Re-run the identical command; it picks up mid-epoch where it left off
python lcm_scripts/train_lcm_blt.py --entropy_model patching_scratch/entropy_model_marathi.pt \
  --fraction 0.25 --epochs 5 --model_dir lcm_models/blt_lcm_25
```

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--resume auto` | default | Continue from this run's own checkpoint / partial output if one exists **and the configuration matches**. |
| `--resume never` | | Ignore and discard partial state; start clean. |
| `--resume PATH` | | Resume from a specific checkpoint file. |
| `--save_interval_steps N` | 200 (1000 for `train_lcm_blt.py`) | Write a resumable checkpoint every N optimizer steps. `0` disables. |
| `--save_interval_seconds S` | 0 (off) | Also checkpoint every S seconds of wall-clock. Useful against a Slurm time limit. |
| `--max_checkpoints N` | 5 | Per-epoch snapshots to retain. |
| `--ckpt_seed N` | 42 | Seed for per-epoch batch shuffling. |

Evaluation and corpus-analysis scripts take `--resume` only — they have no
optimizer state, and resume by record or by stage instead.

### What resume actually restores

Training scripts checkpoint the **model, optimizer, LR scheduler, epoch, step
counter, best-score-so-far, and all RNG states**. Batch order for epoch *N* is a
pure function of `(--ckpt_seed, N)`, so a resumed run replays the same
permutation and skips the batches it already trained on. The result is exact:
`tests/test_checkpoint_utils.py` asserts that a run killed mid-epoch and resumed
ends on **bit-identical weights** to an uninterrupted run.

Three resume granularities are in play, depending on the script:

* **Training loops** (`train_*.py`, `finetune_lcm.py`, `blt_decoder.py`,
  `run_blt_patching.py`) — mid-epoch, at `--save_interval_steps` granularity.
* **Per-record scans** (`sweep_entropy_threshold.py`,
  `morpheme_boundary_alignment.py`, `fixed_chunk_ablation.py`,
  `fertility_audit.py`, `error_analysis_100.py`, `extract_marathi.py`, the
  patching pass of `run_blt_patching.py`, the decode pass of `eval_lcm_blt.py`)
  — rows stream to JSONL as they are produced; a rerun replays the file and
  computes only what is missing.
* **Per-stage memoization** (per-noise-level evaluation, `run_metric_suite.py`,
  `eval_runner.py`) — each finished stage's metrics are cached in a JSON
  sidecar, so an interrupted evaluation does not re-decode levels it finished.

Expensive frozen-encoder passes are cached too. Pass `--embed_cache PATH` (on
`train_lcm_blt.py`, `train_lcm_blt_mt.py`, `train_lcm_sonar.py`,
`train_base_lcm.py`, `finetune_lcm.py`, `eval_lcm_blt.py`, `eval_lcm_sonar.py`)
and a resumed run reloads the encodings instead of recomputing them.

### Configuration changes are refused, not silently mixed

Resume state is keyed to a fingerprint of the run's settings (learning rate,
fraction, epochs, architecture, …; output paths and logging toggles are
excluded, so redirecting `--out_dir` or turning W&B on/off is fine). Resuming
into a run whose hyperparameters changed **errors out** rather than splicing two
configurations into one set of results:

```
RuntimeError: Checkpoint lcm_models/lcm_blt_last.pth was written by a different
configuration (fingerprint a1b2… != c3d4…). Re-run with the original flags,
point --resume at a different checkpoint, or pass --resume never to start a
fresh run.
```

### Files written

```
<model_dir>/<prefix>_last.pth      # rolling checkpoint — the resume target
<model_dir>/<prefix>_best.pth      # best-scoring checkpoint so far
<model_dir>/<prefix>_epoch{N}.pth  # per-epoch snapshots (pruned to --max_checkpoints)
<output>.meta.json                 # fingerprint sidecar for JSONL scans
<output>.state.json                # memoized stage results
```

Checkpoints are written to a `.tmp` file and then atomically renamed, so a
process killed mid-write leaves the previous checkpoint intact rather than a
truncated one. Checkpoint files now carry optimizer/RNG state alongside the
weights; loaders accept both that payload and the bare `state_dict` files older
runs produced, so existing `lcm_models/*.pth` artifacts keep working.

---

## 0.55 Which device a run is on

Every script prints the device it resolved, as its first line of real output,
before it loads a model or touches the dataset:

```
[device] cuda:0 | NVIDIA A100-SXM4-40GB | 39.4 GiB | capability 8.0 | torch 2.5.1+cu121 | CUDA 12.1
```

```
[device] cpu | 64 cores | 32 threads | torch 2.5.1+cu121 | CUDA 12.1
[device] WARNING: running on CPU even though a CUDA device is available -- pass --device cuda to use it.
```

`lcm_scripts/device_utils.py` provides this (`report_device`), and each entry
point calls it right where it resolves its device. Two cases are called out
explicitly, because both otherwise look exactly like a healthy run until the
run is far slower than expected:

* **CPU while a GPU is present** — usually a stale `--device cpu`.
* **`--device cuda` where `torch.cuda.is_available()` is False** — the driver,
  the torch build, or the Slurm `--gres` allocation. The request is reported
  back unchanged rather than silently downgraded to CPU: an unnoticed CPU
  fallback on a cluster is worse than a clear failure.

`CUDA_VISIBLE_DEVICES` is echoed whenever it is set.

The Slurm scripts additionally report what the *node* offered, via
`scripts/report_gpu.sh`, which every `scripts/*.sh` job sources after its start
banner:

```
GPU allocation:
  0, NVIDIA A100-SXM4-40GB, 40960 MiB, 535.104.05
  CUDA_VISIBLE_DEVICES=0
  SLURM_JOB_GPUS=0
```

Comparing the two lines separates "Slurm gave the job no GPU" from "torch could
not use the GPU it was given".

---

## 0.6 Paper-fidelity notes

The implementations follow **BLT** (Pagnoni et al., 2024) and **LCM** (Barrault
et al., 2024). `tests/test_paper_fidelity.py` pins the specific claims below;
run it after any change to the model code.

### BLT

* **Patch boundaries.** `entropy_patch_sentence` starts a patch at byte *k* when
  `H(x_k) > θ_g`, where `H(x_k)` is the entropy of the distribution over byte
  *k* given everything before it (Eq. 1). Because
  `compute_entropies_for_tokens` returns *next-byte* entropies
  (`out[t] = H(x_{t+1} | x_{<=t})`), that is index `k-1`. Positions 0 and 1 are
  always patch starts.
* **Both segmentation rules** from §2.3 are available:
  `--patching_mode global` (default, `H(x_k) > θ_g`) and
  `--patching_mode monotonic` with `--threshold_add` (`H(x_k) − H(x_{k−1}) > θ_r`).
  `--reset_context_on_newline` recomputes entropies per line (§4.4), which stops
  repetitive text from producing runaway patch sizes.
* **Sliding-window attention** for the entropy model via `--attn_window`
  (§4.2 uses 512). Default is unbounded causal attention. Entropy-model defaults
  here (dim 256, 4 layers) are a scaled-down variant of the paper's 100M/14-layer
  model; per Fig. 8, quality rises with both size and context.
* **Hash n-gram embeddings** (§3.2.1, Eqs. 2–4, Appendix C) are implemented in
  `blt_local_encoder.HashNGramEmbedder`: rolling polynomial hash into per-n
  tables, summed onto the byte embedding and normalised by
  `len(ngram_sizes) + 1`.

  **These tables dominate memory.** Cost is
  `len(ngram_sizes) × hash_vocab_size × encoder_dim × 4 bytes`, and AdamW
  quadruples it (gradient + two moments). At `encoder_dim=256`:

  | config | params | committed under AdamW |
  | --- | --- | --- |
  | paper: n=3..8 × 500k (§4.8) | 2.86 GiB | **11.44 GiB** — OOMs a 16 GiB GPU |
  | **default: n=3,4,5 × 100k** | 0.29 GiB | **1.14 GiB** |
  | n=3,4,5 × 200k | 0.57 GiB | 2.29 GiB |
  | n=3,4,5 × 400k | 1.14 GiB | 4.58 GiB |

  The default follows BLT Table 8, which shows per-n vocabulary matters more
  than covering every n and that smaller n are the more impactful ones
  (`3,4,5 @ 100k` scores 0.837 on the train distribution vs 0.826 for the full
  `3..8 @ 400k`). Tune with `--hash_vocab_size` / `--ngram_sizes`, or drop them
  entirely with `--no_hash_ngrams` for the Table 8 ablation. `BLTLoader` prints
  the footprint at construction so a bad configuration is visible before the
  first `optimizer.step()` rather than after it.
* **The local encoder is a separate trainable network** (`BLTLocalEncoder`), not
  the entropy model. The entropy model is frozen and supplies boundaries only.
* **The patch sequence reaches a latent transformer** (`BLTLatentTransformer`,
  §3.1) before being pooled into a sentence concept, so the per-patch compute
  step the paper is about actually runs.

### LCM

* **Base-LCM is decoder-only** (§2.3.1): the concept sequence passes through
  causal self-attention. Use `model.forward_all(src)` to get a prediction at
  every position in one pass; `model(src)` returns the next concept only.
* **PreNet/PostNet implement Eqs. (1)–(4)**: a median/IQR `RobustScaler` fitted
  **once** on sampled concepts and frozen in buffers, with `denormalize` applied
  *after* the PostNet projection so outputs are in raw encoder coordinates. Call
  `model.fit_normalizer(sample)` before training — the training scripts do this
  automatically.
* **End-of-text** is a buffer holding `encode("End of text.")`, installed via
  `model.set_eot_embedding(...)`, and training documents are suffixed with it.
  Generation stops on `cos(x̂_n, eot) > s_eot` **or** `cos(x̂_n, x̂_{n−1}) > s_prev`
  (both default 0.9). Generation raises if the EOT concept was never set.
* **Diffusion and quantized variants** are in `lcm_scripts/diffusion_lcm.py`
  (`TwoTowerDiffusionLCM`, `OneTowerDiffusionLCM`) and
  `lcm_scripts/quant_lcm.py` (`QuantLCM` + `ResidualVectorQuantizer`). They
  provide cosine/quadratic/sigmoid noise schedules with zero-terminal-SNR
  rescaling, classifier-free guidance, guidance rescaling and epsilon-scaling.
  The paper finds these outperform the MSE Base-LCM, which regresses to the
  *mean* of plausible continuations.

### Selecting an LCM variant

`train_lcm_blt.py --lcm_variant` picks the model. Checkpoints are prefixed by
variant, so several can share one `--model_dir`.

```bash
# Base-LCM (MSE) — the paper's baseline, and the weakest of the four
python lcm_scripts/train_lcm_blt.py --entropy_model patching_scratch/entropy_model_marathi.pt \
  --lcm_variant base --fraction 0.25 --epochs 5

# Two-Tower diffusion (the variant the paper scales to 7B)
python lcm_scripts/train_lcm_blt.py --entropy_model patching_scratch/entropy_model_marathi.pt \
  --lcm_variant two_tower --noise_schedule cosine --diffusion_steps 100 \
  --fraction 0.25 --epochs 5

# One-Tower diffusion
python lcm_scripts/train_lcm_blt.py ... --lcm_variant one_tower --noise_schedule sigmoid

# Quant-LCM-d (discrete units) / -c (continuous residual)
python lcm_scripts/train_lcm_blt.py ... --lcm_variant quant --quant_target discrete \
  --n_codebooks 64 --units_per_codebook 8192
```

| Flag | Applies to | Default |
| --- | --- | --- |
| `--lcm_variant` | all | `base` |
| `--model_dim` / `--n_layers` / `--n_heads` | all | 2048 / 12 / 16 |
| `--diffusion_steps` | diffusion | 100 |
| `--noise_schedule` | diffusion | `cosine` |
| `--n_codebooks` / `--units_per_codebook` | quant | 64 / 8192 |
| `--quant_target` | quant | `discrete` |
| `--quant_fit_samples` | quant | 200000 |

The robust scaler is fitted for every variant before training; for `quant` the
RVQ codebooks are then fitted on the **normalized** concepts, so quantization
is not dominated by whichever dimensions have the largest raw scale.

> **Checkpoint compatibility.** BaseLCM checkpoints produced before it became
> decoder-only have an incompatible parameter layout and must be retrained;
> `finetune_lcm.py` reports this explicitly rather than loading garbage.
> Likewise, patching statistics computed before the boundary fix are shifted by
> one byte — re-run the patching and morpheme-alignment analyses.

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
  --out_dir runs/bpe_llama8b_25 --wandb --wandb_project BLT-LCM \
  --wandb_name bpe_llama8b_25
# → runs/bpe_llama8b_25/metrics_fraction0.25.csv

# 25%  (with QLoRA 4-bit — fits ~16GB VRAM)
uv run lcm_scripts/train_bpe_llama8b.py \
  --fraction 0.25 --epochs 1 --batch_size 1 --grad_accum 16 \
  --qlora --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/bpe_llama8b_25_qlora --wandb --wandb_project BLT-LCM \
  --wandb_name bpe_llama8b_25_qlora

# 50%  (with QLoRA)
uv run lcm_scripts/train_bpe_llama8b.py \
  --fraction 0.50 --epochs 1 --batch_size 1 --grad_accum 16 \
  --qlora --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/bpe_llama8b_50_qlora --wandb --wandb_project BLT-LCM \
  --wandb_name bpe_llama8b_50_qlora

# 80%  (with QLoRA)
uv run lcm_scripts/train_bpe_llama8b.py \
  --fraction 0.80 --epochs 1 --batch_size 1 --grad_accum 16 \
  --qlora --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/bpe_llama8b_80_qlora --wandb --wandb_project BLT-LCM \
  --wandb_name bpe_llama8b_80_qlora
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
  --out_dir runs/bpe_transformer_25 --wandb --wandb_project BLT-LCM \
  --wandb_name bpe_transformer_25
# → runs/bpe_transformer_25/metrics_fraction0.25.csv

# 50%
uv run lcm_scripts/train_bpe_transformer.py \
  --fraction 0.50 --epochs 3 --batch_size 32 \
  --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/bpe_transformer_50 --wandb --wandb_project BLT-LCM \
  --wandb_name bpe_transformer_50

# 80%
uv run lcm_scripts/train_bpe_transformer.py \
  --fraction 0.80 --epochs 3 --batch_size 32 \
  --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/bpe_transformer_80 --wandb --wandb_project BLT-LCM \
  --wandb_name bpe_transformer_80
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
  --out_dir runs/bhashasetu_benchmarks --wandb --wandb_project BLT-LCM

# Include Llama-8B (requires ~48GB VRAM or use --llama_qlora)
uv run lcm_scripts/benchmark_bhashasetu_models.py \
  --models bpe_transformer bpe_lcm bpe_llama8b sonar_lcm \
  --fractions 0.25 0.50 0.80 \
  --noise_levels 0.0 0.1 0.2 \
  --epochs 1 --eval_docs 100 --eval_examples 500 \
  --llama_qlora --out_dir runs/bhashasetu_benchmarks --wandb \
  --wandb_project BLT-LCM

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

Set `CLUSTER_PARTITION` in `.env` (copy from `.env.example`) to select the GPU queue.
Then use `scripts/sbatch.sh` as a drop-in replacement for `sbatch` — it reads the partition
from `.env` automatically, so the same commands work on Galvani, Ferranti, and any other cluster.

```bash
# Pre-encode BLT embeddings (run once per fraction before BLT-LCM training)
scripts/sbatch.sh scripts/encode_blt.sh 0.25
scripts/sbatch.sh scripts/encode_blt.sh 0.50
scripts/sbatch.sh scripts/encode_blt.sh 0.80

# BLT-LCM — train
scripts/sbatch.sh scripts/submit_blt.sh 0.25 lcm_blt_25 11:00:00
scripts/sbatch.sh scripts/submit_blt.sh 0.50 lcm_blt_50 21:00:00
scripts/sbatch.sh scripts/submit_blt.sh 0.80 lcm_blt_80 1-09:00

# BLT-LCM — evaluate
scripts/sbatch.sh scripts/eval_blt.sh 0.25 lcm_blt_25
scripts/sbatch.sh scripts/eval_blt.sh 0.50 lcm_blt_50
scripts/sbatch.sh scripts/eval_blt.sh 0.80 lcm_blt_80

# BLT-LCM — full metric suite (requires hypothesis files from eval step)
scripts/sbatch.sh scripts/metric_suite.sh 0.25 lcm_blt_25
scripts/sbatch.sh scripts/metric_suite.sh 0.50 lcm_blt_50
scripts/sbatch.sh scripts/metric_suite.sh 0.80 lcm_blt_80

# BPE-LCM — train & evaluate
scripts/sbatch.sh scripts/submit_bpe_lcm.sh 0.25 lcm_bpe_25
scripts/sbatch.sh scripts/submit_bpe_lcm.sh 0.50 lcm_bpe_50
scripts/sbatch.sh scripts/submit_bpe_lcm.sh 0.80 lcm_bpe_80

# SONAR-LCM — train & evaluate
scripts/sbatch.sh scripts/submit_sonar.sh 0.25 lcm_sonar_25 11:00:00
scripts/sbatch.sh scripts/submit_sonar.sh 0.50 lcm_sonar_50 21:00:00
scripts/sbatch.sh scripts/submit_sonar.sh 0.80 lcm_sonar_80 1-09:00

# SONAR-LCM — evaluate existing checkpoint separately
scripts/sbatch.sh scripts/eval_sonar.sh 0.25 lcm_sonar_25 2

# BPE + Llama-8B — QLoRA (default, ~16GB VRAM)
scripts/sbatch.sh scripts/submit_llama8b.sh 0.25 bpe_llama8b_25_qlora qlora
scripts/sbatch.sh scripts/submit_llama8b.sh 0.50 bpe_llama8b_50_qlora qlora
scripts/sbatch.sh scripts/submit_llama8b.sh 0.80 bpe_llama8b_80_qlora qlora

# BPE + Llama-8B — full precision (~48GB VRAM, 25% only)
scripts/sbatch.sh scripts/submit_llama8b.sh 0.25 bpe_llama8b_25_full full

# BPE + Transformer — train & evaluate
scripts/sbatch.sh scripts/submit_transformer.sh 0.25 bpe_transformer_25
scripts/sbatch.sh scripts/submit_transformer.sh 0.50 bpe_transformer_50
scripts/sbatch.sh scripts/submit_transformer.sh 0.80 bpe_transformer_80
```
