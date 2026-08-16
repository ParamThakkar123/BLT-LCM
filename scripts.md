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

**Every script in this repository reports its device**, as its first line of
real output, before it loads a model or touches the dataset.

Scripts that put tensors on a device print what they resolved:

```
[device] cuda:0 | NVIDIA A100-SXM4-40GB | 39.4 GiB | capability 8.0 | torch 2.5.1+cu121 | CUDA 12.1
```

```
[device] cpu | 64 cores | 32 threads | torch 2.5.1+cu121 | CUDA 12.1
[device] WARNING: running on CPU even though a CUDA device is available -- pass --device cuda to use it.
```

Scripts that do no tensor work — plotting, CSV aggregation, dataset streaming,
the pure-python corpus scans — say so, with the reason, so the line reads as a
fact about the script rather than a missing GPU:

```
[device] cpu | 64 cores | CPU-only: matplotlib rendering from existing result files
[device] cpu | 64 cores | CPU-only: bigram entropy model and Indic NLP morphology, both pure python
```

These never import torch, so the report costs nothing in a script that only
draws a figure.

`lcm_scripts/device_utils.py` provides both (`report_device` and
`report_cpu_only`), and each entry point calls one of them where it resolves
its device. Coverage:

| | |
| --- | --- |
| Training / evaluation / patching | `report_device(args.device)` at the top of `main()` |
| Metric suites (`run_metric_suite.py`, `eval_runner.py`, `eval_metrics.py`) | `report_device(label="metrics")` — BLEU/chrF++/TER are CPU string metrics, COMET runs a model |
| `fertility_audit.py` | `report_device(label="Stanza POS", logger=logger)` — Stanza's tagger is the neural part |
| Plotting / aggregation / streaming | `report_cpu_only(reason)` |
| `BLTLoader`, `SonarLoader`, `SonarLite` | report at construction, confirming they were handed the device the script announced |
| `tests/` | a session fixture in `tests/conftest.py` prints once per run |

The remaining files under `lcm_scripts/` (`base_lcm.py`, `diffusion_lcm.py`,
`quant_lcm.py`, `blt_local_encoder.py`, `checkpoint_utils.py`, …) are
importable modules with no entry point and no device of their own — they live
on whatever device the caller moves them to, and that caller has already
reported it.

Two cases are called out explicitly, because both otherwise look exactly like a
healthy run until the run is far slower than expected:

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

## 0.57 Figures: training curves and evaluation plots

**Every training script draws its own training curve, and every evaluation
script draws its own results figures**, at 300 DPI, next to the run's other
outputs. Nothing extra needs to be run and no flags are required.

```
--plot_dir DIR        where figures go (default: the run's own output dir --
                      --model_dir or --out_dir, whichever the script uses)
--plot_format FMT     png (default) | jpg | both
--no_plots            skip figure generation entirely
```

These three flags are excluded from the run fingerprint, so switching plotting
on or changing the format mid-run does **not** invalidate an in-progress run's
checkpoints or embedding caches.

### What a training run writes

```
<plot_dir>/<run>_training_curve.png      # loss vs epoch (+ validation, best-epoch callout)
<plot_dir>/<run>_training_dashboard.png  # step loss, epoch loss, LR schedule,
                                         #   gradient norm, epoch wall-clock, eval metrics
<plot_dir>/<run>_history.json            # the raw scalars behind both figures
```

The dashboard only draws panels it has data for, so a run without a learning-rate
schedule or without per-epoch evaluation simply gets a smaller sheet.

`<run>_history.json` carries the run fingerprint and is **resume-aware**: a job
that is preempted after epoch 2 and restarted finishes with a curve covering
epochs 1–4, not one that begins at the restart. A history written under a
different configuration is not spliced in — it is discarded, the same way a
mismatched checkpoint is refused (§0.5).

Per-step scalars are *sampled* (every 50 steps, or `--log_every` where the
script has one) rather than recorded per optimizer step. The gradient-norm
reduction walks every parameter, so it stays off the hot path, and the sidecar
stays small on a multi-epoch run.

| Script | Run name | Loss plotted |
| --- | --- | --- |
| `train_lcm_blt.py` | `lcm_blt_<variant>` | MSE / diffusion / RVQ loss per variant |
| `train_lcm_blt_mt.py` | `lcm_blt_mt_s<seed>` | concept MSE (one curve per seed) |
| `train_lcm_sonar.py` | `lcm_sonar_fraction<F>` | masked MSE |
| `train_lcm_bpe.py` | `lcm_bpe_fraction<F>` | masked MSE |
| `train_bpe_transformer.py` | `bpe_transformer_fraction<F>` | cross-entropy (+ perplexity) |
| `train_bpe_llama8b.py` | `bpe_llama8b_fraction<F>` | cross-entropy, transcribed from the HF Trainer log |
| `train_base_lcm.py` | `base_lcm` | MSE, **with a held-out validation curve** |
| `train_sonar.py` | `sonar_lite` | reconstruction + λ·MSE |
| `finetune_lcm.py` | `lcm_finetuned` | MSE |
| `blt_decoder.py` | `blt_decoder` | byte cross-entropy (+ byte perplexity) |
| `run_blt_patching.py` | `entropy_model_marathi` | byte cross-entropy (+ bits/byte) |

### What an evaluation run writes

| Script | Figures |
| --- | --- |
| `eval_lcm_blt.py` | metric bars, metric table, hypothesis-vs-reference length delta |
| `eval_lcm_sonar.py` | metric vs noise level, metric table |
| `evaluate_blt_lcm.py` | MSE by sentence position in the document, per-batch MSE distribution |
| `eval_runner.py` | metric degradation vs noise, **one line per seed**, plus a seed-averaged table |
| `run_metric_suite.py` | metrics across checkpoints (i.e. across training), COMET on its own axis, best-checkpoint bars, summary table |
| `train_lcm_blt_mt.py`, `train_lcm_*`, `train_bpe_*` | noise-robustness curves and a metric table after their eval stage |
| `compare_bhashasetu_metrics.py` | grouped bars and degradation lines, per metric, across BLT / BPE / SONAR |
| `benchmark_bhashasetu_models.py` | cross-model robustness per data fraction, score-by-fraction bars, full summary table |
| `run_blt_patching.py` | patches-per-sentence and patch-length distributions at the chosen τ |

`benchmark_bhashasetu_models.py` forwards `--plot_format` / `--no_plots` to every
sub-job, so a whole benchmark sweep is drawn in one format; `--plot_dir` is not
forwarded, so each run keeps its own figures beside its own outputs.

Figure generation is best-effort by design: a headless node without a writable
font cache, a missing backend, or a corrupt sidecar prints a warning and the run
continues. A finished training run is never lost to a plotting failure.

The paper figures proper are still built separately by `generate_paper_figures.py`
and the per-analysis plotters under `error_analysis/`, `fertility_audit/`,
`morpheme_alignment/`, `fixed_chunk_ablation/` and `tokenization_statistics/`.
`tests/test_plot_utils.py` covers the shared helpers in `lcm_scripts/plot_utils.py`.

---

## 0.58 Variable epochs: train while the loss is still improving

A fixed `--epochs` is a guess. Every training script also accepts a stopping
rule that keeps going while the monitored loss improves:

```
--train_until_plateau   run until the loss stops improving, instead of a fixed --epochs
--patience N            consecutive non-improving epochs tolerated (default 3)
--min_delta X           improvement smaller than X counts as no improvement (default 0)
--min_epochs N          never stop on plateau before N epochs (default 1)
--max_epochs N          hard cap, so an oscillating loss still terminates (default 200)
```

```bash
# Stop once three epochs in a row fail to improve the loss by more than 1e-4,
# but never before 5 epochs and never past 60.
uv run lcm_scripts/train_lcm_blt.py --entropy_model ... \
  --train_until_plateau --patience 3 --min_delta 1e-4 --min_epochs 5 --max_epochs 60
```

Without the flag, nothing changes: `--epochs N` runs exactly N epochs, as every
existing command line and Slurm script expects. With it, `--epochs` becomes a
**floor** (at least that many) and `--max_epochs` becomes the ceiling.

What is monitored:

| Script | Monitored quantity |
| --- | --- |
| `train_base_lcm.py` | **held-out validation MSE** (it has a real 80/20 split) |
| every other script | that script's training loss |
| `train_bpe_llama8b.py` | the HF Trainer's logged training loss, via a `TrainerCallback` that sets `should_training_stop` |

Each epoch prints its position (`Epoch 7/<=60 (until plateau)`), non-improving
epochs print the patience countdown, and the run ends with a line saying why:

```
  [plateau] epoch 9: MSE loss 0.3616 did not beat 0.3554 by > 0.0001 (2/3 before stopping)
[epochs] ran 11 epoch(s); best MSE loss 0.355 at epoch 8; stopped because plateau: ...
```

**Resume-safe.** The patience counter is rebuilt from the run's
`*_history.json`, so a job preempted on its second-to-last tolerated epoch comes
back with the same "epochs since improvement" the original process had — not a
fresh window that would buy several more pointless epochs.

The stopping-rule flags are excluded from the run fingerprint on purpose: they
govern how *long* a run goes, not what any step computes, so adding
`--train_until_plateau` to an existing command **continues** that run from its
checkpoint instead of discarding the epochs already paid for.

---

## 0.59 Publishing results to git (and GitHub)

Every training and evaluation script ends by collecting what it produced into
`results/runs/<run_name>/` and committing it:

```
results/runs/<run_name>/
  README.md                 # metrics table, run info, embedded figures
  run.json                  # ALL hyperparameters, metrics, stop reason, environment
  <run>_history.json        # every recorded loss, LR and gradient norm
  *.png / *.jpg             # the run's figures
  metrics_*.csv             # the run's metric CSV, where it has one
```

```
--push_results        also push the commit to the remote (or BLT_LCM_PUSH_RESULTS=1)
--results_dir DIR     where collected results go (default results/runs)
--results_remote R    remote to push to (default origin)
--results_branch B    branch to push to (default: the current branch)
--results_max_mb N    skip any single file larger than this (default 25)
--no_results          do not collect or commit anything
```

`run.json` is the reproducibility record: the complete argparse namespace, the
final and best losses, the stop reason, the git commit the code was at (flagged
if the working tree was dirty), torch/CUDA versions, GPU name, and the Slurm job
id when there is one.

Deliberate limits, because this runs unattended at the end of every job:

* **Only an explicit file list is staged** — `git add <paths>`, never `-A`. A run
  cannot commit your working-tree edits, and cannot commit a checkpoint, dataset
  shard or embedding cache. Only `.png/.jpg/.pdf/.svg/.csv/.json/.md/.txt` are
  eligible; `.pth`/`.pt`/`.jsonl` are excluded by design.
* **Size-capped** — anything over `--results_max_mb` is skipped with a warning
  rather than written into git history, where it is permanent.
* **Never fatal** — no remote, no upstream, a rejected push, a detached HEAD:
  all print a warning and return. A finished training run is never lost to a
  failed push.
* **Never interactive** — every git call runs with `GIT_TERMINAL_PROMPT=0` and a
  timeout (`--results_timeout`, default 120 s). A missing credential fails in
  seconds instead of hanging a GPU job until its Slurm wall-clock limit, which
  is the single most expensive way for unattended publishing to break.
* **Push is opt-in** — committing locally is cheap and reversible; pushing is
  neither. Set `BLT_LCM_PUSH_RESULTS=1` in `.env` to make it automatic for every
  job without editing any command line.

> **`.gitignore` note.** `runs/` (which matches at any depth) and `*.json` would
> otherwise hide every published result. `.gitignore` ends with
> `!results/runs/` + `!results/runs/**` to re-include them — the directory
> negation has to come first, because git does not descend into an excluded
> directory looking for negations. Removing those two lines silently disables
> all publishing.

### On a GPU cluster

Slurm/PBS/LSF is auto-detected (via `SLURM_JOB_ID` and friends) and switches the
publisher into **isolated commit** mode:

```
--results_commit_mode auto|isolated|worktree
```

Array jobs all `cd $REPO_DIR` into **one clone**. In `worktree` mode they would
contend for `.git/index.lock`, and a job that moved `HEAD` would move it
underneath every other job still running. In `isolated` mode the commit is built
through a private index with plumbing (`read-tree` → `add` → `write-tree` →
`commit-tree`) and pushed straight to the remote, so the only shared state
touched is the remote ref:

* the shared checkout's index, `HEAD` and working tree are never modified;
* pushes that lose a race are retried (`--results_retries`, default 5) with
  randomized backoff, so an array finishing together does not retry in lockstep;
* a result identical to what is already on the remote is not re-pushed.

Verified with 8 concurrent publishes into one checkout: all 8 landed, `HEAD`
unchanged, no `index.lock` left behind.

Because the local `HEAD` does not move, the results are **on the remote, not in
your local checkout** — `git pull` to see them.

### Credentials

A compute node has no credential helper, no ssh agent and no terminal to prompt
on. Put a token in `.env` (gitignored — line 1 of `.gitignore`), which every
`scripts/*.sh` job script sources:

```bash
BLT_LCM_PUSH_RESULTS=1
GITHUB_USERNAME=your-github-username
GITHUB_TOKEN=ghp_...      # classic PAT with `repo` scope, or fine-grained
                          # with "Contents: read and write" on this repo
```

**Never put the token in a tracked file** (`auto_setup.sh`, a submit script,
`.env.example`). The first results push would publish it to GitHub, and GitHub
revokes tokens it detects in pushes — breaking the jobs it was added for.

`results_sync.py` reads `BLT_LCM_GIT_TOKEN`, `GIT_TOKEN`, `GITHUB_TOKEN` or
`GH_TOKEN` (in that order) and injects it into an `https://` remote for the
duration of the push. ssh remotes use the agent/key instead and need no token.
The assembled URL is redacted out of any error message.

`scripts/results_env.sh` is sourced by `auto_setup.sh` and every `scripts/*.sh`
job. It loads `.env`, reports how the job will publish, and **verifies the
credentials with `git ls-remote` before the job starts**, so a bad token is a
message in the first second rather than a lost result eleven hours later:

```
Results publishing:
  push: ON -> origin/main
  auth: token from the environment (user ParamThakkar123)
  credentials: OK (verified before the job starts)
  mode: isolated commit (shared checkout safe; local HEAD is not moved)
```

`benchmark_bhashasetu_models.py` publishes a `bhashasetu_benchmark` record on top
of the per-model ones, holding the cross-model comparison figures and the
clean-input score of every (model, fraction) cell.

`tests/test_results_sync.py` and `tests/test_train_control.py` cover both.

---

## 0.7 Publication-grade evaluation

Five additions aimed squarely at what a reviewer checks first.

### 0.7.1 FLORES-200 — comparable numbers

```bash
uv run lcm_scripts/eval_flores.py \
  --lcm_checkpoint lcm_models/lcm_blt_mt_s42_best.pth \
                   lcm_models/lcm_blt_mt_s43_best.pth \
                   lcm_models/lcm_blt_mt_s44_best.pth \
  --entropy_model patching_scratch/entropy_model_marathi.pt \
  --pooler lcm_models/blt_pooler.pth --decoder lcm_models/blt_decoder.pth \
  --flores_tgt mar_Deva --comet_model Unbabel/wmt22-comet-da \
  --compare nllb-600m=outputs/flores_baselines/nllb-600m_eng_Latn-mar_Deva_noise0.0.hyp.txt
```

`lcm_scripts/flores_utils.py` loads FLORES-200 (`dev` = 997 sentences for
tuning, `devtest` = 1012 for reporting), the benchmark IndicTrans2, NLLB and the
Indic MT literature report on. Every BhashaSetu number in this repo is on an
ad-hoc split nobody else uses, so it cannot be situated against anything;
FLORES fixes that. Friendly aliases work everywhere (`mr`, `marathi`,
`mar_Deva`). 24 Indic languages are available.

FLORES has no train split by design — it is evaluation data only, so there is
nothing to leak. Passing `--flores_max_examples` prints a warning, because a
truncated devtest is **not** comparable to published numbers.

### 0.7.2 Published baselines

```bash
uv run lcm_scripts/eval_public_baselines.py \
  --systems nllb-600m indictrans2 --flores_tgt mar_Deva \
  --noise_levels 0.0 0.1 0.2 --out_dir outputs/flores_baselines
```

Runs NLLB-200 and IndicTrans2 on exactly the segments `eval_flores.py` uses and
writes their hypotheses, so the comparison is on identical data and can be
paired-tested. IndicTrans2 additionally needs `pip install IndicTransToolkit`
for its preprocessor — without it the model is fed unnormalized text and its
scores are not the published ones, so the script refuses rather than reporting
a misleadingly low number.

### 0.7.3 Statistical significance

`lcm_scripts/significance.py` handles the two independent sources of noise:

* **Test-set noise** — `paired_test()` wraps sacrebleu's paired bootstrap
  resampling (Koehn 2004, `--significance_test bs`) or approximate
  randomization (`ar`). The same segments are resampled for both systems, so
  segment difficulty cancels.
* **Training noise** — `seed_summary()` aggregates the per-seed metric CSVs into
  mean ± std with a **t-based** confidence interval. With 3 seeds, t(2)=4.303
  rather than 1.96; a normal interval would be 2.2× too narrow.

> **A trap worth knowing about.** sacrebleu returns its *floor* p-value
> (`1/(n+1)`, e.g. 0.001 at 1000 resamples) when the observed delta is zero,
> because no resample produces a larger delta than zero. Taken at face value
> that reads as "p < 0.05, significant" for two **byte-identical** systems.
> `SystemComparison.significant` checks the delta before the p-value, and
> identical outputs are flagged in the results with a note.
> `tests/test_significance.py` pins this, along with the property that matters:
> a real gap comes out significant and two equally-good systems do not.

### 0.7.4 Ablations

```bash
uv run lcm_scripts/run_ablations.py --ablation variant decode compute \
  --entropy_model patching_scratch/entropy_model_marathi.pt --fraction 0.25
```

| Ablation | Answers |
| --- | --- |
| `variant` | all four LCM variants (base / one-tower / two-tower / quant). The LCM paper reports diffusion beating the MSE baseline, so reporting only `base` invites "did you try the variant your own citation prefers?" |
| `decode` | generative decoding vs the nearest-neighbour retrieval baseline. Retrieval can only emit sentences already in the training corpus, so it flatters corpus metrics without generating anything; reporting both separates "good concept space" from "good decoder" |
| `compute` | BLT-LCM vs a **parameter-matched** BPE Transformer. The match is searched over a size grid using an analytic parameter count (verified against real models in `tests/test_run_ablations.py`), and warns loudly if the closest configuration is more than 5% off rather than quietly calling it matched |

At the paper's configuration (embed 1024 / model 2048 / 12 layers = 610.6M
parameters) the matched baseline is 612.5M — 0.3% off.

### 0.7.5 Cross-language transfer and the entropy-model mismatch

```bash
uv run lcm_scripts/eval_multilingual.py \
  --entropy_model patching_scratch/entropy_model_marathi.pt
```

The entropy model was trained on Marathi, so every other language it patches —
**including the English source side of the translation task** — is out of
distribution. This measures the cost, on FLORES's n-way parallel data, so a
difference between languages is a property of the model rather than of the test
set. Measured on devtest:

| Language | Script | Bits/byte | vs Marathi | Bytes/patch |
| --- | --- | --- | --- | --- |
| Marathi (trained on) | Deva | 0.244 | 1.00× | 23.9 |
| Hindi | Deva | 0.208 | 0.85× | 36.6 |
| Bengali | Beng | 0.220 | 0.90× | 68.4 |
| Tamil | Taml | 0.241 | 0.99× | 71.8 |
| **English** | **Latn** | **0.733** | **3.00×** | **13.0** |

Two things worth stating in the paper rather than leaving for a reviewer:

1. **English costs 3× the bits/byte of Marathi**, and is patched at half the
   granularity. Every reported En→Mr number is computed through that mismatch.
2. **Low entropy is not good patching.** The non-Devanagari Indic scripts score
   *lower* bits/byte than Marathi yet get 3–4× larger patches — the model has
   learned generic UTF-8 continuation structure for byte ranges it never saw in
   training, which is predictable without being linguistically meaningful. The
   compression ratio, not the entropy, is what exposes this.

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
