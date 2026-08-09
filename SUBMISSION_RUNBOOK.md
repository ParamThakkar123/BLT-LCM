# ICLR 2027 Submission Runbook

Everything that must run (and be written) to get **BLT-LCM** submitted to the
Fifteenth International Conference on Learning Representations (ICLR 2027).

| | |
|---|---|
| **Abstract deadline** | **Sep 11, 2026 AOE** — hard gate, see §0d |
| **Full paper deadline** | **Sep 16, 2026 AOE** (supplementary material due same time) |
| Track | Main conference — ≤ 9 pages main text (refs + appendix excluded) |
| Template | ICLR 2027 LaTeX — <https://github.com/ICLR/Master-Template/raw/master/iclr2027.zip> |
| Anonymity | Double-blind. Identity revealed in main text *or* supplementary = **desk reject** |
| Portal | <https://openreview.net/group?id=ICLR.cc/2027/Conference> |
| Reviews released | Oct 29, 2026 · author–reviewer discussion Oct 29 – Nov 11, 2026 |
| Final decisions | Dec 9, 2026 (main text may grow to 10 pages from the discussion phase on) |

Sources: [Call for Papers](https://iclr.cc/Conferences/2027/CallForPapers) ·
[Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)

---

## 0. Read this before running anything

Four things gate the whole submission. (a)–(c) are code/results issues, already
fixed in code but they change what you run and what you throw away. (d) is an
ICLR-specific process gate with an earlier deadline than the paper itself.

**(a) Delete the old root CSVs.** `mt_eval_results_25_percent.csv`,
`mt_eval_results_50_percent.csv`, `mt_eval_results_80_percent.csv` were produced
by `lcm_scripts/eval_runner.py`, which **never runs the model** — it re-scores a
fixed hypothesis file — and which **corrupts the references rather than the
source**. Their "3 seeds" are the same number printed three times. Do not put any
of those numbers in the paper.

**(b) There are no checkpoints.** `lcm_models/`, `embeddings/` and `outputs/` do
not exist. Everything below trains from scratch.

**(c) `comet` was the wrong package.** `pyproject.toml` declared both
`unbabel-comet` and `comet>=3.1.0` — the latter is an unrelated astronomy VOEvent
broker that shadows the `comet` module. It has been removed. **Re-sync your
environment before running anything**, or COMET will keep coming out NaN:

```bash
uv pip uninstall comet          # remove the astronomy package if already installed
uv sync                         # reinstall from the corrected pyproject.toml
python -c "from comet import download_model; print('COMET OK')"
```

**(d) The Sep 11 abstract deadline locks things you cannot change later.**
ICLR requires abstract registration five days before the paper. Once it passes:

- **No new authors may be added.** Author order can still change up to Sep 16;
  the author *set* cannot. Settle the author list before Sep 11.
- Every submission needs **at least one author registered to review**, and that
  author must be qualified (≥ 1 accepted publication at ICLR/NeurIPS/ICML/CVPR/
  AAAI or similar). If nobody on the author list qualifies, find out now, not in
  September. Authors submitting 3+ papers must review ≥ 6.
- Authors with no prior major-venue publication are limited to **1 paper**.

So the abstract — title, abstract text, author list, reviewer registration — is
due while Stage 2 is still running. Write it against the *claim*, not the
numbers.

### Compute reality

The local RTX 3050 (4 GB) cannot run any of this. Use the cluster. Slurm scripts
now exist in `scripts/` for all five model types (BLT, BPE-LCM, SONAR,
BPE-Transformer, Llama-8B) via `scripts/submit_*.sh` — submit through the
`scripts/sbatch.sh` wrapper so `CLUSTER_PARTITION`/`CLUSTER_MEM` from `.env` are
honored instead of the scripts' hardcoded Galvani defaults.

### Bar check

This is a main-conference submission, not a workshop paper. The reviewer pool
will not accept a single-language-pair result with no baselines. The cut line in
the last section is a *floor for submitting at all*, not a target.

---

## Stage 0 — Concept space (gates everything else)

Nothing downstream can run until the pooler and generative decoder exist; every
concept vector depends on this pooler.

```bash
python lcm_scripts/blt_decoder.py \
  --entropy_model patching_scratch/entropy_model_marathi.pt \
  --num_sentences 50000 --epochs 10 \
  --pooler_save_path lcm_models/blt_pooler.pth \
  --save_path lcm_models/blt_decoder.pth
```

Produces `lcm_models/blt_pooler.pth` + `lcm_models/blt_decoder.pth`.
**If you retrain the pooler later, every cached embedding must be regenerated.**

---

## Stage 1 — Pre-encode BLT embeddings (per fraction)

```bash
# 25% (~6h) / 50% (~12h) / 80% (~20h)
uv run lcm_scripts/train_lcm_blt.py \
  --entropy_model patching_scratch/entropy_model_marathi.pt \
  --fraction 0.25 --epochs 0 --batch_size 8 \
  --embed_cache embeddings/blt_embeddings_frac025.pth
```

Repeat with `--fraction 0.50 / 0.80` and matching `--embed_cache` names.
On the cluster: `sbatch scripts/encode_blt.sh 0.25` (and 0.50, 0.80).

---

## Stage 2 — Headline result: EN→MR translation ★ highest priority

One script covers the entire 3 fractions × 3 noise levels × 3 seeds grid for
BLT-LCM. **This is the single most valuable thing to run.**

```bash
for FRAC in 0.25 0.50 0.80; do
  for SEED in 42 43 44; do
    uv run lcm_scripts/train_lcm_blt_mt.py \
      --entropy_model patching_scratch/entropy_model_marathi.pt \
      --pooler lcm_models/blt_pooler.pth \
      --decoder lcm_models/blt_decoder.pth \
      --fraction $FRAC --epochs 3 \
      --seed $SEED --data_seed 42 \
      --noise_levels 0.0 0.1 0.2 \
      --comet_model Unbabel/wmt22-comet-da \
      --out_csv results/blt_lcm_mt_${FRAC}_s${SEED}.csv
  done
done
```

Notes on the flags:
- `--comet_model` **must** be passed — the default is `None` and COMET is NaN without it.
- `--seed` varies model init, batch order and the eval-noise draw → real error bars.
- `--data_seed 42` stays fixed so all seeds share one train/eval split.
- One CSV **per seed** — same `--out_csv` would overwrite. Checkpoints are already
  seed-suffixed (`lcm_blt_mt_s42_best.pth`).

CSV columns: `model, fraction, noise, seed, BLEU, chrF++, TER, METEOR, COMET`.

---

## Stage 3 — Baselines

Three at once:

```bash
uv run lcm_scripts/benchmark_bhashasetu_models.py \
  --models bpe_transformer bpe_lcm sonar_lcm \
  --fractions 0.25 0.50 0.80 --noise_levels 0.0 0.1 0.2 \
  --eval_docs 100 --epochs 1 \
  --out_dir runs/bhashasetu_benchmarks
```

Llama-8B separately (needs a Slurm script you don't have yet):

```bash
uv run lcm_scripts/train_bpe_llama8b.py \
  --fraction 0.25 --epochs 1 --batch_size 1 --grad_accum 16 \
  --qlora --noise_levels 0.0 0.1 0.2 \
  --out_dir runs/bpe_llama8b_25_qlora
```

**Priority if time runs out:** SONAR-LCM first (it is the paper's central
comparison — the whole claim is "replace SONAR"), then BPE-Transformer, then
BPE-LCM, then Llama-8B last. Note that at ICLR, "we compare only against
SONAR-LCM" is itself a likely review criticism — get BPE-Transformer in if the
schedule allows.

---

## Stage 4 — Linguistic analyses (CPU-only, cheap, mostly already done)

`fertility_chrf_scatter.py` **needs Stage 2 hypotheses** — it validates the α
exponent in §4, which is currently the paper's biggest unsupported claim.

```bash
export INDIC_RESOURCES_PATH="$PWD/indic_nlp_resources"

uv run lcm_scripts/fertility_chrf_scatter.py          # ← validates §4 (needs Stage 2)
uv run lcm_scripts/fertility_audit.py                 # λ per morpheme class
uv run morpheme_alignment/morpheme_boundary_alignment.py
uv run morpheme_alignment/patch_morpheme_example.py   # Figure 2 (already in repo)
uv run fixed_chunk_ablation/fixed_chunk_ablation.py   # 4- vs 8-byte ablation
uv run sweep_threshold/sweep_entropy_threshold.py     # justifies τ = 1.335
uv run tokenization_statistics/patch_compression_by_morpheme_class.py
uv run cross_script_sanity/hindi_entropy_sanity.py    # second-script evidence
```

Already-current outputs you can cite without re-running: `results/summary_statistics.json`,
`results/fertility_by_class.json`, `error_analysis/error_analysis_report.md`,
`cross_script_sanity/hindi_entropy_sanity_summary.csv`.

Appendix pages are unlimited at ICLR, so everything here can ship — but reviewers
are not required to read appendices. Anything load-bearing belongs in the 9 pages.

---

## Stage 5 — Writing (not compute-blocked — start now, in parallel)

### 5a. Retarget the LaTeX to the ICLR template

`neurips_paper/` currently builds against `neurips_2026.sty` with the
`[dblblindworkshop]` option. That style file will not be accepted. Migrate:

```bash
curl -L -o iclr2027.zip https://github.com/ICLR/Master-Template/raw/master/iclr2027.zip
unzip iclr2027.zip -d iclr_paper/
cp neurips_paper/paper.tex neurips_paper/references.bib neurips_paper/*.jpeg iclr_paper/
```

Then in `iclr_paper/paper.tex`:

- Replace `\usepackage[dblblindworkshop]{neurips_2026}` with
  `\usepackage{iclr2027_conference}` plus `\input{math_commands.tex}` if the
  template ships one. **Confirm the exact `.sty` basename after unzipping** — use
  whatever style file is in the archive, not the name assumed here.
- Camera-ready later switches to `\usepackage[final]{iclr2027_conference}`.
- Bibliography style becomes the template's `.bst` (`\bibliographystyle{iclr2027_conference}`),
  not `plainnat`.
- Delete the whole NeurIPS track-selection comment block — it is NeurIPS-specific
  and confusing to leave in.

Keep `neurips_paper/` around until `iclr_paper/` compiles, then drop it.

### 5b. Content status

| Item | Status |
|---|---|
| `references.bib` | **empty (0 bytes)**; `paper.tex` has **zero `\cite` calls** |
| §Background and Related Work | empty heading |
| §Tokenization Bias in SONAR | empty heading |
| §Results | placeholder sentence, no tables |
| §Ablation Studies / §Discussion / §Conclusion | empty headings |
| Abstract | must be final by **Sep 11** — write it before the numbers land |

### 5c. Required and recommended statements (replaces the NeurIPS checklist)

ICLR does **not** use the NeurIPS reproducibility checklist. Delete
`checklist.tex` and the `\input` that pulls it in; write these instead. None of
the three count against the 9 pages.

| Statement | Required? | Placement / limit |
|---|---|---|
| **AI Use Statement** | **Mandatory.** Also disclose LLM use in the OpenReview submission form. Authors are fully responsible for anything an LLM produced — a falsehood or plagiarism traced to an LLM is an author-side Code of Ethics violation. | After the main text, no page cost |
| Ethics Statement | Optional, recommended. Covers datasets, human subjects, bias, privacy — relevant here given BhashaSetu provenance and the low-resource-language framing. | ≤ 1 page, no page cost |
| Reproducibility Statement | Encouraged. Point at the anonymized code, the exact seeds/splits from Stage 2, and the dataset description. | No page cost |

### 5d. Build

```bash
cd iclr_paper && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

Check the compiled main text is ≤ 9 pages *before* references. It may grow to 10
during the discussion phase and for camera-ready — do not plan to use that
headroom at submission.

---

## Stage 6 — Anonymization sweep (before uploading anything)

Double-blind, and ICLR **desk rejects** on identity leakage in the main text *or*
the supplementary material. Scrub:

- Git remote `github.com/ParamThakkar123/BLT-LCM`
- HF dataset id `ParamTh/BhashaSetu` — appears in `README.md` (5×) and
  `lcm_scripts/bhashasetu_utils.py:18` (`DEFAULT_DATASET`)
- W&B entity `fyp-team-2513` (`README.md`), and `runs/*/wandb/` run metadata
- `updated_phase2_checklist.md` — names the institution and individual members
- `.env` — must not be included
- PDF metadata (`pdfauthor`, `pdfsubject`) — check with `pdfinfo paper.pdf`

Ship a clean export rather than the working tree:

```bash
git archive --format=zip HEAD -o anon_code.zip   # then scrub the identifiers above
```

The dataset *name* "BhashaSetu" in the paper text is fine; the `ParamTh/`
namespace is not.

ICLR gives three ways to release code — pick one:

1. Anonymized zip as supplementary material (due Sep 16 with the paper).
2. An anonymous repository, linked from the paper.
3. A repository link posted in a reviewer-only comment after forums open — this
   one lets you keep the deanonymized repo, so it is the least work.

Prior arXiv postings by the same authors do **not** break anonymity, but must be
cited in the third person. Same for anything already presented at a workshop —
that is permitted under the dual-submission policy.

---

## Suggested ordering

1. **Now** — env re-sync (§0c), settle the author list and reviewer registration
   (§0d), then launch Stage 0. It gates everything.
2. **While Stage 0/1 run** — Stage 5a template migration, Stage 5b writing,
   Stage 4 CPU analyses.
3. **As soon as Stage 0 lands** — Stage 2, all 9 runs. Do not wait on baselines.
4. **By Sep 11** — abstract + author list + reviewer registration submitted. This
   is a hard gate; nothing about it can be fixed after the fact.
5. **Sep 11–16** — Stage 3 results in, fill Results/Ablations from the CSVs,
   write the three statements (§5c), Stage 6 scrub, upload paper + supplementary.

**Cut line:** Stage 2 + Stage 4 + SONAR-LCM is the minimum submittable paper.
Llama-8B is the first thing to drop. Be aware this is the floor, not the target —
see the bar check in §0.
