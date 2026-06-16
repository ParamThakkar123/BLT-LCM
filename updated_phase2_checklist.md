# Phase 2 — 10-Day Sprint Checklist

> BLT-LCM Marathi MT | VJTI Group 2513
> 

---

## 🔴 Day 1 — Baseline Repair

> Owner focus: Anushka (metrics)
> 
- [x]  **Run full metric suite on Phase 1 model** *(Anushka)*
    - [x]  BLEU (already have 21.7 — reconfirm)
    - [x]  chrF++
    - [x]  METEOR
    - [x]  COMET
    - [x]  TER
    - [x]  ROUGE-L
    - *Use existing Notebook 6 — no retraining needed. ~3 hrs compute.*
- [ ]  **Fertility audit by morpheme class** *(Anushka)*
    - [ ]  Noun roots
    - [ ]  Verb inflections
    - [ ]  Compound words
    - [ ]  Postpositions
    - [ ]  Output: table of fertility (λ) per morpheme class
    - *Use Indic NLP library for POS tagging. This λ feeds the theoretical bound.*

---

## 🔴 Day 2 — BLT Patch Extraction

> Owner focus: Param (BLT setup + sweep script)
> 
- [x]  **Run BLT entropy-based patching on 10K Marathi sentences** *(Param)*
    - [x]  Use pre-trained BLT encoder (Can also train from scratch)
    - [x]  Record patch boundaries per sentence
    - [x]  Record entropy value at each boundary
    - [x]  Save output as new JSONL file
- [x]  **Set up entropy threshold sweep script** *(Param)*
    - [x]  Implement sweep for τ ∈ {0.5, 1.0, 1.5, 2.0, 2.5}
    - [x]  Verify script runs end-to-end without errors
    - [x]  Record number of patches produced per sentence at each τ
- [x]  **Cross-script sanity check on a second non-Latin script** *(Param)*
    - [x]  Ran the Marathi-trained entropy patcher on 100 Hindi (Devanagari) sentences via `cross_script_sanity/hindi_entropy_sanity.py`
    - [x]  Saved per-sentence patch boundaries and proxy boundary F1 to `cross_script_sanity/hindi_entropy_sanity.jsonl`
    - [x]  Recorded summary evidence in `cross_script_sanity/hindi_entropy_sanity_summary.csv`: mean boundary F1@±2 bytes = 0.378, precision = 0.548, recall = 0.296
    - *Interpretation: preliminary second-script evidence supports a cautious script-agnostic sanity claim, but the proxy Hindi boundaries are lightweight and should not be framed as full gold morphological evaluation.*

---

## 🔴 Day 3 — Morpheme Alignment + Launch Training

> Owner focus: Anushka (alignment) · Param (training launch)
> 
- [x]  **Patch boundary vs. morpheme boundary alignment study** *(Anushka)*
    - [x]  Pull 200 Marathi sentences from test set
    - [x]  Get gold morpheme segmentations via Indic NLP morphological analyzer
    - [x]  For each τ: compute precision, recall, F1 of boundary overlap
    - [x]  Plot F1 vs. τ curve
    - [x]  Annotate chosen τ (highest F1) — this is your justified hyperparameter
- [ ]  **Launch BLT-LCM fine-tuning run overnight** *(Param)*
    - [ ]  Connect BLT patch representations to QLoRA pipeline
    - [ ]  Same hyperparameters as Phase 1 (lr=1e-4, cosine, 10K steps)
    - [ ]  Confirm training is running before end of day
    - [ ]  **Fallback if pipeline breaks:** concatenate BLT patch boundary positions as additional features on top of BPE embeddings — do not spend more than 2 hours debugging before switching to fallback

---

## 🔴 Day 4 — Monitor Training + Run Ablations

> Owner focus: Param (training) · Anushka (ablation + noisy test)
> 
- [ ]  **Monitor overnight training run** *(Param)*
    - [ ]  Check TensorBoard loss curves — must be smooth and declining
    - [ ]  Check VRAM stays below 11 GB
    - [ ]  If crashed: diagnose and relaunch immediately
- [ ]  **Fixed-length chunking ablation** *(Anushka)*
    - [ ]  Run BLT with fixed 4-byte chunks on same 200 alignment sentences
    - [ ]  Run BLT with fixed 8-byte chunks on same 200 alignment sentences
    - [ ]  Compute boundary F1 for both
    - [ ]  Compare against entropy-adaptive F1 from Day 3
    - *This proves entropy-adaptivity specifically (not just byte-level) is what matters*
- [ ]  **Build noisy input test set + evaluate Phase 1 model** *(Anushka)*
    - [ ]  Create 100-sentence noisy Marathi test set
        - [ ]  10% noise level: missing matras, character substitutions
        - [ ]  20% noise level: same transformations, higher rate
    - [ ]  Run Phase 1 model on both noise levels
    - [ ]  Record chrF++ at 0%, 10%, 20% noise — Phase 1 baseline degradation curve

---

## 🔴 Day 5 — Training Completion + Data Efficiency

> Owner focus: Param (training + efficiency runs + checkpoint)
> 
- [ ]  **Data efficiency runs** *(Param)*
    - [ ]  Train BLT-LCM on 25% data subset (max_steps = 2500)
    - [ ]  Train BLT-LCM on 50% data subset (max_steps = 5000)
    - [ ]  Record chrF++ for each subset
    - [ ]  Compare against Phase 1 at same data fractions
- [ ]  **Verify main BLT-LCM training has converged** *(Param)*
    - [ ]  Loss curve is smooth and has flattened
    - [ ]  Save best checkpoint
    - [ ]  Do not exceed 10K steps — same budget as Phase 1

---

## 🔴 Day 6 — Full Evaluation Day *(most important day)*

> Owner focus: Anushka (metrics + plots) · Param (latency)
⚠️ No new experiments today. Analysis only.
> 
- [ ]  **Run complete metric suite on BLT-LCM model** *(Anushka)*
    - [ ]  BLEU
    - [ ]  chrF++ ← headline metric
    - [ ]  METEOR
    - [ ]  COMET
    - [ ]  TER
    - [ ]  Run 3× with different decoding seeds, report mean
- [ ]  **Noisy input evaluation on BLT-LCM** *(Anushka)*
    - [ ]  Run Day 4 noisy test set through Phase 2 model
    - [ ]  Plot Phase 1 vs. Phase 2 degradation curves on same graph
- [x]  **Fertility vs. gain scatter plot** *(Anushka)*
    - [x]  X-axis: fertility λ per morpheme class (from Day 1)
    - [x]  Y-axis: Δ chrF++ (Phase 2 minus Phase 1) per morpheme class
    - [x]  If high-fertility classes show larger gains → theoretical bound is empirically validated
    - [x]  Implemented via `lcm_scripts/fertility_chrf_scatter.py`, which joins `fertility_by_class_detail.jsonl` sentence classes to BPE/BLT hypothesis files, writes `results/fertility_chrf_delta_by_class.csv`, plots `results/fertility_chrf_delta_scatter.png`, and reports the empirical bound diagnostic `E(BLT) ≤ λ^-α·E(BPE)`.
- [ ]  **Inference latency comparison** *(Param)*
    - [ ]  Time 500 inference calls on Phase 1 model → sentences/sec + peak VRAM
    - [ ]  Time 500 inference calls on Phase 2 model → sentences/sec + peak VRAM
    - [ ]  Record in a 2-row comparison table

---

## 🟡 Day 7 — Error Analysis + Human Evaluation

> Owner focus: Anushka (error analysis) · All (human eval)
> 
- [ ]  **Error analysis on 100 failure sentences** *(Anushka)*
    - [ ]  Find 100 sentences where Phase 2 scores lower than Phase 1
    - [ ]  Categorize each failure:
        - [ ]  Long compound words
        - [ ]  Code-mixed Marathi-English input
        - [ ]  Rare Unicode / uncommon Devanagari sequences
        - [ ]  Very short sentences (< 5 words)
        - [ ]  Domain-specific vocabulary (legal, medical)
    - [ ]  Output: failure category breakdown table → goes into Limitations section
- [ ]  **Lightweight human evaluation** *(All)*
    - [ ]  2–3 native Marathi speakers (team members or friends acceptable)
    - [ ]  50 sentences, both models, blind rating
    - [ ]  Rate on 1–5 scale: Fluency · Semantic accuracy
    - [ ]  Record mean rating per model
    - [ ]  Note small evaluator pool as a limitation in the report

---

## 🔴 Day 8 — Write Chapters 3 and 4

> Owner focus: Param (Ch3) · Anushka (Ch4)
> 
- [ ]  **Chapter 3 — BLT-LCM Architecture** *(Param)*
    - [ ]  BLT byte-patch encoder description
    - [ ]  Entropy threshold selection with justification (cite Day 3 F1 curve)
    - [ ]  Concept aggregation layer: entropy-weighted pooling explanation
    - [ ]  Integration with QLoRA pipeline
    - [ ]  Simplified theoretical bound: E(BLT) ≤ f(λ) · E(BPE)
    - [ ]  One architecture diagram
- [ ]  **Chapter 4 — Results** *(Anushka)*
    - [ ]  Main results table: Phase 1 vs. BLT-LCM, all metrics
    - [ ]  Morpheme boundary alignment figure (patch F1 vs. τ)
    - [ ]  Fixed-length chunking ablation table
    - [ ]  Noisy input degradation plot
    - [ ]  Fertility vs. Δ chrF++ scatter
    - [ ]  Data efficiency learning curves
    - [ ]  Inference latency table
    - [ ]  Human evaluation results

---

## 🔴 Day 9 — Write Remaining Chapters + First Review

> Owner focus: Anushka + Param (writing) · Param (notebooks) · All (review)
> 
- [ ]  **Chapter 2 — Updated Related Work** *(Anushka + Param)*
    - [ ]  Add BLT paper with differentiation sentence
    - [ ]  Add LCM paper with differentiation sentence
    - [ ]  Add 2–3 recent (2024–25) byte-level / low-resource MT papers
    - [ ]  Every paragraph ends with "unlike X, we..."
- [ ]  **Chapter 5 — Conclusion** *(Anushka + Param)*
    - [ ]  What Phase 2 added over Phase 1 (3 bullet contributions)
    - [ ]  3 main empirical findings
    - [ ]  Honest limitations from Day 7 error analysis
    - [ ]  Future work: multi-language extension · RLHF · speech integration
- [ ]  **Update Colab notebooks** *(Param)*
    - [ ]  Notebook 8: BLT patch extraction
    - [ ]  Notebook 9: BLT-LCM fine-tuning
    - [ ]  Notebook 10: Phase 2 evaluation
    - [ ]  Verify all 10 notebooks run top-to-bottom on a fresh session
- [ ]  **First full draft review** *(All)*
    - [ ]  All four members read independently
    - [ ]  Mark: unclear claims · missing citations · results without experiments
    - [ ]  Compile single shared fix list by end of day

---

## 🔴 Day 10 — Polish and Submit

> Owner focus: All team · Submit by early afternoon, not midnight
> 
- [ ]  **Work through Day 9 fix list** *(All)*
    - [ ]  Every claim has a citation or a result number
    - [ ]  Every figure is referenced in the body text
    - [ ]  Abstract matches what the paper actually shows
    - [ ]  No figure without a takeaway caption
    - [ ]  Page limits checked
- [ ]  **Final read-aloud check** *(All)*
    - [ ]  One person reads the full report aloud
    - [ ]  Every sentence you stumble on → rewrite it
- [ ]  **Submit** *(All)*
    - [ ]  Submit by early afternoon (3-hour buffer before deadline)
    - [ ]  Confirm submission confirmation email received

---

## ⚠️ Standing Rules for All 10 Days

- [ ]  If BLT pipeline integration breaks on Day 3 → switch to fallback immediately (do not debug > 2 hrs)
- [ ]  Day 6 is analysis only — no new experiments, no architecture changes
- [ ]  Writing starts Day 8 with placeholder numbers if needed — do not wait for perfect results
- [ ]  Training must finish by end of Day 5 — if it hasn't, cut max_steps and take what you have

---

*Group 2513 · VJTI Mumbai · Phase 2 Report · 2025–26* LCM Idea L6 — "BLT-LCM: Replacing SONAR with a Byte Latent Encoder for Script-Agnostic Concept Extraction"
The core architectural gap. The AI community is actively speculating about the synergy between LCM and BLT — the Byte Latent Transformer — and the combination is widely anticipated but completely unstudied. BLT's architecture could serve as a scalable encoder and decoder within the LCM framework, and LCM's current reliance on SONAR still uses token-level processing to develop the sentence embedding space. This is a direct invitation for your paper. The problem with SONAR for SEA languages is that it still fundamentally relies on token-level encodings that inherit the biases of BPE tokenization. BLT directly models raw bytes, making it more robust to noisy inputs and better at understanding sub-word aspects of the data including orthography, phonology, and low-resource machine translation.
The novel contribution. Propose a BLT-LCM hybrid: use BLT's entropy-based dynamic byte patching as the encoder/decoder instead of SONAR, passing the patch-level representations into a concept aggregation layer that produces a sentence-level concept vector. This eliminates tokenization bias entirely from the LCM pipeline — the concept space is now built from raw bytes, not from BPE tokens. For your SEA language, this should be a significant improvement because BLT's entropy-based patches will naturally align with morpheme boundaries in agglutinative SEA languages (high entropy at morpheme boundaries → new patch begins), producing concept representations that respect linguistic structure.
The theoretical angle. Derive a formal bound showing that BLT-style byte patching reduces the expected concept encoding error for agglutinative languages compared to SONAR, as a function of the language's morpheme-to-token ratio. Your fertility measurement infrastructure from the Ideas 1+3 paper gives you this ratio directly.
Why NeurIPS loves this. It integrates two of the most discussed unrealized combinations in the 2025 Meta AI ecosystem (LCM + BLT) and provides the first empirical evidence of whether this combination actually helps — specifically for the language class (low-resource, non-Latin-script, agglutinative) where the theoretical benefit is largest. This is a genuine architectural contribution, not an application paper.