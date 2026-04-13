# BLT-LCM Marathi Machine Translation

This project implements BLT-LCM (Byte Latent Transformer - Large Concept Model) for script-agnostic concept extraction in Marathi machine translation, evaluating against baselines like LCM + SONAR.

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd blt-lcm-marathi
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```

## Data Preparation

1. Extract Marathi sentences:
   ```bash
   python extract_marathi.py
   ```

2. Run entropy-based patching:
   ```bash
   python patching_script.py
   ```

3. Load MT data:
   ```bash
   python load_mt_data.py
   ```

## Training

Train LCM with different modes:

- Fallback (entropy-based patching):
  ```bash
  python train_blt_lcm.py --mode fallback --steps 1000
  ```

- SONAR concepts:
  ```bash
  python train_blt_lcm.py --mode sonar --steps 1000
  ```

- BLT (if available, otherwise falls back):
  ```bash
  python train_blt_lcm.py --mode blt --steps 1000
  ```

This saves models as `lcm_{mode}.pth`.

## Evaluation

Evaluate concept quality (reconstruction loss) for each model:

- Fallback:
  ```bash
  python evaluate_concept_quality.py --mode fallback --model_path lcm_fallback.pth
  ```

- SONAR:
  ```bash
  python evaluate_concept_quality.py --mode sonar --model_path lcm_sonar.pth
  ```

- BLT:
  ```bash
  python evaluate_concept_quality.py --mode blt --model_path lcm_blt.pth
  ```

Compare the average reconstruction losses; lower is better. BLT-LCM should outperform LCM + SONAR in concept extraction.

## Results

Run all evaluations and compare losses to demonstrate BLT-LCM superiority for Marathi MT concepts.