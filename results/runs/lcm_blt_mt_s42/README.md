# lcm_blt_mt_s42

* script: `train_lcm_blt_mt.py`
* finished: 2026-08-16T05:35:01Z
* wall clock: 0.0 s
* code: `667c833` on `main` (working tree dirty)
* device: NVIDIA RTX PRO 6000 Blackwell Server Edition

## Results

| metric | value |
| --- | --- |
| final_train_loss | 0.15960174307065353 |
| best_epoch | 3 |
| epochs_run | 3 |
| clean_fraction | 0.25 |
| clean_noise | 0.0 |
| clean_BLEU | 0.04234823765255262 |
| clean_chrF++ | 9.222575762772504 |
| clean_TER | 97.0472205618649 |
| clean_METEOR | 2.5852228284481718 |
| clean_COMET | 0.28030647844076156 |

## Run info

| key | value |
| --- | --- |
| seed | 42 |
| data_seed | 42 |
| train_pairs | 698519 |
| eval_pairs | 500 |
| noise_levels | [0.0, 0.1, 0.2] |
| mode | fixed |
| cap | 3 |
| patience | 3 |
| min_delta | 0.0 |
| min_epochs | 1 |
| epochs_observed | 3 |
| best | 0.15960174307065353 |
| best_epoch | 3 |
| epochs_since_improvement | 0 |
| stop_reason | reached the --epochs cap of 3 |

## Figures

### lcm_blt_mt_s42_metrics_table.png

![lcm_blt_mt_s42_metrics_table.png](lcm_blt_mt_s42_metrics_table.png)

### lcm_blt_mt_s42_noise_robustness.png

![lcm_blt_mt_s42_noise_robustness.png](lcm_blt_mt_s42_noise_robustness.png)

### lcm_blt_mt_s42_training_curve.png

![lcm_blt_mt_s42_training_curve.png](lcm_blt_mt_s42_training_curve.png)

### lcm_blt_mt_s42_training_dashboard.png

![lcm_blt_mt_s42_training_dashboard.png](lcm_blt_mt_s42_training_dashboard.png)


Full hyperparameters, metrics and environment: [`run.json`](run.json).
