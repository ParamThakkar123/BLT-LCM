# Cross-script Hindi sanity check

This directory records a small second-script check for BLT entropy patching.
The experiment applies the Marathi-trained byte entropy model to 100 Hindi
(Devanagari) sentences and compares entropy patch starts with lightweight Hindi
boundary proxies: word starts, postposition starts, and common suffix starts.

Artifacts:

- `hindi_entropy_sanity.py` — reproducible runner.
- `hindi_entropy_sanity.jsonl` — one record per Hindi sentence with patch starts,
  proxy boundaries, and boundary precision/recall/F1 using a ±2 byte tolerance.
- `hindi_entropy_sanity_summary.csv` — aggregate metrics for the 100-sentence run.


## How to run

From the repository root, run:

```bash
python cross_script_sanity/hindi_entropy_sanity.py --num_sentences 100 --device cpu
```

The script defaults to `patching_scratch/entropy_model_marathi.pt`, the BLT default entropy threshold, and `cross_script_sanity/` as the output directory. If CUDA is available, you can omit `--device cpu` or pass `--device cuda` for a faster run. To write outputs elsewhere or use a different checkpoint, run:

```bash
python cross_script_sanity/hindi_entropy_sanity.py \
  --model patching_scratch/entropy_model_marathi.pt \
  --threshold 1.335442066192627 \
  --num_sentences 100 \
  --output_dir cross_script_sanity \
  --device cpu
```

Expected outputs are `hindi_entropy_sanity.jsonl` and `hindi_entropy_sanity_summary.csv` in the chosen output directory.

Current 100-sentence result with `patching_scratch/entropy_model_marathi.pt` and
BLT default threshold `1.335442066192627`:

| Language | Script | Sentences | Precision@±2 | Recall@±2 | F1@±2 | Patches/sentence |
|---|---|---:|---:|---:|---:|---:|
| Hindi | Devanagari | 100 | 0.548 | 0.296 | 0.378 | 4.8 |

Interpretation: this is evidence that the entropy patching signal is not limited
to the Marathi sentence set, but it is a sanity check rather than a full Hindi
morpheme-alignment benchmark because the comparison boundaries are heuristic
proxies rather than expert gold segmentations.
