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

Current 100-sentence result with `patching_scratch/entropy_model_marathi.pt` and
BLT default threshold `1.335442066192627`:

| Language | Script | Sentences | Precision@±2 | Recall@±2 | F1@±2 | Patches/sentence |
|---|---|---:|---:|---:|---:|---:|
| Hindi | Devanagari | 100 | 0.548 | 0.296 | 0.378 | 4.8 |

Interpretation: this is evidence that the entropy patching signal is not limited
to the Marathi sentence set, but it is a sanity check rather than a full Hindi
morpheme-alignment benchmark because the comparison boundaries are heuristic
proxies rather than expert gold segmentations.
