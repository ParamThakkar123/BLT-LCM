"""Extract Marathi sentences from BhashaSetu into marathi_sentences.json.

Resumable: sentences are appended to a JSONL progress file as they stream in,
so an interrupted extraction restarts at the first sentence it had not written
instead of re-streaming the dataset from the beginning. The final
``marathi_sentences.json`` is assembled from that file.

Usage:
  python extract_marathi.py --num_sentences 50000
  python extract_marathi.py --resume never      # ignore partial output
"""

if __name__ != "__main__":
    raise ImportError(
        "This module is a script and should not be imported. Run it directly."
    )

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset

from lcm_scripts.checkpoint_utils import ResumableJsonl, config_fingerprint

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="ParamTh/BhashaSetu")
parser.add_argument("--split", default="train")
parser.add_argument("--num_sentences", type=int, default=50000)
parser.add_argument("--out", default="marathi_sentences.json")
parser.add_argument(
    "--progress_jsonl",
    default="marathi_sentences.progress.jsonl",
    help="Where sentences are appended as they arrive; replayed on resume.",
)
parser.add_argument(
    "--resume",
    default="auto",
    metavar="auto|never",
    help="'auto' (default) continues from the progress file if its "
    "configuration matches; 'never' discards it and starts over.",
)
args = parser.parse_args()

fingerprint = config_fingerprint(
    {
        "dataset": args.dataset,
        "split": args.split,
        "num_sentences": args.num_sentences,
    }
)

writer = ResumableJsonl(
    args.progress_jsonl,
    fingerprint=fingerprint,
    resume=args.resume != "never",
    key="index",
    flush_every=1000,
)
already = len(writer.done)
if already:
    print(f"Resuming: {already}/{args.num_sentences} sentences already extracted")

if already < args.num_sentences:
    # The dataset is streamed in order, so the first `already` rows are exactly
    # the ones on disk; skip past them rather than rewriting them.
    ds = load_dataset(args.dataset, streaming=True, split=args.split)
    for count, sample in enumerate(ds):
        if count >= args.num_sentences:
            break
        if writer.is_done(count):
            continue
        writer.append({"index": count, "marathi": sample["marathi"]})
        if (count + 1) % 5000 == 0:
            print(f"  Extracted {count + 1}/{args.num_sentences}...")
writer.close()

marathi_sentences = [r["marathi"] for r in writer.all_records()]

with open(args.out, "w", encoding="utf-8") as f:
    json.dump(marathi_sentences, f, ensure_ascii=False, indent=2)

print(f"Extracted {len(marathi_sentences)} Marathi sentences to {args.out}.")
