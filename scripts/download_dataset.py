"""
Download the BhashaSetu dataset to the local HuggingFace cache.

Run this once on a node with internet access (e.g. login node) before
launching training jobs on compute nodes with HF_DATASETS_OFFLINE=1.

Usage:
  uv run scripts/download_dataset.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset

repo_id = "ParamTh/BhashaSetu"
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

print(f"Downloading {repo_id} to {hf_home} ...")
ds = load_dataset(repo_id, split="train")
print(f"Done. {len(ds)} rows cached at: {hf_home}")
