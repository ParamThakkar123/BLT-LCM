"""
Download the BhashaSetu dataset to the local HuggingFace cache.

Run this once on a node with internet access (e.g. login node) before
launching training jobs on compute nodes with HF_DATASETS_OFFLINE=1.

Usage:
  uv run scripts/download_dataset.py
"""

import sys

if __name__ != "__main__":
    sys.exit(0)

import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcm_scripts.device_utils import report_cpu_only

from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset

repo_id = "ParamTh/BhashaSetu"
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

report_cpu_only("dataset download, no model involved")
print(f"Downloading {repo_id} to {hf_home} ...")
ds = load_dataset(repo_id, split="train")
print(f"Done. {len(ds)} rows cached at: {hf_home}")
