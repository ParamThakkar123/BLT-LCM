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

from huggingface_hub import snapshot_download

repo_id = "ParamTh/BhashaSetu"
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

print(f"Downloading {repo_id} to {hf_home} ...")
path = snapshot_download(repo_id=repo_id, repo_type="dataset")
print(f"Done. Dataset cached at: {path}")
