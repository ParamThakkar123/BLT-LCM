"""Simple reproducible experiment configuration and logging helpers.

Provides YAML-based configs and a helper to setup TensorBoard logging and a
lightweight VRAM safety check.
"""

import os
import yaml
import torch
from datetime import datetime

# Top-level optional imports
try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TB = True
except Exception:
    SummaryWriter = None
    _HAS_TB = False

try:
    import pynvml

    _HAS_NVML = True
except Exception:
    pynvml = None
    _HAS_NVML = False

try:
    import wandb

    _HAS_WANDB = True
except Exception:
    wandb = None
    _HAS_WANDB = False

try:
    import numpy as _np
except Exception:
    _np = None


def save_config(config: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str):
    # TensorBoard writer
    if not _HAS_TB:
        print("TensorBoard not available; skipping SummaryWriter")
        return None
    os.makedirs(log_dir, exist_ok=True)
    tb = SummaryWriter(log_dir)
    return tb


def setup_wandb(
    log_dir: str,
    project: str = None,
    name: str = None,
    entity: str = None,
    config: dict = None,
):
    """Initialize a W&B run if available. Returns the wandb module or None.

    The function does nothing if wandb is not installed.
    """
    if not _HAS_WANDB:
        print("wandb not available; skipping W&B logging")
        return None
    os.makedirs(log_dir, exist_ok=True)
    try:
        proj = project or "blt-lcm"
        run = wandb.init(
            project=proj, name=name, entity=entity, config=config or {}, dir=log_dir
        )
        return wandb
    except Exception as e:
        print(f"wandb.init failed: {e}; skipping W&B")
        return None


def check_vram_safety(max_req_gb: float = 8.0) -> bool:
    # Simple heuristic: check total CUDA memory if available
    if not torch.cuda.is_available():
        return True
    if not _HAS_NVML:
        print("pynvml not installed or failed; cannot check VRAM precisely")
        return True
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_gb = info.total / (1024**3)
        if total_gb < max_req_gb:
            print(f"Warning: GPU total memory {total_gb:.1f}GB < {max_req_gb}GB")
        return True
    except Exception:
        print("pynvml failed during runtime; cannot check VRAM precisely")
        return True


def new_run_dir(base: str = "runs") -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, ts)
    os.makedirs(path, exist_ok=True)
    return path


if __name__ == "__main__":
    from device_utils import report_device

    report_device()
    print(
        "experiment_config helpers: save_config/load_config/setup_logging/check_vram_safety"
    )
