"""Uniform device reporting for every runnable script in this repository.

Each entry point calls :func:`report_device` once, immediately after it resolves
its device, so the first thing any log shows is whether the run is on a GPU or
on the CPU::

    [device] cuda:0 | NVIDIA A100-SXM4-40GB | 39.4 GiB | capability 8.0 | torch 2.5.1+cu121 | CUDA 12.1

That matters most where it is easiest to miss: a Slurm job whose ``--gres`` did
not take, or a `--device cuda` typo, otherwise looks identical to a healthy run
until it is twenty hours slower than expected.

This module reports; it does not silently change what a script asked for. If
``cuda`` was requested on a machine without it, the request is echoed back with
a loud warning rather than being downgraded to ``cpu`` behind your back — an
unnoticed CPU fallback on a cluster is worse than a clear failure.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

__all__ = ["resolve_device", "describe_device", "report_device"]


def resolve_device(requested: Any = None) -> torch.device:
    """Turn ``None`` / ``"auto"`` / a string / a ``torch.device`` into a device.

    ``None`` and ``"auto"`` select CUDA when it is available and CPU otherwise.
    Anything else is honoured exactly as given.
    """
    if requested is None or (isinstance(requested, str) and requested.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return requested if isinstance(requested, torch.device) else torch.device(requested)


def describe_device(device: Any = None) -> str:
    """One-line description of ``device``: name, memory, and the torch build."""
    dev = resolve_device(device)
    parts: list[str] = []

    if dev.type == "cuda":
        if torch.cuda.is_available():
            index = dev.index if dev.index is not None else torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            parts += [
                f"cuda:{index}",
                props.name,
                f"{props.total_memory / 2**30:.1f} GiB",
                f"capability {props.major}.{props.minor}",
            ]
            if torch.cuda.device_count() > 1:
                parts.append(f"{torch.cuda.device_count()} GPUs visible")
        else:
            parts += [str(dev), "REQUESTED BUT UNAVAILABLE"]
    else:
        parts += [str(dev), f"{os.cpu_count()} cores", f"{torch.get_num_threads()} threads"]

    parts.append(f"torch {torch.__version__}")
    if torch.version.cuda:
        parts.append(f"CUDA {torch.version.cuda}")
    return " | ".join(parts)


def report_device(
    device: Any = None,
    *,
    label: Optional[str] = None,
    logger: Any = None,
    warn_cpu: bool = True,
) -> torch.device:
    """Print which device this run uses and return it as a ``torch.device``.

    Args:
        device: what the script resolved (``None``/``"auto"`` auto-selects).
        label: optional prefix, e.g. the stage name, when one script reports
            more than once.
        logger: a ``logging.Logger`` to write through instead of ``print``.
        warn_cpu: also note that CPU will be slow. Pass ``False`` in scripts
            where CPU is the normal, expected choice.

    Returns:
        The resolved device, so call sites can write ``device = report_device(args.device)``.
    """
    dev = resolve_device(device)

    def emit(message: str, level: str = "info") -> None:
        if logger is not None:
            getattr(logger, level)(message)
        else:
            print(message, flush=True)

    prefix = f"[device] {label}: " if label else "[device] "
    emit(prefix + describe_device(dev))

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        emit(f"[device] CUDA_VISIBLE_DEVICES={visible!r}")

    if dev.type == "cuda" and not torch.cuda.is_available():
        emit(
            "[device] WARNING: CUDA was requested but torch.cuda.is_available() is "
            "False, so this run will fail as soon as a tensor is moved to the GPU. "
            "Check the driver/toolkit, the Slurm --gres allocation, or pass "
            "--device cpu deliberately.",
            "warning",
        )
    elif dev.type == "cpu":
        if torch.cuda.is_available():
            emit(
                "[device] WARNING: running on CPU even though a CUDA device is "
                "available -- pass --device cuda to use it.",
                "warning",
            )
        elif warn_cpu:
            emit("[device] No CUDA device found; this will be substantially slower.")

    return dev
