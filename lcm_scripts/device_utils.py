"""Uniform device reporting for every runnable script in this repository.

Each entry point reports its device once, before it loads a model or touches the
dataset, so the first thing any log shows is where the work will happen::

    [device] cuda:0 | NVIDIA A100-SXM4-40GB | 39.4 GiB | capability 8.0 | torch 2.5.1+cu121 | CUDA 12.1
    [device] cpu | 64 cores | CPU-only: matplotlib rendering, no tensor computation

That matters most where it is easiest to miss: a Slurm job whose ``--gres`` did
not take, or a ``--device cuda`` typo, otherwise looks identical to a healthy
run until it is twenty hours slower than expected.

Two entry points, depending on what the script actually does:

``report_device``
    Scripts that put tensors on a device -- directly, or through a library that
    picks a GPU on its own (Stanza, COMET, HuggingFace Trainer).

``report_cpu_only``
    Scripts that do no tensor work at all -- plotting, CSV aggregation, dataset
    streaming, pure-python corpus scans. These never import torch, so the report
    costs nothing and the ``reason`` says why CPU is not a fallback here.

This module reports; it does not silently change what a script asked for. If
``cuda`` was requested on a machine without it, the request is echoed back with
a loud warning rather than being downgraded to ``cpu`` behind your back -- an
unnoticed CPU fallback on a cluster is worse than a clear failure.
"""

from __future__ import annotations

import os
from typing import Any, Optional

__all__ = [
    "resolve_device",
    "describe_device",
    "enable_tf32",
    "report_device",
    "report_cpu_only",
]


def _emitter(logger: Any = None):
    """Write through a logging.Logger when given one, else print."""

    def emit(message: str, level: str = "info") -> None:
        if logger is not None:
            getattr(logger, level)(message)
        else:
            print(message, flush=True)

    return emit


# ── Scripts that run tensors ─────────────────────────────────────────────────


def resolve_device(requested: Any = None):
    """Turn ``None`` / ``"auto"`` / a string / a ``torch.device`` into a device.

    ``None`` and ``"auto"`` select CUDA when it is available and CPU otherwise.
    Anything else is honoured exactly as given.
    """
    import torch  # imported lazily so CPU-only scripts never pay for it

    if requested is None or (isinstance(requested, str) and requested.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return requested if isinstance(requested, torch.device) else torch.device(requested)


def describe_device(device: Any = None) -> str:
    """One-line description of ``device``: name, memory, and the torch build."""
    import torch

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


def enable_tf32(emit: Any = None) -> bool:
    """Allow TF32 for matmul and cuDNN on Ampere and newer.

    TF32 keeps fp32 range with 10 bits of mantissa and runs on the tensor
    cores, which is roughly a 1.5-2x speedup on the matmul-bound parts of
    training and evaluation. The precision loss is immaterial next to the
    variance of these runs.

    This used to be set in exactly one script (``train_lcm_blt.py``), so every
    other trainer and every eval script silently ran fp32 matmuls on hardware
    that could do better. Reporting the device is the one thing all of them do
    first, which makes it the right place to set it once.

    Returns True when TF32 was actually enabled.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        major = torch.cuda.get_device_properties(torch.cuda.current_device()).major
    except Exception:
        return False
    if major < 8:  # pre-Ampere has no TF32 path
        return False

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if emit is not None:
        emit("[device] TF32 enabled for matmul and cuDNN (Ampere+)")
    return True


def report_device(
    device: Any = None,
    *,
    label: Optional[str] = None,
    logger: Any = None,
    warn_cpu: bool = True,
    tf32: bool = True,
):
    """Print which device this run uses and return it as a ``torch.device``.

    Args:
        device: what the script resolved (``None``/``"auto"`` auto-selects).
        label: optional prefix, e.g. the stage name, when one script reports
            more than once, or the library that will consume the device.
        logger: a ``logging.Logger`` to write through instead of ``print``.
        warn_cpu: emit the CPU notes -- that a GPU was available but not used,
            or that no GPU was found and this will be slow. Pass ``False`` where
            CPU is a deliberate choice (a pinned smoke test) or where the
            calling script has already reported the same device once.
        tf32: allow TF32 matmuls when the device supports them. Pass ``False``
            for a run that needs bit-comparable fp32 arithmetic.

    Returns:
        The resolved device, so call sites can write
        ``device = report_device(args.device)``.
    """
    import torch

    dev = resolve_device(device)
    emit = _emitter(logger)

    prefix = f"[device] {label}: " if label else "[device] "
    emit(prefix + describe_device(dev))

    if tf32 and dev.type == "cuda":
        enable_tf32(emit)

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
    elif dev.type == "cpu" and warn_cpu:
        if torch.cuda.is_available():
            emit(
                "[device] WARNING: running on CPU even though a CUDA device is "
                "available -- pass --device cuda to use it.",
                "warning",
            )
        else:
            emit("[device] No CUDA device found; this will be substantially slower.")

    return dev


# ── Scripts that never touch a tensor ────────────────────────────────────────


def report_cpu_only(reason: str = "", *, label: Optional[str] = None, logger: Any = None) -> str:
    """Report that this script is CPU-only by construction, not by fallback.

    Deliberately does not import torch: these scripts are plotting, aggregation
    and pure-python passes, and paying seconds of import time to print one line
    would be a poor trade. ``reason`` should say what the work actually is, so
    the line reads as a fact about the script rather than a missing GPU.
    """
    prefix = f"[device] {label}: " if label else "[device] "
    line = f"{prefix}cpu | {os.cpu_count()} cores"
    if reason:
        line += f" | CPU-only: {reason}"
    _emitter(logger)(line)
    return "cpu"
