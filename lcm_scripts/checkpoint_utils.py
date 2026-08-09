"""Checkpoint / resume helpers shared by every long-running script in this repo.

Three shapes of resumable work show up across the codebase, and this module has
one primitive for each.

``TrainingCheckpointer``
    Gradient-descent loops (``train_*.py``, ``finetune_lcm.py``,
    ``blt_decoder.py``, ``patching_scratch/run_blt_patching.py``). Persists
    model + optimizer + scheduler + epoch/step counters + RNG state, so a killed
    run restarts on the batch it died on rather than at the top of the epoch.

``ResumableJsonl``
    Per-record scans that emit one output row per input row (the corpus studies
    in ``sweep_threshold/``, ``morpheme_alignment/``, ``fixed_chunk_ablation/``,
    ``fertility_audit.py``, ``extract_marathi.py``). Rows are appended as they
    are produced; a rerun replays the file and skips what is already written.

``StageTracker``
    Coarse loops where each iteration is expensive but there are only a handful
    of them (per-noise-level evaluation, per-checkpoint metric suites).
    Completed stage results are memoized in a JSON sidecar.

All three key their state to a fingerprint of the run configuration. Resuming
into a run whose flags have changed is refused rather than silently splicing two
configurations together; pass ``--resume never`` to start fresh instead.

Checkpoint writes go through ``atomic_torch_save`` / ``atomic_write_text``: the
payload lands in a sibling ``.tmp`` file and is then ``os.replace``-d into
place, so a process killed mid-write leaves the previous checkpoint intact
instead of a truncated one.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

try:  # torch is required for TrainingCheckpointer but not for the JSONL/stage helpers
    import torch
except Exception:  # pragma: no cover - the pure-python analyses run without torch
    torch = None

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


CKPT_FORMAT_VERSION = 2

# Flags that change where output goes or how it is logged, but not what is
# computed. Excluded from the fingerprint so that (say) resuming with W&B off,
# or on a different device, is still allowed.
DEFAULT_FINGERPRINT_IGNORE = frozenset(
    {
        "resume",
        "device",
        "log_dir",
        "wandb",
        "wandb_project",
        "wandb_name",
        "wandb_entity",
        "save_interval_steps",
        "save_interval_seconds",
        "max_checkpoints",
        "comet_model",
        "out_csv",
        "out_dir",
        "model_dir",
        "output",
        "output_dir",
        "save_path",
        "pooler_save_path",
        "dry_run",
    }
)


# --------------------------------------------------------------------------- #
# Atomic writes
# --------------------------------------------------------------------------- #


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def atomic_torch_save(obj: Any, path: str) -> None:
    """``torch.save`` that can't leave a half-written checkpoint behind."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for atomic_torch_save")
    _ensure_parent(path)
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_text(text: str, path: str, encoding: str = "utf-8") -> None:
    """Write text so readers never observe a partial file."""
    _ensure_parent(path)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding=encoding, newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(obj: Any, path: str, indent: int = 2) -> None:
    atomic_write_text(
        json.dumps(obj, indent=indent, ensure_ascii=False, default=str), path
    )


def load_model_state(path_or_blob: Any, map_location: Any = "cpu") -> dict:
    """Extract a plain ``state_dict`` from anything this repo writes.

    Checkpoints gained an enclosing payload (optimizer, RNG, step counters) when
    resume support landed, but ``lcm_models/*.pth`` files produced before that —
    and the ones written by ``blt_decoder``/``run_blt_patching``, which nest the
    weights under their own key — are still around and still valid. This accepts
    all of those shapes so callers can keep doing
    ``model.load_state_dict(load_model_state(path))``.
    """
    blob = (
        torch_load(os.fspath(path_or_blob), map_location=map_location)
        if isinstance(path_or_blob, (str, os.PathLike))
        else path_or_blob
    )
    if isinstance(blob, dict):
        for key in ("model_state_dict", "state_dict", "model", "lcm"):
            if key in blob and isinstance(blob[key], dict):
                return blob[key]
    return blob


def torch_load(path: str, map_location: Any = "cpu") -> Any:
    """Load one of *our* checkpoints.

    ``weights_only=True`` became the torch default in 2.6 and rejects the
    optimizer/RNG/metadata payloads written here, so it is explicitly disabled.
    These files are produced by this repo, not fetched from elsewhere.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for torch_load")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only kwarg
        return torch.load(path, map_location=map_location)


# --------------------------------------------------------------------------- #
# Run fingerprinting
# --------------------------------------------------------------------------- #


def config_fingerprint(
    config: Any,
    ignore: Iterable[str] = DEFAULT_FINGERPRINT_IGNORE,
    extra: Optional[dict] = None,
) -> str:
    """Short stable hash of the settings that determine what a run computes.

    ``config`` may be an ``argparse.Namespace`` or a dict. Keys in ``ignore``
    (I/O destinations, logging toggles, checkpoint cadence) are dropped so that
    changing them does not invalidate an in-progress run.
    """
    if hasattr(config, "__dict__") and not isinstance(config, dict):
        config = vars(config)
    ignore = set(ignore)
    payload = {k: v for k, v in sorted(dict(config).items()) if k not in ignore}
    if extra:
        payload.update({f"__{k}": v for k, v in sorted(extra.items())})
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# RNG capture / restore
# --------------------------------------------------------------------------- #


def capture_rng_state() -> dict:
    """Snapshot every RNG that affects training order and augmentation."""
    state: dict[str, Any] = {"python": random.getstate()}
    if _np is not None:
        state["numpy"] = _np.random.get_state()
    if torch is not None:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[dict]) -> None:
    """Restore a snapshot from :func:`capture_rng_state`, skipping absent RNGs.

    Each generator is restored independently: a checkpoint written on a CUDA box
    and resumed on CPU (or with numpy missing) still restores what it can.
    """
    if not state:
        return
    if "python" in state:
        try:
            random.setstate(state["python"])
        except Exception:
            pass
    if _np is not None and "numpy" in state:
        try:
            _np.random.set_state(state["numpy"])
        except Exception:
            pass
    if torch is not None and "torch" in state:
        try:
            torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
        except Exception:
            pass
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            try:
                torch.cuda.set_rng_state_all(state["torch_cuda"])
            except Exception:
                pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    if _np is not None:
        _np.random.seed(seed % (2**32))
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Training checkpoints
# --------------------------------------------------------------------------- #


@dataclass
class ResumePoint:
    """Where a resumed run should pick up.

    ``start_epoch`` / ``start_batch`` are the *next* unit of work: a checkpoint
    taken after finishing batch 40 of epoch 2 yields ``start_epoch=2,
    start_batch=41``, and a checkpoint taken at the end of epoch 2 yields
    ``start_epoch=3, start_batch=0``.
    """

    start_epoch: int = 0
    start_batch: int = 0
    global_step: int = 0
    best_score: Optional[float] = None
    extra: dict = field(default_factory=dict)
    resumed: bool = False

    def batches_to_skip(self, epoch: int) -> int:
        """Batches already done in ``epoch`` (0 for every later epoch)."""
        return self.start_batch if (self.resumed and epoch == self.start_epoch) else 0


class TrainingCheckpointer:
    """Periodic + end-of-epoch checkpointing with mid-epoch resume.

    Layout inside ``out_dir``::

        <prefix>_last.pt      rolling checkpoint, overwritten (the resume target)
        <prefix>_best.pt      best-scoring checkpoint so far
        <prefix>_epoch{N}.pt  end-of-epoch snapshots, pruned to ``max_keep``

    ``_last.pt`` is what ``--resume auto`` picks up, and it is rewritten every
    ``save_interval_steps`` batches (and/or every ``save_interval_seconds``), so
    the worst case lost work is one interval rather than one epoch.
    """

    def __init__(
        self,
        out_dir: str,
        prefix: str,
        fingerprint: Optional[str] = None,
        max_keep: int = 5,
        save_interval_steps: int = 0,
        save_interval_seconds: float = 0.0,
        suffix: str = ".pth",
        verbose: bool = True,
    ):
        self.out_dir = out_dir
        self.prefix = prefix
        self.suffix = suffix
        self.fingerprint = fingerprint
        self.max_keep = max_keep
        self.save_interval_steps = max(0, int(save_interval_steps))
        self.save_interval_seconds = max(0.0, float(save_interval_seconds))
        self.verbose = verbose
        self._last_save_time = time.time()
        os.makedirs(out_dir, exist_ok=True)

    # -- paths ------------------------------------------------------------- #

    @property
    def last_path(self) -> str:
        return os.path.join(self.out_dir, f"{self.prefix}_last{self.suffix}")

    @property
    def best_path(self) -> str:
        return os.path.join(self.out_dir, f"{self.prefix}_best{self.suffix}")

    def epoch_path(self, epoch: int) -> str:
        return os.path.join(self.out_dir, f"{self.prefix}_epoch{epoch}{self.suffix}")

    # -- saving ------------------------------------------------------------ #

    def _payload(
        self,
        model,
        optimizer=None,
        scheduler=None,
        *,
        epoch: int,
        batch_in_epoch: int,
        global_step: int,
        best_score: Optional[float],
        extra: Optional[dict],
        epoch_completed: bool,
    ) -> dict:
        state = model.state_dict() if hasattr(model, "state_dict") else model
        return {
            "format_version": CKPT_FORMAT_VERSION,
            "fingerprint": self.fingerprint,
            "model_state_dict": state,
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "epoch_completed": epoch_completed,
            "global_step": global_step,
            "best_score": best_score,
            "rng_state": capture_rng_state(),
            "extra": extra or {},
            "saved_at": time.time(),
        }

    def save(
        self,
        model,
        optimizer=None,
        scheduler=None,
        *,
        epoch: int,
        batch_in_epoch: int = 0,
        global_step: int = 0,
        best_score: Optional[float] = None,
        extra: Optional[dict] = None,
        epoch_completed: bool = False,
        also: Sequence[str] = (),
    ) -> str:
        """Write ``_last.pt`` (plus any extra destinations) atomically."""
        payload = self._payload(
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            global_step=global_step,
            best_score=best_score,
            extra=extra,
            epoch_completed=epoch_completed,
        )
        atomic_torch_save(payload, self.last_path)
        for dest in also:
            _ensure_parent(dest)
            shutil.copyfile(self.last_path, dest)
        self._last_save_time = time.time()
        return self.last_path

    def maybe_save(self, model, optimizer=None, scheduler=None, **kw) -> bool:
        """Save only if the step/time interval has elapsed. Returns whether it did."""
        step = int(kw.get("global_step", 0))
        due = False
        if self.save_interval_steps and step and step % self.save_interval_steps == 0:
            due = True
        if (
            self.save_interval_seconds
            and (time.time() - self._last_save_time) >= self.save_interval_seconds
        ):
            due = True
        if not due:
            return False
        self.save(model, optimizer, scheduler, **kw)
        if self.verbose:
            print(
                f"  [checkpoint] step {step} -> {self.last_path}",
                flush=True,
            )
        return True

    def save_epoch(
        self,
        model,
        optimizer=None,
        scheduler=None,
        *,
        epoch: int,
        label: Optional[int] = None,
        **kw,
    ):
        """End-of-epoch snapshot: writes ``_last`` and ``_epoch{label}``.

        ``epoch`` is the 0-based loop index and drives the resume arithmetic;
        ``label`` is what goes in the filename and defaults to the 1-based
        ``epoch + 1`` that the rest of the repo (and the docs) already use.
        """
        path = self.epoch_path(epoch + 1 if label is None else label)
        self.save(
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            batch_in_epoch=0,
            epoch_completed=True,
            also=(path,),
            **kw,
        )
        self.prune()
        if self.verbose:
            print(f"  [checkpoint] epoch {epoch + 1} -> {path}", flush=True)
        return path

    def save_best(self, model, optimizer=None, scheduler=None, **kw) -> str:
        self.save(model, optimizer, scheduler, also=(self.best_path,), **kw)
        if self.verbose:
            print(f"  [checkpoint] new best -> {self.best_path}", flush=True)
        return self.best_path

    def prune(self) -> None:
        """Keep only the ``max_keep`` newest ``_epoch{N}.pt`` snapshots."""
        if self.max_keep <= 0:
            return
        prefix = f"{self.prefix}_epoch"
        found = []
        for name in os.listdir(self.out_dir):
            if not (name.startswith(prefix) and name.endswith(self.suffix)):
                continue
            try:
                n = int(name[len(prefix) : -len(self.suffix)])
            except ValueError:
                continue
            found.append((n, os.path.join(self.out_dir, name)))
        # Sort by epoch number, not lexicographically: epoch10 must outrank epoch9.
        for _, path in sorted(found)[: max(0, len(found) - self.max_keep)]:
            try:
                os.remove(path)
            except OSError:
                pass

    # -- loading ----------------------------------------------------------- #

    def load(self, resume: str = "auto", map_location: Any = "cpu") -> Optional[dict]:
        """Find and read the checkpoint implied by ``resume``.

        ``"never"`` disables resuming, ``"auto"`` picks ``_last.pt`` if present,
        anything else is treated as an explicit checkpoint path. Returns ``None``
        when there is nothing to resume from, and raises when the checkpoint
        exists but was produced by a different configuration.
        """
        if not resume or resume == "never":
            return None
        if resume == "auto":
            path = self.last_path
            if not os.path.exists(path):
                if self.verbose:
                    print(f"[resume] no checkpoint at {path}; starting fresh")
                return None
        else:
            path = resume
            if not os.path.exists(path):
                raise FileNotFoundError(f"--resume checkpoint not found: {path}")

        ckpt = torch_load(path, map_location=map_location)
        saved_fp = ckpt.get("fingerprint")
        if self.fingerprint and saved_fp and saved_fp != self.fingerprint:
            raise RuntimeError(
                f"Checkpoint {path} was written by a different configuration "
                f"(fingerprint {saved_fp} != {self.fingerprint}). Re-run with the "
                f"original flags, point --resume at a different checkpoint, or "
                f"pass --resume never to start a fresh run."
            )
        if self.verbose:
            print(
                f"[resume] loaded {path} "
                f"(epoch {ckpt.get('epoch')}, batch {ckpt.get('batch_in_epoch')}, "
                f"step {ckpt.get('global_step')})"
            )
        return ckpt

    def restore(
        self,
        ckpt: Optional[dict],
        model=None,
        optimizer=None,
        scheduler=None,
        *,
        restore_rng: bool = True,
        strict: bool = True,
    ) -> ResumePoint:
        """Load state into the live objects and report where to resume."""
        if ckpt is None:
            return ResumePoint()

        if model is not None and ckpt.get("model_state_dict") is not None:
            model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if restore_rng:
            restore_rng_state(ckpt.get("rng_state"))

        epoch = int(ckpt.get("epoch", 0))
        batch = int(ckpt.get("batch_in_epoch", 0))
        if ckpt.get("epoch_completed"):
            start_epoch, start_batch = epoch + 1, 0
        else:
            # batch_in_epoch is the index of the last *finished* batch.
            start_epoch, start_batch = epoch, batch + 1

        return ResumePoint(
            start_epoch=start_epoch,
            start_batch=start_batch,
            global_step=int(ckpt.get("global_step", 0)),
            best_score=ckpt.get("best_score"),
            extra=dict(ckpt.get("extra") or {}),
            resumed=True,
        )


# --------------------------------------------------------------------------- #
# Deterministic, skippable epoch iteration
# --------------------------------------------------------------------------- #


def epoch_generator(seed: int, epoch: int):
    """A ``torch.Generator`` seeded from ``(seed, epoch)``.

    Reseeding per epoch — rather than letting one generator run across the whole
    training job — is what makes mid-epoch resume sound: epoch *N*'s batch order
    is a pure function of ``(seed, N)``, so a resumed run reproduces the exact
    permutation it was part-way through and can simply skip the batches it
    already consumed.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for epoch_generator")
    g = torch.Generator()
    g.manual_seed((int(seed) * 1_000_003 + int(epoch)) % (2**63 - 1))
    return g


def iter_epoch(loader, skip: int = 0) -> Iterator[tuple[int, Any]]:
    """Yield ``(batch_index, batch)`` from ``loader``, skipping the first ``skip``.

    Skipped batches are still pulled from the loader so that the sampler and any
    dataset-side RNG advance exactly as they did in the original run; only the
    training step is bypassed.
    """
    for i, batch in enumerate(loader):
        if i < skip:
            continue
        yield i, batch


class ResumableLoader:
    """DataLoader factory whose batch order depends only on ``(seed, epoch)``.

    Build one per training script, then call :meth:`epoch` inside the epoch loop
    instead of iterating a single long-lived ``DataLoader``.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        seed: int = 42,
        shuffle: bool = True,
        collate_fn: Optional[Callable] = None,
        **loader_kwargs,
    ):
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch is required for ResumableLoader")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.loader_kwargs = loader_kwargs

    def __len__(self) -> int:
        n = len(self.dataset)
        return (n + self.batch_size - 1) // self.batch_size

    def loader(self, epoch: int):
        from torch.utils.data import DataLoader

        kwargs = dict(self.loader_kwargs)
        if self.shuffle:
            kwargs["generator"] = epoch_generator(self.seed, epoch)
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def epoch(self, epoch: int, skip: int = 0) -> Iterator[tuple[int, Any]]:
        """Iterate epoch ``epoch``, resuming after batch index ``skip - 1``."""
        if skip:
            print(f"  [resume] skipping {skip} already-trained batches", flush=True)
        return iter_epoch(self.loader(epoch), skip=skip)


# --------------------------------------------------------------------------- #
# Record-level resume for corpus scans
# --------------------------------------------------------------------------- #


class ResumableJsonl:
    """Append-only JSONL sink that lets a rerun skip finished records.

    Each record carries an integer index under ``key``. On construction the
    existing file (if any) is replayed to learn which indices are done; call
    :meth:`is_done` to skip them and :meth:`append` to add new ones.

    A ``.meta.json`` sidecar stores the run fingerprint. If it does not match,
    the partial output is not reused — resuming would otherwise interleave rows
    computed under two different configurations.

    Records land on disk as they are produced, and the trailing partial line
    that a hard kill can leave behind is dropped during replay, so the file is
    always valid JSONL.
    """

    def __init__(
        self,
        path: str,
        fingerprint: Optional[str] = None,
        resume: bool = True,
        key: str = "sentence_id",
        flush_every: int = 200,
        verbose: bool = True,
    ):
        self.path = path
        self.key = key
        self.fingerprint = fingerprint
        self.flush_every = max(1, int(flush_every))
        self.verbose = verbose
        self.done: set[int] = set()
        self.records: list[dict] = []
        self._since_flush = 0
        _ensure_parent(path)

        reuse = resume and os.path.exists(path) and self._fingerprint_matches()
        if reuse:
            self.records = self._replay()
            self.done = {int(r[key]) for r in self.records if key in r}
            if verbose and self.done:
                print(f"[resume] {path}: {len(self.done)} records already written")
        elif os.path.exists(path):
            if verbose:
                reason = "fingerprint mismatch" if resume else "--resume never"
                print(f"[resume] {path}: ignoring existing output ({reason})")
            os.remove(path)

        self._fh = open(path, "a", encoding="utf-8")
        self._write_meta()

    # -- meta sidecar ------------------------------------------------------ #

    @property
    def meta_path(self) -> str:
        return f"{self.path}.meta.json"

    def _fingerprint_matches(self) -> bool:
        if self.fingerprint is None:
            return True
        try:
            with open(self.meta_path, encoding="utf-8") as f:
                return json.load(f).get("fingerprint") == self.fingerprint
        except Exception:
            return False

    def _write_meta(self) -> None:
        atomic_write_json(
            {"fingerprint": self.fingerprint, "key": self.key}, self.meta_path
        )

    # -- replay ------------------------------------------------------------ #

    def _replay(self) -> list[dict]:
        """Read back complete rows, discarding a truncated final line."""
        rows: list[dict] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # Only the last line can be torn; anything earlier means the
                    # file is not ours to resume.
                    if self.verbose:
                        print(f"[resume] {self.path}: dropped incomplete trailing row")
                    break
        # Rewrite without the torn line so future appends stay parseable.
        atomic_write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), self.path
        )
        return rows

    # -- writing ----------------------------------------------------------- #

    def is_done(self, index: int) -> bool:
        return int(index) in self.done

    def append(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records.append(record)
        if self.key in record:
            self.done.add(int(record[self.key]))
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._since_flush = 0

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._fh.close()

    def __enter__(self) -> "ResumableJsonl":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def all_records(self, sort: bool = True) -> list[dict]:
        """Every record written across all runs, ordered by ``key``."""
        if sort and all(self.key in r for r in self.records):
            return sorted(self.records, key=lambda r: int(r[self.key]))
        return list(self.records)


# --------------------------------------------------------------------------- #
# Stage-level resume for coarse, expensive loops
# --------------------------------------------------------------------------- #


class StageTracker:
    """Memoize the results of a handful of expensive stages to a JSON sidecar.

    Use for loops whose iterations cost minutes each and number in the tens —
    per-noise-level evaluation, per-checkpoint metric suites, per-threshold
    sweeps. Each completed stage's result is persisted immediately, so a rerun
    replays finished stages from disk instead of recomputing them.
    """

    def __init__(
        self,
        path: str,
        fingerprint: Optional[str] = None,
        resume: bool = True,
        verbose: bool = True,
    ):
        self.path = path
        self.fingerprint = fingerprint
        self.verbose = verbose
        self.stages: dict[str, Any] = {}
        _ensure_parent(path)

        if resume and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    blob = json.load(f)
                if fingerprint is None or blob.get("fingerprint") == fingerprint:
                    self.stages = dict(blob.get("stages") or {})
                    if verbose and self.stages:
                        print(
                            f"[resume] {path}: reusing {len(self.stages)} completed "
                            f"stage(s): {', '.join(sorted(self.stages))}"
                        )
                elif verbose:
                    print(f"[resume] {path}: ignoring stale state (config changed)")
            except Exception as e:  # corrupt sidecar shouldn't sink the run
                if verbose:
                    print(f"[resume] {path}: could not read state ({e}); starting fresh")

    def _persist(self) -> None:
        atomic_write_json(
            {"fingerprint": self.fingerprint, "stages": self.stages}, self.path
        )

    def done(self, key: str) -> bool:
        return key in self.stages

    def get(self, key: str, default: Any = None) -> Any:
        return self.stages.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.stages[key] = value
        self._persist()

    def run(self, key: str, fn: Callable[[], Any]) -> Any:
        """Return the memoized result for ``key``, computing it only if absent."""
        if self.done(key):
            if self.verbose:
                print(f"[resume] stage '{key}' already complete; reusing result")
            return self.get(key)
        value = fn()
        self.set(key, value)
        return value

    def clear(self) -> None:
        self.stages = {}
        self._persist()


# --------------------------------------------------------------------------- #
# Cached expensive artifacts (encoder passes, corpus builds, ...)
# --------------------------------------------------------------------------- #


def cached_torch(
    path: Optional[str],
    build: Callable[[], Any],
    fingerprint: Optional[str] = None,
    resume: bool = True,
    validate: Optional[Callable[[Any], bool]] = None,
    label: str = "artifact",
    verbose: bool = True,
) -> Any:
    """Compute ``build()`` once and reuse it on later runs.

    Encoding a corpus into concept vectors can dominate wall-clock for a short
    training run, and it is pure with respect to the config fingerprint — so a
    resumed run should never pay for it twice. Pass ``path=None`` to disable
    caching entirely.
    """
    if not path:
        return build()

    if resume and os.path.exists(path):
        try:
            blob = torch_load(path)
            if isinstance(blob, dict) and "payload" in blob:
                saved_fp, payload = blob.get("fingerprint"), blob["payload"]
            else:  # legacy caches written before fingerprinting
                saved_fp, payload = None, blob
            if fingerprint and saved_fp and saved_fp != fingerprint:
                if verbose:
                    print(f"[cache] {path}: config changed; recomputing {label}")
            elif validate is not None and not validate(payload):
                if verbose:
                    print(f"[cache] {path}: failed validation; recomputing {label}")
            else:
                if verbose:
                    print(f"[cache] loaded {label} from {path}")
                return payload
        except Exception as e:
            if verbose:
                print(f"[cache] {path}: could not load ({e}); recomputing {label}")

    payload = build()
    try:
        atomic_torch_save({"fingerprint": fingerprint, "payload": payload}, path)
        if verbose:
            print(f"[cache] saved {label} to {path}")
    except Exception as e:
        if verbose:
            print(f"[cache] failed to save {label} to {path}: {e}")
    return payload


# --------------------------------------------------------------------------- #
# argparse wiring
# --------------------------------------------------------------------------- #


def add_resume_args(
    parser,
    *,
    training: bool = True,
    default_interval_steps: int = 200,
    default_max_keep: int = 5,
) -> None:
    """Add the standard ``--resume`` flags to a script's parser.

    ``training`` also adds checkpoint-cadence flags; leave it False for
    evaluation and corpus-analysis scripts, which resume by record or stage and
    have no optimizer state to persist.
    """
    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        metavar="auto|never|PATH",
        help=(
            "Resume behaviour. 'auto' (default) continues from the run's own "
            "checkpoint/partial output if one exists and the configuration "
            "matches; 'never' starts fresh and discards partial state; a path "
            "resumes from that specific checkpoint."
        ),
    )
    if training:
        parser.add_argument(
            "--save_interval_steps",
            type=int,
            default=default_interval_steps,
            help="Write a resumable checkpoint every N optimizer steps (0 disables).",
        )
        parser.add_argument(
            "--save_interval_seconds",
            type=float,
            default=0.0,
            help="Also checkpoint every N seconds of wall-clock (0 disables).",
        )
        parser.add_argument(
            "--max_checkpoints",
            type=int,
            default=default_max_keep,
            help="Number of per-epoch checkpoints to retain (0 keeps all).",
        )
        parser.add_argument(
            "--ckpt_seed",
            type=int,
            default=42,
            help=(
                "Seed for per-epoch batch shuffling. Fixed across a run so that "
                "mid-epoch resume reproduces the exact batch order."
            ),
        )
