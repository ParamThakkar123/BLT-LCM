"""Epoch budgeting: a fixed epoch count, or "keep going while the loss improves".

A fixed ``--epochs`` is a guess. Too low and the run stops while the loss is
still falling; too high and it burns GPU hours on epochs that no longer move it.
``EpochBudget`` lets a script take either shape from the same loop:

    budget = EpochBudget.from_args(args, history=history)
    for epoch in budget.epochs_from(resume.start_epoch):
        avg_loss = ...train one epoch...
        budget.observe(epoch, avg_loss)
    print(budget.stop_reason)

With ``--train_until_plateau`` the loop runs until the monitored loss has failed
to improve by more than ``--min_delta`` for ``--patience`` consecutive epochs,
subject to ``--min_epochs`` and a hard ``--max_epochs`` cap. Without it, it runs
exactly ``--epochs`` epochs, which is what every existing command line does.

Resume-safe: the plateau counters are rebuilt from the run's ``TrainingHistory``,
so a job preempted at epoch 7 and restarted does not get its patience window
reset -- it continues from the same "epochs since the last improvement" the
original process had reached.
"""

from __future__ import annotations

import math
from typing import Any, Iterator, Optional

# Stopping-rule flags. They govern how LONG a run goes, not what any step
# computes, so checkpoint_utils.DEFAULT_FINGERPRINT_IGNORE lists them: switching
# a fixed-epoch run to a plateau run must be able to CONTINUE that run rather
# than invalidate the epochs it already paid for.
EPOCH_CONTROL_ARG_NAMES = (
    "train_until_plateau",
    "patience",
    "min_delta",
    "min_epochs",
    "max_epochs",
)


def add_epoch_control_args(parser, *, default_patience: int = 3) -> None:
    """Add the stop-when-it-stops-improving flags to a training parser."""
    parser.add_argument(
        "--train_until_plateau",
        action="store_true",
        help="Ignore --epochs as a fixed count and keep training while the "
        "loss keeps improving. Stops after --patience epochs with no "
        "improvement greater than --min_delta, or at --max_epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=default_patience,
        help="Consecutive non-improving epochs tolerated before stopping "
        "(--train_until_plateau only).",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.0,
        help="Improvement smaller than this counts as no improvement. Use it "
        "to stop on a loss that is still creeping down but has stopped "
        "meaningfully falling (e.g. 1e-4).",
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=1,
        help="Never stop on plateau before this many epochs have run.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=200,
        help="Hard cap for --train_until_plateau, so a loss that oscillates "
        "forever still terminates.",
    )


class EpochBudget:
    """Decides, before each epoch, whether the run should keep going."""

    def __init__(
        self,
        epochs: int,
        *,
        until_plateau: bool = False,
        patience: int = 3,
        min_delta: float = 0.0,
        min_epochs: int = 1,
        max_epochs: int = 200,
        label: str = "loss",
    ):
        self.until_plateau = bool(until_plateau)
        self.patience = max(int(patience), 1)
        self.min_delta = float(min_delta)
        self.min_epochs = max(int(min_epochs), 1)
        self.label = label
        # In fixed mode --epochs IS the cap. In plateau mode --epochs is only a
        # floor on how long we are willing to run, and --max_epochs is the cap.
        self.cap = max(int(max_epochs), self.min_epochs) if self.until_plateau else int(epochs)
        if self.until_plateau:
            self.min_epochs = max(self.min_epochs, min(int(epochs), self.cap))
        self.best: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.since_improvement = 0
        self.observed = 0
        self.stop_reason = "not started"

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_args(
        cls,
        args: Any,
        *,
        history: Any = None,
        label: str = "loss",
        epochs: Optional[int] = None,
    ) -> "EpochBudget":
        """Build from a parser that used ``add_epoch_control_args``.

        Parsers without those flags fall back to a plain fixed-epoch budget, so
        this is safe to call from a script that has not opted in. Pass
        ``epochs=`` where the script's fixed-count flag is not called
        ``--epochs`` (e.g. ``--train_epochs``).
        """
        budget = cls(
            epochs if epochs is not None else getattr(args, "epochs", 1),
            until_plateau=getattr(args, "train_until_plateau", False),
            patience=getattr(args, "patience", 3),
            min_delta=getattr(args, "min_delta", 0.0),
            min_epochs=getattr(args, "min_epochs", 1),
            max_epochs=getattr(args, "max_epochs", 200),
            label=label,
        )
        if history is not None:
            budget.seed_from_history(history)
        return budget

    def seed_from_history(self, history: Any) -> None:
        """Replay a resumed run's epoch losses so patience is not reset.

        Without this, a job preempted on its last-tolerated epoch would come
        back with a full patience window and train several more epochs that the
        original process had already decided were not worth running.
        """
        records = sorted(
            getattr(history, "epochs", []) or [], key=lambda r: r.get("epoch", 0)
        )
        for rec in records:
            loss = rec.get("val_loss", rec.get("train_loss"))
            if loss is None:
                continue
            self.observe(int(rec["epoch"]) - 1, float(loss), quiet=True)

    # -- the loop ----------------------------------------------------------- #

    def epochs_from(self, start_epoch: int = 0) -> Iterator[int]:
        """Yield 0-based epoch indices until the budget is exhausted."""
        epoch = int(start_epoch)
        while True:
            if epoch >= self.cap:
                self.stop_reason = (
                    f"reached the {'--max_epochs' if self.until_plateau else '--epochs'} "
                    f"cap of {self.cap}"
                )
                return
            if self.until_plateau and self._plateaued(epoch):
                best = f"{self.best:.6g}" if self.best is not None else "n/a"
                self.stop_reason = (
                    f"plateau: no improvement > {self.min_delta:g} in "
                    f"{self.since_improvement} epochs (best {self.label} {best} "
                    f"at epoch {self.best_epoch})"
                )
                return
            yield epoch
            epoch += 1

    def _plateaued(self, epoch: int) -> bool:
        return epoch >= self.min_epochs and self.since_improvement >= self.patience

    def observe(self, epoch: int, loss: float, *, quiet: bool = False) -> None:
        """Record the epoch's monitored loss and update the patience counter."""
        try:
            value = float(loss)
        except (TypeError, ValueError):
            return
        # A NaN/inf epoch is not an improvement, and must not become `best` --
        # that would make every later epoch compare against a broken baseline.
        if not math.isfinite(value):
            self.since_improvement += 1
            self.observed += 1
            return
        self.observed += 1
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.best_epoch = epoch + 1
            self.since_improvement = 0
        else:
            self.since_improvement += 1
            if self.until_plateau and not quiet:
                print(
                    f"  [plateau] epoch {epoch + 1}: {self.label} {value:.6g} did not "
                    f"beat {self.best:.6g} by > {self.min_delta:g} "
                    f"({self.since_improvement}/{self.patience} before stopping)",
                    flush=True,
                )

    # -- display ------------------------------------------------------------ #

    def describe(self, epoch: int) -> str:
        """Human-readable "3/10" or "3/<=200 (until plateau)" for log lines."""
        if not self.until_plateau:
            return f"{epoch + 1}/{self.cap}"
        return f"{epoch + 1}/<={self.cap} (until plateau)"

    def summary(self) -> str:
        best = f"{self.best:.6g}" if self.best is not None else "n/a"
        return (
            f"[epochs] ran {self.observed} epoch(s); best {self.label} {best} "
            f"at epoch {self.best_epoch}; stopped because {self.stop_reason}"
        )

    def as_dict(self) -> dict:
        """Serialisable record of the stopping rule and how it played out."""
        return {
            "mode": "plateau" if self.until_plateau else "fixed",
            "cap": self.cap,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_epochs": self.min_epochs,
            "epochs_observed": self.observed,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "epochs_since_improvement": self.since_improvement,
            "stop_reason": self.stop_reason,
        }
