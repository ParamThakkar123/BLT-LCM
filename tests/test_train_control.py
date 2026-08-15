"""Tests for the epoch budget: fixed count vs. run-until-the-loss-plateaus."""

import argparse
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lcm_scripts"))

from train_control import (  # noqa: E402
    EPOCH_CONTROL_ARG_NAMES,
    EpochBudget,
    add_epoch_control_args,
)
from checkpoint_utils import DEFAULT_FINGERPRINT_IGNORE, config_fingerprint  # noqa: E402
from plot_utils import TrainingHistory  # noqa: E402


def run(budget, losses, start=0):
    """Feed `losses` to the budget and return the epochs it actually ran."""
    ran = []
    it = iter(losses)
    for epoch in budget.epochs_from(start):
        try:
            loss = next(it)
        except StopIteration:
            break
        ran.append(epoch)
        budget.observe(epoch, loss)
    return ran


# --------------------------------------------------------------------------- #
# Fixed mode -- the existing behaviour of every command line in the repo
# --------------------------------------------------------------------------- #


def test_fixed_mode_runs_exactly_epochs():
    budget = EpochBudget(4)
    assert run(budget, [1.0] * 10) == [0, 1, 2, 3]
    assert "cap of 4" in budget.stop_reason


def test_fixed_mode_ignores_a_plateau():
    """Without the flag, a flat loss must not cut the run short."""
    budget = EpochBudget(5, patience=1)
    assert run(budget, [1.0] * 5) == [0, 1, 2, 3, 4]


def test_fixed_mode_resumes_from_start_epoch():
    assert run(EpochBudget(5), [1.0] * 10, start=3) == [3, 4]


# --------------------------------------------------------------------------- #
# Plateau mode
# --------------------------------------------------------------------------- #


def test_plateau_stops_after_patience_non_improving_epochs():
    budget = EpochBudget(1, until_plateau=True, patience=2, max_epochs=50)
    # Improves for three epochs, then flat.
    ran = run(budget, [1.0, 0.5, 0.25] + [0.25] * 10)
    assert ran == [0, 1, 2, 3, 4]  # two flat epochs tolerated, then stop
    assert "plateau" in budget.stop_reason
    assert budget.best == 0.25
    assert budget.best_epoch == 3


def test_plateau_keeps_going_while_the_loss_falls():
    budget = EpochBudget(1, until_plateau=True, patience=2, max_epochs=20)
    ran = run(budget, [1.0 / (i + 1) for i in range(20)])
    assert len(ran) == 20
    assert "max_epochs" in budget.stop_reason


def test_min_delta_treats_tiny_gains_as_no_improvement():
    budget = EpochBudget(
        1, until_plateau=True, patience=2, min_delta=0.01, max_epochs=50
    )
    # Falls by 0.001 an epoch -- real but below --min_delta.
    ran = run(budget, [1.0 - 0.001 * i for i in range(20)])
    assert len(ran) == 3, ran


def test_max_epochs_caps_an_oscillating_loss():
    budget = EpochBudget(1, until_plateau=True, patience=100, max_epochs=6)
    assert len(run(budget, [1.0, 2.0] * 20)) == 6
    assert "max_epochs" in budget.stop_reason


def test_min_epochs_prevents_an_early_stop():
    budget = EpochBudget(
        1, until_plateau=True, patience=1, min_epochs=5, max_epochs=50
    )
    assert len(run(budget, [1.0] * 20)) == 5


def test_epochs_becomes_a_floor_in_plateau_mode():
    """--epochs N still means "at least N" once --train_until_plateau is on."""
    budget = EpochBudget(6, until_plateau=True, patience=1, max_epochs=50)
    assert len(run(budget, [1.0] * 20)) == 6


def test_nan_epoch_never_becomes_the_best():
    budget = EpochBudget(1, until_plateau=True, patience=5, max_epochs=10)
    budget.observe(0, 1.0)
    budget.observe(1, float("nan"))
    budget.observe(2, float("inf"))
    assert budget.best == 1.0
    assert budget.since_improvement == 2


def test_non_numeric_loss_is_ignored():
    budget = EpochBudget(3)
    budget.observe(0, "not a number")  # type: ignore[arg-type]
    assert budget.observed == 0
    assert budget.best is None


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #


def test_seed_from_history_preserves_the_patience_window(tmp_path):
    """A preempted job must not come back with a fresh patience window."""
    history = TrainingHistory(str(tmp_path), "run")
    for epoch, loss in enumerate([1.0, 0.5, 0.5, 0.5], start=1):
        history.log_epoch(epoch, loss)

    budget = EpochBudget(1, until_plateau=True, patience=3, max_epochs=50)
    budget.seed_from_history(history)
    assert budget.best == 0.5
    assert budget.since_improvement == 2

    # One more non-improving epoch is all it should tolerate.
    assert run(budget, [0.5] * 10, start=4) == [4]


def test_seed_from_history_prefers_validation_loss(tmp_path):
    history = TrainingHistory(str(tmp_path), "run")
    history.log_epoch(1, 1.0, val_loss=2.0)
    history.log_epoch(2, 0.1, val_loss=2.5)  # train improved, val got worse
    budget = EpochBudget(1, until_plateau=True, patience=2, max_epochs=50)
    budget.seed_from_history(history)
    assert budget.best == 2.0
    assert budget.since_improvement == 1


def test_seed_from_empty_history_is_a_no_op(tmp_path):
    budget = EpochBudget(1, until_plateau=True, max_epochs=5)
    budget.seed_from_history(TrainingHistory(str(tmp_path), "run"))
    assert budget.best is None and budget.since_improvement == 0


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #


def _parser():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    add_epoch_control_args(p)
    return p


def test_from_args_defaults_to_fixed_mode():
    budget = EpochBudget.from_args(_parser().parse_args([]))
    assert not budget.until_plateau
    assert budget.cap == 3


def test_from_args_reads_the_plateau_flags():
    args = _parser().parse_args(
        ["--train_until_plateau", "--patience", "7", "--max_epochs", "99",
         "--min_delta", "0.5"]
    )
    budget = EpochBudget.from_args(args)
    assert budget.until_plateau and budget.patience == 7
    assert budget.cap == 99 and budget.min_delta == 0.5


def test_from_args_epochs_override_for_scripts_using_a_different_flag():
    """run_blt_patching.py calls its fixed-count flag --train_epochs."""
    budget = EpochBudget.from_args(argparse.Namespace(), epochs=8)
    assert budget.cap == 8


def test_from_args_on_a_parser_without_the_flags():
    budget = EpochBudget.from_args(argparse.Namespace(epochs=2))
    assert budget.cap == 2 and not budget.until_plateau


def test_stopping_rule_does_not_change_a_run_fingerprint():
    """Switching to --train_until_plateau must CONTINUE a run, not discard it."""
    base = {"epochs": 2, "batch_size": 8, "lr": 1e-4}
    without = config_fingerprint(argparse.Namespace(**base))
    with_flags = config_fingerprint(
        argparse.Namespace(
            **base, train_until_plateau=True, patience=5, min_delta=1e-4,
            min_epochs=2, max_epochs=100,
        )
    )
    assert without == with_flags


def test_every_epoch_control_arg_is_fingerprint_ignored():
    assert set(EPOCH_CONTROL_ARG_NAMES) <= set(DEFAULT_FINGERPRINT_IGNORE)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def test_describe_and_summary():
    fixed = EpochBudget(10)
    assert fixed.describe(2) == "3/10"
    plateau = EpochBudget(1, until_plateau=True, max_epochs=50)
    assert "until plateau" in plateau.describe(2)

    run(plateau, [1.0, 0.9, 0.9, 0.9, 0.9])
    assert "best" in plateau.summary()
    d = plateau.as_dict()
    assert d["mode"] == "plateau" and d["best"] == pytest.approx(0.9)
    assert d["epochs_observed"] == 5
