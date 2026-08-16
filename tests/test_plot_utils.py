"""Tests for the shared training-curve / evaluation-figure helpers.

The figures themselves are rasters and are not compared pixel-by-pixel; what is
checked here is everything a run actually depends on -- that the expected files
appear under the expected names, that a resumed run continues its curve rather
than restarting it, and that degenerate inputs are skipped instead of raising
and taking a training run down with them.
"""

import json
import math
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lcm_scripts"))

from plot_utils import (  # noqa: E402
    PLOT_ARG_NAMES,
    TrainingHistory,
    add_plot_args,
    plot_curve,
    plot_formats,
    plot_grouped_bars,
    plot_histogram,
    plot_lines,
    plot_metric_bars,
    plot_noise_curves,
    plot_table,
    resolve_plot_dir,
)
from checkpoint_utils import DEFAULT_FINGERPRINT_IGNORE, config_fingerprint  # noqa: E402


def _filled_history(out_dir, run_name="run", fingerprint="fp", epochs=3, **kwargs):
    h = TrainingHistory(str(out_dir), run_name, fingerprint=fingerprint, **kwargs)
    step = 0
    for epoch in range(1, epochs + 1):
        for _ in range(5):
            step += 10
            h.log_step(step, 1.0 / step, lr=1e-4, grad_norm=0.5)
        h.log_epoch(epoch, 1.0 / epoch, seconds=12.0, lr=1e-4)
    return h


# --------------------------------------------------------------------------- #
# Filenames
# --------------------------------------------------------------------------- #


def test_run_name_containing_a_fraction_does_not_collapse_filenames(tmp_path):
    """Run names like "lcm_bpe_fraction0.25" carry a dot.

    A blind ``os.path.splitext`` reads ".25_training_curve" as the extension, so
    every figure of the run lands on "lcm_bpe_fraction0.png" and overwrites the
    previous one.
    """
    h = _filled_history(tmp_path, run_name="lcm_bpe_fraction0.25")
    h.plot()
    names = set(os.listdir(tmp_path))
    assert "lcm_bpe_fraction0.25_training_curve.png" in names
    assert "lcm_bpe_fraction0.25_training_dashboard.png" in names
    assert not any(n.startswith("lcm_bpe_fraction0.png") for n in names)


def test_requested_formats_are_all_written(tmp_path):
    h = _filled_history(tmp_path, formats=("png", "jpg"))
    written = h.plot()
    assert any(p.endswith(".png") for p in written)
    assert any(p.endswith(".jpg") for p in written)


def test_plot_formats_maps_both_to_two_formats():
    class A:
        plot_format = "both"

    assert plot_formats(A()) == ("png", "jpg")

    class B:
        plot_format = "jpg"

    assert plot_formats(B()) == ("jpg",)

    assert plot_formats(object()) == ("png",)


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #


def test_resumed_history_continues_the_curve(tmp_path):
    """A restart must extend the earlier run's curve, not start a new one."""
    _filled_history(tmp_path, epochs=2)
    resumed = TrainingHistory(str(tmp_path), "run", fingerprint="fp")
    assert [e["epoch"] for e in resumed.epochs] == [1, 2]
    resumed.log_epoch(3, 0.33)
    assert [e["epoch"] for e in resumed.epochs] == [1, 2, 3]


def test_history_from_another_config_is_not_spliced_in(tmp_path):
    _filled_history(tmp_path, epochs=2, fingerprint="fp-a")
    other = TrainingHistory(str(tmp_path), "run", fingerprint="fp-b")
    assert other.epochs == []


def test_resume_disabled_starts_fresh(tmp_path):
    _filled_history(tmp_path, epochs=2)
    fresh = TrainingHistory(str(tmp_path), "run", fingerprint="fp", resume=False)
    assert fresh.epochs == []


def test_repeated_epoch_is_updated_not_duplicated(tmp_path):
    h = TrainingHistory(str(tmp_path), "run")
    h.log_epoch(1, 5.0)
    h.log_epoch(1, 4.0)
    assert len(h.epochs) == 1
    assert h.epochs[0]["train_loss"] == 4.0


def test_unreadable_history_is_ignored_rather_than_raising(tmp_path):
    (tmp_path / "run_history.json").write_text("{not json", encoding="utf-8")
    h = TrainingHistory(str(tmp_path), "run", fingerprint="fp")
    assert h.epochs == []


def test_history_json_records_the_scalars(tmp_path):
    _filled_history(tmp_path, epochs=2)
    blob = json.loads((tmp_path / "run_history.json").read_text(encoding="utf-8"))
    assert [e["epoch"] for e in blob["epochs"]] == [1, 2]
    assert blob["epochs"][0]["seconds"] == 12.0
    assert blob["steps"] and "grad_norm" in blob["steps"][0]


# --------------------------------------------------------------------------- #
# Degenerate input must never take a run down
# --------------------------------------------------------------------------- #


def test_plotting_disabled_is_a_no_op(tmp_path):
    h = TrainingHistory(None, "run")
    h.log_step(1, 1.0)
    h.log_epoch(1, 1.0)
    h.log_eval(1, {"BLEU": 1.0})
    assert h.plot() == []
    assert not h.enabled
    assert os.listdir(tmp_path) == []


def test_history_with_no_scalars_emits_no_figure(tmp_path):
    assert TrainingHistory(str(tmp_path), "run").plot() == []
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".png")]


def test_nan_metrics_are_dropped(tmp_path):
    h = TrainingHistory(str(tmp_path), "run")
    h.log_eval(1, {"BLEU": float("nan"), "COMET": None, "chrF++": 40.0})
    assert h.evals[0]["metrics"] == {"chrF++": 40.0}
    # An evaluation with nothing finite in it is not recorded at all.
    h.log_eval(2, {"BLEU": float("nan")})
    assert [e["epoch"] for e in h.evals] == [1]


@pytest.mark.parametrize(
    "call",
    [
        lambda p: plot_metric_bars({"BLEU": float("nan")}, p, title="t"),
        lambda p: plot_metric_bars({}, p, title="t"),
        lambda p: plot_noise_curves([], p, title="t"),
        lambda p: plot_noise_curves(
            [{"noise": 0.0, "BLEU": None}], p, title="t"
        ),
        lambda p: plot_grouped_bars([], {}, p, title="t"),
        lambda p: plot_lines([], {}, p, title="t", x_label="x", y_label="y"),
        lambda p: plot_table([], ["a"], p, title="t"),
        lambda p: plot_histogram([], p, title="t", x_label="x"),
        lambda p: plot_histogram(
            [float("nan"), None], p, title="t", x_label="x"
        ),
        lambda p: plot_curve([], [], p, title="t", x_label="x", y_label="y"),
    ],
)
def test_empty_inputs_produce_no_file_and_no_exception(tmp_path, call):
    assert call(str(tmp_path / "fig")) == []


def test_single_epoch_run_still_gets_a_curve(tmp_path):
    """A one-epoch run has no epoch-to-epoch trend; the step samples carry it."""
    h = TrainingHistory(str(tmp_path), "run")
    for step in range(1, 40):
        h.log_step(step, 1.0 / step)
    assert h.plot()
    assert (tmp_path / "run_training_curve.png").exists()


# --------------------------------------------------------------------------- #
# Evaluation figures
# --------------------------------------------------------------------------- #


def test_evaluation_figures_are_written(tmp_path):
    rows = [
        {"model": m, "noise": n, "BLEU": 10 - 20 * n, "chrF++": 40 - 50 * n,
         "TER": 70 + 40 * n}
        for m in ("blt_lcm", "bpe_lcm")
        for n in (0.0, 0.1, 0.2)
    ]
    assert plot_noise_curves(
        rows, str(tmp_path / "noise"), title="t", series_key="model"
    )
    assert plot_metric_bars(
        {"BLEU": 10.0, "TER": 70.0}, str(tmp_path / "bars"), title="t"
    )
    assert plot_grouped_bars(
        ["25%", "50%"], {"a": [1.0, 2.0], "b": [2.0, float("nan")]},
        str(tmp_path / "grouped"), title="t",
    )
    assert plot_lines(
        ["e1", "e2"], {"BLEU": [1.0, 2.0]}, str(tmp_path / "lines"),
        title="t", x_label="x", y_label="y",
    )
    assert plot_table(
        [["0%", "1.0"]], ["Noise", "BLEU"], str(tmp_path / "table"),
        title="t", highlight_row=0,
    )
    assert plot_histogram(
        [float(i % 7) for i in range(50)], str(tmp_path / "hist"),
        title="t", x_label="x",
    )
    assert plot_curve(
        [1, 2, 3], [0.4, 0.5, 0.6], str(tmp_path / "curve"),
        title="t", x_label="x", y_label="y",
    )


def test_noise_curves_skip_metrics_with_no_finite_values(tmp_path):
    rows = [{"noise": n, "BLEU": 10.0, "COMET": float("nan")} for n in (0.0, 0.1)]
    written = plot_noise_curves(
        rows, str(tmp_path / "fig"), title="t", metrics=("BLEU", "COMET")
    )
    assert written  # BLEU alone is enough to draw


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #


def test_plot_flags_do_not_change_a_run_fingerprint():
    """Turning plotting on must not invalidate an in-progress run's checkpoints."""
    import argparse

    base = {"epochs": 2, "batch_size": 8, "lr": 1e-4}
    without = config_fingerprint(argparse.Namespace(**base))
    with_defaults = config_fingerprint(
        argparse.Namespace(**base, plot_dir=None, plot_format="png", no_plots=False)
    )
    with_values = config_fingerprint(
        argparse.Namespace(**base, plot_dir="figs", plot_format="jpg", no_plots=True)
    )
    assert without == with_defaults == with_values


def test_every_plot_arg_is_fingerprint_ignored():
    assert set(PLOT_ARG_NAMES) <= set(DEFAULT_FINGERPRINT_IGNORE)


def test_add_plot_args_and_resolve_plot_dir():
    import argparse

    p = argparse.ArgumentParser()
    add_plot_args(p)

    args = p.parse_args([])
    assert resolve_plot_dir(args, "runs/x") == "runs/x"

    args = p.parse_args(["--plot_dir", "figs"])
    assert resolve_plot_dir(args, "runs/x") == "figs"

    args = p.parse_args(["--no_plots"])
    assert resolve_plot_dir(args, "runs/x") is None


def test_history_is_disabled_when_out_dir_is_none():
    assert not TrainingHistory(None, "run").enabled
    assert TrainingHistory("somewhere", "run", resume=False).enabled


# --------------------------------------------------------------------------- #
# Dashboard panel selection
# --------------------------------------------------------------------------- #


def test_dashboard_is_skipped_when_there_is_only_one_panel(tmp_path):
    """One panel adds nothing over the standalone loss curve."""
    h = TrainingHistory(str(tmp_path), "run")
    for epoch in range(1, 4):
        h.log_epoch(epoch, 1.0 / epoch)  # epoch loss only, no lr/steps/time
    h.plot()
    assert (tmp_path / "run_training_curve.png").exists()
    assert not (tmp_path / "run_training_dashboard.png").exists()


def test_dashboard_includes_eval_panel(tmp_path):
    h = _filled_history(tmp_path)
    h.log_eval(1, {"BLEU": 10.0})
    h.log_eval(2, {"BLEU": 12.0})
    written = h.plot()
    assert any("dashboard" in p for p in written)


def test_epoch_perplexity_extras_are_recorded(tmp_path):
    """Scripts pass derived scalars (perplexity, bits/byte, VRAM) as extras."""
    h = TrainingHistory(str(tmp_path), "run")
    h.log_epoch(1, 4.0, peak_vram_gb=7.5, bits_per_byte=4.0 / math.log(2))
    rec = h.epochs[0]
    assert rec["peak_vram_gb"] == 7.5
    assert rec["bits_per_byte"] == pytest.approx(5.7708, abs=1e-3)
