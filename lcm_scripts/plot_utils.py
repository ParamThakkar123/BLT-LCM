"""Shared plotting helpers: training curves and evaluation figures.

Every training script in this repo records its loss (and, where available, the
learning rate, gradient norm, epoch wall-clock and per-epoch eval metrics) into a
``TrainingHistory``. The history is a plain JSON sidecar next to the run's
checkpoints, so it survives a restart: a resumed run reloads it and the finished
curve covers the whole run rather than only the last attempt.

Nothing here is allowed to take a run down. Rendering is best-effort -- a missing
backend, a headless node without a writable font cache, or a corrupt sidecar
prints a warning and the training loop carries on.

Figures are written at 300 DPI in ``--plot_format`` (png by default, jpg or both
also accepted), so they drop straight into the paper.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Iterable, Mapping, Optional, Sequence

# Names added by `add_plot_args`. They are pure output/IO switches, so
# checkpoint_utils.DEFAULT_FINGERPRINT_IGNORE lists them and toggling a plot
# never invalidates an in-progress run's checkpoints.
PLOT_ARG_NAMES = ("plot_dir", "plot_format", "no_plots")

# Metrics where a *lower* score is better. Plotted with an inverted emphasis so
# a reader does not mistake a tall TER bar for a good result.
LOWER_IS_BETTER = ("TER", "loss", "MSE")

_PALETTE = [
    "#3b82f6",  # blue
    "#22c55e",  # green
    "#f59e0b",  # amber
    "#ef4444",  # red
    "#8b5cf6",  # violet
    "#14b8a6",  # teal
    "#ec4899",  # pink
    "#64748b",  # slate
]

_RC = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

_PLT = None
_IMPORT_FAILED = False


def _plt():
    """Import pyplot on the Agg backend, once, and never raise."""
    global _PLT, _IMPORT_FAILED
    if _PLT is not None or _IMPORT_FAILED:
        return _PLT
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update(_RC)
        _PLT = plt
    except Exception as e:  # pragma: no cover - environment dependent
        _IMPORT_FAILED = True
        print(f"[plot] matplotlib unavailable ({e}); skipping figures")
    return _PLT


def color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #


def add_plot_args(parser, *, default_dir: Optional[str] = None) -> None:
    """Add the standard figure flags to a script's parser."""
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=default_dir,
        help="Directory for training-curve / evaluation figures. Defaults to "
        "the run's output directory.",
    )
    parser.add_argument(
        "--plot_format",
        type=str,
        default="png",
        choices=["png", "jpg", "both"],
        help="Image format for saved figures (default: png).",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip figure generation entirely.",
    )


def plot_formats(args: Any) -> tuple[str, ...]:
    fmt = getattr(args, "plot_format", "png") or "png"
    return ("png", "jpg") if fmt == "both" else (fmt,)


def resolve_plot_dir(args: Any, fallback: str) -> Optional[str]:
    """Where this run's figures go, or None when plotting is switched off."""
    if getattr(args, "no_plots", False):
        return None
    return getattr(args, "plot_dir", None) or fallback


# --------------------------------------------------------------------------- #
# Figure primitives
# --------------------------------------------------------------------------- #


def save_figure(fig, out_path: str, formats: Sequence[str] = ("png",)) -> list[str]:
    """Write ``fig`` once per requested format. Returns the paths written.

    ``out_path`` may carry an image extension; it is replaced by each format's.
    Only *image* extensions are stripped -- run names in this repo routinely
    contain a fraction ("...fraction0.25_training_curve"), and a blind
    ``splitext`` would treat ".25_training_curve" as the extension and collapse
    every figure of a run onto one filename. JPEG has no alpha channel, so the
    face colour is forced opaque white rather than flattened to black by the
    encoder.
    """
    plt = _plt()
    if plt is None:
        return []
    stem, ext = os.path.splitext(out_path)
    if ext.lower() not in (".png", ".jpg", ".jpeg", ".pdf", ".svg"):
        stem = out_path
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    written = []
    for fmt in formats:
        path = f"{stem}.{fmt}"
        try:
            fig.savefig(
                path,
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                format="jpeg" if fmt in ("jpg", "jpeg") else fmt,
            )
            written.append(path)
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[plot] could not write {path}: {e}")
    plt.close(fig)
    for path in written:
        print(f"[plot] wrote {path}")
    return written


def _style_axis(ax, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.3)


def _ema(values: Sequence[float], alpha: float = 0.1) -> list[float]:
    out: list[float] = []
    acc = None
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append(float("nan"))
            continue
        acc = v if acc is None else alpha * v + (1 - alpha) * acc
        out.append(acc)
    return out


def _finite(values: Iterable[Any]) -> list[float]:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


# --------------------------------------------------------------------------- #
# Training history
# --------------------------------------------------------------------------- #


class TrainingHistory:
    """Collects per-step and per-epoch training scalars and plots them.

    The history is persisted as JSON after every epoch, so a run that is killed
    mid-training still leaves a plottable record, and a resumed run reloads what
    the earlier attempt wrote instead of drawing a curve that starts at the
    restart point.
    """

    def __init__(
        self,
        out_dir: Optional[str],
        run_name: str,
        *,
        title: Optional[str] = None,
        fingerprint: Optional[str] = None,
        resume: bool = True,
        formats: Sequence[str] = ("png",),
        loss_label: str = "Loss",
    ):
        self.out_dir = out_dir
        self.run_name = run_name
        self.title = title or run_name
        self.fingerprint = fingerprint
        self.formats = tuple(formats) or ("png",)
        self.loss_label = loss_label
        self.enabled = bool(out_dir)
        self.steps: list[dict] = []
        self.epochs: list[dict] = []
        self.evals: list[dict] = []
        self.json_path = (
            os.path.join(out_dir, f"{run_name}_history.json") if out_dir else None
        )
        if self.enabled and resume:
            self._load()

    # -- persistence -------------------------------------------------------- #

    def _load(self) -> None:
        if not self.json_path or not os.path.exists(self.json_path):
            return
        try:
            with open(self.json_path, encoding="utf-8") as f:
                blob = json.load(f)
        except Exception as e:
            print(f"[plot] ignoring unreadable history {self.json_path}: {e}")
            return
        # A history written under a different configuration describes a
        # different run; keeping it would splice two curves together.
        if self.fingerprint and blob.get("fingerprint") not in (None, self.fingerprint):
            print(f"[plot] history {self.json_path} is from another config; starting fresh")
            return
        self.steps = list(blob.get("steps", []))
        self.epochs = list(blob.get("epochs", []))
        self.evals = list(blob.get("evals", []))
        if self.epochs or self.steps:
            print(
                f"[plot] resumed history: {len(self.epochs)} epochs, "
                f"{len(self.steps)} step samples"
            )

    def save_json(self) -> Optional[str]:
        if not self.enabled or not self.json_path or not self.out_dir:
            return None
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "run_name": self.run_name,
                        "title": self.title,
                        "fingerprint": self.fingerprint,
                        "loss_label": self.loss_label,
                        "steps": self.steps,
                        "epochs": self.epochs,
                        "evals": self.evals,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            return self.json_path
        except Exception as e:
            print(f"[plot] could not write {self.json_path}: {e}")
            return None

    # -- recording ---------------------------------------------------------- #

    @staticmethod
    def _upsert(records: list[dict], key: str, value: Any, payload: dict) -> None:
        for rec in records:
            if rec.get(key) == value:
                rec.update(payload)
                return
        records.append(payload)

    def log_step(
        self,
        step: int,
        loss: float,
        *,
        lr: Optional[float] = None,
        grad_norm: Optional[float] = None,
        **extra: float,
    ) -> None:
        """Record one sampled step. Call it on the same cadence as --log_every.

        Sampling rather than recording every step keeps the sidecar small and
        keeps the gradient-norm reduction (which walks every parameter) off the
        hot path.
        """
        if not self.enabled:
            return
        rec: dict[str, Any] = {"step": int(step), "loss": float(loss)}
        if lr is not None:
            rec["lr"] = float(lr)
        if grad_norm is not None:
            rec["grad_norm"] = float(grad_norm)
        rec.update({k: float(v) for k, v in extra.items() if v is not None})
        self._upsert(self.steps, "step", int(step), rec)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        *,
        val_loss: Optional[float] = None,
        seconds: Optional[float] = None,
        lr: Optional[float] = None,
        peak_vram_gb: Optional[float] = None,
        **extra: float,
    ) -> None:
        """Record one finished epoch. ``epoch`` is 1-based in the figures."""
        if not self.enabled:
            return
        rec: dict[str, Any] = {"epoch": int(epoch), "train_loss": float(train_loss)}
        for name, value in (
            ("val_loss", val_loss),
            ("seconds", seconds),
            ("lr", lr),
            ("peak_vram_gb", peak_vram_gb),
        ):
            if value is not None:
                rec[name] = float(value)
        rec.update({k: float(v) for k, v in extra.items() if v is not None})
        self._upsert(self.epochs, "epoch", int(epoch), rec)
        self.save_json()

    def log_eval(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        """Record an evaluation pass taken at the end of ``epoch``."""
        if not self.enabled:
            return
        clean = {}
        for k, v in metrics.items():
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                clean[k] = f
        if not clean:
            return
        self._upsert(
            self.evals, "epoch", int(epoch), {"epoch": int(epoch), "metrics": clean}
        )
        self.save_json()

    # -- plotting ----------------------------------------------------------- #

    def _sorted(self) -> tuple[list[dict], list[dict], list[dict]]:
        return (
            sorted(self.steps, key=lambda r: r.get("step", 0)),
            sorted(self.epochs, key=lambda r: r.get("epoch", 0)),
            sorted(self.evals, key=lambda r: r.get("epoch", 0)),
        )

    def plot(self) -> list[str]:
        """Render the loss curve and the full training dashboard."""
        if not self.enabled:
            return []
        self.save_json()
        plt = _plt()
        if plt is None:
            return []
        written: list[str] = []
        try:
            written += self._plot_loss_curve()
            written += self._plot_dashboard()
        except Exception as e:  # pragma: no cover - never fail a finished run
            print(f"[plot] figure generation failed: {e}")
        return written

    def _plot_loss_curve(self) -> list[str]:
        """The headline training curve: loss against epoch."""
        plt = _plt()
        steps, epochs, _ = self._sorted()
        if plt is None or not self.out_dir:
            return []
        if not epochs and not steps:
            print("[plot] no training scalars recorded; skipping training curve")
            return []

        fig, ax = plt.subplots(figsize=(7, 4.5))
        if epochs:
            xs = [r["epoch"] for r in epochs]
            ys = [r["train_loss"] for r in epochs]
            ax.plot(
                xs, ys, "o-", color=color(0), linewidth=2.5, markersize=8,
                label="Train", zorder=3,
            )
            val = [(r["epoch"], r["val_loss"]) for r in epochs if "val_loss" in r]
            if val:
                ax.plot(
                    [x for x, _ in val], [y for _, y in val], "s--", color=color(3),
                    linewidth=2, markersize=7, label="Validation", zorder=3,
                )
            best = min(range(len(ys)), key=lambda i: ys[i])
            for i, (x, y) in enumerate(zip(xs, ys)):
                # The best point's value is already in the callout below;
                # annotating it as well just overlaps the arrow.
                if i == best:
                    continue
                ax.annotate(
                    f"{y:.4g}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=color(0),
                )
            ax.annotate(
                f"best epoch {xs[best]}\n{self.loss_label.lower()}={ys[best]:.4g}",
                xy=(xs[best], ys[best]),
                xytext=(12, 24),
                textcoords="offset points",
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.4),
                bbox=dict(
                    boxstyle="round,pad=0.35", facecolor="#fef3c7",
                    edgecolor="#f59e0b", alpha=0.95,
                ),
            )
            ax.set_xlabel("Epoch")
            ax.set_xticks(xs)
            ax.legend(loc="best")
        else:
            # Single-epoch runs still have a curve -- it just lives in the
            # step samples.
            xs = [r["step"] for r in steps]
            ys = [r["loss"] for r in steps]
            ax.plot(xs, ys, color=color(0), alpha=0.3, linewidth=1, label="Step loss")
            ax.plot(
                xs, _ema(ys), color=color(0), linewidth=2.5,
                label="Step loss (EMA)", zorder=3,
            )
            ax.set_xlabel("Training step")
            ax.legend(loc="best")

        ax.set_ylabel(self.loss_label)
        ax.set_title(f"Training Curve — {self.title}")
        _style_axis(ax)
        return save_figure(
            fig,
            os.path.join(self.out_dir, f"{self.run_name}_training_curve"),
            self.formats,
        )

    def _plot_dashboard(self) -> list[str]:
        """Loss / LR / gradient norm / epoch time / eval metrics in one sheet."""
        plt = _plt()
        steps, epochs, evals = self._sorted()
        if plt is None or not self.out_dir:
            return []

        panels: list[tuple[str, Any]] = []
        # A single sample draws an empty axis with no visible mark, which reads
        # as a broken panel rather than a short run; those panels are dropped.
        if len(steps) >= 2:
            panels.append(("step_loss", steps))
        if epochs:
            panels.append(("epoch_loss", epochs))
        lr_source = steps if sum("lr" in r for r in steps) >= 2 else epochs
        if sum("lr" in r for r in lr_source) >= 2:
            panels.append(("lr", lr_source))
        if sum("grad_norm" in r for r in steps) >= 2:
            panels.append(("grad_norm", steps))
        if any("seconds" in r for r in epochs):
            panels.append(("time", epochs))
        if evals:
            panels.append(("eval", evals))
        if len(panels) < 2:
            # A dashboard of one panel adds nothing over the loss curve.
            return []

        ncols = 2
        nrows = math.ceil(len(panels) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.8 * nrows))
        axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for ax, (kind, data) in zip(axes, panels):
            if kind == "step_loss":
                xs = [r["step"] for r in data]
                ys = [r["loss"] for r in data]
                ax.plot(xs, ys, color=color(0), alpha=0.25, linewidth=1)
                ax.plot(xs, _ema(ys), color=color(0), linewidth=2, label="EMA")
                ax.set_xlabel("Training step")
                ax.set_ylabel(self.loss_label)
                ax.set_title("Step loss")
                ax.legend(loc="best")
            elif kind == "epoch_loss":
                xs = [r["epoch"] for r in data]
                ax.plot(
                    xs, [r["train_loss"] for r in data], "o-", color=color(0),
                    linewidth=2.2, markersize=7, label="Train",
                )
                val = [(r["epoch"], r["val_loss"]) for r in data if "val_loss" in r]
                if val:
                    ax.plot(
                        [x for x, _ in val], [y for _, y in val], "s--",
                        color=color(3), linewidth=2, markersize=6, label="Validation",
                    )
                ax.set_xlabel("Epoch")
                ax.set_ylabel(self.loss_label)
                ax.set_title("Epoch loss")
                ax.set_xticks(xs)
                ax.legend(loc="best")
            elif kind == "lr":
                by_step = "step" in data[0]
                pts = [
                    (r["step"] if by_step else r["epoch"], r["lr"])
                    for r in data
                    if "lr" in r
                ]
                ax.plot(
                    [x for x, _ in pts], [y for _, y in pts], color=color(2),
                    linewidth=2,
                )
                ax.set_xlabel("Training step" if by_step else "Epoch")
                ax.set_ylabel("Learning rate")
                ax.set_title("Learning-rate schedule")
                if not by_step:
                    # Whole-numbered epochs; the default locator would tick at
                    # 1.25, 1.5, ... on a short run.
                    ax.set_xticks([x for x, _ in pts])
                if min((y for _, y in pts), default=0) > 0:
                    ax.set_yscale("log")
            elif kind == "grad_norm":
                pts = [(r["step"], r["grad_norm"]) for r in data if "grad_norm" in r]
                ys = [y for _, y in pts]
                ax.plot(
                    [x for x, _ in pts], ys, color=color(4), alpha=0.3, linewidth=1
                )
                ax.plot([x for x, _ in pts], _ema(ys), color=color(4), linewidth=2)
                ax.set_xlabel("Training step")
                ax.set_ylabel("Gradient norm")
                ax.set_title("Gradient norm")
            elif kind == "time":
                xs = [r["epoch"] for r in data if "seconds" in r]
                ys = [r["seconds"] for r in data if "seconds" in r]
                ax.bar(xs, ys, color=color(5), edgecolor="white", width=0.6)
                for x, y in zip(xs, ys):
                    # Sub-second epochs (smoke tests) would all read "0s".
                    label = f"{y:.0f}s" if y >= 10 else f"{y:.2g}s"
                    ax.text(x, y, label, ha="center", va="bottom", fontsize=8)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Seconds")
                ax.set_title("Epoch wall-clock")
                ax.set_xticks(xs)
            elif kind == "eval":
                names: list[str] = []
                for r in data:
                    for k in r["metrics"]:
                        if k not in names:
                            names.append(k)
                for i, name in enumerate(names):
                    pts = [
                        (r["epoch"], r["metrics"][name])
                        for r in data
                        if name in r["metrics"]
                    ]
                    ax.plot(
                        [x for x, _ in pts], [y for _, y in pts], "o-",
                        color=color(i), linewidth=2, markersize=6,
                        label=name + (" ↓" if name in LOWER_IS_BETTER else ""),
                    )
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Score")
                ax.set_title("Evaluation metrics")
                # Epochs are whole numbers; the default locator would put
                # ticks at 1.5, 2.5, ... on a short run.
                ax.set_xticks([r["epoch"] for r in data])
                ax.legend(loc="best", fontsize=9)
            _style_axis(ax)

        for ax in axes[len(panels) :]:
            ax.axis("off")

        fig.suptitle(f"Training Diagnostics — {self.title}", fontsize=15, y=1.0)
        fig.tight_layout()
        return save_figure(
            fig,
            os.path.join(self.out_dir, f"{self.run_name}_training_dashboard"),
            self.formats,
        )


# --------------------------------------------------------------------------- #
# Evaluation figures
# --------------------------------------------------------------------------- #


def plot_metric_bars(
    metrics: Mapping[str, Any],
    out_path: str,
    *,
    title: str,
    subtitle: Optional[str] = None,
    formats: Sequence[str] = ("png",),
) -> list[str]:
    """Bar chart of one evaluation's metrics, NaNs dropped.

    Lower-is-better metrics are hatched and flagged in the label so a tall TER
    bar is not read as a good result.
    """
    plt = _plt()
    if plt is None:
        return []
    names, values = [], []
    for k, v in metrics.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            names.append(k)
            values.append(f)
    if not names:
        print("[plot] no finite metrics to plot")
        return []

    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(names)), 4.5))
    xs = range(len(names))
    bars = ax.bar(
        xs,
        values,
        width=0.55,
        color=[color(i) for i in xs],
        edgecolor="white",
        linewidth=1.2,
        hatch=["//" if n in LOWER_IS_BETTER else "" for n in names],
    )
    span = max(values) - min(0.0, min(values)) or 1.0
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + span * 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        [n + (" ↓" if n in LOWER_IS_BETTER else " ↑") for n in names], rotation=0
    )
    ax.set_ylabel("Score")
    ax.set_ylim(min(0.0, min(values)), max(values) * 1.18)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""))
    _style_axis(ax)
    return save_figure(fig, out_path, formats)


def plot_noise_curves(
    rows: Sequence[Mapping[str, Any]],
    out_path: str,
    *,
    title: str,
    x_key: str = "noise",
    x_label: str = "Input character-noise rate",
    metrics: Sequence[str] = ("BLEU", "chrF++", "TER"),
    series_key: Optional[str] = None,
    formats: Sequence[str] = ("png",),
) -> list[str]:
    """One panel per metric, metric value against the noise level.

    ``series_key`` (e.g. "model" or "seed") draws one line per distinct value,
    which is what turns a benchmark CSV into a robustness comparison.
    """
    plt = _plt()
    if plt is None:
        return []
    present = [
        m for m in metrics if any(_finite([r.get(m)]) for r in rows)
    ]
    if not present or not rows:
        print("[plot] no finite metric rows to plot")
        return []

    fig, axes = plt.subplots(
        1, len(present), figsize=(5.2 * len(present), 4.3), squeeze=False
    )
    axes = list(axes[0])

    if series_key:
        series: dict[str, list[Mapping[str, Any]]] = {}
        for r in rows:
            series.setdefault(str(r.get(series_key, "")), []).append(r)
    else:
        series = {"": list(rows)}

    for ax, metric in zip(axes, present):
        for i, (name, srows) in enumerate(sorted(series.items())):
            pts = sorted(
                (float(r[x_key]), float(r[metric]))
                for r in srows
                if _finite([r.get(x_key)]) and _finite([r.get(metric)])
            )
            if not pts:
                continue
            ax.plot(
                [x for x, _ in pts],
                [y for _, y in pts],
                "o-",
                color=color(i),
                linewidth=2.2,
                markersize=7,
                label=name or metric,
            )
            # Only label points when there is a single line. With several
            # series the labels of near-identical values print on top of each
            # other; the accompanying table carries the exact numbers.
            if len(series) == 1:
                for x, y in pts:
                    ax.annotate(
                        f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=color(i),
                    )
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric + (" (lower is better)" if metric in LOWER_IS_BETTER else ""))
        ax.set_title(metric)
        # Tick only at the levels actually evaluated. The default locator
        # invents intermediate ticks (0.025, 0.075, ...) at which nothing
        # was measured.
        ticks = sorted({float(r[x_key]) for r in rows if _finite([r.get(x_key)])})
        if ticks:
            ax.set_xticks(ticks)
            ax.set_xticklabels(
                [f"{t:.0%}" if max(ticks) <= 1.0 else f"{t:g}" for t in ticks]
            )
        if series_key and len(series) > 1:
            ax.legend(loc="best", fontsize=9)
        _style_axis(ax)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return save_figure(fig, out_path, formats)


def plot_grouped_bars(
    categories: Sequence[str],
    series: Mapping[str, Sequence[float]],
    out_path: str,
    *,
    title: str,
    x_label: str = "",
    y_label: str = "Score",
    formats: Sequence[str] = ("png",),
) -> list[str]:
    """Grouped bars: one group per category, one bar per series."""
    plt = _plt()
    if plt is None or not categories or not series:
        return []
    n = len(series)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(categories)), 4.6))
    for i, (name, values) in enumerate(series.items()):
        offsets = [j - 0.4 + width * (i + 0.5) for j in range(len(categories))]
        vals = [
            float(v) if _finite([v]) else float("nan") for v in values
        ]
        ax.bar(
            offsets, vals, width=width, label=name, color=color(i),
            edgecolor="white", linewidth=0.8,
        )
        for x, v in zip(offsets, vals):
            if math.isfinite(v):
                ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    _style_axis(ax)
    return save_figure(fig, out_path, formats)


def plot_lines(
    x: Sequence[Any],
    series: Mapping[str, Sequence[float]],
    out_path: str,
    *,
    title: str,
    x_label: str,
    y_label: str,
    rotate_xticks: int = 0,
    formats: Sequence[str] = ("png",),
) -> list[str]:
    """Generic multi-series line chart over a shared (possibly textual) x axis."""
    plt = _plt()
    if plt is None or not len(x) or not series:
        return []
    idx = list(range(len(x)))
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(x)), 4.5))
    for i, (name, values) in enumerate(series.items()):
        pts = [
            (j, float(v))
            for j, v in zip(idx, values)
            if _finite([v])
        ]
        if not pts:
            continue
        ax.plot(
            [j for j, _ in pts], [v for _, v in pts], "o-", color=color(i),
            linewidth=2.2, markersize=6, label=name,
        )
    ax.set_xticks(idx)
    ax.set_xticklabels([str(v) for v in x], rotation=rotate_xticks,
                       ha="right" if rotate_xticks else "center")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    _style_axis(ax)
    return save_figure(fig, out_path, formats)


def plot_table(
    rows: Sequence[Sequence[Any]],
    col_labels: Sequence[str],
    out_path: str,
    *,
    title: str,
    highlight_row: Optional[int] = None,
    formats: Sequence[str] = ("png",),
) -> list[str]:
    """Render a results table as an image, for dropping into a paper/slide."""
    plt = _plt()
    if plt is None or not rows:
        return []
    fig, ax = plt.subplots(
        figsize=(max(7, 1.5 * len(col_labels)), 1.1 + 0.42 * len(rows))
    )
    ax.axis("off")
    table = ax.table(
        cellText=[[str(c) for c in row] for row in rows],
        colLabels=list(col_labels),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#1e3a5f")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
    for i in range(len(rows)):
        bg = "#fef3c7" if i == highlight_row else ("#f8fafc" if i % 2 == 0 else "#e2e8f0")
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(bg)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=18)
    return save_figure(fig, out_path, formats)


def plot_histogram(
    values: Sequence[float],
    out_path: str,
    *,
    title: str,
    x_label: str,
    bins: int = 40,
    formats: Sequence[str] = ("png",),
) -> list[str]:
    """Distribution of a per-example quantity, with mean/median marked."""
    plt = _plt()
    vals = _finite(values)
    if plt is None or not vals:
        return []
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.hist(vals, bins=bins, color=color(0), alpha=0.8, edgecolor="white")
    mean = sum(vals) / len(vals)
    median = sorted(vals)[len(vals) // 2]
    ax.axvline(mean, color=color(3), linestyle="--", linewidth=1.8,
               label=f"mean = {mean:.4g}")
    ax.axvline(median, color=color(1), linestyle=":", linewidth=1.8,
               label=f"median = {median:.4g}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    ax.set_title(f"{title}\n(n = {len(vals):,})")
    ax.legend(loc="best", fontsize=9)
    _style_axis(ax)
    return save_figure(fig, out_path, formats)


def plot_curve(
    x: Sequence[float],
    y: Sequence[float],
    out_path: str,
    *,
    title: str,
    x_label: str,
    y_label: str,
    label: Optional[str] = None,
    formats: Sequence[str] = ("png",),
) -> list[str]:
    """Single-series curve (e.g. loss against sentence position)."""
    plt = _plt()
    pts = [(float(a), float(b)) for a, b in zip(x, y) if _finite([a]) and _finite([b])]
    if plt is None or not pts:
        return []
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(
        [a for a, _ in pts], [b for _, b in pts], "o-", color=color(0),
        linewidth=2.2, markersize=6, label=label,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if label:
        ax.legend(loc="best", fontsize=9)
    _style_axis(ax)
    return save_figure(fig, out_path, formats)
