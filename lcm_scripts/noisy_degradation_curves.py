"""Create BPE-LCM vs BLT-LCM noisy-input chrF++ degradation curves.

The script is intentionally evidence-aware: it only plots a comparison when the
requested metric CSVs contain both models at all requested noise levels. This
prevents a placeholder figure from being mistaken for a completed BLT-LCM
robustness result.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Iterable


DEFAULT_MODELS = ("bpe_lcm", "blt_lcm")
DEFAULT_NOISE_LEVELS = (0.0, 0.10, 0.20)
DEFAULT_METRIC = "chrF++"


@dataclass(frozen=True)
class NoiseMetricRow:
    model: str
    noise: float
    metric_value: float
    fraction: float | None = None
    source_csv: str | None = None


def _norm_model(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _get(row: dict[str, str], candidates: Iterable[str]) -> str:
    lowered = {k.lower(): k for k in row}
    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is not None and str(row.get(key, "")).strip():
            return str(row[key]).strip()
    return ""


def read_metric_rows(
    paths: list[str], metric: str = DEFAULT_METRIC
) -> list[NoiseMetricRow]:
    rows: list[NoiseMetricRow] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                model = _get(raw, ("model", "model_name", "system"))
                noise = _get(raw, ("noise", "noise_level", "character_noise"))
                value = _get(
                    raw,
                    (
                        metric,
                        metric.lower(),
                        metric.replace("++", "pp"),
                        "chrf",
                        "chrf++",
                    ),
                )
                if not (model and noise and value):
                    continue
                fraction_text = _get(raw, ("fraction", "data_fraction"))
                rows.append(
                    NoiseMetricRow(
                        model=_norm_model(model),
                        noise=float(noise),
                        metric_value=float(value),
                        fraction=float(fraction_text) if fraction_text else None,
                        source_csv=path,
                    )
                )
    return rows


def select_curve_rows(
    rows: list[NoiseMetricRow],
    models: tuple[str, ...] = DEFAULT_MODELS,
    noise_levels: tuple[float, ...] = DEFAULT_NOISE_LEVELS,
    fraction: float | None = None,
) -> list[NoiseMetricRow]:
    wanted_models = tuple(_norm_model(m) for m in models)
    selected: dict[tuple[str, float], NoiseMetricRow] = {}
    for row in rows:
        if row.model not in wanted_models:
            continue
        if fraction is not None and (
            row.fraction is None or abs(row.fraction - fraction) > 1e-9
        ):
            continue
        for noise in noise_levels:
            if abs(row.noise - noise) < 1e-9:
                key = (row.model, noise)
                if key in selected:
                    print(
                        f"Warning: duplicate {row.model}@{noise} from {selected[key].source_csv} and {row.source_csv}; keeping last"
                    )
                selected[key] = row
                break

    missing = [
        (model, noise)
        for model in wanted_models
        for noise in noise_levels
        if (model, noise) not in selected
    ]
    if missing:
        formatted = ", ".join(f"{model}@{noise:g}" for model, noise in missing)
        raise ValueError(
            "Cannot create noisy-input degradation curve because required "
            f"model/noise metric rows are missing: {formatted}"
        )
    return [
        selected[(model, noise)] for model in wanted_models for noise in noise_levels
    ]


def write_curve_csv(
    rows: list[NoiseMetricRow], out_csv: str, metric: str = DEFAULT_METRIC
) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "noise", metric, "fraction", "source_csv"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row.model,
                    "noise": f"{row.noise:.2f}",
                    metric: f"{row.metric_value:.6g}",
                    "fraction": "" if row.fraction is None else f"{row.fraction:.6g}",
                    "source_csv": row.source_csv or "",
                }
            )


def plot_curve(
    rows: list[NoiseMetricRow], out_png: str, metric: str = DEFAULT_METRIC
) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    by_model: dict[str, list[NoiseMetricRow]] = {}
    for row in rows:
        by_model.setdefault(row.model, []).append(row)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for model, model_rows in by_model.items():
        model_rows = sorted(model_rows, key=lambda r: r.noise)
        ax.plot(
            [r.noise * 100 for r in model_rows],
            [r.metric_value for r in model_rows],
            marker="o",
            label=model,
        )
    ax.set_xlabel("Character noise (%)")
    ax.set_ylabel(metric)
    ax.set_title("Noisy input degradation: BPE-LCM vs BLT-LCM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Metric CSV files with model, noise, and chrF++ columns",
    )
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument(
        "--noise-levels", nargs="+", type=float, default=list(DEFAULT_NOISE_LEVELS)
    )
    p.add_argument("--metric", default=DEFAULT_METRIC)
    p.add_argument("--fraction", type=float, default=None)
    p.add_argument("--out-csv", default="results/noisy_input_degradation_curve.csv")
    p.add_argument("--out-png", default="results/noisy_input_degradation_curve.png")
    from device_utils import report_cpu_only

    args = p.parse_args()
    report_cpu_only("CSV aggregation and matplotlib rendering")

    all_rows = read_metric_rows(args.inputs, metric=args.metric)
    rows = select_curve_rows(
        all_rows, tuple(args.models), tuple(args.noise_levels), args.fraction
    )
    write_curve_csv(rows, args.out_csv, metric=args.metric)
    plot_curve(rows, args.out_png, metric=args.metric)
    print(f"Wrote {args.out_csv} and {args.out_png}")


if __name__ == "__main__":
    main()
