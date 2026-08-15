"""Statistical significance for MT metric comparisons.

A table of single BLEU numbers cannot answer the question a reviewer will ask:
is the gap between two systems larger than the noise? Two independent sources of
noise matter here, and they need different treatment.

**Test-set noise** -- would the gap survive a different sample of sentences?
Answered by paired bootstrap resampling (Koehn, 2004) or approximate
randomization: resample the *same* segments for both systems, so the pairing is
preserved, and count how often the gap reverses. That is what
``paired_test`` does, delegating to sacrebleu's implementation so the numbers
match what everyone else reports.

**Training noise** -- would the gap survive a different random seed? No amount of
bootstrapping on one output file can answer that; it needs several training
runs. ``seed_summary`` aggregates the per-seed metric CSVs this repo already
produces into mean, standard deviation and a t-based confidence interval, which
is what belongs in the paper's table as ``x.y ± z``.

Reporting both is the difference between "our system scores 1.2 BLEU higher" and
a claim a reviewer can check.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional, Sequence

try:
    import sacrebleu
    from sacrebleu.significance import PairedTest

    _HAS_SACREBLEU = True
except Exception:  # pragma: no cover - dependency checked at runtime
    sacrebleu = None
    PairedTest = None
    _HAS_SACREBLEU = False


DEFAULT_METRICS = ("BLEU", "chrF++", "TER")
DEFAULT_N_SAMPLES = 1000
SIGNIFICANCE_LEVEL = 0.05


# Score differences below this are numerical noise, not a difference between
# systems. chrF/BLEU are reported on a 0-100 scale, so this is far below any
# gap anybody would report.
NEGLIGIBLE_DELTA = 1e-6


@dataclass
class SystemComparison:
    """One system scored against the baseline, with a p-value for the gap."""

    system: str
    metric: str
    baseline_score: float
    system_score: float
    p_value: Optional[float]
    n_segments: int
    test_type: str
    note: str = ""

    @property
    def delta(self) -> float:
        return self.system_score - self.baseline_score

    @property
    def significant(self) -> bool:
        # A zero-sized difference is never significant, whatever the test
        # says. sacrebleu's paired bootstrap returns its floor p-value
        # (1/(n+1), e.g. 0.001 at 1000 resamples) when the observed delta is
        # zero, because no resample produces a *larger* delta than zero --
        # the degenerate case reads as "p < 0.05, significant" for two
        # byte-identical systems. Reporting that in a paper would be a real
        # error, so the delta is checked before the p-value.
        if abs(self.delta) < NEGLIGIBLE_DELTA:
            return False
        return self.p_value is not None and self.p_value < SIGNIFICANCE_LEVEL

    def marker(self) -> str:
        """The conventional table annotation: ``*`` for p < 0.05."""
        if self.p_value is None:
            return ""
        return "*" if self.p_value < SIGNIFICANCE_LEVEL else ""

    def format(self) -> str:
        p = "n/a" if self.p_value is None else f"{self.p_value:.4f}"
        note = f"  [{self.note}]" if self.note else ""
        return (
            f"{self.system} {self.metric}: {self.system_score:.2f} vs "
            f"{self.baseline_score:.2f} (Δ{self.delta:+.2f}, p={p})"
            f"{' significant' if self.significant else ''}{note}"
        )

    def as_dict(self) -> dict:
        return {
            "system": self.system,
            "metric": self.metric,
            "baseline_score": self.baseline_score,
            "system_score": self.system_score,
            "delta": self.delta,
            "p_value": self.p_value,
            "significant": self.significant,
            "n_segments": self.n_segments,
            "test_type": self.test_type,
            "note": self.note,
        }


@dataclass
class SeedSummary:
    """Mean ± std over training seeds, with a confidence interval."""

    metric: str
    values: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values) if self.values else float("nan")

    @property
    def std(self) -> float:
        # Sample standard deviation: these seeds are a sample of the runs that
        # could have been made, not the population of them.
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Two-sided t interval for the mean.

        With the 3 seeds this repo runs, the normal approximation is far too
        narrow -- t(2) at 95% is 4.303, not 1.96 -- so a t multiplier is used
        and small-n intervals come out honestly wide.
        """
        if self.n < 2:
            return (self.mean, self.mean)
        t = _t_critical(self.n - 1, level)
        half = t * self.std / math.sqrt(self.n)
        return (self.mean - half, self.mean + half)

    def format(self, level: float = 0.95) -> str:
        if self.n == 0:
            return f"{self.metric}: n/a"
        if self.n == 1:
            return f"{self.metric}: {self.mean:.2f} (1 seed, no error bar)"
        lo, hi = self.confidence_interval(level)
        return (
            f"{self.metric}: {self.mean:.2f} ± {self.std:.2f} "
            f"({self.n} seeds, {level:.0%} CI [{lo:.2f}, {hi:.2f}])"
        )

    def as_dict(self, level: float = 0.95) -> dict:
        lo, hi = self.confidence_interval(level)
        return {
            "metric": self.metric,
            "n_seeds": self.n,
            "mean": self.mean,
            "std": self.std,
            "ci_low": lo,
            "ci_high": hi,
            "values": list(self.values),
        }


# Two-sided t critical values at 95% / 99%, by degrees of freedom. Small n is
# exactly where the normal approximation misleads, and scipy is not a
# dependency of this project.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042,
    60: 2.000,
}
_T99 = {
    1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032, 6: 3.707, 7: 3.499,
    8: 3.355, 9: 3.250, 10: 3.169, 12: 3.055, 15: 2.947, 20: 2.845, 30: 2.750,
    60: 2.660,
}


def _t_critical(df: int, level: float = 0.95) -> float:
    table = _T99 if level >= 0.99 else _T95
    for k in sorted(table):
        if df <= k:
            return table[k]
    return 2.576 if level >= 0.99 else 1.96


# --------------------------------------------------------------------------- #
# Paired significance testing
# --------------------------------------------------------------------------- #


def _metric_objects(metrics: Sequence[str]):
    """Map our metric names onto sacrebleu metric objects."""
    if not _HAS_SACREBLEU:
        return {}
    out = {}
    for name in metrics:
        key = name.strip()
        if key.upper() == "BLEU":
            out["BLEU"] = sacrebleu.metrics.BLEU()
        elif key.lower().startswith("chrf"):
            # chrF++ is chrF with word bigrams, matching eval_metrics.py.
            out["chrF++"] = sacrebleu.metrics.CHRF(word_order=2)
        elif key.upper() == "TER":
            out["TER"] = sacrebleu.metrics.TER()
    return out


def paired_test(
    baseline: Sequence[str],
    systems: dict[str, Sequence[str]],
    references: Sequence[str],
    metrics: Sequence[str] = DEFAULT_METRICS,
    test_type: str = "bs",
    n_samples: int = DEFAULT_N_SAMPLES,
) -> list[SystemComparison]:
    """Paired significance test of each system against ``baseline``.

    ``test_type`` is ``"bs"`` for paired bootstrap resampling (Koehn 2004, the
    convention in MT papers) or ``"ar"`` for approximate randomization
    (sacrebleu's default, less optimistic on small sets).

    The pairing is the point: every resample draws the *same* segment indices
    for both systems, so segment difficulty cancels out and what remains is the
    difference between the systems. An unpaired test on the same data would be
    far less sensitive.
    """
    if not _HAS_SACREBLEU or PairedTest is None:
        print("[significance] sacrebleu unavailable; skipping significance tests")
        return []
    if not systems:
        return []

    lengths = {len(baseline), len(references)} | {len(v) for v in systems.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"paired test needs equal-length outputs; got lengths {sorted(lengths)}"
        )

    metric_objs = _metric_objects(metrics)
    if not metric_objs:
        return []

    # An output file identical to the baseline's is almost always a mistake --
    # the wrong file passed to --compare, or a system that silently copied the
    # baseline. It is called out rather than run through a test that cannot say
    # anything useful about it.
    identical = {
        name for name, out in systems.items() if list(out) == list(baseline)
    }
    for name in identical:
        print(
            f"[significance] '{name}' produced output identical to the "
            "baseline; a paired test cannot distinguish them"
        )

    named = [("baseline", list(baseline))] + [
        (name, list(out)) for name, out in systems.items()
    ]
    try:
        test = PairedTest(
            named,
            metric_objs,
            references=[list(references)],
            test_type=test_type,
            n_samples=n_samples,
        )
        _signatures, scores = test()
    except Exception as e:
        print(f"[significance] paired test failed: {e}")
        return []

    # sacrebleu returns {metric: [Result per system]}, baseline first, and each
    # non-baseline Result carries the p-value of its gap to the baseline.
    out: list[SystemComparison] = []
    system_names = [n for n, _ in named]
    for metric_name, results in scores.items():
        if metric_name == "System":
            continue
        base_score = _result_score(results[0])
        for name, res in zip(system_names[1:], results[1:]):
            score = _result_score(res)
            p = _result_pvalue(res)
            note = ""
            if name in identical:
                note = "identical output to the baseline"
            elif abs(score - base_score) < NEGLIGIBLE_DELTA:
                # The bootstrap floor p-value would otherwise read as
                # "significant" for a zero-sized gap; see
                # SystemComparison.significant.
                note = "no measurable difference; p-value not meaningful"
            out.append(
                SystemComparison(
                    system=name,
                    metric=_canonical_metric(metric_name),
                    baseline_score=base_score,
                    system_score=score,
                    p_value=p,
                    n_segments=len(references),
                    test_type=test_type,
                    note=note,
                )
            )
    return out


def _canonical_metric(name: str) -> str:
    """sacrebleu labels chrF++ by its signature ("chrF2++"); our CSVs say chrF++.

    Without this the comparison rows cannot be joined against the metric tables
    every other script in the repo writes.
    """
    if name.lower().startswith("chrf"):
        return "chrF++" if name.endswith("++") else "chrF"
    return name


def _result_score(res) -> float:
    for attr in ("score",):
        val = getattr(res, attr, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    # sacrebleu >= 2.1 returns a formatted string like "42.1 (p = 0.0010)".
    try:
        return float(str(res).split()[0])
    except Exception:
        return float("nan")


def _result_pvalue(res) -> Optional[float]:
    val = getattr(res, "p_value", None)
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None
    text = str(res)
    if "p =" in text:
        try:
            return float(text.split("p =")[1].split(")")[0].strip())
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# Seed aggregation
# --------------------------------------------------------------------------- #


def seed_summary(
    rows: Sequence[dict],
    metrics: Sequence[str] = DEFAULT_METRICS,
    group_keys: Sequence[str] = ("model", "fraction", "noise"),
) -> dict[tuple, dict[str, SeedSummary]]:
    """Aggregate per-seed metric rows into mean/std/CI per condition.

    ``rows`` are the metric CSV records this repo already writes (one per
    ``model x fraction x noise x seed``). Returns ``{condition: {metric: summary}}``
    keyed by the ``group_keys`` values, so a table can be built directly.
    """
    grouped: dict[tuple, dict[str, SeedSummary]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        bucket = grouped.setdefault(key, {})
        for metric in metrics:
            raw = row.get(metric)
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(val):
                bucket.setdefault(metric, SeedSummary(metric)).values.append(val)
    return grouped


def format_seed_table(
    grouped: dict[tuple, dict[str, SeedSummary]],
    group_keys: Sequence[str] = ("model", "fraction", "noise"),
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> list[list[str]]:
    """Rows of ``[*condition, "mean ± std", ...]`` ready for a table figure."""
    table = []
    for key in sorted(grouped, key=lambda k: tuple(str(x) for x in k)):
        cells = [str(v) for v in key]
        for metric in metrics:
            s = grouped[key].get(metric)
            if s is None or s.n == 0:
                cells.append("—")
            elif s.n == 1:
                cells.append(f"{s.mean:.2f}")
            else:
                cells.append(f"{s.mean:.2f} ± {s.std:.2f}")
        table.append(cells)
    return table


def add_significance_args(parser) -> None:
    """Add paired-test flags to a script's parser."""
    parser.add_argument(
        "--significance_test",
        choices=["bs", "ar", "none"],
        default="bs",
        help="Paired significance test against the baseline system: 'bs' = "
        "paired bootstrap resampling (Koehn 2004), 'ar' = approximate "
        "randomization, 'none' to skip.",
    )
    parser.add_argument(
        "--significance_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Resamples for the paired test. 1000 is the usual reporting "
        "standard; more is slower and changes p-values only marginally.",
    )
