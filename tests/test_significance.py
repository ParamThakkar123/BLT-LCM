"""Tests for paired significance testing and seed aggregation.

The property that matters is discrimination: a real quality gap must come out
significant, and a non-gap must not. A test that calls everything significant
is worse than no test, because it launders noise into a paper claim.
"""

import argparse
import os
import random
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lcm_scripts"))

from significance import (  # noqa: E402
    NEGLIGIBLE_DELTA,
    SIGNIFICANCE_LEVEL,
    SeedSummary,
    SystemComparison,
    add_significance_args,
    format_seed_table,
    paired_test,
    seed_summary,
)

WORDS = "नमस्कार जग भाषा सेतू मराठी वाक्य शब्द अर्थ माणूस पाणी".split()


def make_corpus(n=200, length=9, seed=0):
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(length)) for _ in range(n)]


def degrade(refs, p, seed=1):
    """A system that corrupts a fraction `p` of tokens -- higher p, worse."""
    rng = random.Random(seed)
    return [
        " ".join(w if rng.random() > p else rng.choice(WORDS) for w in r.split())
        for r in refs
    ]


# --------------------------------------------------------------------------- #
# Discrimination
# --------------------------------------------------------------------------- #


def test_a_real_gap_is_significant():
    refs = make_corpus()
    base = degrade(refs, 0.55, seed=1)
    better = degrade(refs, 0.15, seed=2)
    results = paired_test(base, {"better": better}, refs, n_samples=200)
    assert results, "sacrebleu unavailable?"
    bleu = next(r for r in results if r.metric == "BLEU")
    assert bleu.delta > 0
    assert bleu.p_value is not None and bleu.p_value < SIGNIFICANCE_LEVEL
    assert bleu.significant


def test_two_equally_good_systems_are_not_significant():
    """Different outputs, same quality: the test must not manufacture a win."""
    refs = make_corpus()
    base = degrade(refs, 0.4, seed=11)
    twin = degrade(refs, 0.4, seed=12)
    results = paired_test(base, {"twin": twin}, refs, n_samples=300)
    bleu = next(r for r in results if r.metric == "BLEU")
    assert abs(bleu.delta) < 8.0  # same quality by construction
    assert not bleu.significant, f"noise reported as significant: {bleu.format()}"


def test_identical_output_is_never_significant():
    """sacrebleu returns its FLOOR p-value (1/(n+1)) for a zero-sized gap.

    Taken at face value that reads as "p < 0.05, significant" for two
    byte-identical systems -- exactly the kind of claim that should not reach
    a paper.
    """
    refs = make_corpus()
    base = degrade(refs, 0.4, seed=3)
    results = paired_test(base, {"twin": list(base)}, refs, n_samples=200)
    for r in results:
        assert r.delta == pytest.approx(0.0, abs=NEGLIGIBLE_DELTA)
        assert not r.significant, f"identical systems flagged significant: {r.format()}"
        assert "identical" in r.note


def test_negligible_delta_is_not_significant_even_with_a_tiny_p():
    c = SystemComparison(
        system="s", metric="BLEU", baseline_score=42.0,
        system_score=42.0 + NEGLIGIBLE_DELTA / 10,
        p_value=0.0001, n_segments=100, test_type="bs",
    )
    assert not c.significant


def test_a_worse_system_gets_a_negative_delta():
    refs = make_corpus()
    base = degrade(refs, 0.2, seed=5)
    worse = degrade(refs, 0.7, seed=6)
    results = paired_test(base, {"worse": worse}, refs, n_samples=200)
    bleu = next(r for r in results if r.metric == "BLEU")
    assert bleu.delta < 0 and bleu.significant


def test_ter_lower_is_better_still_reports_a_signed_delta():
    refs = make_corpus()
    base = degrade(refs, 0.6, seed=7)
    better = degrade(refs, 0.1, seed=8)
    ter = next(
        r for r in paired_test(base, {"b": better}, refs, n_samples=200)
        if r.metric == "TER"
    )
    # A better system has LOWER TER, so the delta is negative.
    assert ter.delta < 0 and ter.significant


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


def test_metric_names_match_the_repo_csv_columns():
    """sacrebleu labels chrF++ "chrF2++"; joining on that would silently fail."""
    refs = make_corpus(60)
    base = degrade(refs, 0.5, seed=9)
    names = {r.metric for r in paired_test(base, {"x": degrade(refs, 0.2, seed=10)},
                                           refs, n_samples=100)}
    assert names == {"BLEU", "chrF++", "TER"}


def test_mismatched_lengths_are_refused():
    refs = make_corpus(50)
    with pytest.raises(ValueError, match="equal-length"):
        paired_test(refs, {"x": refs[:10]}, refs)


def test_no_systems_returns_nothing():
    refs = make_corpus(20)
    assert paired_test(refs, {}, refs) == []


def test_comparison_serialises():
    refs = make_corpus(60)
    r = paired_test(refs, {"x": degrade(refs, 0.5, seed=13)}, refs, n_samples=100)[0]
    d = r.as_dict()
    assert set(d) >= {"system", "metric", "delta", "p_value", "significant", "note"}


def test_approximate_randomization_also_works():
    refs = make_corpus(100)
    results = paired_test(
        degrade(refs, 0.6, seed=14), {"b": degrade(refs, 0.1, seed=15)},
        refs, test_type="ar", n_samples=200,
    )
    assert results and results[0].test_type == "ar"


# --------------------------------------------------------------------------- #
# Seed aggregation
# --------------------------------------------------------------------------- #


def _rows(values, model="blt_lcm", noise=0.0):
    return [
        {"model": model, "fraction": 0.25, "noise": noise, "seed": 42 + i,
         "BLEU": b, "chrF++": c, "TER": t}
        for i, (b, c, t) in enumerate(values)
    ]


def test_seed_summary_computes_mean_std_and_ci():
    g = seed_summary(_rows([(14.1, 42.0, 70.2), (13.6, 41.2, 71.0), (14.8, 42.9, 69.4)]))
    s = g[("blt_lcm", 0.25, 0.0)]["BLEU"]
    assert s.n == 3
    assert s.mean == pytest.approx(14.1667, abs=1e-3)
    assert s.std == pytest.approx(0.6028, abs=1e-3)
    lo, hi = s.confidence_interval()
    assert lo < s.mean < hi


def test_small_n_uses_a_t_interval_not_a_normal_one():
    """t(2) at 95% is 4.303. Using 1.96 would understate the interval 2.2x."""
    s = SeedSummary("BLEU", [10.0, 12.0, 14.0])
    lo, hi = s.confidence_interval()
    half = (hi - lo) / 2
    assert half == pytest.approx(4.303 * s.std / (3**0.5), rel=1e-3)


def test_single_seed_has_no_error_bar():
    s = SeedSummary("BLEU", [12.0])
    assert s.std == 0.0
    assert s.confidence_interval() == (12.0, 12.0)
    assert "no error bar" in s.format()


def test_empty_summary_is_reported_as_na():
    assert "n/a" in SeedSummary("BLEU", []).format()


def test_non_finite_values_are_dropped():
    g = seed_summary(
        [
            {"model": "m", "fraction": 1, "noise": 0, "BLEU": 10.0, "COMET": float("nan")},
            {"model": "m", "fraction": 1, "noise": 0, "BLEU": None, "chrF++": 40.0},
        ],
        metrics=("BLEU", "chrF++", "COMET"),
    )
    b = g[("m", 1, 0)]
    assert b["BLEU"].n == 1 and b["chrF++"].n == 1
    assert "COMET" not in b


def test_seed_table_formats_mean_and_std():
    g = seed_summary(_rows([(14.1, 42.0, 70.2), (13.6, 41.2, 71.0)]))
    table = format_seed_table(g)
    assert len(table) == 1
    assert "±" in table[0][3]


def test_seed_table_marks_missing_metrics():
    g = seed_summary([{"model": "m", "fraction": 1, "noise": 0}])
    assert format_seed_table(g)[0][3] == "—"


def test_grouping_separates_conditions():
    rows = _rows([(10.0, 30.0, 80.0)], noise=0.0) + _rows(
        [(5.0, 20.0, 90.0)], noise=0.2
    )
    g = seed_summary(rows)
    assert len(g) == 2
    assert g[("blt_lcm", 0.25, 0.0)]["BLEU"].mean == 10.0
    assert g[("blt_lcm", 0.25, 0.2)]["BLEU"].mean == 5.0


def test_significance_args_are_added():
    p = argparse.ArgumentParser()
    add_significance_args(p)
    a = p.parse_args([])
    assert a.significance_test == "bs" and a.significance_samples == 1000
    assert p.parse_args(["--significance_test", "none"]).significance_test == "none"
