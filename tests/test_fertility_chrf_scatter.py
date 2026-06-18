import math

from lcm_scripts.fertility_chrf_scatter import (
    compute_rows,
    pearson_correlation,
    spearman_correlation,
    summarize_rows,
    theoretical_factor,
    validate_parallel_files,
)


def toy_metric(hyps, refs):
    matches = sum(1 for hyp, ref in zip(hyps, refs) if hyp == ref)
    return 100.0 * matches / len(refs)


def test_compute_rows_groups_sentence_subsets_and_delta_chrf():
    fertility = {
        "noun_root": {"display_name": "Noun roots", "num_words": 4, "fertility_lambda": 1.0},
        "compound": {"display_name": "Compound words", "num_words": 3, "fertility_lambda": 2.0},
    }
    sentence_ids = {
        "noun_root": {0, 1},
        "compound": {1, 2},
    }
    refs = ["a", "b", "c"]
    bpe = ["a", "x", "x"]
    blt = ["a", "b", "c"]

    rows = compute_rows(
        fertility,
        sentence_ids,
        bpe,
        blt,
        refs,
        ["noun_root", "compound"],
        metric_fn=toy_metric,
    )

    by_class = {row.morpheme_class: row for row in rows}
    assert by_class["noun_root"].sentence_count == 2
    assert by_class["noun_root"].bpe_chrf == 50.0
    assert by_class["noun_root"].blt_chrf == 100.0
    assert by_class["noun_root"].delta_chrf == 50.0

    assert by_class["compound"].sentence_count == 2
    assert by_class["compound"].bpe_chrf == 0.0
    assert by_class["compound"].blt_chrf == 100.0
    assert by_class["compound"].delta_chrf == 100.0
    assert by_class["compound"].f_lambda == 0.5
    assert by_class["compound"].bound_satisfied


def test_theoretical_factor_is_monotone_inverse_lambda():
    assert theoretical_factor(1.0, 1.0) == 1.0
    assert theoretical_factor(2.0, 1.0) == 0.5
    assert math.isclose(theoretical_factor(4.0, 0.5), 0.5)


def test_summary_reports_bound_rate_and_lambda_delta_correlation():
    fertility = {
        "noun_root": {"display_name": "Noun roots", "num_words": 4, "fertility_lambda": 1.0},
        "compound": {"display_name": "Compound words", "num_words": 3, "fertility_lambda": 2.0},
    }
    rows = compute_rows(
        fertility,
        {"noun_root": {0, 1}, "compound": {1, 2}},
        ["a", "x", "x"],
        ["a", "b", "c"],
        ["a", "b", "c"],
        ["noun_root", "compound"],
        metric_fn=toy_metric,
    )

    summary = summarize_rows(rows, bound_exponent=1.0)

    assert summary["num_classes"] == 2
    assert summary["bound_satisfied_count"] == 2
    assert summary["bound_satisfied_fraction"] == 1.0
    assert math.isclose(summary["pearson_lambda_delta_chrf"], 1.0)
    assert math.isclose(summary["spearman_lambda_delta_chrf"], 1.0)


def test_correlation_helpers_handle_ties_and_undefined_cases():
    assert math.isclose(pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]), 1.0)
    assert math.isclose(spearman_correlation([1.0, 2.0, 2.0], [1.0, 3.0, 2.0]), 0.8660254037844387)
    assert math.isnan(pearson_correlation([1.0], [2.0]))


def test_validate_parallel_files_rejects_mismatched_lengths():
    try:
        validate_parallel_files(["a"], ["a", "b"], ["a"])
    except ValueError as exc:
        assert "identical line counts" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched file lengths")
