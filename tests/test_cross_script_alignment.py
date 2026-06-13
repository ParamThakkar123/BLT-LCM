from scripts.run_cross_script_alignment import boundary_metrics, hindi_boundary_offsets, run_alignment


def test_boundary_metrics_counts_tolerance_matches():
    precision, recall, f1 = boundary_metrics({9, 20}, {10, 30}, tolerance=1)
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


def test_hindi_boundary_offsets_adds_suffix_boundary():
    offsets = hindi_boundary_offsets("लड़कों ने खेला।")
    assert len(offsets) >= 2


def test_run_alignment_compares_to_marathi_reference():
    results = run_alignment(["मैं घर जाता है।", "वह किताब पढ़ता है।"], tolerance=3, marathi_baseline_f1=0.6341)
    assert results["language"] == "hindi"
    assert results["num_sentences"] == 2
    assert "delta_vs_marathi_best_f1" in results
    assert results["claim_guidance"].startswith("Do not call")
