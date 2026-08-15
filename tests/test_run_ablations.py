"""Tests for the ablation driver's parameter matching.

The compute-matched comparison only means anything if the two models really do
have comparable parameter budgets, so the counting is what gets tested here.
Actually running the ablations trains models and is out of scope for a unit
test; `--dry_run` covers the command construction.
"""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lcm_scripts"))

from run_ablations import (  # noqa: E402
    DECODE_METHODS,
    LCM_VARIANTS,
    count_parameters,
    lcm_parameter_count,
    match_transformer_to_lcm,
    transformer_parameter_count,
)


@pytest.mark.parametrize(
    "vocab,d_model,nhead,layers,dim_ff",
    [
        (500, 64, 4, 2, 128),
        (1000, 128, 8, 3, 512),
        (16000, 256, 8, 2, 1024),
        (2000, 384, 8, 4, 768),
        (8000, 512, 8, 1, 2048),
    ],
)
def test_analytic_count_matches_a_real_model(vocab, d_model, nhead, layers, dim_ff):
    """The formula must be exact, not approximate.

    The search grid reaches past a billion parameters; instantiating each
    candidate to call numel() exhausts memory, so the count is computed from
    the layer shapes instead. If nn.Transformer's layout ever changes, this
    catches it rather than letting a silently wrong "match" into a paper.
    """
    from train_bpe_transformer import BPETransformer

    analytic = transformer_parameter_count(vocab, d_model, nhead, layers, dim_ff)
    actual = count_parameters(
        BPETransformer(vocab, d_model, nhead, layers, dim_ff, 0.1)
    )
    assert analytic == actual


def test_lcm_parameter_count_is_a_real_count():
    from base_lcm import BaseLCM

    n = lcm_parameter_count(64, 128, 2, 8)
    assert n == count_parameters(
        BaseLCM(embed_dim=64, model_dim=128, n_layers=2, n_heads=8)
    )


def test_matching_finds_a_close_configuration_at_the_paper_size():
    """The configuration the paper actually uses must be matchable."""
    target = lcm_parameter_count(1024, 2048, 12, 16)
    m = match_transformer_to_lcm(target, 16000)
    assert m, "no configuration found"
    assert m["relative_error"] < 0.05, (
        f"closest match is {m['relative_error']:.1%} off: {m}"
    )


def test_matching_reports_its_error_honestly(capsys):
    """A target the grid cannot reach must warn, not silently claim a match."""
    m = match_transformer_to_lcm(1_000, 16000)  # far below the smallest config
    assert m["relative_error"] > 0.05
    assert "WARNING" in capsys.readouterr().out


def test_matched_configuration_is_constructible():
    from train_bpe_transformer import BPETransformer

    target = lcm_parameter_count(256, 512, 4, 8)
    m = match_transformer_to_lcm(target, 4000)
    model = BPETransformer(
        4000, m["d_model"], m["nhead"], m["num_layers"], m["dim_ff"], 0.1
    )
    assert count_parameters(model) == m["parameters"]


def test_head_divisibility_is_respected():
    m = match_transformer_to_lcm(50_000_000, 16000, nhead=8)
    assert m["d_model"] % m["nhead"] == 0


def test_all_lcm_variants_are_covered():
    """The ablation must offer every variant train_lcm_blt.py implements."""
    import argparse

    from train_lcm_blt import main  # noqa: F401  (import check only)

    assert set(LCM_VARIANTS) == {"base", "one_tower", "two_tower", "quant"}
    assert set(DECODE_METHODS) == {"generative", "retrieval"}
    del argparse
