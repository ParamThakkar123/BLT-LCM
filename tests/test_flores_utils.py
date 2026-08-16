"""Tests for FLORES-200 loading.

The dataset download is skipped when unavailable, so these run offline on a
machine with the HF cache and skip cleanly on one without.
"""

import argparse
import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lcm_scripts"))

from flores_utils import (  # noqa: E402
    DEFAULT_INDIC_PANEL,
    INDIC_LANGUAGES,
    FloresUnavailable,
    add_flores_args,
    available_languages,
    language_name,
    load_flores_pairs,
    resolve_lang,
)


def flores_or_skip(fn, *a, **k):
    """Skip when the dataset cannot be loaded at all.

    Only ``FloresUnavailable`` counts as "not available" -- a ValueError from
    the loader is a genuine assertion about behaviour (an unknown language,
    say) and must propagate, or the test that checks for it silently passes
    as a skip.
    """
    try:
        return fn(*a, **k)
    except FloresUnavailable as e:
        pytest.skip(f"FLORES unavailable: {e}")


# --------------------------------------------------------------------------- #
# Language codes (no dataset needed)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("mr", "mar_Deva"), ("marathi", "mar_Deva"), ("Marathi", "mar_Deva"),
        ("mar_Deva", "mar_Deva"), ("en", "eng_Latn"), ("english", "eng_Latn"),
        ("hi", "hin_Deva"), ("ta", "tam_Taml"), ("bn", "ben_Beng"),
    ],
)
def test_alias_resolution(alias, expected):
    assert resolve_lang(alias) == expected


def test_unknown_code_passes_through():
    """An unrecognised FLORES code must reach the dataset, not be swallowed."""
    assert resolve_lang("zho_Hans") == "zho_Hans"


def test_empty_language_is_refused():
    with pytest.raises(ValueError):
        resolve_lang("")


def test_language_names():
    assert language_name("mr") == "Marathi"
    assert language_name("eng_Latn") == "English"
    assert language_name("tam_Taml") == "Tamil"


def test_default_panel_spans_several_scripts():
    """A byte-level claim tested only within Devanagari would prove little."""
    scripts = {code.split("_")[1] for code in DEFAULT_INDIC_PANEL}
    assert len(scripts) >= 4, scripts
    assert "mar_Deva" in DEFAULT_INDIC_PANEL


def test_indic_registry_codes_are_well_formed():
    for code in INDIC_LANGUAGES:
        lang, _, script = code.partition("_")
        assert len(lang) == 3 and script, code


# --------------------------------------------------------------------------- #
# Loading (needs the dataset)
# --------------------------------------------------------------------------- #


def test_devtest_is_the_full_benchmark():
    pairs = flores_or_skip(load_flores_pairs, "en", "mr", "devtest")
    # FLORES-200 devtest is 1012 sentences; a different count means a
    # different benchmark, and the numbers would not be comparable.
    assert len(pairs) == 1012
    assert all(p.source and p.target for p in pairs)


def test_dev_split_is_separate():
    dev = flores_or_skip(load_flores_pairs, "en", "mr", "dev")
    assert len(dev) == 997


def test_n_way_parallel_means_identical_sources_across_targets():
    """The same source sentences for every target is what makes the
    cross-language comparison a comparison of models, not of test sets."""
    mr = flores_or_skip(load_flores_pairs, "en", "mr", "devtest", 50)
    ta = flores_or_skip(load_flores_pairs, "en", "ta", "devtest", 50)
    assert [p.source for p in mr] == [p.source for p in ta]
    assert [p.target for p in mr] != [p.target for p in ta]


def test_max_examples_truncates():
    assert len(flores_or_skip(load_flores_pairs, "en", "mr", "devtest", 25)) == 25


def test_missing_language_is_a_clear_error():
    with pytest.raises(ValueError, match="no column"):
        flores_or_skip(load_flores_pairs, "en", "not_ALang", "devtest")


def test_marathi_targets_are_devanagari():
    pairs = flores_or_skip(load_flores_pairs, "en", "mr", "devtest", 20)
    devanagari = sum(
        any("ऀ" <= ch <= "ॿ" for ch in p.target) for p in pairs
    )
    assert devanagari == len(pairs)


def test_available_languages_covers_the_panel():
    langs = set(flores_or_skip(available_languages, "devtest"))
    missing = [c for c in DEFAULT_INDIC_PANEL if c not in langs]
    assert not missing, missing


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_flores_args():
    p = argparse.ArgumentParser()
    add_flores_args(p)
    a = p.parse_args([])
    assert a.flores_split == "devtest" and a.flores_tgt == "mar_Deva"
    b = p.parse_args(["--flores_split", "dev", "--flores_tgt", "ta"])
    assert b.flores_split == "dev" and resolve_lang(b.flores_tgt) == "tam_Taml"
