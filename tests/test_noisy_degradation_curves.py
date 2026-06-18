import csv

import pytest

from lcm_scripts.bhashasetu_utils import add_character_noise
from lcm_scripts.noisy_degradation_curves import read_metric_rows, select_curve_rows, write_curve_csv


def test_add_character_noise_can_drop_devanagari_matras():
    text = "काला पीला"
    variants = {add_character_noise(text, 1.0, seed=i) for i in range(50)}
    assert any(len(v) < len(text) for v in variants)


def test_select_curve_rows_requires_both_models_all_noise_levels(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "noise", "chrF++", "fraction"])
        writer.writeheader()
        writer.writerow({"model": "bpe_lcm", "noise": "0", "chrF++": "40", "fraction": "0.5"})
        writer.writerow({"model": "bpe_lcm", "noise": "0.1", "chrF++": "35", "fraction": "0.5"})
        writer.writerow({"model": "blt_lcm", "noise": "0", "chrF++": "42", "fraction": "0.5"})

    rows = read_metric_rows([str(csv_path)])
    with pytest.raises(ValueError, match="blt_lcm@0.1"):
        select_curve_rows(rows, noise_levels=(0.0, 0.1), fraction=0.5)


def test_write_curve_csv_for_complete_comparison(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "noise", "chrF++", "fraction"])
        writer.writeheader()
        for model, base in [("bpe_lcm", 40), ("blt_lcm", 44)]:
            for noise in (0.0, 0.1, 0.2):
                writer.writerow({"model": model, "noise": noise, "chrF++": base - noise * 50, "fraction": "0.5"})

    selected = select_curve_rows(read_metric_rows([str(csv_path)]), fraction=0.5)
    out_csv = tmp_path / "curve.csv"
    write_curve_csv(selected, str(out_csv))

    out_rows = list(csv.DictReader(out_csv.open(newline="", encoding="utf-8")))
    assert len(out_rows) == 6
    assert out_rows[0]["model"] == "bpe_lcm"
    assert out_rows[-1]["noise"] == "0.20"
