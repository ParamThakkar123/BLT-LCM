"""Helpers for locating Indic NLP resource data."""

from __future__ import annotations

import os
from pathlib import Path

from indicnlp import common

_RESOURCE_SENTINEL = Path("script") / "all_script_phonetic_data.csv"
_ENV_VAR = "INDIC_RESOURCES_PATH"


def _candidate_paths() -> list[Path]:
    """Return likely Indic NLP resource directories in priority order."""
    candidates: list[Path] = []

    env_path = os.environ.get(_ENV_VAR)
    if env_path:
        candidates.append(Path(env_path).expanduser())

    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repo_root / "indic_nlp_resources",
            repo_root / "resources" / "indic_nlp_resources",
            Path.cwd() / "indic_nlp_resources",
            Path.home() / "indic_nlp_resources",
        ]
    )

    # Keep a Windows path only as a last-resort compatibility fallback for
    # existing local setups. This should not be the default on other machines.
    candidates.append(Path("D:/phase2/indic_nlp_resources"))

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def configure_indic_resources() -> Path:
    """Locate and configure the Indic NLP resources directory.

    The ``indic-nlp-library`` Python package does not include the resource CSVs
    required by ``indicnlp.loader.load()`` and the Marathi morphology analyzer.
    Users can set ``INDIC_RESOURCES_PATH`` to the cloned
    ``indic_nlp_resources`` directory; otherwise common in-repo and home
    locations are checked.
    """
    tried = _candidate_paths()
    for path in tried:
        if (path / _RESOURCE_SENTINEL).is_file():
            resolved = path.resolve()
            common.set_resources_path(str(resolved))
            return resolved

    tried_text = "\n".join(f"  - {path}" for path in tried)
    raise FileNotFoundError(
        "Indic NLP resources were not found. Clone/download the "
        "indic_nlp_resources repository and set INDIC_RESOURCES_PATH to that "
        "directory, or place it at ./indic_nlp_resources. Expected to find "
        f"{_RESOURCE_SENTINEL}. Tried:\n{tried_text}"
    )
