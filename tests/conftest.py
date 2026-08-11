"""Shared pytest setup.

Reports the device once per session, for the same reason every script in this
repository does: the tests exercise real torch code (``test_checkpoint_utils.py``
trains tiny models, ``test_paper_fidelity.py`` runs the model implementations),
and a run that silently landed on CPU is otherwise indistinguishable from one
that used the GPU.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcm_scripts.device_utils import report_device


@pytest.fixture(scope="session", autouse=True)
def _report_device():
    """Print the device once, before the first test runs."""
    # -s is not always passed, so write straight to the terminal reporter's stream.
    report_device(label="tests", warn_cpu=False)
