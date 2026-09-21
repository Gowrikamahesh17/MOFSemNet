"""Shared pytest config: gates tests that need network access (downloading ChemBERTa/MolT5 from
the Hugging Face Hub) behind an explicit --run-network flag, so `pytest` alone stays fast and
runnable offline, matching the rest of this test suite.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.network (downloads pretrained models from the Hugging Face Hub).",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "network: test requires network access to download a pretrained model")
    config.addinivalue_line("markers", "slow: test trains a real model end-to-end and takes longer than the rest of the suite")


@pytest.fixture
def run_network(request):
    if not request.config.getoption("--run-network"):
        pytest.skip("Requires network access — run with `pytest --run-network` to include this test.")
