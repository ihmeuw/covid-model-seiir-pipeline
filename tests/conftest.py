import pathlib

import pytest


@pytest.fixture(scope="session")
def fixture_dir():
    here = pathlib.Path(__file__).parent
    return here / "fixtures"
