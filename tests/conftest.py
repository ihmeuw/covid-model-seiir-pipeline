import pathlib

import pytest


@pytest.fixture
def fixture_dir():
    here = pathlib.Path(__file__).parent
    return here / "fixtures"
