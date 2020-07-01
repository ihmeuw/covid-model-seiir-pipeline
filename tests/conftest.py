import os
import pathlib

import pytest


@pytest.fixture(scope="session")
def fixture_dir():
    here = pathlib.Path(__file__).parent
    return here / "fixtures"


@pytest.fixture
def tmpdir_file_count(tmpdir):
    def file_count():
        "Return count of files in our ODEDataInterface storage location."
        cnt = 0
        for _, _, files in os.walk(tmpdir):
            for _ in files:
                cnt += 1
        return cnt

    return file_count
