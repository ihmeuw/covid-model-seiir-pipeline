import os
import pathlib

import pandas
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


# Data fixtures
#
# These represent actual data in various parts of the system that needs to be
# marshalled/unmarshalled
@pytest.fixture
def fit_beta():
    "Example beta result from an ODE model fit."
    return pandas.DataFrame({
        'location_id': [10],
        'date': ['2020-03-02'],
        'days': [3],
        'beta': [0.7805482566415091],
        'S': [16603117.700039685],
        'E': [0.00024148647168971383],
        'I1': [8.662533567671615e-05],
        'I2': [3.626634800151582e-05],
        'R': [4.0834675478217315e-05],
        'newE': [9.88132445500906e-05],
        'newE_obs': [9.88132445500906e-05],
    })


@pytest.fixture
def parameters():
    "Example parameters data from an ODE model."
    return pandas.DataFrame([
        ['alpha', 0.9967029839013677],
        ['sigma', 0.2729460588151238],
        ['gamma1', 0.5],
        ['gamma2', 0.809867822982699],
        ['day_shift', 5.0],
    ], columns=['params', 'values'])


@pytest.fixture
def dates():
    "Example dates data from an ODE model."
    return pandas.DataFrame([
        [523, '2020-03-06', '2020-05-05'],
        [526, '2020-03-08', '2020-05-05'],
        [533, '2020-02-23', '2020-05-05'],
        [537, '2020-02-26', '2020-05-05'],
    ], columns=['loc_id', 'start_date', 'end_date'])


@pytest.fixture
def coefficients():
    "Example coefficients data from regression."
    return pandas.DataFrame([
        [523, -0.2356716680846281, 0.010188535227465946, 0.27234473836297995, -598.8754409180113],
        [526, -0.24334559662138652, 0.019963189989381125, 0.27234473836297995, -598.8754409180113],
        [533, -0.214475560389406, 0.011172361940536456, 0.27234473836297995, -598.8754409180113],
        [537, -0.09571280930682702, 0.011990915850960831, 0.27234473836297995, -598.8754409180113],
        [538, 0.0988105530655817, 0.0187165992693182, 0.27234473836297995, -598.8754409180113],
    ], columns=["group_id", "intercept", "mobility", "proportion_over_1k", "testing"])
