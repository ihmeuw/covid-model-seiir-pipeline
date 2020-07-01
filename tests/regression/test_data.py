import datetime as dt
from pathlib import Path

import pandas
import pytest

from covid_model_seiir_pipeline.marshall import (
    CSVMarshall,
    Keys as MKeys,
)
from covid_model_seiir_pipeline.paths import ODEPaths
from covid_model_seiir_pipeline.regression.data import RegressionDataInterface


@pytest.fixture
def ode_fit_beta():
    """
    beta result created by ODEProcess.create_result_df()
    """
    return pandas.DataFrame({
            'location_id': [60887],
            'date': [dt.datetime(2020, 6, 14)],
            'days': [115],
            'beta': [0.30910847173899497],
            'S': [3680864.701899232],
            'E': [1625.747531399217],
            'I1': [898.9826448859556],
            'I2': [559.2951366672883],
            'R': [57997.25978282046],
            'newE': [432.884735422745],
            'newE_obs': [432.884735422745],
    })


class TestRegressionDataInterfaceIO:
    def test_ode_fit_io(self, tmpdir, tmpdir_file_count, ode_fit_beta):
        """
        Test I/O relating to the fit stage.

        This only includes loading files.
        """
        di = RegressionDataInterface(
            regression_paths=None,
            ode_paths=ODEPaths(Path(tmpdir)),
            covariate_paths=None,
        )

        # Step 1: create files
        m = CSVMarshall(di.ode_paths.root_dir)
        m.dump(ode_fit_beta, key=MKeys.beta(draw_id=4))

        # Step 2: load files
        # Note: 60887 corresponds to a value in the ode_fit_beta fixture
        loaded = di.load_ode_fits(draw_id=4, location_ids=[60887])

        pandas.testing.assert_frame_equal(ode_fit_beta, loaded)
