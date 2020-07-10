from pathlib import Path

import numpy
import pandas

from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.marshall import (
    CSVMarshall,
)
from covid_model_seiir_pipeline.paths import RegressionPaths
from covid_model_seiir_pipeline.regression.data import RegressionDataInterface


class TestForecastDataInterfaceIO:
    def test_regression_io(self, tmpdir, coefficients, parameters):
        """
        Test I/O relating to regression stage.

        This only includes loading files, as they are all saved by the
        RegressionDataInterface.
        """
        regress_paths = RegressionPaths(Path(tmpdir))
        rdi = RegressionDataInterface(
            infection_paths=None,
            regression_paths=regress_paths,
            covariate_paths=None,
            regression_marshall=CSVMarshall(regress_paths.root_dir),
        )

        fdi = ForecastDataInterface(
            forecast_paths=None,
            regression_paths=None,
            covariate_paths=None,
            regression_marshall=CSVMarshall.from_paths(regress_paths),
            forecast_marshall=None,
        )

        # Step 1: save files (normally done in regression)
        rdi.save_regression_coefficients(coefficients, draw_id=4)
        rdi.save_draw_beta_param_file(parameters, draw_id=4)

        # Step 2: load files as they would be loaded in forecast
        loaded_coefficients = fdi.load_regression_coefficients(draw_id=4)
        loaded_parameters = fdi.load_beta_params(draw_id=4)

        # Step 3: test files
        pandas.testing.assert_frame_equal(coefficients, loaded_coefficients)

        # load_beta_params does not return a DataFrame but instead a dict
        # in addition, some rounding error occurs in the save/load from CSV
        expected_parameters = parameters.set_index('params')['values'].to_dict()
        try:
            assert expected_parameters == loaded_parameters
        except AssertionError:
            # assert keys are identical
            assert set(expected_parameters) == set(loaded_parameters)
            # assert each value is accurate to 15 decimal places
            for k, expected in expected_parameters.items():
                loaded = loaded_parameters[k]
                numpy.testing.assert_almost_equal(loaded, expected, decimal=15)
