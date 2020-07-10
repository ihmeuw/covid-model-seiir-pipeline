from pathlib import Path

import pandas

from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.marshall import (
    CSVMarshall,
)
from covid_model_seiir_pipeline.paths import RegressionPaths
from covid_model_seiir_pipeline.regression.data import RegressionDataInterface


class TestForecastDataInterfaceIO:
    def test_regression_io(self, tmpdir, coefficients):
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

        # Step 2: load files as they would be loaded in forecast
        loaded = fdi.load_regression_coefficients(draw_id=4)

        pandas.testing.assert_frame_equal(coefficients, loaded)
