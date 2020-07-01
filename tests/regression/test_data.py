from pathlib import Path

import pandas

from covid_model_seiir_pipeline.marshall import (
    CSVMarshall,
    Keys as MKeys,
)
from covid_model_seiir_pipeline.paths import ODEPaths, RegressionPaths
from covid_model_seiir_pipeline.regression.data import RegressionDataInterface


class TestRegressionDataInterfaceIO:
    def test_ode_fit_io(self, tmpdir, tmpdir_file_count, fit_beta):
        """
        Test I/O relating to the fit stage.

        This only includes loading files.
        """
        ode_paths = ODEPaths(Path(tmpdir))
        di = RegressionDataInterface(
            regression_paths=None,
            ode_paths=ode_paths,
            covariate_paths=None,
            ode_marshall=CSVMarshall(ode_paths.root_dir),
            regression_marshall=None,
        )

        # Step 1: create files
        m = CSVMarshall(di.ode_paths.root_dir)
        m.dump(fit_beta, key=MKeys.fit_beta(draw_id=4))

        # Step 2: load files
        # Note: 10 corresponds to a value in the ode_fit_beta fixture
        loaded = di.load_ode_fits(draw_id=4, location_ids=[10])

        # Step 3: test
        # Note: load_ode_fits explicitly converts the "date" column
        assert pandas.core.dtypes.common.is_datetime64_any_dtype(loaded['date'])
        loaded['date'] = loaded['date'].dt.strftime("%Y-%m-%d")
        pandas.testing.assert_frame_equal(fit_beta, loaded)

    def test_regression_io(self, tmpdir, tmpdir_file_count, coefficients, regression_beta):
        """
        Test I/O relating to regression stage.

        This only includes saving files
        """
        regress_paths = RegressionPaths(Path(tmpdir))
        di = RegressionDataInterface(
            regression_paths=regress_paths,
            ode_paths=None,
            covariate_paths=None,
            ode_marshall=None,
            regression_marshall=CSVMarshall(regress_paths.root_dir),
        )
        # Step 1: count files
        assert tmpdir_file_count() == 0, "Files somehow already exist in storage dir"
        # Step 2: save files
        di.save_regression_coefficients(coefficients, draw_id=4)
        di.save_regression_betas(regression_beta, draw_id=4)
        # Step 3: count files (again)
        assert tmpdir_file_count() == 2
