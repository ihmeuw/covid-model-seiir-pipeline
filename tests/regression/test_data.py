from pathlib import Path

from covid_model_seiir_pipeline.marshall import (
    CSVMarshall,
)
from covid_model_seiir_pipeline.paths import RegressionPaths
from covid_model_seiir_pipeline.regression.data import RegressionDataInterface


class TestRegressionDataInterfaceIO:
    def test_regression_io(self, tmpdir, tmpdir_file_count, parameters, dates, coefficients, regression_beta):
        """
        Test I/O relating to regression stage.

        This only includes saving files
        """
        regress_paths = RegressionPaths(Path(tmpdir))
        di = RegressionDataInterface(
            infection_paths=None,
            regression_paths=regress_paths,
            covariate_paths=None,
            regression_marshall=CSVMarshall(regress_paths.root_dir),
        )
        # Step 1: count files

        assert tmpdir_file_count() == 0, "Files somehow already exist in storage dir"

        # Step 2: save files
        di.save_beta_param_file(parameters, draw_id=4)
        di.save_date_file(dates, draw_id=4)
        di.save_regression_coefficients(coefficients, draw_id=4)
        di.save_regression_betas(regression_beta, draw_id=4)

        # Step 3: count files (again)
        assert tmpdir_file_count() == 4
