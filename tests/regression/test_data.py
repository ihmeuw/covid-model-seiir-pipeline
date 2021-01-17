from covid_model_seiir_pipeline.lib import io
from covid_model_seiir_pipeline.pipeline.regression.data import RegressionDataInterface


class TestRegressionDataInterfaceIO:
    def test_regression_io(self, tmpdir, tmpdir_file_count,
                           parameters, dates, coefficients, regression_beta, location_data):
        """
        Test I/O relating to regression stage.

        This only includes saving files
        """

        di = RegressionDataInterface(
            infection_root=None,
            covariate_root=None,
            coefficient_root=None,
            regression_root=io.RegressionRoot(tmpdir),
        )
        di.make_dirs()
        # Step 1: count files

        assert tmpdir_file_count() == 0, "Files somehow already exist in storage dir"

        # Step 2: save files
        di.save_beta_param_file(parameters, draw_id=4)
        di.save_date_file(dates, draw_id=4)
        di.save_regression_coefficients(coefficients, draw_id=4)
        di.save_regression_betas(regression_beta, draw_id=4)
        di.save_infection_data(location_data, draw_id=4)

        # Step 3: count files (again)
        assert tmpdir_file_count() == 5
