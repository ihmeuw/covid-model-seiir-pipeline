from pathlib import Path
import warnings

import numpy
import pandas
import pytest

from covid_model_seiir_pipeline.lib import io
from covid_model_seiir_pipeline.pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.pipeline.regression.data import RegressionDataInterface


class TestForecastDataInterfaceIO:
    def test_regression_io(self, tmpdir, coefficients, dates, regression_beta, location_data, parameters):
        """
        Test I/O relating to regression stage.

        This only includes loading files, as they are all saved by the
        RegressionDataInterface.
        """
        rdi = RegressionDataInterface(
            infection_root=None,
            covariate_root=None,
            coefficient_root=None,
            regression_root=io.RegressionRoot(tmpdir),
        )
        rdi.make_dirs()
        fdi = ForecastDataInterface(
            regression_root=io.RegressionRoot(tmpdir),
            covariate_root=None,
            forecast_root=None,
            fh_subnationals=False,
        )
        # Patch on the regression data interface
        fdi._get_regression_data_interface = lambda: rdi

        # Step 1: save files (normally done in regression)

        rdi.save_regression_coefficients(coefficients, draw_id=4)
        rdi.save_beta_param_file(parameters, draw_id=4)
        rdi.save_date_file(dates, draw_id=4)
        rdi.save_regression_betas(regression_beta, draw_id=4)
        rdi.save_infection_data(location_data, draw_id=4)

        # Step 2: load files as they would be loaded in forecast
        loaded_coefficients = fdi.load_regression_coefficients(draw_id=4)
        loaded_parameters = fdi.load_beta_params(draw_id=4)
        loaded_transition_dates = fdi.load_transition_date(draw_id=4)
        loaded_regression_beta = fdi.load_beta_regression(draw_id=4)
        loaded_location_data = fdi.load_infection_data(draw_id=4)

        # Step 3: test files
        pandas.testing.assert_frame_equal(coefficients, loaded_coefficients)
        transition_dates = pandas.to_datetime(dates['end_date']).rename('date').reset_index()
        loaded_transition_dates = loaded_transition_dates.reset_index()
        pandas.testing.assert_frame_equal(transition_dates, loaded_transition_dates)
        pandas.testing.assert_frame_equal(regression_beta, loaded_regression_beta)
        pandas.testing.assert_frame_equal(location_data, loaded_location_data)

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
            warnings.warn("beta fit parameters accurate only to 15 decimal places after save/load cycle")

    def test_forecast_io(self, tmpdir, components, beta_scales, forecast_outputs):
        di = ForecastDataInterface(
            regression_root=None,
            covariate_root=None,
            forecast_root=io.ForecastRoot(tmpdir),
            fh_subnationals=False,
        )
        di.make_dirs(scenario=['happy'])
        # Step 1: save files
        di.save_components(components, scenario="happy", draw_id=4)
        di.save_beta_scales(beta_scales, scenario="happy", draw_id=4)
        di.save_raw_outputs(forecast_outputs, scenario="happy", draw_id=4)

        # Step 2: test save location
        # this is sort of cheating, but it ensures that scenario things are
        # nicely nested as they should be
        assert (Path(tmpdir) / "happy" / "component_draws" / "draw_4.csv").exists()
        assert (Path(tmpdir) / "happy" / "beta_scaling" / "draw_4.csv").exists()
        assert (Path(tmpdir) / "happy" / "raw_outputs" / "draw_4.csv").exists()

        # Step 3: load those files
        loaded_components = di.load_components(scenario="happy", draw_id=4)
        loaded_beta_scales = di.load_beta_scales(scenario="happy", draw_id=4)
        loaded_forecast_outputs = di.load_raw_outputs(scenario="happy", draw_id=4)

        # Step 4: test files
        pandas.testing.assert_frame_equal(components, loaded_components)
        pandas.testing.assert_frame_equal(beta_scales, loaded_beta_scales)
        pandas.testing.assert_frame_equal(forecast_outputs, loaded_forecast_outputs)


def assert_equal_after_date_conversion(expected, actual, date_cols):
    with pytest.raises(AssertionError):
        pandas.testing.assert_frame_equal(expected, actual)

    for col in date_cols:
        assert pandas.api.types.is_string_dtype(expected[col])
        expected[col] = pandas.to_datetime(expected[col])

    pandas.testing.assert_frame_equal(expected, actual)
