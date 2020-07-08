from unittest.mock import (
    patch,
    sentinel,
    MagicMock,
    Mock,
)

import pytest


from covid_model_seiir_pipeline.regression.task import (
    parse_arguments,
    run_beta_regression,
    main,
    RegressionDataInterface,
    RegressionSpecification,
    model,
    np,
)


def test_parse_arguments():
    argstr = "--draw-id 4 --regression-version FOO"
    result = parse_arguments(argstr)

    assert result.draw_id == 4
    assert result.regression_version == "FOO"


def test_main():
    """
    Tests main() calls parse_arguments and then passes two result values to run_ode_fit.
    """
    ns = "covid_model_seiir_pipeline.regression.task."
    with patch(ns + "parse_arguments",
               return_value=Mock(
                   draw_id=sentinel.DRAW_ID,
                   regression_version=sentinel.REGRESSION_VERSION)), \
         patch(ns + "run_beta_regression") as mocked_run_beta_regression:
        main()

    assert mocked_run_beta_regression.called_once_with(draw_id=sentinel.DRAW_ID,
                                                       regression_version=sentinel.REGRESSION_VERSION)


class Test_run_beta_regression:
    """
    Tests unit task of SEIIR pipeline "regression" stage.

    run_beta_regression is an entry point of the application and is meant to
    run at the per-draw level. Running this once for each draw represents the
    non-bookkeeping work done in the entire pipeline "regression" stage.
    """

    # FIXME: Adapting this test to run for both fit and regression requires
    #  more complicated mocking than I'm willing to do right now.  - J.C.
    # def test_happy_path(self, draw_id, regression_version, data_interface, ode_model, numpy_seed):
    #     """
    #     Test happy path logic where nothing fails.
    #
    #     This is primarily a test of RegressionDataInterface usage, as it has
    #     responsibilities for both loading and persisting data.
    #     """
    #     # run imperative function
    #     run_beta_regression(draw_id=draw_id, regression_version=regression_version)
    #
    #     # data_interface should have loaded location data
    #     data_interface.load_location_ids.assert_called_once()
    #     data_interface.load_all_location_data.assert_called_once_with(
    #         location_ids=sentinel.LOCATION_IDS,
    #         draw_id=draw_id)
    #     data_interface.load_covariates.assert_called_onece_with(
    #         sentinel.COVARIATES,
    #         sentinel.LOCATION_IDS
    #     )
    #
    #     numpy_seed.assert_called_once_with(draw_id)
    #
    #     # UNTESTED - ODEProcessInput is created and used to initialize an ODEProcess
    #
    #     # ode_model.process() is called
    #     ode_model.process.assert_called_once()
    #
    # FIXME: here we need mocks for
    #   - align_beta_with_covariates
    #   - build regressor
    #   - predict
    #
    #     # parameters dataframe must be created and saved
    #     ode_model.create_params_df.assert_called_once()
    #     data_interface.save_draw_beta_param_file.assert_called_once_with(
    #         ode_model.create_params_df(), draw_id)
    #
    #     # start/end date DataFrame is created and saved
    #     ode_model.create_start_end_date_df.assert_called_once()
    #     data_interface.save_draw_date_file.assert_called_once_with(
    #         ode_model.create_start_end_date_df(), draw_id)

    # fixtures for testing internal behavior of run_beta_regression
    @pytest.fixture(autouse=True)
    def specification(self):
        """
        mock RegressionSpecification.from_path to avoid I/O. Could instead
        use fixture file.

        This mock basically just exists to allow us to ignore the details of
        creating a RegressionDataInterface object.
        """
        mocked_spec = MagicMock()
        mocked_spec.covariates.return_value = sentinel.COVARIATES

        with patch.object(RegressionSpecification, "from_path", return_value=mocked_spec):
            yield mocked_spec

    @pytest.fixture
    def data_interface(self):
        """
        data_interface does a significant amount of work in this function and
        the calls to it will be highly tested.
        """
        mocked_di = MagicMock()
        mocked_di.load_location_ids.return_value = sentinel.LOCATION_IDS

        with patch.object(RegressionDataInterface, "from_specification", return_value=mocked_di):
            yield mocked_di

    @pytest.fixture
    def ode_model(self):
        """
        Mock away the creation of an ODE model via ODEProcess.

        This object is responsibile for processing data and creating several
        results.
        """
        mocked_ode_model = MagicMock()

        with patch.object(model, "ODEProcess", return_value=mocked_ode_model):
            yield mocked_ode_model

    @pytest.fixture
    def numpy_seed(self):
        """
        Spy on (as opposed to mock) numpy.random.seed to ensure it is called.
        """
        with patch.object(np.random, "seed",
                          wraps=np.random.seed) as seed_spy:
            yield seed_spy

    # fixtures for calling run_beta_regression
    @pytest.fixture
    def draw_id(self):
        return 42

    @pytest.fixture
    def regression_version(self, tmpdir):
        """
        Return a pathlib.Path-like value where regression_version is to be
        saved.

        Use pytest's tmpdir construct as it is suitable and requires no effort.
        """
        return tmpdir
