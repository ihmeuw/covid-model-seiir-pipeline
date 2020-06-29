from unittest.mock import (
    patch,
    sentinel,
    MagicMock,
    Mock,
)

import pytest


from covid_model_seiir_pipeline.ode_fit.task import (
    parse_arguments,
    run_ode_fit,
    main,
    ODEDataInterface,
    FitSpecification,
    model,
)


def test_parse_arguments():
    argstr = "--draw-id 4 --ode-version FOO"
    result = parse_arguments(argstr)

    assert result.draw_id == 4
    assert result.ode_version == "FOO"


def test_main():
    """
    Tests main() calls parse_arguments and then passes two result values to run_ode_fit.
    """
    ns = "covid_model_seiir_pipeline.ode_fit.task."
    with patch(ns + "parse_arguments", return_value=Mock(draw_id=sentinel.DRAW_ID,
                                                         ode_version=sentinel.ODE_VERSION)), \
            patch(ns + "run_ode_fit") as mocked_run_ode_fit:
        main()

    assert mocked_run_ode_fit.called_once_with(draw_id=sentinel.DRAW_ID,
                                               ode_version=sentinel.ODE_VERSION)


class Test_run_ode_fit:
    """
    Tests unit task of SEIIR pipeline "fit" stage.

    run_ode_fit is an entry point of the application and is meant to run at the
    per-draw level. Running this once for each draw represents the
    non-bookkeeping work done in the entire pipeline "fit" stage.
    """
    def test_happy_path(self, draw_id, ode_version, data_interface, ode_model):
        """
        Test happy path logic where nothing fails.

        This is primarily a test of ODEDataInterface usage, as it has
        responsibilities for both loading and persisting data.
        """
        # run imperative function
        run_ode_fit(draw_id=draw_id, ode_version=ode_version)

        # data_interface should have loaded location data
        data_interface.load_location_ids.assert_called_once()
        data_interface.load_all_location_data.assert_called_once_with(
            location_ids=sentinel.LOCATION_IDS,
            draw_id=draw_id)

        # UNTESTED - ODEProcessInput is created and used to initialize an ODEProcess

        # ode_model.process() is called
        ode_model.process.assert_called_once()

        # result DataFrame must be created and saved
        ode_model.create_result_df.assert_called_once()
        data_interface.save_beta_fit_file.assert_called_once_with(
            ode_model.create_result_df(), draw_id)

        # parameters dataframe must be created and saved
        ode_model.create_params_df.assert_called_once()
        data_interface.save_draw_beta_param_file.assert_called_once_with(
            ode_model.create_params_df(), draw_id)

        # start/end date DataFrame is created and saved
        ode_model.create_start_end_date_df.assert_called_once()
        data_interface.save_draw_date_file.assert_called_once_with(
            ode_model.create_start_end_date_df(), draw_id)

    # fixtures for testing internal behavior of run_ode_fit
    @pytest.fixture(autouse=True)
    def fit_spec(self):
        """
        mock FitSpecification.from_path to avoid I/O. Could instead use fixture file.

        This mock basically just exists to allow us to ignore the details of
        creating an ODEDataInterface object.
        """
        with patch.object(FitSpecification, "from_path") as mocked_fs:
            yield mocked_fs

    @pytest.fixture
    def data_interface(self):
        """
        data_interface does a significant amount of work in this function and
        the calls to it will be highly tested.
        """
        mocked_di = MagicMock()
        mocked_di.load_location_ids.return_value = sentinel.LOCATION_IDS

        with patch.object(ODEDataInterface, "from_specification", return_value=mocked_di):
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

    # fixtures for calling run_ode_fit
    @pytest.fixture
    def draw_id(self):
        return 42

    @pytest.fixture
    def ode_version(self, tmpdir):
        """
        Return a pathlib.Path-like value where ode_version is to be saved.

        Use pytest's tmpdir construct as it is suitable and requires no effort.
        """
        return tmpdir
