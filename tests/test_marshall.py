import pathlib

import numpy
import pandas
import pytest

from covid_model_seiir_pipeline.marshall import (
    Keys,
    CSVMarshall,
    Hdf5Marshall,
    ZipMarshall,
    DtypeMarshall,
)


class MarshallInterfaceTests:
    """
    Mixin class for testing the marshall interface.
    """
    def test_parameters_marshall(self, instance, parameters):
        self.assert_load_dump_workflow_correct(instance, parameters, key=Keys.parameter(4))

    def test_date_marshall(self, instance, dates):
        self.assert_load_dump_workflow_correct(instance, dates, key=Keys.date(4))

    def test_coefficients_marshall(self, instance, coefficients):
        self.assert_load_dump_workflow_correct(instance, coefficients, key=Keys.coefficient(4))

    def test_beta_scales_marshall(self, instance, beta_scales):
        self.assert_load_dump_workflow_correct(instance, beta_scales, key=Keys.beta_scales(scenario='happy', draw_id=4))

    def test_components_marshall(self, instance, components):
        self.assert_load_dump_workflow_correct(instance, components, key=Keys.components(scenario='happy', draw_id=4))

    def test_no_overwriting(self, instance, parameters):
        self.assert_no_accidental_overwrites(instance, parameters, key=Keys.parameter(4))

    def test_interface_methods(self, instance):
        "Test mandatory interface methods exist."
        assert hasattr(instance, "dump")
        assert hasattr(instance, "load")
        assert hasattr(instance, "resolve_key")
        assert hasattr(instance, "from_paths")

    def assert_load_dump_workflow_correct(self, instance, data, key):
        "Helper method for testing load/dump marshalling does not change data."
        assert instance.dump(data, key=key) is None, ".dump() returns non-None value"
        loaded = instance.load(key=key)

        pandas.testing.assert_frame_equal(data, loaded)

    def assert_no_accidental_overwrites(self, instance, data, key):
        "Test overwriting data implicitly is not supported."
        instance.dump(data, key=key)
        with pytest.raises(LookupError):
            instance.dump(data, key=key)


class TestCSVMarshall(MarshallInterfaceTests):
    @pytest.fixture
    def instance(self, tmpdir):
        return CSVMarshall(pathlib.Path(tmpdir))


class TestZipMarshall(MarshallInterfaceTests):
    @pytest.fixture
    def instance(self, tmpdir):
        zip_path = str(tmpdir / "data.zip")
        return ZipMarshall(zip_path)


class TestHdf5Marshall(MarshallInterfaceTests):
    @pytest.fixture
    def instance(self, tmpdir):
        hdf5_path = str(tmpdir / "data.hdf")
        return Hdf5Marshall(hdf5_path)


class TestHdf5Marshall_noniface:
    @pytest.fixture
    def instance(self, tmpdir):
        hdf5_path = str(tmpdir / "data.hdf")
        return Hdf5Marshall(hdf5_path)

    def test_datetime(self, instance, regression_beta):
        """
        Re-use regression_Beta fixture but cast date as a datetime.
        """
        regression_beta['date'] = pandas.to_datetime(regression_beta['date'])
        key = Keys.regression_beta(4)

        instance.dump(regression_beta, key=key)
        loaded = instance.load(key)

        pandas.testing.assert_frame_equal(regression_beta, loaded)


class TestDtypeMarshall:
    """
    Test marshalling data from one dtype to another.

    Normally these are identical as pandas is built on top of numpy.
    Differences are noted in tests.

    https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#dtypes
    https://numpy.org/doc/stable/reference/arrays.dtypes.html

    There is an excellent diagram of the numpy dtype hierarchy here:
    https://numpy.org/doc/stable/reference/arrays.scalars.html
    """
    @pytest.fixture
    def instance(self):
        return DtypeMarshall()

    @pytest.mark.parametrize(["pd_dtype", "hdf_dtype", "dtype"], [
        [numpy.datetime64("2020-07-07"), numpy.int64(1594080000), "datetime64"]
    ])
    def test_type_marshall_pandas_hdf(self, instance, pd_dtype, hdf_dtype, dtype):
        """
        Test marshalling between pandas and hdf (h5py) dtypes.
        """
        p = pandas.Series(pd_dtype)
        n = numpy.array([hdf_dtype])

        numpy.testing.assert_array_equal(instance.as_hdf_dtype(p), n)
        pandas.testing.assert_series_equal(instance.as_pandas_dtype(n, dtype), p)


def test_CSVMarshall_uses_same_paths_as_paths_module(tmpdir):
    """
    Runs the gamut of paths produced by paths.py and compares to CSVMarshall
    path computation.
    """
    from covid_model_seiir_pipeline.paths import (
        RegressionPaths,
        ForecastPaths,
    )

    root = pathlib.Path(tmpdir)
    m = CSVMarshall(root)
    r = RegressionPaths(root)
    f = ForecastPaths(root, scenarios=["happy"])

    dargs = {'draw_id': 4}
    sdargs = {'scenario': "happy", 'draw_id': 4}

    # forecasting/data.py
    # load_regression_coefficients
    assert r.get_coefficient_file(**dargs) == m.resolve_key(Keys.coefficient(**dargs))
    # load_dates_df
    assert r.get_date_file(**dargs) == m.resolve_key(Keys.date(**dargs))
    # load_beta_regression
    assert r.get_beta_regression_file(**dargs) == m.resolve_key(Keys.regression_beta(**dargs))
    # load_infection_data
    assert r.get_infection_file(**dargs) == m.resolve_key(Keys.location_data(**dargs))
    # load_beta_params
    assert r.get_beta_param_file(**dargs) == m.resolve_key(Keys.parameter(**dargs))
    # save_components
    assert f.get_components_path(**sdargs) == m.resolve_key(Keys.components(**sdargs))
    # save_beta_scales
    assert f.get_beta_scaling_path(**sdargs) == m.resolve_key(Keys.beta_scales(**sdargs))
    # save_outputs
    assert f.get_outputs_path(**sdargs) == m.resolve_key(Keys.forecast_outputs(**sdargs))

    # regression/data.py
    # save_beta_param_file
    assert r.get_beta_param_file(**dargs) == m.resolve_key(Keys.parameter(**dargs))
    # save_date_file
    assert r.get_date_file(**dargs) == m.resolve_key(Keys.date(**dargs))
    # save_regression_coefficients
    assert r.get_coefficient_file(**dargs) == m.resolve_key(Keys.coefficient(**dargs))
    # save_regression_betas
    assert r.get_beta_regression_file(**dargs) == m.resolve_key(Keys.regression_beta(**dargs))
    # save_location_data
    assert r.get_data_file(**dargs) == m.resolve_key(Keys.location_data(**dargs))
