import pandas
import pytest

from covid_model_seiir_pipeline.io import RegressionRoot
from covid_model_seiir_pipeline.io.marshall import (
    CSVMarshall,
    ZipMarshall,
    HDF5Marshall,
)


class MarshallInterfaceTests:
    """
    Mixin class for testing the marshall interface.
    """

    def test_parameters_marshall(self, instance, regression_root, parameters):
        self.assert_load_dump_workflow_correct(instance, parameters, key=regression_root.parameters(draw_id=4))

    def test_date_marshall(self, instance, regression_root, dates):
        self.assert_load_dump_workflow_correct(instance, dates, key=regression_root.dates(draw_id=4))

    def test_coefficients_marshall(self, instance, regression_root, coefficients):
        self.assert_load_dump_workflow_correct(instance, coefficients, key=regression_root.coefficients(draw_id=4))

    def test_regression_beta_marshall(self, instance, regression_root, regression_beta):
        self.assert_load_dump_workflow_correct(instance, regression_beta, key=regression_root.beta(draw_id=4))

    def test_location_data_marshall(self, instance, regression_root, location_data):
        self.assert_load_dump_workflow_correct(instance, location_data, key=regression_root.data(draw_id=4))

    def test_no_overwriting(self, instance, regression_root, parameters):
        self.assert_load_dump_workflow_correct(instance, parameters, key=regression_root.parameters(draw_id=4))

    def test_interface_methods(self, instance):
        "Test mandatory interface methods exist."
        assert hasattr(instance, "dump")
        assert hasattr(instance, "load")
        assert hasattr(instance, "exists")

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
    def regression_root(self, tmpdir):
        return RegressionRoot(tmpdir, data_format='csv')

    @pytest.fixture
    def instance(self):
        return CSVMarshall


class TestZipMarshall(MarshallInterfaceTests):
    @pytest.fixture
    def regression_root(self, tmpdir):
        return RegressionRoot(tmpdir, data_format='zip')

    @pytest.fixture
    def instance(self):
        return ZipMarshall


class TestHdf5Marshall(MarshallInterfaceTests):
    @pytest.fixture
    def regression_root(self, tmpdir):
        return RegressionRoot(tmpdir, data_format='hdf')

    @pytest.fixture
    def instance(self):
        return HDF5Marshall


class TestHdf5Marshall_noniface:
    @pytest.fixture
    def regression_root(self, tmpdir):
        return RegressionRoot(tmpdir, data_format='hdf')

    @pytest.fixture
    def instance(self):
        return HDF5Marshall

    def test_datetime(self, instance, regression_root, regression_beta):
        """
        Re-use regression_Beta fixture but cast date as a datetime.
        """
        regression_beta['date'] = pandas.to_datetime(regression_beta['date'])
        key = regression_root.beta(draw_id=4)

        instance.dump(regression_beta, key=key)
        loaded = instance.load(key)

        pandas.testing.assert_frame_equal(regression_beta, loaded)

