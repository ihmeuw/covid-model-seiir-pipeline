import pandas
import pytest

from covid_model_seiir_pipeline.lib.io import RegressionRoot
from covid_model_seiir_pipeline.lib.io.marshall import (
    CSVMarshall,
)


class MarshallInterfaceTests:
    """
    Mixin class for testing the marshall interface.
    """

    def test_parameters_marshall(self, instance, regression_root, parameters):
        self.assert_load_dump_workflow_correct(instance, regression_root,
                                               parameters, key=regression_root.parameters(draw_id=4))

    def test_date_marshall(self, instance, regression_root, dates):
        self.assert_load_dump_workflow_correct(instance, regression_root,
                                               dates, key=regression_root.dates(draw_id=4))

    def test_coefficients_marshall(self, instance, regression_root, coefficients):
        self.assert_load_dump_workflow_correct(instance, regression_root,
                                               coefficients, key=regression_root.coefficients(draw_id=4))

    def test_regression_beta_marshall(self, instance, regression_root, regression_beta):
        self.assert_load_dump_workflow_correct(instance, regression_root,
                                               regression_beta, key=regression_root.beta(draw_id=4))

    def test_location_data_marshall(self, instance, regression_root, location_data):
        self.assert_load_dump_workflow_correct(instance, regression_root,
                                               location_data, key=regression_root.infection_data(draw_id=4))

    def test_no_overwriting(self, instance, regression_root, parameters):
        self.assert_load_dump_workflow_correct(instance, regression_root,
                                               parameters, key=regression_root.parameters(draw_id=4))

    def test_interface_methods(self, instance):
        "Test mandatory interface methods exist."
        assert hasattr(instance, "dump")
        assert hasattr(instance, "load")
        assert hasattr(instance, "exists")

    def assert_load_dump_workflow_correct(self, instance, regression_root, data, key):
        "Helper method for testing load/dump marshalling does not change data."
        instance.touch(*regression_root.terminal_paths())
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
        return RegressionRoot(tmpdir)


    @pytest.fixture
    def instance(self):
        return CSVMarshall
