import pathlib

import pandas
import pytest

from covid_model_seiir_pipeline.marshall import (
    Keys,
    CSVMarshall,
    ZipMarshall,
)


class MarshallInterfaceTests:
    """
    Mixin class for testing the marshall interface.
    """
    def test_beta_marshall(self, instance, fit_beta):
        self.assert_load_dump_workflow_correct(instance, fit_beta, key=Keys.fit_beta(4))

    def test_parameters_marshall(self, instance, parameters):
        self.assert_load_dump_workflow_correct(instance, parameters, key=Keys.parameter(4))

    def test_date_marshall(self, instance, dates):
        self.assert_load_dump_workflow_correct(instance, dates, key=Keys.date(4))

    def test_coefficients_marshall(self, instance, coefficients):
        self.assert_load_dump_workflow_correct(instance, coefficients, key=Keys.coefficient(4))

    def test_no_overwriting(self, instance, fit_beta, parameters):
        self.assert_no_accidental_overwrites(instance, fit_beta, key=Keys.fit_beta(4))
        self.assert_no_accidental_overwrites(instance, parameters, key=Keys.parameter(4))

    def test_interface_methods(self, instance):
        "Test mandatory interface methods exist."
        assert hasattr(instance, "dump")
        assert hasattr(instance, "load")
        assert hasattr(instance, "resolve_key")

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
