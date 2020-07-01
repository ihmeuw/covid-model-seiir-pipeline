import pathlib

import pandas
import pytest

from covid_model_seiir_pipeline.marshall import (
    Keys,
    CSVMarshall,
)


class MarshallInterfaceTests:
    """
    Mixin class for testing the marshall interface.
    """
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


@pytest.fixture
def beta_result():
    "Example beta result from an ODE model."
    return pandas.DataFrame({
        'location_id': [10],
        'date': ['2020-03-02'],
        'days': [3],
        'beta': [0.7805482566415091],
        'S': [16603117.700039685],
        'E': [0.00024148647168971383],
        'I1': [8.662533567671615e-05],
        'I2': [3.626634800151582e-05],
        'R': [4.0834675478217315e-05],
        'newE': [9.88132445500906e-05],
        'newE_obs': [9.88132445500906e-05],
    })


@pytest.fixture
def parameters():
    "Example parameters data."
    return pandas.DataFrame([
        ['alpha', 0.9967029839013677],
        ['sigma', 0.2729460588151238],
        ['gamma1', 0.5],
        ['gamma2', 0.809867822982699],
        ['day_shift', 5.0],
    ], columns=['params', 'values'])


@pytest.fixture
def dates():
    "Example dates data."
    return pandas.DataFrame([
        [523, '2020-03-06', '2020-05-05'],
        [526, '2020-03-08', '2020-05-05'],
        [533, '2020-02-23', '2020-05-05'],
        [537, '2020-02-26', '2020-05-05'],
    ], columns=['loc_id', 'start_date', 'end_date'])


class TestCSVMarshall(MarshallInterfaceTests):
    @pytest.fixture
    def instance(self, tmpdir):
        return CSVMarshall(pathlib.Path(tmpdir))

    def test_beta_marshall(self, instance, beta_result):
        self.assert_load_dump_workflow_correct(instance, beta_result, key=Keys.beta(4))

    def test_parameters_marshall(self, instance, parameters):
        self.assert_load_dump_workflow_correct(instance, parameters, key=Keys.parameter(4))

    def test_date_marshall(self, instance, dates):
        self.assert_load_dump_workflow_correct(instance, dates, key=Keys.date(4))

    def test_no_overwriting(self, instance, beta_result, parameters):
        self.assert_no_accidental_overwrites(instance, beta_result, key=Keys.beta(4))
        self.assert_no_accidental_overwrites(instance, parameters, key=Keys.parameter(4))
