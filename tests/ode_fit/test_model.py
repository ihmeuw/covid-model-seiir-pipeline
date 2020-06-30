import numpy
import pandas
import pytest

from covid_model_seiir_pipeline.ode_fit import model
from covid_model_seiir_pipeline.static_vars import INFECTION_COL_DICT


class Test_ODEProcess_results:
    """
    Black-box test for ODEProcess to ensure outputs remain consistent.
    """

    # ignore a very specific RuntimeWarning
    @pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide:RuntimeWarning")  # noqa
    def test_create_params_df(self, ode_model):
        """
        Test outputs from ode_model.process remain good.
        """
        params = ode_model.create_params_df()

        expected = pandas.DataFrame([
            ['alpha', 0.9967029839013677],
            ['sigma', 0.2729460588151238],
            ['gamma1', 0.5],
            ['gamma2', 0.809867822982699],
            ['day_shift', 5.0],
        ], columns=['params', 'values'])

        assert expected.equals(params)

    # ignore a very specific RuntimeWarning
    @pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide:RuntimeWarning")  # noqa
    def test_create_result_df(self, ode_model):
        beta = ode_model.create_result_df()

        assert beta.shape == (225, 11)

        # pick arbitrary rows to test
        mask1 = (beta.location_id == 10) & (beta.date == "2020-03-02")
        r1 = beta.loc[mask1, :].iloc[0].to_dict()
        e1 = {
            'location_id': 10,
            'date': '2020-03-02',
            'days': 3,
            'beta': 0.7805482566415091,
            'S': 16603117.700039685,
            'E': 0.00024148647168971383,
            'I1': 8.662533567671615e-05,
            'I2': 3.626634800151582e-05,
            'R': 4.0834675478217315e-05,
            'newE': 9.88132445500906e-05,
            'newE_obs': 9.88132445500906e-05,
        }
        assert r1 == e1

        mask2 = (beta.location_id == 60887) & (beta.date == "2020-06-10")
        r2 = beta.loc[mask2, :].iloc[0].to_dict()
        e2 = {
            'location_id': 60887,
            'date': '2020-06-10',
            'days': 111,
            'beta': 0.310171949194388,
            'S': 3682620.648469025,
            'E': 1668.104871971228,
            'I1': 921.078728218612,
            'I2': 572.4297222704859,
            'R': 56163.725831125324,
            'newE': 445.045338509743,
            'newE_obs': 445.045338509743,
        }
        assert r2 == e2

    @pytest.fixture(scope="module")
    def ode_model(self, location_data):
        """
        Return an ODEProcess model ready to have its outputs tested.
        """
        numpy.random.seed(4)  # make outputs deterministic

        inputs = model.ODEProcessInput(
            df_dict=location_data,
            col_date=INFECTION_COL_DICT['COL_DATE'],
            col_cases=INFECTION_COL_DICT['COL_CASES'],
            col_pop=INFECTION_COL_DICT['COL_POP'],
            col_loc_id=INFECTION_COL_DICT['COL_LOC_ID'],
            col_lag_days=INFECTION_COL_DICT['COL_ID_LAG'],
            col_observed=INFECTION_COL_DICT['COL_OBS_DEATHS'],
            alpha=[0.9, 1.0],
            sigma=[0.2, 0.3333],
            gamma1=[0.5, 0.5],
            gamma2=[0.3333, 1.0],
            solver_dt=0.1,
            day_shift=[0, 8],
        )
        result = model.ODEProcess(inputs)
        result.process()
        return result

    @pytest.fixture(scope="module")
    def location_data(self, fixture_dir):
        """
        Returns location data to run ode_fit on.

        This is a dict of {location_id: DataFrame_of_data} pairs.
        """
        fixture_path = fixture_dir / "ode_fit/location_data_2020_06_28.02.csv"
        big_df = pandas.read_csv(fixture_path)
        return {loc_id: loc_data for loc_id, loc_data in big_df.groupby("loc_id")}
