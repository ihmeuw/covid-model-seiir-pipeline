from pathlib import Path

import numpy
import pandas
import pytest

from covid_model_seiir_pipeline.ode_fit import model
from covid_model_seiir_pipeline.static_vars import INFECTION_COL_DICT


class Test_ODEProcess_results:
    """
    Black-box test for ODEProcess to ensure outputs remain consistent.
    """
    location_data_root = "/ihme/covid-19/seir-inputs"
    # NOTE: draw and version_result are tied together - you cannot modify draw
    # without updating all result values
    draw = 861
    version_result = {
        "2020_06_28.02": pandas.DataFrame([
            ['alpha', 0.9967029839013677],
            ['sigma', 0.2729460588151238],
            ['gamma1', 0.5],
            ['gamma2', 0.809867822982699],
            ['day_shift', 5.0],
        ], columns=['params', 'values']),
    }

    @pytest.mark.parametrize(["location_data_version", "expected"],
                             list(version_result.items()),
                             indirect=["location_data_version"])
    # ignore a very specific RuntimeWarning
    @pytest.mark.filterwarnings("ignore:invalid value encountered in true_divide:RuntimeWarning")  # noqa
    def test_create_params_df(self, ode_model, expected):
        """
        Test outputs from ode_model.process remain good.
        """
        params = ode_model.create_params_df()

        assert expected.equals(params)

    @pytest.fixture
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

    @pytest.fixture
    def location_data_version(self, request):
        try:
            return request.param
        except AttributeError:
            pytest.fail("Must parametrize location_data_version to run this test")

    @pytest.fixture
    def location_data(self, fixture_dir, location_data_version):
        """
        Returns location data to run ode_fit on.

        This is a dict of {location_id: DataFrame_of_data} pairs.

        This method serves double-duty for generating it's own input fixtures.
        """
        fixture_path = fixture_dir / "ode_fit" / f"location_data_{location_data_version}.csv"

        try:
            big_df = pandas.read_csv(fixture_path)
        except FileNotFoundError:
            result = self._build_location_data_from_source(location_data_version)

            # save fixture for future use
            pandas.concat(result.values()).to_csv(fixture_path)

            return result
        else:
            return {loc_id: loc_data for loc_id, loc_data in big_df.groupby("loc_id")}

    def _build_location_data_from_source(self, location_data_version):
        """
        Load location data from source directory.

        This is a helper method to automatically generate test fixture data
        that has not been prepared ahead of time.

        Note: several data files exist that are not interesting - we only load
        those data files which are interesting.
        """
        def is_interesting(df):
            return 'obs_deaths' in df

        result = {}

        root = Path(f"{self.location_data_root}/{location_data_version}/")
        if not root.exists():
            pytest.fail(f"Cannot build data for {location_data_version} - "
                        f"path {root} does not exist!")

        for f in root.glob(f"*/draw{self.draw:04d}_prepped_deaths_and_cases_all_age.csv"):
            # f.parent is e.g., 'Kerala_4857'. Get location_id (4857)
            location_id = int(f.parent.name.rsplit("_", 1)[1])
            df = pandas.read_csv(f)
            if is_interesting(df):
                result[location_id] = df

        return result
