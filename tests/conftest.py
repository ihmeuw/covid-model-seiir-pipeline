import os
import pathlib

import pandas
import pytest


@pytest.fixture(scope="session")
def fixture_dir():
    here = pathlib.Path(__file__).parent
    return here / "fixtures"


@pytest.fixture
def tmpdir_file_count(tmpdir):
    def file_count():
        "Return count of files in our ODEDataInterface storage location."
        cnt = 0
        for _, _, files in os.walk(tmpdir):
            for _ in files:
                cnt += 1
        return cnt

    return file_count


# Data fixtures
#
# These represent actual data in various parts of the system that needs to be
# marshalled/unmarshalled
@pytest.fixture
def parameters():
    "Example parameters data from an ODE model."
    return pandas.DataFrame([
        ['alpha', 0.9967029839013677],
        ['sigma', 0.2729460588151238],
        ['gamma1', 0.5],
        ['gamma2', 0.809867822982699],
        ['day_shift', 5.0],
    ], columns=['params', 'values'])


@pytest.fixture
def dates():
    "Example dates data from an ODE model."
    return pandas.DataFrame([
        [523, '2020-03-06', '2020-05-05'],
        [526, '2020-03-08', '2020-05-05'],
        [533, '2020-02-23', '2020-05-05'],
        [537, '2020-02-26', '2020-05-05'],
    ], columns=['location_id', 'start_date', 'end_date'])


@pytest.fixture
def coefficients():
    "Example coefficients data from regression."
    return pandas.DataFrame([
        [523, -0.2356716680846281, 0.010188535227465946, 0.27234473836297995, -598.8754409180113],
        [526, -0.24334559662138652, 0.019963189989381125, 0.27234473836297995, -598.8754409180113],
        [533, -0.214475560389406, 0.011172361940536456, 0.27234473836297995, -598.8754409180113],
        [537, -0.09571280930682702, 0.011990915850960831, 0.27234473836297995, -598.8754409180113],
        [538, 0.0988105530655817, 0.0187165992693182, 0.27234473836297995, -598.8754409180113],
    ], columns=["group_id", "intercept", "mobility", "proportion_over_1k", "testing"])


@pytest.fixture
def regression_beta():
    "Example beta from regression. Not be confused with fit beta."
    return pandas.DataFrame([
        [523, "2020-03-07", 1, 2.5739807207048058, 4969159.687697555, 345.52948890230834, 99.9253559769042,
         30.788494966654106, 8.07443820131525, 253.22564174849842, 203.997904056312, 523.0, 1.0, 5.00183501163904,
         0.9999738227721054, 3.49564272841683e-07, 0.9741367332124444],
        [523, "2020-03-08", 2, 1.6254293885720643, 4968898.260275219, 483.625404827741, 158.98914338609208,
         68.82741106261548, 34.30607014785281, 269.80262968215254, 249.16085187099705, 523.0, 1.0, 4.8648570764761,
         0.9999738227721054, 5.44970658216924e-07, 0.9725766504737438],
        [523, "2020-03-09", 3, 1.1884395563959436, 4968619.718949269, 601.1713114959216, 224.2559581939473,
         115.57601170740372, 83.28614944679695, 287.46480206654036, 52.025099521832, 523.0, 1.0, 4.46264044478716,
         0.9999738227721054, 9.1305795481005e-07, 0.9681075127865911],
        [523, "2020-03-10", 4, 0.9567975916014266, 4968322.943386977, 704.4042770583011, 289.31826711,
         168.3186879239724, 159.02196884064057, 306.2831986645444, 441.972798383768, 523.0, 1.0, 3.7269486319883103,
         0.9999738227721054, 1.6100022880651301e-06, 0.9599749904639846],
    ], columns=[
        "loc_id", "date", "days", "beta", "S", "E", "I1", "I2", "R", "newE",
        "newE_obs", "location_id", "intercept", "mobility", "proportion_over_1k",
        "testing", "beta_pred",
    ])


@pytest.fixture
def location_data():
    """
    Example location data for one location.

    Normal location data would include values for many locations for a much
    larger date range.
    """
    return pandas.DataFrame([
        ["Georgia", 35, "age1", "2020-01-10", 0, 0, 0, 3664751.9351746, 17, 1, 1],
        ["Georgia", 35, "age1", "2020-01-11", 0, 0, 0, 3664751.9351746, 17, 1, 1],
        ["Georgia", 35, "age1", "2020-01-12", 0, 0, 0, 3664751.9351746, 17, 1, 1],
    ], columns=["location", "loc_id", "age", "date", "cases_draw",
                "deaths_draw", "deaths_mean", "pop", "i_d_lag", "obs_deaths",
                "obs_infecs"])


@pytest.fixture
def beta_scales():
    "Example beta_scales from forecasting. This is before re-orientation to per-location"
    return pandas.DataFrame([
        [523, 42, 19, 0.5102710680186469, 1.9536616573451457, 0.23937731744901722, -1.4297142378449315],
        [524, 42, 19, 0.337742881941994, 2.288522214755621, 0.12839967017032047, -2.05260745649655],
        [525, 42, 19, 0.4407337644202013, 1.8260965896095822, 0.27446608975078923, -1.2929275602690833],
        [526, 42, 19, 0.33158362481548936, 1.9785807689859762, 0.2106927417577173, -1.5573544069963998],

    ], columns=['location_id', "window_size", "history_days", "fit_final",
                "pred_start", "beta_ratio_mean", "beta_residual_mean"])


@pytest.fixture
def components():
    "Example component draws from forecasting."
    return pandas.DataFrame([
        [523, 4863604.1266471315, 5444.82727070014, 3143.398035154036, 2867.3620553313203, 102628.58881427224,
         0.0, 0.47457643145023976, 2.0, "2020-06-14"],
        [524, 786251.4055127738, 88.26780716536842, 52.89251926343722, 49.59755184317147, 1585.3514153472793,
         0.0, 0.3246745187254681, -25.0, "2020-06-14"],
        [525, 7052421.641838077, 11963.26570146024, 6796.7742256542615, 6089.061664358247, 172409.07925075648,
         0.0, 0.5224992562447484, 20.0, "2020-06-14"],
        [526, 3025082.20659897, 1956.2847143040935, 1210.0356364200209, 1158.6355734951346, 27941.34379648508,
         0.0, 0.35164960033097264, 0.0, "2020-06-14"],
    ], columns=["location_id", "S", "E", "I1", "I2", "R", "t", "beta", "theta", "date"])


@pytest.fixture
def forecast_outputs():
    "Example forecast outputs."
    # TODO: Mike couldn't find real data for this
    return pandas.DataFrame({'forecasts': ['good', 'ok', 'not great']})
