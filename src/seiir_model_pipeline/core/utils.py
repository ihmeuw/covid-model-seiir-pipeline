import pandas as pd
import numpy as np
from seiir_model.ode_model import ODEProcessInput
from seiir_model_pipeline.core.versioner import INFECTION_COL_DICT

from slime.model import CovModel, CovModelSet
from seiir_model_pipeline.core.versioner import PEAK_DATE_FILE
from slime.core.data import MRData

SEIIR_COMPARTMENTS = ['S', 'E', 'I1', 'I2', 'R']

COL_BETA = 'beta'
COL_GROUP = 'loc_id'
COL_DATE = 'date'

LOCATION_SET_ID = 111


def date_to_days(date):
    date = pd.to_datetime(date)
    return np.array((date - date.min()).days)


def get_locations(location_metadata_file):
    df = pd.read_csv(location_metadata_file)
    return df.location_id.unique().tolist()


def get_peaked_dates_from_file():
    df = pd.read_csv(PEAK_DATE_FILE)
    return dict(zip(df.location_id, df.peak_date))


def process_ode_process_input(settings, location_data, peak_data):
    """Convert to ODEProcessInput.
    """
    return ODEProcessInput(
        df_dict=location_data,
        col_date=INFECTION_COL_DICT['COL_DATE'],
        col_cases=INFECTION_COL_DICT['COL_CASES'],
        col_pop=INFECTION_COL_DICT['COL_POP'],
        col_loc_id=INFECTION_COL_DICT['COL_LOC_ID'],
        alpha=settings.alpha,
        sigma=settings.sigma,
        gamma1=settings.gamma1,
        gamma2=settings.gamma2,
        peak_date_dict=peak_data,
        day_shift=settings.day_shift,
        solver_dt=settings.solver_dt,
        spline_options={
            'spline_knots': np.array(settings.knots),
            'spline_degree': settings.degree
        }
    )


def convert_to_covmodel(cov_dict):
    cov_models = []
    for name, dct in cov_dict.items():
        cov_models.append(CovModel(
            name,
            use_re=dct['use_re'],
            bounds=np.array(dct['bounds']),
            gprior=np.array(dct['gprior']),
            re_var=dct['re_var'],
        ))
    covmodel_set = CovModelSet(cov_models)
    return covmodel_set


def convert_inputs_for_beta_model(data_cov, df_beta, covmodel_set):
    df_cov, col_t_cov, col_group_cov = data_cov
    df = df_beta.merge(
        df_cov,
        left_on=[COL_DATE, COL_GROUP],
        right_on=[col_t_cov, col_group_cov],
    ).copy()
    df.sort_values(inplace=True, by=[COL_GROUP, COL_DATE])
    cov_names = [covmodel.col_cov for covmodel in covmodel_set.cov_models]
    mrdata = MRData(df, col_group=COL_GROUP, col_obs=COL_BETA, col_covs=cov_names)

    return mrdata


def get_df(file):
    """Get the data frame.

    Args:
        file (str | pd.DataFrame):
            If file is data frame return as it is, if file is the string of
            path to the file, read the csv file as data frame.
    """
    if isinstance(file, pd.DataFrame):
        return file
    else:
        assert isinstance(file, str)
        assert file[-4:] == '.csv'
        return pd.read_csv(file)


def get_ode_init_cond(beta_ode_fit, current_date,
                      col_components=None):
    """Get the initial condition for the ODE.

    Args:
        beta_ode_fit (str | pd.DataFrame):
            The result for the beta_ode_fit, either file or path to file.
        current_date (str | pd.DataFrame):
            Current date for each location we try to predict off. Either file
            or path to file.

    Returns:
         pd.DataFrame: Initial conditions by location.
    """
    # process input
    beta_ode_fit = get_df(beta_ode_fit)
    current_date = get_df(current_date)
    if col_components is None:
        col_components = SEIIR_COMPARTMENTS
    else:
        assert isinstance(col_components, list)
    assert all([c in beta_ode_fit for c in col_components])
    assert (COL_GROUP in beta_ode_fit) and (COL_GROUP in current_date)
    assert (COL_DATE in beta_ode_fit) and (COL_DATE in current_date)

    beta_ode_fit = beta_ode_fit[[COL_GROUP, COL_DATE] + col_components].copy()
    current_date = current_date[[COL_GROUP, COL_DATE]].copy()
    return pd.merge(current_date, beta_ode_fit, how='left')
