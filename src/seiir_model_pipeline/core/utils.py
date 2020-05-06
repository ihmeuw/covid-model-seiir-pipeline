import pandas as pd
import numpy as np

from seiir_model.ode_model import ODEProcessInput
from slime.model import CovModel, CovModelSet
from slime.core.data import MRData

from seiir_model_pipeline.core.versioner import PEAK_DATE_FILE
from seiir_model_pipeline.core.versioner import INFECTION_COL_DICT
from seiir_model_pipeline.core.data import get_missing_locations


SEIIR_COMPARTMENTS = ['S', 'E', 'I1', 'I2', 'R']

COL_BETA = 'beta'
COL_GROUP = 'loc_id'
COL_DATE = 'date'

LOCATION_SET_ID = 111


def get_location_name_from_id(location_id, metadata_path):
    df = pd.read_csv(metadata_path)
    location_name = df.loc[df.location_id == location_id]['location_name'].iloc[0]
    return location_name


def date_to_days(date):
    date = pd.to_datetime(date)
    return np.array((date - date.min()).days)


def get_locations(directories, location_set_version_id, covariate_version):
    df = pd.read_csv(
        directories.get_location_metadata_file(location_set_version_id),
    )
    missing = get_missing_locations(
        directories=directories, location_ids=df.location_id.unique().tolist(), covariate_version=covariate_version
    )
    locations = set(df.location_id.unique().tolist()) - set(missing)
    # locations = [x for x in locations if x not in [60407, 60406, 60405]]
    return list(locations)


def get_peaked_dates_from_file():
    df = pd.read_csv(PEAK_DATE_FILE)
    return dict(zip(df.location_id, df.peak_date))


def process_ode_process_input(settings, location_data):
    """Convert to ODEProcessInput.
    """
    return ODEProcessInput(
        df_dict=location_data,
        col_date=INFECTION_COL_DICT['COL_DATE'],
        col_cases=INFECTION_COL_DICT['COL_CASES'],
        col_pop=INFECTION_COL_DICT['COL_POP'],
        col_loc_id=INFECTION_COL_DICT['COL_LOC_ID'],
        col_lag_days=INFECTION_COL_DICT['COL_ID_LAG'],
        alpha=settings.alpha,
        sigma=settings.sigma,
        gamma1=settings.gamma1,
        gamma2=settings.gamma2,
        solver_dt=settings.solver_dt,
        spline_options={
            'spline_knots': np.array(settings.knots),
            'spline_degree': settings.degree
        },
        day_shift=settings.day_shift
    )


def convert_to_covmodel(cov_dict, cov_order_list):
    cov_models = []
    cov_names = []
    for name, dct in cov_dict.items():
        cov_names.append(name)
        cov_models.append(CovModel(
            name,
            use_re=dct['use_re'],
            bounds=np.array(dct['bounds']),
            gprior=np.array(dct['gprior']),
            re_var=dct['re_var'],
        ))
    cov_names_id = {name: i for i, name in enumerate(cov_names)}
    
    ordered_covmodel_sets = []
    for names in cov_order_list:
        ordered_covmodel_sets.append(
            CovModelSet([cov_models[cov_names_id[name]] for name in names])
        )
    all_covmodels_set = CovModelSet(cov_models)
    return ordered_covmodel_sets, all_covmodels_set


def convert_inputs_for_beta_model(data_cov, df_beta, covmodel_set):
    df_cov, col_t_cov, col_group_cov = data_cov
    df = df_beta.merge(
        df_cov,
        left_on=[COL_DATE, COL_GROUP],
        right_on=[col_t_cov, col_group_cov],
    ).copy()
    df = df.loc[df[COL_BETA] != 0].copy()
    df.sort_values(inplace=True, by=[COL_GROUP, COL_DATE])
    df['ln_'+COL_BETA] = np.log(df[COL_BETA])
    cov_names = [covmodel.col_cov for covmodel in covmodel_set.cov_models]
    mrdata = MRData(df, col_group=COL_GROUP, col_obs='ln_'+COL_BETA, col_covs=cov_names)
    
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


def get_ode_init_cond(location_id, beta_ode_fit, current_date,
                      col_components=None):
    """Get the initial condition for the ODE.

    Args:
        location_id (init):
            Location ids.
        beta_ode_fit (str | pd.DataFrame):
            The result for the beta_ode_fit, either file or path to file.
        current_date (str | np.datetime64):
            Current date for each location we try to predict off. Either file
            or path to file.


    Returns:
         pd.DataFrame: Initial conditions by location.
    """
    # process input
    beta_ode_fit = get_df(beta_ode_fit)
    assert (COL_GROUP in beta_ode_fit)
    assert (COL_DATE in beta_ode_fit)
    beta_ode_fit = beta_ode_fit[beta_ode_fit[COL_GROUP] == location_id].copy()

    if isinstance(current_date, str):
        current_date = np.datetime64(current_date)
    else:
        assert isinstance(current_date, np.datetime64)

    dt = np.abs((pd.to_datetime(beta_ode_fit[COL_DATE]) - current_date).dt.days)
    beta_ode_fit = beta_ode_fit.iloc[np.argmin(dt)]

    if col_components is None:
        col_components = SEIIR_COMPARTMENTS
    else:
        assert isinstance(col_components, list)
    assert all([c in beta_ode_fit for c in col_components])

    return beta_ode_fit[col_components].values.ravel()
