from typing import List

import pandas as pd
import numpy as np

from covid_model_seiir.ode_model import ODEProcessInput
from slime.model import CovModel, CovModelSet
from slime.core.data import MRData

from covid_model_seiir_pipeline.regression.specification import CovariateSpecification
from covid_model_seiir_pipeline.core.versioner import INFECTION_COL_DICT

SEIIR_COMPARTMENTS = ['S', 'E', 'I1', 'I2', 'R']

COL_BETA = 'beta'
COL_GROUP = 'loc_id'
COL_DATE = 'date'
COL_INTERCEPT = 'intercept'

LOCATION_SET_ID = 111


def process_ode_process_input(settings, location_data):
    """
    Convert to ODEProcessInput.
    """
    locs_na = []
    locs_neg = []
    for loc, df in location_data.items():
        if df[INFECTION_COL_DICT['COL_CASES']].isna().any():
            locs_na.append(loc)
        if (df[INFECTION_COL_DICT['COL_CASES']].to_numpy() < 0.0).any():
            locs_neg.append(loc)
    if len(locs_na) > 0 and len(locs_neg) > 0:
        raise ValueError(
            'NaN in infection data: ' + str(locs_na) + '. Negatives in infection data: ' + str(locs_neg)
        )
    if len(locs_na) > 0:
        raise ValueError('NaN in infection data: ' + str(locs_na))
    if len(locs_neg) > 0:
        raise ValueError('Negatives in infection data:' + str(locs_neg))

    return ODEProcessInput(
        df_dict=location_data,
        col_date=INFECTION_COL_DICT['COL_DATE'],
        col_cases=INFECTION_COL_DICT['COL_CASES'],
        col_pop=INFECTION_COL_DICT['COL_POP'],
        col_loc_id=INFECTION_COL_DICT['COL_LOC_ID'],
        col_lag_days=INFECTION_COL_DICT['COL_ID_LAG'],
        col_observed=INFECTION_COL_DICT['COL_OBS_DEATHS'],
        alpha=settings.alpha,
        sigma=settings.sigma,
        gamma1=settings.gamma1,
        gamma2=settings.gamma2,
        solver_dt=settings.solver_dt,
        spline_options={
            'spline_knots': np.array(settings.knots),
            'spline_degree': settings.degree,
            'prior_spline_convexity': None if not settings.concavity else 'concave',
            'prior_spline_monotonicity': None if not settings.increasing else 'increasing',
            'spline_knots_type': settings.spline_knots_type,
            'spline_r_linear': settings.spline_r_linear,
            'spline_l_linear': settings.spline_l_linear,
        },
        day_shift=settings.day_shift,
        spline_se_power=settings.spline_se_power,
        spline_space=settings.spline_space
    )


def convert_to_covmodel(cov_dict, cov_order_list):
    """
    Based on a covariate dictionary and an ordered list of lists of covariates,
    create a CovModelSet.

    :param cov_dict:
    :param cov_order_list:
    :return:
    """
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


def convert_to_covmodel_regression(covariates: List[CovariateSpecification]):
    """
    Based on a list of `CovariateSpecification`s and an ordered list of lists of covariate
    names, create a CovModelSet.
    """

    # construct each CovModel independently. add to dict of list by covariate order
    cov_models = []
    cov_model_order_dict = {}
    for covariate in covariates:
        cov_model = CovModel(
            covariate.name,
            use_re=covariate.use_re,
            bounds=np.array(covariate.bounds),
            gprior=np.array(covariate.gprior),
            re_var=covariate.re_var,
        )
        cov_models.append(cov_model)
        ordered_cov_set = cov_model_order_dict.get(covariate.order, [])
        cov_model_order_dict[covariate.order] = ordered_cov_set.append(cov_model)

    # constuct a CovModelSet for each order
    ordered_covmodel_sets = []
    cov_orders = list(cov_model_order_dict.keys())
    for order in cov_orders.sort():
        ordered_covmodel_sets.append(
            CovModelSet(cov_model_order_dict.order)
        )

    # constuct a CovModelSet for all
    all_covmodels_set = CovModelSet(cov_models)
    return ordered_covmodel_sets, all_covmodels_set


def convert_inputs_for_beta_model(data_cov, df_beta, covmodel_set):
    """
    Convert inputs for the beta regression model.

    :param data_cov: covariate specifications
    :param df_beta: data frame with beta outputs from the spline
    :param covmodel_set: set for a covariate model
    :return: MRData object
    """
    df_cov, col_t_cov, col_group_cov = data_cov
    df = df_beta.merge(
        df_cov,
        left_on=[COL_DATE, COL_GROUP],
        right_on=[col_t_cov, col_group_cov],
    ).copy()
    df = df.loc[df[COL_BETA] != 0].copy()
    df.sort_values(inplace=True, by=[COL_GROUP, COL_DATE])
    df['ln_' + COL_BETA] = np.log(df[COL_BETA])
    cov_names = [covmodel.col_cov for covmodel in covmodel_set.cov_models]
    covs_na = []
    for name in cov_names:
        if name != COL_INTERCEPT:
            if df[name].isna().values.any():
                covs_na.append(name)
    if len(covs_na) > 0:
        raise ValueError('NaN in covariate data: ' + str(covs_na))

    mrdata = MRData(df, col_group=COL_GROUP, col_obs='ln_' + COL_BETA, col_covs=cov_names)

    return mrdata


def get_df(file):
    """
    Get the data frame.

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
    """
    Get the initial condition for the ODE.

    Args:
        location_id (init):
            Location ids.
        beta_ode_fit (str | pd.DataFrame):
            The result for the beta_ode_fit, either file or path to file.
        current_date (str | np.datetime64):
            Current date for each location we try to predict off. Either file
            or path to file.
        col_components (List[str] | None): the column names of the ODE components, by default SEIIR model


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
