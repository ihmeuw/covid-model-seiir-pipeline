from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import tqdm

from covid_model_seiir_pipeline.lib import math
from covid_model_seiir_pipeline.lib.ode_mk2.containers import (
    Parameters,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT,
    VARIANT_NAMES,
    RISK_GROUP_NAMES,
    COMPARTMENTS_NAMES,
    TRACKING_COMPARTMENTS_NAMES,
)
from covid_model_seiir_pipeline.lib.ode_mk2 import (
    solver,
)
from covid_model_seiir_pipeline.pipeline.forecasting.model.containers import (
    Indices,
    PostprocessingParameters,
)

if TYPE_CHECKING:
    # The model subpackage is a library for the pipeline stage and shouldn't
    # explicitly depend on things outside the subpackage.
    from covid_model_seiir_pipeline.pipeline.forecasting.specification import (
        ScenarioSpecification,
    )
    # Support type checking but keep the pipeline stages as isolated as possible.
    from covid_model_seiir_pipeline.pipeline.regression.specification import (
        HospitalParameters,
    )


##############################
# ODE parameter construction #
##############################

def build_beta_final(indices: Indices,
                     beta_regression: pd.DataFrame,
                     covariates: pd.DataFrame,
                     coefficients: pd.DataFrame,
                     beta_shift_parameters: pd.DataFrame,
                     log_beta_shift: Tuple[float, pd.Timestamp],
                     beta_scale: Tuple[float, pd.Timestamp]):
    log_beta_hat = math.compute_beta_hat(covariates, coefficients)
    log_beta_hat.loc[pd.IndexSlice[:, log_beta_shift[1]:]] += log_beta_shift[0]
    beta_hat = np.exp(log_beta_hat).loc[indices.future].rename('beta_hat').reset_index()

    beta = (beta_shift(beta_hat, beta_shift_parameters)
            .set_index(['location_id', 'date'])
            .beta_hat
            .rename('beta'))
    beta = beta_regression.loc[indices.past, 'beta'].append(beta).sort_index()
    beta.loc[pd.IndexSlice[:, beta_scale[1]:]] *= beta_scale[0]
    return beta, beta_hat.set_index(['location_id', 'date']).reindex(beta.index)


def build_model_parameters(indices: Indices,
                           beta: pd.Series,
                           ode_parameters: pd.DataFrame,
                           rhos: pd.DataFrame,
                           vaccinations: pd.DataFrame,
                           all_etas: pd.DataFrame,
                           natural_waning_dist: pd.Series,
                           natural_waning_matrix: pd.DataFrame) -> Parameters:
    keep_cols = ['alpha_all', 'sigma_all', 'gamma_all', 'pi_all'] + [f'kappa_{v}' for v in VARIANT_NAMES]
    ode_params = (ode_parameters
                  .reindex(indices.full)
                  .groupby('location_id')
                  .ffill()
                  .loc[:, keep_cols]
                  .to_dict('series'))
    new_e_all = pd.Series(np.nan, index=indices.full, name='new_e_all')
    rhos = rhos.reindex(indices.full, fill_value=0.).to_dict('series')
    rhos['rho_none'] = pd.Series(0., index=indices.full, name='rho_none')

    vaccinations = vaccinations.reindex(indices.full, fill_value=0.).to_dict('series')
    etas = process_etas(all_etas, indices.full)

    return Parameters(
        **ode_params,
        new_e_all=new_e_all,
        beta_all=beta,
        **rhos,
        **vaccinations,
        **etas,
        natural_waning_distribution=natural_waning_dist.loc['infection'],
        phi=natural_waning_matrix.loc[list(VARIANT_NAMES), list(VARIANT_NAMES)],
    )


def beta_shift(beta_hat: pd.DataFrame,
               beta_scales: pd.DataFrame) -> pd.DataFrame:
    """Shift the raw predicted beta to line up with beta in the past.

    This method performs both an intercept shift and a scaling based on the
    residuals of the ode fit beta and the beta hat regression in the past.

    Parameters
    ----------
        beta_hat
            Dataframe containing the date, location_id, and beta hat in the
            future.
        beta_scales
            Dataframe containing precomputed parameters for the scaling.

    Returns
    -------
        Predicted beta, after scaling (shift).

    """
    beta_hat = beta_hat.sort_values(['location_id', 'date']).set_index('location_id')
    scale_init = beta_scales['scale_init']
    scale_final = beta_scales['scale_final']
    window_size = beta_scales['window_size']

    beta_final = []
    for location_id in beta_hat.index.unique():
        if window_size is not None:
            t = np.arange(len(beta_hat.loc[location_id])) / window_size.at[location_id]
            scale = scale_init.at[location_id] + (scale_final.at[location_id] - scale_init.at[location_id]) * t
            scale[(window_size.at[location_id] + 1):] = scale_final.at[location_id]
        else:
            scale = scale_init.at[location_id]
        loc_beta_hat = beta_hat.loc[location_id].set_index('date', append=True)['beta_hat']
        loc_beta_final = loc_beta_hat * scale
        beta_final.append(loc_beta_final)

    beta_final = pd.concat(beta_final).reset_index()

    return beta_final


#######################################
# Construct postprocessing parameters #
#######################################

def build_postprocessing_parameters(indices: Indices,
                                    past_infections: pd.Series,
                                    past_deaths: pd.Series,
                                    ratio_data,
                                    model_parameters: Parameters,
                                    correction_factors,
                                    hospital_parameters) -> PostprocessingParameters:
    ratio_data = correct_ratio_data(indices, ratio_data, model_parameters)

    correction_factors = forecast_correction_factors(
        indices,
        correction_factors,
        hospital_parameters,
    )

    return PostprocessingParameters(
        past_infections=past_infections,
        past_deaths=past_deaths,
        **ratio_data.to_dict(),
        **correction_factors.to_dict()
    )


def correct_ratio_data(indices: Indices,
                       ratio_data,
                       model_params: Parameters):
    variant_prevalence = (model_params
                          .get_params()
                          .filter(like='rho')
                          .drop(columns=['rho_none', 'rho_ancestral'])
                          .sum(axis=1))
    p_start = variant_prevalence.loc[indices.initial_condition].reset_index(level='date', drop=True)
    variant_prevalence -= p_start.reindex(variant_prevalence.index, level='location_id')
    variant_prevalence[variant_prevalence < 0] = 0.0
    
    ifr_scalar = ratio_data.ifr_scalar * variant_prevalence + (1 - variant_prevalence)
    ifr_scalar = ifr_scalar.groupby('location_id').shift(ratio_data.infection_to_death).fillna(0.)
    ratio_data.ifr = ifr_scalar * _expand_rate(ratio_data.ifr, indices.full)
    ratio_data.ifr_lr = ifr_scalar * _expand_rate(ratio_data.ifr_lr, indices.full)
    ratio_data.ifr_hr = ifr_scalar * _expand_rate(ratio_data.ifr_hr, indices.full)
    
    ihr_scalar = ratio_data.ihr_scalar * variant_prevalence + (1 - variant_prevalence)
    ihr_scalar = ihr_scalar.groupby('location_id').shift(ratio_data.infection_to_admission).fillna(0.)
    ratio_data.ihr = ihr_scalar * _expand_rate(ratio_data.ihr, indices.full)

    ratio_data.idr = _expand_rate(ratio_data.idr, indices.full)
    ratio_data.ihr = _expand_rate(ratio_data.ihr, indices.full)
    return ratio_data


def _expand_rate(rate: pd.Series, index: pd.MultiIndex):
    return (rate
            .reindex(index)
            .groupby('location_id')
            .fillna(method='ffill')
            .fillna(method='bfill'))


def forecast_correction_factors(indices: Indices,
                                correction_factors,
                                hospital_parameters):
    averaging_window = pd.Timedelta(days=hospital_parameters.correction_factor_average_window)
    application_window = pd.Timedelta(days=hospital_parameters.correction_factor_application_window)

    new_cfs = {}
    for cf_name, cf in correction_factors.to_dict().items():
        cf = cf.reindex(indices.full)
        loc_cfs = []
        for loc_id, loc_today in indices.initial_condition.tolist():
            loc_cf = cf.loc[loc_id]
            mean_cf = loc_cf.loc[loc_today - averaging_window: loc_today].mean()
            loc_cf.loc[loc_today:] = np.nan
            loc_cf.loc[loc_today + application_window:] = mean_cf
            loc_cf = loc_cf.interpolate().reset_index()
            loc_cf['location_id'] = loc_id
            loc_cfs.append(loc_cf.set_index(['location_id', 'date'])[cf_name])
        new_cfs[cf_name] = pd.concat(loc_cfs).sort_index()
    return HospitalCorrectionFactors(**new_cfs)


###########
# Run ODE #
###########

def run_ode_forecast(initial_conditions: pd.DataFrame,
                     ode_parameters: Parameters):
    full_compartments, chis = solver.run_ode_model(
        initial_conditions,
        *ode_parameters.to_dfs(),
        forecast=True,
        num_cores=5,
    )
    return full_compartments, chis
