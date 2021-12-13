from typing import Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd


from covid_model_seiir_pipeline.lib import math
from covid_model_seiir_pipeline.lib.ode_mk2.containers import (
    Parameters,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    REPORTED_EPI_MEASURE_NAMES,
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
    beta = beta_regression.reindex(indices.past).loc[:, 'beta'].append(beta).sort_index()
    beta.loc[pd.IndexSlice[:, beta_scale[1]:]] *= beta_scale[0]
    return beta, beta_hat.set_index(['location_id', 'date']).reindex(beta.index)


def build_model_parameters(indices: Indices,
                           beta: pd.Series,
                           posterior_epi_measures: pd.DataFrame,
                           prior_ratios: pd.DataFrame,
                           ode_parameters: pd.Series,
                           rhos: pd.DataFrame,
                           vaccinations: pd.DataFrame,
                           etas: pd.DataFrame,
                           phis: pd.DataFrame) -> Parameters:
    keep = ['alpha', 'sigma', 'gamma', 'pi', 'kappa']
    ode_params = pd.DataFrame(
        {key: value for key, value in ode_parameters.to_dict().items() if key.split('_')[0] in keep},
        index=indices.full
    )
    ode_params.loc[:, 'beta_all_infection'] = beta
    measure_map = {
        'death': ('deaths', 'ifr'),
        'admission': ('hospitalizations', 'ihr'),
        'case': ('cases', 'idr')
    }

    posterior_epi_measures = posterior_epi_measures.loc[posterior_epi_measures['round'] == 2]
    scalars = []
    for epi_measure in REPORTED_EPI_MEASURE_NAMES:
        ode_params.loc[:, f'count_all_{epi_measure}'] = -1
        ode_params.loc[:, f'weight_all_{epi_measure}'] = -1
        lag = ode_parameters.loc[f'exposure_to_{epi_measure}']
        infections = (posterior_epi_measures
                      .loc[:, 'daily_naive_unvaccinated_infections']
                      .reindex(indices.full)
                      .groupby('location_id')
                      .shift(lag))
        ratio_measure, ratio_name = measure_map[epi_measure]
        numerator = posterior_epi_measures.loc[:, f'daily_{ratio_measure}'].reindex(indices.full)
        prior_ratio = prior_ratios.loc[:, ratio_name].groupby('location_id').last()
        ode_params.loc[:, f'rate_all_{epi_measure}'] = build_ratio(infections, numerator, prior_ratio)

        for risk_group in RISK_GROUP_NAMES:
            scalars.append(
                (prior_ratio[f'{ratio_name}_{risk_group}'] / prior_ratio[ratio_name])
                .rename(f'{epi_measure}_{risk_group}')
                .reindex(indices.full)
                .groupby('location_id')
                .ffill()
                .groupby('location_id')
                .bfill()
            )
    scalars = pd.concat(scalars, axis=1)

    rhos = rhos.reindex(indices.full, fill_value=0.)
    rhos.columns = [f'rho_{c}_infection' for c in rhos.columns]
    rhos.loc[:, 'rho_none_infection'] = 0
    base_parameters = pd.concat([ode_params, rhos], axis=1)

    vaccinations = vaccinations.reindex(indices.full, fill_value=0.)
    etas = etas.sort_index().reindex(indices.full, fill_value=0.)

    return Parameters(
        base_parameters=base_parameters,
        vaccinations=vaccinations,
        age_scalars=scalars,
        etas=etas,
        phis=phis,
    )


def build_ratio(shifted_infections: pd.Series, numerator: pd.Series, prior_ratio: pd.Series):
    posterior_ratio = (numerator / shifted_infections).rename('value')
    posterior_ratio.loc[(posterior_ratio == 0) | ~np.isfinite(posterior_ratio)] = np.nan    
    locs = posterior_ratio.reset_index().location_id.unique()    
    for location_id in locs:
        count = posterior_ratio.loc[location_id].notnull().sum()
        if not count:
            posterior_ratio.loc[location_id] = prior_ratio.loc[location_id]
    
    pr_gb = posterior_ratio.dropna().reset_index().groupby('location_id')
    date = pr_gb.date.last()
    final_posterior_ratio = pr_gb.value.last()
    lr_ratio = posterior_ratio.dropna().groupby('location_id').apply(lambda x: x.iloc[-60:].mean())

    window = 60
    scale = (lr_ratio - final_posterior_ratio) / window
    t = pd.Series(np.tile(np.arange(window + 1), len(scale)), 
                  index=pd.MultiIndex.from_product((locs, np.arange(window+1)), 
                                                   names=('location_id', 't')))
    rate_scaleup = (final_posterior_ratio + scale * t).rename('value').reset_index(level='t')
    rate_scaleup['date'] = pd.to_timedelta(rate_scaleup['t'], unit='D') + date
    rate_scaleup = rate_scaleup.set_index('date', append=True).value
    ratio = (posterior_ratio
             .drop(posterior_ratio.index.intersection(rate_scaleup.index))
             .append(rate_scaleup)
             .sort_index()
             .reindex(shifted_infections.index)
             .groupby('location_id')
             .ffill()
             .groupby('location_id')
             .bfill())

    return ratio



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

# def build_postprocessing_parameters(indices: Indices,
#                                     past_infections: pd.Series,
#                                     past_deaths: pd.Series,
#                                     ratio_data,
#                                     model_parameters: Parameters,
#                                     correction_factors,
#                                     hospital_parameters) -> PostprocessingParameters:
#     ratio_data = correct_ratio_data(indices, ratio_data, model_parameters)
#
#     correction_factors = forecast_correction_factors(
#         indices,
#         correction_factors,
#         hospital_parameters,
#     )
#
#     return PostprocessingParameters(
#         past_infections=past_infections,
#         past_deaths=past_deaths,
#         **ratio_data.to_dict(),
#         **correction_factors.to_dict()
#     )


#
# def forecast_correction_factors(indices: Indices,
#                                 correction_factors,
#                                 hospital_parameters):
#     averaging_window = pd.Timedelta(days=hospital_parameters.correction_factor_average_window)
#     application_window = pd.Timedelta(days=hospital_parameters.correction_factor_application_window)
#
#     new_cfs = {}
#     for cf_name, cf in correction_factors.to_dict().items():
#         cf = cf.reindex(indices.full)
#         loc_cfs = []
#         for loc_id, loc_today in indices.initial_condition.tolist():
#             loc_cf = cf.loc[loc_id]
#             mean_cf = loc_cf.loc[loc_today - averaging_window: loc_today].mean()
#             loc_cf.loc[loc_today:] = np.nan
#             loc_cf.loc[loc_today + application_window:] = mean_cf
#             loc_cf = loc_cf.interpolate().reset_index()
#             loc_cf['location_id'] = loc_id
#             loc_cfs.append(loc_cf.set_index(['location_id', 'date'])[cf_name])
#         new_cfs[cf_name] = pd.concat(loc_cfs).sort_index()
#     return HospitalCorrectionFactors(**new_cfs)


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
