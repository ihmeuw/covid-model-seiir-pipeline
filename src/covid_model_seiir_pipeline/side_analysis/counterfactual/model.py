import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib.ode_mk2.containers import (
    Parameters,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    RISK_GROUP_NAMES,
)
from covid_model_seiir_pipeline.pipeline.forecasting.model.containers import (
    Indices,
)
from covid_model_seiir_pipeline.side_analysis.counterfactual.specification import (
    CounterfactualScenarioParameters
)


def build_indices(scenario_spec: CounterfactualScenarioParameters,
                  beta: pd.Series,
                  past_compartments: pd.DataFrame):
    past_start_dates = (past_compartments
                        .reset_index(level='date')
                        .date
                        .groupby('location_id')
                        .min())

    with_covid = past_compartments.filter(like='Infection_all_all_all').sum(axis=1) > 0.
    min_start_date = (past_compartments[with_covid]
                      .reset_index(level='date')
                      .groupby('location_id')
                      .date
                      .min()) + pd.Timedelta(days=1)
    desired_start_date = pd.Series(pd.Timestamp(scenario_spec.start_date), index=min_start_date.index)
    forecast_start_dates = np.maximum(min_start_date, desired_start_date).rename('date')
    beta_fit_end_dates = forecast_start_dates.copy()
    forecast_end_dates = beta.reset_index().groupby('location_id').date.max()

    return Indices(
        past_start_dates,
        beta_fit_end_dates,
        forecast_start_dates,
        forecast_end_dates,
    )


def build_model_parameters(indices: Indices,
                           counterfactual_beta: pd.Series,
                           forecast_ode_parameters: pd.DataFrame,
                           prior_ratios: pd.DataFrame,
                           vaccinations: pd.DataFrame,
                           etas: pd.DataFrame,
                           phis: pd.DataFrame) -> Parameters:
    ode_params = (forecast_ode_parameters
                  .reindex(indices.full)
                  .groupby('location_id')
                  .ffill()
                  .groupby('location_id')
                  .bfill())
    ode_params.loc[:, 'beta_all_infection'] = (counterfactual_beta
                                               .rename('beta_all_infection')
                                               .reindex(indices.full)
                                               .groupby('location_id')
                                               .ffill()
                                               .groupby('location_id')
                                               .bfill())

    scalars = []
    ratio_map = {
        'death': 'ifr',
        'admission': 'ihr',
        'case': 'idr',
    }
    for epi_measure, ratio_name in ratio_map.items():
        for risk_group in RISK_GROUP_NAMES:
            scalars.append(
                (prior_ratios[f'{ratio_name}_{risk_group}'] / prior_ratios[ratio_name])
                .rename(f'{epi_measure}_{risk_group}')
                .reindex(indices.full)
                .groupby('location_id')
                .ffill()
                .groupby('location_id')
                .bfill()
            )
    scalars = pd.concat(scalars, axis=1)

    return Parameters(
        base_parameters=ode_params,
        vaccinations=vaccinations.reindex(indices.full, fill_value=0.),
        etas=etas.reindex(indices.full, fill_value=0.),
        age_scalars=scalars,
        phis=phis,
    )

