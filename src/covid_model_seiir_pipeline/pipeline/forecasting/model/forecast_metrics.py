from typing import Dict, List, Tuple, Union, TYPE_CHECKING

import pandas as pd

from covid_model_seiir_pipeline.pipeline.forecasting.model.containers import (
    RatioData,
    CompartmentInfo,
    HospitalCorrectionFactors,
    HospitalMetrics,
    OutputMetrics,
)
from covid_model_seiir_pipeline.pipeline.regression.model import (
    compute_hospital_usage,
)


if TYPE_CHECKING:
    # Support type checking but keep the pipeline stages as isolated as possible.
    from covid_model_seiir_pipeline.pipeline.regression.specification import (
        HospitalParameters,
    )


def compute_output_metrics(infection_data: pd.DataFrame,
                           ratio_data: RatioData,
                           components_past: pd.DataFrame,
                           components_forecast: pd.DataFrame,
                           seiir_params: Dict[str, float],
                           compartment_info: CompartmentInfo) -> OutputMetrics:
    components = splice_components(components_past, components_forecast)

    if compartment_info.group_suffixes:
        modeled_infections, modeled_deaths = 0, 0
        for group in compartment_info.group_suffixes:
            group_compartments = [c for c in compartment_info.compartments if group in c]
            group_infections, vulnerable_infections = compute_infections(components[['date'] + group_compartments])

            group_ifr = getattr(ratio_data, f'ifr_{group}').rename('ifr')
            group_deaths = compute_deaths(vulnerable_infections, ratio_data.infection_to_death, group_ifr)

            modeled_infections += group_infections
            modeled_deaths += group_deaths
    else:
        modeled_infections, vulnerable_infections = compute_infections(
            components[['date'] + compartment_info.compartments]
        )
        modeled_deaths = compute_deaths(vulnerable_infections, ratio_data.infection_to_death, ratio_data.ifr)

    past_infections_idx = components_past.set_index('date', append=True).index
    modeled_infections = modeled_infections.to_frame()
    modeled_deaths = modeled_deaths.reset_index(level='observed')
    infection_data = infection_data.set_index(['location_id', 'date'])
    infections = infection_data.loc[past_infections_idx, ['infections']].combine_first(modeled_infections).infections
    deaths = infection_data[['deaths']]
    deaths['observed'] = 1
    deaths = deaths.combine_first(modeled_deaths).fillna(0)

    cases = ((infections
              .groupby('location_id')
              .shift(ratio_data.infection_to_case)
              * ratio_data.idr)
             .dropna()
             .rename('cases'))
    admissions = ((infections
                   .groupby('location_id')
                   .shift(ratio_data.infection_to_admission)
                   * ratio_data.ihr)
                  .dropna()
                  .rename('admissions'))

    r_controlled, r_effective = compute_effective_r(
        components,
        seiir_params,
        compartment_info.compartments
    )
    components = components.set_index('date', append=True)
    susceptible_columns = [c for c in components.columns if 'S' in c]
    immune_columns = [c for c in components.columns if 'M' in c or 'R' in c]
    return OutputMetrics(
        components=components,
        infections=infections,
        cases=cases,
        admissions=admissions,
        deaths=deaths,
        r_controlled=r_controlled,
        r_effective=r_effective,
        herd_immunity=(1 - 1 / r_controlled).rename('herd_immunity'),
        total_susceptible=components[susceptible_columns].sum(axis=1).rename('total_susceptible'),
        total_immune=components[immune_columns].sum(axis=1).rename('total_immune'),
    )


def compute_corrected_hospital_usage(admissions: pd.Series,
                                     hospital_fatality_ratio: pd.Series,
                                     hospital_parameters: 'HospitalParameters',
                                     correction_factors: HospitalCorrectionFactors) -> HospitalMetrics:
    hospital_usage = compute_hospital_usage(
        admissions,
        hospital_fatality_ratio,
        hospital_parameters,
    )
    hospital_usage.hospital_census = (hospital_usage.hospital_census
                                      * correction_factors.hospital_census).fillna(method='ffill')
    hospital_usage.icu_census = (hospital_usage.icu_census
                                 * correction_factors.icu_census).fillna(method='ffill')
    hospital_usage.ventilator_census = (hospital_usage.ventilator_census
                                        * correction_factors.ventilator_census).fillna(method='ffill')
    return hospital_usage


def splice_components(components_past: pd.DataFrame, components_forecast: pd.DataFrame):
    components_past = components_past.reindex(components_forecast.columns, axis='columns').reset_index()
    components_forecast = components_forecast.reset_index()
    components = (pd.concat([components_past, components_forecast])
                  .sort_values(['location_id', 'date'])
                  .set_index(['location_id']))
    return components


def compute_infections(components: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:

    def _get_daily_subgroup(data: pd.DataFrame, sub_group_columns: List[str]) -> Union[pd.Series, int]:
        if sub_group_columns:
            daily_data = (data[sub_group_columns]
                          .sum(axis=1, skipna=False)
                          .groupby('location_id')
                          .apply(lambda x: x.shift(1) - x)
                          .fillna(0)
                          .rename('infections'))
        else:
            daily_data = 0
        return daily_data

    def _cleanup(infections: pd.Series) -> pd.Series:
        return (pd.concat([components['date'], infections], axis=1)
                .reset_index()
                .set_index(['location_id', 'date'])
                .sort_index()['infections'])

    # Columns that will, when summed, give the desired group.
    susceptible_columns = [c for c in components.columns if 'S' in c]
    # E_p has both inflows and outflows so we have to sum
    # everything downstream of it.
    new_e_protected_columns = [c for c in components.columns if '_p' in c and 'S' not in c]
    immune_cols = [c for c in components.columns if 'M' in c]
    delta_susceptible = _get_daily_subgroup(components, susceptible_columns)
    delta_new_e_protected = _get_daily_subgroup(components, new_e_protected_columns)
    delta_immune = _get_daily_subgroup(components, immune_cols)

    # noinspection PyTypeChecker
    modeled_infections = _cleanup(delta_susceptible + delta_immune)
    # noinspection PyTypeChecker
    vulnerable_infections = _cleanup(delta_susceptible + delta_immune + delta_new_e_protected)

    return modeled_infections, vulnerable_infections


def compute_deaths(modeled_infections: pd.Series, infection_death_lag: int, ifr: pd.Series) -> pd.Series:
    modeled_deaths = (modeled_infections.shift(infection_death_lag) * ifr).dropna().rename('deaths').reset_index()
    modeled_deaths['observed'] = 0
    modeled_deaths = modeled_deaths.set_index(['location_id', 'date', 'observed'])['deaths']
    return modeled_deaths


def compute_effective_r(components: pd.DataFrame,
                        beta_params: Dict[str, float],
                        compartments: List[str]) -> Tuple[pd.Series, pd.Series]:
    alpha, sigma = beta_params['alpha'], beta_params['sigma']
    gamma1, gamma2 = beta_params['gamma1'], beta_params['gamma2']

    components = components.reset_index().set_index(['location_id', 'date'])

    beta, theta = components['beta'], components['theta']
    theta = theta.fillna(0.)
    susceptible = components[[c for c in compartments if 'S' in c]].sum(axis=1)
    infected = components[[c for c in compartments if 'I' in c]].sum(axis=1)
    n = components[compartments].sum(axis=1).groupby('location_id').max()
    avg_gamma = 1 / (1 / (gamma1*(sigma - theta)) + 1 / (gamma2*(sigma - theta)))

    r_controlled = (beta * alpha * sigma / avg_gamma * infected**(alpha - 1)).rename('r_controlled')
    r_effective = (r_controlled * susceptible / n).rename('r_effective')

    return r_controlled, r_effective
