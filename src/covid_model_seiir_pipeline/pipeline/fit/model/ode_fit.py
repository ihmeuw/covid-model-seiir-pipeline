import itertools
from typing import List, Tuple

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib.ode_mk2.containers import (
    Parameters,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT_NAMES,
    RISK_GROUP_NAMES,
    COMPARTMENTS_NAMES,
    TRACKING_COMPARTMENTS_NAMES,
)
from covid_model_seiir_pipeline.lib.ode_mk2 import (
    solver,
)
from covid_model_seiir_pipeline.pipeline.fit.specification import (
    FitParameters,
)
from covid_model_seiir_pipeline.pipeline.fit.model.rates import (
    Rates,
)
from covid_model_seiir_pipeline.pipeline.fit.model.sampled_params import (
    Durations,
    VariantRR,
    sample_ode_params,
    sample_parameter,
)


def prepare_ode_fit_parameters(rates: Rates,
                               epi_measures: pd.DataFrame,
                               rhos: pd.DataFrame,
                               vaccinations: pd.DataFrame,
                               etas: pd.DataFrame,
                               natural_waning_dist: pd.Series,
                               natural_waning_matrix: pd.DataFrame,
                               variant_severity: VariantRR,
                               fit_params: FitParameters,
                               hierarchy: pd.DataFrame,
                               draw_id: int) -> Tuple[Parameters, pd.DataFrame]:
    epi_measures, rates = prepare_epi_measures_and_rates(rates, epi_measures, hierarchy)
    past_index = epi_measures.index

    sampled_params = sample_ode_params(variant_severity, fit_params, draw_id)
    sampled_params = pd.DataFrame(sampled_params, index=past_index)

    weights = []
    for measure in ['death', 'admission', 'case']:
        parameter = f'weight_all_{measure}'
        weights.append(
            pd.Series(sample_parameter(parameter, draw_id, 0., 1.), name=parameter, index=past_index)
        )
    weights = [w / sum(weights).rename(w.name) for w in weights]

    rhos = rhos.reindex(past_index, fill_value=0.)
    rhos.columns = [f'rho_{c}_infection' for c in rhos.columns]
    rhos['rho_none_infection'] = pd.Series(0., index=past_index, name='rho_none_infection')
    
    base_parameters = pd.concat([
        sampled_params,
        epi_measures.rename(columns=lambda x: f'count_all_{x}'),
        pd.Series(-1, index=past_index, name='beta_all_infection'),
        rhos,
        *weights,
    ], axis=1)

    vaccinations = vaccinations.reindex(past_index, fill_value=0.)
    etas = etas.sort_index().reindex(past_index, fill_value=0.)

    phis = []
    for endpoint in ['infection', 'death', 'admission', 'case']:
        if endpoint == 'infection':
            w_base = pd.Series(0., index=natural_waning_dist.index)
        else:
            w_base = natural_waning_dist['infection']
        w_target = natural_waning_dist[endpoint]

        for from_variant, to_variant in itertools.product(VARIANT_NAMES, VARIANT_NAMES):
            cvi = natural_waning_matrix.loc[from_variant, to_variant]
            phi = 1 - (1 - cvi * w_target) / (1 - cvi * w_base)
            phis.append(phi.rename(f'{from_variant}_{to_variant}_{endpoint}'))
    phis = pd.concat(phis, axis=1)

    rates_map = {'ifr': 'death', 'ihr': 'admission', 'idr': 'case'}
    keep_cols = [f'{r}_{g}' for r, g in itertools.product(rates_map, RISK_GROUP_NAMES)]
    ode_rates = rates.loc[:, keep_cols].rename(columns=lambda x: f"{rates_map[x.split('_')[0]]}_{x.split('_')[1]}")
    ode_params = Parameters(
        base_parameters=base_parameters,
        vaccinations=vaccinations,
        rates=ode_rates,
        etas=etas,
        phis=phis,
    )
    return ode_params, rates


def prepare_epi_measures_and_rates(rates: Rates, epi_measures: pd.DataFrame, hierarchy: pd.DataFrame):
    most_detailed = hierarchy[hierarchy.most_detailed == 1].location_id.unique().tolist()
    measures = (
        ('deaths', 'death', 'ifr'),
        ('cases', 'case', 'idr'),
        ('hospitalizations', 'admission', 'ihr')
    )

    out_measures = []
    out_rates = []
    for in_measure, out_measure, rate in measures:
        epi_data = (epi_measures[f'smoothed_daily_{in_measure}'].rename(out_measure) + 3e-2).to_frame()
        epi_rates = rates._asdict()[rate]
        lag = epi_rates['lag'].iloc[0]

        epi_data = reindex_to_infection_day(epi_data, lag, most_detailed)
        epi_rates = reindex_to_infection_day(epi_rates.drop(columns='lag'), lag, most_detailed)

        out_measures.append(epi_data)
        out_rates.append(epi_rates)

    out_measures = pd.concat(out_measures, axis=1)
    out_measures = out_measures.loc[out_measures.notnull().any(axis=1)]

    dates = out_measures.reset_index().date
    global_date_range = pd.date_range(dates.min() - pd.Timedelta(days=1), dates.max())
    square_idx = pd.MultiIndex.from_product((most_detailed, global_date_range),
                                            names=['location_id', 'date']).sort_values()

    out_measures = out_measures.reindex(square_idx).sort_index()
    out_rates = pd.concat(out_rates, axis=1).reindex(square_idx).sort_index()

    return out_measures, out_rates


def reindex_to_infection_day(data: pd.DataFrame, lag: int, most_detailed: List[int]) -> pd.DataFrame:
    data = data.reset_index()
    data = (data
            .loc[data.location_id.isin(most_detailed)]
            .set_index(['location_id', 'date'])
            .groupby('location_id')
            .shift(-lag))
    return data


def make_initial_condition(parameters: Parameters, full_rates: pd.DataFrame, population: pd.DataFrame):
    base_params = parameters.base_parameters
    
    crude_infections = get_crude_infections(base_params, full_rates, population, threshold=50)    
    new_e_start = crude_infections.reset_index(level='date').groupby('location_id').first()
    start_date, new_e_start = new_e_start['date'], new_e_start['infections']
    end_date = base_params.filter(like='count')
    end_date = (end_date.loc[end_date.notnull().any(axis=1)]
                .reset_index(level='date')
                .groupby('location_id')                
                .last()
                .date + pd.Timedelta(days=1))
    # Alpha is time-invariant
    alpha = base_params.alpha_all_infection.groupby('location_id').first()
    compartments = [f'{compartment}_{risk_group}'
                    for risk_group, compartment
                    in itertools.product(RISK_GROUP_NAMES, COMPARTMENTS_NAMES + TRACKING_COMPARTMENTS_NAMES)]
    initial_condition = []
    for location_id, loc_start_date in start_date.iteritems():
        loc_end_date = end_date.loc[location_id]
        loc_initial_condition = pd.DataFrame(0., columns=compartments, index=full_rates.loc[location_id].index)
        for risk_group in RISK_GROUP_NAMES:
            pop = population.loc[location_id, risk_group]
            new_e = new_e_start.loc[location_id] * pop / population.loc[location_id].sum()
            suffix = f'_unvaccinated_{risk_group}'
            # Backfill everyone susceptible
            loc_initial_condition.loc[:loc_start_date, f'S_none{suffix}'] = pop
            # Set initial condition on start date
            infectious = (new_e / 5) ** (1 / alpha.loc[location_id])
            loc_initial_condition.loc[loc_start_date, f'S_none{suffix}'] = pop - new_e - infectious
            loc_initial_condition.loc[loc_start_date, f'E_ancestral{suffix}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'NewE_ancestral{suffix}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'NewENaive_ancestral{suffix}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'infection_ancestral_all_{risk_group}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'I_ancestral{suffix}'] = infectious
            for variant in VARIANT_NAMES:
                loc_initial_condition.loc[:loc_start_date, f'EffectiveSusceptible_{variant}{suffix}'] = pop
        loc_initial_condition.loc[loc_end_date:, :] = np.nan
        loc_initial_condition['location_id'] = location_id
        loc_initial_condition = loc_initial_condition.set_index('location_id', append=True).reorder_levels(['location_id', 'date'])
        initial_condition.append(loc_initial_condition)
    initial_condition = pd.concat(initial_condition)

    return initial_condition


def get_crude_infections(base_params, rates, population, threshold=50):
    crude_infections = pd.DataFrame(index=rates.index)
    for measure, rate in [('death', 'ifr'), ('admission', 'ihr'), ('case', 'idr')]:
        infections = base_params[f'count_all_{measure}'] / rates[rate]
        crude_infections[measure] = infections
    crude_infections = crude_infections.max(axis=1).rename('infections')
    crude_infections = crude_infections.loc[crude_infections > threshold]
    return crude_infections


def compute_posterior_epi_measures(compartments: pd.DataFrame,
                                   durations: Durations) -> Tuple[pd.DataFrame, pd.Series]:
    compartments_diff = compartments.groupby('location_id').diff().fillna(compartments)
    
    naive = compartments.filter(like='S_none').sum(axis=1).rename('naive')
    naive_unvaccinated = compartments.filter(like='S_none_unvaccinated').sum(axis=1).rename('naive_unvaccinated')
    
    naive_infections = compartments_diff.filter(like='NewENaive').sum(axis=1).rename('daily_naive_infections')
    total_infections = compartments_diff.filter(like='NewE_').sum(axis=1).rename('daily_total_infections')
    
    inf_cols = [f'infection_ancestral_all_{risk_group}' for risk_group in RISK_GROUP_NAMES]
    naive_unvaccinated_infections = (compartments_diff
                                     .loc[:, inf_cols]
                                     .sum(axis=1)
                                     .rename('daily_naive_unvaccinated_infections'))

    pct_unvaccinated = ((_to_cumulative(naive_unvaccinated_infections) / _to_cumulative(naive_unvaccinated_infections))
                        .clip(0, 1)
                        .rename('pct_unvaccinated')
                        .groupby('location_id')
                        .shift(durations.exposure_to_seroconversion)
                        .dropna()
                        .reset_index())

    measure_map = {
        'death': ('deaths', 'ifr'),
        'admission': ('hospitalizations', 'ihr'),
        'case': ('cases', 'idr'),
    }

    measures = []
    for ode_measure, (rates_measure, rate_name) in measure_map.items():
        cols = [f'{ode_measure}_ancestral_all_{risk_group}' for risk_group in RISK_GROUP_NAMES]
        lag = durations._asdict()[f'exposure_to_{ode_measure}']
        daily_measure = (compartments_diff
                         .loc[:, cols]
                         .sum(axis=1)
                         .groupby('location_id')
                         .shift(lag)
                         .rename(f'daily_{rates_measure}'))
        cumulative_measure = _to_cumulative(daily_measure)
        measures.extend([cumulative_measure, daily_measure])

    epi_measures = pd.concat([
        naive, naive_unvaccinated,
        naive_unvaccinated_infections, _to_cumulative(naive_unvaccinated_infections),
        naive_infections, _to_cumulative(naive_infections),
        total_infections, _to_cumulative(total_infections),
        *measures
    ], axis=1)

    return epi_measures, pct_unvaccinated


def _shift(lag: int):
    def _inner(x: pd.Series):
        return (x
                .reset_index(level='location_id', drop=True)
                .shift(periods=lag, freq='D'))
    return _inner


def _to_cumulative(data: pd.Series):
    daily = (data
             .groupby('location_id')
             .cumsum()
             .rename(f'{str(data.name).replace("daily", "cumulative")}'))
    
    daily[(daily < 0) & (data == 0)] = np.nan
    return daily
    

def run_ode_fit(initial_condition: pd.DataFrame,
                ode_parameters: Parameters,
                num_cores: int,
                progress_bar: bool,
                location_ids: List[int] = None):
    if location_ids is None:
        location_ids = initial_condition.reset_index().location_id.unique().tolist()
    full_compartments, chis = solver.run_ode_model(
        initial_condition,
        **ode_parameters.to_dict(),
        location_ids=location_ids,
        forecast=False,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )
    # Set all the forecast stuff to nan
    full_compartments.loc[full_compartments.sum(axis=1) == 0., :] = np.nan

    betas = (full_compartments
             .filter(like='beta_none')
             .filter(like='lr')
             .groupby('location_id')
             .diff()
             .rename(columns=lambda x: f'beta_{x.split("_")[2]}')
             .rename(columns={'beta_all': 'beta'}))
    counts = ode_parameters.base_parameters.filter(like='count')
    assert betas.index.equals(counts.index)
    for measure in ['death', 'admission', 'case']:
        betas.loc[counts[f'count_all_{measure}'].isnull(), f'beta_{measure}'] = np.nan

    return full_compartments, betas, chis



