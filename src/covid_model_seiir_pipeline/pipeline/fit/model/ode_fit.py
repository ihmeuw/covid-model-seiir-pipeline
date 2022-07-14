import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib.ode_mk2.containers import (
    Parameters,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    SYSTEM_TYPE,
    VARIANT_NAMES,
    RISK_GROUP_NAMES,
    COMPARTMENTS_NAMES,
    TRACKING_COMPARTMENTS_NAMES,
)
from covid_model_seiir_pipeline.lib.ode_mk2 import (
    solver,
)
from covid_model_seiir_pipeline.lib import (
    parallel,
)
from covid_model_seiir_pipeline.pipeline.fit.model.sampled_params import (
    Durations,
    sample_parameter,
)
from covid_model_seiir_pipeline.pipeline.fit.model.epi_measures import (
    aggregate_data_from_md,
)


def prepare_ode_fit_parameters(measure: str,
                               rates: pd.DataFrame,
                               epi_measures: pd.DataFrame,
                               rhos: pd.DataFrame,
                               vaccinations: pd.DataFrame,
                               etas: pd.DataFrame,
                               natural_waning_dist: pd.Series,
                               natural_waning_matrix: pd.DataFrame,
                               sampled_ode_params: Dict[str, float],
                               hierarchy: pd.DataFrame,
                               population: pd.Series,
                               draw_id: int) -> Tuple[Parameters, pd.DataFrame]:
    measures_and_rates, age_scalars = prepare_epi_measures_and_rates(
        measure, rates, epi_measures, hierarchy, population,
    )
    past_index = measures_and_rates.index
    scalar_params = {k: p for k, p in sampled_ode_params.items() if isinstance(p, (int, float))}
    series_params = [p.reindex(past_index, level='location_id').rename(k)
                     for k, p in sampled_ode_params.items() if isinstance(p, pd.Series)]
    sampled_params = pd.concat([
        pd.DataFrame(scalar_params, index=past_index),
        *series_params,
    ], axis=1)

    rhos = rhos.reindex(past_index, fill_value=0.)
    rhos.columns = [f'rho_{c}_infection' for c in rhos.columns]
    rhos['rho_none_infection'] = pd.Series(0., index=past_index, name='rho_none_infection')

    base_parameters = pd.concat([
        sampled_params,
        measures_and_rates,
        pd.Series(-1, index=past_index, name='beta_all_infection'),
        rhos,
    ], axis=1)

    vaccinations = vaccinations.reindex(past_index, fill_value=0.)
    etas = etas.sort_index().reindex(past_index, fill_value=0.)
    phis = compute_phis(
        natural_waning_dist, natural_waning_matrix, draw_id
    )

    ode_params = Parameters(
        base_parameters=base_parameters,
        vaccinations=vaccinations,
        age_scalars=age_scalars,
        etas=etas,
        phis=phis,
    )
    return ode_params, rates


def prepare_past_infections_parameters(beta: pd.Series,
                                       rates: pd.DataFrame,
                                       durations: Durations,
                                       epi_measures: pd.DataFrame,
                                       rhos: pd.DataFrame,
                                       vaccinations: pd.DataFrame,
                                       etas: pd.DataFrame,
                                       natural_waning_dist: pd.Series,
                                       natural_waning_matrix: pd.DataFrame,
                                       resampled_params: pd.DataFrame,
                                       hierarchy: pd.DataFrame,
                                       draw_id: int):
    measures_and_rates, age_scalars = prepare_epi_measures_for_past_infections(
        epi_measures=epi_measures,
        epi_rates=rates,
        durations=durations,
        hierarchy=hierarchy,
    )
    loc_ids = beta.reset_index().location_id.unique().tolist()
    past_index = measures_and_rates.loc[loc_ids].sort_index().index
    resampled_params = (resampled_params
                        .reindex(past_index)
                        .groupby('location_id')
                        .ffill()
                        .groupby('location_id')
                        .bfill())

    rhos = rhos.reindex(past_index, fill_value=0.)
    rhos.columns = [f'rho_{c}_infection' for c in rhos.columns]
    rhos['rho_none_infection'] = pd.Series(0., index=past_index, name='rho_none_infection')

    base_parameters = pd.concat([
        resampled_params,
        measures_and_rates,
        beta.reindex(past_index),
        rhos,
    ], axis=1)

    vaccinations = vaccinations.reindex(past_index, fill_value=0.)
    etas = etas.sort_index().reindex(past_index, fill_value=0.)
    phis = compute_phis(
        natural_waning_dist, natural_waning_matrix, draw_id
    )

    return Parameters(
        base_parameters=base_parameters,
        vaccinations=vaccinations,
        age_scalars=age_scalars,
        etas=etas,
        phis=phis,
    )


def prepare_epi_measures_and_rates(measure: str,
                                   rates: pd.DataFrame,
                                   epi_measures: pd.DataFrame,
                                   hierarchy: pd.DataFrame,
                                   population: pd.Series):
    metrics = ['count', 'rate', 'weight']
    measures = {
        'death': ('deaths', 'ifr'),
        'case': ('cases', 'idr'),
        'admission': ('hospitalizations', 'ihr'),
    }
    in_measure, rate = measures[measure]
    most_detailed = hierarchy[hierarchy.most_detailed == 1].location_id.unique().tolist()
    total_measure = epi_measures[f'smoothed_daily_{in_measure}'].dropna().groupby('location_id').sum()
    to_model = total_measure[total_measure > 0].index.intersection(most_detailed).tolist()
    model_idx = epi_measures.loc[to_model].index

    offset = epi_measures[f'smoothed_daily_{in_measure}'].groupby('location_id').max() * 0.01
    offset = offset.clip(0, 1)
    lag = rates['lag'].iloc[0]
    measure_data = reindex_to_infection_day(
        epi_measures[f'smoothed_daily_{in_measure}'] + offset,
        lag,
        to_model
    ).reindex(model_idx)

    out_data = pd.DataFrame(
        np.nan,
        columns=[f'{metric}_all_{m}' for metric, m in itertools.product(metrics, measures)],
        index=model_idx,
    )
    out_data.loc[:, f'count_all_{measure}'] = measure_data[f'smoothed_daily_{in_measure}']

    out_data.loc[:, [c for c in out_data if 'weight' in c]] = 0.
    out_data.loc[:, f'weight_all_{measure}'] = 1.0
    out_scalars = pd.DataFrame(
        np.nan,
        columns=[f'{m}_{rg}' for m, rg in itertools.product(measures, RISK_GROUP_NAMES)],
        index=model_idx,
    )

    rates = reindex_to_infection_day(rates.drop(columns='lag'), lag, most_detailed)
    for risk_group in RISK_GROUP_NAMES:
        out_scalars.loc[:, f'{measure}_{risk_group}'] = (
            rates[f'{rate}_{risk_group}'] / rates[rate]
        )
    out_data.loc[:, f'rate_all_{measure}'] = rates[rate]

    return out_data, out_scalars


def prepare_epi_measures_for_past_infections(epi_measures: pd.DataFrame,
                                             epi_rates: pd.DataFrame,
                                             durations: Durations,
                                             hierarchy: pd.DataFrame):
    most_detailed = hierarchy[hierarchy.most_detailed == 1].location_id.unique().tolist()
    measures = (
        ('deaths', 'death', 'ifr'),
        ('cases', 'case', 'idr'),
        ('hospitalizations', 'admission', 'ihr')
    )

    out_measures = []
    out_scalars = []
    out_rates = []
    for in_measure, out_measure, rate in measures:
        epi_data = (epi_measures[f'smoothed_daily_{in_measure}'].rename(f'count_all_{out_measure}') + 3e-2).to_frame()
        lag = durations._asdict()[f'exposure_to_{out_measure}']

        epi_data = reindex_to_infection_day(epi_data, lag, most_detailed)        
        rates = reindex_to_infection_day(epi_rates.filter(like=rate), lag, most_detailed)
        epi_scalars = rates.copy()
        drop_cols = epi_scalars.columns
        for risk_group in RISK_GROUP_NAMES:
            epi_scalars.loc[:, f'{out_measure}_{risk_group}'] = epi_scalars[f'{rate}_{risk_group}'] / epi_scalars[rate]
        epi_scalars = epi_scalars.drop(columns=drop_cols)

        out_measures.append(epi_data)
        out_scalars.append(epi_scalars)
        out_rates.append(rates[rate].rename(f'rate_all_{out_measure}'))

    out_measures = pd.concat(out_measures, axis=1).reset_index()
    out_measures = (out_measures
                    .loc[out_measures.location_id.isin(most_detailed)]
                    .set_index(['location_id', 'date'])
                    .sort_index())
    out_rates = pd.concat(out_rates, axis=1).reindex(out_measures.index).groupby('location_id').ffill().bfill()    
    out_weights = out_measures.copy().rename(columns=lambda x: x.replace('count', 'weight'))
    out_weights.loc[:, :] = np.nan
    out_data = pd.concat([out_measures, out_rates, out_weights], axis=1)
    out_scalars = pd.concat(out_scalars, axis=1).reindex(out_measures.index).sort_index()
    return out_data, out_scalars


def reindex_to_infection_day(data: pd.DataFrame, lag: int, most_detailed: List[int]) -> pd.DataFrame:
    data = data.reset_index()
    data = (data
            .loc[data.location_id.isin(most_detailed)]
            .set_index(['location_id', 'date'])
            .groupby('location_id')
            .shift(-lag))
    return data


def compute_phis(natural_waning_dist: pd.Series,
                 natural_waning_matrix: pd.DataFrame,
                 draw_id: int):
    phis = []
    phi_scalar = sample_parameter('phi_scalar', draw_id, 0.85, 0.95)
    phi_scalars = {'death': phi_scalar, 'admission': phi_scalar}
    for endpoint in ['infection', 'death', 'admission', 'case']:
        if endpoint == 'infection':
            w_base = pd.Series(0., index=natural_waning_dist.index)
        else:
            w_base = natural_waning_dist['infection']
        w_target = natural_waning_dist[endpoint]

        for from_variant, to_variant in itertools.product(VARIANT_NAMES, VARIANT_NAMES):
            if endpoint == 'case' and to_variant == 'omicron':
                # ## MID-POINT
                # w_target = natural_waning_dist[['infection', 'admission']].mean(axis=1).rename('case')
                ## SYMPTOMATIC == SEVERE
                w_target = natural_waning_dist['admission'].rename('case')
            cvi = natural_waning_matrix.loc[from_variant, to_variant]
            numerator_scalar = phi_scalars.get(endpoint, cvi)
            phi = 1 - (1 - numerator_scalar * w_target) / (1 - cvi * w_base)
            phi[phi.cummax() < phi.max()] = phi.max()
            phis.append(phi.rename(f'{from_variant}_{to_variant}_{endpoint}'))
    phis = pd.concat(phis, axis=1)
    return phis


def make_initial_condition(measure: str,
                           parameters: Parameters,
                           rates: pd.DataFrame,
                           population: pd.DataFrame, 
                           infections: pd.Series = None):
    base_params = parameters.base_parameters
    rates = rates.loc[base_params.index]
    
    crude_infections = get_crude_infections(measure, base_params, rates, infections)
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
        loc_initial_condition = pd.DataFrame(0., columns=compartments, index=rates.loc[location_id].index)
        for risk_group in RISK_GROUP_NAMES:
            pop = population.loc[location_id, risk_group]
            if measure == 'final':
                beta_init = base_params.at[(location_id, loc_start_date), 'beta_all_infection']
            else:
                beta_init = 2.0                
            new_e = new_e_start.loc[location_id] * pop / population.loc[location_id].sum()
            infectious = (new_e / beta_init) ** (1 / alpha.loc[location_id])            
            susceptible = pop - new_e - infectious
            
            # Backfill everyone susceptible
            loc_initial_condition.loc[:loc_start_date, f'S_course_0_none_{risk_group}'] = pop
            loc_initial_condition.loc[loc_start_date, f'S_course_0_none_{risk_group}'] = susceptible
            loc_initial_condition.loc[loc_start_date, f'E_course_0_ancestral_{risk_group}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'Infection_none_ancestral_course_0_{risk_group}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'Infection_none_all_course_0_{risk_group}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'Infection_none_all_all_{risk_group}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'Infection_all_all_course_0_{risk_group}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'Infection_all_all_all_{risk_group}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'Infection_all_ancestral_all_{risk_group}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'I_unvaccinated_ancestral_{risk_group}'] = infectious
            beta_measure = 'all' if measure == 'final' else measure
            loc_initial_condition.loc[loc_start_date, f'Beta_none_none_{beta_measure}_{risk_group}'] = beta_init
            loc_initial_condition.loc[loc_start_date, f'Beta_none_none_all_{risk_group}'] = beta_init
            for variant in VARIANT_NAMES:
                label = f'EffectiveSusceptible_all_{variant}_all_{risk_group}'
                loc_initial_condition.loc[:loc_start_date, label] = susceptible

        loc_initial_condition.loc[loc_end_date:, :] = np.nan
        loc_initial_condition['location_id'] = location_id
        loc_initial_condition = loc_initial_condition.set_index('location_id', append=True).reorder_levels(['location_id', 'date'])
        initial_condition.append(loc_initial_condition)
    initial_condition = pd.concat(initial_condition)

    return initial_condition


def get_crude_infections(measure: str, base_params, rates, infections):
    rate_map = {'death': 'ifr', 'admission': 'ihr', 'case': 'idr'}
    if measure in rate_map:
        crude_infections = base_params[f'count_all_{measure}'] / rates[rate_map[measure]]
        crude_infections = crude_infections.loc[crude_infections > 0.].rename('infections')
    elif infections is not None:
        crude_infections = infections
    else:
        crude_infections = pd.DataFrame(index=rates.index)
        for measure, rate in rate_map.items():
            infections = base_params[f'count_all_{measure}'] / rates[rate]
            infections[infections <= 0.] = np.nan
            crude_infections[measure] = infections
        mask = base_params['beta_all_infection'] > 0
        crude_infections = crude_infections.loc[mask].mean(axis=1).rename('infections').dropna()
    return crude_infections


def compute_posterior_epi_measures(compartments: pd.DataFrame,
                                   durations: Durations) -> pd.DataFrame:
    compartments_diff = compartments.groupby('location_id').diff().fillna(compartments)
    
    naive = compartments.filter(like='S_').filter(like='none').sum(axis=1).rename('naive')
    naive_unvaccinated = compartments.filter(like='S_unvaccinated_none').sum(axis=1).rename('naive_unvaccinated')
    
    inf_map = {'Infection_none_all_unvaccinated': 'daily_naive_unvaccinated_infections',
               'Infection_none_all_all': 'daily_naive_infections',
               'Infection_all_all_all': 'daily_total_infections'}
    infections = []
    for col, name in inf_map.items():
        daily = compartments_diff.filter(like=col).sum(axis=1, min_count=1).rename(name)
        infections.extend([daily, _to_cumulative(daily)])
    infections = pd.concat(infections, axis=1)

    measure_map = {
        'Death': 'deaths',
        'Admission': 'hospitalizations',
        'Case': 'cases',
    }

    measures = []
    for ode_measure, rates_measure in measure_map.items():
        lag = durations._asdict()[f'exposure_to_{ode_measure.lower()}']
        daily_measure = (compartments_diff
                         .filter(like=f'{ode_measure}_none_all_unvaccinated')
                         .sum(axis=1, min_count=1)
                         .groupby('location_id')
                         .shift(lag)
                         .rename(f'daily_{rates_measure}'))
        cumulative_measure = _to_cumulative(daily_measure)
        measures.extend([cumulative_measure, daily_measure])

    epi_measures = pd.concat([
        naive, naive_unvaccinated,
        infections,
        *measures
    ], axis=1)

    return epi_measures


def aggregate_posterior_epi_measures(measure: str,
                                     epi_measures: pd.DataFrame,
                                     posterior_epi_measures: pd.DataFrame,
                                     hierarchy: pd.DataFrame) -> pd.DataFrame:
    output_measure = {'death': 'deaths', 'case': 'cases', 'admission': 'hospitalizations'}[measure]
    posterior_locs = posterior_epi_measures.reset_index()['location_id'].unique().tolist()
    epi_measures = epi_measures.loc[posterior_locs]
    agg_posterior_epi_measures = []
    for measure in ['cumulative_naive_unvaccinated_infections',
                    'cumulative_naive_infections',
                    'cumulative_total_infections',
                    f'cumulative_{output_measure}']:
        if 'infections' in measure:
            pem = (posterior_epi_measures.loc[:, [measure]]
                   .dropna()
                   .reset_index())
        else:
            pem = (posterior_epi_measures.loc[epi_measures.loc[epi_measures[measure].notnull()].index,
                                              [measure]]
                   .dropna()
                   .reset_index())
        agg_posterior_epi_measures.append(
            aggregate_data_from_md(pem, hierarchy, measure).set_index(['location_id', 'date'])
        )
    agg_posterior_epi_measures = pd.concat(agg_posterior_epi_measures, axis=1)
    agg_posterior_epi_measures = pd.concat([
        agg_posterior_epi_measures,
        (agg_posterior_epi_measures
         .groupby(level=0).diff().fillna(agg_posterior_epi_measures)
         .rename(columns=lambda x: x.replace('cumulative', 'daily')))
    ], axis=1)

    return agg_posterior_epi_measures


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
        system_type=SYSTEM_TYPE.rates_and_measures,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )
    # Set all the forecast stuff to nan
    full_compartments.loc[full_compartments.sum(axis=1) == 0., :] = np.nan
    
    betas = (full_compartments
             .filter(like='Beta_none_none')
             .filter(like='lr'))

    betas = (betas
             .groupby('location_id')
             .diff()
             .fillna(betas)
             .rename(columns=lambda x: f'beta_{x.split("_")[3]}')
             .rename(columns={'beta_all': 'beta'}))
        
    # Can have a composite beta if we don't have measure betas
    no_beta = betas[[f'beta_{measure}' for measure in ['death', 'admission', 'case']]].isnull().all(axis=1)
    betas.loc[no_beta, 'beta'] = np.nan
    betas[np.abs(betas) <= 1e-5] = np.nan

    return full_compartments, betas, chis


def run_posterior_fit(initial_condition: pd.DataFrame,
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
        system_type=SYSTEM_TYPE.beta_and_measures,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )
    # Set all the forecast stuff to nan
    full_compartments.loc[full_compartments.sum(axis=1) == 0., :] = np.nan
    return full_compartments, chis


def fill_from_hierarchy(df: pd.DataFrame, hierarchy: pd.DataFrame, level: str) -> pd.DataFrame:
    assert level in ['parent_id', 'region_id', 'super_region_id', 'global']
    if level == 'global':
        df[level] = 1
    else:
        df[level] = hierarchy.set_index('location_id').reindex(df.index, level='location_id')[level]
    fill = df.groupby([level, 'date']).transform('mean')
    df = df.drop(columns=level).fillna(fill)
    return df

