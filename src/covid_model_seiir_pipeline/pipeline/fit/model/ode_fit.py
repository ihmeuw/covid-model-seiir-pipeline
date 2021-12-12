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
        _weights = pd.Series(sample_parameter(parameter, draw_id, 0., 1.), name=parameter, index=past_index)
        _weights = add_transition_period(
            weights=_weights,
            data_period=epi_measures.loc[epi_measures[measure].notnull()].index,
        )
        weights.append(_weights)
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

    out_measures = pd.concat(out_measures, axis=1).reset_index()
    out_measures = out_measures.loc[out_measures.location_id.isin(most_detailed)].set_index(['location_id', 'date'])
    out_rates = pd.concat(out_rates, axis=1).reindex(out_measures.index).sort_index()
    return out_measures, out_rates


def add_transition_period(weights: pd.Series, data_period: pd.Index, window_len: int = 30) -> pd.Series:
    w_ = pd.Series(np.nan, name=weights.name, index=data_period)
    w0 = w_.reset_index().groupby('location_id')[['date']].min()
    w1 = w0 + pd.Timedelta(days=window_len)
    w3 = w_.reset_index().groupby('location_id')[['date']].max()
    w2 = w3 - pd.Timedelta(days=window_len)
    
    req = w2[w2 > w1].dropna().index
    w0 = w0.loc[req].set_index('date', append=True).index
    w1 = w1.loc[req].set_index('date', append=True).index
    w2 = w2.loc[req].set_index('date', append=True).index
    w3 = w3.loc[req].set_index('date', append=True).index
    
    w_.loc[w0] = 1 / window_len
    w_.loc[w1] = 1
    w_.loc[w2] = 1
    w_.loc[w3] = 1 / window_len
    w_ = w_.groupby(level=0).apply(lambda x: x.interpolate())
    
    return (weights * w_).fillna(0)


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
    
    crude_infections = get_crude_infections(base_params, full_rates, threshold=50)
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
            # Backfill everyone susceptible
            loc_initial_condition.loc[:loc_start_date, f'S_unvaccinated_none_{risk_group}'] = pop
            # Set initial condition on start date
            infectious = (new_e / 5) ** (1 / alpha.loc[location_id])
            susceptible = pop - new_e - infectious

            loc_initial_condition.loc[loc_start_date, f'S_unvaccinated_none_{risk_group}'] = susceptible
            loc_initial_condition.loc[loc_start_date, f'E_unvaccinated_ancestral_{risk_group}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'Infection_none_all_unvaccinated_{risk_group}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'Infection_none_all_all_{risk_group}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'Infection_all_all_all_{risk_group}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'Infection_all_ancestral_all_{risk_group}'] = new_e
            loc_initial_condition.loc[loc_start_date, f'I_unvaccinated_ancestral_{risk_group}'] = infectious
            for variant in VARIANT_NAMES:
                label = f'EffectiveSusceptible_all_{variant}_all_{risk_group}'
                loc_initial_condition.loc[:loc_start_date, label] = susceptible

#            for measure, rate in [('death', 'ifr'), ('admission', 'ihr'), ('case', 'idr')]:
#                measure_data = base_params.loc[(location_id, loc_start_date), f'count_all_{measure}']
#                if np.isnan(measure_data):
#                    measure_data = new_e * full_rates.loc[(location_id, loc_start_date), rate]
#                label = measure.capitalize()
#                loc_initial_condition.loc[loc_start_date, f'{label}_none_all_unvaccinated_{risk_group}'] = measure_data
#                loc_initial_condition.loc[loc_start_date, f'{label}_none_all_all_{risk_group}'] = measure_data
#                loc_initial_condition.loc[loc_start_date, f'{label}_all_all_all_{risk_group}'] = measure_data
#                loc_initial_condition.loc[loc_start_date, f'{label}_all_ancestral_all_{risk_group}'] = measure_data

        loc_initial_condition.loc[loc_end_date:, :] = np.nan
        loc_initial_condition['location_id'] = location_id
        loc_initial_condition = loc_initial_condition.set_index('location_id', append=True).reorder_levels(['location_id', 'date'])
        initial_condition.append(loc_initial_condition)
    initial_condition = pd.concat(initial_condition)

    return initial_condition


def get_crude_infections(base_params, rates, threshold=50):
    crude_infections = pd.DataFrame(index=rates.index)
    for measure, rate in [('death', 'ifr'), ('admission', 'ihr'), ('case', 'idr')]:
        infections = base_params[f'count_all_{measure}'] / rates[rate]
        crude_infections[measure] = infections
    mask = crude_infections.max(axis=1) > threshold
    crude_infections = crude_infections.loc[mask].min(axis=1).rename('infections')
    return crude_infections


def compute_posterior_epi_measures(compartments: pd.DataFrame,
                                   durations: Durations) -> Tuple[pd.DataFrame, pd.Series]:
    compartments_diff = compartments.groupby('location_id').diff().fillna(compartments)
    
    naive = compartments.filter(like='S_').filter(like='none').sum(axis=1).rename('naive')
    naive_unvaccinated = compartments.filter(like='S_unvaccinated_none').sum(axis=1).rename('naive_unvaccinated')
    
    inf_map = {'Infection_none_all_unvaccinated': 'daily_naive_unvaccinated_infections',
               'Infection_none_all_all': 'daily_naive_infections',
               'Infection_all_all_all': 'daily_total_infections'}
    infections = []
    for col, name in inf_map.items():
        daily = compartments_diff.filter(like=col).sum(axis=1).rename(name)
        infections.extend([daily, _to_cumulative(daily)])
    infections = pd.concat(infections, axis=1)

    # FIXME: Ratio is wrong.
    nui = _to_cumulative(infections['daily_naive_unvaccinated_infections'])
    pct_unvaccinated = ((nui / nui)
                        .clip(0, 1)
                        .rename('pct_unvaccinated')
                        .groupby('location_id')
                        .shift(durations.exposure_to_seroconversion)
                        .dropna()
                        .reset_index())

    measure_map = {
        'Death': ('deaths', 'ifr'),
        'Admission': ('hospitalizations', 'ihr'),
        'Case': ('cases', 'idr'),
    }

    measures = []
    for ode_measure, (rates_measure, rate_name) in measure_map.items():
        cols = [f'{ode_measure}_none_all_unvaccinated_{risk_group}' for risk_group in RISK_GROUP_NAMES]
        lag = durations._asdict()[f'exposure_to_{ode_measure.lower()}']
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
        infections,
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
             .filter(like='Beta_none_none')
             .filter(like='lr'))
    betas = (betas
             .groupby('location_id')
             .diff()
             .fillna(betas)
             .rename(columns=lambda x: f'beta_{x.split("_")[3]}')
             .rename(columns={'beta_all': 'beta'}))
        
    # Clean up beta
    # counts and beta should be identically indexed in production, 
    # but subset to beta in case we only ran a subset of locations for debugging
    counts = ode_parameters.base_parameters.filter(like='count').loc[betas.index]    
    for measure in ['death', 'admission', 'case']:
        betas.loc[counts[f'count_all_{measure}'].isnull(), f'beta_{measure}'] = np.nan
    # Can have a composite beta if we don't have measure betas
    no_beta = betas[[f'beta_{measure}' for measure in ['death', 'admission', 'case']]].isnull().all(axis=1)
    betas.loc[no_beta, 'beta'] = np.nan

    return full_compartments, betas, chis



