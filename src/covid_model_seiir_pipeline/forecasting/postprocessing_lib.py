import functools
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.forecasting.specification import FORECAST_SCALING_CORES, ResamplingSpecification


# TODO: make a model subpackage and put this there.


def load_deaths(scenario: str, data_interface: ForecastDataInterface):
    deaths, *_ = load_output_data(scenario, data_interface)
    return deaths


def load_infections(scenario: str, data_interface: ForecastDataInterface):
    _, infections, *_ = load_output_data(scenario, data_interface)
    return infections


def load_r_effective(scenario: str, data_interface: ForecastDataInterface):
    _, _, r_effective = load_output_data(scenario, data_interface)
    return r_effective


def load_output_data(scenario: str, data_interface: ForecastDataInterface):
    _runner = functools.partial(
        load_output_data_by_draw,
        scenario=scenario,
        data_interface=data_interface,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        outputs = pool.map(_runner, draws)
    deaths, infections, r_effective = zip(*outputs)

    return deaths, infections, r_effective


def load_output_data_by_draw(draw_id: int, scenario: str,
                             data_interface: ForecastDataInterface) -> Tuple[pd.Series, pd.Series, pd.Series]:
    draw_df = data_interface.load_raw_outputs(scenario, draw_id)
    draw_df = draw_df.set_index(['location_id', 'date']).sort_index()
    deaths = draw_df.reset_index().set_index(['location_id', 'date', 'observed'])['deaths'].rename(draw_id)
    infections = draw_df['infections'].rename(draw_id)
    r_effective = draw_df['r_effective'].rename(draw_id)
    return deaths, infections, r_effective


def load_coefficients(scenario: str, data_interface: ForecastDataInterface):
    _runner = functools.partial(
        load_coefficients_by_draw,
        data_interface=data_interface
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        outputs = pool.map(_runner, draws)
    return outputs


def load_coefficients_by_draw(draw_id: int, data_interface: ForecastDataInterface) -> pd.Series:
    coefficients = data_interface.load_regression_coefficients(draw_id)
    coefficients = coefficients.set_index('location_id').stack().reset_index()
    coefficients.columns = ['location_id', 'covariate', draw_id]
    coefficients = coefficients.set_index(['location_id', 'covariate'])[draw_id]
    return coefficients


def load_scaling_parameters(scenario: str, data_interface: ForecastDataInterface):
    _runner = functools.partial(
        load_scaling_parameters_by_draw,
        scenario=scenario,
        data_interface=data_interface,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        outputs = pool.map(_runner, draws)
    return outputs


def load_scaling_parameters_by_draw(draw_id: int, scenario: str, data_interface: ForecastDataInterface) -> pd.Series:
    scaling_parameters = data_interface.load_beta_scales(scenario, draw_id)
    scaling_parameters = scaling_parameters.set_index('location_id').stack().reset_index()
    scaling_parameters.columns = ['location_id', 'scaling_parameter', draw_id]
    scaling_parameters = scaling_parameters.set_index(['location_id', 'scaling_parameter'])[draw_id]
    return scaling_parameters


def load_covariate(covariate: str, time_varying: bool, scenario: str,
                   data_interface: ForecastDataInterface) -> List[pd.Series]:
    _runner = functools.partial(
        load_covariate_by_draw,
        covariate=covariate,
        time_varying=time_varying,
        scenario=scenario,
        data_interface=data_interface,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        outputs = pool.map(_runner, draws)

    return outputs


def load_covariate_by_draw(draw_id: int,
                           covariate: str,
                           time_varying: bool,
                           scenario: str,
                           data_interface: ForecastDataInterface) -> pd.Series:
    covariate_df = data_interface.load_raw_covariates(scenario, draw_id)
    covariate_df = covariate_df.set_index(['location_id', 'date']).sort_index()
    if time_varying:
        covariate_data = covariate_df[covariate].rename(draw_id)
    else:
        covariate_data = covariate_df.groupby(level='location_id')[covariate].max().rename(draw_id)
    return covariate_data


def load_betas(scenario: str, data_interface: ForecastDataInterface) -> List[pd.Series]:
    _runner = functools.partial(
        load_betas_by_draw,
        scenario=scenario,
        data_interface=data_interface,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        betas = pool.map(_runner, draws)
    return betas


def load_betas_by_draw(draw_id: int, scenario: str, data_interface: ForecastDataInterface) -> pd.Series:
    components = data_interface.load_components(scenario, draw_id)
    draw_betas = (components
                  .sort_index()['beta']
                  .rename(draw_id))
    return draw_betas


def load_beta_residuals(scenario: str, data_interface: ForecastDataInterface) -> List[pd.Series]:
    _runner = functools.partial(
        load_beta_residuals_by_draw,
        data_interface=data_interface,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        beta_residuals = pool.map(_runner, draws)
    return beta_residuals


def load_beta_residuals_by_draw(draw_id: int, data_interface: ForecastDataInterface) -> pd.Series:
    beta_regression = data_interface.load_beta_regression(draw_id)
    beta_regression = (beta_regression
                       .set_index(['location_id', 'date'])
                       .sort_index()[['beta', 'beta_pred']])
    beta_residual = np.log(beta_regression['beta'] / beta_regression['beta_pred']).rename(draw_id)
    return beta_residual


def load_elastispliner_inputs(data_interface: ForecastDataInterface) -> pd.DataFrame:
    es_inputs = data_interface.load_elastispliner_inputs()
    es_inputs = es_inputs.set_index(['location_id', 'date'])
    cumulative_cases = (es_inputs['Confirmed case rate'] * es_inputs['population']).rename('cumulative_cases')
    cumulative_deaths = (es_inputs['Death rate'] * es_inputs['population']).rename('cumulative_deaths')
    cumulative_hospitalizations = (es_inputs['Hospitalization rate'] * es_inputs['population'])
    cumulative_hospitalizations = cumulative_hospitalizations.rename('cumulative_hospitalizations')
    es_inputs = pd.concat([cumulative_cases, cumulative_deaths, cumulative_hospitalizations], axis=1)
    return es_inputs


def load_es_noisy(scenario: str, data_interface: ForecastDataInterface):
    return load_elastispliner_outputs(data_interface, noisy=True)


def load_es_smoothed(scenario: str, data_interface: ForecastDataInterface):
    return load_elastispliner_outputs(data_interface, noisy=False)


def load_elastispliner_outputs(data_interface: ForecastDataInterface, noisy: bool):
    es_noisy, es_smoothed = data_interface.load_elastispliner_outputs()
    es_outputs = es_noisy if noisy else es_smoothed
    es_outputs = es_outputs.set_index(['location_id', 'date', 'observed'])
    n_draws = data_interface.get_n_draws()
    es_outputs = es_outputs.rename(columns={f'draw_{i}': i for i in range(n_draws)})
    es_outputs = es_outputs.groupby(level='location_id').apply(lambda x: x - x.shift(fill_value=0))
    return es_outputs


def load_full_data(data_interface: ForecastDataInterface) -> pd.DataFrame:
    full_data = data_interface.load_full_data()
    full_data = full_data.set_index(['location_id', 'date'])
    full_data = full_data.rename(columns={
        'Deaths': 'cumulative_deaths',
        'Confirmed': 'cumulative_cases',
        'Hospitalizations': 'cumulative_hospitalizations',
    })
    full_data = full_data[['cumulative_cases', 'cumulative_deaths', 'cumulative_hospitalizations']]
    return full_data


def build_version_map(data_interface: ForecastDataInterface) -> pd.Series:
    version_map = {}
    version_map['forecast_version'] = data_interface.forecast_paths.root_dir.name
    version_map['regression_version'] = data_interface.regression_paths.root_dir.name
    version_map['covariate_version'] = data_interface.covariate_paths.root_dir.name

    # FIXME: infectionator doesn't do metadata the right way.
    inf_metadata = data_interface.get_infectionator_metadata()
    inf_output_dir = inf_metadata['wrapped_R_call'][-1].split()[1].strip("'")
    version_map['infectionator_version'] = Path(inf_output_dir).name

    death_metadata = inf_metadata['death']['metadata']
    version_map['elastispliner_version'] = Path(death_metadata['output_path']).name

    model_inputs_metadata = death_metadata['model_inputs_metadata']
    version_map['model_inputs_version'] = Path(model_inputs_metadata['output_path']).name

    snapshot_metadata = model_inputs_metadata['snapshot_metadata']
    version_map['snapshot_version'] = Path(snapshot_metadata['output_path']).name
    jhu_snapshot_metadata = model_inputs_metadata['jhu_snapshot_metadata']
    version_map['jhu_snapshot_version'] = Path(jhu_snapshot_metadata['output_path']).name
    try:
        # There is a typo in the process that generates this key.
        # Protect ourselves in case they fix it without warning.
        webscrape_metadata = model_inputs_metadata['webcrape_metadata']
    except KeyError:
        webscrape_metadata = model_inputs_metadata['webscrape_metadata']
    version_map['webscrape_version'] = Path(webscrape_metadata['output_path']).name

    version_map['location_set_version_id'] = int(model_inputs_metadata['run_arguments']['lsvid'])
    version_map['data_date'] = Path(snapshot_metadata['output_path']).name.split('.')[0].replace('_', '-')

    version_map = pd.Series(version_map)
    version_map = version_map.reset_index()
    version_map.columns = ['name', 'version']
    return version_map


def load_populations(data_interface: ForecastDataInterface):
    metadata = data_interface.get_infectionator_metadata()
    model_inputs_path = Path(
        metadata['death']['metadata']['model_inputs_metadata']['output_path']
    )
    population_path = model_inputs_path / 'output_measures' / 'population' / 'all_populations.csv'
    populations = pd.read_csv(population_path)
    return populations


def load_hierarchy(data_interface: ForecastDataInterface):
    metadata = data_interface.get_infectionator_metadata()
    model_inputs_path = Path(
        metadata['death']['metadata']['model_inputs_metadata']['output_path']
    )
    hierarchy_path = model_inputs_path / 'locations' / 'modeling_hierarchy.csv'
    hierarchy = pd.read_csv(hierarchy_path)
    return hierarchy


def get_locations_modeled_and_missing(data_interface: ForecastDataInterface):
    hierarchy = load_hierarchy(data_interface)
    modeled_locations = data_interface.load_location_ids()
    most_detailed_locs = hierarchy.loc[hierarchy.most_detailed == 1, 'location_id'].unique().tolist()
    missing_locations = list(set(most_detailed_locs).difference(modeled_locations))
    locations_modeled_and_missing = {'modeled': modeled_locations, 'missing': missing_locations}
    return locations_modeled_and_missing


def load_modeled_hierarchy(data_interface: ForecastDataInterface):
    hierarchy = load_hierarchy(data_interface)
    modeled_locs = get_locations_modeled_and_missing(data_interface)['modeled']
    not_most_detailed = hierarchy.most_detailed == 0
    modeled = hierarchy.location_id.isin(modeled_locs)
    return hierarchy[not_most_detailed | modeled]


def build_resampling_map(deaths, resampling_params: ResamplingSpecification):
    cumulative_deaths = deaths.groupby(level='location_id').cumsum().reset_index()
    cumulative_deaths['date'] = pd.to_datetime(cumulative_deaths['date'])
    reference_deaths = (cumulative_deaths[cumulative_deaths.date == pd.Timestamp(resampling_params.reference_date)]
                        .set_index('location_id')
                        .drop(columns=['date', 'observed']))
    upper_deaths = reference_deaths.quantile(resampling_params.upper_quantile, axis=1)
    lower_deaths = reference_deaths.quantile(resampling_params.lower_quantile, axis=1)
    
    resample_map = {}
    for location_id in reference_deaths.index:
        upper, lower = upper_deaths.at[location_id], lower_deaths.at[location_id]
        loc_deaths = reference_deaths.loc[location_id]
        to_resample = loc_deaths[(upper < loc_deaths) | (loc_deaths < lower)].index.tolist()
        np.random.seed(location_id)
        to_fill = np.random.choice(loc_deaths.index.difference(to_resample),
                                   len(to_resample), replace=False).tolist()
        resample_map[location_id] = {'to_resample': to_resample,
                                     'to_fill': to_fill}
    return resample_map


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    mean = data.mean(axis=1).rename('mean')
    upper = data.quantile(.975, axis=1).rename('upper')
    lower = data.quantile(.025, axis=1).rename('lower')
    return pd.concat([mean, upper, lower], axis=1)


def resample_draws(measure_data: pd.DataFrame, resampling_map: Dict[int, Dict[str, List[int]]]):
    output = []
    locs = measure_data.reset_index().location_id.unique()
    for location_id, loc_map in resampling_map.items():
        if location_id not in locs:
            continue
        loc_data = measure_data.loc[location_id]
        loc_data[loc_map['to_resample']] = loc_data[loc_map['to_fill']]
        loc_data = pd.concat({location_id: loc_data}, names=['location_id'])
        if isinstance(loc_data, pd.Series):
            loc_data = loc_data.unstack()
        output.append(loc_data)

    resampled = pd.concat(output)
    resampled.columns = [f'draw_{draw}' for draw in resampled.columns]
    return resampled


def sum_aggregator(measure_data: pd.DataFrame, hierarchy: pd.DataFrame, _population: pd.DataFrame) -> pd.DataFrame:
    """Aggregates global and national data from subnational models by addition.

    For use with daily count space data.

    The ``_population`` argument is here for api consistency and is not used.

    """
    most_detailed = hierarchy.loc[hierarchy.most_detailed == 1, 'location_id'].tolist()
    # The observed column is in the elastispliner data.
    if 'observed' in measure_data.index.names:
        measure_data = measure_data.reset_index(level='observed')

    # Produce the global cause it's good to see.
    global_aggregate = measure_data.loc[most_detailed].groupby(level='date').sum()
    global_aggregate = pd.concat({1: global_aggregate}, names=['location_id'])
    measure_data = measure_data.append(global_aggregate)

    # Otherwise, all aggregation is to the national level.
    national_level = 3
    subnational_levels = sorted([level for level in hierarchy['level'].unique().tolist() if level > national_level])
    for level in subnational_levels[::-1]:  # From most detailed to least_detailed
        locs_at_level = hierarchy[hierarchy.level == level]
        for parent_id, children in locs_at_level.groupby('parent_id'):
            child_locs = children.location_id.tolist()
            aggregate = measure_data.loc[child_locs].groupby(level='date').sum()
            aggregate = pd.concat({parent_id: aggregate}, names=['location_id'])
            measure_data = measure_data.append(aggregate)

    # We'll call any aggregate with at least one observed point observed.
    if 'observed' in measure_data.columns:
        measure_data.loc[measure_data['observed'] >= 1, 'observed'] = 1
        measure_data = measure_data.set_index('observed', append=True)
    return measure_data


def mean_aggregator(measure_data: pd.DataFrame, hierarchy: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Aggregates global and national data from subnational models by a
    population weighted mean.

    """
    # Get all age/sex population and append to the data.
    population = collapse_population(population, hierarchy)
    measure_columns = measure_data.columns
    measure_data = measure_data.join(population)

    # We'll work in the space of pop*measure where aggregation is just a sum.
    weighted_measure_data = measure_data.loc[hierarchy.loc[hierarchy.most_detailed == 1, 'location_id'].tolist()]
    weighted_measure_data[measure_columns] = measure_data[measure_columns].mul(measure_data['population'], axis=0)

    # Some covariates are time-invariant
    if 'date' in weighted_measure_data.index.names:
        global_aggregate = weighted_measure_data.groupby(level='date').sum()
        global_aggregate = pd.concat({1: global_aggregate}, names=['location_id'])
    else:
        global_aggregate = weighted_measure_data.sum()
        global_aggregate = pd.concat({1: global_aggregate}, names=['location_id']).unstack()
    # Invert the weighting.  Pop column is also aggregated now.
    global_aggregate = global_aggregate[measure_columns].div(global_aggregate['population'], axis=0)
    measure_data = measure_data.append(global_aggregate)

    # Do the same thing again to produce national aggregates.
    national_level = 3
    subnational_levels = sorted([level for level in hierarchy['level'].unique().tolist() if level > national_level])
    for level in subnational_levels[::-1]:  # From most detailed to least_detailed
        locs_at_level = hierarchy[hierarchy.level == level]
        for parent_id, children in locs_at_level.groupby('parent_id'):
            child_locs = children.location_id.tolist()
            if 'date' in weighted_measure_data.index.names:
                aggregate = weighted_measure_data.loc[child_locs].groupby(level='date').sum()
                aggregate = pd.concat({parent_id: aggregate}, names=['location_id'])

            else:
                aggregate = weighted_measure_data.loc[child_locs].sum()
                aggregate = pd.concat({parent_id: aggregate}, names=['location_id']).unstack()

            weighted_measure_data = weighted_measure_data.append(aggregate)
            aggregate = aggregate[measure_columns].div(aggregate['population'], axis=0)
            measure_data = measure_data.append(aggregate)
    measure_data = measure_data.drop(columns='population')
    return measure_data


def collapse_population(population_data: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
    """Collapse the larger population table to all age and sex population."""
    most_detailed = population_data.location_id.isin(
        hierarchy.loc[hierarchy.most_detailed == 1, 'location_id'].tolist()
    )
    all_sexes = population_data.sex_id == 3
    all_ages = population_data.age_group_id == 22
    population_data = population_data.loc[most_detailed & all_ages & all_sexes, ['location_id', 'population']]
    population_data = population_data.set_index('location_id')['population']
    return population_data
