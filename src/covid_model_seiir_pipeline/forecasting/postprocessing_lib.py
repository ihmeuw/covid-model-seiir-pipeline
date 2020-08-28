import functools
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.forecasting.workflow import FORECAST_SCALING_CORES


# TODO: make a model subpackage and put this there.

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
    deaths = draw_df['deaths'].rename(draw_id)
    infections = draw_df['infections'].rename(draw_id)
    r_effective = draw_df['r_effective'].rename(draw_id)
    return deaths, infections, r_effective


def load_coefficients(data_interface: ForecastDataInterface):
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


def load_covariates(scenario: str, cov_order: Dict[str, List[str]],
                    data_interface: ForecastDataInterface) -> Dict[str, List[pd.Series]]:
    _runner = functools.partial(
        load_covariates_by_draw,
        scenario=scenario,
        cov_order=cov_order,
        data_interface=data_interface,
    )
    draws = range(data_interface.get_n_draws())
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        outputs = pool.map(_runner, draws)

    cov_names = [*cov_order['time_varying'], *cov_order['non_time_varying']]
    covariates = dict(zip(cov_names, zip(*outputs)))
    return covariates


def load_covariates_by_draw(draw_id: int, scenario: str,
                            cov_order: Dict[str, List[str]],
                            data_interface: ForecastDataInterface) -> Tuple[pd.Series, ...]:
    covariate_df = data_interface.load_raw_covariates(scenario, draw_id)
    covariate_df = covariate_df.set_index(['location_id', 'date']).sort_index()
    covariate_grouped = covariate_df.groupby(level='location_id')

    time_varying = [covariate_df[col].rename(draw_id) for col in cov_order['time_varying']]
    non_time_varying = [covariate_grouped[col].max().rename(draw_id) for col in cov_order['non_time_varying']]

    return (*time_varying, *non_time_varying)


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


def load_beta_residuals(data_interface: ForecastDataInterface) -> List[pd.Series]:
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


def load_elastispliner_outputs(data_interface: ForecastDataInterface):
    es_noisy, es_smoothed = data_interface.load_elastispliner_outputs()
    es_noisy = es_noisy.set_index(['location_id', 'date', 'observed'])
    es_smoothed = es_smoothed.set_index(['location_id', 'date', 'observed'])
    return es_noisy, es_smoothed


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
    metadata = data_interface.get_regression_metadata()
    version_map = {}
    version_map['forecast_version'] = data_interface.forecast_paths.root_dir.name
    version_map['regression_version'] = Path(metadata['output_path']).name
    version_map['covariate_version'] = Path(metadata['covariates_metadata']['output_path']).name

    # FIXME: infectionator doesn't do metadata the right way.
    inf_metadata = metadata['infectionator_metadata']
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
    metadata = data_interface.get_regression_metadata()
    model_inputs_path = Path(
        metadata['infectionator_metadata']['death']['metadata']['model_inputs_metadata']['output_path']
    )
    population_path = model_inputs_path / 'output_measures' / 'population' / 'all_populations.csv'
    populations = pd.read_csv(population_path)
    return populations


def load_location_information(data_interface: ForecastDataInterface):
    metadata = data_interface.get_regression_metadata()
    model_inputs_path = Path(
        metadata['infectionator_metadata']['death']['metadata']['model_inputs_metadata']['output_path']
    )
    hierarchy_path = model_inputs_path / 'locations' / 'modeling_hierarchy.csv'
    hierarchy = pd.read_csv(hierarchy_path)
    modeled_locations = data_interface.load_location_ids()
    most_detailed_locs = hierarchy.loc[hierarchy.most_detailed == 1, 'location_id'].unique().tolist()
    missing_locations = list(set(most_detailed_locs).difference(modeled_locations))
    locations_modeled_and_missing = {'modeled': modeled_locations, 'missing': missing_locations}
    return hierarchy, locations_modeled_and_missing



def build_resampling_map(deaths, resampling_params):
    cumulative_deaths = deaths.groupby(level='location_id').cumsum()
    max_deaths = cumulative_deaths.groupby(level='location_id').max()
    upper_deaths = max_deaths.quantile(resampling_params['upper_quantile'], axis=1)
    lower_deaths = max_deaths.quantile(resampling_params['lower_quantile'], axis=1)
    resample_map = {}
    for location_id in max_deaths.index:
        upper, lower = upper_deaths.at[location_id], lower_deaths.at[location_id]
        loc_deaths = max_deaths.loc[location_id]
        to_resample = loc_deaths[(upper < loc_deaths) | (loc_deaths < lower)].index.tolist()
        np.random.seed(12345)
        to_fill = np.random.choice(loc_deaths.index, len(to_resample), replace=False).tolist()
        resample_map[location_id] = {'to_resample': to_resample,
                                     'to_fill': to_fill}
    return resample_map


def summarize(data: pd.DataFrame):
    data['mean'] = data.mean(axis=1)
    data['upper'] = data.quantile(.975, axis=1)
    data['lower'] = data.quantile(.025, axis=1)
    return data[['mean', 'upper', 'lower']]


def resample_draws(resampling_map, *measure_data: pd.DataFrame):
    #_runner = functools.partial(
    #    resample_draws_by_measure,
    #    resampling_map=resampling_map
    #)
    #with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
    #    resampled = pool.map(_runner, measure_data)
    resampled = [resample_draws_by_measure(measure, resampling_map) for measure in measure_data]
    return resampled


def resample_draws_by_measure(measure_data: pd.DataFrame, resampling_map):
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
