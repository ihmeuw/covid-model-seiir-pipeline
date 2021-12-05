import itertools
from typing import Callable, Dict, Optional, Union

import pandas as pd
import tqdm

from covid_model_seiir_pipeline.lib import (
    aggregate,
    parallel,
)
from covid_model_seiir_pipeline.pipeline.fit.data import (
    FitDataInterface,
)


class MeasureConfig:
    def __init__(self,
                 loader: Callable,
                 label: str,
                 cumulative_label: str = None,
                 round_specific: bool = True,
                 aggregator: Callable = None,
                 summary_metric: Optional[str] = 'mean',
                 ci: float = .95,
                 ci2: float = None,
                 description: str = None,
                 cumulative_description: str = None):
        self.loader = loader
        self.label = label
        self.cumulative_label = cumulative_label
        self.round_specific = round_specific
        self.aggregator = aggregator
        self.summary_metric = summary_metric
        self.ci = ci
        self.ci2 = ci2
        self.description = description
        self.cumulative_description = cumulative_description


class CompositeMeasureConfig:
    def __init__(self,
                 base_measures: Dict[str, MeasureConfig],
                 label: str,
                 duration_label: str,
                 combiner: Callable,
                 description: str):
        self.base_measures = base_measures
        self.label = label
        self.duration_label = duration_label
        self.combiner = combiner
        self.description = description


MEASURES: Dict[str, Union[MeasureConfig, CompositeMeasureConfig]] = {}

for prefix in ['', 'smoothed_']:
    for measure in ['deaths', 'hospitalizations', 'cases']:
        description = f'Reported and corrected {{metric}} {measure}. This data is a primary model input.'
        if prefix:
            description += ' This data has been smoothed with a rolling 7-day average.'

        MEASURES[f'{prefix}{measure}'] = MeasureConfig(
            loader=parallel.make_loader(
                FitDataInterface.load_input_epi_measures, f'{prefix}daily_{measure}'
            ),
            label=f'{prefix}daily_{measure}',
            round_specific=False,
            cumulative_label=f'{prefix}cumulative_{measure}',
            aggregator=aggregate.sum_aggregator,
            description=description.format(metric='daily'),
            cumulative_description=description.format(metric='cumulative'),
        )


for measure in ['naive_unvaccinated_infections', 'naive_infections', 'total_infections',
                'deaths', 'cases', 'hospitalizations']:
    description_measure = 'infections' if 'infections' in measure else measure
    if 'total' in measure:
        denominator = 'total population'
    else:
        denominator = 'COVID-naive (those without a prior covid infection)'
        if 'unvaccinated' in measure:
            denominator = 'unvaccinated and ' + denominator

    description = (
        f'Posterior {{metric}} {description_measure} among the {denominator} '
        'This data is an output of the past infections model.'
    )
    MEASURES[f'posterior_{measure}'] = MeasureConfig(
        loader=parallel.make_loader(
            FitDataInterface.load_posterior_epi_measures, f'daily_{measure}',
        ),
        label=f'posterior_daily_{measure}',
        cumulative_label=f'posterior_cumulative_{measure}',
        aggregator=aggregate.sum_aggregator,
        ci=1.,
        ci2=.95,
        description=description.format(metric='daily'),
        cumulative_description=description.format(metric='cumulative')
    )


for measure in ['naive', 'naive_unvaccinated']:
    vax = ' unvaccinated ' if 'unvaccinated' in measure else ''

    description = (
        f'Count of {vax} people never exposed to covid. '
        f'This data is an output of the past infections model.'
    )
    MEASURES[f'posterior_{measure}'] = MeasureConfig(
        loader=parallel.make_loader(
            FitDataInterface.load_posterior_epi_measures, measure,
        ),
        label=f'posterior_{measure}',
        aggregator=aggregate.sum_aggregator,
    )

for rate, suffix in itertools.product(['ifr', 'ihr', 'idr'], ['', '_lr', '_hr']):
    denominator = {'': 'entire population', '_lr': 'population under 65', '_hr': 'population over 65'}[suffix]
    description = (
        f'{rate.upper()} prior among the {denominator}. '
        f'This data is an output of the rates model, an intermediate step of the past infections model.'
    )
    measure = f'{rate}{suffix}'
    MEASURES[f'prior_{measure}'] = MeasureConfig(
        loader=parallel.make_loader(
            FitDataInterface.load_rates, measure,
        ),
        label=f'prior_{measure}',
        description=description,
    )

for suffix, label_suffix in zip(['', '_death', '_admission', '_case'],
                                ['', '_deaths', '_hospitalizations', '_cases']):
    measure_used = label_suffix[1:] if label_suffix else 'all reported epi data'
    description = (
        f'Estimate of beta using {measure_used}. '
        f'This data is an output of the past infections model.'
    )
    MEASURES[f'beta{label_suffix}'] = MeasureConfig(
        loader=parallel.make_loader(
            FitDataInterface.load_beta, f'beta{suffix}',
        ),
        label=f'beta{label_suffix}',
        description=description,
    )


def load_seroprevalence(data_interface: FitDataInterface,
                        *,  # Disallow other positional args
                        num_draws: int = 1,
                        progress_bar: bool = False,
                        **_) -> pd.DataFrame:
    idx_cols = ['data_id', 'location_id', 'date', 'is_outlier']

    input_sero_data = (data_interface
                       .load_seroprevalence()
                       .reset_index()
                       .set_index(idx_cols)
                       .loc[:, ['reported_seroprevalence']])

    adjusted_sero_data = []
    for draw in tqdm.trange(num_draws, disable=not progress_bar):
        df = (data_interface
              .load_final_seroprevalence(draw)
              .reset_index()
              .set_index(idx_cols)
              .loc[:, ['seroprevalence', 'adjusted_seroprevalence']])
        adjusted_sero_data.append(df)
    adjusted_data = pd.concat(adjusted_sero_data).groupby(idx_cols).mean()
    seroprevalence = (pd.concat([input_sero_data, adjusted_data], axis=1)
                      .reset_index()
                      .drop(columns='data_id')
                      .set_index(['location_id', 'date']))
    return seroprevalence


MEASURES['seroprevalence'] = MeasureConfig(
    loader=load_seroprevalence,
    label='seroprevalence',
    summary_metric=None,
    description=('Reported, sampled, and adjusted seroprevalence data. '
                 'This data is a primary model input.')
)


def load_ode_params(data_interface: FitDataInterface,
                    *,  # Disallow other positional args
                    num_draws: int = 1,
                    progress_bar: bool = False,
                    **_) -> pd.DataFrame:

    parameter_data = []
    for draw in tqdm.trange(num_draws, disable=not progress_bar):
        df = data_interface.load_ode_params(draw).set_index('parameter').value
        parameter_data.append(df)
    parameter_data = pd.concat(parameter_data, axis=1)
    return parameter_data


MEASURES['ode_parameters'] = MeasureConfig(
    loader=load_ode_params,
    label='ode_parameters',
    description=('ODE system and rates parameters. '
                 'These parameters are sampled from distributions based on literature and assumptions')
)


def load_rates_data(data_interface: FitDataInterface,
                    *,  # Disallow other positional args
                    num_draws: int = 1,
                    progress_bar: bool = False,
                    **_) -> pd.DataFrame:
    rates_data = []
    for draw in tqdm.trange(num_draws, disable=not progress_bar):
        df = data_interface.load_rates_data(draw)
        rates_data.append(df)
    rates_data = pd.concat(rates_data).groupby(['measure', 'round', 'data_id', 'location_id'])
    rates_data = (pd.concat([rates_data['date'].median(), rates_data['value'].mean()], axis=1)
                  .reset_index()
                  .drop(columns='data_id'))
    return rates_data


MEASURES['rates_data'] = MeasureConfig(
    loader=load_rates_data,
    label='rates_data',
    summary_metric=None,
    description=('Sampled seroprevalence data points shifted into the space of the rates analysis. '
                 'This data is an output of the rates model, an intermediate step of the past infections model.')
)


def make_ratio(numerator: pd.DataFrame, denominator: pd.DataFrame, duration: pd.Series):
    out = []
    for draw in numerator.columns:
        draw_duration = duration.loc[draw]
        out.append(numerator[draw] / denominator[draw].groupby('location_id').shift(draw_duration))
    out = pd.concat(out, axis=1)
    return out


for ratio, measure, duration_measure in zip(['ifr', 'ihr', 'idr'],
                                            ['deaths', 'hospitalizations', 'cases'],
                                            ['death', 'admission', 'case']):
    MEASURES[f'posterior_{ratio}'] = CompositeMeasureConfig(
        base_measures={'numerator': MEASURES[f'posterior_{measure}'],
                       'denominator': MEASURES['posterior_naive_unvaccinated_infections']},
        label=f'posterior_{ratio}',
        duration_label=f'exposure_to_{duration_measure}',
        combiner=make_ratio,
        description=(
            f'Posterior {ratio.upper()} among the unvaccinated and COVID-naive (those without a '
            f'prior covid infection). This data is a composite of results from the past infections model.'
        )
    )


def get_data_dictionary() -> pd.DataFrame:
    data_dictionary = {}
    for measure_config in MEASURES.values():
        data_dictionary[measure_config.label] = measure_config.description
        if isinstance(measure_config, MeasureConfig) and measure_config.cumulative_label:
            data_dictionary[measure_config.cumulative_label] = measure_config.cumulative_description
    data_dictionary = pd.Series(data_dictionary).reset_index()
    data_dictionary.columns = ['output', 'description']
    return data_dictionary


