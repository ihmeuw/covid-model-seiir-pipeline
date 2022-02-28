import functools
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
                 combiner: Callable,
                 label: str,
                 cumulative_label: str = None,
                 summary_metric: Optional[str] = 'mean',
                 ci: float = .95,
                 ci2: float = None,
                 duration_label: str = None,
                 description: str = None,
                 cumulative_description: str = None):
        self.base_measures = base_measures
        self.combiner = combiner
        self.label = label
        self.cumulative_label = cumulative_label
        self.summary_metric = summary_metric
        self.ci = ci
        self.ci2 = ci2
        self.duration_label = duration_label
        self.description = description
        self.cumulative_description = cumulative_description


MEASURES: Dict[str, Union[MeasureConfig, CompositeMeasureConfig]] = {}

for prefix in ['', 'smoothed_']:
    for measure in ['deaths', 'hospitalizations', 'cases']:
        description = f'Reported and corrected {{metric}} {measure}. This data is a primary model input.'
        if prefix:
            description += ' This data has been smoothed with a rolling 7-day average.'

        MEASURES[f'{prefix}{measure}'] = MeasureConfig(
            loader=parallel.make_loader(
                FitDataInterface.load_input_epi_measures,
                measure=f'{prefix}daily_{measure}',
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
        round_specific=False,
        description=description.format(metric='daily'),
        cumulative_description=description.format(metric='cumulative')
    )


for reported_measure, infection_type in itertools.product(['death', 'admission', 'case'], ['naive', 'total']):
    description = (
        f'Posterior {{metric}} {infection_type} infections according to '
        f'reported {reported_measure}.'
    )
    MEASURES[f'posterior_{reported_measure}_based_{infection_type}_infections'] = MeasureConfig(
        loader=parallel.make_loader(
            FitDataInterface.load_posterior_epi_measures,
            f'daily_{infection_type}_infections',
            measure_version=reported_measure,
        ),
        label=f'posterior_{reported_measure}_based_daily_{infection_type}_infections',
        aggregator=aggregate.sum_aggregator,
        cumulative_label=f'posterior_{reported_measure}_based_cumulative_{infection_type}_infections',
        description=description.format(metric='daily'),
        cumulative_description=description.format(metric='cumulative'),
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
        description=description,
        round_specific=False,
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
        round_specific=False,
    )

for suffix, label_suffix in zip(['_death', '_admission', '_case'],
                                ['_deaths', '_hospitalizations', '_cases']):
    measure_used = label_suffix[1:]
    description = (
        f'Estimate of beta using {label_suffix[1:]}. '
        f'This data is an output of the past infections model.'
    )
    MEASURES[f'beta{label_suffix}'] = MeasureConfig(
        loader=parallel.make_loader(
            FitDataInterface.load_fit_beta,
            f'beta{suffix}',
            measure_version=suffix[1:],
        ),
        label=f'beta{label_suffix}',
        description=description,
    )

MEASURES['beta'] = MeasureConfig(
    loader=parallel.make_loader(
        FitDataInterface.load_fit_beta, 'beta',
    ),
    label='beta',
    description=(
        f'Estimate of beta using all epi measures. '
        f'This data is an output of the past infections model.'
    ),
)


def load_seroprevalence(data_interface: FitDataInterface,
                        *,  # Disallow other positional args
                        num_draws: int = 1,
                        progress_bar: bool = False,
                        **_) -> pd.DataFrame:
    input_sero_data = (data_interface
                       .load_seroprevalence()
                       .reset_index()
                       .rename(columns={'date': 'sero_date'})
                       .set_index(['data_id', 'location_id', 'sero_date', 'is_outlier'])
                       .loc[:, ['reported_seroprevalence']])

    idx_cols = ['data_id', 'location_id', 'sero_date', 'is_outlier']
    adjusted_sero_data = []
    measures_and_draws = list(itertools.product(['case', 'death', 'admission'], range(num_draws)))
    for measure_version, draw in tqdm.tqdm(measures_and_draws, disable=not progress_bar):
        df = (data_interface
              .load_final_seroprevalence(draw, measure_version=measure_version)
              .loc[:, idx_cols + ['date', 'seroprevalence', 'adjusted_seroprevalence']])
        adjusted_sero_data.append(df)
    adjusted_data = pd.concat(adjusted_sero_data).groupby(idx_cols)
    adjusted_data = (pd.concat([adjusted_data['date'].median(),
                                adjusted_data[['seroprevalence', 'adjusted_seroprevalence']].mean()],
                               axis=1)
                     .reset_index()
                     .set_index(['data_id', 'location_id', 'date', 'sero_date', 'is_outlier']))
    seroprevalence = input_sero_data.join(adjusted_data, how='outer').reset_index()
    seroprevalence['date'] = seroprevalence['date'].fillna(seroprevalence['sero_date'])
    seroprevalence = (seroprevalence
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


def load_durations(data_interface: FitDataInterface,
                   *,  # Disallow other positional args
                   num_draws: int = 1,
                   progress_bar: bool = False,
                   **_) -> pd.Series:
    durations = [f'exposure_to_{m}' for m in ['death', 'admission', 'case']]
    parameter_data = []
    for draw in tqdm.trange(num_draws, disable=not progress_bar):
        d = data_interface.load_ode_params(draw, columns=durations).iloc[0]
        parameter_data.append(d)
    parameter_data = pd.concat(parameter_data, axis=1)
    parameter_data.columns = [f'draw_{i}' for i in range(num_draws)]
    return parameter_data


MEASURES['durations'] = MeasureConfig(
    loader=load_durations,
    label='durations',
    description=f'Durations from exposure to death, admission, and case.',
)


def load_rates_data(data_interface: FitDataInterface,
                    *,  # Disallow other positional args
                    num_draws: int = 1,
                    progress_bar: bool = False,
                    **_) -> pd.DataFrame:
    rates_data = []
    measures_and_draws = list(itertools.product(['case', 'death', 'admission'], range(num_draws)))
    for measure_version, draw in tqdm.tqdm(measures_and_draws, disable=not progress_bar):
        df = data_interface.load_rates_data(draw, measure_version=measure_version)
        rates_data.append(df)
    rates_data = pd.concat(rates_data).groupby(['measure', 'round', 'data_id', 'location_id'])
    rates_data = (pd.concat([rates_data['date'].median(), rates_data['value'].mean()], axis=1)
                  .reset_index()
                  .drop(columns='data_id'))
    return rates_data.set_index(['measure', 'round', 'location_id', 'date'])


MEASURES['rates_data'] = MeasureConfig(
    loader=load_rates_data,
    label='rates_data',
    summary_metric=None,
    description=(
        'Sampled seroprevalence data points shifted into the '
        'space of the rates analysis. This data is an output of the rates model, '
        'an intermediate step of the past infections model.'
    )
)


def make_ratio(numerator: pd.DataFrame, denominator: pd.DataFrame, duration: pd.Series):
    out = []
    for draw in numerator.columns:
        draw_duration = duration.loc[draw]
        out.append(numerator[draw] / denominator[draw].groupby(['location_id', 'round']).shift(draw_duration))
    out = pd.concat(out, axis=1)
    return out


for ratio, measure, duration_measure in zip(['ifr', 'ihr', 'idr'],
                                            ['deaths', 'hospitalizations', 'cases'],
                                            ['death', 'admission', 'case']):
    MEASURES[f'posterior_{ratio}'] = CompositeMeasureConfig(
        base_measures={'numerator': MEASURES[f'smoothed_{measure}'],
                       'denominator': MEASURES['posterior_total_infections']},
        label=f'posterior_{ratio}',
        duration_label=f'exposure_to_{duration_measure}',
        combiner=make_ratio,
        description=(
            f'Posterior {ratio.upper()}. This data is a composite of '
            f'results from the past infections model.'
        )
    )



def get_data_dictionary() -> pd.DataFrame:
    data_dictionary = {}
    for measure_config in MEASURES.values():
        data_dictionary[measure_config.label] = measure_config.description
        if measure_config.cumulative_label:
            data_dictionary[measure_config.cumulative_label] = measure_config.cumulative_description
    data_dictionary = pd.Series(data_dictionary).reset_index()
    data_dictionary.columns = ['output', 'description']
    return data_dictionary
