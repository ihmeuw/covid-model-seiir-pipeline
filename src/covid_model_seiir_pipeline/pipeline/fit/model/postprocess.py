import itertools
from typing import Callable, Dict, Optional

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
                 ci2: float = None):
        self.loader = loader
        self.label = label
        self.cumulative_label = cumulative_label
        self.round_specific = round_specific
        self.aggregator = aggregator
        self.summary_metric = summary_metric
        self.ci = ci
        self.ci2 = ci2


MEASURES: Dict[str, MeasureConfig] = {}

for prefix in ['', 'smoothed_']:
    for measure in ['deaths', 'hospitalizations', 'cases']:
        MEASURES[f'{prefix}daily_{measure}'] = MeasureConfig(
            loader=parallel.make_loader(
                FitDataInterface.load_input_epi_measures, f'{prefix}daily_{measure}'
            ),
            label=f'{prefix}daily_{measure}',
            cumulative_label=f'{prefix}cumulative_{measure}',
            aggregator=aggregate.sum_aggregator,
        )

for measure in ['naive_unvaccinated_infections', 'naive_infections', 'total_infections',
                'deaths', 'cases', 'hospitalizations']:
    MEASURES[f'posterior_daily_{measure}'] = MeasureConfig(
        loader=parallel.make_loader(
            FitDataInterface.load_posterior_epi_measures, f'daily_{measure}',
        ),
        label=f'posterior_daily_{measure}',
        cumulative_label=f'posterior_cumulative_{measure}',
        aggregator=aggregate.sum_aggregator,
    )

for measure in ['naive', 'naive_unvaccinated']:
    MEASURES[f'posterior_{measure}'] = MeasureConfig(
        loader=parallel.make_loader(
            FitDataInterface.load_posterior_epi_measures, measure,
        ),
        label=f'posterior_{measure}',
        aggregator=aggregate.sum_aggregator,
    )

for measure in [f'{rate}{suffix}' for rate, suffix
                in itertools.product(['ifr', 'ihr', 'idr'], ['', '_lr', '_hr'])]:
    MEASURES[f'prior_{measure}'] = MeasureConfig(
        loader=parallel.make_loader(
            FitDataInterface.load_rates, measure,
        ),
        label=f'prior_{measure}',
    )

for suffix, label_suffix in zip(['', '_death', '_admission', '_case'],
                                ['', '_deaths', '_hospitalizations', '_cases']):
    MEASURES[f'beta{label_suffix}'] = MeasureConfig(
        loader=parallel.make_loader(
            FitDataInterface.load_beta, f'beta{suffix}',
        ),
        label=f'beta{label_suffix}',
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
)

